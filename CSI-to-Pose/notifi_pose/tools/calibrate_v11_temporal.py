"""Validation-only calibration of fixed latency and temporal smoothing.

The transform is global: one shift/window/blend is shared by every trial.
No per-trial alignment and no test split are used for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .. import contract as C
from ..hybrid_v10 import RootExpertBlend, build_residual_hybrid
from ..seen_v4 import AlignmentRobustTrajectoryNet
from ..trainer import set_seed
from .evaluate_sealed import make_model
from .train_p2_v9_hybrid import root_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, load_v3, make_loaders


def _smooth(values: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return values
    shape = values.shape
    flattened = values.flatten(2).transpose(1, 2)
    weight = valid[:, None].to(values.dtype)
    numerator = F.avg_pool1d(
        flattened * weight, window, stride=1, padding=window // 2,
        count_include_pad=False,
    )
    denominator = F.avg_pool1d(
        weight, window, stride=1, padding=window // 2,
        count_include_pad=False,
    ).clamp_min(1e-6)
    return (numerator / denominator).transpose(1, 2).reshape(shape)


def _shift(values: torch.Tensor, valid: torch.Tensor, frames: int) -> torch.Tensor:
    if not frames:
        return values
    length = values.shape[1]
    source = (torch.arange(length, device=values.device) + frames).clamp(0, length - 1)
    shifted = values.index_select(1, source)
    source_valid = valid.index_select(1, source)
    mask = source_valid
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, shifted, values)


class GlobalTemporalCalibration(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.pose_shift = 0
        self.pose_window = 1
        self.pose_blend = 0.0
        self.root_shift = 0
        self.root_window = 1
        self.root_blend = 0.0

    def set_pose(self, shift: int, window: int, blend: float) -> None:
        self.pose_shift, self.pose_window, self.pose_blend = shift, window, blend

    def set_root(self, shift: int, window: int, blend: float) -> None:
        self.root_shift, self.root_window, self.root_blend = shift, window, blend

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = dict(self.base(csi, link_mask))
        valid = link_mask.any(-1)
        pose = _shift(output["pose_rel"], valid, self.pose_shift)
        pose_smooth = _smooth(pose, valid, self.pose_window)
        output["pose_rel"] = pose + self.pose_blend * (pose_smooth - pose)
        root = _shift(output["root"], valid, self.root_shift)
        root_smooth = _smooth(root, valid, self.root_window)
        output["root"] = root + self.root_blend * (root_smooth - root)
        return output


def _pose_score(metrics: dict) -> float:
    return (
        float(metrics["mpjpe_m"])
        + 0.20 * float(metrics["dynamic_mpjpe_m"])
        + 0.15 * float(metrics["danger_mpjpe_m"])
        + 0.05 * float(metrics["danger_endpoint_mpjpe_m"])
    )


def _pose_feasible(metrics: dict, baseline: dict) -> bool:
    return (
        metrics["mpjpe_m"] <= baseline["mpjpe_m"] + 0.001
        and metrics["dynamic_mpjpe_m"] <= baseline["dynamic_mpjpe_m"] * 1.02
        and metrics["danger_endpoint_mpjpe_m"]
        <= baseline["danger_endpoint_mpjpe_m"] * 1.03
        and 0.85 <= metrics["pose_speed_ratio"] <= 1.20
    )


def _load_model(args, device: str) -> tuple[nn.Module, dict]:
    p2_checkpoint = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)
    hybrid_checkpoint = torch.load(
        args.hybrid_checkpoint, map_location=device, weights_only=False
    )
    primary = build_residual_hybrid(
        make_model(p2_checkpoint, device),
        hybrid_checkpoint.get("residual_decoder", "dense"),
    ).to(device)
    primary.load_state_dict(hybrid_checkpoint["model"])
    calibration = json.loads(args.hybrid_calibration.read_text(encoding="utf-8"))
    selected = calibration["selected"]
    pose_override = getattr(args, "pose_strength_override", None)
    if pose_override is not None:
        selected = {**selected, "pose_strength": pose_override}
    primary.set_calibration(
        selected["pose_strength"], selected["root_strength"],
        selected["class_strength"], selected["risk_strength"],
    )

    checkpoint = torch.load(
        args.root_expert_checkpoint, map_location=device, weights_only=False
    )
    checkpoint_protocol = checkpoint.get("protocol")
    if checkpoint_protocol is None and not getattr(
        args, "allow_unverified_root_protocol", False
    ):
        raise RuntimeError(
            "root expert checkpoint has no protocol metadata; "
            "use a clean-protocol checkpoint or explicitly allow legacy evaluation"
        )
    if checkpoint_protocol is not None and checkpoint_protocol != args.exp:
        raise RuntimeError(
            "root expert protocol mismatch: "
            f"checkpoint={checkpoint_protocol}, requested={args.exp}"
        )
    if getattr(args, "root_expert_kind", "v9") == "p2_hybrid":
        if checkpoint.get("objective") != "root_only":
            raise RuntimeError(
                "p2_hybrid root expert must be trained with objective=root_only"
            )
        root_expert = build_residual_hybrid(
            make_model(p2_checkpoint, device),
            checkpoint.get("residual_decoder", "dense"),
        ).to(device)
        root_expert.load_state_dict(checkpoint["model"])
        root_expert.set_calibration(0.0, 1.0, 0.0, 0.0)
    else:
        root_expert = AlignmentRobustTrajectoryNet(load_v3(args, device)).to(device)
        missing, unexpected = root_expert.load_state_dict(
            checkpoint["model"], strict=False
        )
        allowed_missing = ("class_head.", "risk_head.")
        invalid_missing = [
            key for key in missing if not key.startswith(allowed_missing)
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "incompatible root expert checkpoint: "
                f"missing={invalid_missing}, unexpected={list(unexpected)}"
            )
        root_expert.set_calibration(
            float(checkpoint.get("pose_strength", 0.0)),
            float(checkpoint.get("root_strength", 0.0)),
        )
    model = RootExpertBlend(primary, root_expert).to(device)
    model.set_root_strength(args.root_expert_strength)
    model.eval()
    return model, selected


def _evaluate_candidate(model, loader, device: str, max_shift: int,
                        kind: str, shift: int, window: int, blend: float) -> dict:
    if kind == "pose":
        model.set_pose(shift, window, blend)
    else:
        model.set_root(shift, window, blend)
    metrics = evaluate_trajectory(model, loader, device, max_shift)
    return {
        "kind": kind, "shift": shift, "window": window, "blend": blend,
        "pose_score": _pose_score(metrics),
        "root_score": root_selection_score(metrics),
        "validation": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument(
        "--hybrid-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v9_hybrid_v10_clean" / "best_model.pt",
    )
    parser.add_argument(
        "--hybrid-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v9_hybrid_v10_clean" / "recalibrated_results.json",
    )
    parser.add_argument(
        "--root-expert-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v4_v9a_lmh_e01_multitask_recall93" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--root-expert-kind", choices=("v9", "p2_hybrid"), default="v9"
    )
    parser.add_argument("--allow-unverified-root-protocol", action="store_true")
    parser.add_argument(
        "--v3-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v3_contact_root" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--v2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
    )
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--pose-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--root-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--root-expert-strength", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v11_temporal_calibration" / "validation.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    base, source_calibration = _load_model(args, device)
    model = GlobalTemporalCalibration(base).to(device)
    baseline = evaluate_trajectory(model, loaders["val"], device, args.max_shift)

    pose_candidates = []
    for shift in (-12, -8, -4, 0, 4, 8, 12):
        pose_candidates.append(_evaluate_candidate(
            model, loaders["val"], device, args.max_shift,
            "pose", shift, 1, 0.0,
        ))
    shift_best = min(
        [item for item in pose_candidates if _pose_feasible(item["validation"], baseline)]
        or pose_candidates,
        key=lambda item: item["pose_score"],
    )
    for window in (3, 5, 7):
        for blend in (0.25, 0.50, 0.75, 1.0):
            pose_candidates.append(_evaluate_candidate(
                model, loaders["val"], device, args.max_shift,
                "pose", shift_best["shift"], window, blend,
            ))
    pose_best = min(
        [item for item in pose_candidates if _pose_feasible(item["validation"], baseline)]
        or pose_candidates,
        key=lambda item: item["pose_score"],
    )
    model.set_pose(pose_best["shift"], pose_best["window"], pose_best["blend"])

    root_candidates = []
    for shift in (-12, -8, -4, 0, 4, 8, 12):
        root_candidates.append(_evaluate_candidate(
            model, loaders["val"], device, args.max_shift,
            "root", shift, 1, 0.0,
        ))
    root_shift = min(root_candidates, key=lambda item: item["root_score"])
    for window in (3, 5, 7):
        for blend in (0.25, 0.50, 0.75, 1.0):
            root_candidates.append(_evaluate_candidate(
                model, loaders["val"], device, args.max_shift,
                "root", root_shift["shift"], window, blend,
            ))
    root_best = min(root_candidates, key=lambda item: item["root_score"])
    model.set_root(root_best["shift"], root_best["window"], root_best["blend"])
    selected_validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    result = {
        "run": "p2_v11_global_temporal_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "constraints": "one global transform shared by every trial",
        "source_calibration": source_calibration,
        "baseline_validation": baseline,
        "selected": {
            "pose_shift": pose_best["shift"],
            "pose_window": pose_best["window"],
            "pose_blend": pose_best["blend"],
            "root_shift": root_best["shift"],
            "root_window": root_best["window"],
            "root_blend": root_best["blend"],
        },
        "selected_validation": selected_validation,
        "pose_candidates": pose_candidates,
        "root_candidates": root_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selected": result["selected"],
        "baseline_validation": baseline,
        "selected_validation": selected_validation,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
