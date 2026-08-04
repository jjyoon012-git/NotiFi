"""Validation-only smoothing of the learned pose residual.

The frozen P2 pose is left untouched. Only the correction predicted by the
hybrid decoder is smoothed, which avoids blurring the coarse fall trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..hybrid_v10 import (
    RootExpertBlend,
    SequenceBoneCalibration,
    build_residual_hybrid,
)
from ..trainer import set_seed
from .evaluate_sealed import make_model
from .train_p2_v9_hybrid import pose_selection_score, root_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def _masked_smooth(values: torch.Tensor, valid: torch.Tensor,
                   window: int) -> torch.Tensor:
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


class ResidualTemporalCalibration(nn.Module):
    """Smooth the hybrid correction while preserving the P2 prediction."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.window = 1
        self.blend = 0.0
        self.risk_adaptive = "none"
        self.danger_logit_bias = 0.0
        self.root_window = 1
        self.root_blend = 0.0

    def set_calibration(self, window: int, blend: float,
                        risk_adaptive: str = "none",
                        danger_logit_bias: float = 0.0) -> None:
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be between 0 and 1")
        if risk_adaptive not in {"none", "probability", "hard"}:
            raise ValueError(f"unknown risk-adaptive mode: {risk_adaptive}")
        self.window = window
        self.blend = blend
        self.risk_adaptive = risk_adaptive
        self.danger_logit_bias = danger_logit_bias

    def set_root_calibration(self, window: int, blend: float) -> None:
        if window < 1 or window % 2 == 0:
            raise ValueError("root window must be a positive odd integer")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("root blend must be between 0 and 1")
        self.root_window = window
        self.root_blend = blend

    def _adaptive_amount(self, output: dict, values: torch.Tensor,
                         blend: float) -> torch.Tensor:
        smoothing = values.new_full((len(values),), blend)
        if self.risk_adaptive == "none":
            return smoothing
        logits = output["risk_logits"].clone()
        logits[:, 2] += self.danger_logit_bias
        if self.risk_adaptive == "probability":
            danger_gate = torch.softmax(logits, dim=-1)[:, 2]
        else:
            danger_gate = logits.argmax(-1).eq(2).to(values.dtype)
        return smoothing * (1.0 - danger_gate)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = dict(self.base(csi, link_mask))
        if "pose_p2" not in output:
            raise KeyError("hybrid output must expose pose_p2")
        valid = link_mask.any(-1)
        residual = output["pose_rel"] - output["pose_p2"]
        smoothed = _masked_smooth(residual, valid, self.window)
        smoothing = self._adaptive_amount(output, residual, self.blend)
        calibrated = residual + smoothing[:, None, None, None] * (
            smoothed - residual
        )
        output["pose_rel"] = output["pose_p2"] + calibrated
        if "root_primary" in output and self.root_blend:
            root_residual = output["root"] - output["root_primary"]
            root_smoothed = _masked_smooth(
                root_residual, valid, self.root_window
            )
            root_amount = self._adaptive_amount(
                output, root_residual, self.root_blend
            )
            output["root"] = output["root_primary"] + root_residual + (
                root_amount[:, None, None] * (root_smoothed - root_residual)
            )
        return output


def _checked_checkpoint(path: Path, device: str, protocol: str) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("protocol") != protocol:
        raise RuntimeError(
            f"checkpoint protocol mismatch for {path}: "
            f"{checkpoint.get('protocol')!r} != {protocol!r}"
        )
    return checkpoint


def _build_model(args, device: str) -> nn.Module:
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    hybrid_checkpoint = _checked_checkpoint(
        args.hybrid_checkpoint, device, args.exp
    )
    primary = build_residual_hybrid(
        make_model(p2_checkpoint, device),
        hybrid_checkpoint.get("residual_decoder", "dense"),
    ).to(device)
    primary.load_state_dict(hybrid_checkpoint["model"])
    primary.set_calibration(args.pose_strength, 0.0, 0.0, 0.0)

    root_checkpoint = _checked_checkpoint(
        args.root_expert_checkpoint, device, args.exp
    )
    if root_checkpoint.get("objective") != "root_only":
        raise RuntimeError("root expert must have objective=root_only")
    root_expert = build_residual_hybrid(
        make_model(p2_checkpoint, device),
        root_checkpoint.get("residual_decoder", "dense"),
    ).to(device)
    root_expert.load_state_dict(root_checkpoint["model"])
    root_expert.set_calibration(0.0, 1.0, 0.0, 0.0)

    combined = RootExpertBlend(primary, root_expert).to(device)
    combined.set_root_strength(args.root_strength)
    temporal = ResidualTemporalCalibration(combined).to(device)
    return SequenceBoneCalibration(
        temporal, blend=args.bone_blend, symmetric=args.bone_symmetric
    ).to(device)


def _feasible(metrics: dict, baseline: dict) -> bool:
    return (
        0.90 <= metrics["pose_speed_ratio"] <= 1.10
        and metrics["mpjpe_m"] <= baseline["mpjpe_m"] + 0.001
        and metrics["danger_mpjpe_m"] <= baseline["danger_mpjpe_m"] + 0.003
        and metrics["danger_endpoint_mpjpe_m"]
        <= baseline["danger_endpoint_mpjpe_m"] * 1.02
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--pose-strength", type=float, default=0.35)
    parser.add_argument("--root-strength", type=float, default=1.0)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument(
        "--risk-adaptive", choices=("none", "probability", "hard"),
        default="none",
    )
    parser.add_argument("--danger-logit-bias", type=float, default=1.1)
    parser.add_argument("--calibrate-root-residual", action="store_true")
    parser.add_argument(
        "--root-windows", type=int, nargs="+", default=(3, 5, 9, 13, 21)
    )
    parser.add_argument(
        "--root-blends", type=float, nargs="+", default=(0.5, 0.75, 1.0)
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument(
        "--windows", type=int, nargs="+", default=(3, 5, 7, 9)
    )
    parser.add_argument(
        "--blends", type=float, nargs="+", default=(0.25, 0.50, 0.75, 1.0)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model = _build_model(args, device)
    temporal = model.base

    candidates = []
    settings = [(1, 0.0)] + [
        (window, blend) for window in args.windows for blend in args.blends
    ]
    for window, blend in settings:
        temporal.set_calibration(
            window, blend, args.risk_adaptive, args.danger_logit_bias
        )
        metrics = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        candidates.append({
            "window": window,
            "blend": blend,
            "feasible": False,
            "score": pose_selection_score(metrics),
            "validation": metrics,
        })
    baseline = candidates[0]
    for candidate in candidates:
        candidate["feasible"] = _feasible(
            candidate["validation"], baseline["validation"]
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = min(feasible or candidates, key=lambda item: item["score"])
    temporal.set_calibration(
        selected["window"], selected["blend"],
        args.risk_adaptive, args.danger_logit_bias,
    )

    root_candidates = []
    root_settings = [(1, 0.0)]
    if args.calibrate_root_residual:
        root_settings += [
            (window, blend)
            for window in args.root_windows for blend in args.root_blends
        ]
    for window, blend in root_settings:
        temporal.set_root_calibration(window, blend)
        metrics = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        root_candidates.append({
            "window": window,
            "blend": blend,
            "score": root_selection_score(metrics),
            "validation": metrics,
        })
    root_baseline = root_candidates[0]
    feasible_root = [candidate for candidate in root_candidates if (
        candidate["validation"]["root_error_m"]
        <= root_baseline["validation"]["root_error_m"] + 0.0005
        and candidate["validation"]["danger_root_error_m"]
        <= root_baseline["validation"]["danger_root_error_m"] + 0.001
        and candidate["validation"]["danger_endpoint_mpjpe_m"]
        <= root_baseline["validation"]["danger_endpoint_mpjpe_m"] * 1.01
    )]
    selected_root = min(
        feasible_root or root_candidates, key=lambda item: item["score"]
    )

    result = {
        "run": "p2_v11_residual_temporal_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "root_expert_checkpoint": str(args.root_expert_checkpoint),
            "pose_strength": args.pose_strength,
            "root_strength": args.root_strength,
            "bone_blend": args.bone_blend,
            "bone_symmetric": args.bone_symmetric,
            "risk_adaptive": args.risk_adaptive,
            "danger_logit_bias": args.danger_logit_bias,
        },
        "selected": {
            "window": selected["window"],
            "blend": selected["blend"],
            "root_window": selected_root["window"],
            "root_blend": selected_root["blend"],
        },
        "baseline_validation": baseline["validation"],
        "selected_validation": selected_root["validation"],
        "candidates": candidates,
        "root_candidates": root_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
