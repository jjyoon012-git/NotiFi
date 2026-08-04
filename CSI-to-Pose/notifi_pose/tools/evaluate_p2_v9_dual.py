"""Validation-gated P2/V9 pose model with the V9C root expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..hybrid_v10 import RootExpertBlend, build_residual_hybrid
from ..seen_v4 import AlignmentRobustTrajectoryNet
from ..trainer import set_seed
from .diagnose_observability import report_path
from .evaluate_sealed import make_model
from .train_p2_v9_hybrid import root_selection_score
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    load_v3,
    make_loaders,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-calibration", type=Path, required=True)
    parser.add_argument(
        "--root-expert-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v4_v9a_lmh_e01_multitask_recall93" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--root-expert-kind", choices=("v9", "p2_hybrid"), default="v9",
        help="architecture stored in --root-expert-checkpoint",
    )
    parser.add_argument(
        "--allow-unverified-root-protocol", action="store_true",
        help="allow legacy root checkpoints without protocol metadata",
    )
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
    parser.add_argument(
        "--exp", default="single_split_lmh_e01",
        choices=("single_split", "single_split_lmh_e01"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--minimum-root-gain", type=float, default=0.005)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument(
        "--pose-strength-override", type=float, default=None,
        help="validation-selected pose strength to use instead of the JSON default",
    )
    parser.add_argument(
        "--root-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)

    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
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
    if args.pose_strength_override is not None:
        selected = {**selected, "pose_strength": args.pose_strength_override}
    primary.set_calibration(
        selected["pose_strength"], selected["root_strength"],
        selected["class_strength"], selected["risk_strength"],
    )
    primary.eval()

    root_checkpoint = torch.load(
        args.root_expert_checkpoint, map_location=device, weights_only=False
    )
    checkpoint_protocol = root_checkpoint.get("protocol")
    if checkpoint_protocol is None and not args.allow_unverified_root_protocol:
        raise RuntimeError(
            "root expert checkpoint has no protocol metadata; "
            "use a clean-protocol checkpoint or explicitly allow legacy evaluation"
        )
    if checkpoint_protocol is not None and checkpoint_protocol != args.exp:
        raise RuntimeError(
            "root expert protocol mismatch: "
            f"checkpoint={checkpoint_protocol}, requested={args.exp}"
        )
    if args.root_expert_kind == "p2_hybrid":
        if root_checkpoint.get("objective") != "root_only":
            raise RuntimeError(
                "p2_hybrid root expert must be trained with objective=root_only"
            )
        root_expert = build_residual_hybrid(
            make_model(p2_checkpoint, device),
            root_checkpoint.get("residual_decoder", "dense"),
        ).to(device)
        root_expert.load_state_dict(root_checkpoint["model"])
        root_expert.set_calibration(0.0, 1.0, 0.0, 0.0)
    else:
        root_expert = AlignmentRobustTrajectoryNet(load_v3(args, device)).to(device)
        missing, unexpected = root_expert.load_state_dict(
            root_checkpoint["model"], strict=False
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
            float(root_checkpoint.get("pose_strength", 0.0)),
            float(root_checkpoint.get("root_strength", 0.0)),
        )
    root_expert.eval()
    model = RootExpertBlend(primary, root_expert).to(device)

    candidates = []
    for strength in args.root_strengths:
        model.set_root_strength(strength)
        metrics = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        candidates.append({
            "root_strength": strength,
            "score": root_selection_score(metrics),
            "validation": metrics,
        })
    baseline = candidates[0]
    candidate = min(candidates, key=lambda item: item["score"])
    gain = (
        baseline["validation"]["root_error_m"]
        - candidate["validation"]["root_error_m"]
    )
    if gain < args.minimum_root_gain:
        candidate = baseline
    model.set_root_strength(candidate["root_strength"])

    danger_bias = float(selected["danger_logit_bias"])
    result = {
        "run": "p2_v9_dual_root_v10",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_p2": report_path(args.p2_checkpoint),
        "source_hybrid": report_path(args.hybrid_checkpoint),
        "source_root_expert": report_path(args.root_expert_checkpoint),
        "root_expert_kind": args.root_expert_kind,
        "selected": {
            **selected,
            "root_expert_strength": candidate["root_strength"],
        },
        "selected_validation": candidate["validation"],
        "root_candidates": candidates,
    }
    if not args.skip_test_eval:
        result["test"] = evaluate_trajectory(
            model, loaders["test"], device, args.max_shift
        )
        result["test_classification"] = evaluate_classification(
            model, loaders["test_class"], device, danger_bias
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "selected": result["selected"],
        "source_p2": result["source_p2"],
        "source_hybrid": result["source_hybrid"],
        "source_root_expert": result["source_root_expert"],
        "validation": result["selected_validation"],
        **({
            "test": result["test"],
            "test_classification": result["test_classification"],
        } if "test" in result else {}),
    }, args.output.with_name("dual_model.pt"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
