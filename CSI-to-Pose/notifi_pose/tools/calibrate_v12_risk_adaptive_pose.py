"""Validation-only risk-gated mixture of clean pose specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..hybrid_v10 import (
    RiskAdaptivePoseBlend,
    RootExpertBlend,
    SequenceBoneCalibration,
    build_residual_hybrid,
)
from ..trainer import set_seed
from .calibrate_v11_residual_temporal import (
    ResidualTemporalCalibration,
    _checked_checkpoint,
)
from .evaluate_sealed import make_model
from .train_p2_v9_hybrid import pose_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def _pose_model(p2: dict, checkpoint: dict, device: str):
    model = build_residual_hybrid(
        make_model(p2, device), checkpoint.get("residual_decoder", "subcarrier")
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.set_calibration(1.0, 0.0, 0.0, 0.0)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--danger-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--non-danger-strengths", type=float, nargs="+", default=(1.0,),
    )
    parser.add_argument(
        "--gate-modes", nargs="+", choices=("probability", "hard"),
        default=("probability", "hard"),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=(21, 25, 31))
    parser.add_argument("--blends", type=float, nargs="+", default=(0.75, 1.0))
    parser.add_argument("--danger-logit-bias", type=float, default=1.1)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)
    primary_checkpoint = _checked_checkpoint(
        args.primary_checkpoint, device, args.exp
    )
    expert_checkpoint = _checked_checkpoint(
        args.expert_checkpoint, device, args.exp
    )
    adaptive = RiskAdaptivePoseBlend(
        _pose_model(p2, primary_checkpoint, device),
        _pose_model(p2, expert_checkpoint, device),
        danger_logit_bias=args.danger_logit_bias,
    ).to(device)

    root_checkpoint = _checked_checkpoint(
        args.root_expert_checkpoint, device, args.exp
    )
    root_expert = build_residual_hybrid(
        make_model(p2, device),
        root_checkpoint.get("residual_decoder", "direct_root"),
    ).to(device)
    root_expert.load_state_dict(root_checkpoint["model"])
    root_expert.set_calibration(0.0, 1.0, 0.0, 0.0)
    rooted = RootExpertBlend(adaptive, root_expert).to(device)
    rooted.set_root_strength(1.0)
    temporal = ResidualTemporalCalibration(rooted).to(device)
    model = SequenceBoneCalibration(
        temporal, blend=args.bone_blend, symmetric=args.bone_symmetric
    ).to(device)

    candidates = []
    for gate_mode in args.gate_modes:
        for non_danger_strength in args.non_danger_strengths:
            for danger_strength in args.danger_strengths:
                adaptive.set_calibration(
                    non_danger_strength, danger_strength,
                    args.danger_logit_bias, gate_mode,
                )
                for window in args.windows:
                    for blend in args.blends:
                        temporal.set_calibration(
                            window, blend, "probability", args.danger_logit_bias
                        )
                        metrics = evaluate_trajectory(
                            model, loaders["val"], device, args.max_shift
                        )
                        candidates.append({
                            "gate_mode": gate_mode,
                            "non_danger_strength": non_danger_strength,
                            "danger_strength": danger_strength,
                            "window": window,
                            "blend": blend,
                            "score": pose_selection_score(metrics),
                            "validation": metrics,
                        })
    baseline = next(
        candidate for candidate in candidates
        if candidate["non_danger_strength"] == 1.0
        and candidate["danger_strength"] == 1.0
        and candidate["window"] == 31 and candidate["blend"] == 1.0
    )
    for candidate in candidates:
        metrics = candidate["validation"]
        reference = baseline["validation"]
        candidate["feasible"] = (
            0.90 <= metrics["pose_speed_ratio"] <= 1.10
            and metrics["mpjpe_m"] <= reference["mpjpe_m"] + 0.0005
            and metrics["danger_mpjpe_m"] <= reference["danger_mpjpe_m"]
            and metrics["danger_distal_mpjpe_m"]
            <= reference["danger_distal_mpjpe_m"]
            and metrics["danger_endpoint_mpjpe_m"]
            <= reference["danger_endpoint_mpjpe_m"] + 0.0005
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = min(feasible or candidates, key=lambda item: item["score"])
    report = {
        "run": "p2_v12_risk_adaptive_pose_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "primary_checkpoint": str(args.primary_checkpoint),
            "expert_checkpoint": str(args.expert_checkpoint),
            "root_expert_checkpoint": str(args.root_expert_checkpoint),
            "danger_logit_bias": args.danger_logit_bias,
            "bone_blend": args.bone_blend,
            "bone_symmetric": args.bone_symmetric,
        },
        "baseline_validation": baseline["validation"],
        "selected": {
            key: selected[key] for key in (
                "gate_mode", "non_danger_strength", "danger_strength",
                "window", "blend", "score",
            )
        },
        "selected_validation": selected["validation"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "baseline": baseline["validation"],
        "selected": report["selected"],
        "validation": selected["validation"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
