"""Validation-only calibration of two clean pose experts.

The root trajectory is supplied by a separately trained root expert.  Model
mixture and temporal calibration are selected together under explicit fall
quality constraints; the test split is never constructed or evaluated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ..hybrid_v10 import (
    PoseModelEnsemble,
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


def _load_pose_model(p2_checkpoint: dict, checkpoint: dict,
                     device: str) -> nn.Module:
    model = build_residual_hybrid(
        make_model(p2_checkpoint, device),
        checkpoint.get("residual_decoder", "dense"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.set_calibration(1.0, 0.0, 0.0, 0.0)
    return model


def _fall_feasible(metrics: dict, baseline: dict) -> bool:
    return (
        0.90 <= metrics["pose_speed_ratio"] <= 1.10
        and metrics["danger_mpjpe_m"] <= baseline["danger_mpjpe_m"] + 0.001
        and metrics["danger_distal_mpjpe_m"]
        <= baseline["danger_distal_mpjpe_m"] + 0.001
        and metrics["danger_endpoint_mpjpe_m"]
        <= baseline["danger_endpoint_mpjpe_m"] + 0.001
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--expert-weights", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=(21, 25, 31))
    parser.add_argument("--blends", type=float, nargs="+", default=(0.75, 1.0))
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument(
        "--risk-adaptive", choices=("none", "probability", "hard"),
        default="probability",
    )
    parser.add_argument("--danger-logit-bias", type=float, default=1.1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(weight < 0.0 or weight > 1.0 for weight in args.expert_weights):
        raise ValueError("expert weights must be in [0, 1]")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    primary_checkpoint = _checked_checkpoint(
        args.primary_checkpoint, device, args.exp
    )
    expert_checkpoint = _checked_checkpoint(
        args.expert_checkpoint, device, args.exp
    )
    ensemble = PoseModelEnsemble([
        _load_pose_model(p2_checkpoint, primary_checkpoint, device),
        _load_pose_model(p2_checkpoint, expert_checkpoint, device),
    ]).to(device)

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

    combined = RootExpertBlend(ensemble, root_expert).to(device)
    combined.set_root_strength(1.0)
    temporal = ResidualTemporalCalibration(combined).to(device)
    model = SequenceBoneCalibration(
        temporal, blend=args.bone_blend, symmetric=args.bone_symmetric
    ).to(device)

    settings = [
        (weight, window, blend)
        for weight in args.expert_weights
        for window in args.windows
        for blend in args.blends
    ]
    candidates = []
    for weight, window, blend in settings:
        ensemble.set_weights([1.0 - weight, weight])
        temporal.set_calibration(
            window, blend, args.risk_adaptive, args.danger_logit_bias
        )
        metrics = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        candidates.append({
            "expert_weight": weight,
            "window": window,
            "blend": blend,
            "score": pose_selection_score(metrics),
            "validation": metrics,
        })

    baseline = next(
        candidate for candidate in candidates
        if candidate["expert_weight"] == 0.0
        and candidate["window"] == 31
        and candidate["blend"] == 1.0
    )
    for candidate in candidates:
        candidate["feasible"] = _fall_feasible(
            candidate["validation"], baseline["validation"]
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = min(feasible or candidates, key=lambda item: item["score"])

    result = {
        "run": "p2_v12_pose_ensemble_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "primary_checkpoint": str(args.primary_checkpoint),
            "expert_checkpoint": str(args.expert_checkpoint),
            "root_expert_checkpoint": str(args.root_expert_checkpoint),
            "bone_blend": args.bone_blend,
            "bone_symmetric": args.bone_symmetric,
            "risk_adaptive": args.risk_adaptive,
            "danger_logit_bias": args.danger_logit_bias,
        },
        "baseline_validation": baseline["validation"],
        "selected": {
            key: selected[key]
            for key in ("expert_weight", "window", "blend", "score")
        },
        "selected_validation": selected["validation"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
