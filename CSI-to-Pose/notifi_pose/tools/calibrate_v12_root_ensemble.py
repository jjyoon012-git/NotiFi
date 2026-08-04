"""Validation-only root-seed ensemble with a locked pose ensemble."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

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
from .train_p2_v9_hybrid import root_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def _load(p2: dict, checkpoint: dict, device: str,
          pose_strength: float, root_strength: float):
    model = build_residual_hybrid(
        make_model(p2, device), checkpoint.get("residual_decoder", "subcarrier")
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.set_calibration(pose_strength, root_strength, 0.0, 0.0)
    return model


def simplex_weights(count: int, step: float) -> list[list[float]]:
    """Enumerate non-negative ensemble weights that sum exactly to one."""
    if count < 1:
        raise ValueError("at least one root checkpoint is required")
    units = round(1.0 / step)
    if step <= 0.0 or abs(units * step - 1.0) > 1e-8:
        raise ValueError("root weight step must evenly divide one")
    return [
        [value / units for value in values]
        for values in itertools.product(range(units + 1), repeat=count)
        if sum(values) == units
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--pose-weights", type=float, nargs="+", required=True)
    parser.add_argument("--root-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--root-expert-weights", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="legacy two-expert weight sweep",
    )
    parser.add_argument(
        "--root-weight-step", type=float, default=0.1,
        help="simplex resolution when three or more root experts are supplied",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--danger-logit-bias", type=float, default=1.1)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.pose_checkpoints) != len(args.pose_weights):
        raise ValueError("pose checkpoint and weight counts must match")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)
    pose_checkpoints = [
        _checked_checkpoint(path, device, args.exp)
        for path in args.pose_checkpoints
    ]
    pose = PoseModelEnsemble([
        _load(p2, checkpoint, device, 1.0, 0.0)
        for checkpoint in pose_checkpoints
    ], list(args.pose_weights)).to(device)

    root_checkpoints = [
        _checked_checkpoint(path, device, args.exp)
        for path in args.root_checkpoints
    ]
    if any(checkpoint.get("objective") != "root_only"
           for checkpoint in root_checkpoints):
        raise RuntimeError("all root checkpoints must have objective=root_only")
    root = PoseModelEnsemble([
        _load(p2, checkpoint, device, 0.0, 1.0)
        for checkpoint in root_checkpoints
    ]).to(device)
    rooted = RootExpertBlend(pose, root).to(device)
    rooted.set_root_strength(1.0)
    temporal = ResidualTemporalCalibration(rooted).to(device)
    temporal.set_calibration(
        args.window, args.blend, "probability", args.danger_logit_bias
    )
    model = SequenceBoneCalibration(
        temporal, blend=args.bone_blend, symmetric=args.bone_symmetric
    ).to(device)

    if len(root_checkpoints) == 2:
        weight_candidates = [
            [1.0 - value, value] for value in args.root_expert_weights
        ]
    else:
        weight_candidates = simplex_weights(
            len(root_checkpoints), args.root_weight_step
        )
    candidates = []
    for weights in weight_candidates:
        root.set_weights(weights)
        metrics = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        candidates.append({
            "root_weights": weights,
            "score": root_selection_score(metrics),
            "validation": metrics,
        })
    baseline = next(
        candidate for candidate in candidates
        if candidate["root_weights"][0] == 1.0
    )
    for candidate in candidates:
        metrics = candidate["validation"]
        reference = baseline["validation"]
        candidate["feasible"] = (
            metrics["root_error_m"] <= reference["root_error_m"] + 0.0005
            and metrics["danger_root_error_m"]
            <= reference["danger_root_error_m"] + 0.001
            and metrics["danger_endpoint_mpjpe_m"]
            <= reference["danger_endpoint_mpjpe_m"] * 1.01
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = min(feasible or candidates, key=lambda item: item["score"])
    report = {
        "run": "p2_v12_root_seed_ensemble_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "p2_checkpoint": str(args.p2_checkpoint),
            "pose_checkpoints": [str(path) for path in args.pose_checkpoints],
            "pose_weights": list(args.pose_weights),
            "root_checkpoints": [str(path) for path in args.root_checkpoints],
            "window": args.window,
            "blend": args.blend,
            "danger_logit_bias": args.danger_logit_bias,
            "bone_blend": args.bone_blend,
            "bone_symmetric": args.bone_symmetric,
        },
        "baseline_validation": baseline["validation"],
        "selected": {
            "root_weights": selected["root_weights"],
            "score": selected["score"],
        },
        "selected_validation": selected["validation"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selected": report["selected"],
        "validation": selected["validation"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
