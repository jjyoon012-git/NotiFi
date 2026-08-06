"""Compare frozen pose experts on validation-only danger geometry metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..hybrid_v10 import (
    PoseModelEnsemble,
    SequenceBoneCalibration,
    SharedBackboneCache,
    SharedBackboneExecution,
)
from ..trainer import set_seed
from .calibrate_v11_residual_temporal import ResidualTemporalCalibration
from .evaluate_v12_final import _load_hybrid
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


SUMMARY_KEYS = (
    "mpjpe_m",
    "dynamic_mpjpe_m",
    "pose_speed_ratio",
    "danger_pose_mpjpe_m",
    "danger_pose_distal_mpjpe_m",
    "danger_pose_endpoint_mpjpe_m",
    "danger_speed_correlation",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--ensemble-weight-candidates", nargs="*", default=(),
        help="comma-separated weights in checkpoint order",
    )
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--danger-logit-bias", type=float, default=1.1)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)

    first, first_checkpoint = _load_hybrid(
        p2, args.checkpoints[0], args.exp, device, 1.0, 0.0
    )
    shared_backbone = SharedBackboneCache(first.base).to(device)
    first.base = shared_backbone
    candidates = []
    skipped = []
    accepted_models = []
    for index, path in enumerate(args.checkpoints):
        try:
            if index == 0:
                model, checkpoint = first, first_checkpoint
            else:
                model, checkpoint = _load_hybrid(
                    p2, path, args.exp, device, 1.0, 0.0, shared_backbone
                )
        except RuntimeError as error:
            skipped.append({"checkpoint": str(path), "reason": str(error)})
            continue
        if checkpoint.get("objective") != "pose_only":
            raise RuntimeError(f"pose candidate is not pose-only: {path}")
        execution = SharedBackboneExecution(model, shared_backbone).to(device)
        metrics = evaluate_trajectory(
            execution, loaders["val"], device, args.max_shift
        )
        candidates.append({
            "checkpoint": str(path),
            "residual_decoder": checkpoint.get("residual_decoder"),
            "epoch": int(checkpoint.get("epoch", -1)),
            "training_config": checkpoint.get("training_config", {}),
            "validation": {
                key: float(metrics[key]) for key in SUMMARY_KEYS
            },
        })
        accepted_models.append(model)
        del execution

    ensembles = []
    for value in args.ensemble_weight_candidates:
        weights = [float(item) for item in value.split(",")]
        if len(weights) != len(accepted_models):
            raise ValueError(
                "ensemble weights must match protocol-valid checkpoints"
            )
        ensemble = PoseModelEnsemble(accepted_models, weights).to(device)
        temporal = ResidualTemporalCalibration(ensemble).to(device)
        temporal.set_calibration(
            args.window, args.blend, "probability", args.danger_logit_bias
        )
        calibrated = SequenceBoneCalibration(
            temporal, blend=args.bone_blend, symmetric=True
        ).to(device)
        execution = SharedBackboneExecution(
            calibrated, shared_backbone
        ).to(device)
        metrics = evaluate_trajectory(
            execution, loaders["val"], device, args.max_shift
        )
        ensembles.append({
            "weights": list(ensemble.weights.cpu().tolist()),
            "validation": {
                key: float(metrics[key]) for key in SUMMARY_KEYS
            },
        })

    result = {
        "run": "p2_v13_pose_candidate_audit",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "p2_checkpoint": str(args.p2_checkpoint),
        "candidates": candidates,
        "skipped": skipped,
        "ensembles": ensembles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "candidates": {
            item["checkpoint"]: item["validation"] for item in candidates
        },
        "skipped": skipped,
        "ensembles": ensembles,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
