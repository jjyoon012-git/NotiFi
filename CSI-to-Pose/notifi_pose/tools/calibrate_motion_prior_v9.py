"""Calibrate the V9C denoising prior on validation and evaluate test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..motion_prior_v9 import MotionPriorTrajectoryWrapper, TemporalMotionDenoiser
from ..seen_v4 import AlignmentRobustTrajectoryNet
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, report_path
from .train_seen_v4_trajectory import (
    calibrate_danger_bias,
    evaluate_classification,
    evaluate_trajectory,
    load_v3,
    make_loaders,
    selection_score,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp", default="single_split",
        choices=("single_split", "single_split_lmh_e01"),
    )
    parser.add_argument(
        "--v9-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v4_v9a_no_impact" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--prior-checkpoint", type=Path,
        default=C.WORK_ROOT / "priors" / "temporal_denoiser_v9" / "best_model.pt",
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
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.10, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--max-validation-speed", type=float, default=1.20)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=4.0)
    parser.add_argument("--minimum-danger-recall", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v4_v9c_motion_prior",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, loaders = make_loaders(args, device)
    _, validation, test = datasets

    trajectory = AlignmentRobustTrajectoryNet(load_v3(args, device)).to(device)
    trajectory_checkpoint = torch.load(
        args.v9_checkpoint, map_location=device, weights_only=False
    )
    trajectory.load_state_dict(trajectory_checkpoint["model"])
    trajectory.set_calibration(
        float(trajectory_checkpoint.get("pose_strength", 0.0)),
        float(trajectory_checkpoint.get("root_strength", 0.0)),
    )
    trajectory.eval()

    prior = TemporalMotionDenoiser().to(device)
    prior_checkpoint = torch.load(
        args.prior_checkpoint, map_location=device, weights_only=False
    )
    prior.load_state_dict(prior_checkpoint["model"])
    prior.eval()
    model = MotionPriorTrajectoryWrapper(trajectory, prior).to(device)

    candidates = []
    for strength in args.strengths:
        model.set_prior_strength(strength)
        metrics = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        candidates.append({
            "prior_strength": strength,
            "feasible_speed": 0.80 <= metrics["pose_speed_ratio"]
            <= args.max_validation_speed,
            "score": selection_score(metrics),
            "validation": metrics,
        })
        print(
            f"strength={strength:.2f} mpjpe={metrics['mpjpe_m'] * 100:.2f}cm "
            f"danger={metrics['danger_mpjpe_m'] * 100:.2f}cm "
            f"speed={metrics['pose_speed_ratio']:.3f}"
        )
    feasible = [item for item in candidates if item["feasible_speed"]]
    selected = min(feasible or candidates, key=lambda item: item["score"])

    model.set_prior_strength(0.0)
    baseline_test = evaluate_trajectory(
        model, loaders["test"], device, args.max_shift
    )
    model.set_prior_strength(selected["prior_strength"])
    test_metrics = evaluate_trajectory(
        model, loaders["test"], device, args.max_shift
    )
    selected_risk, risk_candidates = calibrate_danger_bias(
        model, loaders["val_class"], device, args.minimum_danger_recall
    )
    validation_classification = selected_risk["validation"]
    danger_bias = selected_risk["danger_logit_bias"]
    raw_test_classification = evaluate_classification(
        model, loaders["test_class"], device
    )
    test_classification = evaluate_classification(
        model, loaders["test_class"], device, danger_bias
    )
    result = {
        "run": "seen_v4_v9c_temporal_motion_prior",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_v9": report_path(args.v9_checkpoint),
        "source_prior": report_path(args.prior_checkpoint),
        "selected": {"prior_strength": selected["prior_strength"]},
        "selected_validation": selected["validation"],
        "baseline_test": baseline_test,
        "test": test_metrics,
        "validation_classification": validation_classification,
        "test_classification": test_classification,
        "raw_test_classification": raw_test_classification,
        "risk_calibration": {
            "minimum_validation_danger_recall": args.minimum_danger_recall,
            "selected_danger_logit_bias": danger_bias,
            "candidates": risk_candidates,
        },
        "shuffled_test": evaluate_model(
            model, ShuffledSignalDataset(test, args.seed),
            device, args.batch_size, 5,
        ),
        "calibration_candidates": candidates,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "prior_strength": selected["prior_strength"],
        "source_v9": report_path(args.v9_checkpoint),
        "source_prior": report_path(args.prior_checkpoint),
        "validation": selected["validation"],
        "test": test_metrics,
        "validation_classification": validation_classification,
        "test_classification": test_classification,
        "raw_test_classification": raw_test_classification,
        "danger_logit_bias": danger_bias,
    }, args.run_dir / "calibrated_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
