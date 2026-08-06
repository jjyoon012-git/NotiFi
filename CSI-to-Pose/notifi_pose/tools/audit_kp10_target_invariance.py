"""Verify that KP10 predictions are invariant to poisoned evaluation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from .audit_kp10_paired_bootstrap import kp10_prediction
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .diagnose_observability import report_path


def poison_evaluation_targets(data: dict) -> dict:
    poisoned = dict(data)
    poisoned["target_pose"] = torch.full_like(data["target_pose"], 123.0)
    poisoned["target_valid"] = ~data["target_valid"]
    poisoned["target_class"] = (
        data["target_class"] + 7
    ) % C.N_CLASSES
    poisoned["target_risk"] = (
        data["target_risk"] + 1
    ) % C.N_RISK
    poisoned["pool"] = dict(data["pool"])
    poisoned["pool"]["target_cost"] = torch.full_like(
        data["pool"]["target_cost"], -999.0
    )
    return poisoned


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--profile-ranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker_seed127"
        / "best_model.pt",
    )
    parser.add_argument(
        "--strength-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "target_invariance_validation.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, args.split, device)
    reference = kp10_prediction(data, args, device)
    poisoned = kp10_prediction(poison_evaluation_targets(data), args, device)
    maximum = float((reference - poisoned).abs().max())
    exact = bool(torch.equal(reference, poisoned))
    result = {
        "status": "passed" if exact else "failed",
        "protocol": args.exp,
        "split": args.split,
        "test_split_touched": False,
        "test_used_for_selection": False,
        "purpose": "inference target-invariance audit only",
        "poisoned_fields": [
            "target_pose", "target_valid", "target_class", "target_risk",
            "pool.target_cost",
        ],
        "exact_prediction_equality": exact,
        "maximum_absolute_pose_difference": maximum,
        "trials": int(len(reference)),
        "strength_calibration": report_path(args.strength_calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
