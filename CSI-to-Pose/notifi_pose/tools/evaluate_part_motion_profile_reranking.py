"""One-shot fixed-test evaluation of anatomical motion-profile KP5."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_part_motion_profile_reranking import (
    prepare,
    render_mixture,
    weighted_part_distance,
)
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--scalar-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71" / "best_model.pt",
    )
    parser.add_argument(
        "--part-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83" / "best_model.pt",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "reranking_calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    selected = calibration["selection"]["name"]
    match = re.fullmatch(
        r"p(?P<weight>\d{4})_(?P<pattern>uniform|distal|limbs)"
        r"_t(?P<temperature>\d{3})_top(?P<top>\d+)"
        r"_s(?P<strength>\d{3})",
        selected,
    )
    if match is None:
        raise RuntimeError(f"unsupported locked validation selection: {selected}")
    patterns = {
        "uniform": (1, 1, 1, 1, 1, 1),
        "distal": (1.2, 0.7, 1.4, 1.4, 1.4, 1.4),
        "limbs": (0.8, 0.6, 1.7, 1.7, 1.7, 1.7),
    }
    part_weight = int(match["weight"]) / 1000.0
    temperature = int(match["temperature"]) / 100.0
    top_k = int(match["top"])
    strength = int(match["strength"]) / 1000.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "test", device)
    part_distance = weighted_part_distance(
        data["part_distance"], patterns[match["pattern"]]
    )
    adjusted = (
        data["logits"] - 0.20 * data["scalar_distance"]
        - part_weight * part_distance
    )
    candidate = render_mixture(data, adjusted, temperature, top_k)
    predicted = (1.0 - strength) * data["baseline"] + strength * candidate
    metrics = {
        "locked_kp2_dh": _metric_batch(
            data["baseline"], data["target_pose"],
            data["target_valid"], data["target_risk"],
        ),
        "kp5_part_motion_profile": _metric_batch(
            predicted, data["target_pose"],
            data["target_valid"], data["target_risk"],
        ),
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation",
        "fixed_configuration": calibration["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
        "scalar_profile_checkpoint": report_path(args.scalar_profile_checkpoint),
        "part_profile_checkpoint": report_path(args.part_profile_checkpoint),
        "calibration": report_path(args.calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
