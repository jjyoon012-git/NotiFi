"""One-shot fixed-test evaluation of validation-locked CSI time warping."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_motion_profile_warping import (
    activity_sources,
    add_common_arguments,
    monotonic_energy_warp,
    parse_promoted_selection,
    promoted_candidate,
)
from .calibrate_part_motion_profile_reranking import prepare
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        "--warp-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_warp"
        / "warping_calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_warp" / "test_fixed.json",
    )
    args = parser.parse_args()
    part_calibration = json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    )
    base_config = parse_promoted_selection(part_calibration)
    warp_calibration = json.loads(
        args.warp_calibration.read_text(encoding="utf-8")
    )
    selected = warp_calibration["selection"]["name"]
    match = re.fullmatch(
        r"(?P<source>scalar|parts_mean|limbs|scalar_limbs)"
        r"_f(?P<floor>\d{2})_w(?P<warp>\d{3})_b(?P<blend>\d{3})",
        selected,
    )
    if match is None:
        raise RuntimeError(f"unsupported locked warp selection: {selected}")
    floor = int(match["floor"]) / 100.0
    warp_strength = int(match["warp"]) / 100.0
    blend = int(match["blend"]) / 1000.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "test", device)
    candidate = promoted_candidate(data, base_config)
    activity = activity_sources(data)[match["source"]]
    warped = monotonic_energy_warp(
        candidate, activity, data["target_valid"], warp_strength, floor
    )
    predicted = (1.0 - blend) * data["baseline"] + blend * warped
    unwarped = (
        (1.0 - base_config["strength"]) * data["baseline"]
        + base_config["strength"] * candidate
    )
    metrics = {
        "kp5_part_motion_profile": _metric_batch(
            unwarped, data["target_pose"],
            data["target_valid"], data["target_risk"],
        ),
        "kp5_motion_warp": _metric_batch(
            predicted, data["target_pose"],
            data["target_valid"], data["target_risk"],
        ),
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation",
        "base_configuration": base_config,
        "fixed_configuration": warp_calibration["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "part_calibration": report_path(args.part_calibration),
        "warp_calibration": report_path(args.warp_calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
