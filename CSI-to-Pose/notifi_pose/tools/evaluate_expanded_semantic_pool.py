"""One-shot fixed-test evaluation of the validation-locked 30-motion pool."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_expanded_semantic_pool import slice_candidate_data
from .calibrate_motion_profile_warping import (
    activity_sources,
    monotonic_energy_warp,
    parse_promoted_selection,
)
from .calibrate_part_motion_profile_reranking import prepare, render_mixture
from .calibrate_semantic_candidate_prior import (
    add_arguments,
    base_adjusted_logits,
    candidate_semantic_scores,
)
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument(
        "--expanded-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_expanded_semantic_pool"
        / "refined_calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_expanded_semantic_pool"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    part_config = parse_promoted_selection(json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    ))
    calibration = json.loads(
        args.expanded_calibration.read_text(encoding="utf-8")
    )
    selected = calibration["selection"]["name"]
    match = re.fullmatch(
        r"pool(?P<pool>\d+)_a(?P<action>\d{4})_r(?P<risk>\d{4})"
        r"_t(?P<temperature>\d{3})_top(?P<top>\d+)",
        selected,
    )
    if match is None:
        raise RuntimeError(f"unsupported locked expanded selection: {selected}")
    pool_size = int(match["pool"])
    action_weight = int(match["action"]) / 1000.0
    risk_weight = int(match["risk"]) / 1000.0
    temperature = int(match["temperature"]) / 100.0
    top_k = int(match["top"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = slice_candidate_data(
        prepare(args, "test", device, pool_top_k=pool_size), pool_size
    )
    base = base_adjusted_logits(data, part_config)
    action_score, risk_score = candidate_semantic_scores(data)
    activity = activity_sources(data)["scalar_limbs"]

    def predict(adjusted, temp, choices):
        candidate = render_mixture(data, adjusted, temp, choices)
        warped = monotonic_energy_warp(
            candidate, activity, data["target_valid"], 0.50, 0.30
        )
        return 0.25 * data["baseline"] + 0.75 * warped

    semantic = predict(
        base + 0.35 * action_score + 0.50 * risk_score, 0.50, 5
    )
    expanded = predict(
        base + action_weight * action_score + risk_weight * risk_score,
        temperature, top_k,
    )
    metrics = {
        "kp6_semantic_prior_pool30_reference": _metric_batch(
            semantic, data["target_pose"], data["target_valid"], data["target_risk"]
        ),
        "kp6_expanded_semantic_pool": _metric_batch(
            expanded, data["target_pose"], data["target_valid"], data["target_risk"]
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
        "expanded_calibration": report_path(args.expanded_calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
