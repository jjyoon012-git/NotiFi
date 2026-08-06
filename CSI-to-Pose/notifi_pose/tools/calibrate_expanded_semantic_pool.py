"""Validation-only larger train-motion pool for the CSI semantic prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
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


CANDIDATE_KEYS = (
    "indices", "retrieval_score", "action_log_probability", "target_cost",
)
TENSOR_KEYS = (
    "logits", "scalar_distance", "part_distance",
    "candidate_scalar_profiles", "candidate_part_profiles",
)


def slice_candidate_data(data, candidates):
    result = dict(data)
    result["pool"] = dict(data["pool"])
    for key in CANDIDATE_KEYS:
        result["pool"][key] = data["pool"][key][:, :candidates]
    for key in TENSOR_KEYS:
        result[key] = data[key][:, :candidates]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_expanded_semantic_pool"
        / "coarse_calibration.json",
    )
    args = parser.parse_args()
    part_calibration = json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    )
    part_config = parse_promoted_selection(part_calibration)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = prepare(args, "val", device, pool_top_k=50)
    metrics = {}
    for pool_size in (20, 30, 50):
        data = slice_candidate_data(full, pool_size)
        base = base_adjusted_logits(data, part_config)
        action_score, risk_score = candidate_semantic_scores(data)
        activity = activity_sources(data)["scalar_limbs"]
        for action_weight in (0.20, 0.35, 0.50, 0.75, 1.0):
            for risk_weight in (0.25, 0.50, 0.75, 1.0):
                adjusted = (
                    base + action_weight * action_score + risk_weight * risk_score
                )
                candidate = render_mixture(data, adjusted, 0.50, 5)
                warped = monotonic_energy_warp(
                    candidate, activity, data["target_valid"], 0.50, 0.30
                )
                key = (
                    f"pool{pool_size}_a{int(action_weight * 1000):04d}"
                    f"_r{int(risk_weight * 1000):04d}_t050_top5"
                )
                metrics[key] = _metric_batch(
                    0.25 * data["baseline"] + 0.75 * warped,
                    data["target_pose"], data["target_valid"],
                    data["target_risk"],
                )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_expanded_semantic_pool_coarse",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "base_part_selection": part_config,
        "locked_warp": "scalar_limbs_f30_w050_b750",
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "part_calibration": report_path(args.part_calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"], "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
