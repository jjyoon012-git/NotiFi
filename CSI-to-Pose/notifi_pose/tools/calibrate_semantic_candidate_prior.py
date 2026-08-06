"""Validation-only explicit CSI action/risk consistency for KP6 candidates."""

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
from .calibrate_part_motion_profile_reranking import (
    prepare,
    render_mixture,
    standardize,
    weighted_part_distance,
)
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def candidate_semantic_scores(data):
    indices = data["pool"]["indices"]
    candidate_class = data["checkpoint"]["train_class"][indices]
    candidate_risk = torch.where(
        candidate_class < 9, 0,
        torch.where(candidate_class < 12, 1, 2),
    )
    risk_log = data["risk_probability"].clamp_min(1e-6).log()
    risk_log = risk_log.gather(1, candidate_risk)
    action_log = data["pool"]["action_log_probability"]
    return standardize(action_log), standardize(risk_log)


def base_adjusted_logits(data, part_config):
    patterns = {
        "uniform": (1, 1, 1, 1, 1, 1),
        "distal": (1.2, 0.7, 1.4, 1.4, 1.4, 1.4),
        "limbs": (0.8, 0.6, 1.7, 1.7, 1.7, 1.7),
    }
    part_distance = weighted_part_distance(
        data["part_distance"], patterns[part_config["pattern"]]
    )
    return (
        data["logits"] - 0.20 * data["scalar_distance"]
        - part_config["weight"] * part_distance
    )


def add_arguments(parser):
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
        "--part-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "reranking_calibration.json",
    )
    parser.add_argument(
        "--warp-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_warp"
        / "warping_calibration.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_semantic_prior"
        / "calibration.json",
    )
    args = parser.parse_args()
    part_calibration = json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    )
    part_config = parse_promoted_selection(part_calibration)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "val", device)
    base = base_adjusted_logits(data, part_config)
    action_score, risk_score = candidate_semantic_scores(data)
    activity = activity_sources(data)["scalar_limbs"]
    metrics = {}
    for action_weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        for risk_weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75):
            adjusted = (
                base + action_weight * action_score + risk_weight * risk_score
            )
            candidate = render_mixture(data, adjusted, 0.50, 5)
            warped = monotonic_energy_warp(
                candidate, activity, data["inference_valid"], 0.50, 0.30
            )
            key = (
                f"a{int(action_weight * 1000):04d}"
                f"_r{int(risk_weight * 1000):04d}"
            )
            metrics[key] = _metric_batch(
                0.25 * data["baseline"] + 0.75 * warped,
                data["target_pose"], data["target_valid"], data["target_risk"],
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_semantic_candidate_prior",
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
