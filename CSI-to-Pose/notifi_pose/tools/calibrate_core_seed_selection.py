"""Validation-only selection between independently trained retrieval cores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import (
    prepare,
    render_mixture,
    weighted_part_distance,
)
from .calibrate_risk_adaptive_blend import adaptive_strength
from .calibrate_semantic_candidate_prior import candidate_semantic_scores
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--current-selector", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "best_model.pt",
    )
    parser.add_argument(
        "--current-reranker", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "best_model.pt",
    )
    parser.add_argument(
        "--alternative-selector", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_mpr_selector_seed23"
        / "best_model.pt",
    )
    parser.add_argument(
        "--alternative-reranker", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_mpr_reranker_seed23"
        / "best_model.pt",
    )
    parser.add_argument(
        "--scalar-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed79"
        / "best_model.pt",
    )
    parser.add_argument(
        "--part-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed101"
        / "best_model.pt",
    )
    parser.add_argument(
        "--adaptive-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend"
        / "calibration.json",
    )


def core_args(args, selector: Path, reranker: Path):
    return argparse.Namespace(
        exp=args.exp,
        selector_checkpoint=selector,
        reranker_checkpoint=reranker,
        scalar_profile_checkpoint=args.scalar_profile_checkpoint,
        part_profile_checkpoint=args.part_profile_checkpoint,
    )


def predict_locked(data, adaptive_config):
    inference_valid = data["inference_valid"]
    action_score, risk_score = candidate_semantic_scores(data)
    part_distance = weighted_part_distance(
        data["part_distance"], (0.8, 0.6, 1.7, 1.7, 1.7, 1.7)
    )
    adjusted = (
        data["logits"] - 0.20 * data["scalar_distance"]
        - 0.10 * part_distance
        + 0.35 * action_score + 0.50 * risk_score
    )
    candidate = render_mixture(data, adjusted, 0.50, 5)
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    warped = monotonic_energy_warp(
        candidate, activity, inference_valid, 0.50, 0.30
    )
    strength = adaptive_strength(
        data["risk_probability"], adaptive_config["base"],
        adaptive_config["danger_delta"],
        adaptive_config["uncertainty_delta"],
    )
    return (
        (1.0 - strength)[:, None, None, None] * data["baseline"]
        + strength[:, None, None, None] * warped
    )


@torch.no_grad()
def evaluate_core(args, selector, reranker, split, adaptive_config, device):
    data = prepare(core_args(args, selector, reranker), split, device)
    predicted = predict_locked(data, adaptive_config)
    return _metric_batch(
        predicted, data["target_pose"], data["target_valid"],
        data["target_risk"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_core_seed_selection"
        / "calibration.json",
    )
    args = parser.parse_args()
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configurations = {
        "seed17": (args.current_selector, args.current_reranker),
        "seed23": (args.alternative_selector, args.alternative_reranker),
    }
    metrics = {
        name: evaluate_core(
            args, selector, reranker, "val", adaptive_config, device
        )
        for name, (selector, reranker) in configurations.items()
    }
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_retrieval_core_seed",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "locked_pipeline": {
            "profile_seeds": "scalar79_part101",
            "profile_weights": "scalar0200_part0100_limbs",
            "semantic": "action0350_risk0500",
            "warp": "scalar_limbs_f30_w050",
            "adaptive_blend": adaptive_config,
        },
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "configurations": {
            name: {
                "selector": report_path(selector),
                "reranker": report_path(reranker),
            }
            for name, (selector, reranker) in configurations.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"], "metrics": metrics,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
