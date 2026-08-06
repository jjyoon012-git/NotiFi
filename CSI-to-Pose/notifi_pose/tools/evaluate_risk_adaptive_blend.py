"""One-shot fixed-test evaluation of risk-adaptive KP6 blending."""

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
from .calibrate_risk_adaptive_blend import adaptive_strength
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
        "--adaptive-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend"
        / "calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    part_config = parse_promoted_selection(json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    ))
    calibration = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )
    selected = calibration["selection"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "test", device)
    base = base_adjusted_logits(data, part_config)
    action_score, risk_score = candidate_semantic_scores(data)
    adjusted = base + 0.35 * action_score + 0.50 * risk_score
    candidate = render_mixture(data, adjusted, 0.50, 5)
    warped = monotonic_energy_warp(
        candidate, activity_sources(data)["scalar_limbs"],
        data["inference_valid"], 0.50, 0.30,
    )
    fixed = 0.25 * data["baseline"] + 0.75 * warped
    strength = adaptive_strength(
        data["risk_probability"], selected["base"],
        selected["danger_delta"], selected["uncertainty_delta"],
    )
    adaptive = (
        (1.0 - strength)[:, None, None, None] * data["baseline"]
        + strength[:, None, None, None] * warped
    )
    metrics = {
        "kp6_semantic_warp": _metric_batch(
            fixed, data["target_pose"], data["target_valid"], data["target_risk"]
        ),
        "kp6_risk_adaptive": _metric_batch(
            adaptive, data["target_pose"],
            data["target_valid"], data["target_risk"],
        ),
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation",
        "fixed_configuration": selected,
        "test_strength_summary": {
            "mean": float(strength.mean()),
            "danger_probability_weighted_mean": float(
                (strength * data["risk_probability"][:, 2]).sum()
                / data["risk_probability"][:, 2].sum().clamp_min(1e-6)
            ),
        },
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "adaptive_calibration": report_path(args.adaptive_calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
