"""Validation-only risk/uncertainty-adaptive pose-prior blending for KP6."""

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


def adaptive_strength(
    risk_probability,
    base,
    danger_delta,
    uncertainty_delta,
):
    entropy = -(
        risk_probability.clamp_min(1e-6)
        * risk_probability.clamp_min(1e-6).log()
    ).sum(-1) / torch.log(risk_probability.new_tensor(3.0))
    return (
        base
        + danger_delta * risk_probability[:, 2]
        + uncertainty_delta * entropy
    ).clamp(0.40, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend"
        / "calibration.json",
    )
    args = parser.parse_args()
    part_config = parse_promoted_selection(json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    ))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "val", device)
    base_logits = base_adjusted_logits(data, part_config)
    action_score, risk_score = candidate_semantic_scores(data)
    adjusted = base_logits + 0.35 * action_score + 0.50 * risk_score
    candidate = render_mixture(data, adjusted, 0.50, 5)
    warped = monotonic_energy_warp(
        candidate, activity_sources(data)["scalar_limbs"],
        data["inference_valid"], 0.50, 0.30,
    )
    metrics = {}
    strengths = {}
    for base in (0.65, 0.70, 0.75, 0.80, 0.85):
        for danger_delta in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
            for uncertainty_delta in (-0.10, 0.0, 0.10):
                strength = adaptive_strength(
                    data["risk_probability"], base,
                    danger_delta, uncertainty_delta,
                )
                predicted = (
                    (1.0 - strength)[:, None, None, None] * data["baseline"]
                    + strength[:, None, None, None] * warped
                )
                key = (
                    f"b{int(base * 1000):03d}"
                    f"_d{int((danger_delta + 0.20) * 1000):03d}"
                    f"_u{int((uncertainty_delta + 0.15) * 1000):03d}"
                )
                metrics[key] = _metric_batch(
                    predicted, data["target_pose"], data["target_valid"],
                    data["target_risk"],
                )
                strengths[key] = {
                    "base": base,
                    "danger_delta": danger_delta,
                    "uncertainty_delta": uncertainty_delta,
                    "mean": float(strength.mean()),
                    "danger_probability_weighted_mean": float(
                        (strength * data["risk_probability"][:, 2]).sum()
                        / data["risk_probability"][:, 2].sum().clamp_min(1e-6)
                    ),
                }
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_risk_adaptive_blend",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "locked_semantic": "a0350_r0500",
        "locked_warp": "scalar_limbs_f30_w050",
        "selection": {
            "name": best, "score": scores[best], **strengths[best],
        },
        "scores": scores,
        "metrics": metrics,
        "strengths": strengths,
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
