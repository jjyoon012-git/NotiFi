"""Validation-only calibration of monotonic CSI progress time warping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import (
    MotionProgressHead, ProfileCandidateRanker, TemporalMotionSelector,
)
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_independent_risk_fusion import classifier_outputs
from .calibrate_motion_profile_warping import monotonic_energy_warp, pose_speed
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_csi_motion_progress import predict_progress
from .train_kinetic_pose import pose_selection_score
from .train_profile_candidate_ranker import render_ranked_action


def progress_warp(pose, query_progress, valid, warp_strength,
                  floor_fraction=0.30):
    """Retiming from a predicted monotonic progress curve."""
    candidate_activity = pose_speed(pose, valid)
    output = torch.zeros_like(pose)
    for item, mask in enumerate(valid):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        length = len(positions)
        if length < 2:
            output[item, positions] = pose[item, positions]
            continue
        activity = candidate_activity[item, positions].clamp_min(0)
        floor = floor_fraction * activity.mean().clamp_min(0.02)
        candidate = torch.cumsum(activity + floor, dim=0)
        candidate = candidate / candidate[-1].clamp_min(1e-6)
        query = query_progress[item, positions].clamp(0, 1)
        query = torch.cummax(query, dim=0).values
        upper = torch.searchsorted(candidate, query).clamp(0, length - 1)
        lower = (upper - 1).clamp_min(0)
        fraction = (query - candidate[lower]) / (
            candidate[upper] - candidate[lower]
        ).clamp_min(1e-6)
        mapped = lower.float() + fraction * (upper - lower).float()
        identity = torch.arange(length, dtype=pose.dtype, device=pose.device)
        mapped = (1.0 - warp_strength) * identity + warp_strength * mapped
        source_lower = mapped.floor().long().clamp(0, length - 1)
        source_upper = (source_lower + 1).clamp_max(length - 1)
        alpha = (mapped - source_lower.float())[:, None, None]
        source = pose[item, positions]
        output[item, positions] = (
            (1.0 - alpha) * source[source_lower]
            + alpha * source[source_upper]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--progress-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp15_motion_progress_seed263"
        / "best_model.pt",
    )
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--ranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker_seed127"
        / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp15_motion_progress_warp"
        / "calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptive = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    action_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    action_model = TemporalMotionSelector(
        **action_checkpoint["model_config"]
    ).to(device)
    action_model.load_state_dict(action_checkpoint["model"])
    ranker_checkpoint = torch.load(
        args.ranker_checkpoint, map_location="cpu", weights_only=False
    )
    ranker = ProfileCandidateRanker(
        **ranker_checkpoint["model_config"]
    ).to(device)
    ranker.load_state_dict(ranker_checkpoint["model"])
    progress_checkpoint = torch.load(
        args.progress_checkpoint, map_location="cpu", weights_only=False
    )
    progress_model = MotionProgressHead(
        **progress_checkpoint["model_config"]
    ).to(device)
    progress_model.load_state_dict(progress_checkpoint["model"])
    data = prepare(args, "val", device)
    extra_action, _ = classifier_outputs(action_model, data["cache"], device)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * extra_action
    inference_valid = data["inference_valid"]
    current = predict_locked(data, adaptive)
    prior = render_ranked_action(
        ranker, data, retrieval_features(data, 3, 1.0), device
    )
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    existing = monotonic_energy_warp(
        prior, activity, inference_valid, 0.50, 0.30
    )
    existing_low = smooth_valid_delta(
        existing - current, inference_valid, 17
    )
    metrics = {
        "existing_energy_warp_s045": _metric_batch(
            current + 0.45 * existing_low,
            data["target_pose"], data["target_valid"], data["target_risk"],
        )
    }
    progress = predict_progress(
        progress_model, data["cache"], inference_valid, device
    )
    for floor in (0.15, 0.30, 0.45):
        for warp_strength in (0.25, 0.50, 0.75, 1.0):
            warped = progress_warp(
                prior, progress, inference_valid, warp_strength, floor
            )
            low = smooth_valid_delta(
                warped - current, inference_valid, 17
            )
            for blend in (0.35, 0.40, 0.45):
                name = (
                    f"progress_f{int(floor * 100):02d}"
                    f"_w{int(warp_strength * 100):03d}"
                    f"_s{int(blend * 100):03d}"
                )
                metrics[name] = _metric_batch(
                    current + blend * low,
                    data["target_pose"], data["target_valid"],
                    data["target_risk"],
                )
    scores = {
        name: pose_selection_score(value) for name, value in metrics.items()
    }
    selected = min(scores, key=scores.get)
    result = {
        "status": (
            "validation_promoted" if selected != "existing_energy_warp_s045"
            else "validation_rejected"
        ),
        "protocol": args.exp,
        "selection": {"name": selected, "score": scores[selected]},
        "metrics": metrics,
        "scores": scores,
        "progress_checkpoint": report_path(args.progress_checkpoint),
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"], "selection": result["selection"],
        "metrics": metrics[selected],
        "existing_score": scores["existing_energy_warp_s045"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
