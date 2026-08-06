"""One-shot fixed-test evaluation of validation-locked KP20 yaw alignment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_independent_risk_fusion import classifier_outputs
from .calibrate_kp20_yaw_alignment import yaw_align
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score
from .train_profile_candidate_ranker import render_ranked_action


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp20_yaw_alignment"
        / "calibration.json",
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
        default=C.WORK_ROOT / "runs" / "kp20_yaw_alignment"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    match = re.fullmatch(r"yaw_(\d{3})", calibration["selection"]["name"])
    if match is None or calibration["status"] != "validation_promoted":
        raise RuntimeError("KP20 calibration did not promote a yaw configuration")
    selected_strength = int(match.group(1)) / 100.0
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
    data = prepare(args, "test", device)
    action, _ = classifier_outputs(action_model, data["cache"], device)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * action
    inference_valid = data["inference_valid"]
    current = predict_locked(data, adaptive)
    prior = render_ranked_action(
        ranker, data, retrieval_features(data, 3, 1.0), device
    )
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    prior = monotonic_energy_warp(
        prior, activity, inference_valid, 0.50, 0.30
    )
    metrics = {}
    for name, strength in (
        ("kp10_yaw_000", 0.0),
        (f"kp20_yaw_{int(selected_strength * 100):03d}", selected_strength),
    ):
        aligned = yaw_align(prior, current, inference_valid, strength)
        low = smooth_valid_delta(
            aligned - current, inference_valid, 17
        )
        predicted = current + 0.45 * low
        metrics[name] = _metric_batch(
            predicted, data["target_pose"], data["target_valid"],
            data["target_risk"],
        )
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "fixed_configuration": calibration["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "calibration": report_path(args.calibration),
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
