"""One-shot fixed-test evaluation of validation-locked KP10 prior strength."""

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
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .kp10_inference import inference_view
from .train_kinetic_pose import pose_selection_score
from .train_profile_candidate_ranker import render_ranked_action


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
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
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    match = re.fullmatch(
        r"strength_(\d{3})", calibration["selection"]["name"]
    )
    if match is None:
        raise RuntimeError("Unsupported validation-locked strength")
    selected_strength = int(match.group(1)) / 100.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    classifier = TemporalMotionSelector(
        **classifier_checkpoint["model_config"]
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    ranker_checkpoint = torch.load(
        args.ranker_checkpoint, map_location="cpu", weights_only=False
    )
    ranker = ProfileCandidateRanker(**ranker_checkpoint["model_config"]).to(device)
    ranker.load_state_dict(ranker_checkpoint["model"])
    data = prepare(args, "test", device)
    action_logits, _ = classifier_outputs(classifier, data["cache"], device)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * action_logits
    inference = inference_view(data)
    inference_valid = inference["inference_valid"]
    current = predict_locked(inference, adaptive_config)
    prior = render_ranked_action(
        ranker, inference, retrieval_features(inference, 3, 1.0), device
    )
    activity = (
        0.50 * inference["predicted_scalar_profile"]
        + 0.50 * inference["predicted_part_profile"][..., 2:].mean(-1)
    )
    prior = monotonic_energy_warp(
        prior, activity, inference_valid, 0.50, 0.30
    )
    low = smooth_valid_delta(prior - current, inference_valid, 17)
    metrics = {}
    for name, strength in (
        ("kp10_strength_035", 0.35),
        (f"kp10_strength_{int(selected_strength * 100):03d}", selected_strength),
    ):
        predicted = current + strength * low
        predicted = predicted - predicted[:, :, :1]
        metrics[name] = _metric_batch(
            predicted, data["target_pose"], data["target_valid"],
            data["target_risk"],
        )
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation pose composite",
        "inference_inputs": "CSI and link mask only",
        "fixed_configuration": calibration["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(metric)
            for name, metric in metrics.items()
        },
        "calibration": report_path(args.calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
