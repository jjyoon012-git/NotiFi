"""One-shot fixed-test evaluation of validation-selected action fusion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_action_classifier_pose import classifier_logits
from .calibrate_action_logit_fusion import calibrated_action_logits
from .calibrate_core_seed_selection import predict_locked
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score
from .train_profile_candidate_ranker import evaluate_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--action-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_action_logit_fusion"
        / "calibration.json",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_pose"
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
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_pose"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    selected = calibration["selection"]["name"]
    match = re.fullmatch(r"base_plus_extra_(\d{3})", selected)
    if match is None:
        raise RuntimeError("Unsupported validation-locked action fusion")
    extra_weight = int(match.group(1)) / 100.0
    action_config = json.loads(
        args.action_calibration.read_text(encoding="utf-8")
    )["selection"]
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    current = predict_locked(data, adaptive_config)

    reference = dict(data)
    reference["fused_action"] = calibrated_action_logits(data, action_config)
    _, reference_metrics = evaluate_pose(
        ranker, reference, retrieval_features(reference, 3, 1.0),
        current, device,
    )
    extra = classifier_logits(classifier, data["cache"], device)
    promoted = dict(data)
    promoted["fused_action"] = (
        1.50 * data["base_action_logits"] + extra_weight * extra
    )
    _, promoted_metrics = evaluate_pose(
        ranker, promoted, retrieval_features(promoted, 3, 1.0),
        current, device,
    )
    metrics = {
        "kp8_profile_ranker_seed127": reference_metrics,
        "kp10_independent_action_fusion": promoted_metrics,
    }
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
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "ranker_checkpoint": report_path(args.ranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
