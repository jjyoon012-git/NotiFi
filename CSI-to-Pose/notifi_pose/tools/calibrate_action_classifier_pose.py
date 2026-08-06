"""Validation-only pose selection for an independent CSI action classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from .calibrate_action_logit_fusion import calibrated_action_logits
from .calibrate_core_seed_selection import predict_locked
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score
from .train_profile_candidate_ranker import evaluate_pose


@torch.no_grad()
def classifier_logits(model, cache, device):
    model.eval()
    values = []
    for start in range(0, len(cache["features"]), 64):
        stop = min(start + 64, len(cache["features"]))
        values.append(model(
            cache["features"][start:stop].to(device).float(),
            cache["frame_mask"][start:stop].to(device),
        )["action_logits"].float().cpu())
    return torch.cat(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--action-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_action_logit_fusion"
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
        / "calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_config = json.loads(
        args.action_calibration.read_text(encoding="utf-8")
    )["selection"]
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
    data = prepare(args, "val", device)
    extra = classifier_logits(classifier, data["cache"], device)
    current_logits = calibrated_action_logits(data, action_config)
    configurations = {"current": current_logits}
    for weight in (0.25, 0.50, 0.75):
        configurations[f"current_plus_extra_{int(weight * 100):03d}"] = (
            current_logits + weight * extra
        )
    for weight in (0.50, 0.75, 1.00):
        configurations[f"base_plus_extra_{int(weight * 100):03d}"] = (
            1.50 * data["base_action_logits"] + weight * extra
        )
    current = predict_locked(data, adaptive_config)
    metrics, action_accuracy = {}, {}
    for name, logits in configurations.items():
        variant = dict(data)
        variant["fused_action"] = logits
        _, metrics[name] = evaluate_pose(
            ranker, variant, retrieval_features(variant, 3, 1.0),
            current, device,
        )
        action_accuracy[name] = float(
            (logits.argmax(-1) == data["target_class"]).float().mean()
        )
    scores = {
        name: pose_selection_score(metric) for name, metric in metrics.items()
    }
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_action_classifier_pose",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "selection": {"name": best, "score": scores[best]},
        "scores": scores, "metrics": metrics,
        "action_accuracy": action_accuracy,
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "ranker_checkpoint": report_path(args.ranker_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"],
        "action_accuracy": action_accuracy,
        "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
