"""Validation-only use of the independent CSI classifier's risk evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from .calibrate_core_seed_selection import predict_locked
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score
from .train_profile_candidate_ranker import evaluate_pose


@torch.no_grad()
def classifier_outputs(model, cache, device):
    model.eval()
    actions, risks = [], []
    for start in range(0, len(cache["features"]), 64):
        stop = min(start + 64, len(cache["features"]))
        output = model(
            cache["features"][start:stop].to(device).float(),
            cache["frame_mask"][start:stop].to(device),
        )
        actions.append(output["action_logits"].float().cpu())
        risks.append(output["risk_logits"].float().cpu())
    return torch.cat(actions), torch.cat(risks)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
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
        default=C.WORK_ROOT / "runs" / "kp10_independent_risk_fusion"
        / "calibration.json",
    )
    args = parser.parse_args()
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
    data = prepare(args, "val", device)
    action_logits, risk_logits = classifier_outputs(
        classifier, data["cache"], device
    )
    fixed_action = 1.50 * data["base_action_logits"] + 0.75 * action_logits
    risk_configurations = {
        "current": data["base_risk_logits"] + data["selector_risk_logits"],
        "base_plus_extra_075": data["base_risk_logits"] + 0.75 * risk_logits,
        "base_plus_extra_100": data["base_risk_logits"] + risk_logits,
        "current_plus_extra_050": (
            data["base_risk_logits"] + data["selector_risk_logits"]
            + 0.50 * risk_logits
        ),
    }
    metrics, risk_accuracy = {}, {}
    for name, logits in risk_configurations.items():
        variant = dict(data)
        variant["risk_probability"] = torch.softmax(logits, dim=-1)
        variant["fused_action"] = fixed_action
        current = predict_locked(variant, adaptive_config)
        _, metrics[name] = evaluate_pose(
            ranker, variant, retrieval_features(variant, 3, 1.0),
            current, device,
        )
        risk_accuracy[name] = float(
            (logits.argmax(-1) == data["target_risk"]).float().mean()
        )
    scores = {
        name: pose_selection_score(metric) for name, metric in metrics.items()
    }
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_independent_risk_fusion",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "fixed_action_fusion": "1.50 * base + 0.75 * independent classifier",
        "selection": {"name": best, "score": scores[best]},
        "scores": scores, "metrics": metrics,
        "risk_accuracy": risk_accuracy,
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "ranker_checkpoint": report_path(args.ranker_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"],
        "risk_accuracy": risk_accuracy,
        "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
