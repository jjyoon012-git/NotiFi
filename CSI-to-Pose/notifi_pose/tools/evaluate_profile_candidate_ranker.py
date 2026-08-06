"""One-shot fixed-test evaluation of the validation-selected profile ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_action_logit_fusion import calibrated_action_logits
from .calibrate_calibrated_action_prior import render_final
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
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker"
        / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("test_used_for_selection", True):
        raise RuntimeError("Ranker checkpoint is not validation-locked")
    action_config = json.loads(
        args.action_calibration.read_text(encoding="utf-8")
    )["selection"]
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "test", device)
    data["fused_action"] = calibrated_action_logits(data, action_config)
    current = predict_locked(data, adaptive_config)
    reference = render_final(data, current)
    model = ProfileCandidateRanker(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    features = retrieval_features(data, 3, 1.0)
    promoted, promoted_metrics = evaluate_pose(
        model, data, features, current, device
    )
    metrics = {
        "kp7_action_calibrated": _metric_batch(
            reference, data["target_pose"], data["target_valid"],
            data["target_risk"],
        ),
        "kp8_profile_candidate_ranker": promoted_metrics,
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation epoch and locked KP7 rendering",
        "inference_inputs": "CSI and link mask only",
        "train_bank_only": True,
        "train_leave_self_out": checkpoint["train_leave_self_out"],
        "fixed_configuration": checkpoint["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(metric)
            for name, metric in metrics.items()
        },
        "checkpoint": report_path(args.checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
