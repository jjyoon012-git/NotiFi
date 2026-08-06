"""Validation-only observability of whether the KP10 residual will help."""

from __future__ import annotations

import argparse
import csv
import json
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
from .train_kinetic_pose import DISTAL_JOINTS
from .train_profile_candidate_ranker import render_ranked_action


def trial_error(pose, target, valid, joints=None):
    if joints is None:
        joints = tuple(range(C.N_JOINTS))
    error = torch.linalg.vector_norm(
        pose[:, :, list(joints)] - target[:, :, list(joints)], dim=-1
    )
    return (error * valid[..., None]).sum((1, 2)) / (
        valid.sum(1) * len(joints)
    ).clamp_min(1)


def masked_stat(values, valid, name):
    if name == "mean":
        return (values * valid).sum(1) / valid.sum(1).clamp_min(1)
    if name == "max":
        return values.masked_fill(~valid, float("-inf")).amax(1)
    raise ValueError(name)


def correlation(left, right):
    left = left - left.mean()
    right = right - right.mean()
    return float((left * right).sum() / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(1e-8))


def feature_report(feature, improvement):
    order = feature.argsort()
    buckets = []
    for indices in order.tensor_split(4):
        values = improvement.index_select(0, indices)
        buckets.append({
            "feature_mean": float(feature.index_select(0, indices).mean()),
            "improvement_mean_cm": float(values.mean() * 100),
            "win_rate": float((values > 0).float().mean()),
            "trials": int(len(indices)),
        })
    return {
        "pearson_improvement": correlation(feature, improvement),
        "quartiles_low_to_high": buckets,
    }


def label_names():
    with C.LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["class_id"]): row["detail_label"]
            for row in csv.DictReader(handle)
        }


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
        default=C.WORK_ROOT / "runs" / "kp26_confidence_observability"
        / "validation.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptive = json.loads(
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
    ranker = ProfileCandidateRanker(
        **ranker_checkpoint["model_config"]
    ).to(device)
    ranker.load_state_dict(ranker_checkpoint["model"])
    data = prepare(args, "val", device)
    extra, _ = classifier_outputs(classifier, data["cache"], device)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * extra
    probability = torch.softmax(data["fused_action"], dim=-1)
    current = predict_locked(data, adaptive)
    prior = render_ranked_action(
        ranker, data, retrieval_features(data, 3, 1.0), device
    )
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    valid = data["inference_valid"]
    prior = monotonic_energy_warp(prior, activity, valid, 0.50, 0.30)
    residual = smooth_valid_delta(prior - current, valid, 17)
    point = current + 0.40 * residual
    target, metric_valid = data["target_pose"], data["target_valid"]
    current_error = trial_error(current, target, metric_valid)
    point_error = trial_error(point, target, metric_valid)
    current_distal = trial_error(
        current, target, metric_valid, DISTAL_JOINTS
    )
    point_distal = trial_error(point, target, metric_valid, DISTAL_JOINTS)
    improvement = current_error - point_error
    distal_improvement = current_distal - point_distal
    top = probability.topk(2, dim=-1).values
    entropy = -(
        probability.clamp_min(1e-7) * probability.clamp_min(1e-7).log()
    ).sum(-1) / torch.log(probability.new_tensor(float(C.N_CLASSES)))
    residual_size = torch.linalg.vector_norm(residual, dim=-1).mean(-1)
    features = {
        "action_confidence": top[:, 0],
        "action_margin": top[:, 0] - top[:, 1],
        "negative_action_entropy": -entropy,
        "predicted_danger_probability": probability[:, 12:].sum(-1),
        "activity_mean": masked_stat(activity, valid, "mean"),
        "activity_peak": masked_stat(activity, valid, "max"),
        "residual_mean": masked_stat(residual_size, valid, "mean"),
        "residual_peak": masked_stat(residual_size, valid, "max"),
    }
    reports = {
        name: {
            "pose": feature_report(feature, improvement),
            "distal": feature_report(feature, distal_improvement),
        } for name, feature in features.items()
    }
    predicted_class = probability.argmax(-1)
    correct = predicted_class == data["target_class"]
    correctness = {}
    for name, mask in (("correct", correct), ("incorrect", ~correct)):
        correctness[name] = {
            "trials": int(mask.sum()),
            "pose_improvement_cm": float(improvement[mask].mean() * 100),
            "distal_improvement_cm": float(
                distal_improvement[mask].mean() * 100
            ),
            "win_rate": float((improvement[mask] > 0).float().mean()),
        }
    names = label_names()
    by_class = {}
    for class_id in sorted(data["target_class"].unique().tolist()):
        mask = data["target_class"] == class_id
        by_class[f"{class_id:02d}_{names.get(class_id, 'unknown')}"] = {
            "trials": int(mask.sum()),
            "risk_id": int(data["target_risk"][mask][0]),
            "current_pose_cm": float(current_error[mask].mean() * 100),
            "kp10_pose_cm": float(point_error[mask].mean() * 100),
            "pose_improvement_cm": float(improvement[mask].mean() * 100),
            "distal_improvement_cm": float(
                distal_improvement[mask].mean() * 100
            ),
            "trial_win_rate": float((improvement[mask] > 0).float().mean()),
            "action_accuracy": float(correct[mask].float().mean()),
        }
    result = {
        "status": "validation_only_confidence_observability",
        "protocol": args.exp,
        "test_split_touched": False,
        "purpose": "decide whether a CSI-only selective residual gate is learnable",
        "trials": len(improvement),
        "action_accuracy": float(correct.float().mean()),
        "overall": {
            "pose_improvement_cm": float(improvement.mean() * 100),
            "distal_improvement_cm": float(distal_improvement.mean() * 100),
            "trial_win_rate": float((improvement > 0).float().mean()),
        },
        "action_correctness": correctness,
        "by_class": by_class,
        "csi_feature_observability": reports,
        "metrics": {
            "current": _metric_batch(
                current, target, metric_valid, data["target_risk"]
            ),
            "kp10": _metric_batch(
                point, target, metric_valid, data["target_risk"]
            ),
        },
        "inference_features_use_gt": False,
        "gt_use": "validation diagnostic outcome and action-correctness audit only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "overall": result["overall"],
        "action_accuracy": result["action_accuracy"],
        "action_correctness": correctness,
        "feature_correlations": {
            name: {
                "pose": value["pose"]["pearson_improvement"],
                "distal": value["distal"]["pearson_improvement"],
            } for name, value in reports.items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
