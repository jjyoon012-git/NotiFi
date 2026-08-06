"""Post-hoc paired bootstrap audit of locked KP6 and KP10 test predictions."""

from __future__ import annotations

import argparse
import copy
import csv
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
from .calibrate_motion_profile_warping import (
    activity_sources,
    monotonic_energy_warp,
    parse_promoted_selection,
)
from .calibrate_part_motion_profile_reranking import prepare, render_mixture
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .calibrate_risk_adaptive_blend import adaptive_strength
from .calibrate_semantic_candidate_prior import (
    base_adjusted_logits,
    candidate_semantic_scores,
)
from .kp10_inference import inference_view
from .train_kinetic_pose import DISTAL_JOINTS
from .train_profile_candidate_ranker import render_ranked_action


@torch.no_grad()
def kp6_prediction(data, args):
    data = inference_view(data)
    part_config = parse_promoted_selection(json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    ))
    selected = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    base = base_adjusted_logits(data, part_config)
    action_score, risk_score = candidate_semantic_scores(data)
    adjusted = base + 0.35 * action_score + 0.50 * risk_score
    candidate = render_mixture(data, adjusted, 0.50, 5)
    warped = monotonic_energy_warp(
        candidate, activity_sources(data)["scalar_limbs"],
        data["inference_valid"], 0.50, 0.30,
    )
    strength = adaptive_strength(
        data["risk_probability"], selected["base"],
        selected["danger_delta"], selected["uncertainty_delta"],
    )
    return (
        (1.0 - strength)[:, None, None, None] * data["baseline"]
        + strength[:, None, None, None] * warped
    )


@torch.no_grad()
def kp10_prediction(data, args, device):
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
        args.profile_ranker_checkpoint,
        map_location="cpu", weights_only=False,
    )
    ranker = ProfileCandidateRanker(
        **ranker_checkpoint["model_config"]
    ).to(device)
    ranker.load_state_dict(ranker_checkpoint["model"])
    extra, _ = classifier_outputs(classifier, data["cache"], device)
    data = dict(data)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * extra
    data = inference_view(data)
    current = predict_locked(data, adaptive)
    prior = render_ranked_action(
        ranker, data, retrieval_features(data, 3, 1.0), device
    )
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    prior = monotonic_energy_warp(
        prior, activity, data["inference_valid"], 0.50, 0.30
    )
    residual = smooth_valid_delta(
        prior - current, data["inference_valid"], 17
    )
    calibration = json.loads(
        args.strength_calibration.read_text(encoding="utf-8")
    )
    match = re.fullmatch(r"strength_(\d{3})", calibration["selection"]["name"])
    if match is None:
        raise RuntimeError("invalid KP10 strength calibration")
    strength = int(match.group(1)) / 100.0
    predicted = current + strength * residual
    return predicted - predicted[:, :, :1]


def trial_errors(predicted, target, valid):
    error = torch.linalg.vector_norm(predicted - target, dim=-1)
    overall = (error * valid[..., None]).sum((1, 2)) / (
        valid.sum(1) * C.N_JOINTS
    ).clamp_min(1)
    distal = (
        error[..., list(DISTAL_JOINTS)] * valid[..., None]
    ).sum((1, 2)) / (
        valid.sum(1) * len(DISTAL_JOINTS)
    ).clamp_min(1)
    return {"pose": overall, "distal": distal}


def paired_bootstrap(kp6, kp10, indices, samples, generator):
    difference = (kp10 - kp6).index_select(0, indices)
    draw = torch.randint(
        len(difference), (samples, len(difference)), generator=generator
    )
    means = difference[draw].mean(1)
    interval = torch.quantile(means, torch.tensor([0.025, 0.975]))
    return {
        "kp10_minus_kp6_mean_cm": float(difference.mean() * 100),
        "ci95_cm": [float(interval[0] * 100), float(interval[1] * 100)],
        "probability_improvement": float((means < 0).float().mean()),
        "trial_win_rate": float((difference < 0).float().mean()),
        "trials": int(len(difference)),
    }


def class_delta_audit(kp6_error, kp10_error, target_class):
    with C.LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        names = {
            int(row["class_id"]): row["detail_label"]
            for row in csv.DictReader(handle)
        }
    result = {}
    for class_id in sorted(target_class.unique().tolist()):
        mask = target_class == class_id
        pose = (kp10_error["pose"] - kp6_error["pose"])[mask]
        distal = (kp10_error["distal"] - kp6_error["distal"])[mask]
        result[f"{int(class_id):02d}_{names[int(class_id)]}"] = {
            "trials": int(mask.sum()),
            "pose_delta_cm": float(pose.mean() * 100),
            "distal_delta_cm": float(distal.mean() * 100),
            "pose_trial_win_rate": float((pose < 0).float().mean()),
            "distal_trial_win_rate": float((distal < 0).float().mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--profile-ranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker_seed127"
        / "best_model.pt",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--strength-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "paired_bootstrap_audit.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kp6_args = copy.copy(args)
    kp6_args.scalar_profile_checkpoint = (
        C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71"
        / "best_model.pt"
    )
    kp6_args.part_profile_checkpoint = (
        C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "best_model.pt"
    )
    kp6_data = prepare(kp6_args, "test", device)
    kp10_data = prepare(args, "test", device)
    kp6 = kp6_prediction(kp6_data, kp6_args)
    kp10 = kp10_prediction(kp10_data, args, device)
    if not torch.equal(kp6_data["target_valid"], kp10_data["target_valid"]):
        raise RuntimeError("KP6 and KP10 test target masks differ")
    data = kp10_data
    kp6_error = trial_errors(kp6, data["target_pose"], data["target_valid"])
    kp10_error = trial_errors(kp10, data["target_pose"], data["target_valid"])
    all_indices = torch.arange(len(kp6))
    danger_indices = torch.nonzero(
        data["target_risk"] == 2, as_tuple=False
    ).flatten()
    generator = torch.Generator().manual_seed(args.seed)
    bootstrap = {}
    for metric in ("pose", "distal"):
        bootstrap[f"overall_{metric}"] = paired_bootstrap(
            kp6_error[metric], kp10_error[metric], all_indices,
            args.samples, generator,
        )
        bootstrap[f"danger_{metric}"] = paired_bootstrap(
            kp6_error[metric], kp10_error[metric], danger_indices,
            args.samples, generator,
        )
    result = {
        "status": "posthoc_locked_test_audit",
        "protocol": args.exp,
        "purpose": "uncertainty audit only; no model or hyperparameter selection",
        "test_used_for_selection": False,
        "future_tuning_from_this_audit": False,
        "inference_inputs": "CSI and link mask only",
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "locked_profile_checkpoints": {
            "kp6_scalar": "work_v2/runs/kp5_motion_profile_seed71/best_model.pt",
            "kp6_part": "work_v2/runs/kp5_part_motion_profile_seed83/best_model.pt",
            "kp10_scalar": str(args.scalar_profile_checkpoint).replace("\\", "/"),
            "kp10_part": str(args.part_profile_checkpoint).replace("\\", "/"),
        },
        "bootstrap": bootstrap,
        "aggregate_metrics": {
            "kp6": _metric_batch(
                kp6, data["target_pose"], data["target_valid"],
                data["target_risk"],
            ),
            "kp10": _metric_batch(
                kp10, data["target_pose"], data["target_valid"],
                data["target_risk"],
            ),
        },
        "by_class_posthoc": class_delta_audit(
            kp6_error, kp10_error, data["target_class"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
