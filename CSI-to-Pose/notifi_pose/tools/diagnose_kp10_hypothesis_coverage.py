"""Validation-only coverage of three CSI-predicted action hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from .audit_motion_retrieval_oracle import _metric_batch, _render
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_independent_risk_fusion import classifier_outputs
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .kp10_inference import inference_view
from .train_kinetic_pose import DISTAL_JOINTS, pose_selection_score
from .train_profile_candidate_ranker import group_candidate_values, render_ranked_action


def label_names():
    with C.LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["class_id"]): row["detail_label"]
            for row in csv.DictReader(handle)
        }


def paired_bootstrap(values, samples=10_000, seed=20260806):
    """Bootstrap a trial-level point-minus-set coverage gain in centimeters."""
    values = values.detach().float().cpu() * 100.0
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(values), (samples, len(values)), generator=generator
    )
    means = values.index_select(0, indices.flatten()).reshape(
        samples, len(values)
    ).mean(1)
    return {
        "mean_gain_cm": float(values.mean()),
        "ci95_cm": [
            float(torch.quantile(means, 0.025)),
            float(torch.quantile(means, 0.975)),
        ],
        "trial_win_rate": float((values > 0).float().mean()),
        "trials": len(values),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def export_hypotheses(output_dir, indices, predicted, probabilities,
                      classes, valid, rows, split):
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    normalized = probabilities / probabilities.sum(
        1, keepdim=True
    ).clamp_min(1e-6)
    for index in indices:
        if index < 0 or index >= len(predicted):
            raise IndexError(f"hypothesis export index out of range: {index}")
        row = int(rows[index])
        name = f"{split}_row_{row:06d}.npz"
        np.savez_compressed(
            output_dir / name,
            pose_hypotheses=predicted[index].half().numpy(),
            action_probability=normalized[index].float().numpy(),
            action_class_id=classes[index].short().numpy(),
            frame_mask=valid[index].numpy(),
            dataset_row=np.asarray(row, dtype=np.int64),
        )
        exported.append(name)
    metadata = {
        "format": "notifi_kp28_multi5_csi_only_v1",
        "split": split,
        "hypotheses": predicted.shape[1],
        "pose_shape": [predicted.shape[2], C.N_JOINTS, 3],
        "inference_inputs": ["CSI", "link mask"],
        "contains_gt": False,
        "files": exported,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


@torch.no_grad()
def render_hypotheses(model, data, features, device, count=3):
    model.eval()
    inference_valid = data["inference_valid"]
    all_hypotheses, all_probability, all_classes = [], [], []
    for groups, valid in zip(features, inference_valid):
        hypotheses, probabilities, classes = [], [], []
        for group in groups[:count]:
            values = group_candidate_values(group).to(device)
            class_id = torch.full(
                (len(values),), int(group["class_id"]),
                dtype=torch.long, device=device,
            )
            score = model(values, class_id).float().cpu()
            local = score.topk(min(2, len(score))).indices
            weight = torch.softmax(score.index_select(0, local), dim=0)
            bank_indices = group["indices"].index_select(0, local)
            canonical = (
                data["train_bank"].index_select(0, bank_indices)
                * weight[:, None, None, None]
            ).sum(0)
            hypotheses.append(_render(canonical, valid, C.CACHE_FRAMES))
            probabilities.append(group["probability"])
            classes.append(int(group["class_id"]))
        while len(hypotheses) < count:
            hypotheses.append(hypotheses[-1].clone())
            probabilities.append(probabilities[-1].new_zeros(()))
            classes.append(classes[-1])
        all_hypotheses.append(torch.stack(hypotheses))
        all_probability.append(torch.stack(probabilities))
        all_classes.append(torch.tensor(classes))
    return (
        torch.stack(all_hypotheses), torch.stack(all_probability),
        torch.stack(all_classes),
    )


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
        default=C.WORK_ROOT / "runs" / "kp22_hypothesis_coverage"
        / "validation.json",
    )
    parser.add_argument("--hypotheses", type=int, default=3)
    parser.add_argument(
        "--strength-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "calibration.json",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument(
        "--export-indices", default=None,
        help="Comma-separated split-local indices; required with --export-dir.",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptive = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    strength_config = json.loads(
        args.strength_calibration.read_text(encoding="utf-8")
    )
    strength_match = re.fullmatch(
        r"strength_(\d{3})", strength_config["selection"]["name"]
    )
    if strength_match is None:
        raise RuntimeError("invalid KP10 strength calibration")
    strength = int(strength_match.group(1)) / 100.0
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
    data = prepare(args, args.split, device)
    action, _ = classifier_outputs(action_model, data["cache"], device)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * action
    inference = inference_view(data)
    current = predict_locked(inference, adaptive)
    point_features = retrieval_features(inference, 3, 1.0)
    hypothesis_features = retrieval_features(
        inference, args.hypotheses, 1.0
    )
    point_prior = render_ranked_action(
        ranker, inference, point_features, device
    )
    hypotheses, probabilities, hypothesis_classes = render_hypotheses(
        ranker, inference, hypothesis_features, device, args.hypotheses
    )
    set_prior = render_ranked_action(
        ranker, inference, hypothesis_features, device
    )
    batch, choices = hypotheses.shape[:2]
    activity = (
        0.50 * inference["predicted_scalar_profile"]
        + 0.50 * inference["predicted_part_profile"][..., 2:].mean(-1)
    )
    valid = inference["inference_valid"]
    point_prior = monotonic_energy_warp(
        point_prior, activity, valid, 0.50, 0.30
    )
    set_prior = monotonic_energy_warp(
        set_prior, activity, valid, 0.50, 0.30
    )
    point_low = smooth_valid_delta(point_prior - current, valid, 17)
    point = current + strength * point_low
    set_low = smooth_valid_delta(set_prior - current, valid, 17)
    set_point = current + strength * set_low
    flat = hypotheses.reshape(
        batch * choices, *hypotheses.shape[2:]
    )
    repeated_activity = activity[:, None].expand(-1, choices, -1).reshape(
        batch * choices, -1
    )
    repeated_valid = valid[:, None].expand(-1, choices, -1).reshape(
        batch * choices, -1
    )
    flat = monotonic_energy_warp(
        flat, repeated_activity, repeated_valid, 0.50, 0.30
    )
    repeated_current = current[:, None].expand(
        -1, choices, -1, -1, -1
    ).reshape(batch * choices, *current.shape[1:])
    flat_low = smooth_valid_delta(
        flat - repeated_current, repeated_valid, 17
    )
    predicted = (repeated_current + strength * flat_low).reshape(
        batch, choices, *current.shape[1:]
    )
    export = None
    if args.export_dir is not None:
        if not args.export_indices:
            raise ValueError("--export-indices is required with --export-dir")
        export = export_hypotheses(
            args.export_dir,
            [int(value) for value in args.export_indices.split(",")],
            predicted, probabilities, hypothesis_classes, valid,
            data["cache"]["rows"], args.split,
        )
    target = data["target_pose"]
    metric_valid = data["target_valid"]
    error = torch.linalg.vector_norm(
        predicted - target[:, None], dim=-1
    )
    trial_error = (
        error * metric_valid[:, None, :, None]
    ).sum((2, 3)) / (
        metric_valid.sum(1)[:, None] * C.N_JOINTS
    ).clamp_min(1)
    best_index = trial_error.argmin(1)
    oracle = predicted[torch.arange(batch), best_index]
    top1 = predicted[:, 0]
    action_match = hypothesis_classes == data["target_class"][:, None]
    true_action_index = action_match.float().argmax(1)
    true_action_index = torch.where(
        action_match.any(1), true_action_index,
        torch.zeros_like(true_action_index),
    )
    true_action = predicted[torch.arange(batch), true_action_index]
    pairwise = []
    for left in range(choices):
        for right in range(left + 1, choices):
            pairwise.append(torch.linalg.vector_norm(
                predicted[:, left] - predicted[:, right], dim=-1
            ).mean((1, 2)))
    diversity = torch.stack(pairwise, dim=1).mean()
    oracle_name = f"best_of_{choices}_oracle_coverage"
    metrics = {
        "deployed_point_estimate": _metric_batch(
            point, target, metric_valid, data["target_risk"]
        ),
        "top1_action_hypothesis": _metric_batch(
            top1, target, metric_valid, data["target_risk"]
        ),
        "true_action_label_oracle_diagnostic": _metric_batch(
            true_action, target, metric_valid, data["target_risk"]
        ),
        f"top{choices}_probability_mixture": _metric_batch(
            set_point, target, metric_valid, data["target_risk"]
        ),
        oracle_name: _metric_batch(
            oracle, target, metric_valid, data["target_risk"]
        ),
    }
    point_error = torch.linalg.vector_norm(point - target, dim=-1)
    oracle_error = torch.linalg.vector_norm(oracle - target, dim=-1)
    point_trial = (
        point_error * metric_valid[..., None]
    ).sum((1, 2)) / (
        metric_valid.sum(1) * C.N_JOINTS
    ).clamp_min(1)
    oracle_trial = (
        oracle_error * metric_valid[..., None]
    ).sum((1, 2)) / (
        metric_valid.sum(1) * C.N_JOINTS
    ).clamp_min(1)
    distal = list(DISTAL_JOINTS)
    point_distal = (
        point_error[..., distal] * metric_valid[..., None]
    ).sum((1, 2)) / (
        metric_valid.sum(1) * len(distal)
    ).clamp_min(1)
    oracle_distal = (
        oracle_error[..., distal] * metric_valid[..., None]
    ).sum((1, 2)) / (
        metric_valid.sum(1) * len(distal)
    ).clamp_min(1)
    names = label_names()
    by_class = {}
    for class_id in sorted(data["target_class"].unique().tolist()):
        mask = data["target_class"] == class_id
        by_class[f"{class_id:02d}_{names.get(class_id, 'unknown')}"] = {
            "trials": int(mask.sum()),
            "risk_id": int(data["target_risk"][mask][0]),
            "point_pose_cm": float(point_trial[mask].mean() * 100),
            "oracle_pose_cm": float(oracle_trial[mask].mean() * 100),
            "oracle_pose_gain_cm": float(
                (point_trial[mask] - oracle_trial[mask]).mean() * 100
            ),
            "point_distal_cm": float(point_distal[mask].mean() * 100),
            "oracle_distal_cm": float(oracle_distal[mask].mean() * 100),
            "oracle_rank_counts": torch.bincount(
                best_index[mask], minlength=args.hypotheses
            ).tolist(),
        }
    danger = data["target_risk"] == 2
    action_hit = action_match.any(1)
    action_top1 = hypothesis_classes[:, 0] == data["target_class"]
    result = {
        "status": (
            "validation_only_hypothesis_coverage_diagnostic"
            if args.split == "val"
            else "fixed_test_hypothesis_coverage_no_test_tuning"
        ),
        "protocol": args.exp,
        "evaluation_split": args.split,
        "test_split_touched": args.split == "test",
        "selection_source": (
            f"validation top-{args.hypotheses} coverage"
            if args.split == "test"
            else "not applicable"
        ),
        "test_used_for_selection": False,
        "inference_hypotheses_use_gt": False,
        "coverage_metric_uses_gt": True,
        "gt_selects_deployment_hypothesis": False,
        "hypotheses": args.hypotheses,
        "motion_strength": strength,
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "mean_pairwise_diversity_m": float(diversity),
        "oracle_selected_rank_counts": torch.bincount(
            best_index, minlength=args.hypotheses
        ).tolist(),
        "action_set_audit": {
            "top1_accuracy": float(action_top1.float().mean()),
            "top_k_recall": float(action_hit.float().mean()),
            "danger_top1_accuracy": float(action_top1[danger].float().mean()),
            "danger_top_k_recall": float(action_hit[danger].float().mean()),
            "uses_gt_for_evaluation_only": True,
        },
        "paired_coverage_audit": {
            "overall_pose": paired_bootstrap(
                point_trial - oracle_trial, seed=20260806
            ),
            "overall_distal": paired_bootstrap(
                point_distal - oracle_distal, seed=20260807
            ),
            "danger_pose": paired_bootstrap(
                point_trial[danger] - oracle_trial[danger], seed=20260808
            ),
            "danger_distal": paired_bootstrap(
                point_distal[danger] - oracle_distal[danger], seed=20260809
            ),
            "selection_note": (
                "The oracle rank is chosen by whole-pose trial error; this "
                "audit is coverage evidence, not deployable selection."
            ),
        },
        "by_class": by_class,
        "csi_only_export": export,
        "interpretation": (
            f"Best-of-{args.hypotheses} uses {args.split} GT only after "
            "inference to measure coverage; it is not a deployable selector "
            "or point-estimate performance."
        ),
        "inference_inputs_for_hypotheses": "CSI and link mask only",
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "ranker_checkpoint": report_path(args.ranker_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "scores": result["scores"],
        "metrics": metrics,
        "mean_pairwise_diversity_m": result["mean_pairwise_diversity_m"],
        "oracle_selected_rank_counts": result["oracle_selected_rank_counts"],
        "action_set_audit": result["action_set_audit"],
        "paired_coverage_audit": result["paired_coverage_audit"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
