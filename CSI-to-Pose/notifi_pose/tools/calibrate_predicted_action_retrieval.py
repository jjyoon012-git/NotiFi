"""Validation-only retrieval constrained by CSI-predicted action classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch, _render
from .calibrate_core_seed_selection import predict_locked
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_semantic_candidate_prior import add_arguments
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


@torch.no_grad()
def predicted_action_motion(data, action_top_k):
    probability = torch.softmax(data["fused_action"], dim=-1)
    classes = probability.topk(action_top_k, dim=-1).indices
    train_class = data["checkpoint"]["train_class"]
    motions = []
    selected_classes = []
    for item, valid in enumerate(data["target_valid"]):
        candidates = []
        weights = []
        chosen = []
        for class_id in classes[item].tolist():
            indices = torch.nonzero(
                train_class == class_id, as_tuple=False
            ).flatten()
            if len(indices) == 0:
                continue
            members = data["train_bank"].index_select(0, indices)
            distance = torch.linalg.vector_norm(
                members - data["baseline_bank"][item][None], dim=-1
            ).mean((1, 2))
            candidates.append(members[distance.argmin()])
            weights.append(probability[item, class_id])
            chosen.append(class_id)
        if not candidates:
            candidates = [data["train_bank"][0]]
            weights = [probability.new_tensor(1.0)]
            chosen = [int(train_class[0])]
        weight = torch.stack(weights)
        weight = weight / weight.sum().clamp_min(1e-6)
        canonical = (
            torch.stack(candidates) * weight[:, None, None, None]
        ).sum(0)
        motions.append(_render(canonical, valid, C.CACHE_FRAMES))
        selected_classes.append(chosen)
    return torch.stack(motions), selected_classes


def add_action_arguments(parser):
    add_arguments(parser)
    parser.set_defaults(
        scalar_profile_checkpoint=(
            C.WORK_ROOT / "runs" / "kp5_motion_profile_seed79"
            / "best_model.pt"
        ),
        part_profile_checkpoint=(
            C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed101"
            / "best_model.pt"
        ),
    )
    parser.add_argument(
        "--adaptive-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend"
        / "calibration.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_predicted_action_retrieval"
        / "calibration.json",
    )
    args = parser.parse_args()
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "val", device)
    current = predict_locked(data, adaptive_config)
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    metrics = {
        "current": _metric_batch(
            current, data["target_pose"], data["target_valid"],
            data["target_risk"],
        )
    }
    class_trace = {}
    for action_top_k in (1, 2, 3):
        action_motion, selected = predicted_action_motion(data, action_top_k)
        class_trace[f"top{action_top_k}"] = selected
        warped = monotonic_energy_warp(
            action_motion, activity, data["target_valid"], 0.50, 0.30
        )
        for weight in (0.05, 0.10, 0.15, 0.20, 0.30):
            name = f"top{action_top_k}_w{int(weight * 1000):03d}"
            metrics[name] = _metric_batch(
                (1.0 - weight) * current + weight * warped,
                data["target_pose"], data["target_valid"],
                data["target_risk"],
            )
    scores = {name: pose_selection_score(metric) for name, metric in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_predicted_action_retrieval",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "train_bank_only": True,
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "selected_predicted_classes": class_trace,
        "adaptive_calibration": report_path(args.adaptive_calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"], "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
