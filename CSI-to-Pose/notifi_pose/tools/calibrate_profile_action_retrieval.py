"""Validation-only profile-aware retrieval inside CSI-predicted actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch, _render
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_motion_profile_reranking import profile_distance, standardize
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import (
    part_profile_distance,
    prepare,
    weighted_part_distance,
)
from .calibrate_predicted_action_retrieval import add_action_arguments
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def bank_profiles(train_bank):
    velocity = torch.zeros_like(train_bank)
    velocity[:, 1:] = (
        train_bank[:, 1:] - train_bank[:, :-1]
    ) * C.TARGET_FPS
    scalar = torch.linalg.vector_norm(velocity, dim=-1).mean(-1)
    parts = []
    for joints in C.JOINT_GROUPS.values():
        parts.append(torch.linalg.vector_norm(
            velocity[:, :, list(joints)], dim=-1
        ).mean(-1))
    return scalar, torch.stack(parts, dim=-1)


def bank_proximity_profiles(train_bank):
    joints = [0, 1, 2, 4, 5, 10, 11, 15, 20, 21]
    floor = train_bank[..., C.UP_AXIS].amin(-1)
    height = train_bank[:, :, joints, C.UP_AXIS] - floor[..., None]
    return (height < 0.12).to(train_bank.dtype)


def proximity_distance(predicted, candidates, valid):
    batch, choices, frames, contacts = candidates.shape
    predicted = F.avg_pool1d(
        predicted.permute(0, 2, 1).reshape(batch * contacts, 1, frames),
        9, stride=1, padding=4,
    ).reshape(batch, contacts, frames).permute(0, 2, 1)
    candidates = F.avg_pool1d(
        candidates.permute(0, 1, 3, 2).reshape(
            batch * choices * contacts, 1, frames
        ), 9, stride=1, padding=4,
    ).reshape(batch, choices, contacts, frames).permute(0, 1, 3, 2)
    mask = valid[:, None, :, None]
    count = mask.sum(2).clamp_min(1)
    return ((predicted[:, None] - candidates).abs() * mask).sum(2) / count


def query_motion_context(data, item, valid, action_probability):
    scalar = data["predicted_scalar_profile"][item][valid]
    part = data["predicted_part_profile"][item][valid]
    if len(scalar) == 0:
        scalar = data["predicted_scalar_profile"].new_zeros(1)
        part = data["predicted_part_profile"].new_zeros(
            1, len(C.JOINT_GROUPS)
        )
    peak = scalar.argmax().to(scalar.dtype) / max(len(scalar) - 1, 1)
    scalar_stats = torch.stack((
        scalar.mean(), scalar.std(unbiased=False), scalar.amax(), peak,
    ))
    return torch.cat((
        data["risk_probability"][item],
        action_probability.reshape(1),
        scalar_stats,
        part.mean(0), part.amax(0),
    ))


def retrieval_features(data, action_top_k=3, action_temperature=1.0,
                       self_indices=None):
    train_class = data["checkpoint"]["train_class"]
    scalar_bank, part_bank = bank_profiles(data["train_bank"])
    proximity_bank = (
        bank_proximity_profiles(data["train_bank"])
        if "predicted_proximity" in data else None
    )
    probability = torch.softmax(
        data["fused_action"] / float(action_temperature), dim=-1
    )
    top_classes = probability.topk(action_top_k, dim=-1).indices
    result = []
    inference_valid = data["inference_valid"]
    for item, valid in enumerate(inference_valid):
        groups = []
        for class_id in top_classes[item].tolist():
            indices = torch.nonzero(
                train_class == class_id, as_tuple=False
            ).flatten()
            if self_indices is not None:
                indices = indices[indices != int(self_indices[item])]
            if len(indices) == 0:
                continue
            members = data["train_bank"].index_select(0, indices)
            pose = torch.linalg.vector_norm(
                members - data["baseline_bank"][item][None], dim=-1
            ).mean((1, 2))[None]
            scalar = profile_distance(
                data["predicted_scalar_profile"][item:item + 1],
                scalar_bank.index_select(0, indices)[None],
                valid[None],
            )
            part = part_profile_distance(
                data["predicted_part_profile"][item:item + 1],
                part_bank.index_select(0, indices)[None],
                valid[None],
            )
            part_values = (
                part - part.mean(1, keepdim=True)
            ) / part.std(1, keepdim=True).clamp_min(1e-5)
            part = weighted_part_distance(
                part, (0.8, 0.6, 1.7, 1.7, 1.7, 1.7)
            )
            contact_values = None
            if proximity_bank is not None:
                contact = proximity_distance(
                    data["predicted_proximity"][item:item + 1],
                    proximity_bank.index_select(0, indices)[None],
                    valid[None],
                )
                contact_values = (
                    contact - contact.mean(1, keepdim=True)
                ) / contact.std(1, keepdim=True).clamp_min(1e-5)
            selector_values = None
            if data.get("include_selector_distance", False):
                selector = torch.linalg.vector_norm(
                    data["checkpoint"]["train_embedding"].index_select(
                        0, indices
                    ) - data["selector_embedding"][item][None],
                    dim=-1,
                )[None]
                selector_values = standardize(selector)[0]
            groups.append({
                "class_id": class_id,
                "indices": indices,
                "probability": probability[item, class_id],
                "pose": standardize(pose)[0],
                "scalar": standardize(scalar)[0],
                "part": part[0],
                "part_values": part_values[0],
                "context": query_motion_context(
                    data, item, valid, probability[item, class_id]
                ),
                "contact_values": (
                    contact_values[0] if contact_values is not None else None
                ),
                "selector_values": selector_values,
            })
        result.append(groups)
    return result


def render_profile_action(data, features, scalar_weight, part_weight,
                          inner_top_k=1, inner_temperature=0.5):
    motions = []
    inference_valid = data["inference_valid"]
    for groups, valid in zip(features, inference_valid):
        candidates = []
        probabilities = []
        for group in groups:
            score = (
                group["pose"] + scalar_weight * group["scalar"]
                + part_weight * group["part"]
            )
            count = min(int(inner_top_k), len(score))
            local = score.topk(count, largest=False).indices
            bank_indices = group["indices"].index_select(0, local)
            inner_weight = torch.softmax(
                -score.index_select(0, local) / float(inner_temperature), dim=0
            )
            candidate = (
                data["train_bank"].index_select(0, bank_indices)
                * inner_weight[:, None, None, None]
            ).sum(0)
            candidates.append(candidate)
            probabilities.append(group["probability"])
        weight = torch.stack(probabilities)
        weight = weight / weight.sum().clamp_min(1e-6)
        canonical = (
            torch.stack(candidates) * weight[:, None, None, None]
        ).sum(0)
        motions.append(_render(canonical, valid, C.CACHE_FRAMES))
    return torch.stack(motions)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_profile_action_retrieval"
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
    features = retrieval_features(data)
    metrics = {}
    for scalar_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        for part_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
            action = render_profile_action(
                data, features, scalar_weight, part_weight
            )
            warped = monotonic_energy_warp(
                action, activity, data["target_valid"], 0.50, 0.30
            )
            low = smooth_valid_delta(
                warped - current, data["target_valid"], 31
            )
            predicted = current + 0.15 * low
            predicted = predicted - predicted[:, :, :1]
            name = (
                f"s{int(scalar_weight * 1000):04d}"
                f"_p{int(part_weight * 1000):04d}"
            )
            metrics[name] = _metric_batch(
                predicted, data["target_pose"], data["target_valid"],
                data["target_risk"],
            )
    scores = {name: pose_selection_score(metric) for name, metric in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_profile_action_retrieval",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "train_bank_only": True,
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
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
