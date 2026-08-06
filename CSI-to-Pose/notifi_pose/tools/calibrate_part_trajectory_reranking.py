"""Validation-only 3D anatomical-trajectory reranking on top of KP6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from ..motion_retrieval import PartTrajectoryHead
from .audit_motion_retrieval_oracle import _metric_batch, _render
from .calibrate_motion_profile_warping import (
    activity_sources,
    monotonic_energy_warp,
    parse_promoted_selection,
)
from .calibrate_part_motion_profile_reranking import (
    prepare,
    render_mixture,
    weighted_part_distance,
)
from .diagnose_observability import report_path
from .train_csi_part_trajectory import (
    part_trajectory_targets,
    predict_part_trajectory,
)
from .train_kinetic_pose import pose_selection_score


def candidate_part_trajectories(train_bank, pool, target_valid):
    values = []
    for item, valid in enumerate(target_valid):
        poses = torch.stack([
            _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            for index in pool["indices"][item]
        ])
        mask = valid[None].expand(len(poses), -1)
        values.append(part_trajectory_targets(poses, mask))
    return torch.stack(values)


def smooth_trajectory(values):
    shape = values.shape
    frames = shape[-3]
    flat = values.movedim(-3, -1).reshape(-1, 1, frames)
    flat = F.avg_pool1d(flat, 7, stride=1, padding=3)
    return flat.reshape(*shape[:-3], shape[-2], shape[-1], frames).movedim(-1, -3)


def trajectory_distance_components(predicted, candidates, valid):
    predicted = smooth_trajectory(predicted)
    candidates = smooth_trajectory(candidates)
    mask = valid[:, None, :, None]
    count = mask.sum(2).clamp_min(1)
    position = torch.linalg.vector_norm(
        candidates - predicted[:, None], dim=-1
    )
    position = (position * mask).sum(2) / count

    query_velocity = (predicted[:, 1:] - predicted[:, :-1]) * C.TARGET_FPS
    candidate_velocity = (
        candidates[:, :, 1:] - candidates[:, :, :-1]
    ) * C.TARGET_FPS
    velocity_mask = (valid[:, 1:] & valid[:, :-1])[:, None, :, None]
    velocity_count = velocity_mask.sum(2).clamp_min(1)
    velocity = torch.linalg.vector_norm(
        candidate_velocity - query_velocity[:, None], dim=-1
    )
    velocity = (velocity * velocity_mask).sum(2) / velocity_count
    query_speed = torch.linalg.vector_norm(query_velocity, dim=-1)
    candidate_speed = torch.linalg.vector_norm(candidate_velocity, dim=-1)
    cosine = (
        candidate_velocity * query_velocity[:, None]
    ).sum(-1) / (candidate_speed * query_speed[:, None]).clamp_min(1e-5)
    moving_weight = query_speed[:, None] * velocity_mask
    direction = ((1.0 - cosine) * moving_weight).sum(2) / (
        moving_weight.sum(2).clamp_min(1e-5)
    )

    endpoints = []
    for item, mask_item in enumerate(valid):
        positions = torch.nonzero(mask_item, as_tuple=False).flatten()
        endpoints.append(0.5 * (
            torch.linalg.vector_norm(
                candidates[item, :, positions[0]] - predicted[item, positions[0]],
                dim=-1,
            )
            + torch.linalg.vector_norm(
                candidates[item, :, positions[-1]] - predicted[item, positions[-1]],
                dim=-1,
            )
        ))
    return {
        "position": position,
        "velocity": velocity,
        "direction": direction,
        "endpoint": torch.stack(endpoints),
    }


def standardized_trajectory_distance(components, variant):
    part_weights = components["position"].new_tensor(
        (0.8, 0.6, 1.7, 1.7, 1.7, 1.7)
    )
    part_weights /= part_weights.sum()
    variants = {
        "position": (1.0, 0.0, 0.0, 0.25),
        "balanced": (1.0, 0.20, 0.15, 0.25),
        "motion": (0.50, 0.45, 0.30, 0.20),
    }
    weights = variants[variant]
    result = 0.0
    for weight, name in zip(
        weights, ("position", "velocity", "direction", "endpoint")
    ):
        if weight:
            combined = (components[name] * part_weights).sum(-1)
            result = result + weight * (
                (combined - combined.mean(-1, keepdim=True))
                / combined.std(-1, keepdim=True).clamp_min(1e-5)
            )
    return result


def add_arguments(parser):
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--scalar-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71" / "best_model.pt",
    )
    parser.add_argument(
        "--part-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83" / "best_model.pt",
    )
    parser.add_argument(
        "--trajectory-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_part_trajectory_seed97" / "best_model.pt",
    )
    parser.add_argument(
        "--part-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "reranking_calibration.json",
    )
    parser.add_argument(
        "--warp-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_warp"
        / "warping_calibration.json",
    )


@torch.no_grad()
def prepare_trajectory(args, split, device):
    data = prepare(args, split, device)
    checkpoint = torch.load(
        args.trajectory_checkpoint, map_location="cpu", weights_only=False
    )
    model = PartTrajectoryHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predicted = predict_part_trajectory(
        model, data["cache"], data["target_valid"], device
    )
    candidates = candidate_part_trajectories(
        data["train_bank"], data["pool"], data["target_valid"]
    )
    data["trajectory_components"] = trajectory_distance_components(
        predicted, candidates, data["target_valid"]
    )
    data["trajectory_checkpoint"] = checkpoint
    return data


def adjusted_logits(data, part_config, trajectory_weight, variant):
    patterns = {
        "uniform": (1, 1, 1, 1, 1, 1),
        "distal": (1.2, 0.7, 1.4, 1.4, 1.4, 1.4),
        "limbs": (0.8, 0.6, 1.7, 1.7, 1.7, 1.7),
    }
    part_distance = weighted_part_distance(
        data["part_distance"], patterns[part_config["pattern"]]
    )
    trajectory_distance = standardized_trajectory_distance(
        data["trajectory_components"], variant
    )
    return (
        data["logits"] - 0.20 * data["scalar_distance"]
        - part_config["weight"] * part_distance
        - trajectory_weight * trajectory_distance
    )


def apply_locked_warp(data, candidate):
    activity = activity_sources(data)["scalar_limbs"]
    return monotonic_energy_warp(
        candidate, activity, data["target_valid"],
        warp_strength=0.50, floor_fraction=0.30,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_part_trajectory_seed97"
        / "reranking_calibration.json",
    )
    args = parser.parse_args()
    part_calibration = json.loads(
        args.part_calibration.read_text(encoding="utf-8")
    )
    part_config = parse_promoted_selection(part_calibration)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_trajectory(args, "val", device)
    metrics = {}
    for variant in ("position", "balanced", "motion"):
        for trajectory_weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75):
            if trajectory_weight == 0 and variant != "position":
                continue
            adjusted = adjusted_logits(
                data, part_config, trajectory_weight, variant
            )
            for temperature in (0.35, 0.50, 0.75):
                for top_k in (3, 5, 8):
                    candidate = render_mixture(
                        data, adjusted, temperature, top_k
                    )
                    warped = apply_locked_warp(data, candidate)
                    key = (
                        f"{variant}_g{int(trajectory_weight * 1000):04d}"
                        f"_t{int(temperature * 100):03d}_top{top_k}"
                    )
                    metrics[key] = _metric_batch(
                        0.25 * data["baseline"] + 0.75 * warped,
                        data["target_pose"], data["target_valid"],
                        data["target_risk"],
                    )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_part_trajectory_reranking",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "base_part_selection": part_config,
        "locked_warp": "scalar_limbs_f30_w050_b750",
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "trajectory_validation": data["trajectory_checkpoint"]["selection"],
        "trajectory_checkpoint": report_path(args.trajectory_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"], "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
