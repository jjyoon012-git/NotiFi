"""Validation-only CSI motion-profile time warping for the promoted KP5 prior."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_part_motion_profile_reranking import (
    prepare,
    render_mixture,
    weighted_part_distance,
)
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def smooth_profile(values: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    return F.avg_pool1d(
        values[:, None], kernel, stride=1, padding=kernel // 2
    )[:, 0]


def pose_speed(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = pose.new_zeros(valid.shape)
    values[:, 1:] = torch.linalg.vector_norm(
        pose[:, 1:] - pose[:, :-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    return values * valid


def monotonic_energy_warp(
    pose: torch.Tensor,
    query_activity: torch.Tensor,
    valid: torch.Tensor,
    warp_strength: float,
    floor_fraction: float,
) -> torch.Tensor:
    """Retiming via monotonic cumulative energy, without target pose access."""
    candidate_activity = pose_speed(pose, valid)
    output = torch.zeros_like(pose)
    for item, mask in enumerate(valid):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        length = len(positions)
        if length < 2:
            output[item, positions] = pose[item, positions]
            continue
        query = smooth_profile(query_activity[item, positions][None])[0].clamp_min(0)
        candidate = smooth_profile(
            candidate_activity[item, positions][None]
        )[0].clamp_min(0)
        query_floor = floor_fraction * query.mean().clamp_min(0.02)
        candidate_floor = floor_fraction * candidate.mean().clamp_min(0.02)
        query_cdf = torch.cumsum(query + query_floor, dim=0)
        candidate_cdf = torch.cumsum(candidate + candidate_floor, dim=0)
        query_cdf = query_cdf / query_cdf[-1].clamp_min(1e-6)
        candidate_cdf = candidate_cdf / candidate_cdf[-1].clamp_min(1e-6)
        upper = torch.searchsorted(candidate_cdf, query_cdf).clamp(0, length - 1)
        lower = (upper - 1).clamp_min(0)
        lower_cdf = candidate_cdf[lower]
        upper_cdf = candidate_cdf[upper]
        fraction = (query_cdf - lower_cdf) / (
            upper_cdf - lower_cdf
        ).clamp_min(1e-6)
        mapped = lower.float() + fraction * (upper - lower).float()
        identity = torch.arange(length, device=pose.device, dtype=pose.dtype)
        mapped = (1.0 - warp_strength) * identity + warp_strength * mapped
        source_lower = mapped.floor().long().clamp(0, length - 1)
        source_upper = (source_lower + 1).clamp_max(length - 1)
        alpha = (mapped - source_lower.float())[:, None, None]
        source = pose[item, positions]
        output[item, positions] = (
            (1.0 - alpha) * source[source_lower] + alpha * source[source_upper]
        )
    return output


def parse_promoted_selection(calibration):
    selected = calibration["selection"]["name"]
    match = re.fullmatch(
        r"p(?P<weight>\d{4})_(?P<pattern>uniform|distal|limbs)"
        r"_t(?P<temperature>\d{3})_top(?P<top>\d+)"
        r"_s(?P<strength>\d{3})",
        selected,
    )
    if match is None:
        raise RuntimeError(f"unsupported promoted selection: {selected}")
    return {
        "name": selected,
        "weight": int(match["weight"]) / 1000.0,
        "pattern": match["pattern"],
        "temperature": int(match["temperature"]) / 100.0,
        "top_k": int(match["top"]),
        "strength": int(match["strength"]) / 1000.0,
    }


def promoted_candidate(data, config):
    patterns = {
        "uniform": (1, 1, 1, 1, 1, 1),
        "distal": (1.2, 0.7, 1.4, 1.4, 1.4, 1.4),
        "limbs": (0.8, 0.6, 1.7, 1.7, 1.7, 1.7),
    }
    part_distance = weighted_part_distance(
        data["part_distance"], patterns[config["pattern"]]
    )
    adjusted = (
        data["logits"] - 0.20 * data["scalar_distance"]
        - config["weight"] * part_distance
    )
    return render_mixture(
        data, adjusted, config["temperature"], config["top_k"]
    )


def activity_sources(data):
    part = data["predicted_part_profile"]
    return {
        "scalar": data["predicted_scalar_profile"],
        "parts_mean": part.mean(-1),
        "limbs": (
            0.10 * part[..., 0] + 0.05 * part[..., 1]
            + 0.2125 * part[..., 2] + 0.2125 * part[..., 3]
            + 0.2125 * part[..., 4] + 0.2125 * part[..., 5]
        ),
        "scalar_limbs": 0.50 * data["predicted_scalar_profile"] + 0.50 * (
            part[..., 2:].mean(-1)
        ),
    }


def add_common_arguments(parser):
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
        "--part-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "reranking_calibration.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_warp"
        / "warping_calibration.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.part_calibration.read_text(encoding="utf-8"))
    config = parse_promoted_selection(calibration)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "val", device)
    candidate = promoted_candidate(data, config)
    metrics = {}
    locked = (1.0 - config["strength"]) * data["baseline"] + config["strength"] * candidate
    metrics["part_profile_locked"] = _metric_batch(
        locked, data["target_pose"], data["target_valid"], data["target_risk"]
    )
    for source_name, activity in activity_sources(data).items():
        for floor in (0.05, 0.15, 0.30):
            for warp_strength in (0.25, 0.50, 0.75, 1.0):
                warped = monotonic_energy_warp(
                    candidate, activity, data["inference_valid"],
                    warp_strength, floor,
                )
                for blend in (0.625, 0.75, 0.875):
                    key = (
                        f"{source_name}_f{int(floor * 100):02d}"
                        f"_w{int(warp_strength * 100):03d}"
                        f"_b{int(blend * 1000):03d}"
                    )
                    metrics[key] = _metric_batch(
                        (1.0 - blend) * data["baseline"] + blend * warped,
                        data["target_pose"], data["target_valid"],
                        data["target_risk"],
                    )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_csi_profile_time_warp",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "base_selection": config,
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "part_calibration": report_path(args.part_calibration),
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
