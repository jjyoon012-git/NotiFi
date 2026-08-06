"""Validation-only temporal alignment of a retrieved motion to CSI activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import _load_pose_arrays, _metric_batch, _render
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score


def speed_profile(pose: torch.Tensor) -> torch.Tensor:
    speed = torch.linalg.vector_norm(pose[1:] - pose[:-1], dim=-1).mean(-1)
    speed = F.avg_pool1d(speed[None, None], 9, stride=1, padding=4)[0, 0]
    return speed


def normalized_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-8))


def shift_sequence(values: torch.Tensor, lag: int) -> torch.Tensor:
    if lag == 0:
        return values
    result = torch.empty_like(values)
    if lag > 0:
        result[:lag] = values[0]
        result[lag:] = values[:-lag]
    else:
        amount = -lag
        result[-amount:] = values[-1]
        result[:-amount] = values[amount:]
    return result


def resample_sequence(values: torch.Tensor, frames: int) -> torch.Tensor:
    return F.interpolate(
        values.flatten(1).T[None], size=frames,
        mode="linear", align_corners=True,
    )[0].T.reshape(frames, C.N_JOINTS, 3)


def align_affine(candidate: torch.Tensor, baseline: torch.Tensor,
                 lag_radius: int, rates: tuple[float, ...]) -> torch.Tensor:
    target_speed = speed_profile(baseline)
    if float(target_speed.mean()) < 1e-4:
        return candidate
    best_score, best = -float("inf"), candidate
    frames = len(candidate)
    for rate in rates:
        resized = resample_sequence(
            candidate, max(2, int(round(frames * rate)))
        )
        if len(resized) >= frames:
            start = (len(resized) - frames) // 2
            resized = resized[start:start + frames]
        else:
            pad = frames - len(resized)
            left = pad // 2
            resized = torch.cat((
                resized[:1].expand(left, -1, -1), resized,
                resized[-1:].expand(pad - left, -1, -1),
            ))
        for lag in range(-lag_radius, lag_radius + 1, 3):
            shifted = shift_sequence(resized, lag)
            score = normalized_correlation(
                target_speed, speed_profile(shifted)
            )
            if score > best_score:
                best_score, best = score, shifted
    return best


def cumulative_phase_align(candidate: torch.Tensor, baseline: torch.Tensor,
                           strength: float) -> torch.Tensor:
    target_speed = speed_profile(baseline).clamp_min(1e-5)
    source_speed = speed_profile(candidate).clamp_min(1e-5)
    target_phase = torch.cat((target_speed.new_zeros(1), target_speed.cumsum(0)))
    source_phase = torch.cat((source_speed.new_zeros(1), source_speed.cumsum(0)))
    target_phase = target_phase / target_phase[-1].clamp_min(1e-6)
    source_phase = source_phase / source_phase[-1].clamp_min(1e-6)
    source_index = torch.searchsorted(source_phase, target_phase).clamp(
        1, len(candidate) - 1
    )
    lower = source_index - 1
    upper = source_index
    denominator = (source_phase[upper] - source_phase[lower]).clamp_min(1e-6)
    fraction = (target_phase - source_phase[lower]) / denominator
    warped = (
        candidate[lower] * (1.0 - fraction[:, None, None])
        + candidate[upper] * fraction[:, None, None]
    )
    return (1.0 - strength) * candidate + strength * warped


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--feature-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "val_features.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "time_alignment_calibration.json",
    )
    args = parser.parse_args()
    selector = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    reranker = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    validation = QualityWeightedDataset(
        pose_only(datasets["val"]), protocol_audit_path(args.exp)
    )
    target_pose, target_valid, _, target_risk = _load_pose_arrays(validation)
    baseline = cache["baseline_pose"].float()
    bank = selector["train_bank"].float()
    indices = reranker["selected_validation_bank_indices"].long()
    candidates = torch.stack([
        _render(bank[int(index)], valid, C.CACHE_FRAMES)
        for index, valid in zip(indices, target_valid)
    ])
    aligned: dict[str, list[torch.Tensor]] = {
        "uniform": [], "lag15": [], "lag30": [], "lag45": [],
        "affine": [], "phase25": [], "phase50": [], "phase75": [],
    }
    for candidate, base, valid in zip(candidates, baseline, target_valid):
        frames = torch.nonzero(valid, as_tuple=False).flatten()
        if len(frames) < 3:
            for values in aligned.values():
                values.append(candidate)
            continue
        sequence = candidate[frames]
        reference = base[frames]
        variants = {
            "uniform": sequence,
            "lag15": align_affine(sequence, reference, 15, (1.0,)),
            "lag30": align_affine(sequence, reference, 30, (1.0,)),
            "lag45": align_affine(sequence, reference, 45, (1.0,)),
            "affine": align_affine(sequence, reference, 30, (0.85, 1.0, 1.15)),
            "phase25": cumulative_phase_align(sequence, reference, 0.25),
            "phase50": cumulative_phase_align(sequence, reference, 0.50),
            "phase75": cumulative_phase_align(sequence, reference, 0.75),
        }
        for name, values in variants.items():
            rendered = candidate.new_zeros(candidate.shape)
            rendered[frames] = values
            aligned[name].append(rendered)
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for name, values in aligned.items():
        candidate = torch.stack(values)
        for strength in (0.375, 0.50, 0.625):
            key = f"{name}_cartesian_{int(strength * 1000):03d}"
            metrics[key] = _metric_batch(
                (1.0 - strength) * baseline + strength * candidate,
                target_pose, target_valid, target_risk,
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_time_alignment",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "alignment_signal": "locked CSI pose mean-joint speed only",
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
