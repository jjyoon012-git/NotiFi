"""Validation-only low-frequency action prior that preserves CSI dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_core_seed_selection import predict_locked
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import (
    add_action_arguments,
    predicted_action_motion,
)
from .diagnose_observability import report_path
from .train_kinetic_pose import pose_selection_score


def smooth_valid_delta(delta, valid, window):
    result = torch.zeros_like(delta)
    for item, mask in enumerate(valid):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(positions) == 0:
            continue
        values = delta[item, positions].flatten(1).T[None]
        left = (window - 1) // 2
        values = F.pad(values, (left, left), mode="replicate")
        values = F.avg_pool1d(values, window, stride=1)
        result[item, positions] = values[0].T.reshape(
            len(positions), C.N_JOINTS, 3
        )
    return result


@torch.no_grad()
def action_components(args, split, adaptive_config, device):
    data = prepare(args, split, device)
    current = predict_locked(data, adaptive_config)
    action_motion, _ = predicted_action_motion(data, 3)
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    warped = monotonic_energy_warp(
        action_motion, activity, data["target_valid"], 0.50, 0.30
    )
    return data, current, warped


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_frequency_action_prior"
        / "calibration.json",
    )
    args = parser.parse_args()
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, current, warped = action_components(
        args, "val", adaptive_config, device
    )
    metrics = {
        "current": _metric_batch(
            current, data["target_pose"], data["target_valid"],
            data["target_risk"],
        ),
        "direct_w150": _metric_batch(
            0.85 * current + 0.15 * warped,
            data["target_pose"], data["target_valid"], data["target_risk"],
        ),
    }
    delta = warped - current
    for window in (9, 17, 31, 61):
        low = smooth_valid_delta(delta, data["target_valid"], window)
        for weight in (0.10, 0.15, 0.20, 0.25, 0.30):
            predicted = current + weight * low
            predicted = predicted - predicted[:, :, :1]
            name = f"low{window:02d}_w{int(weight * 1000):03d}"
            metrics[name] = _metric_batch(
                predicted, data["target_pose"], data["target_valid"],
                data["target_risk"],
            )
    scores = {name: pose_selection_score(metric) for name, metric in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_frequency_preserving_action_prior",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "locked_action_selection": "CSI top3 train-only motion retrieval",
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
