"""Audit velocity, acceleration, and jerk without opening the test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from .. import contract as C
from ..trainer import set_seed
from .calibrate_v11_residual_temporal import _build_model
from .evaluate_sealed import smooth_valid
from .train_seen_v4_trajectory import make_loaders


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else math.nan


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 3:
        return math.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 1e-8:
        return math.nan
    return float((left * right).sum() / denominator)


def _motion(sequence: torch.Tensor, order: int) -> torch.Tensor:
    for _ in range(order):
        sequence = torch.diff(sequence, dim=0) * C.TARGET_FPS
    return torch.linalg.vector_norm(sequence, dim=-1).mean(-1)


def _rows(predicted: torch.Tensor, target: torch.Tensor,
          valid: torch.Tensor, danger: torch.Tensor) -> list[dict]:
    rows = []
    for item in range(len(predicted)):
        row = {"danger": bool(danger[item])}
        for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            mask = valid[item][order:].clone()
            for offset in range(order):
                mask &= valid[item][offset:len(valid[item]) - order + offset]
            pred_motion = _motion(predicted[item], order)
            target_motion = _motion(target[item], order)
            if not mask.any():
                row[f"{name}_ratio"] = math.nan
                row[f"{name}_correlation"] = math.nan
                continue
            pred_selected = pred_motion[mask]
            target_selected = target_motion[mask]
            target_mean = float(target_selected.mean())
            row[f"{name}_ratio"] = (
                float(pred_selected.mean()) / max(target_mean, 1e-8)
            )
            row[f"{name}_correlation"] = _correlation(
                pred_selected, target_selected
            )
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    keys = (
        "velocity_ratio", "velocity_correlation",
        "acceleration_ratio", "acceleration_correlation",
        "jerk_ratio", "jerk_correlation",
    )
    return {key: _finite_mean([row[key] for row in rows]) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--pose-strength", type=float, required=True)
    parser.add_argument("--root-strength", type=float, default=1.0)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument("--residual-window", type=int, default=1)
    parser.add_argument("--residual-blend", type=float, default=0.0)
    parser.add_argument(
        "--risk-adaptive", choices=("none", "probability", "hard"),
        default="none",
    )
    parser.add_argument("--danger-logit-bias", type=float, default=1.1)
    parser.add_argument("--root-residual-window", type=int, default=1)
    parser.add_argument("--root-residual-blend", type=float, default=0.0)
    parser.add_argument("--evaluation-smooth-window", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model = _build_model(args, device)
    model.base.set_calibration(
        args.residual_window, args.residual_blend,
        args.risk_adaptive, args.danger_logit_bias,
    )
    model.base.set_root_calibration(
        args.root_residual_window, args.root_residual_blend
    )
    model.eval()

    relative_rows = []
    absolute_rows = []
    with torch.no_grad():
        for batch in loaders["val"]:
            output = model(
                batch["csi"].to(device), batch["link_mask"].to(device)
            )
            valid = batch["valid"].bool()
            predicted = smooth_valid(
                output["pose_rel"].float().cpu(), valid,
                args.evaluation_smooth_window,
            )
            predicted_root = smooth_valid(
                output["root"].float().cpu(), valid,
                args.evaluation_smooth_window,
            )
            target = batch["pose_rel"].float()
            target_root = batch["root"].float()
            danger = batch["risk_id"].eq(2)
            relative_rows.extend(_rows(
                predicted, target, valid, danger,
            ))
            absolute_rows.extend(_rows(
                predicted + predicted_root[:, :, None],
                target + target_root[:, :, None], valid, danger,
            ))
    def report(rows: list[dict]) -> dict:
        return {
            "overall": _aggregate(rows),
            "danger": _aggregate([row for row in rows if row["danger"]]),
        }
    result = {
        "run": "p2_v11_temporal_dynamics_audit",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "settings": {
            "pose_strength": args.pose_strength,
            "residual_window": args.residual_window,
            "residual_blend": args.residual_blend,
            "risk_adaptive": args.risk_adaptive,
            "danger_logit_bias": args.danger_logit_bias,
            "root_residual_window": args.root_residual_window,
            "root_residual_blend": args.root_residual_blend,
            "evaluation_smooth_window": args.evaluation_smooth_window,
        },
        "relative_pose": report(relative_rows),
        "absolute_pose": report(absolute_rows),
        "trials": len(relative_rows),
        "danger_trials": sum(row["danger"] for row in relative_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
