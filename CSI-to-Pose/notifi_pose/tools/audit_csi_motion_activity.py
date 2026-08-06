"""Audit whether frozen CSI activity contains deployable temporal alignment signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (
        torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    ).clamp_min(1e-7))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    args = parser.parse_args()
    cache = torch.load(
        args.feature_root / f"{args.split}_features.pt",
        map_location="cpu", weights_only=False,
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    dataset = QualityWeightedDataset(
        pose_only(datasets[args.split]), protocol_audit_path(args.exp)
    )
    pose, valid, _, risk = _load_pose_arrays(dataset)
    rows = []
    for item in range(len(dataset)):
        mask = valid[item]
        length = int(mask.sum())
        target = pose[item, :length]
        speed = torch.zeros(length)
        speed[1:] = torch.linalg.vector_norm(
            target[1:] - target[:-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        activity = cache["motion_activity"][item, :length].float()
        activity = F.avg_pool1d(
            activity[None, None], 9, stride=1, padding=4
        )[0, 0]
        speed = F.avg_pool1d(speed[None, None], 9, stride=1, padding=4)[0, 0]
        raw = correlation(activity, speed)
        best = (-2.0, 0)
        for lag in range(-30, 31):
            if lag < 0:
                value = correlation(activity[-lag:], speed[:lag])
            elif lag > 0:
                value = correlation(activity[:-lag], speed[lag:])
            else:
                value = raw
            if value > best[0]:
                best = (value, lag)
        rows.append({
            "risk": int(risk[item]), "raw": raw,
            "max_correlation": best[0], "lag_frames": best[1],
        })
    result = {}
    for name, keep in (
        ("all", [True] * len(rows)),
        ("danger", [row["risk"] == 2 for row in rows]),
    ):
        selected = [row for row, flag in zip(rows, keep) if flag]
        result[name] = {
            "trials": len(selected),
            "raw_mean": float(np.mean([row["raw"] for row in selected])),
            "raw_median": float(np.median([row["raw"] for row in selected])),
            "max_mean": float(np.mean([row["max_correlation"] for row in selected])),
            "max_median": float(np.median([row["max_correlation"] for row in selected])),
            "lag_median_frames": float(np.median([row["lag_frames"] for row in selected])),
            "lag_abs_median_frames": float(np.median([
                abs(row["lag_frames"]) for row in selected
            ])),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
