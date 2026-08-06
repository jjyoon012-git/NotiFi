"""Audit how much sealed yja/E02 pose error is explained by one fixed rotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataio.dataset import build_datasets
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_sealed import smooth_valid
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_calibration_aware_v14 import split_support_queries, subset_dataset


def fit_rotation(predicted: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    covariance = predicted.reshape(-1, 3).T @ target.reshape(-1, 3)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(predicted - target, dim=-1).mean())


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=Path("work_v2/runs/p2_sub_single_clean_finetune/best_model.pt"),
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=Path("docs/results/v13s_pruned_pose_root_ensemble.json"),
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=Path(
            "work_v2/runs/p2_v12w_robust_classification_ensemble/validation.json"
        ),
    )
    parser.add_argument("--source-exp", default="single_split_lmh_e01")
    parser.add_argument("--target-fold", default="yja_E02")
    parser.add_argument("--support-per-pose", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/yja_fixed_orientation_audit.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.exp = args.source_exp
    root_lock = _read_locked(args.root_calibration, args.source_exp)
    class_lock = _read_locked(
        args.classification_calibration, args.source_exp
    )
    model, _ = build_locked_model(args, device, root_lock, class_lock)
    model.eval()
    target = pose_only(build_datasets(
        exp="sealed", fold=args.target_fold, baseline="sub", seed=args.seed
    )["test"])
    support, query = split_support_queries(
        target, args.support_per_pose, args.seed
    )
    selected = subset_dataset(target, query[args.target_fold])
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False)

    predictions = []
    targets = []
    dangers = []
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        valid = batch["valid"].bool()
        predicted = smooth_valid(
            output["pose_rel"].float().cpu(), valid, 5
        )
        predictions.append(predicted[valid])
        targets.append(batch["pose_rel"].float()[valid])
        danger = batch["risk_id"].eq(2)[:, None].expand_as(valid)
        dangers.append(danger[valid])
    predicted = torch.cat(predictions)
    target_pose = torch.cat(targets)
    danger_mask = torch.cat(dangers)
    rotation = fit_rotation(predicted, target_pose)
    rotated = predicted @ rotation
    denominator = rotated.square().sum().clamp_min(1e-8)
    scale = (rotated * target_pose).sum() / denominator
    scaled = scale * rotated

    result = {
        "run": "yja_fixed_orientation_audit",
        "protocol": f"sealed/{args.target_fold}",
        "purpose": "diagnostic only; target GT is not used for training or model selection",
        "support_trials_excluded": len(support[args.target_fold]),
        "test_trials": len(selected),
        "danger_test_trials": int((selected.index.risk_id == 2).sum()),
        "rotation_matrix": rotation.tolist(),
        "rotation_determinant": float(torch.linalg.det(rotation)),
        "similarity_scale": float(scale),
        "all": {
            "raw_pose_mpjpe_m": error(predicted, target_pose),
            "fixed_rotation_mpjpe_m": error(rotated, target_pose),
            "fixed_rotation_scale_mpjpe_m": error(scaled, target_pose),
        },
        "danger": {
            "raw_pose_mpjpe_m": error(
                predicted[danger_mask], target_pose[danger_mask]
            ),
            "fixed_rotation_mpjpe_m": error(
                rotated[danger_mask], target_pose[danger_mask]
            ),
            "fixed_rotation_scale_mpjpe_m": error(
                scaled[danger_mask], target_pose[danger_mask]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
