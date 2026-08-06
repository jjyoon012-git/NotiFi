"""Train a monotonic CSI-only action-progress head on train data."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import MotionProgressHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only, report_path
from .train_csi_contact_profile import contact_features


def target_progress(pose, valid, floor_fraction=0.30):
    speed = pose.new_zeros(valid.shape)
    speed[:, 1:] = torch.linalg.vector_norm(
        pose[:, 1:] - pose[:, :-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    speed = F.avg_pool1d(
        speed[:, None], 9, stride=1, padding=4
    )[:, 0].clamp_min(0) * valid
    count = valid.sum(1).clamp_min(1)
    mean = speed.sum(1) / count
    increment = (speed + floor_fraction * mean[:, None].clamp_min(0.02)) * valid
    cumulative = torch.cumsum(increment, dim=1)
    total = cumulative.gather(
        1, (valid.long().sum(1) - 1).clamp_min(0)[:, None]
    ).clamp_min(1e-6)
    return cumulative / total * valid


@torch.no_grad()
def predict_progress(model, cache, valid, device):
    model.eval()
    values = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(model(
            contact_features(cache, indices).to(device),
            valid.index_select(0, indices).to(device),
        )["progress"].float().cpu())
    return torch.cat(values)


@torch.no_grad()
def evaluate(model, cache, target, valid, risk, device):
    inference_valid = cache["frame_mask"].bool()
    predicted = predict_progress(model, cache, inference_valid, device)
    valid = valid & inference_valid
    weight = valid.float() * torch.where(
        risk[:, None] == 2, 2.0, 1.0
    )
    mae = ((predicted - target).abs() * weight).sum() / weight.sum().clamp_min(1)
    delta_valid = valid[:, 1:] & valid[:, :-1]
    delta = predicted[:, 1:] - predicted[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    delta_mae = (
        (delta - target_delta).abs() * delta_valid
    ).sum() / delta_valid.sum().clamp_min(1)
    danger = valid & (risk[:, None] == 2)
    danger_mae = (
        (predicted - target).abs() * danger
    ).sum() / danger.sum().clamp_min(1)
    return {
        "progress_mae": float(mae),
        "danger_progress_mae": float(danger_mae),
        "increment_mae": float(delta_mae),
        "selection_score": float(mae + 0.35 * danger_mae + 2.0 * delta_mae),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp15_motion_progress_seed263",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cache = torch.load(
        args.feature_root / "train_features.pt", map_location="cpu",
        weights_only=False,
    )
    val_cache = torch.load(
        args.feature_root / "val_features.pt", map_location="cpu",
        weights_only=False,
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, _, train_risk = _load_pose_arrays(train)
    val_pose, val_valid, _, val_risk = _load_pose_arrays(validation)
    train_target = target_progress(train_pose, train_valid)
    val_target = target_progress(val_pose, val_valid)
    model_config = {
        "input_dim": contact_features(train_cache, torch.arange(1)).shape[-1]
    }
    model = MotionProgressHead(**model_config).to(device)
    weight = train.sampler_weights() * torch.where(
        train_risk == 2, torch.tensor(2.5, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))), batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weight, len(train), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ), num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (
            1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)
        ),
    )
    best, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            target_valid = train_valid.index_select(0, indices).to(device)
            valid = train_cache["frame_mask"].index_select(
                0, indices
            ).to(device).bool()
            supervision_valid = valid & target_valid
            target = train_target.index_select(0, indices).to(device)
            risk = train_risk.index_select(0, indices).to(device)
            feature = contact_features(train_cache, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(feature, valid)["progress"]
            frame_weight = supervision_valid * torch.where(
                risk[:, None] == 2, 1.75, 1.0
            )
            level = F.smooth_l1_loss(
                output, target, beta=0.03, reduction="none"
            )
            level = (level * frame_weight).sum() / frame_weight.sum().clamp_min(1)
            delta_valid = (
                supervision_valid[:, 1:] & supervision_valid[:, :-1]
            )
            delta = output[:, 1:] - output[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            increment = F.smooth_l1_loss(
                delta, target_delta, beta=0.005, reduction="none"
            )
            increment = (increment * delta_valid).sum() / delta_valid.sum().clamp_min(1)
            loss = level + 0.50 * increment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(
            model, val_cache, val_target, val_valid, val_risk, device
        )
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation": metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or metrics["selection_score"] < best["score"] - 1e-6:
            best = {
                "epoch": epoch, "score": metrics["selection_score"],
                "validation": metrics,
                "model": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best["model"], "model_config": model_config,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "protocol": args.exp, "seed": args.seed,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "feature_root": report_path(args.feature_root),
    }, args.run_dir / "best_model.pt")
    result = {
        "status": "validation_selected_monotonic_motion_progress",
        "protocol": args.exp,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "history": history,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
