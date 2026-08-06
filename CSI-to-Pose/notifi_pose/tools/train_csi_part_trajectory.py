"""Train a low-dimensional CSI-to-anatomical-trajectory bottleneck."""

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
from ..motion_retrieval import PartTrajectoryHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only, report_path
from .train_csi_motion_profile import profile_features
from .train_csi_part_motion_profile import DISTAL_PART_WEIGHT, PART_NAMES


def part_trajectory_targets(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = [pose[:, :, list(joints)].mean(2) for joints in C.JOINT_GROUPS.values()]
    return torch.stack(values, dim=2) * valid[..., None, None]


@torch.no_grad()
def predict_part_trajectory(model, cache, valid, device):
    model.eval()
    values = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(model(
            profile_features(cache, indices).to(device),
            valid.index_select(0, indices).to(device),
        )["part_trajectory"].float().cpu())
    return torch.cat(values)


@torch.no_grad()
def evaluate(model, cache, target, valid, risk, device):
    predicted = predict_part_trajectory(model, cache, valid, device)
    error = torch.linalg.vector_norm(predicted - target, dim=-1)
    danger_mask = valid & (risk[:, None] == 2)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    velocity_error = torch.linalg.vector_norm(
        velocity_predicted - velocity_target, dim=-1
    ) * C.TARGET_FPS
    velocity_mask = valid[:, 1:] & valid[:, :-1]
    danger_velocity_mask = velocity_mask & (risk[:, None] == 2)
    weights = DISTAL_PART_WEIGHT / DISTAL_PART_WEIGHT.sum()
    part_mae = torch.stack([error[..., part][valid].mean() for part in range(6)])
    danger_part_mae = torch.stack([
        error[..., part][danger_mask].mean() for part in range(6)
    ])
    part_velocity = torch.stack([
        velocity_error[..., part][velocity_mask].mean() for part in range(6)
    ])
    danger_part_velocity = torch.stack([
        velocity_error[..., part][danger_velocity_mask].mean() for part in range(6)
    ])
    result = {
        "mae_m": float((part_mae * weights).sum()),
        "danger_mae_m": float((danger_part_mae * weights).sum()),
        "velocity_mae_mps": float((part_velocity * weights).sum()),
        "danger_velocity_mae_mps": float((danger_part_velocity * weights).sum()),
        "per_part": {
            name: {
                "mae_m": float(part_mae[part]),
                "danger_mae_m": float(danger_part_mae[part]),
                "velocity_mae_mps": float(part_velocity[part]),
                "danger_velocity_mae_mps": float(danger_part_velocity[part]),
            }
            for part, name in enumerate(PART_NAMES)
        },
    }
    result["selection_score"] = (
        result["mae_m"] + 1.25 * result["danger_mae_m"]
        + 0.10 * result["velocity_mae_mps"]
        + 0.15 * result["danger_velocity_mae_mps"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_part_trajectory_seed97",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cache = torch.load(
        args.feature_root / "train_features.pt", map_location="cpu", weights_only=False
    )
    val_cache = torch.load(
        args.feature_root / "val_features.pt", map_location="cpu", weights_only=False
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, _, train_risk = _load_pose_arrays(train)
    val_pose, val_valid, _, val_risk = _load_pose_arrays(validation)
    train_target = part_trajectory_targets(train_pose, train_valid)
    val_target = part_trajectory_targets(val_pose, val_valid)
    input_dim = profile_features(train_cache, torch.arange(1)).shape[-1]
    model = PartTrajectoryHead(input_dim=input_dim).to(device)

    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(3.0, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))), batch_size=args.batch_size,
        sampler=sampler, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (
            1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    part_weight = DISTAL_PART_WEIGHT.to(device)
    best, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            features = profile_features(train_cache, indices).to(device)
            mask = train_valid.index_select(0, indices).to(device)
            target = train_target.index_select(0, indices).to(device)
            danger = train_risk.index_select(0, indices).to(device) == 2
            speed = torch.zeros(target.shape[:-1], device=device)
            speed[:, 1:] = torch.linalg.vector_norm(
                target[:, 1:] - target[:, :-1], dim=-1
            ) * C.TARGET_FPS
            features = features + 0.012 * torch.randn_like(features)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                predicted = model(features, mask)["part_trajectory"]
                frame_weight = 1.0 + 1.5 * (speed / 0.5).clamp_max(2.0)
                frame_weight *= part_weight[None, None]
                frame_weight *= torch.where(danger[:, None, None], 1.8, 1.0)
                frame_weight *= mask[..., None]
                position = F.smooth_l1_loss(
                    predicted, target, reduction="none", beta=0.10
                ).mean(-1)
                position = (position * frame_weight).sum() / frame_weight.sum()
                predicted_velocity = predicted[:, 1:] - predicted[:, :-1]
                target_velocity = target[:, 1:] - target[:, :-1]
                delta_mask = (mask[:, 1:] & mask[:, :-1])[..., None]
                velocity = F.smooth_l1_loss(
                    predicted_velocity, target_velocity,
                    reduction="none", beta=0.04,
                ).mean(-1)
                velocity_weight = frame_weight[:, 1:] * delta_mask
                velocity = (
                    velocity * velocity_weight
                ).sum() / velocity_weight.sum().clamp_min(1.0)
                predicted_pair = predicted[:, :, :, None] - predicted[:, :, None, :]
                target_pair = target[:, :, :, None] - target[:, :, None, :]
                configuration = F.smooth_l1_loss(
                    predicted_pair, target_pair, reduction="none", beta=0.10
                ).mean(-1).mean((2, 3))
                configuration = (
                    configuration * mask
                ).sum() / mask.sum().clamp_min(1)
                loss = position + 0.35 * velocity + 0.20 * configuration
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append((
                float(loss.detach()), float(position.detach()),
                float(velocity.detach()), float(configuration.detach()),
            ))
        validation_result = evaluate(
            model, val_cache, val_target, val_valid, val_risk, device
        )
        values = np.asarray(losses)
        row = {
            "epoch": epoch,
            "train": {
                "loss": float(values[:, 0].mean()),
                "position": float(values[:, 1].mean()),
                "velocity": float(values[:, 2].mean()),
                "configuration": float(values[:, 3].mean()),
            },
            "validation": validation_result,
        }
        history.append(row)
        print(json.dumps({
            "epoch": epoch, "train_loss": row["train"]["loss"],
            "validation": {key: value for key, value in validation_result.items()
                           if key != "per_part"},
        }), flush=True)
        score = validation_result["selection_score"]
        if best is None or score < best["score"] - 1e-5:
            best = {
                "epoch": epoch, "score": score,
                "state": copy.deepcopy(model.state_dict()),
                "validation": validation_result,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best is not None
    result = {
        "run": "KP6-PART-TRAJECTORY-EXP01",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "part_order": PART_NAMES,
        "config": {
            **{key: value for key, value in vars(args).items()
               if not isinstance(value, Path)},
            "feature_root": report_path(args.feature_root),
            "run_dir": report_path(args.run_dir),
        },
        "selection": {"epoch": best["epoch"], "score": best["score"]},
        "validation": best["validation"],
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "model": best["state"],
        "model_config": {"input_dim": input_dim, "parts": len(PART_NAMES)},
        "selection": result["selection"], "part_order": PART_NAMES,
        "feature_root": report_path(args.feature_root),
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
