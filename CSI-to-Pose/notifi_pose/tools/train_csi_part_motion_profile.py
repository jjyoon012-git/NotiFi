"""Train CSI-only framewise motion profiles for six anatomical regions."""

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
from ..motion_retrieval import PartMotionProfileHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only, report_path
from .train_csi_motion_profile import correlation, profile_features


PART_NAMES = tuple(C.JOINT_GROUPS)
DISTAL_PART_WEIGHT = torch.tensor((1.15, 0.75, 1.35, 1.35, 1.35, 1.35))


def part_speed_targets(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute root-relative joint speed separately for each body region."""
    velocity = torch.zeros_like(pose)
    velocity[:, 1:] = (pose[:, 1:] - pose[:, :-1]) * C.TARGET_FPS
    values = []
    for joints in C.JOINT_GROUPS.values():
        values.append(
            torch.linalg.vector_norm(velocity[:, :, list(joints)], dim=-1).mean(-1)
        )
    return torch.stack(values, dim=-1).clamp_max(3.0) * valid[..., None]


@torch.no_grad()
def predict_part_profile(model, cache, valid, device):
    model.eval()
    values = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(model(
            profile_features(cache, indices).to(device),
            valid.index_select(0, indices).to(device),
        )["part_speed"].float().cpu())
    return torch.cat(values)


@torch.no_grad()
def evaluate(model, cache, target, valid, risk, device):
    predicted = predict_part_profile(model, cache, valid, device)
    absolute = (predicted - target).abs()
    danger_mask = valid & (risk[:, None] == 2)
    per_part = {}
    all_corr, danger_corr = [], []
    for part, name in enumerate(PART_NAMES):
        correlations, danger_correlations = [], []
        for item, mask in enumerate(valid):
            value = correlation(predicted[item, mask, part], target[item, mask, part])
            correlations.append(value)
            if int(risk[item]) == 2:
                danger_correlations.append(value)
        all_corr.append(float(np.mean(correlations)))
        danger_corr.append(float(np.mean(danger_correlations)))
        per_part[name] = {
            "mae_mps": float(absolute[..., part][valid].mean()),
            "danger_mae_mps": float(absolute[..., part][danger_mask].mean()),
            "correlation_mean": all_corr[-1],
            "danger_correlation_mean": danger_corr[-1],
        }
    weights = DISTAL_PART_WEIGHT / DISTAL_PART_WEIGHT.sum()
    mae = torch.stack([torch.tensor(per_part[name]["mae_mps"]) for name in PART_NAMES])
    danger_mae = torch.stack([
        torch.tensor(per_part[name]["danger_mae_mps"]) for name in PART_NAMES
    ])
    result = {
        "mae_mps": float((mae * weights).sum()),
        "danger_mae_mps": float((danger_mae * weights).sum()),
        "correlation_mean": float(np.average(all_corr, weights=weights.numpy())),
        "danger_correlation_mean": float(
            np.average(danger_corr, weights=weights.numpy())
        ),
        "per_part": per_part,
    }
    result["selection_score"] = (
        result["mae_mps"] + result["danger_mae_mps"]
        - 0.05 * result["correlation_mean"]
        - 0.10 * result["danger_correlation_mean"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83",
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
    train_target = part_speed_targets(train_pose, train_valid)
    val_target = part_speed_targets(val_pose, val_valid)
    input_dim = profile_features(train_cache, torch.arange(1)).shape[-1]
    model = PartMotionProfileHead(input_dim=input_dim).to(device)

    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(2.75, dtype=torch.double),
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
            features = features + 0.012 * torch.randn_like(features)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                output = model(features, mask)
                predicted = output["part_speed"]
                frame_weight = 1.0 + 1.5 * (target / 0.6).clamp_max(2.0)
                frame_weight *= part_weight[None, None]
                frame_weight *= torch.where(danger[:, None, None], 1.8, 1.0)
                frame_weight *= mask[..., None]
                regression = F.smooth_l1_loss(
                    torch.log1p(predicted), torch.log1p(target),
                    reduction="none", beta=0.10,
                )
                regression = (regression * frame_weight).sum() / frame_weight.sum()
                motion = F.binary_cross_entropy_with_logits(
                    output["part_motion_logits"], (target > 0.14).float(),
                    reduction="none",
                )
                motion = (motion * frame_weight).sum() / frame_weight.sum()
                delta_mask = (mask[:, 1:] & mask[:, :-1])[..., None]
                acceleration = F.smooth_l1_loss(
                    predicted[:, 1:] - predicted[:, :-1],
                    target[:, 1:] - target[:, :-1],
                    reduction="none", beta=0.05,
                )
                acceleration = (
                    acceleration * delta_mask * part_weight[None, None]
                ).sum() / (delta_mask.sum() * part_weight.sum()).clamp_min(1.0)
                loss = regression + 0.20 * motion + 0.12 * acceleration
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append((float(loss), float(regression), float(motion), float(acceleration)))
        validation_result = evaluate(
            model, val_cache, val_target, val_valid, val_risk, device
        )
        values = np.asarray(losses)
        row = {
            "epoch": epoch,
            "train": {
                "loss": float(values[:, 0].mean()),
                "regression": float(values[:, 1].mean()),
                "motion": float(values[:, 2].mean()),
                "acceleration": float(values[:, 3].mean()),
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
        "run": "KP5-PART-MOTION-PROFILE-EXP01",
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
