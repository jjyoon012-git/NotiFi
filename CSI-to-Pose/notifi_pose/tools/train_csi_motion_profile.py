"""Train a small CSI-only framewise motion-profile predictor."""

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
from ..motion_retrieval import MotionProfileHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only, report_path


def profile_features(cache: dict, indices: torch.Tensor | None = None) -> torch.Tensor:
    def take(value):
        return value if indices is None else value.index_select(0, indices)
    return torch.cat((
        take(cache["features"]).float(),
        take(cache["motion_activity"]).float()[..., None],
        torch.softmax(take(cache["phase_logits"]).float(), dim=-1),
    ), dim=-1)


def speed_targets(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    speed = torch.zeros(valid.shape, dtype=pose.dtype)
    speed[:, 1:] = torch.linalg.vector_norm(
        pose[:, 1:] - pose[:, :-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    return speed.clamp_max(2.0) * valid


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (
        torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    ).clamp_min(1e-7))


@torch.no_grad()
def evaluate(model, cache, target_speed, valid, risk, device):
    model.eval()
    predictions = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        output = model(
            profile_features(cache, indices).to(device),
            valid.index_select(0, indices).to(device),
        )
        predictions.append(output["speed"].float().cpu())
    predicted = torch.cat(predictions)
    correlations, danger_correlations = [], []
    peak_error, danger_peak_error = [], []
    absolute = (predicted - target_speed).abs()
    for item, mask in enumerate(valid):
        corr = correlation(predicted[item, mask], target_speed[item, mask])
        correlations.append(corr)
        peak = abs(int(predicted[item, mask].argmax()) - int(target_speed[item, mask].argmax()))
        peak_error.append(peak)
        if int(risk[item]) == 2:
            danger_correlations.append(corr)
            danger_peak_error.append(peak)
    mae = float(absolute[valid].mean())
    danger_mask = valid & (risk[:, None] == 2)
    danger_mae = float(absolute[danger_mask].mean())
    result = {
        "mae_mps": mae,
        "danger_mae_mps": danger_mae,
        "correlation_mean": float(np.mean(correlations)),
        "correlation_median": float(np.median(correlations)),
        "danger_correlation_mean": float(np.mean(danger_correlations)),
        "danger_correlation_median": float(np.median(danger_correlations)),
        "peak_error_frames_median": float(np.median(peak_error)),
        "danger_peak_error_frames_median": float(np.median(danger_peak_error)),
    }
    result["selection_score"] = (
        mae + danger_mae
        - 0.05 * result["correlation_mean"]
        - 0.08 * result["danger_correlation_mean"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71",
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
    train_speed = speed_targets(train_pose, train_valid)
    val_speed = speed_targets(val_pose, val_valid)
    input_dim = profile_features(train_cache, torch.arange(1)).shape[-1]
    model = MotionProfileHead(input_dim=input_dim).to(device)

    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(2.5, dtype=torch.double),
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
    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            features = profile_features(train_cache, indices).to(device)
            mask = train_valid.index_select(0, indices).to(device)
            target = train_speed.index_select(0, indices).to(device)
            danger = (train_risk.index_select(0, indices).to(device) == 2)
            features = features + 0.01 * torch.randn_like(features)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                output = model(features, mask)
                predicted = output["speed"]
                frame_weight = 1.0 + 2.0 * (target / 0.5).clamp_max(2.0)
                frame_weight = frame_weight * torch.where(
                    danger[:, None], 1.75, 1.0
                )
                frame_weight = frame_weight * mask
                regression = F.smooth_l1_loss(
                    torch.log1p(predicted), torch.log1p(target),
                    reduction="none", beta=0.10,
                )
                regression = (
                    regression * frame_weight
                ).sum() / frame_weight.sum().clamp_min(1.0)
                motion = F.binary_cross_entropy_with_logits(
                    output["motion_logits"], (target > 0.12).float(),
                    reduction="none",
                )
                motion = (motion * frame_weight).sum() / frame_weight.sum().clamp_min(1.0)
                acceleration = F.smooth_l1_loss(
                    predicted[:, 1:] - predicted[:, :-1],
                    target[:, 1:] - target[:, :-1],
                    reduction="none", beta=0.04,
                )
                acceleration_mask = mask[:, 1:] & mask[:, :-1]
                acceleration = acceleration[acceleration_mask].mean()
                loss = regression + 0.20 * motion + 0.15 * acceleration
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append((
                float(loss.detach()), float(regression.detach()),
                float(motion.detach()), float(acceleration.detach()),
            ))
        validation_result = evaluate(
            model, val_cache, val_speed, val_valid, val_risk, device
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
        print(json.dumps(row), flush=True)
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
        "run": "KP5-MOTION-PROFILE-EXP01",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
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
        "model": best["state"], "model_config": {"input_dim": input_dim},
        "selection": result["selection"],
        "feature_root": report_path(args.feature_root),
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
