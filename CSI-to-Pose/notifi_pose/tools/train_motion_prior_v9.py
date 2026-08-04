"""Train the V9C temporal denoising prior on train-split GT motions only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import build_datasets
from ..motion_prior_v9 import TemporalMotionDenoiser, corrupt_motion
from ..trainer import set_seed
from .diagnose_observability import pose_only, report_path


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def denoising_loss(output: dict, target: torch.Tensor,
                   valid: torch.Tensor,
                   risk: torch.Tensor) -> tuple[torch.Tensor, dict]:
    pose_element = F.smooth_l1_loss(
        output["pose_rel"], target, beta=0.05, reduction="none"
    ).mean((-1, -2))
    pose_per_sample = (
        pose_element * valid
    ).sum(1) / valid.sum(1).clamp_min(1)
    pose = pose_per_sample.mean()
    lag = 5
    interval = valid[:, lag:] & valid[:, :-lag]
    predicted_velocity = (
        output["pose_rel"][:, lag:] - output["pose_rel"][:, :-lag]
    ) * (C.TARGET_FPS / lag)
    target_velocity = (
        target[:, lag:] - target[:, :-lag]
    ) * (C.TARGET_FPS / lag)
    velocity_element = F.smooth_l1_loss(
        predicted_velocity, target_velocity, beta=0.20, reduction="none"
    ).mean((-1, -2))
    velocity = (
        velocity_element * interval
    ).sum(1) / interval.sum(1).clamp_min(1)
    bone = L.BoneLoss().to(target.device).per_sample(
        output["pose_rel"], target, valid
    )
    endpoint_error = torch.linalg.vector_norm(
        output["pose_rel"][:, -15:] - target[:, -15:], dim=-1
    ).mean(-1)
    endpoint_valid = valid[:, -15:]
    endpoint = (
        endpoint_error * endpoint_valid
    ).sum(1) / endpoint_valid.sum(1).clamp_min(1)
    per_sample = (
        pose_per_sample + 0.15 * velocity + 0.05 * bone + 0.10 * endpoint
    )
    weight = torch.where(
        risk.eq(2), per_sample.new_tensor(2.0), per_sample.new_tensor(1.0)
    )
    total = (per_sample * weight).sum() / weight.sum().clamp_min(1)
    return total, {
        "total": float(total.detach()),
        "pose": float(pose.detach()),
        "velocity": float(velocity.mean().detach()),
        "bone": float(bone.mean().detach()),
        "endpoint": float(endpoint.mean().detach()),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str,
             corrupted: bool, seed: int) -> dict:
    model.eval()
    torch.manual_seed(seed)
    total_error = total_source_error = total_loss = count = 0.0
    correction = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        pose = batch["pose_rel"]
        valid = batch["valid"].bool()
        if corrupted:
            source, observed = corrupt_motion(pose, valid)
        else:
            source = pose
            observed = valid[..., None].expand(-1, -1, C.N_JOINTS)
        output = model(source, valid, observed)
        loss, _ = denoising_loss(output, pose, valid, batch["risk_id"])
        error = torch.linalg.vector_norm(
            output["pose_rel"] - pose, dim=-1
        )
        error = (error * valid[..., None]).sum() / (
            valid.sum() * C.N_JOINTS
        ).clamp_min(1)
        source_error = torch.linalg.vector_norm(source - pose, dim=-1)
        source_error = (source_error * valid[..., None]).sum() / (
            valid.sum() * C.N_JOINTS
        ).clamp_min(1)
        size = len(pose)
        total_loss += float(loss) * size
        total_error += float(error) * size
        total_source_error += float(source_error) * size
        correction += float(torch.linalg.vector_norm(
            output["pose_correction"], dim=-1
        )[valid].mean()) * size
        count += size
    return {
        "loss": total_loss / max(count, 1),
        "mpjpe_m": total_error / max(count, 1),
        "source_mpjpe_m": total_source_error / max(count, 1),
        "correction_m": correction / max(count, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "priors" / "temporal_denoiser_v9",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp="single_split", baseline="none", seed=args.seed)
    train = pose_only(datasets["train"])
    validation = pose_only(datasets["val"])
    labels = train.index.class_id.to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=C.N_CLASSES)
    weights = 1.0 / np.sqrt(np.maximum(counts[labels], 1))
    weights *= np.where(train.index.risk_id.to_numpy() == 2, 3.0, 1.0)
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loaders = {
        "train": DataLoader(
            train, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=device == "cuda",
        ),
        "val": DataLoader(
            validation, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        ),
    }
    model = TemporalMotionDenoiser().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    best = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            pose = batch["pose_rel"]
            valid = batch["valid"].bool()
            corrupted, observed = corrupt_motion(pose, valid)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(corrupted, valid, observed)
                loss, parts = denoising_loss(
                    output, pose, valid, batch["risk_id"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            size = len(pose)
            examples += size
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * size
        train_metrics = {
            key: value / max(examples, 1) for key, value in totals.items()
        }
        noisy = evaluate(model, loaders["val"], device, True, args.seed + epoch)
        clean = evaluate(model, loaders["val"], device, False, args.seed)
        score = noisy["mpjpe_m"] + 0.50 * clean["mpjpe_m"]
        history.append({
            "epoch": epoch, "train": train_metrics,
            "score": score, "validation_noisy": noisy,
            "validation_clean": clean,
        })
        print(
            f"epoch={epoch:02d} loss={train_metrics['total']:.4f} "
            f"noisy={noisy['mpjpe_m'] * 100:.2f}cm "
            f"clean={clean['mpjpe_m'] * 100:.2f}cm score={score:.5f}"
        )
        if score < best:
            best = score
            stale = 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "validation_noisy": noisy, "validation_clean": clean,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = {
        "run": "temporal_denoising_motion_prior_v9",
        "protocol": "single_split_train_gt_only",
        "best_epoch": checkpoint["epoch"],
        "validation_noisy": checkpoint["validation_noisy"],
        "validation_clean": checkpoint["validation_clean"],
        "history": history,
        "checkpoint": report_path(checkpoint_path),
    }
    (args.run_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
