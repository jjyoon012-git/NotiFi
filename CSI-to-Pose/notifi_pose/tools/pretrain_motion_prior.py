"""Pretrain a GT-only kinematic motion autoencoder without held-out leakage."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import PoseDataset, build_datasets
from ..trainer import fit_bone_lengths, set_seed
from ..v3 import MotionPriorAutoencoder


@dataclass
class PriorConfig:
    exp: str
    fold: str | None
    hidden: int
    layers: int
    heads: int
    graph_blocks: int
    epochs: int
    batch_size: int
    lr: float
    seed: int


def pose_only(dataset) -> PoseDataset:
    keep = dataset.index.task.to_numpy() == C.TASK_POSE
    valid_any = np.asarray(dataset.cache.arrays["valid"][dataset.rows]).any(1)
    keep &= valid_any
    return PoseDataset(
        dataset.rows[np.flatnonzero(keep)], dataset.cache, dataset.link_ok,
        train=False, seed=dataset.seed, baseline=dataset.baseline,
    )


def move(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prior_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid"]
    pair = valid[:, 1:] & valid[:, :-1]
    pose = L.masked_smooth_l1(output["pose_rel"], batch["pose_rel"], valid)
    velocity = L.velocity_loss(output["pose_rel"], batch["pose_rel"], pair)
    acceleration = L.derivative_per_sample(
        output["pose_rel"], batch["pose_rel"], pair, 2
    ).mean()
    total = pose + 0.10 * velocity + 0.02 * acceleration
    return total, {
        "total": float(total.detach()), "pose": float(pose.detach()),
        "velocity": float(velocity.detach()),
        "acceleration": float(acceleration.detach()),
    }


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    total = count = 0
    mpjpe = 0.0
    for batch in loader:
        batch = move(batch, device)
        output = model(batch["pose_rel"], batch["valid"])
        loss, _ = prior_loss(output, batch)
        size = len(batch["pose_rel"])
        total += float(loss) * size
        mpjpe += L.mpjpe(output["pose_rel"], batch["pose_rel"], batch["valid"]) * size
        count += size
    return {"loss": total / max(count, 1), "mpjpe": mpjpe / max(count, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp", choices=("single_split", "yja_holdout", "loso"),
        default="yja_holdout",
    )
    parser.add_argument("--fold", default=None)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--graph-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.exp == "loso" and not args.fold:
        parser.error("--fold is required for --exp loso")

    set_seed(args.seed)
    selected = build_datasets(exp=args.exp, fold=args.fold, baseline="none")
    train_data = pose_only(selected["train"])
    val_data = pose_only(selected["val"])
    loaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_data, batch_size=args.batch_size, shuffle=False),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MotionPriorAutoencoder(
        args.hidden, args.layers, args.heads, args.graph_blocks
    ).to(device)
    fit_bone_lengths(model, train_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    tag = args.tag or f"motion_prior_{args.exp}{'_' + args.fold if args.fold else ''}"
    output_dir = C.WORK_ROOT / "priors" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = count = 0
        for batch in loaders["train"]:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(batch["pose_rel"], batch["valid"])
                loss, _ = prior_loss(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(batch["pose_rel"])
            count += len(batch["pose_rel"])
        scheduler.step()
        validation = evaluate(model, loaders["val"], device)
        history.append({
            "epoch": epoch, "train_loss": running / max(count, 1), **validation,
        })
        if validation["mpjpe"] < best:
            best, best_epoch = validation["mpjpe"], epoch
            torch.save({
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "cfg": vars(args), "best_epoch": epoch,
                "val_mpjpe": validation["mpjpe"],
            }, output_dir / "motion_prior.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"ep {epoch:3d} train={history[-1]['train_loss']:.4f} "
                f"val_mpjpe={validation['mpjpe'] * 100:.2f}cm "
                f"best={best * 100:.2f}cm"
            )

    config = PriorConfig(
        args.exp, args.fold, args.hidden, args.layers, args.heads,
        args.graph_blocks, args.epochs, args.batch_size, args.lr, args.seed,
    )
    (output_dir / "result.json").write_text(json.dumps({
        "config": asdict(config), "best_epoch": best_epoch,
        "best_val_mpjpe": best, "history": history,
    }, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
