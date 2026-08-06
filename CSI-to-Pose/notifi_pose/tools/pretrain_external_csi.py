"""Pretrain the shared amplitude-Doppler encoder on an external CSI cache."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..external_cache import ExternalCSICacheDataset
from ..external_pretraining import MultiSourceCSIPretrainer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pose_velocity_target(pose: torch.Tensor) -> torch.Tensor:
    """Body-scale-normalized root-relative native-keypoint velocity."""

    joints = pose.shape[-2]
    if joints == 17:
        root = 0.5 * (pose[:, :, 11] + pose[:, :, 12])
        shoulder = 0.5 * (pose[:, :, 5] + pose[:, :, 6])
    else:
        root = pose[:, :, 0]
        shoulder = pose[:, :, min(1, joints - 1)]
    scale = torch.linalg.vector_norm(shoulder - root, dim=-1)
    valid_scale = torch.where(scale > 1e-4, scale, torch.nan)
    body_scale = torch.nanmedian(valid_scale, dim=1).values
    body_scale = torch.nan_to_num(body_scale, nan=1.0).clamp_min(1e-4)
    relative = (pose - root[:, :, None]) / body_scale[:, None, None, None]
    velocity = torch.zeros_like(relative)
    velocity[:, 1:] = relative[:, 1:] - relative[:, :-1]
    return velocity.flatten(-2)


def _drop_links(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    output = mask.clone()
    for batch in range(len(output)):
        alive = torch.where(output[batch].any(0))[0]
        if len(alive) > 1 and torch.rand(()) < probability:
            dropped = alive[torch.randint(len(alive), ())]
            output[batch, :, dropped] = False
    return output


def _run_epoch(model, loader, optimizer, device, dataset_id: str,
               lambda_motion: float, link_dropout: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "class_loss": 0.0, "motion_loss": 0.0,
              "correct": 0.0, "samples": 0.0}
    for batch in loader:
        csi = batch["csi"].to(device)
        mask = batch["link_mask"].to(device)
        pose = batch["pose_native"].to(device)
        labels = batch["action_id"].to(device)
        if training:
            mask = _drop_links(mask, link_dropout)
            csi = csi * mask[..., None, None].to(csi.dtype)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(csi, mask, dataset_id)
            class_loss = F.cross_entropy(output["class_logits"], labels)
            target = _pose_velocity_target(pose)
            valid = output["frame_mask"]
            per_frame = F.smooth_l1_loss(
                output["motion_embedding"], target, reduction="none"
            ).mean(-1)
            motion_loss = (
                per_frame * valid.to(per_frame.dtype)
            ).sum() / valid.sum().clamp_min(1)
            loss = class_loss + lambda_motion * motion_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        count = len(labels)
        totals["loss"] += float(loss.detach()) * count
        totals["class_loss"] += float(class_loss.detach()) * count
        totals["motion_loss"] += float(motion_loss.detach()) * count
        totals["correct"] += float(
            (output["class_logits"].argmax(-1) == labels).sum()
        )
        totals["samples"] += count
    count = max(totals.pop("samples"), 1.0)
    return {
        **{key: value / count for key, value in totals.items() if key != "correct"},
        "accuracy": totals["correct"] / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-subjects", nargs="+", default=["S09", "S10"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-motion", type=float, default=2.0)
    parser.add_argument("--link-dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    _seed_everything(args.seed)
    caches = [ExternalCSICacheDataset(path) for path in args.cache]
    dataset_ids = {str(cache.manifest["dataset"]) for cache in caches}
    if len(dataset_ids) != 1:
        for cache in caches:
            cache.close()
        raise ValueError("all external caches must belong to the same dataset")
    pose_shapes = {tuple(cache.pose.shape[2:]) for cache in caches}
    if len(pose_shapes) != 1:
        for cache in caches:
            cache.close()
        raise ValueError("external caches have incompatible pose shapes")
    dataset = ConcatDataset(caches)
    dataset_id = dataset_ids.pop()
    subjects = np.concatenate([
        cache.metadata.subject.to_numpy() for cache in caches
    ])
    validation_subjects = set(args.val_subjects)
    is_validation = np.isin(subjects, list(validation_subjects))
    train_rows = np.flatnonzero(~is_validation).tolist()
    val_rows = np.flatnonzero(is_validation).tolist()
    if not train_rows or not val_rows:
        for cache in caches:
            cache.close()
        raise ValueError("subject-disjoint train/validation split is empty")

    train_loader = DataLoader(
        Subset(dataset, train_rows), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_rows), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    pose_shape = next(iter(pose_shapes))
    motion_dim = int(np.prod(pose_shape))
    classes = max(int(cache.metadata.action_id.max()) for cache in caches) + 1
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = MultiSourceCSIPretrainer(
        {dataset_id: classes}, hidden=args.hidden,
        temporal_layers=args.temporal_layers, heads=args.heads,
        motion_dim=motion_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = math.inf
    stale = 0
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(
            model, train_loader, optimizer, device, dataset_id,
            args.lambda_motion, args.link_dropout,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, None, device, dataset_id,
                args.lambda_motion, 0.0,
            )
        scheduler.step()
        score = val_metrics["loss"]
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={score:.4f} val_acc={val_metrics['accuracy']:.3f}",
            flush=True,
        )
        if score < best:
            best = score
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "shared": model.shared_checkpoint(),
                "args": vars(args),
                "split": {"train_rows": train_rows, "val_rows": val_rows},
                "validation": val_metrics,
            }, output / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    (output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    for cache in caches:
        cache.close()
    print(f"best validation loss={best:.6f} -> {output / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
