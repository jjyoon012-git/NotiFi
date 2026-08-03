"""Measure data loading and GraphPoseNet train-step throughput."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from ..dataio.dataset import DropoutConfig, build_datasets
from ..losses import PoseLoss
from ..nets import GraphPoseNet
from ..trainer import fit_norm


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("none", "sub", "sub_z"), default="none")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    dataset = build_datasets(
        exp="single_split",
        dropout=DropoutConfig(p=0.25),
        baseline=args.baseline,
    )["train"]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    started = time.perf_counter()
    batches = []
    for index, batch in enumerate(loader):
        batches.append(batch)
        if index + 1 >= args.batches:
            break
    data_seconds = time.perf_counter() - started

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphPoseNet(hidden=args.hidden, n_blocks=3).to(device)
    synchronize()
    started = time.perf_counter()
    fit_norm(model, loader, device, max_batches=20)
    synchronize()
    norm_seconds = time.perf_counter() - started
    criterion = PoseLoss(
        dataset.class_counts(), dataset.risk_counts(),
        lambda_velocity=0.1, lambda_motion=0.05, motion_weight=3.0,
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    synchronize()
    started = time.perf_counter()
    for batch in batches:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            output = model(batch["csi"], batch["link_mask"])
            loss, _ = criterion(output, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    synchronize()
    train_seconds = time.perf_counter() - started

    print(
        f"baseline={args.baseline} batch={args.batch_size} steps={len(batches)} "
        f"norm={norm_seconds:.2f}s "
        f"data={data_seconds:.2f}s ({data_seconds / len(batches):.3f}s/step) "
        f"train={train_seconds:.2f}s ({train_seconds / len(batches):.3f}s/step)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
