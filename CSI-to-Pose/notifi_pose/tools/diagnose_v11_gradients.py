"""Measure task-gradient agreement in the shared P2 temporal representation."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from .evaluate_sealed import make_model


def _to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.flatten()
    second = second.flatten()
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-12:
        return float("nan")
    return float(torch.dot(first, second) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "v11_diagnostics" / "gradient_conflict.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(
        exp=args.exp, baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=args.seed,
    )
    loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = make_model(checkpoint, device)
    model.eval()

    names = ("pose", "root", "class", "risk", "velocity")
    cosine_rows = {f"{a}__{b}": [] for a, b in combinations(names, 2)}
    magnitudes = {name: [] for name in names}
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        batch = _to_device(raw_batch, device)
        output = model(batch["csi"], batch["link_mask"])
        feature = output["temporal_features"]
        valid = batch["valid"].bool()
        pair_valid = valid[:, 1:] & valid[:, :-1]
        losses = {
            "pose": L.smooth_l1_per_sample(
                output["pose_rel"], batch["pose_rel"], valid
            ).mean(),
            "root": L.smooth_l1_per_sample(
                output["root"], batch["root"], valid
            ).mean(),
            "class": F.cross_entropy(output["class_logits"], batch["class_id"]),
            "risk": F.cross_entropy(output["risk_logits"], batch["risk_id"]),
            "velocity": L.derivative_per_sample(
                output["pose_rel"], batch["pose_rel"], pair_valid, 1
            ).mean(),
        }
        gradients = {}
        for task_index, name in enumerate(names):
            gradients[name] = torch.autograd.grad(
                losses[name], feature,
                retain_graph=task_index < len(names) - 1,
                allow_unused=False,
            )[0].detach()
            magnitudes[name].append(float(gradients[name].norm()))
        for first, second in combinations(names, 2):
            cosine_rows[f"{first}__{second}"].append(
                _cosine(gradients[first], gradients[second])
            )

    result = {
        "checkpoint": str(args.checkpoint),
        "protocol": args.exp,
        "split": "train",
        "batches": min(args.batches, len(loader)),
        "representation": "shared temporal feature",
        "cosine": {
            key: {
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "negative_fraction": float(np.mean(np.asarray(values) < 0.0)),
            }
            for key, values in cosine_rows.items()
        },
        "gradient_norm": {
            key: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
            for key, values in magnitudes.items()
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
