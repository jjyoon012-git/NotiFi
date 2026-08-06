"""Pretrain the GT-only KP2-B kinematic VQ motion tokenizer."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_tokens import (
    FactorizedMotionTokenizer,
    KinematicMotionTokenizer,
    pose_to_bones,
)
from ..trainer import set_seed
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import (
    DISTAL_JOINTS,
    _aggregate_rows,
    _pose_rows,
    _weighted_mean,
    pose_selection_score,
    relative_pose_speed,
)


def tokenizer_loss(output: dict, batch: dict, args) -> tuple[torch.Tensor, dict]:
    predicted = output["pose_rel"]
    target = batch["pose_rel"]
    valid = batch["valid"].bool()
    risk = batch["risk_id"]
    speed = relative_pose_speed(target, valid)
    weight = 1.0 + args.motion_weight * (speed / 0.50).clamp(0.0, 2.0)
    weight = weight * (1.0 + args.danger_boost * (risk == 2).float()[:, None])
    weight = weight * valid.to(predicted.dtype)

    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.03
    ).mean(-1)
    joint_weight = coordinate.new_ones(C.N_JOINTS)
    joint_weight[list(DISTAL_JOINTS)] = args.distal_weight
    position = _weighted_mean(coordinate, weight[..., None] * joint_weight)

    target_direction, _ = pose_to_bones(target)
    cosine = (
        output["bone_direction"] * target_direction
    ).sum(-1).clamp(-1.0, 1.0)
    bone_mask = weight[..., None].expand_as(cosine).clone()
    bone_mask[:, :, C.ROOT_JOINT] = 0.0
    direction = _weighted_mean(1.0 - cosine, bone_mask)

    pair = valid[:, 1:] & valid[:, :-1]
    predicted_velocity = (predicted[:, 1:] - predicted[:, :-1]) * C.TARGET_FPS
    target_velocity = (target[:, 1:] - target[:, :-1]) * C.TARGET_FPS
    velocity = _weighted_mean(
        F.smooth_l1_loss(
            predicted_velocity, target_velocity,
            reduction="none", beta=0.20,
        ).mean((-1, -2)),
        weight[:, 1:] * pair.to(weight.dtype),
    )
    total = (
        position
        + args.lambda_direction * direction
        + args.lambda_velocity * velocity
        + output["codebook_loss"]
        + output["commitment_loss"]
        + args.lambda_diversity * output["diversity_loss"]
    )
    return total, {
        "total": float(total.detach()),
        "position": float(position.detach()),
        "bone_direction": float(direction.detach()),
        "velocity": float(velocity.detach()),
        "codebook": float(output["codebook_loss"].detach()),
        "commitment": float(output["commitment_loss"].detach()),
        "diversity": float(output["diversity_loss"].detach()),
        "perplexity": float(output["codebook_perplexity"].detach()),
        "active_codes": float(output["active_codes"].detach()),
    }


def make_train_loader(train, args) -> DataLoader:
    labels = train.index.class_id.to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=C.N_CLASSES)
    weights = 1.0 / np.sqrt(np.maximum(counts[labels], 1))
    danger = train.index.risk_id.to_numpy(dtype=np.int64) == 2
    weights = weights * np.where(danger, args.danger_sample_weight, 1.0)
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return DataLoader(
        train, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def train_epoch(model, loader, optimizer, scaler, device: str, args) -> dict:
    model.train()
    totals: dict[str, list[float]] = {}
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            output = model(batch["pose_rel"], batch["valid"].bool())
            loss, parts = tokenizer_loss(output, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        for key, value in parts.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, device: str) -> dict:
    model.eval()
    rows = []
    token_ids = []
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        output = model(
            batch["pose_rel"].to(device), batch["valid"].to(device).bool()
        )
        rows.extend(_pose_rows(output["pose_rel"].cpu(), batch))
        selected = output["token_ids"][output["token_mask"]].cpu()
        if selected.ndim == 1:
            selected = selected[:, None]
        token_ids.append(selected)
    metrics = _aggregate_rows(rows)
    selected = (
        torch.cat(token_ids) if token_ids
        else torch.empty(0, model.quantizer_levels, dtype=torch.long)
    )
    active, perplexity = [], []
    for level in range(model.quantizer_levels):
        histogram = torch.bincount(
            selected[:, level], minlength=model.codes
        ).float()
        probability = histogram / histogram.sum().clamp_min(1.0)
        active.append(int((histogram > 0).sum()))
        perplexity.append(float(torch.exp(
            -(probability * probability.clamp_min(1e-12).log()).sum()
        )))
    metrics["active_codes_by_level"] = active
    metrics["codebook_perplexity_by_level"] = perplexity
    metrics["active_codes"] = min(active)
    metrics["codebook_perplexity"] = float(np.mean(perplexity))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--codes", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--quantizer-levels", type=int, default=1)
    parser.add_argument(
        "--continuous", action=argparse.BooleanOptionalAction, default=False,
        help="Bypass VQ to measure the continuous motion bottleneck.",
    )
    parser.add_argument(
        "--factorized", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--part-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--danger-boost", type=float, default=0.75)
    parser.add_argument("--danger-sample-weight", type=float, default=3.0)
    parser.add_argument("--distal-weight", type=float, default=1.5)
    parser.add_argument("--lambda-direction", type=float, default=0.25)
    parser.add_argument("--lambda-velocity", type=float, default=0.10)
    parser.add_argument("--lambda-diversity", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2b_motion_tokenizer",
    )
    args = parser.parse_args()

    if args.continuous and args.factorized:
        parser.error("--continuous and --factorized cannot be combined")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp=args.exp, baseline="none", seed=args.seed)
    train = pose_only(datasets["train"])
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    train_loader = make_train_loader(train, args)
    if args.factorized:
        model = FactorizedMotionTokenizer(
            args.hidden, args.code_dim, args.codes, args.dropout,
            args.commitment, args.downsample, args.part_hidden,
        ).to(device)
    else:
        model = KinematicMotionTokenizer(
            args.hidden, args.code_dim, args.codes, args.dropout, args.commitment,
            args.downsample, args.quantizer_levels, args.continuous,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = {"score": math.inf, "epoch": 0, "state": None, "metrics": None}
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, args
        )
        scheduler.step()
        validation_metrics = evaluate(
            model, validation, args.batch_size * 2, device
        )
        score = pose_selection_score(validation_metrics)
        record = {
            "epoch": epoch, "train": train_metrics,
            "validation_score": score, "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score < best["score"] - 1e-5:
            best = {
                "score": score, "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "metrics": validation_metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best["state"] is None:
        raise RuntimeError("tokenizer training produced no checkpoint")
    model.load_state_dict(best["state"])
    test_metrics = evaluate(model, test, args.batch_size * 2, device)
    gate = {
        "validation_mpjpe_below_3cm": best["metrics"]["mpjpe_m"] < 0.03,
        "validation_dynamic_below_4cm": best["metrics"]["dynamic_mpjpe_m"] < 0.04,
    }
    if not args.continuous:
        gate["at_least_32_active_codes"] = best["metrics"]["active_codes"] >= 32
    gate["passed"] = all(gate.values())
    result = {
        "run": "KP2-B-TOKENIZER-EXP01",
        "model_family": "NotiFi-KP2",
        "stage": "GT-only motion tokenizer",
        "protocol": args.exp,
        "train_only_fit": True,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "device": device,
        "config": vars(args) | {"run_dir": report_path(args.run_dir)},
        "dataset": {
            "train": train.describe(),
            "validation": validation.describe(),
            "test": test.describe(),
        },
        "selection": {
            "epoch": best["epoch"], "score": best["score"]
        },
        "validation": best["metrics"],
        "test": test_metrics,
        "stage2_gate": gate,
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "model": best["state"],
        "architecture": {
            "kind": (
                "factorized" if args.factorized
                else "continuous" if args.continuous
                else "whole_body"
            ),
            "hidden": args.hidden, "code_dim": args.code_dim,
            "codes": args.codes, "dropout": args.dropout,
            "commitment": args.commitment, "downsample": args.downsample,
            "quantizer_levels": args.quantizer_levels,
            "continuous": args.continuous,
            "part_hidden": args.part_hidden,
        },
        "selection": result["selection"],
        "validation": result["validation"],
        "test": result["test"],
        "stage2_gate": gate,
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
