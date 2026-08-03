"""Run the motion-first pose memorization gate or full yja holdout training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, PoseDataset, build_datasets
from ..motion_first import MotionFirstPoseNet, masked_temporal_average
from ..trainer import _selection_score, evaluate, set_seed
from .diagnose_observability import (
    ShuffledSignalDataset,
    dynamic_score_for_rows,
    evaluate_model,
    pose_only,
    report_path,
)
from .train_motion_first import compute_loss as motion_loss


def make_model(checkpoint_path: Path, device: str) -> tuple[MotionFirstPoseNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MotionFirstPoseNet(
        hidden=int(config["hidden"]),
        temporal_layers=int(config["temporal_layers"]),
        heads=int(config["heads"]),
    ).to(device)
    model.load_motion_backbone(checkpoint["model"])
    return model, checkpoint


def pose_loss(train: PoseDataset, device: str) -> L.PoseLoss:
    return L.PoseLoss(
        class_counts=train.class_counts(), risk_counts=train.risk_counts(),
        lambda_root=1.0, lambda_bone=0.1,
        lambda_cls=0.05, lambda_risk=0.05,
        lambda_velocity=0.20, lambda_motion=0.15,
        lambda_acceleration=0.0005, lambda_impact=0.20,
        lambda_coarse=0.15, lambda_displacement=0.25,
        lambda_phase=0.10, motion_weight=3.0, device=device,
    ).to(device)


def velocity_head_loss(output: dict, batch: dict) -> torch.Tensor:
    target = (batch["pose_rel"][:, 1:] - batch["pose_rel"][:, :-1]) * C.TARGET_FPS
    predicted = output["pose_velocity"][:, 1:]
    valid = batch["valid"][:, 1:] & batch["valid"][:, :-1]
    target = masked_temporal_average(target, valid, width=5)
    element = torch.nn.functional.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.20
    )
    selected = element[valid]
    return selected.mean() if selected.numel() else element.new_zeros(())


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def overfit_gate(model: MotionFirstPoseNet, source: PoseDataset, criterion: L.PoseLoss,
                 device: str, trials: int, steps: int, batch_size: int,
                 output_path: Path) -> dict:
    score = dynamic_score_for_rows(source)
    selected = np.argsort(-score)[:trials]
    subset = PoseDataset(
        source.rows[selected], source.cache, source.link_ok,
        train=False, seed=source.seed, baseline=source.baseline,
    )
    loader = DataLoader(
        subset, batch_size=min(batch_size, trials), shuffle=True, num_workers=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    iterator = iter(loader)
    model.train()
    last_loss = math.nan
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        output = model(batch["csi"], batch["link_mask"])
        primary, _ = criterion(output, batch)
        auxiliary, _ = motion_loss(output, batch)
        velocity_loss = velocity_head_loss(output, batch)
        loss = primary + 0.10 * auxiliary + 0.30 * velocity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach())
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"overfit trials={trials} step={step:04d} loss={last_loss:.4f}")
    raw_metrics = evaluate_model(
        model, subset, device, batch_size=min(batch_size, trials), smooth_window=1
    )
    metrics = evaluate_model(
        model, subset, device, batch_size=min(batch_size, trials), smooth_window=5
    )
    result = {
        "run": "EXP-004A4_keyframe_trajectory_overfit",
        "trials": trials,
        "steps": steps,
        "trial_ids": subset.index.trial_id.tolist(),
        "last_train_loss": last_loss,
        "velocity_mix": 0.0,
        "raw_metrics": raw_metrics,
        "metrics": metrics,
        "gate": {
            "mpjpe_lt_3cm": metrics["mpjpe_m"] < 0.03,
            "speed_ratio_0.9_to_1.1": 0.9 <= metrics["pose_speed_ratio"] <= 1.1,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def full_training(model: MotionFirstPoseNet, train: PoseDataset,
                  validation: PoseDataset, test: PoseDataset,
                  criterion: L.PoseLoss, device: str, args) -> dict:
    loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device == "cuda",
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.learning_rate * 0.15},
        {"params": list(model.pose_decoder.parameters())
         + list(model.pose_velocity_head.parameters())
         + [model.velocity_mix_logit]
         + list(model.root_decoder.parameters()), "lr": args.learning_rate},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best_score = float("inf")
    stale = 0
    history = []
    checkpoint_path = args.run_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch)
        model.set_backbone_trainable(epoch > args.freeze_epochs)
        model.train()
        total_loss = examples = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(batch["csi"], batch["link_mask"])
                primary, _ = criterion(output, batch)
                auxiliary, _ = motion_loss(output, batch)
                direct_velocity = velocity_head_loss(output, batch)
                loss = (
                    primary + args.motion_preserve_weight * auxiliary
                    + args.velocity_head_weight * direct_velocity
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["class_id"])
            total_loss += float(loss.detach()) * count
            examples += count
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        score = _selection_score(validation_metrics)
        history.append({
            "epoch": epoch, "train_loss": total_loss / max(examples, 1),
            "selection_score": score, "validation": validation_metrics,
        })
        print(
            f"epoch={epoch:02d} loss={total_loss / max(examples, 1):.4f} "
            f"val_mpjpe={validation_metrics['mpjpe'] * 100:.2f}cm "
            f"impact={validation_metrics['impact_mpjpe'] * 100:.2f}cm "
            f"score={score:.4f}"
        )
        if score < best_score:
            best_score = score
            stale = 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "validation": validation_metrics,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loader = DataLoader(
        test, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )
    result = {
        "run": f"motion_first_pose_{args.exp}",
        "protocol": args.exp,
        "best_epoch": checkpoint["epoch"],
        "source_validation": checkpoint["validation"],
        "test": {
            **evaluate(model, test_loader, criterion, device),
            **evaluate_model(model, test, device, args.batch_size * 2, 5),
        },
        "shuffled_test": evaluate_model(
            model, ShuffledSignalDataset(test, args.seed),
            device, args.batch_size * 2, 5,
        ),
        "checkpoint": report_path(checkpoint_path),
        "history": history,
    }
    result_path = args.run_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overfit", "full"), default="overfit")
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_yja_e02" / "best_model.pt",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--motion-preserve-weight", type=float, default=0.15)
    parser.add_argument("--velocity-head-weight", type=float, default=0.30)
    parser.add_argument("--baseline", default="sub", choices=("none", "sub", "sub_z"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exp", default="yja_holdout", choices=("single_split", "yja_holdout")
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_pose_yja_e02",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dropout = DropoutConfig(rf_augment=True)
    datasets = build_datasets(
        exp=args.exp, baseline=args.baseline, dropout=dropout, seed=args.seed
    )
    train = pose_only(datasets["train"])
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    train.train = args.mode == "full"
    train.dropout = dropout
    model, _ = make_model(args.motion_checkpoint, device)
    criterion = pose_loss(train, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "overfit":
        overfit_gate(
            model, train, criterion, device, args.trials, args.steps,
            args.batch_size, args.run_dir / "overfit_results.json",
        )
    else:
        full_training(model, train, validation, test, criterion, device, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
