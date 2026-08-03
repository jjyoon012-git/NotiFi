"""Train a validation-gated seen-domain residual over the established pose baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..motion_first import MotionFirstEncoder, MotionResidualPoseNet
from ..trainer import _selection_score, evaluate, set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, pose_only, report_path
from .evaluate_sealed import make_model as make_pose_baseline


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_model(baseline_path: Path, motion_path: Path,
               device: str) -> MotionResidualPoseNet:
    baseline_checkpoint = torch.load(
        baseline_path, map_location=device, weights_only=False
    )
    baseline = make_pose_baseline(baseline_checkpoint, device)
    motion_checkpoint = torch.load(
        motion_path, map_location=device, weights_only=False
    )
    config = motion_checkpoint["config"]
    motion = MotionFirstEncoder(
        hidden=int(config["hidden"]),
        temporal_layers=int(config["temporal_layers"]),
        heads=int(config["heads"]),
    ).to(device)
    motion.load_state_dict(motion_checkpoint["model"])
    return MotionResidualPoseNet(
        baseline, motion, hidden=int(config["hidden"])
    ).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
    )
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_residual_pose_seen",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dropout = DropoutConfig(rf_augment=True)
    datasets = build_datasets(
        exp="single_split", baseline="sub", dropout=dropout, seed=args.seed
    )
    train = pose_only(datasets["train"])
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    train.train = True
    train.dropout = dropout
    loaders = {
        "train": DataLoader(
            train, batch_size=args.batch_size, shuffle=True, num_workers=0,
            pin_memory=device == "cuda",
        ),
        "val": DataLoader(
            validation, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        ),
    }
    model = make_model(args.baseline_checkpoint, args.motion_checkpoint, device)
    criterion = L.PoseLoss(
        lambda_root=1.0, lambda_bone=0.1,
        lambda_cls=0.0, lambda_risk=0.0,
        lambda_velocity=0.10, lambda_acceleration=0.0005,
        lambda_impact=0.20, lambda_displacement=0.10,
        motion_weight=3.0, device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    baseline_validation = evaluate(model, loaders["val"], criterion, device)
    best_score = _selection_score(baseline_validation)
    torch.save({
        "model": model.state_dict(), "epoch": 0,
        "validation": baseline_validation,
    }, checkpoint_path)
    print(
        f"epoch=00 val_mpjpe={baseline_validation['mpjpe'] * 100:.2f}cm "
        f"impact={baseline_validation['impact_mpjpe'] * 100:.2f}cm "
        f"score={best_score:.4f}"
    )
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch)
        model.train()
        total = examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(batch["csi"], batch["link_mask"])
                loss, _ = criterion(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["class_id"])
            total += float(loss.detach()) * count
            examples += count
        validation_metrics = evaluate(model, loaders["val"], criterion, device)
        score = _selection_score(validation_metrics)
        history.append({
            "epoch": epoch, "train_loss": total / max(examples, 1),
            "selection_score": score, "validation": validation_metrics,
        })
        print(
            f"epoch={epoch:02d} loss={total / max(examples, 1):.4f} "
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
    result = {
        "run": "motion_residual_pose_single_split",
        "protocol": "single_split",
        "best_epoch": checkpoint["epoch"],
        "baseline_validation": baseline_validation,
        "source_validation": checkpoint["validation"],
        "test": {
            **evaluate(model, loaders["test"], criterion, device),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
