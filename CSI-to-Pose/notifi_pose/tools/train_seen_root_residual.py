"""Train a coherent root-only residual on the provided seen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..motion_first import KeyframeRootResidualNet
from ..trainer import evaluate, set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, pose_only, report_path
from .train_seen_action_residual import make_model as make_pose_model


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def root_loss(output: dict, batch: dict) -> torch.Tensor:
    valid = batch["valid"].bool()
    distance = torch.linalg.vector_norm(output["root"] - batch["root"], dim=-1)
    impact = L.impact_window(
        batch["pose_rel"], batch["root"], valid, batch["risk_id"]
    )
    weight = valid.to(distance.dtype) * (
        1.0 + 1.5 * impact.to(distance.dtype)
    )
    position = (distance * weight).sum() / weight.sum().clamp_min(1.0)

    lag = 5
    interval = valid[:, lag:] & valid[:, :-lag]
    predicted_delta = output["root"][:, lag:] - output["root"][:, :-lag]
    target_delta = batch["root"][:, lag:] - batch["root"][:, :-lag]
    displacement = torch.linalg.vector_norm(
        predicted_delta - target_delta, dim=-1
    )
    displacement = (
        (displacement * interval).sum()
        / interval.sum().clamp_min(1)
    )
    return position + 0.20 * displacement


def selection_score(metrics: dict) -> float:
    return (
        float(metrics["root_err"])
        + 0.25 * float(metrics["impact_mpjpe"])
        + 0.05 * float(metrics["mpjpe"])
    )


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
    parser.add_argument(
        "--pose-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
    )
    parser.add_argument("--pose-residual-scale", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=args.seed,
    )
    train = pose_only(datasets["train"])
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    train.train = False
    loaders = {
        "train": DataLoader(
            train, batch_size=args.batch_size, shuffle=True, num_workers=0,
            pin_memory=device == "cuda",
        ),
        "val": DataLoader(
            validation, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=0,
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=0,
        ),
    }
    pose_model = make_pose_model(
        args.baseline_checkpoint, args.motion_checkpoint, device
    )
    pose_checkpoint = torch.load(
        args.pose_residual_checkpoint, map_location=device, weights_only=False
    )
    pose_model.load_state_dict(pose_checkpoint["model"])
    pose_model.set_residual_scale(args.pose_residual_scale)
    model = KeyframeRootResidualNet(
        pose_model, hidden=pose_model.motion.hidden
    ).to(device)
    criterion = L.PoseLoss(
        lambda_root=1.0, lambda_bone=0.1, lambda_cls=0.0, lambda_risk=0.0,
        lambda_velocity=0.1, lambda_impact=0.2,
        lambda_displacement=0.1, motion_weight=3.0, device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    initial = evaluate(model, loaders["val"], criterion, device)
    best_score = selection_score(initial)
    torch.save({"model": model.state_dict(), "epoch": 0, "validation": initial}, checkpoint_path)
    print(
        f"epoch=00 root={initial['root_err'] * 100:.2f}cm "
        f"impact={initial['impact_mpjpe'] * 100:.2f}cm score={best_score:.4f}"
    )
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            output = model(batch["csi"], batch["link_mask"])
            loss = root_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["class_id"])
            total += float(loss.detach()) * count
            examples += count
        metrics = evaluate(model, loaders["val"], criterion, device)
        score = selection_score(metrics)
        history.append({
            "epoch": epoch,
            "train_loss": total / max(examples, 1),
            "selection_score": score,
            "validation": metrics,
        })
        print(
            f"epoch={epoch:02d} loss={total / max(examples, 1):.4f} "
            f"root={metrics['root_err'] * 100:.2f}cm "
            f"impact={metrics['impact_mpjpe'] * 100:.2f}cm score={score:.4f}"
        )
        if score < best_score:
            best_score = score
            stale = 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "validation": metrics,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "run": "keyframe_root_residual_single_split",
        "protocol": "single_split",
        "pose_residual_scale": args.pose_residual_scale,
        "best_epoch": checkpoint["epoch"],
        "baseline_validation": initial,
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
