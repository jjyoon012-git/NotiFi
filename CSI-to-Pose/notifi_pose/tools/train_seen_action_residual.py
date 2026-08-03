"""Train an action-conditioned, metric-aligned seen-domain pose residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..motion_first import ActionMotionResidualPoseNet, MotionFirstEncoder
from ..trainer import _selection_score, evaluate, set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, pose_only, report_path
from .evaluate_sealed import make_model as make_pose_baseline


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_model(baseline_path: Path, motion_path: Path,
               device: str) -> ActionMotionResidualPoseNet:
    baseline = make_pose_baseline(torch.load(
        baseline_path, map_location=device, weights_only=False
    ), device)
    checkpoint = torch.load(motion_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    motion = MotionFirstEncoder(
        hidden=int(config["hidden"]),
        temporal_layers=int(config["temporal_layers"]),
        heads=int(config["heads"]),
    ).to(device)
    motion.load_state_dict(checkpoint["model"])
    return ActionMotionResidualPoseNet(
        baseline, motion, hidden=int(config["hidden"])
    ).to(device)


def metric_aligned_loss(output: dict, batch: dict) -> torch.Tensor:
    valid = batch["valid"].bool()
    distance = torch.linalg.vector_norm(
        output["pose_rel"] - batch["pose_rel"], dim=-1
    )
    joint_weight = distance.new_ones(C.N_JOINTS)
    joint_weight[list(L.DISTAL_JOINTS)] = 1.5
    impact = L.impact_window(
        batch["pose_rel"], batch["root"], valid, batch["risk_id"]
    )
    frame_weight = 1.0 + 2.0 * impact.to(distance.dtype)
    weight = (
        valid[..., None].to(distance.dtype)
        * frame_weight[..., None]
        * joint_weight[None, None]
    )
    position = (distance * weight).sum() / weight.sum().clamp_min(1.0)

    lag = 5
    interval = valid[:, lag:] & valid[:, :-lag]
    predicted_delta = output["pose_rel"][:, lag:] - output["pose_rel"][:, :-lag]
    target_delta = batch["pose_rel"][:, lag:] - batch["pose_rel"][:, :-lag]
    displacement = torch.linalg.vector_norm(
        predicted_delta - target_delta, dim=-1
    )
    displacement_weight = interval[..., None].to(displacement.dtype)
    displacement = (
        (displacement * displacement_weight).sum()
        / displacement_weight.expand_as(displacement).sum().clamp_min(1.0)
    )
    bone = L.BoneLoss()(output["pose_rel"], batch["pose_rel"], valid)
    return position + 0.20 * displacement + 0.05 * bone


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
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen",
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
            validation, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        ),
    }
    model = make_model(args.baseline_checkpoint, args.motion_checkpoint, device)
    reporting_loss = L.PoseLoss(
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
    baseline_validation = evaluate(model, loaders["val"], reporting_loss, device)
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
        model.train()
        total = examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            output = model(batch["csi"], batch["link_mask"])
            loss = metric_aligned_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["class_id"])
            total += float(loss.detach()) * count
            examples += count
        validation_metrics = evaluate(model, loaders["val"], reporting_loss, device)
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
        "run": "action_motion_residual_single_split",
        "protocol": "single_split",
        "best_epoch": checkpoint["epoch"],
        "baseline_validation": baseline_validation,
        "source_validation": checkpoint["validation"],
        "test": {
            **evaluate(model, loaders["test"], reporting_loss, device),
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
