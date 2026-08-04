"""Train the event-centric impact timing and body-part localizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import contract as C
from ..dataio.dataset import DropoutConfig, build_datasets
from ..impact_event import (
    ImpactEventLocalizer,
    impact_event_loss,
    physical_impact_targets,
)
from ..quality import QualityWeightedDataset, quality_summary
from ..seen_v3 import ContactGuidedRootNet
from ..trainer import set_seed
from .diagnose_observability import pose_only, report_path
from .train_seen_v2 import evaluate_injury
from .train_seen_v3_root import load_v2


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_v3(args, device: str) -> ContactGuidedRootNet:
    model = ContactGuidedRootNet(load_v2(args, device)).to(device)
    checkpoint = torch.load(
        args.v3_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.set_root_strength(float(checkpoint.get("root_strength", 0.5)))
    model.eval()
    return model


@torch.no_grad()
def evaluate_event(model, loader: DataLoader, device: str) -> dict:
    model.eval()
    timing_error = []
    joint_correct = joint_top3 = joint_time2 = joint_time5 = 0
    region_correct = region_top2 = region_time2 = region_time5 = 0
    event_hit2 = event_hit5 = 0
    target_joint_counts = None
    predicted_joint_counts = None
    target_region_counts = None
    predicted_region_counts = None
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["csi"], batch["link_mask"])
        target = physical_impact_targets(
            batch["pose_rel"], batch["root"], batch["valid"].bool(),
            batch["risk_id"],
        )
        selected = target["event_valid"]
        if not selected.any():
            continue
        event_prediction = output["event_logits_v8"][selected].argmax(1)
        joint_prediction = output["first_contact_logits"][selected].argmax(1)
        region_prediction = output["first_region_logits_v8"][selected].argmax(1)
        top3 = output["first_contact_logits"][selected].topk(3, dim=-1).indices
        target_frame = target["event_frame"][selected]
        target_joint = target["event_joint"][selected]
        target_region = target["event_region"][selected]
        error = (event_prediction - target_frame).abs()
        timing_error.extend(error.cpu().tolist())
        event_hit2 += int((error <= 2).sum())
        event_hit5 += int((error <= 5).sum())
        matches = joint_prediction.eq(target_joint)
        joint_correct += int(matches.sum())
        joint_top3 += int(top3.eq(target_joint[:, None]).any(-1).sum())
        joint_time2 += int((matches & error.le(2)).sum())
        joint_time5 += int((matches & error.le(5)).sum())
        region_matches = region_prediction.eq(target_region)
        region_correct += int(region_matches.sum())
        region_top2 += int(
            output["first_region_logits_v8"][selected].topk(2, dim=-1).indices
            .eq(target_region[:, None]).any(-1).sum()
        )
        region_time2 += int((region_matches & error.le(2)).sum())
        region_time5 += int((region_matches & error.le(5)).sum())
        size = output["first_contact_logits"].shape[-1]
        target_count = torch.bincount(target_joint, minlength=size).cpu().numpy()
        predicted_count = torch.bincount(
            joint_prediction, minlength=size
        ).cpu().numpy()
        region_size = output["first_region_logits_v8"].shape[-1]
        target_region_count = torch.bincount(
            target_region, minlength=region_size
        ).cpu().numpy()
        predicted_region_count = torch.bincount(
            region_prediction, minlength=region_size
        ).cpu().numpy()
        if target_joint_counts is None:
            target_joint_counts = target_count
            predicted_joint_counts = predicted_count
            target_region_counts = target_region_count
            predicted_region_counts = predicted_region_count
        else:
            target_joint_counts += target_count
            predicted_joint_counts += predicted_count
            target_region_counts += target_region_count
            predicted_region_counts += predicted_region_count
    count = len(timing_error)
    values = np.asarray(timing_error, dtype=np.float64)
    return {
        "event_trials": count,
        "timing_mae_frames": float(values.mean()) if count else float("nan"),
        "timing_median_frames": float(np.median(values)) if count else float("nan"),
        "timing_hit_at_2": event_hit2 / max(count, 1),
        "timing_hit_at_5": event_hit5 / max(count, 1),
        "joint_accuracy": joint_correct / max(count, 1),
        "joint_top3_accuracy": joint_top3 / max(count, 1),
        "joint_time_hit_at_2": joint_time2 / max(count, 1),
        "joint_time_hit_at_5": joint_time5 / max(count, 1),
        "region_accuracy": region_correct / max(count, 1),
        "region_top2_accuracy": region_top2 / max(count, 1),
        "region_time_hit_at_2": region_time2 / max(count, 1),
        "region_time_hit_at_5": region_time5 / max(count, 1),
        "target_joint_counts": (
            target_joint_counts.tolist() if target_joint_counts is not None else []
        ),
        "predicted_joint_counts": (
            predicted_joint_counts.tolist()
            if predicted_joint_counts is not None else []
        ),
        "target_region_counts": (
            target_region_counts.tolist() if target_region_counts is not None else []
        ),
        "predicted_region_counts": (
            predicted_region_counts.tolist()
            if predicted_region_counts is not None else []
        ),
    }


def selection_score(metrics: dict) -> float:
    return (
        float(metrics["timing_mae_frames"]) / 10.0
        + 1.0 - float(metrics["region_accuracy"])
        + 0.50 * (1.0 - float(metrics["region_time_hit_at_5"]))
        + 0.25 * (1.0 - float(metrics["joint_top3_accuracy"]))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v3_contact_root" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--v2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibrated_model.pt",
    )
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
    parser.add_argument(
        "--root-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--danger-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "impact_event_v8",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=args.seed,
    )
    train = QualityWeightedDataset(pose_only(datasets["train"]))
    validation = QualityWeightedDataset(pose_only(datasets["val"]))
    test = QualityWeightedDataset(pose_only(datasets["test"]))
    weights = train.sampler_weights()
    danger = torch.tensor(
        train.index.risk_id.to_numpy(dtype=np.int64) == 2,
        dtype=torch.bool,
    )
    weights = weights * torch.where(
        danger, torch.tensor(args.danger_weight, dtype=weights.dtype),
        torch.tensor(1.0, dtype=weights.dtype),
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True, generator=generator
    )
    loaders = {
        "train": DataLoader(
            train, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=device == "cuda",
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
    model = ImpactEventLocalizer(load_v3(args, device)).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    initial_validation = evaluate_event(model, loaders["val"], device)
    best_score = selection_score(initial_validation)
    torch.save({
        "model": model.state_dict(), "epoch": 0,
        "validation": initial_validation,
    }, checkpoint_path)
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            output = model(batch["csi"], batch["link_mask"])
            loss, parts = impact_event_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["class_id"])
            examples += count
            totals["total"] = totals.get("total", 0.0) + float(loss.detach()) * count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * count
        train_metrics = {
            key: value / max(examples, 1) for key, value in totals.items()
        }
        validation_metrics = evaluate_event(model, loaders["val"], device)
        score = selection_score(validation_metrics)
        history.append({
            "epoch": epoch, "train": train_metrics,
            "selection_score": score, "validation": validation_metrics,
        })
        print(
            f"epoch={epoch:02d} loss={train_metrics['total']:.4f} "
            f"timing={validation_metrics['timing_mae_frames']:.2f}f "
            f"region={validation_metrics['region_accuracy']:.3f} "
            f"region_time5={validation_metrics['region_time_hit_at_5']:.3f} "
            f"score={score:.4f}", flush=True,
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
                print(f"early stop at epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    selected_validation = evaluate_event(model, loaders["val"], device)
    test_metrics = evaluate_event(model, loaders["test"], device)
    result = {
        "run": "impact_event_v8_single_split",
        "protocol": "single_split",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "best_epoch": checkpoint["epoch"],
        "initial_validation": initial_validation,
        "selected_validation": selected_validation,
        "test": test_metrics,
        "legacy_injury_test": evaluate_injury(model, loaders["test"], device),
        "quality": {
            "train": quality_summary(train),
            "validation": quality_summary(validation),
            "test": quality_summary(test),
        },
        "config": {
            "danger_weight": args.danger_weight,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        "history": history,
        "checkpoint": report_path(checkpoint_path),
    }
    result_path = args.run_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
