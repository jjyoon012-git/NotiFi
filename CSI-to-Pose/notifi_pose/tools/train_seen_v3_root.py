"""Train and calibrate the contact-guided root stage of Seen V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..quality import QualityWeightedDataset, quality_summary
from ..seen_v2 import SeenReconstructionV2Net
from ..seen_v3 import ContactGuidedRootNet, contact_guided_root_loss
from ..trainer import evaluate, set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, pose_only, report_path
from .train_seen_v2 import load_motion, make_final_seen_backbone


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_v2(args, device: str) -> SeenReconstructionV2Net:
    backbone = make_final_seen_backbone(
        args.baseline_checkpoint, args.motion_checkpoint,
        args.pose_residual_checkpoint, args.root_residual_checkpoint,
        0.5, device,
    )
    motion = load_motion(args.motion_checkpoint, device)
    model = SeenReconstructionV2Net(backbone, motion, hidden=motion.hidden).to(device)
    checkpoint = torch.load(
        args.v2_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    calibration = checkpoint.get("calibration", {
        "rotation_strength": 0.10,
        "high_pose_strength": 0.0,
        "root_strength": 0.50,
    })
    model.set_calibration(
        rotation=calibration["rotation_strength"],
        high_pose=calibration["high_pose_strength"],
        root=calibration["root_strength"],
    )
    model.set_partial_finetune(False)
    model.eval()
    return model


def selection_score(metrics: dict) -> float:
    return (
        float(metrics["root_err"])
        + 0.35 * float(metrics["impact_mpjpe"])
        + 0.05 * float(metrics["mpjpe"])
    )


def measure(model, dataset, loader, criterion, device: str,
            batch_size: int) -> dict:
    return {
        **evaluate(model, loader, criterion, device),
        **evaluate_model(model, dataset, device, batch_size, 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--root-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v3_contact_root",
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
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train.sampler_weights(), len(train), replacement=True,
        generator=generator,
    )
    loaders = {
        "train": DataLoader(
            train, batch_size=args.batch_size,
            sampler=sampler, num_workers=0,
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
    model = ContactGuidedRootNet(load_v2(args, device)).to(device)
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

    model.set_root_strength(0.0)
    initial = measure(
        model, validation, loaders["val"], criterion,
        device, args.batch_size * 2,
    )
    model.set_root_strength(1.0)
    best_score = float("inf")
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            output = model(batch["csi"], batch["link_mask"])
            loss, parts = contact_guided_root_loss(output, batch)
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
        validation_metrics = measure(
            model, validation, loaders["val"], criterion,
            device, args.batch_size * 2,
        )
        score = selection_score(validation_metrics)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "selection_score": score,
            "validation": validation_metrics,
        })
        print(
            f"epoch={epoch:02d} loss={train_metrics['total']:.4f} "
            f"root={validation_metrics['root_err'] * 100:.2f}cm "
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
    candidates = []
    for strength in args.root_strengths:
        model.set_root_strength(strength)
        metrics = measure(
            model, validation, loaders["val"], criterion,
            device, args.batch_size * 2,
        )
        candidates.append({
            "root_strength": strength,
            "score": selection_score(metrics),
            "validation": metrics,
        })
    selected = min(candidates, key=lambda item: item["score"])
    model.set_root_strength(selected["root_strength"])
    test_metrics = measure(
        model, test, loaders["test"], criterion,
        device, args.batch_size * 2,
    )
    from .train_seen_v2 import evaluate_injury
    injury_metrics = evaluate_injury(model, loaders["test"], device)
    result = {
        "run": "seen_v3_contact_guided_root_single_split",
        "protocol": "single_split",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "best_epoch": checkpoint["epoch"],
        "baseline_validation": initial,
        "selected": {"root_strength": selected["root_strength"]},
        "selected_validation": selected["validation"],
        "test": test_metrics,
        "injury_test": injury_metrics,
        "shuffled_test": evaluate_model(
            model, ShuffledSignalDataset(test, args.seed),
            device, args.batch_size * 2, 5,
        ),
        "calibration_candidates": candidates,
        "quality": {
            "train": quality_summary(train),
            "validation": quality_summary(validation),
            "test": quality_summary(test),
        },
        "history": history,
        "checkpoint": report_path(checkpoint_path),
    }
    result_path = args.run_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calibrated_path = args.run_dir / "calibrated_model.pt"
    torch.save({
        "model": model.state_dict(),
        "root_strength": selected["root_strength"],
        "source_checkpoint": report_path(args.v2_checkpoint),
        "validation": selected["validation"],
        "test": test_metrics,
    }, calibrated_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
