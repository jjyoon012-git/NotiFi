"""Train the seven-part seen-first reconstruction model on single_split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..motion_first import (
    ActionMotionResidualPoseNet,
    KeyframeRootResidualNet,
    MotionFirstEncoder,
)
from ..quality import QualityWeightedDataset, quality_summary
from ..seen_v2 import SeenReconstructionV2Net, injury_targets, weighted_seen_v2_loss
from ..trainer import _selection_score, evaluate, set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, pose_only, report_path
from .evaluate_sealed import make_model as make_graphformer


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_motion(path: Path, device: str) -> MotionFirstEncoder:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MotionFirstEncoder(
        hidden=int(config["hidden"]),
        temporal_layers=int(config["temporal_layers"]),
        heads=int(config["heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def make_final_seen_backbone(baseline_path: Path, motion_path: Path,
                             pose_residual_path: Path, root_residual_path: Path,
                             pose_scale: float, device: str) -> KeyframeRootResidualNet:
    baseline_checkpoint = torch.load(
        baseline_path, map_location=device, weights_only=False
    )
    baseline = make_graphformer(baseline_checkpoint, device)
    motion = load_motion(motion_path, device)
    pose_model = ActionMotionResidualPoseNet(
        baseline, motion, hidden=motion.hidden
    ).to(device)
    pose_checkpoint = torch.load(
        pose_residual_path, map_location=device, weights_only=False
    )
    pose_model.load_state_dict(pose_checkpoint["model"])
    pose_model.set_residual_scale(pose_scale)
    root_model = KeyframeRootResidualNet(
        pose_model, hidden=motion.hidden
    ).to(device)
    root_checkpoint = torch.load(
        root_residual_path, map_location=device, weights_only=False
    )
    root_model.load_state_dict(root_checkpoint["model"])
    return root_model


def make_optimizer(model: SeenReconstructionV2Net, learning_rate: float,
                   partial: bool) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("baseline.", "motion.")):
            backbone.append(parameter)
        else:
            head.append(parameter)
    groups = [{"params": head, "lr": learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": learning_rate * 0.10})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def train_epoch(model: SeenReconstructionV2Net, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: str,
                teacher: SeenReconstructionV2Net | None = None) -> dict:
    model.train()
    total = examples = 0
    aggregate: dict[str, float] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["csi"], batch["link_mask"])
        teacher_output = None
        if teacher is not None:
            with torch.no_grad():
                teacher_output = teacher(batch["csi"], batch["link_mask"])
        loss, parts = weighted_seen_v2_loss(output, batch, teacher_output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        count = len(batch["class_id"])
        total += float(loss.detach()) * count
        examples += count
        for key, value in parts.items():
            aggregate[key] = aggregate.get(key, 0.0) + value * count
    aggregate = {key: value / max(examples, 1) for key, value in aggregate.items()}
    aggregate["total"] = total / max(examples, 1)
    return aggregate


def binary_counts(logits: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> tuple[int, int, int]:
    predicted = logits >= 0.0
    while mask.dim() < predicted.dim():
        mask = mask.unsqueeze(-1)
    predicted = predicted[mask.expand_as(predicted)]
    target = target[mask.expand_as(target)].bool()
    return (
        int((predicted & target).sum()),
        int((predicted & ~target).sum()),
        int((~predicted & target).sum()),
    )


@torch.no_grad()
def evaluate_injury(model: SeenReconstructionV2Net, loader: DataLoader,
                    device: str) -> dict:
    model.eval()
    injury_counts = np.zeros(3, dtype=np.int64)
    foot_counts = np.zeros(3, dtype=np.int64)
    first_correct = first_count = 0
    speed_sum = speed_count = floor_sum = floor_count = 0
    pose_scale, root_scale = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["csi"], batch["link_mask"])
        target = injury_targets(
            batch["pose_rel"], batch["root"], batch["valid"].bool(),
            batch["risk_id"],
        )
        injury_counts += binary_counts(
            output["injury_contact_logits"], target["injury_contact"],
            batch["valid"].bool(),
        )
        feet, _ = L.contact_targets(
            batch["pose_rel"], batch["root"], batch["valid"].bool()
        )
        foot_counts += binary_counts(
            output["contact_logits"], feet, batch["valid"].bool()
        )
        selected = target["first_contact_valid"]
        if selected.any():
            first_correct += int((
                output["first_contact_logits"][selected].argmax(-1)
                == target["first_contact"][selected]
            ).sum())
            first_count += int(selected.sum())
        impact_mask = target["impact_mask"]
        if impact_mask.any():
            difference = (
                output["joint_impact_speed"] - target["joint_speed"]
            ).abs()
            speed_sum += float((difference * impact_mask[..., None]).sum())
            speed_count += int(impact_mask.sum()) * difference.shape[-1]
        floor_sum += float((
            output["floor_height"] - target["floor_height"]
        ).abs().sum())
        floor_count += len(batch["class_id"])
        valid = batch["valid"].bool()
        pose_scale.extend(output["pose_scale"][valid].detach().cpu().tolist())
        root_scale.extend(output["root_scale"][valid].detach().cpu().tolist())

    def f1(counts: np.ndarray) -> float:
        tp, fp, fn = (int(value) for value in counts)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "injury_contact_f1": f1(injury_counts),
        "foot_contact_f1": f1(foot_counts),
        "first_contact_accuracy": first_correct / max(first_count, 1),
        "first_contact_trials": first_count,
        "impact_joint_speed_mae_mps": speed_sum / max(speed_count, 1),
        "floor_height_mae_m": floor_sum / max(floor_count, 1),
        "pose_scale_mean": float(np.mean(pose_scale)),
        "root_scale_mean": float(np.mean(root_scale)),
    }


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
    parser.add_argument(
        "--root-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
    )
    parser.add_argument("--pose-residual-scale", type=float, default=0.5)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--quality-audit", type=Path,
        default=C.WORK_ROOT / "reports" / "motion_alignment_audit.csv",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=args.seed,
    )
    weighted = {
        split: QualityWeightedDataset(pose_only(dataset), args.quality_audit)
        for split, dataset in datasets.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weighted["train"].sampler_weights(), len(weighted["train"]),
        replacement=True, generator=generator,
    )
    loaders = {
        "train": DataLoader(
            weighted["train"], batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=device == "cuda",
        ),
        "val": DataLoader(
            weighted["val"], batch_size=args.batch_size * 2,
            shuffle=False, num_workers=0,
        ),
        "test": DataLoader(
            weighted["test"], batch_size=args.batch_size * 2,
            shuffle=False, num_workers=0,
        ),
    }

    final_backbone = make_final_seen_backbone(
        args.baseline_checkpoint, args.motion_checkpoint,
        args.pose_residual_checkpoint, args.root_residual_checkpoint,
        args.pose_residual_scale, device,
    )
    external_motion = load_motion(args.motion_checkpoint, device)
    model = SeenReconstructionV2Net(
        final_backbone, external_motion, hidden=external_motion.hidden
    ).to(device)
    reporting_loss = L.PoseLoss(
        lambda_root=1.0, lambda_bone=0.1, lambda_cls=0.0, lambda_risk=0.0,
        lambda_velocity=0.1, lambda_impact=0.2,
        lambda_displacement=0.1, motion_weight=3.0, device=device,
    ).to(device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    initial = evaluate(model, loaders["val"], reporting_loss, device)
    best_score = _selection_score(initial)
    torch.save({
        "model": model.state_dict(), "epoch": 0, "stage": "identity",
        "validation": initial,
    }, checkpoint_path)
    print(
        f"epoch=00 stage=identity val_mpjpe={initial['mpjpe'] * 100:.2f}cm "
        f"root={initial['root_err'] * 100:.2f}cm "
        f"impact={initial['impact_mpjpe'] * 100:.2f}cm score={best_score:.4f}"
    )

    history = []
    global_epoch = 0
    for stage, epochs, partial in (
        ("heads", args.head_epochs, False),
        ("partial_finetune", args.finetune_epochs, True),
    ):
        if epochs <= 0:
            continue
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        teacher = None
        if partial:
            teacher = copy.deepcopy(model).to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        model.set_partial_finetune(partial)
        optimizer = make_optimizer(model, args.learning_rate, partial)
        stale = 0
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_metrics = train_epoch(
                model, loaders["train"], optimizer, device, teacher
            )
            validation = evaluate(model, loaders["val"], reporting_loss, device)
            score = _selection_score(validation)
            history.append({
                "epoch": global_epoch, "stage": stage,
                "stage_epoch": stage_epoch, "train": train_metrics,
                "selection_score": score, "validation": validation,
            })
            print(
                f"epoch={global_epoch:02d} stage={stage} "
                f"loss={train_metrics['total']:.4f} "
                f"val_mpjpe={validation['mpjpe'] * 100:.2f}cm "
                f"root={validation['root_err'] * 100:.2f}cm "
                f"impact={validation['impact_mpjpe'] * 100:.2f}cm "
                f"score={score:.4f}"
            )
            if score < best_score:
                best_score = score
                stale = 0
                torch.save({
                    "model": model.state_dict(), "epoch": global_epoch,
                    "stage": stage, "validation": validation,
                }, checkpoint_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop stage={stage} at stage_epoch={stage_epoch}")
                    break
        del teacher

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.set_partial_finetune(False)
    test_metrics = {
        **evaluate(model, loaders["test"], reporting_loss, device),
        **evaluate_model(
            model, weighted["test"], device, args.batch_size * 2, 5
        ),
    }
    shuffled = ShuffledSignalDataset(weighted["test"], args.seed)
    result = {
        "run": "seen_reconstruction_v2_single_split",
        "protocol": "single_split",
        "best_epoch": checkpoint["epoch"],
        "best_stage": checkpoint["stage"],
        "identity_validation": initial,
        "selected_validation": checkpoint["validation"],
        "test": test_metrics,
        "injury_test": evaluate_injury(model, loaders["test"], device),
        "shuffled_test": evaluate_model(
            model, shuffled, device, args.batch_size * 2, 5
        ),
        "quality": {
            split: quality_summary(dataset) for split, dataset in weighted.items()
        },
        "history": history,
        "config": vars(args) | {"device": device},
        "checkpoint": report_path(checkpoint_path),
    }
    result["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result["config"].items()
    }
    result_path = args.run_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
