"""Train KP3-GPA with link geometry and token-level phase alignment."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import contract as C
from ..continuous_pose import CSILatentPoseRegressor
from ..dataio.dataset import build_datasets
from ..geometry_phase import temporal_phase_contrastive
from ..geometry_phase_pose import GeometryPhasePoseRegressor
from ..hierarchical_pose import HierarchicalCSIPoseRegressor
from ..trainer import set_seed
from .diagnose_observability import report_path
from .evaluate_sealed import make_model
from .train_continuous_pose import (
    load_teacher,
    make_loaders,
    normalized_target_latent,
)
from .train_hierarchical_pose import evaluate, hierarchy_loss
from .train_kinetic_pose import pose_selection_score


class ParameterEMA:
    """EMA only for trainable adapters, avoiding frozen backbone duplication."""

    def __init__(self, named_parameters: dict[str, torch.nn.Parameter],
                 decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.names = tuple(named_parameters)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in named_parameters.items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach(), 1.0 - self.decay)

    def model_state(self, model: HierarchicalCSIPoseRegressor) -> dict:
        state = model.trainable_state_dict()
        for name, average in self.shadow.items():
            state[name] = average.detach().cpu().clone()
        return state


def _trainable_groups(model: HierarchicalCSIPoseRegressor) -> dict[str, dict]:
    hierarchy_prefixes = (
        "torso_direction_head.", "limb_direction_head.",
        "endpoint_head.", "velocity_head.",
    )
    groups = {
        "geometry": {"named": {}},
        "phase": {"named": {}},
        "hierarchy": {"named": {}},
        "backbone": {"named": {}},
    }
    for name, parameter in model.named_parameters():
        if "geometry_projection." in name:
            groups["geometry"]["named"][name] = parameter
        elif name.startswith("phase_head."):
            groups["phase"]["named"][name] = parameter
        elif name.startswith(hierarchy_prefixes):
            groups["hierarchy"]["named"][name] = parameter
        elif parameter.requires_grad and model._is_trainable_key(name):
            groups["backbone"]["named"][name] = parameter
    if any(not group["named"] for group in groups.values()):
        sizes = {key: len(value["named"]) for key, value in groups.items()}
        raise RuntimeError(f"empty KP3 parameter group: {sizes}")
    return groups


def set_training_stage(groups: dict[str, dict], stage: str) -> None:
    if stage not in {"geometry_warmup", "joint_finetune"}:
        raise ValueError(f"unknown training stage: {stage}")
    for group_name, group in groups.items():
        enabled = (
            stage == "joint_finetune"
            or group_name in {"geometry", "phase"}
        )
        for parameter in group["named"].values():
            parameter.requires_grad_(enabled)


def make_optimizer(model: HierarchicalCSIPoseRegressor, args):
    groups = _trainable_groups(model)
    optimizer = torch.optim.AdamW([
        {
            "params": list(groups["geometry"]["named"].values()),
            "lr": args.geometry_learning_rate,
            "name": "geometry",
        },
        {
            "params": list(groups["phase"]["named"].values()),
            "lr": args.phase_learning_rate,
            "name": "phase",
        },
        {
            "params": list(groups["hierarchy"]["named"].values()),
            "lr": args.hierarchy_learning_rate,
            "name": "hierarchy",
        },
        {
            "params": list(groups["backbone"]["named"].values()),
            "lr": args.backbone_learning_rate,
            "name": "backbone",
        },
    ], weight_decay=args.weight_decay)
    named = {}
    for group in groups.values():
        named.update(group["named"])
    return optimizer, groups, named


def warmup_cosine_factor(epoch: int, epochs: int, warmup: int,
                         minimum_ratio: float) -> float:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if warmup < 0 or warmup >= epochs:
        raise ValueError("warmup must be in [0, epochs)")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum ratio must be in (0, 1]")
    if epoch < warmup:
        return float(epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def train_epoch(model, teacher, loader, optimizer, scaler, ema,
                device: str, args) -> dict:
    model.train()
    totals: dict[str, list[float]] = {}
    for step, batch in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            output = model(batch["csi"], batch["link_mask"])
            pose_loss, parts = hierarchy_loss(output, batch, args)
            target_latent, token_mask = normalized_target_latent(
                teacher, batch, model.backbone
            )
            latent_error = F.smooth_l1_loss(
                output["normalized_motion_latent"], target_latent,
                reduction="none", beta=0.10,
            ).mean(-1)
            token_weight = token_mask.to(latent_error.dtype)
            token_weight *= 1.0 + args.danger_boost * (
                batch["risk_id"] == 2
            ).to(latent_error.dtype)[:, None]
            latent_loss = (
                latent_error * token_weight
            ).sum() / token_weight.sum().clamp_min(1.0)
            phase_loss, phase_stats = temporal_phase_contrastive(
                output["phase_motion_latent"], target_latent,
                token_mask,
                temperature=args.phase_temperature,
                positive_radius=args.phase_positive_radius,
                motion_quantile=args.phase_motion_quantile,
                minimum_queries=args.phase_minimum_queries,
                max_queries=args.phase_max_queries,
            )
            loss = (
                pose_loss
                + args.lambda_latent * latent_loss
                + args.lambda_phase * phase_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        parts.update({
            "latent": float(latent_loss.detach()),
            "phase_contrastive": float(phase_loss.detach()),
            "phase_top1": float(phase_stats["phase_top1"]),
            "phase_queries": float(phase_stats["phase_queries"]),
            "total_with_phase": float(loss.detach()),
        })
        for key, value in parts.items():
            if math.isfinite(value):
                totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def phase_audit(model, teacher, loader, device: str, args) -> dict:
    model.eval()
    losses = []
    accuracies = []
    queries = 0.0
    for batch in loader:
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(device_batch["csi"], device_batch["link_mask"])
        target, mask = normalized_target_latent(
            teacher, device_batch, model.backbone
        )
        loss, stats = temporal_phase_contrastive(
            output["phase_motion_latent"], target, mask,
            temperature=args.phase_temperature,
            positive_radius=args.phase_positive_radius,
            motion_quantile=args.phase_motion_quantile,
            minimum_queries=args.phase_minimum_queries,
            max_queries=args.phase_max_queries,
        )
        count = float(stats["phase_queries"])
        if count:
            losses.append(float(loss) * count)
            accuracies.append(float(stats["phase_top1"]) * count)
            queries += count
    return {
        "phase_contrastive": sum(losses) / max(queries, 1.0),
        "phase_top1": sum(accuracies) / max(queries, 1.0),
        "phase_queries": queries,
    }


def evaluate_ema(model, ema, teacher, loader, device: str,
                 args) -> tuple[dict, dict]:
    live = copy.deepcopy(model.trainable_state_dict())
    model.load_trainable_state_dict(ema.model_state(model))
    metrics = evaluate(model, teacher, loader, device)
    phase = phase_audit(model, teacher, loader, device, args)
    model.load_trainable_state_dict(live)
    return metrics, phase


def build_model(checkpoint: dict, device: str):
    source = checkpoint["source"]
    p2_path = C.PROJECT_ROOT / source["p2_checkpoint"]
    motion_path = C.PROJECT_ROOT / source["motion_checkpoint"]
    kp2c_path = C.PROJECT_ROOT / source["kp2c_checkpoint"]
    teacher, motion_architecture = load_teacher(motion_path, device)
    kp2c = torch.load(kp2c_path, map_location="cpu", weights_only=False)
    p2 = torch.load(p2_path, map_location=device, weights_only=False)
    base_model = make_model(p2, device)
    architecture = kp2c["architecture"]
    backbone = CSILatentPoseRegressor(
        base_model, teacher.decoder,
        checkpoint["latent_mean"], checkpoint["latent_std"],
        checkpoint["bone_lengths"],
        hidden=int(architecture["hidden"]),
        code_dim=int(motion_architecture["code_dim"]),
        temporal_layers=int(architecture.get("temporal_layers", 2)),
        heads=int(architecture.get("heads", 4)),
        dropout=float(architecture.get("dropout", 0.08)),
    ).to(device)
    model = GeometryPhasePoseRegressor(
        backbone, dropout=float(architecture.get("dropout", 0.08))
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.initialize_phase_head_from_pose()
    return model, teacher, architecture, motion_architecture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--hierarchical-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dh_hierarchical_pose" / "best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--geometry-learning-rate", type=float, default=3e-5)
    parser.add_argument("--phase-learning-rate", type=float, default=5e-5)
    parser.add_argument("--hierarchy-learning-rate", type=float, default=5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--lambda-phase", type=float, default=0.01)
    parser.add_argument("--phase-temperature", type=float, default=0.08)
    parser.add_argument("--phase-positive-radius", type=int, default=1)
    parser.add_argument("--phase-motion-quantile", type=float, default=0.60)
    parser.add_argument("--phase-minimum-queries", type=int, default=12)
    parser.add_argument("--phase-max-queries", type=int, default=96)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lambda-torso-direction", type=float, default=0.15)
    parser.add_argument("--lambda-limb-direction", type=float, default=0.15)
    parser.add_argument("--lambda-endpoint", type=float, default=0.50)
    parser.add_argument("--lambda-aux-velocity", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.002)
    parser.add_argument("--lambda-alignment", type=float, default=0.0)
    parser.add_argument("--alignment-radius", type=int, default=3)
    parser.add_argument("--alignment-temperature", type=float, default=0.05)
    parser.add_argument("--alignment-lag-penalty", type=float, default=0.002)
    parser.add_argument("--lambda-danger-keyframe", type=float, default=0.0)
    parser.add_argument("--danger-keyframe-frames", type=int, default=45)
    parser.add_argument("--danger-keyframe-distal-scale", type=float, default=0.75)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--danger-boost", type=float, default=0.75)
    parser.add_argument("--danger-sample-weight", type=float, default=3.0)
    parser.add_argument("--distal-weight", type=float, default=1.5)
    parser.add_argument("--lambda-direction", type=float, default=0.10)
    parser.add_argument("--lambda-velocity", type=float, default=0.10)
    parser.add_argument("--lambda-diversity", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--experiment-name", default="KP3-GPA-EXP01")
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp3_gpa_seed17",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.freeze_epochs >= args.epochs:
        raise ValueError("freeze epochs must be smaller than total epochs")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_checkpoint = torch.load(
        args.hierarchical_checkpoint, map_location="cpu", weights_only=False
    )
    if source_checkpoint.get("protocol") != args.exp:
        raise RuntimeError("KP2-DH checkpoint protocol mismatch")
    model, teacher, source_architecture, motion_architecture = build_model(
        source_checkpoint, device
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    datasets_wrapped, loaders = make_loaders(datasets, args, device)
    train, validation, test = datasets_wrapped
    optimizer, groups, ema_parameters = make_optimizer(model, args)
    set_training_stage(groups, "geometry_warmup")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: warmup_cosine_factor(
            epoch, args.epochs, args.warmup_epochs, args.minimum_lr_ratio
        ),
    )
    ema = ParameterEMA(ema_parameters, args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    initial_metrics = evaluate(model, teacher, loaders["val"], device)
    initial_phase = phase_audit(model, teacher, loaders["val"], device, args)
    initial_score = pose_selection_score(initial_metrics)
    best = {
        "score": initial_score,
        "epoch": 0,
        "state": copy.deepcopy(model.trainable_state_dict()),
        "metrics": initial_metrics,
        "phase": initial_phase,
    }
    history = [{
        "epoch": 0,
        "stage": "KP2-DH initialization",
        "train": None,
        "learning_rates": {
            group["name"]: group["lr"] for group in optimizer.param_groups
        },
        "validation_score": initial_score,
        "validation": initial_metrics,
        "validation_phase": initial_phase,
    }]
    print(json.dumps(history[0], ensure_ascii=False), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        stage = (
            "geometry_warmup"
            if epoch <= args.freeze_epochs else "joint_finetune"
        )
        set_training_stage(groups, stage)
        train_metrics = train_epoch(
            model, teacher, loaders["train"], optimizer, scaler, ema,
            device, args,
        )
        validation_metrics, validation_phase = evaluate_ema(
            model, ema, teacher, loaders["val"], device, args
        )
        score = pose_selection_score(validation_metrics)
        record = {
            "epoch": epoch,
            "stage": stage,
            "train": train_metrics,
            "learning_rates": {
                group["name"]: group["lr"] for group in optimizer.param_groups
            },
            "validation_score": score,
            "validation": validation_metrics,
            "validation_phase": validation_phase,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score < best["score"] - 1e-5:
            best = {
                "score": score,
                "epoch": epoch,
                "state": copy.deepcopy(ema.model_state(model)),
                "metrics": validation_metrics,
                "phase": validation_phase,
            }
            stale = 0
        elif epoch > args.freeze_epochs:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    model.load_trainable_state_dict(best["state"])
    test_metrics = evaluate(model, teacher, loaders["test"], device)
    test_phase = phase_audit(model, teacher, loaders["test"], device, args)
    baseline_score = float(source_checkpoint["blend_selection"][
        "candidates"
    ]["0.0"]["score"])
    result = {
        "run": args.experiment_name,
        "model_family": "NotiFi-KP3",
        "candidate_version": "KP3-GPA",
        "promotion_status": (
            "standalone_gate_passed"
            if best["score"] < baseline_score else "experimental"
        ),
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "inference_inputs": ["csi", "link_mask"],
        "config": vars(args) | {
            "hierarchical_checkpoint": report_path(
                args.hierarchical_checkpoint
            ),
            "run_dir": report_path(args.run_dir),
        },
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
        },
        "architecture": {
            "initialization": "KP2-DH",
            "link_geometry": {
                "rx": "north",
                "tx1": "south",
                "tx2": "west",
                "tx3": "east",
                "distance_or_height_used": False,
                "representation": list(map(list, C.LINK_GEOMETRY)),
            },
            "phase_alignment": "within-trial token multi-positive InfoNCE",
            "positive_neighborhood_tokens": args.phase_positive_radius,
            "decoder": "frozen continuous kinematic decoder",
            "pose_heads": source_checkpoint["architecture"]["explicit_heads"],
            "staged_training": {
                "geometry_phase_warmup_epochs": args.freeze_epochs,
                "geometry_lr": args.geometry_learning_rate,
                "phase_lr": args.phase_learning_rate,
                "hierarchy_lr": args.hierarchy_learning_rate,
                "backbone_lr": args.backbone_learning_rate,
                "ema_decay": args.ema_decay,
            },
            "source_hidden": source_architecture["hidden"],
            "motion_code_dim": motion_architecture["code_dim"],
        },
        "selection": {
            "epoch": best["epoch"],
            "score": best["score"],
            "standalone_v13s_gate_score": baseline_score,
            "standalone_gate_passed": best["score"] < baseline_score,
        },
        "validation": best["metrics"],
        "validation_phase": best["phase"],
        "test": test_metrics,
        "test_phase": test_phase,
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"],
        "protocol": args.exp,
        "trainable_model": best["state"],
        "latent_mean": source_checkpoint["latent_mean"],
        "latent_std": source_checkpoint["latent_std"],
        "bone_lengths": source_checkpoint["bone_lengths"],
        "architecture": result["architecture"],
        "source": {
            "hierarchical_checkpoint": report_path(
                args.hierarchical_checkpoint
            ),
            **source_checkpoint["source"],
        },
        "selection": result["selection"],
        "validation": result["validation"],
        "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
