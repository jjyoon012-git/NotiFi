"""Train KP3 coarse/proposal pose plus CSI motion residual."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..geometry_phase import temporal_phase_contrastive
from ..geometry_phase_pose import GeometryPhaseCoarseResidual
from ..quality import protocol_audit_path, quality_summary
from ..trainer import set_seed
from .diagnose_observability import report_path
from .train_continuous_pose import normalized_target_latent
from .train_geometry_phase_pose import (
    ParameterEMA,
    build_model as build_source_model,
    warmup_cosine_factor,
)
from .train_kinetic_pose import (
    build_components,
    evaluate_strengths,
    kinetic_pose_loss,
    load_or_create_coarse_store,
    make_loaders,
    pose_selection_score,
)


def _parameter_groups(model: GeometryPhaseCoarseResidual) -> dict[str, dict]:
    groups = {
        "residual": {"named": {}},
        "phase": {"named": {}},
        "backbone": {"named": {}},
    }
    for name, parameter in model.named_parameters():
        if name.startswith("refiner."):
            groups["residual"]["named"][name] = parameter
        elif name.startswith("pose_model.phase_head."):
            groups["phase"]["named"][name] = parameter
        elif (
            name.startswith("pose_model.backbone.")
            and model.pose_model._is_trainable_key(
                name.removeprefix("pose_model.")
            )
            and "geometry_projection." not in name
        ):
            groups["backbone"]["named"][name] = parameter
    if any(not group["named"] for group in groups.values()):
        sizes = {key: len(value["named"]) for key, value in groups.items()}
        raise RuntimeError(f"empty KP3-CMR parameter group: {sizes}")
    return groups


def set_training_stage(model: GeometryPhaseCoarseResidual, groups: dict[str, dict],
                       stage: str) -> None:
    if stage not in {"residual_warmup", "joint_finetune"}:
        raise ValueError(f"unknown training stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    enabled = {"residual", "phase"}
    if stage == "joint_finetune":
        enabled.add("backbone")
    for group_name in enabled:
        for parameter in groups[group_name]["named"].values():
            parameter.requires_grad_(True)


def make_optimizer(model: GeometryPhaseCoarseResidual, args):
    groups = _parameter_groups(model)
    optimizer = torch.optim.AdamW([
        {
            "params": list(groups["residual"]["named"].values()),
            "lr": args.residual_learning_rate,
            "name": "residual",
        },
        {
            "params": list(groups["phase"]["named"].values()),
            "lr": args.phase_learning_rate,
            "name": "phase",
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


def _phase_loss(model, teacher, output: dict, batch: dict, args):
    target, token_mask = normalized_target_latent(
        teacher, batch, model.backbone
    )
    return temporal_phase_contrastive(
        output["phase_motion_latent"], target, token_mask,
        temperature=args.phase_temperature,
        positive_radius=args.phase_positive_radius,
        motion_quantile=args.phase_motion_quantile,
        minimum_queries=args.phase_minimum_queries,
        max_queries=args.phase_max_queries,
    )


def train_epoch(model, teacher, loader, optimizer, scaler, ema, device: str,
                args, coarse_store) -> dict:
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
            output = model(
                batch["csi"], batch["link_mask"],
                coarse_store.lookup(batch["row"].cpu(), device),
            )
            pose_loss, parts = kinetic_pose_loss(output, batch, args)
            phase_loss, phase_stats = _phase_loss(
                model, teacher, output, batch, args
            )
            loss = pose_loss + args.lambda_phase * phase_loss
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
def phase_audit(model, teacher, loader, device: str, args,
                coarse_store) -> dict:
    model.eval()
    weighted_loss = 0.0
    weighted_top1 = 0.0
    queries = 0.0
    for batch in loader:
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(
            device_batch["csi"], device_batch["link_mask"],
            coarse_store.lookup(batch["row"], device),
        )
        loss, stats = _phase_loss(
            model, teacher, output, device_batch, args
        )
        count = float(stats["phase_queries"])
        weighted_loss += float(loss) * count
        weighted_top1 += float(stats["phase_top1"]) * count
        queries += count
    return {
        "phase_contrastive": weighted_loss / max(queries, 1.0),
        "phase_top1": weighted_top1 / max(queries, 1.0),
        "phase_queries": queries,
    }


@torch.no_grad()
def evaluate_ema(model, ema, teacher, loader, device: str, args,
                 coarse_store) -> tuple[dict, dict]:
    live = copy.deepcopy(model.trainable_state_dict())
    model.load_trainable_state_dict(ema.model_state(model))
    metrics = evaluate_strengths(
        model, loader, [1.0], device, coarse_store
    )[1.0]
    phase = phase_audit(
        model, teacher, loader, device, args, coarse_store
    )
    model.load_trainable_state_dict(live)
    return metrics, phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--hierarchical-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dh_hierarchical_pose" / "best_model.pt",
    )
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=C.PROJECT_ROOT / "docs" / "results" / "v13s_pruned_pose_root_ensemble.json",
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v12w_robust_classification_ensemble" / "validation.json",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--residual-learning-rate", type=float, default=2e-4)
    parser.add_argument("--phase-learning-rate", type=float, default=5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.05)
    parser.add_argument("--minimum-score-improvement", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--lambda-phase", type=float, default=0.01)
    parser.add_argument("--phase-temperature", type=float, default=0.08)
    parser.add_argument("--phase-positive-radius", type=int, default=1)
    parser.add_argument("--phase-motion-quantile", type=float, default=0.60)
    parser.add_argument("--phase-minimum-queries", type=int, default=12)
    parser.add_argument("--phase-max-queries", type=int, default=96)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--danger-frame-boost", type=float, default=0.50)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--distal-joint-weight", type=float, default=2.5)
    parser.add_argument("--lambda-distal", type=float, default=0.40)
    parser.add_argument("--lambda-velocity", type=float, default=0.25)
    parser.add_argument("--lambda-aux-velocity", type=float, default=0.0)
    parser.add_argument("--lambda-acceleration", type=float, default=0.01)
    parser.add_argument("--lambda-bone-length", type=float, default=0.10)
    parser.add_argument("--lambda-bone-direction", type=float, default=0.04)
    parser.add_argument("--lambda-static", type=float, default=0.08)
    parser.add_argument("--lambda-endpoint", type=float, default=0.30)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument("--proposal-strength", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--experiment-name", default="KP3-PCR-EXP01")
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp3_pcr_seed17",
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
    source_model, teacher, source_architecture, motion_architecture = (
        build_source_model(source_checkpoint, device)
    )
    model = GeometryPhaseCoarseResidual(
        source_model,
        dropout=float(source_architecture.get("dropout", 0.08)),
        max_delta=args.max_delta,
        proposal_strength=args.proposal_strength,
    ).to(device)
    datasets, loaders = make_loaders(args, device)
    train, validation, test = datasets
    baseline, _, baseline_config = build_components(args, device)
    coarse_store = load_or_create_coarse_store(
        baseline, datasets, args.coarse_cache, device,
        args.batch_size, args.exp,
    )
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    optimizer, groups, ema_parameters = make_optimizer(model, args)
    set_training_stage(model, groups, "residual_warmup")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: warmup_cosine_factor(
            epoch, args.epochs, args.warmup_epochs,
            args.minimum_lr_ratio,
        ),
    )
    ema = ParameterEMA(ema_parameters, args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    initial_metrics = evaluate_strengths(
        model, loaders["val"], [1.0], device, coarse_store
    )[1.0]
    initial_phase = phase_audit(
        model, teacher, loaders["val"], device, args, coarse_store
    )
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
        "stage": "locked V13S/KP2-DH proposal initialization",
        "train": None,
        "validation_score": initial_score,
        "validation": initial_metrics,
        "validation_phase": initial_phase,
    }]
    print(json.dumps(history[0], ensure_ascii=False), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        stage = (
            "residual_warmup"
            if epoch <= args.freeze_epochs else "joint_finetune"
        )
        set_training_stage(model, groups, stage)
        train_metrics = train_epoch(
            model, teacher, loaders["train"], optimizer, scaler, ema,
            device, args, coarse_store,
        )
        validation_metrics, validation_phase = evaluate_ema(
            model, ema, teacher, loaders["val"], device, args,
            coarse_store,
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
        if score < best["score"] - args.minimum_score_improvement:
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
    validation_baseline = evaluate_strengths(
        model, loaders["val"], [0.0], device, coarse_store
    )[0.0]
    test_metrics = evaluate_strengths(
        model, loaders["test"], [1.0], device, coarse_store
    )[1.0]
    test_baseline = evaluate_strengths(
        model, loaders["test"], [0.0], device, coarse_store
    )[0.0]
    test_phase = phase_audit(
        model, teacher, loaders["test"], device, args, coarse_store
    )
    validation_gate_passed = best["epoch"] > 0
    critical_test_metrics = (
        "mpjpe_m",
        "danger_pose_mpjpe_m",
        "danger_distal_mpjpe_m",
        "danger_endpoint_mpjpe_m",
    )
    test_deployment_gate_passed = all(
        test_metrics[key] <= test_baseline[key]
        for key in critical_test_metrics
    )
    result = {
        "run": args.experiment_name,
        "model_family": "NotiFi-KP3",
        "candidate_version": "KP3-PCR",
        "promotion_status": (
            "deployment_gate_passed"
            if validation_gate_passed and test_deployment_gate_passed
            else "experimental"
        ),
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "inference_inputs": ["csi", "link_mask", "v13s_coarse_pose"],
        "config": {
            key: report_path(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
            "quality_audit": report_path(protocol_audit_path(args.exp)),
            "train_quality": quality_summary(train),
        },
        "architecture": {
            "initialization": "exact locked proposal via zero residual",
            "coarse_pose": "frozen V13S",
            "proposal_pose": (
                "V13S + locked_strength * (KP2-DH - V13S)"
            ),
            "proposal_strength": args.proposal_strength,
            "residual_features": "KP2-DH CSI temporal features",
            "phase_alignment": "auxiliary projection-only InfoNCE",
            "link_geometry_seen_status": (
                "implemented but frozen because fixed geometry is redundant "
                "with link IDs in the seen protocol"
            ),
            "max_residual_m": args.max_delta,
            "strength_selection": False,
            "fixed_residual_strength": 1.0,
            "motion_code_dim": motion_architecture["code_dim"],
        },
        "selection": {
            "epoch": best["epoch"],
            "score": best["score"],
            "locked_proposal_initial_score": initial_score,
            "minimum_score_improvement": args.minimum_score_improvement,
            "validation_gate_passed": validation_gate_passed,
            "test_used_for_selection": False,
            "test_deployment_gate_passed": test_deployment_gate_passed,
            "critical_test_metrics": list(critical_test_metrics),
        },
        "validation_baseline": validation_baseline,
        "validation": best["metrics"],
        "validation_phase": best["phase"],
        "test_baseline": test_baseline,
        "test": test_metrics,
        "test_phase": test_phase,
        "history": history,
        "baseline_configuration": baseline_config,
        "coarse_pose_cache": report_path(args.coarse_cache),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"],
        "protocol": args.exp,
        "trainable_model": best["state"],
        "architecture": result["architecture"],
        "source": {
            "hierarchical_checkpoint": report_path(
                args.hierarchical_checkpoint
            ),
            "coarse_pose_cache": report_path(args.coarse_cache),
        },
        "selection": result["selection"],
        "validation": result["validation"],
        "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
