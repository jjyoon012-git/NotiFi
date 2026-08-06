"""Train KP2-DHG: CSI-conditioned frame/joint confidence over KP2-DH."""

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
from ..hierarchical_pose import (
    DISTAL_JOINTS,
    HierarchicalCSIPoseRegressor,
    JointConfidencePoseGate,
)
from ..trainer import set_seed
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .train_continuous_pose import load_teacher, make_loaders
from .train_hierarchical_pose import JointErrorAccumulator
from .train_kinetic_pose import (
    CoarsePoseStore,
    _aggregate_rows,
    _pose_rows,
    _weighted_mean,
    pose_selection_score,
    relative_pose_speed,
)


def load_hierarchical_model(path: Path, device: str
                            ) -> tuple[HierarchicalCSIPoseRegressor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_kp2c = C.PROJECT_ROOT / checkpoint["source"]["kp2c_checkpoint"]
    kp2c = torch.load(source_kp2c, map_location="cpu", weights_only=False)
    p2_path = C.PROJECT_ROOT / checkpoint["source"]["p2_checkpoint"]
    motion_path = C.PROJECT_ROOT / checkpoint["source"]["motion_checkpoint"]
    p2_checkpoint = torch.load(p2_path, map_location=device, weights_only=False)
    base_model = make_model(p2_checkpoint, device)
    teacher, motion_architecture = load_teacher(motion_path, device)
    kp2c_architecture = kp2c["architecture"]
    architecture = checkpoint.get("architecture", {})
    backbone = CSILatentPoseRegressor(
        base_model, teacher.decoder, checkpoint["latent_mean"],
        checkpoint["latent_std"], checkpoint["bone_lengths"],
        hidden=int(architecture.get("hidden", kp2c_architecture["hidden"])),
        code_dim=int(architecture.get(
            "code_dim", motion_architecture["code_dim"]
        )),
        temporal_layers=int(architecture.get(
            "temporal_layers", kp2c_architecture.get("temporal_layers", 2)
        )),
        heads=int(architecture.get(
            "heads", kp2c_architecture.get("heads", 4)
        )),
        dropout=float(kp2c_architecture.get("dropout", 0.08)),
    ).to(device)
    model = HierarchicalCSIPoseRegressor(
        backbone,
        direction_scale=float(architecture.get("direction_scale", 1.0)),
        endpoint_scale=float(architecture.get("endpoint_scale", 0.40)),
        dropout=float(architecture.get("dropout", 0.08)),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.eval()
    return model, checkpoint


def oracle_gate(output: dict, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = output["pose_delta"]
    target_delta = target - output["pose_coarse"]
    denominator = delta.square().sum(-1)
    strength = (
        (delta * target_delta).sum(-1) / denominator.clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    reliable = denominator > 1e-5
    return strength, reliable


def gate_loss(output: dict, batch: dict, args) -> tuple[torch.Tensor, dict]:
    predicted = output["pose_rel"]
    target = batch["pose_rel"]
    valid = batch["valid"].bool()
    risk = batch["risk_id"]
    quality = batch["quality_weight"].to(predicted.device)
    speed = relative_pose_speed(target, valid)
    frame_weight = 1.0 + args.motion_weight * (speed / 0.50).clamp(0.0, 2.0)
    frame_weight *= 1.0 + args.danger_boost * (risk == 2).float()[:, None]
    frame_weight *= quality[:, None] * valid.to(frame_weight.dtype)

    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.03
    ).mean(-1)
    joint_weight = coordinate.new_ones(C.N_JOINTS)
    joint_weight[list(DISTAL_JOINTS)] = args.distal_weight
    position = _weighted_mean(
        coordinate, frame_weight[..., None] * joint_weight
    )
    distal = _weighted_mean(
        coordinate[:, :, DISTAL_JOINTS], frame_weight[..., None]
    )

    velocity_terms = []
    for lag in (1, 3, 7):
        pair = valid[:, lag:] & valid[:, :-lag]
        predicted_velocity = (
            predicted[:, lag:] - predicted[:, :-lag]
        ) * (C.TARGET_FPS / lag)
        target_velocity = (
            target[:, lag:] - target[:, :-lag]
        ) * (C.TARGET_FPS / lag)
        error = F.smooth_l1_loss(
            predicted_velocity, target_velocity,
            reduction="none", beta=0.20,
        ).mean((-1, -2))
        velocity_terms.append(_weighted_mean(
            error, frame_weight[:, lag:] * pair.to(frame_weight.dtype)
        ))
    velocity = sum(velocity_terms) / len(velocity_terms)

    target_gate, reliable = oracle_gate(output, target)
    gate = output["joint_confidence_gate"]
    oracle_error = F.smooth_l1_loss(
        gate, target_gate, reduction="none", beta=0.05
    )
    oracle = _weighted_mean(
        oracle_error,
        frame_weight[..., None] * reliable.to(frame_weight.dtype),
    )
    pair = valid[:, 1:] & valid[:, :-1]
    temporal = _weighted_mean(
        (gate[:, 1:] - gate[:, :-1]).square(),
        pair[..., None].to(gate.dtype),
    )
    prior = _weighted_mean(
        (gate - args.initial_strength).square(),
        valid[..., None].to(gate.dtype),
    )
    total = (
        position
        + args.lambda_distal * distal
        + args.lambda_velocity * velocity
        + args.lambda_oracle * oracle
        + args.lambda_temporal * temporal
        + args.lambda_prior * prior
    )
    return total, {
        "total": float(total.detach()),
        "position": float(position.detach()),
        "distal": float(distal.detach()),
        "velocity": float(velocity.detach()),
        "oracle_gate": float(oracle.detach()),
        "gate_temporal": float(temporal.detach()),
        "gate_prior": float(prior.detach()),
        "gate_mean": float(gate[valid].mean().detach()),
        "gate_distal_mean": float(gate[valid][:, DISTAL_JOINTS].mean().detach()),
    }


def train_epoch(model, loader, coarse_store, optimizer, scaler,
                device: str, args) -> dict:
    model.train()
    totals: dict[str, list[float]] = {}
    for batch in loader:
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        coarse = coarse_store.lookup(batch["row"], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            output = model(
                device_batch["csi"], device_batch["link_mask"], coarse
            )
            loss, parts = gate_loss(output, device_batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.gate_head.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        for key, value in parts.items():
            if math.isfinite(value):
                totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def evaluate(model, loader, coarse_store, device: str) -> dict:
    model.eval()
    rows = []
    joint_error = JointErrorAccumulator()
    gates = []
    distal_gates = []
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            coarse_store.lookup(batch["row"], device),
        )
        pose = output["pose_rel"].cpu()
        rows.extend(_pose_rows(pose, batch))
        joint_error.update(pose, batch)
        valid = batch["valid"].bool().to(device)
        gate = output["joint_confidence_gate"]
        gates.append(gate[valid].mean().cpu())
        distal_gates.append(gate[valid][:, DISTAL_JOINTS].mean().cpu())
    metrics = _aggregate_rows(rows)
    metrics["joint_gate_mean"] = float(torch.stack(gates).mean())
    metrics["distal_gate_mean"] = float(torch.stack(distal_gates).mean())
    metrics["danger_joint_mpjpe_m"] = joint_error.result()
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--hierarchical-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dh_hierarchical_pose" / "best_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--initial-strength", type=float, default=0.30)
    parser.add_argument("--max-adjustment", type=float, default=0.20)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--danger-boost", type=float, default=0.75)
    parser.add_argument("--danger-sample-weight", type=float, default=3.0)
    parser.add_argument("--distal-weight", type=float, default=2.5)
    parser.add_argument("--lambda-distal", type=float, default=0.35)
    parser.add_argument("--lambda-velocity", type=float, default=0.10)
    parser.add_argument("--lambda-oracle", type=float, default=0.02)
    parser.add_argument("--lambda-temporal", type=float, default=0.02)
    parser.add_argument("--lambda-prior", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dhg_joint_gate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_model, source_checkpoint = load_hierarchical_model(
        args.hierarchical_checkpoint, device
    )
    if source_checkpoint.get("protocol") != args.exp:
        raise RuntimeError("KP2-DH checkpoint protocol mismatch")
    model = JointConfidencePoseGate(
        pose_model, initial_strength=args.initial_strength,
        hidden=args.gate_hidden, dropout=args.dropout,
        max_adjustment=args.max_adjustment,
    ).to(device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    datasets_wrapped, loaders = make_loaders(datasets, args, device)
    train, validation, test = datasets_wrapped
    coarse_checkpoint = torch.load(
        args.coarse_cache, map_location="cpu", weights_only=False
    )
    if coarse_checkpoint.get("protocol") != args.exp:
        raise RuntimeError("coarse cache protocol mismatch")
    coarse_store = CoarsePoseStore(
        coarse_checkpoint["rows"], coarse_checkpoint["pose"]
    )
    optimizer = torch.optim.AdamW(
        model.gate_head.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    initial_validation = evaluate(model, loaders["val"], coarse_store, device)
    initial_score = pose_selection_score(initial_validation)
    best = {
        "score": initial_score, "epoch": 0,
        "state": copy.deepcopy(model.trainable_state_dict()),
        "metrics": initial_validation,
    }
    history = [{
        "epoch": 0, "train": None,
        "validation_score": initial_score,
        "validation": initial_validation,
        "note": "uniform 0.3 gate before confidence training",
    }]
    print(json.dumps(history[0], ensure_ascii=False), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, loaders["train"], coarse_store, optimizer, scaler,
            device, args,
        )
        scheduler.step()
        validation_metrics = evaluate(
            model, loaders["val"], coarse_store, device
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
                "state": copy.deepcopy(model.trainable_state_dict()),
                "metrics": validation_metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_trainable_state_dict(best["state"])
    test_metrics = evaluate(model, loaders["test"], coarse_store, device)
    result = {
        "run": "KP2-DHG-EXP01",
        "model_family": "NotiFi-KP2",
        "candidate_version": "KP2-DHG",
        "promotion_status": "experimental",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "inference_inputs": ["csi", "link_mask", "v13s_coarse_pose"],
        "config": vars(args) | {
            "hierarchical_checkpoint": report_path(
                args.hierarchical_checkpoint
            ),
            "coarse_cache": report_path(args.coarse_cache),
            "run_dir": report_path(args.run_dir),
        },
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
        },
        "architecture": {
            "frozen_pose_candidate": "KP2-DH",
            "coarse_pose": "V13S",
            "gate": "CSI-conditioned framewise joint confidence",
            "gate_initialization": args.initial_strength,
            "gate_range": [
                args.initial_strength - args.max_adjustment,
                args.initial_strength + args.max_adjustment,
            ],
            "gate_supervision": "bounded GT projection oracle on train only",
        },
        "selection": {"epoch": best["epoch"], "score": best["score"]},
        "validation": best["metrics"],
        "test": test_metrics,
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "gate_model": best["state"],
        "architecture": result["architecture"],
        "source": {
            "hierarchical_checkpoint": report_path(
                args.hierarchical_checkpoint
            ),
            "coarse_cache": report_path(args.coarse_cache),
        },
        "selection": result["selection"],
        "validation": result["validation"], "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
