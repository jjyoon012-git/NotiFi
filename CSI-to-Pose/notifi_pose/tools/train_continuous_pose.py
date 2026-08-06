"""Train KP2-C: CSI to a frozen continuous kinematic motion latent."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import contract as C
from ..continuous_pose import CSILatentPoseRegressor
from ..dataio.dataset import build_datasets
from ..external_pretraining import transplant_external_encoder
from ..motion_tokens import KinematicMotionTokenizer, trial_bone_lengths
from ..quality import (
    QualityWeightedDataset,
    protocol_audit_path,
    quality_summary,
)
from ..trainer import set_seed
from .audit_kinetic_pose import SignalCounterfactualDataset, delta
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .pretrain_motion_tokenizer import tokenizer_loss
from .train_kinetic_pose import _aggregate_rows, _pose_rows, pose_selection_score


def load_teacher(path: Path, device: str) -> tuple[KinematicMotionTokenizer, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if architecture.get("kind") != "continuous":
        raise ValueError("KP2-C requires a continuous motion checkpoint")
    model = KinematicMotionTokenizer(
        hidden=int(architecture["hidden"]),
        code_dim=int(architecture["code_dim"]),
        codes=int(architecture["codes"]),
        dropout=float(architecture["dropout"]),
        commitment=float(architecture["commitment"]),
        downsample=int(architecture["downsample"]),
        quantizer_levels=int(architecture["quantizer_levels"]),
        continuous=True,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, architecture


@torch.no_grad()
def fit_train_priors(teacher, dataset, batch_size: int,
                     device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = []
    lengths = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        pose = batch["pose_rel"].to(device)
        valid = batch["valid"].to(device).bool()
        encoded = teacher.encode(pose, valid)
        latents.append(encoded["latent"][encoded["token_mask"]].cpu())
        lengths.append(trial_bone_lengths(pose, valid).cpu())
    latent = torch.cat(latents)
    mean = latent.mean(0)
    std = latent.std(0).clamp_min(0.05)
    global_lengths = torch.cat(lengths).median(dim=0).values
    return mean, std, global_lengths


def make_loaders(datasets: dict, args, device: str):
    audit_path = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(
        pose_only(datasets["train"]), audit_path
    )
    validation = QualityWeightedDataset(
        pose_only(datasets["val"]), audit_path
    )
    test = QualityWeightedDataset(
        pose_only(datasets["test"]), audit_path
    )
    weights = train.sampler_weights()
    danger = torch.tensor(
        train.index.risk_id.to_numpy(dtype=np.int64) == 2, dtype=torch.bool
    )
    weights *= torch.where(
        danger, torch.tensor(args.danger_sample_weight, dtype=weights.dtype),
        torch.tensor(1.0, dtype=weights.dtype),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return (train, validation, test), {
        "train": DataLoader(
            train, batch_size=args.batch_size, sampler=sampler, num_workers=0,
            pin_memory=device == "cuda",
        ),
        "val": DataLoader(
            validation, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0,
        ),
    }


def normalized_target_latent(teacher, batch: dict, model) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        encoded = teacher.encode(batch["pose_rel"], batch["valid"].bool())
        target = (encoded["latent"] - model.latent_mean) / model.latent_std
    return target, encoded["token_mask"]


def train_epoch(model, teacher, loader, optimizer, scaler,
                device: str, args) -> dict:
    model.train()
    totals: dict[str, list[float]] = {}
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            output = model(batch["csi"], batch["link_mask"])
            zero = output["pose_rel"].sum() * 0.0
            pose_output = {
                **output,
                "codebook_loss": zero,
                "commitment_loss": zero,
                "diversity_loss": zero,
                "codebook_perplexity": zero + 1.0,
                "active_codes": zero + 1.0,
            }
            pose_loss, parts = tokenizer_loss(pose_output, batch, args)
            target_latent, token_mask = normalized_target_latent(
                teacher, batch, model
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
            loss = pose_loss + args.lambda_latent * latent_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        parts["latent"] = float(latent_loss.detach())
        parts["total_with_latent"] = float(loss.detach())
        for key, value in parts.items():
            if math.isfinite(value):
                totals.setdefault(key, []).append(value)
    return {key: float(np.mean(value)) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, teacher, loader, device: str) -> dict:
    model.eval()
    rows = []
    latent_errors = []
    gates = []
    for batch in loader:
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(device_batch["csi"], device_batch["link_mask"])
        rows.extend(_pose_rows(output["pose_rel"].cpu(), batch))
        target, mask = normalized_target_latent(teacher, device_batch, model)
        error = (output["normalized_motion_latent"] - target).square().mean(-1)
        latent_errors.append(error[mask].cpu())
        gates.append(output["fusion_gate"].mean((0, 1)).cpu())
    metrics = _aggregate_rows(rows)
    metrics["normalized_latent_rmse"] = float(
        torch.cat(latent_errors).mean().sqrt()
    )
    metrics["static_fusion_gate_mean"] = float(torch.stack(gates).mean())
    return metrics


@torch.no_grad()
def counterfactual_audit(model, teacher, test, args, device: str) -> dict:
    metrics = {}
    for mode in ("clean", "matched_shuffle", "temporal_reverse", "temporal_mean"):
        loader = DataLoader(
            SignalCounterfactualDataset(test, mode, args.seed + 54),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
        )
        metrics[mode] = evaluate(model, teacher, loader, device)
    return {
        "clean": metrics["clean"],
        "counterfactuals": {
            mode: {
                "metrics": metrics[mode],
                "delta_from_clean": delta(metrics["clean"], metrics[mode]),
            }
            for mode in ("matched_shuffle", "temporal_reverse", "temporal_mean")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2b_continuous_motion_autoencoder" / "best_model.pt",
    )
    parser.add_argument(
        "--external-csi-checkpoint", type=Path, default=None,
        help="Optional amplitude-Doppler pretraining checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lambda-latent", type=float, default=0.30)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--danger-boost", type=float, default=0.75)
    parser.add_argument("--danger-sample-weight", type=float, default=3.0)
    parser.add_argument("--distal-weight", type=float, default=1.5)
    parser.add_argument("--lambda-direction", type=float, default=0.25)
    parser.add_argument("--lambda-velocity", type=float, default=0.10)
    parser.add_argument("--lambda-diversity", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2c_continuous_csi_pose",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    train_pose = pose_only(datasets["train"])
    teacher, motion_architecture = load_teacher(args.motion_checkpoint, device)
    latent_mean, latent_std, global_lengths = fit_train_priors(
        teacher, train_pose, args.batch_size * 2, device
    )
    datasets_wrapped, loaders = make_loaders(datasets, args, device)
    train, validation, test = datasets_wrapped
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    base_model = make_model(p2_checkpoint, device)
    model = CSILatentPoseRegressor(
        base_model, teacher.decoder, latent_mean, latent_std, global_lengths,
        hidden=args.hidden, code_dim=int(motion_architecture["code_dim"]),
        temporal_layers=args.temporal_layers, heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    external_transfer = None
    if args.external_csi_checkpoint is not None:
        external_checkpoint = torch.load(
            args.external_csi_checkpoint, map_location="cpu", weights_only=False
        )
        shared = external_checkpoint.get("shared", external_checkpoint)
        external_transfer = {
            "checkpoint": report_path(args.external_csi_checkpoint),
            **transplant_external_encoder(model.dynamic, shared),
        }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = {"score": math.inf, "epoch": 0, "state": None, "metrics": None}
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, teacher, loaders["train"], optimizer, scaler, device, args
        )
        scheduler.step()
        validation_metrics = evaluate(model, teacher, loaders["val"], device)
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

    if best["state"] is None:
        raise RuntimeError("KP2-C training produced no checkpoint")
    model.load_trainable_state_dict(best["state"])
    test_metrics = evaluate(model, teacher, loaders["test"], device)
    counterfactual = counterfactual_audit(model, teacher, test, args, device)
    result = {
        "run": "KP2-C-EXP01",
        "model_family": "NotiFi-KP2",
        "candidate_version": "KP2-C",
        "promotion_status": "experimental",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "inference_inputs": ["csi", "link_mask"],
        "test_gt_bone_lengths_used": False,
        "config": vars(args) | {
            "p2_checkpoint": report_path(args.p2_checkpoint),
            "motion_checkpoint": report_path(args.motion_checkpoint),
            "external_csi_checkpoint": (
                report_path(args.external_csi_checkpoint)
                if args.external_csi_checkpoint is not None else None
            ),
            "run_dir": report_path(args.run_dir),
        },
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
            "quality": quality_summary(train),
        },
        "architecture": {
            "frozen_static_encoder": "P2",
            "dynamic_encoder": "multi_resolution_doppler",
            "fusion": "framewise_static_dynamic_gate_plus_residual",
            "motion_target": "continuous_2frame_kinematic_latent",
            "frozen_decoder": True,
            "skeleton": "train_only_global_median",
            "hidden": args.hidden,
            "code_dim": int(motion_architecture["code_dim"]),
            "temporal_layers": args.temporal_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "external_csi_initialization": external_transfer,
        },
        "selection": {"epoch": best["epoch"], "score": best["score"]},
        "validation": best["metrics"],
        "test": test_metrics,
        "counterfactual": counterfactual,
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "trainable_model": best["state"],
        "latent_mean": latent_mean, "latent_std": latent_std,
        "bone_lengths": global_lengths,
        "architecture": result["architecture"],
        "source": {
            "p2_checkpoint": report_path(args.p2_checkpoint),
            "motion_checkpoint": report_path(args.motion_checkpoint),
            "external_csi_checkpoint": (
                report_path(args.external_csi_checkpoint)
                if args.external_csi_checkpoint is not None else None
            ),
            "external_csi_transfer": external_transfer,
        },
        "selection": result["selection"],
        "validation": result["validation"], "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
