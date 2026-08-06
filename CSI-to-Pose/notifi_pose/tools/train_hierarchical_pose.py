"""Train KP2-DH: explicit torso, limb, distal, and motion supervision."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import contract as C
from ..continuous_pose import CSILatentPoseRegressor
from ..dataio.dataset import build_datasets
from ..hierarchical_pose import (
    DISTAL_JOINTS,
    LIMB_BONES,
    TORSO_BONES,
    HierarchicalCSIPoseRegressor,
)
from ..motion_tokens import pose_to_bones
from ..trainer import set_seed
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model, smooth_valid
from .pretrain_motion_tokenizer import tokenizer_loss
from .train_continuous_pose import (
    fit_train_priors,
    load_teacher,
    make_loaders,
    normalized_target_latent,
)
from .train_kinetic_pose import (
    CoarsePoseStore,
    _aggregate_rows,
    _pose_rows,
    _weighted_mean,
    pose_selection_score,
    relative_pose_speed,
)


def _shift_frames(values: torch.Tensor, amount: int) -> torch.Tensor:
    shifted = torch.zeros_like(values)
    if amount == 0:
        return values
    if amount > 0:
        shifted[:, :-amount] = values[:, amount:]
    else:
        shifted[:, -amount:] = values[:, :amount]
    return shifted


def banded_velocity_alignment(predicted: torch.Tensor, target: torch.Tensor,
                              valid: torch.Tensor, frame_weight: torch.Tensor,
                              radius: int, temperature: float,
                              lag_penalty: float) -> torch.Tensor:
    """Softly match each high-motion frame within a small temporal band."""
    if radius < 1:
        raise ValueError("alignment radius must be positive")
    if temperature <= 0.0:
        raise ValueError("alignment temperature must be positive")
    predicted_velocity = torch.zeros_like(predicted)
    target_velocity = torch.zeros_like(target)
    pair = valid[:, 1:] & valid[:, :-1]
    velocity_valid = torch.zeros_like(valid)
    velocity_valid[:, 1:] = pair
    predicted_velocity[:, 1:] = (
        predicted[:, 1:] - predicted[:, :-1]
    ) * C.TARGET_FPS
    target_velocity[:, 1:] = (
        target[:, 1:] - target[:, :-1]
    ) * C.TARGET_FPS

    costs = []
    masks = []
    for lag in range(-radius, radius + 1):
        shifted_target = _shift_frames(target_velocity, lag)
        shifted_valid = _shift_frames(velocity_valid, lag)
        cost = F.smooth_l1_loss(
            predicted_velocity, shifted_target,
            reduction="none", beta=0.20,
        ).mean((-1, -2))
        cost = cost + lag_penalty * abs(lag)
        costs.append(cost)
        masks.append(velocity_valid & shifted_valid)
    cost_stack = torch.stack(costs, dim=-1)
    mask_stack = torch.stack(masks, dim=-1)
    logits = (-cost_stack / temperature).masked_fill(~mask_stack, -1e4)
    probability = torch.softmax(logits, dim=-1) * mask_stack.to(logits.dtype)
    probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-6)
    aligned_cost = (probability * cost_stack).sum(-1)

    target_speed = torch.linalg.vector_norm(target_velocity, dim=-1).mean(-1)
    high_motion = velocity_valid & (target_speed > 0.08)
    return _weighted_mean(
        aligned_cost,
        frame_weight * high_motion.to(frame_weight.dtype),
    )


def danger_keyframe_loss(predicted: torch.Tensor, target: torch.Tensor,
                         valid: torch.Tensor, risk: torch.Tensor,
                         speed: torch.Tensor, frames: int,
                         distal_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Emphasize the fastest danger frames without collision heuristics."""
    if frames < 1:
        raise ValueError("danger keyframe count must be positive")
    eligible = valid & (risk == 2)[:, None]
    ranking = speed.masked_fill(~eligible, -torch.inf)
    count = min(frames, ranking.shape[1])
    indices = ranking.topk(count, dim=1).indices
    selected = torch.zeros_like(eligible)
    selected.scatter_(1, indices, True)
    selected &= eligible & torch.isfinite(ranking)
    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.03
    ).mean(-1)
    pose = _weighted_mean(coordinate, selected[..., None].to(coordinate.dtype))
    distal = _weighted_mean(
        coordinate[:, :, DISTAL_JOINTS],
        selected[..., None].to(coordinate.dtype),
    )
    return pose + distal_scale * distal, selected


def hierarchy_loss(output: dict, batch: dict, args) -> tuple[torch.Tensor, dict]:
    zero = output["pose_rel"].sum() * 0.0
    pose_output = {
        **output,
        "codebook_loss": zero,
        "commitment_loss": zero,
        "diversity_loss": zero,
        "codebook_perplexity": zero + 1.0,
        "active_codes": zero + 1.0,
    }
    base_loss, parts = tokenizer_loss(pose_output, batch, args)
    target = batch["pose_rel"]
    valid = batch["valid"].bool()
    risk = batch["risk_id"]
    quality = batch["quality_weight"].to(target.device)
    speed = relative_pose_speed(target, valid)
    frame_weight = 1.0 + args.motion_weight * (speed / 0.50).clamp(0.0, 2.0)
    frame_weight *= 1.0 + args.danger_boost * (risk == 2).float()[:, None]
    frame_weight *= quality[:, None] * valid.to(frame_weight.dtype)

    target_direction, _ = pose_to_bones(target)
    direction_error = 1.0 - (
        output["bone_direction"] * target_direction
    ).sum(-1).clamp(-1.0, 1.0)
    torso_direction = _weighted_mean(
        direction_error[:, :, TORSO_BONES], frame_weight[..., None]
    )
    limb_direction = _weighted_mean(
        direction_error[:, :, LIMB_BONES], frame_weight[..., None]
    )

    endpoint_error = F.smooth_l1_loss(
        output["pose_rel"][:, :, DISTAL_JOINTS],
        target[:, :, DISTAL_JOINTS],
        reduction="none", beta=0.03,
    ).mean(-1)
    endpoint = _weighted_mean(endpoint_error, frame_weight[..., None])

    target_velocity = torch.zeros_like(target)
    velocity_valid = valid.clone()
    velocity_valid[:, 0] = False
    target_velocity[:, 1:] = (
        target[:, 1:] - target[:, :-1]
    ) * C.TARGET_FPS
    velocity_error = F.smooth_l1_loss(
        output["kinetic_velocity"], target_velocity,
        reduction="none", beta=0.20,
    ).mean(-1)
    velocity_joint_weight = velocity_error.new_ones(C.N_JOINTS)
    velocity_joint_weight[list(DISTAL_JOINTS)] = args.distal_weight
    auxiliary_velocity = _weighted_mean(
        velocity_error,
        frame_weight[..., None]
        * velocity_valid[..., None].to(frame_weight.dtype)
        * velocity_joint_weight,
    )
    residual_regularization = (
        output["endpoint_delta"].square().mean()
        + 0.25 * output["torso_direction_delta"].square().mean()
        + 0.25 * output["limb_direction_delta"].square().mean()
    )
    alignment = banded_velocity_alignment(
        output["pose_rel"], target, valid, frame_weight,
        radius=args.alignment_radius,
        temperature=args.alignment_temperature,
        lag_penalty=args.alignment_lag_penalty,
    )
    keyframe, keyframe_mask = danger_keyframe_loss(
        output["pose_rel"], target, valid, risk, speed,
        frames=args.danger_keyframe_frames,
        distal_scale=args.danger_keyframe_distal_scale,
    )
    total = (
        base_loss
        + args.lambda_torso_direction * torso_direction
        + args.lambda_limb_direction * limb_direction
        + args.lambda_endpoint * endpoint
        + args.lambda_aux_velocity * auxiliary_velocity
        + args.lambda_residual * residual_regularization
        + args.lambda_alignment * alignment
        + args.lambda_danger_keyframe * keyframe
    )
    parts.update({
        "torso_direction_explicit": float(torso_direction.detach()),
        "limb_direction_explicit": float(limb_direction.detach()),
        "endpoint_explicit": float(endpoint.detach()),
        "auxiliary_velocity": float(auxiliary_velocity.detach()),
        "hierarchy_residual_regularization": float(
            residual_regularization.detach()
        ),
        "banded_velocity_alignment": float(alignment.detach()),
        "danger_keyframe": float(keyframe.detach()),
        "danger_keyframe_fraction": float(
            keyframe_mask.float().mean().detach()
        ),
        "hierarchy_total": float(total.detach()),
    })
    return total, parts


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
            loss = pose_loss + args.lambda_latent * latent_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        parts["latent"] = float(latent_loss.detach())
        parts["total_with_latent"] = float(loss.detach())
        for key, value in parts.items():
            if math.isfinite(value):
                totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


class JointErrorAccumulator:
    def __init__(self):
        self.danger_sum = torch.zeros(C.N_JOINTS, dtype=torch.float64)
        self.danger_count = 0

    def update(self, predicted: torch.Tensor, batch: dict) -> None:
        valid = batch["valid"].bool().cpu()
        danger = batch["risk_id"].cpu() == 2
        predicted = smooth_valid(predicted.float().cpu(), valid, 5)
        target = batch["pose_rel"].float().cpu()
        mask = valid & danger[:, None]
        error = torch.linalg.vector_norm(predicted - target, dim=-1)
        if mask.any():
            self.danger_sum += (
                error * mask[..., None].to(error.dtype)
            ).sum((0, 1)).double()
            self.danger_count += int(mask.sum())

    def result(self) -> dict[str, float]:
        denominator = max(self.danger_count, 1)
        return {
            name: float(self.danger_sum[index] / denominator)
            for index, name in enumerate(C.JOINT_NAMES)
        }


@torch.no_grad()
def evaluate(model, teacher, loader, device: str,
             coarse_store: CoarsePoseStore | None = None,
             strength: float = 1.0) -> dict:
    model.eval()
    rows = []
    latent_errors = []
    gates = []
    joint_error = JointErrorAccumulator()
    for batch in loader:
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(device_batch["csi"], device_batch["link_mask"])
        pose = output["pose_rel"].cpu()
        if coarse_store is not None:
            coarse = coarse_store.lookup(batch["row"], "cpu").float()
            pose = coarse + strength * (pose - coarse)
            pose -= pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        rows.extend(_pose_rows(pose, batch))
        joint_error.update(pose, batch)
        target, mask = normalized_target_latent(
            teacher, device_batch, model.backbone
        )
        error = (
            output["normalized_motion_latent"] - target
        ).square().mean(-1)
        latent_errors.append(error[mask].cpu())
        gates.append(output["fusion_gate"].mean((0, 1)).cpu())
    metrics = _aggregate_rows(rows)
    metrics["normalized_latent_rmse"] = float(
        torch.cat(latent_errors).mean().sqrt()
    )
    metrics["static_fusion_gate_mean"] = float(torch.stack(gates).mean())
    metrics["danger_joint_mpjpe_m"] = joint_error.result()
    return metrics


@torch.no_grad()
def evaluate_blends(model, loader, coarse_store: CoarsePoseStore,
                    strengths: list[float], device: str) -> dict[float, dict]:
    model.eval()
    rows = {strength: [] for strength in strengths}
    for batch in loader:
        pose = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )["pose_rel"].cpu()
        coarse = coarse_store.lookup(batch["row"], "cpu").float()
        for strength in strengths:
            candidate = coarse + strength * (pose - coarse)
            candidate -= candidate[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
            rows[strength].extend(_pose_rows(candidate, batch))
    return {
        strength: _aggregate_rows(items) for strength, items in rows.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--kp2c-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2c_continuous_csi_pose" / "best_model.pt",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--direction-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-scale", type=float, default=0.40)
    parser.add_argument("--dropout", type=float, default=0.08)
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--experiment-name", default="KP2-DH-EXP01")
    parser.add_argument("--candidate-version", default="KP2-DH")
    parser.add_argument(
        "--blend-strengths", type=float, nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dh_hierarchical_pose",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial = torch.load(
        args.kp2c_checkpoint, map_location="cpu", weights_only=False
    )
    if initial.get("protocol") != args.exp:
        raise RuntimeError("KP2-C checkpoint protocol mismatch")
    source = initial["source"]
    p2_path = C.PROJECT_ROOT / source["p2_checkpoint"]
    motion_path = C.PROJECT_ROOT / source["motion_checkpoint"]

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    train_pose = pose_only(datasets["train"])
    teacher, motion_architecture = load_teacher(motion_path, device)
    latent_mean, latent_std, global_lengths = fit_train_priors(
        teacher, train_pose, args.batch_size * 2, device
    )
    datasets_wrapped, loaders = make_loaders(datasets, args, device)
    train, validation, test = datasets_wrapped
    p2_checkpoint = torch.load(p2_path, map_location=device, weights_only=False)
    base_model = make_model(p2_checkpoint, device)
    source_architecture = initial["architecture"]
    backbone = CSILatentPoseRegressor(
        base_model, teacher.decoder, latent_mean, latent_std, global_lengths,
        hidden=int(source_architecture["hidden"]),
        code_dim=int(motion_architecture["code_dim"]),
        temporal_layers=int(source_architecture.get("temporal_layers", 2)),
        heads=int(source_architecture.get("heads", 4)),
        dropout=float(source_architecture.get("dropout", 0.08)),
    ).to(device)
    model = HierarchicalCSIPoseRegressor(
        backbone, direction_scale=args.direction_scale,
        endpoint_scale=args.endpoint_scale, dropout=args.dropout,
    ).to(device)
    model.load_kp2c_state_dict(initial["trainable_model"])
    if args.resume_checkpoint is not None:
        resumed = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resumed.get("protocol") != args.exp:
            raise RuntimeError("hierarchical resume checkpoint protocol mismatch")
        model.load_trainable_state_dict(resumed["trainable_model"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    initial_validation = evaluate(model, teacher, loaders["val"], device)
    initial_score = pose_selection_score(initial_validation)
    best = {
        "score": initial_score,
        "epoch": 0,
        "state": copy.deepcopy(model.trainable_state_dict()),
        "metrics": initial_validation,
    }
    history = [{
        "epoch": 0,
        "train": None,
        "validation_score": initial_score,
        "validation": initial_validation,
        "note": (
            "resumed hierarchical checkpoint before ablation fine-tuning"
            if args.resume_checkpoint is not None
            else "KP2-C initialization before hierarchical fine-tuning"
        ),
    }]
    print(json.dumps(history[0], ensure_ascii=False), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, teacher, loaders["train"], optimizer, scaler, device, args
        )
        scheduler.step()
        validation_metrics = evaluate(
            model, teacher, loaders["val"], device
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

    if best["state"] is None:
        raise RuntimeError("KP2-DH training produced no checkpoint")
    model.load_trainable_state_dict(best["state"])
    coarse_checkpoint = torch.load(
        args.coarse_cache, map_location="cpu", weights_only=False
    )
    if coarse_checkpoint.get("protocol") != args.exp:
        raise RuntimeError("coarse cache protocol mismatch")
    coarse_store = CoarsePoseStore(
        coarse_checkpoint["rows"], coarse_checkpoint["pose"]
    )
    strengths = [float(value) for value in args.blend_strengths]
    validation_blends = evaluate_blends(
        model, loaders["val"], coarse_store, strengths, device
    )
    blend_scores = {
        strength: pose_selection_score(metrics)
        for strength, metrics in validation_blends.items()
    }
    selected_strength = min(blend_scores, key=blend_scores.get)
    standalone_test = evaluate(model, teacher, loaders["test"], device)
    blended_test = evaluate(
        model, teacher, loaders["test"], device,
        coarse_store=coarse_store, strength=selected_strength,
    )

    result = {
        "run": args.experiment_name,
        "model_family": "NotiFi-KP2",
        "candidate_version": args.candidate_version,
        "promotion_status": "experimental",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "inference_inputs": ["csi", "link_mask"],
        "test_gt_bone_lengths_used": False,
        "config": vars(args) | {
            "kp2c_checkpoint": report_path(args.kp2c_checkpoint),
            "resume_checkpoint": (
                report_path(args.resume_checkpoint)
                if args.resume_checkpoint is not None else None
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
            "initialization": "KP2-C",
            "whole_body_prior": "continuous_2frame_kinematic_latent",
            "explicit_heads": [
                "torso_bone_direction", "limb_bone_direction",
                "distal_cartesian_residual", "joint_velocity",
            ],
            "pose_decoder": "global_length_FK_plus_direct_distal_residual",
            "frozen_teacher_decoder": True,
            "skeleton": "train_only_global_median",
            "hidden": int(source_architecture["hidden"]),
            "code_dim": int(motion_architecture["code_dim"]),
            "temporal_layers": int(source_architecture.get("temporal_layers", 2)),
            "heads": int(source_architecture.get("heads", 4)),
            "dropout": args.dropout,
            "direction_scale": args.direction_scale,
            "endpoint_scale": args.endpoint_scale,
            "temporal_alignment": {
                "weight": args.lambda_alignment,
                "radius_frames": args.alignment_radius,
                "temperature": args.alignment_temperature,
                "lag_penalty": args.alignment_lag_penalty,
                "scope": "GT high-motion velocity frames only",
            },
            "danger_keyframes": {
                "weight": args.lambda_danger_keyframe,
                "frames_per_trial": args.danger_keyframe_frames,
                "ranking": "GT mean joint speed on train only",
                "distal_scale": args.danger_keyframe_distal_scale,
            },
        },
        "selection": {"epoch": best["epoch"], "score": best["score"]},
        "validation": best["metrics"],
        "blend_selection": {
            "strength": selected_strength,
            "score": blend_scores[selected_strength],
            "candidates": {
                str(strength): {
                    "score": blend_scores[strength],
                    "metrics": validation_blends[strength],
                }
                for strength in strengths
            },
        },
        "test": {
            "standalone": standalone_test,
            "selected_blend": blended_test,
        },
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
            "kp2c_checkpoint": report_path(args.kp2c_checkpoint),
            "resume_checkpoint": (
                report_path(args.resume_checkpoint)
                if args.resume_checkpoint is not None else None
            ),
            "p2_checkpoint": report_path(p2_path),
            "motion_checkpoint": report_path(motion_path),
        },
        "selection": result["selection"],
        "blend_selection": result["blend_selection"],
        "validation": result["validation"], "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
