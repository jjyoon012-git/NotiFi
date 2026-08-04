"""Train and validate a P2 coarse model plus bounded V9 residual decoder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import contract as C
from .. import losses as L
from ..hybrid_v10 import build_residual_hybrid
from ..seen_v4 import (
    bounded_piecewise_alignment_loss,
    trajectory_descriptor,
    trajectory_reconstruction_loss,
)
from ..trainer import set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, report_path
from .evaluate_sealed import make_model
from .train_seen_v4_trajectory import (
    _balanced_weight,
    classification_metrics,
    collect_classification_logits,
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
    move_batch,
)


def pose_selection_score(metrics: dict) -> float:
    speed = max(float(metrics["pose_speed_ratio"]), 1e-3)
    return (
        float(metrics["mpjpe_m"])
        + 0.15 * float(metrics["dynamic_mpjpe_m"])
        + 0.35 * float(metrics["danger_mpjpe_m"])
        + 0.10 * float(metrics["danger_endpoint_mpjpe_m"])
        + 0.15 * abs(math.log(speed))
    )


def pose_geometry_score(metrics: dict) -> float:
    """Geometry-only checkpoint score; dynamics are gated after training."""
    return (
        float(metrics["mpjpe_m"])
        + 0.15 * float(metrics["dynamic_mpjpe_m"])
        + 0.35 * float(metrics["danger_mpjpe_m"])
        + 0.10 * float(metrics["danger_endpoint_mpjpe_m"])
    )


def _pose_checkpoint_score(metrics: dict, mode: str) -> float:
    if mode == "geometry":
        return pose_geometry_score(metrics)
    if mode == "composite":
        return pose_selection_score(metrics)
    raise ValueError(f"unknown pose checkpoint score: {mode}")


def root_selection_score(metrics: dict) -> float:
    return (
        float(metrics["root_error_m"])
        + 0.40 * float(metrics["danger_root_error_m"])
        + 0.20 * float(metrics["danger_root_drop_mae_m"])
    )


class DeterministicValidationPerturbation(Dataset):
    """Apply a fixed validation corruption without modifying source examples."""

    def __init__(self, target: Dataset, mode: str):
        self.target = target
        self.mode = mode

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict:
        sample = self.target[index]
        sample = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        if self.mode == "drop_one_link":
            mask = sample["link_mask"]
            alive = torch.nonzero(mask.any(0), as_tuple=False).flatten()
            if len(alive) >= 2:
                selected = int(alive[index % len(alive)])
                mask[:, selected] = False
                sample["csi"][:, selected] = 0.0
        elif self.mode != "clean":
            raise ValueError(f"unknown validation perturbation: {self.mode}")
        return sample


def _replace_validation_loaders(loaders: dict, args) -> None:
    """Select a robustness expert on deterministic validation only."""
    if args.validation_perturbation == "clean":
        return
    for key in ("val", "val_class"):
        loaders[key] = DataLoader(
            DeterministicValidationPerturbation(
                loaders[key].dataset, args.validation_perturbation
            ),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )


def _training_metadata(args: argparse.Namespace) -> dict:
    """Small, serializable record of settings that affect learned weights."""
    return {
        "protocol": args.exp,
        "objective": args.objective,
        "residual_decoder": args.residual_decoder,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "danger_weight": args.danger_weight,
        "pose_velocity_weight": args.pose_velocity_weight,
        "pose_velocity_lags": list(args.pose_velocity_lags),
        "pose_bone_weight": args.pose_bone_weight,
        "pose_endpoint_weight": args.pose_endpoint_weight,
        "pose_euclidean_weight": args.pose_euclidean_weight,
        "pose_danger_distal_weight": args.pose_danger_distal_weight,
        "pose_shift_robust_weight": args.pose_shift_robust_weight,
        "pose_checkpoint_score": args.pose_checkpoint_score,
        "root_velocity_weight": args.root_velocity_weight,
        "root_velocity_lags": list(args.root_velocity_lags),
        "root_displacement_weight": args.root_displacement_weight,
        "root_endpoint_weight": args.root_endpoint_weight,
        "root_shift_robust_weight": args.root_shift_robust_weight,
        "alignment_weight": args.alignment_weight,
        "link_dropout_p": args.link_dropout_p,
        "max_link_drop": args.max_link_drop,
        "rf_augment": args.rf_augment,
        "validation_perturbation": args.validation_perturbation,
        "source_p2": report_path(args.p2_checkpoint),
        "warm_start": (
            report_path(args.init_hybrid_checkpoint)
            if args.init_hybrid_checkpoint is not None else None
        ),
    }


def global_shift_pose_loss(predicted: torch.Tensor, target: torch.Tensor,
                           valid: torch.Tensor, max_shift: int,
                           shift_penalty: float = 0.001) -> torch.Tensor:
    """Pose loss tolerant to one small, constant trial-level time offset."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    candidates = []
    for shift in range(-max_shift, max_shift + 1):
        shifted = torch.zeros_like(target)
        shifted_valid = torch.zeros_like(valid)
        if shift >= 0:
            stop = target.shape[1] - shift
            if stop > 0:
                shifted[:, :stop] = target[:, shift:]
                shifted_valid[:, :stop] = valid[:, shift:]
        else:
            start = -shift
            if start < target.shape[1]:
                shifted[:, start:] = target[:, :target.shape[1] - start]
                shifted_valid[:, start:] = valid[:, :valid.shape[1] - start]
        overlap = valid & shifted_valid
        error = F.smooth_l1_loss(
            predicted, shifted, beta=0.10, reduction="none"
        ).mean((2, 3))
        mask = overlap.to(error.dtype)
        cost = (error * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        cost = cost + shift_penalty * abs(shift) / max(max_shift, 1)
        candidates.append(cost)
    return torch.stack(candidates, dim=-1).min(-1).values


def global_shift_root_loss(predicted: torch.Tensor, target: torch.Tensor,
                           valid: torch.Tensor, max_shift: int,
                           shift_penalty: float = 0.001) -> torch.Tensor:
    """Root loss tolerant to one bounded, constant trial-level offset."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    candidates = []
    for shift in range(-max_shift, max_shift + 1):
        shifted = torch.zeros_like(target)
        shifted_valid = torch.zeros_like(valid)
        if shift >= 0:
            stop = target.shape[1] - shift
            if stop > 0:
                shifted[:, :stop] = target[:, shift:]
                shifted_valid[:, :stop] = valid[:, shift:]
        else:
            start = -shift
            if start < target.shape[1]:
                shifted[:, start:] = target[:, :target.shape[1] - start]
                shifted_valid[:, start:] = valid[:, :valid.shape[1] - start]
        overlap = valid & shifted_valid
        error = F.smooth_l1_loss(
            predicted, shifted, beta=0.10, reduction="none"
        ).mean(2)
        mask = overlap.to(error.dtype)
        cost = (error * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        cost = cost + shift_penalty * abs(shift) / max(max_shift, 1)
        candidates.append(cost)
    return torch.stack(candidates, dim=-1).min(-1).values


def pose_only_reconstruction_loss(output: dict, batch: dict,
                                  velocity_weight: float,
                                  bone_weight: float,
                                  endpoint_weight: float = 0.0,
                                  alignment_weight: float = 0.0,
                                  max_shift: int = 15,
                                  euclidean_weight: float = 0.0,
                                  danger_distal_weight: float = 0.0,
                                  shift_robust_weight: float = 0.0,
                                  velocity_lags: tuple[int, ...] = (5,),
                                  ) -> tuple[torch.Tensor, dict]:
    """Train the residual pose path without gradients from unused heads."""
    valid = batch["valid"].bool()
    quality = batch.get(
        "quality_weight", torch.ones(len(valid), device=valid.device)
    ).to(output["pose_rel"].dtype)
    pose = L.smooth_l1_per_sample(
        output["pose_rel"], batch["pose_rel"], valid
    )
    joint_distance = torch.linalg.vector_norm(
        output["pose_rel"] - batch["pose_rel"], dim=-1
    )
    euclidean = L.masked_per_sample(joint_distance, valid)
    danger = batch["risk_id"].eq(2)
    shift_robust = torch.zeros_like(pose)
    if shift_robust_weight and danger.any():
        shift_robust = global_shift_pose_loss(
            output["pose_rel"], batch["pose_rel"], valid, max_shift
        ) * danger.to(pose.dtype)
    distal = L.masked_per_sample(
        joint_distance[:, :, list(L.DISTAL_JOINTS)], valid
    ) * danger.to(pose.dtype)
    bone = L.BoneLoss().to(output["pose_rel"].device).per_sample(
        output["pose_rel"], batch["pose_rel"], valid
    )
    if not velocity_lags or any(lag < 1 for lag in velocity_lags):
        raise ValueError("velocity_lags must contain positive integers")
    velocity_terms = []
    for lag in velocity_lags:
        interval = valid[:, lag:] & valid[:, :-lag]
        predicted_delta = (
            output["pose_rel"][:, lag:] - output["pose_rel"][:, :-lag]
        ) * (C.TARGET_FPS / lag)
        target_delta = (
            batch["pose_rel"][:, lag:] - batch["pose_rel"][:, :-lag]
        ) * (C.TARGET_FPS / lag)
        velocity_terms.append(L.smooth_l1_per_sample(
            predicted_delta, target_delta, interval, beta=0.20
        ))
    velocity = torch.stack(velocity_terms).mean(0)
    endpoint = torch.zeros_like(pose)
    if endpoint_weight:
        for item in range(len(valid)):
            if int(batch["risk_id"][item]) != 2:
                continue
            frames = torch.nonzero(valid[item], as_tuple=False).flatten()
            if len(frames):
                selected = frames[-int(round(C.TARGET_FPS)):]
                endpoint[item] = torch.linalg.vector_norm(
                    output["pose_rel"][item, selected]
                    - batch["pose_rel"][item, selected], dim=-1,
                ).mean()
    alignment = torch.zeros_like(pose)
    if alignment_weight and danger.any():
        predicted_descriptor = trajectory_descriptor(
            output["pose_rel"], output["root"], valid
        )
        target_descriptor = trajectory_descriptor(
            batch["pose_rel"], batch["root"], valid
        )
        alignment = bounded_piecewise_alignment_loss(
            predicted_descriptor, target_descriptor, valid,
            max_shift=max_shift,
        ) * danger.to(pose.dtype)
    per_sample = (
        pose + bone_weight * bone + velocity_weight * velocity
        + endpoint_weight * endpoint + alignment_weight * alignment
        + euclidean_weight * euclidean
        + danger_distal_weight * distal
        + shift_robust_weight * (shift_robust - pose) * danger.to(pose.dtype)
    )
    total = (per_sample * quality).sum() / quality.sum().clamp_min(1e-6)
    return total, {
        "total": float(total.detach()),
        "pose": float(pose.mean().detach()),
        "bone": float(bone.mean().detach()),
        "velocity": float(velocity.mean().detach()),
        "endpoint": float(endpoint.mean().detach()),
        "euclidean": float(euclidean.mean().detach()),
        "danger_distal": float(
            distal[danger].mean().detach() if danger.any()
            else distal.new_zeros(())
        ),
        "shift_robust": float(
            shift_robust[danger].mean().detach() if danger.any()
            else shift_robust.new_zeros(())
        ),
        "alignment": float(
            alignment[danger].mean().detach() if danger.any()
            else alignment.new_zeros(())
        ),
    }


def root_only_reconstruction_loss(
    output: dict, batch: dict, velocity_weight: float,
    displacement_weight: float, endpoint_weight: float,
    velocity_lags: tuple[int, ...] = (5,),
    shift_robust_weight: float = 0.0, max_shift: int = 15,
) -> tuple[torch.Tensor, dict]:
    """Train a clean-protocol root expert without pose/logit gradients."""
    valid = batch["valid"].bool()
    quality = batch.get(
        "quality_weight", torch.ones(len(valid), device=valid.device)
    ).to(output["root"].dtype)
    danger = batch["risk_id"].eq(2)
    root = L.smooth_l1_per_sample(output["root"], batch["root"], valid)
    shift_robust = torch.zeros_like(root)
    if shift_robust_weight and danger.any():
        shifted = global_shift_root_loss(
            output["root"], batch["root"], valid, max_shift
        )
        shift_robust = shifted * danger.to(root.dtype)
        mix = shift_robust_weight * danger.to(root.dtype)
        root = root + mix * (shifted - root)

    predicted_relative = output["root"] - output["root"][:, :1]
    target_relative = batch["root"] - batch["root"][:, :1]
    displacement = L.smooth_l1_per_sample(
        predicted_relative, target_relative, valid, beta=0.10
    )

    velocity_terms = []
    for lag in velocity_lags:
        if lag < 1 or lag >= valid.shape[1]:
            raise ValueError(f"invalid root velocity lag: {lag}")
        interval = valid[:, lag:] & valid[:, :-lag]
        scale = C.TARGET_FPS / lag
        predicted_velocity = (
            output["root"][:, lag:] - output["root"][:, :-lag]
        ) * scale
        target_velocity = (
            batch["root"][:, lag:] - batch["root"][:, :-lag]
        ) * scale
        velocity_terms.append(L.smooth_l1_per_sample(
            predicted_velocity, target_velocity, interval, beta=0.20
        ))
    velocity = torch.stack(velocity_terms).mean(0)

    endpoint = torch.zeros_like(root)
    if endpoint_weight:
        for item in range(len(valid)):
            if not bool(danger[item]):
                continue
            frames = torch.nonzero(valid[item], as_tuple=False).flatten()
            if len(frames):
                selected = frames[-int(round(C.TARGET_FPS)):]
                endpoint[item] = torch.linalg.vector_norm(
                    output["root"][item, selected]
                    - batch["root"][item, selected], dim=-1,
                ).mean()

    per_sample = (
        root + displacement_weight * displacement
        + velocity_weight * velocity + endpoint_weight * endpoint
    )
    total = (per_sample * quality).sum() / quality.sum().clamp_min(1e-6)
    return total, {
        "total": float(total.detach()),
        "root": float(root.mean().detach()),
        "displacement": float(displacement.mean().detach()),
        "velocity": float(velocity.mean().detach()),
        "endpoint": float(endpoint.mean().detach()),
        "shift_robust": float(shift_robust.mean().detach()),
    }


def classification_only_loss(
    output: dict, batch: dict, class_weight: torch.Tensor,
    risk_weight: torch.Tensor, lambda_class: float, lambda_risk: float,
) -> tuple[torch.Tensor, dict]:
    """Train class and risk adapters without pose/root gradient interference."""
    quality = batch.get(
        "quality_weight",
        torch.ones(len(batch["class_id"]), device=batch["class_id"].device),
    ).to(output["class_logits"].dtype)
    class_loss = F.cross_entropy(
        output["class_logits"], batch["class_id"],
        weight=class_weight, reduction="none",
    )
    risk_loss = F.cross_entropy(
        output["risk_logits"], batch["risk_id"],
        weight=risk_weight, reduction="none",
    )
    per_sample = lambda_class * class_loss + lambda_risk * risk_loss
    total = (per_sample * quality).sum() / quality.sum().clamp_min(1e-6)
    return total, {
        "total": float(total.detach()),
        "classification": float(class_loss.mean().detach()),
        "risk": float(risk_loss.mean().detach()),
    }


def classification_selection_score(metrics: dict) -> float:
    """Lower is better; danger misses and safe false alarms are explicit."""
    class_metrics = metrics["class"]
    risk = metrics["risk"]
    return (
        1.0 - float(class_metrics["macro_f1"])
        + 0.50 * (1.0 - float(risk["macro_f1"]))
        + 1.00 * (1.0 - float(risk["danger_recall"]))
        + 0.50 * float(risk["safe_to_danger_rate"])
    )


def _candidate(model, loaders, device: str, max_shift: int,
               pose: float, root: float, classification: float,
               risk: float) -> dict:
    model.set_calibration(pose, root, classification, risk)
    return evaluate_trajectory(model, loaders["val"], device, max_shift)


def select_calibration(model, loaders, device: str, args) -> dict:
    pose_candidates = []
    for strength in args.pose_strengths:
        metrics = _candidate(
            model, loaders, device, args.max_shift,
            strength, 0.0, 0.0, 0.0,
        )
        pose_candidates.append({
            "strength": strength,
            "feasible_speed": 0.85 <= metrics["pose_speed_ratio"] <= 1.20,
            "score": pose_selection_score(metrics),
            "validation": metrics,
        })
    feasible_pose = [item for item in pose_candidates if item["feasible_speed"]]
    pose = min(feasible_pose or pose_candidates, key=lambda item: item["score"])

    root_candidates = []
    for strength in args.root_strengths:
        metrics = _candidate(
            model, loaders, device, args.max_shift,
            pose["strength"], strength, 0.0, 0.0,
        )
        root_candidates.append({
            "strength": strength,
            "score": root_selection_score(metrics),
            "validation": metrics,
        })
    root = min(root_candidates, key=lambda item: item["score"])
    root_baseline = next(
        item for item in root_candidates if float(item["strength"]) == 0.0
    )
    root_gain = (
        root_baseline["validation"]["root_error_m"]
        - root["validation"]["root_error_m"]
    )
    if root_gain < args.minimum_root_gain:
        root = root_baseline

    class_candidates = []
    for strength in args.logit_strengths:
        model.set_calibration(
            pose["strength"], root["strength"], strength, 0.0
        )
        metrics = evaluate_classification(model, loaders["val_class"], device)
        class_candidates.append({
            "strength": strength,
            "score": metrics["class"]["macro_f1"] + 0.25 * metrics["class"]["accuracy"],
            "validation": metrics["class"],
        })
    classification = max(class_candidates, key=lambda item: item["score"])

    risk_candidates = []
    for strength in args.logit_strengths:
        model.set_calibration(
            pose["strength"], root["strength"],
            classification["strength"], strength,
        )
        logits = collect_classification_logits(model, loaders["val_class"], device)
        bias_steps = int(round(args.maximum_danger_bias / 0.05)) + 1
        for bias in np.linspace(0.0, args.maximum_danger_bias, bias_steps):
            metrics = classification_metrics(logits, float(bias))["risk"]
            risk_candidates.append({
                "strength": strength,
                "danger_logit_bias": float(bias),
                "feasible": metrics["danger_recall"] >= args.minimum_danger_recall,
                "score": (
                    metrics["macro_f1"] + 0.25 * metrics["accuracy"]
                    - 0.50 * metrics["safe_to_danger_rate"]
                ),
                "validation": metrics,
            })
    feasible_risk = [item for item in risk_candidates if item["feasible"]]
    risk = max(feasible_risk or risk_candidates, key=lambda item: item["score"])
    return {
        "pose": pose,
        "root": root,
        "classification": classification,
        "risk": risk,
        "pose_candidates": pose_candidates,
        "root_candidates": root_candidates,
        "class_candidates": class_candidates,
        "risk_candidates": risk_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--init-hybrid-checkpoint", type=Path, default=None,
        help="warm-start a matching residual decoder for a short fine-tune",
    )
    parser.add_argument(
        "--exp", default="single_split_lmh_e01",
        choices=("single_split", "single_split_lmh_e01"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--lambda-class", type=float, default=0.10)
    parser.add_argument("--lambda-risk", type=float, default=0.10)
    parser.add_argument("--risk-danger-boost", type=float, default=1.25)
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument(
        "--objective",
        choices=("full", "pose_only", "root_only", "classification_only"),
        default="full",
    )
    parser.add_argument(
        "--residual-decoder",
        choices=(
            "dense", "graph", "spectral", "cartesian", "subcarrier",
            "direct_root",
        ),
        default="dense"
    )
    parser.add_argument("--pose-velocity-weight", type=float, default=0.05)
    parser.add_argument(
        "--pose-velocity-lags", type=int, nargs="+", default=(5,),
        help="frame intervals used by the multi-scale pose velocity loss",
    )
    parser.add_argument("--pose-bone-weight", type=float, default=0.05)
    parser.add_argument("--pose-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--pose-euclidean-weight", type=float, default=0.0)
    parser.add_argument("--pose-danger-distal-weight", type=float, default=0.0)
    parser.add_argument(
        "--pose-shift-robust-weight", type=float, default=0.0,
        help="replace this fraction of danger pose loss with bounded global-shift loss",
    )
    parser.add_argument(
        "--pose-checkpoint-score", choices=("geometry", "composite"),
        default="composite",
        help="geometry defers temporal realism checks to validation calibration",
    )
    parser.add_argument("--root-velocity-weight", type=float, default=0.02)
    parser.add_argument(
        "--root-velocity-lags", type=int, nargs="+", default=(5,),
        help="frame intervals used by the multi-scale root velocity loss",
    )
    parser.add_argument("--root-displacement-weight", type=float, default=0.25)
    parser.add_argument("--root-endpoint-weight", type=float, default=0.10)
    parser.add_argument(
        "--root-shift-robust-weight", type=float, default=0.0,
        help="danger-root loss fraction using bounded trial-level alignment",
    )
    parser.add_argument("--link-dropout-p", type=float, default=0.0)
    parser.add_argument("--max-link-drop", type=int, default=2)
    parser.add_argument("--rf-augment", action="store_true")
    parser.add_argument(
        "--validation-perturbation",
        choices=("clean", "drop_one_link"),
        default="clean",
        help="select a specialist on a deterministic validation corruption",
    )
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="stop after validation checkpoint selection for seed ablations",
    )
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--minimum-danger-recall", type=float, default=0.94)
    parser.add_argument("--maximum-danger-bias", type=float, default=4.0)
    parser.add_argument(
        "--minimum-root-gain", type=float, default=0.005,
        help="minimum validation root-error gain in metres before enabling the residual",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pose-strengths", type=float, nargs="+",
        default=(0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--root-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--logit-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v9_hybrid_v10",
    )
    args = parser.parse_args()

    if not 0.0 <= args.pose_shift_robust_weight <= 1.0:
        raise ValueError("pose shift-robust weight must be in [0, 1]")
    if not 0.0 <= args.root_shift_robust_weight <= 1.0:
        raise ValueError("root shift-robust weight must be in [0, 1]")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, loaders = make_loaders(args, device)
    _replace_validation_loaders(loaders, args)
    train, _, test = datasets
    checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    base = make_model(checkpoint, device)
    model = build_residual_hybrid(base, args.residual_decoder).to(device)
    if args.init_hybrid_checkpoint is not None:
        initial = torch.load(
            args.init_hybrid_checkpoint, map_location=device, weights_only=False
        )
        initial_decoder = initial.get("residual_decoder", "dense")
        if initial_decoder != args.residual_decoder:
            raise RuntimeError(
                "hybrid decoder mismatch: "
                f"checkpoint={initial_decoder}, requested={args.residual_decoder}"
            )
        initial_protocol = initial.get("protocol")
        if initial_protocol is not None and initial_protocol != args.exp:
            raise RuntimeError(
                "hybrid protocol mismatch: "
                f"checkpoint={initial_protocol}, requested={args.exp}"
            )
        model.load_state_dict(initial["model"])
    class_weight = _balanced_weight(train.index, "class_id", C.N_CLASSES, device)
    risk_weight = _balanced_weight(
        train.index, "risk_id", C.N_RISK, device, args.risk_danger_boost
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"

    model.set_calibration(0.0, 0.0, 0.0, 0.0)
    baseline_validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    baseline_classification = evaluate_classification(
        model, loaders["val_class"], device
    )
    # Root, pose trajectory, and logits are gated independently after training.
    # Pose-only runs must also preserve fall endpoints and realistic speed;
    # otherwise a low-MPJPE but visibly jittery checkpoint can win.
    initial_validation = baseline_validation
    initial_classification = baseline_classification
    initial_source = "identity_p2"
    if args.init_hybrid_checkpoint is not None:
        model.set_calibration(1.0, 1.0, 1.0, 1.0)
        initial_validation = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        initial_classification = evaluate_classification(
            model, loaders["val_class"], device
        )
        initial_source = "warm_start_full_strength"
    if args.objective == "root_only":
        best_score = root_selection_score(initial_validation)
    elif args.objective == "pose_only":
        best_score = _pose_checkpoint_score(
            initial_validation, args.pose_checkpoint_score
        )
    elif args.objective == "classification_only":
        best_score = classification_selection_score(initial_classification)
    else:
        best_score = float(initial_validation["mpjpe_m"])
    torch.save({
        "model": model.state_dict(),
        "epoch": 0,
        "initial_source": initial_source,
        "protocol": args.exp,
        "residual_decoder": args.residual_decoder,
        "objective": args.objective,
        "training_config": _training_metadata(args),
        "validation": initial_validation,
        "validation_classification": initial_classification,
    }, checkpoint_path)

    model.set_calibration(1.0, 1.0, 1.0, 1.0)
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if hasattr(train, "target") and hasattr(train.target, "set_epoch"):
            train.target.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(batch["csi"], batch["link_mask"])
                if args.objective == "pose_only":
                    loss, parts = pose_only_reconstruction_loss(
                        output, batch,
                        velocity_weight=args.pose_velocity_weight,
                        bone_weight=args.pose_bone_weight,
                        endpoint_weight=args.pose_endpoint_weight,
                        alignment_weight=args.alignment_weight,
                        max_shift=args.max_shift,
                        euclidean_weight=args.pose_euclidean_weight,
                        danger_distal_weight=args.pose_danger_distal_weight,
                        shift_robust_weight=args.pose_shift_robust_weight,
                        velocity_lags=tuple(args.pose_velocity_lags),
                    )
                elif args.objective == "root_only":
                    loss, parts = root_only_reconstruction_loss(
                        output, batch,
                        velocity_weight=args.root_velocity_weight,
                        displacement_weight=args.root_displacement_weight,
                        endpoint_weight=args.root_endpoint_weight,
                        velocity_lags=tuple(args.root_velocity_lags),
                        shift_robust_weight=args.root_shift_robust_weight,
                        max_shift=args.max_shift,
                    )
                elif args.objective == "classification_only":
                    loss, parts = classification_only_loss(
                        output, batch, class_weight, risk_weight,
                        args.lambda_class, args.lambda_risk,
                    )
                else:
                    loss, parts = trajectory_reconstruction_loss(
                        output, batch,
                        alignment_weight=args.alignment_weight,
                        max_shift=args.max_shift,
                        class_weight=class_weight,
                        risk_weight=risk_weight,
                        lambda_class=args.lambda_class,
                        lambda_risk=args.lambda_risk,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["class_id"])
            examples += count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * count

        train_metrics = {
            key: value / max(examples, 1) for key, value in totals.items()
        }
        validation = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        validation_classification = evaluate_classification(
            model, loaders["val_class"], device
        )
        if args.objective == "root_only":
            score = root_selection_score(validation)
        elif args.objective == "pose_only":
            score = _pose_checkpoint_score(
                validation, args.pose_checkpoint_score
            )
        elif args.objective == "classification_only":
            score = classification_selection_score(validation_classification)
        else:
            score = float(validation["mpjpe_m"])
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "selection_score": score,
            "validation": validation,
            "validation_classification": validation_classification,
        })
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "protocol": args.exp,
            "residual_decoder": args.residual_decoder,
            "objective": args.objective,
            "training_config": _training_metadata(args),
            "validation": validation,
            "validation_classification": validation_classification,
        }, args.run_dir / "last_model.pt")
        print(
            f"epoch={epoch:02d} loss={train_metrics['total']:.4f} "
            f"mpjpe={validation['mpjpe_m'] * 100:.2f}cm "
            f"danger={validation['danger_mpjpe_m'] * 100:.2f}cm "
            f"root={validation['root_error_m'] * 100:.2f}cm "
            f"class={validation_classification['class']['accuracy']:.3f} "
            f"danger-R={validation_classification['risk']['danger_recall']:.3f}"
        )
        if score < best_score:
            best_score = score
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "protocol": args.exp,
                "residual_decoder": args.residual_decoder,
                "objective": args.objective,
                "training_config": _training_metadata(args),
                "validation": validation,
                "validation_classification": validation_classification,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    selected_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(selected_checkpoint["model"])
    if args.skip_calibration:
        result = {
            "run": "p2_v9_hybrid_validation_only",
            "protocol": args.exp,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "source_p2": report_path(args.p2_checkpoint),
            "training_config": _training_metadata(args),
            "best_epoch": selected_checkpoint["epoch"],
            "selected_validation_full_strength": selected_checkpoint["validation"],
            "validation_classification_full_strength": selected_checkpoint[
                "validation_classification"
            ],
            "history": history,
        }
        (args.run_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    calibration = select_calibration(model, loaders, device, args)
    selected = {
        "pose_strength": calibration["pose"]["strength"],
        "root_strength": calibration["root"]["strength"],
        "class_strength": calibration["classification"]["strength"],
        "risk_strength": calibration["risk"]["strength"],
        "danger_logit_bias": calibration["risk"]["danger_logit_bias"],
    }
    model.set_calibration(
        selected["pose_strength"], selected["root_strength"],
        selected["class_strength"], selected["risk_strength"],
    )
    result = {
        "run": "p2_v9_hybrid_v10",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_p2": report_path(args.p2_checkpoint),
        "training_config": _training_metadata(args),
        "best_epoch": selected_checkpoint["epoch"],
        "selected": selected,
        "baseline_validation": baseline_validation,
        "baseline_validation_classification": baseline_classification,
        "selected_validation": calibration["root"]["validation"],
        "validation_classification": {
            "class": calibration["classification"]["validation"],
            "risk": calibration["risk"]["validation"],
        },
        "calibration_candidates": {
            "pose": calibration["pose_candidates"],
            "root": calibration["root_candidates"],
            "class": calibration["class_candidates"],
            "risk": calibration["risk_candidates"],
        },
        "history": history,
    }
    if not args.skip_test_eval:
        test_metrics = evaluate_trajectory(
            model, loaders["test"], device, args.max_shift
        )
        raw_test_classification = evaluate_classification(
            model, loaders["test_class"], device
        )
        test_classification = evaluate_classification(
            model, loaders["test_class"], device, selected["danger_logit_bias"]
        )
        result.update({
            "test": test_metrics,
            "raw_test_classification": raw_test_classification,
            "test_classification": test_classification,
            "shuffled_test": evaluate_model(
                model, ShuffledSignalDataset(test, args.seed),
                device, args.batch_size * 2, 5,
            ),
        })
    (args.run_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "source_p2": report_path(args.p2_checkpoint),
        "selected": selected,
        "residual_decoder": args.residual_decoder,
        "protocol": args.exp,
        "objective": args.objective,
        "training_config": _training_metadata(args),
        "validation": result["selected_validation"],
        **({
            "test": result["test"],
            "test_classification": result["test_classification"],
        } if "test" in result else {}),
    }, args.run_dir / "calibrated_model.pt")
    print(json.dumps({
        "selected": selected,
        "validation": result["selected_validation"],
        **({
            "test": result["test"],
            "test_classification": result["test_classification"],
        } if "test" in result else {}),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
