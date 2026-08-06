"""Train KP1-EXP01: dynamic-only CSI residuals over the frozen V13S pose."""

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
from ..dataio.dataset import build_datasets
from ..kinetic_pose import KineticPoseResidual
from ..quality import (
    QualityWeightedDataset,
    protocol_audit_path,
    quality_summary,
)
from ..trainer import set_seed
from .diagnose_observability import finite_mean, pose_only, report_path
from .evaluate_sealed import make_model, smooth_valid
from .evaluate_v12_final import _read_locked, build_locked_model


DISTAL_JOINTS = tuple(sorted(set(
    C.JOINT_GROUPS["head"]
    + C.JOINT_GROUPS["left_arm"][-1:]
    + C.JOINT_GROUPS["right_arm"][-1:]
    + C.JOINT_GROUPS["left_leg"][-2:]
    + C.JOINT_GROUPS["right_leg"][-2:]
)))


class CoarsePoseStore:
    """CPU store for deterministic outputs of the frozen V13S pose stack."""

    def __init__(self, rows: torch.Tensor, pose: torch.Tensor):
        self.rows = rows.long().cpu()
        self.pose = pose.cpu()
        self.position = {
            int(row): index for index, row in enumerate(self.rows.tolist())
        }

    def lookup(self, rows: torch.Tensor, device: str) -> torch.Tensor:
        positions = torch.tensor(
            [self.position[int(row)] for row in rows.tolist()], dtype=torch.long
        )
        return self.pose.index_select(0, positions).to(
            device=device, dtype=torch.float32, non_blocking=True
        )


@torch.no_grad()
def load_or_create_coarse_store(baseline, datasets: tuple,
                                path: Path, device: str,
                                batch_size: int, protocol: str) -> CoarsePoseStore:
    expected_rows = sorted({
        int(row) for dataset in datasets for row in dataset.rows.tolist()
    })
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("protocol") != protocol:
            raise RuntimeError(f"coarse cache protocol mismatch: {path}")
        if cached["rows"].tolist() != expected_rows:
            raise RuntimeError(f"coarse cache row set mismatch: {path}")
        return CoarsePoseStore(cached["rows"], cached["pose"])

    baseline.eval()
    by_row: dict[int, torch.Tensor] = {}
    for dataset in datasets:
        loader = DataLoader(
            dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0
        )
        for batch in loader:
            output = baseline(
                batch["csi"].to(device), batch["link_mask"].to(device)
            )
            pose = output["pose_rel"].detach().cpu().to(torch.float16)
            for row, value in zip(batch["row"].tolist(), pose):
                by_row[int(row)] = value
    missing = sorted(set(expected_rows) - set(by_row))
    if missing:
        raise RuntimeError(f"coarse cache is missing {len(missing)} rows")
    rows = torch.tensor(expected_rows, dtype=torch.long)
    pose = torch.stack([by_row[row] for row in expected_rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": protocol,
        "rows": rows,
        "pose": pose,
        "dtype": "float16",
        "source": "frozen_v13s",
    }, path)
    return CoarsePoseStore(rows, pose)


def relative_pose_speed(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    speed = pose.new_zeros(pose.shape[:2])
    if pose.shape[1] > 1:
        pair = valid[:, 1:] & valid[:, :-1]
        delta = torch.linalg.vector_norm(pose[:, 1:] - pose[:, :-1], dim=-1)
        speed[:, 1:] = delta.mean(-1) * C.TARGET_FPS * pair.to(pose.dtype)
    return speed


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def kinetic_pose_loss(output: dict, batch: dict, args) -> tuple[torch.Tensor, dict]:
    predicted = output["pose_rel"]
    target = batch["pose_rel"]
    valid = batch["valid"].bool()
    quality = batch["quality_weight"].to(predicted.device)
    risk = batch["risk_id"].to(predicted.device)
    speed = relative_pose_speed(target, valid)
    frame_weight = 1.0 + args.motion_weight * (speed / 0.50).clamp(0.0, 2.0)
    frame_weight = frame_weight * (
        1.0 + args.danger_frame_boost * (risk == 2).float()[:, None]
    )
    frame_weight = frame_weight * quality[:, None] * valid.to(predicted.dtype)

    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.05
    ).mean(-1)
    joint_weight = coordinate.new_ones(C.N_JOINTS)
    joint_weight[list(DISTAL_JOINTS)] = args.distal_joint_weight
    position = _weighted_mean(
        coordinate, frame_weight[..., None] * joint_weight
    )
    distal = _weighted_mean(
        coordinate[:, :, DISTAL_JOINTS], frame_weight[..., None]
    )

    velocity_terms = []
    for lag in (1, 3, 7):
        pair = valid[:, lag:] & valid[:, :-lag]
        pred_velocity = (
            predicted[:, lag:] - predicted[:, :-lag]
        ) * (C.TARGET_FPS / lag)
        target_velocity = (
            target[:, lag:] - target[:, :-lag]
        ) * (C.TARGET_FPS / lag)
        element = F.smooth_l1_loss(
            pred_velocity, target_velocity, reduction="none", beta=0.20
        ).mean((-1, -2))
        lag_weight = frame_weight[:, lag:] * pair.to(frame_weight.dtype)
        velocity_terms.append(_weighted_mean(element, lag_weight))
    velocity = sum(velocity_terms) / len(velocity_terms)
    target_joint_velocity = torch.zeros_like(target)
    target_joint_velocity[:, 1:] = (
        target[:, 1:] - target[:, :-1]
    ) * C.TARGET_FPS
    auxiliary_velocity_element = F.smooth_l1_loss(
        output["kinetic_velocity"], target_joint_velocity,
        reduction="none", beta=0.20,
    ).mean((-1, -2))
    auxiliary_velocity = _weighted_mean(
        auxiliary_velocity_element,
        frame_weight * valid.to(frame_weight.dtype),
    )

    acceleration_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    predicted_acceleration = (
        predicted[:, 2:] - 2.0 * predicted[:, 1:-1] + predicted[:, :-2]
    ) * (C.TARGET_FPS ** 2)
    target_acceleration = (
        target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    ) * (C.TARGET_FPS ** 2)
    acceleration_element = F.smooth_l1_loss(
        predicted_acceleration, target_acceleration,
        reduction="none", beta=1.0,
    ).mean((-1, -2))
    acceleration = _weighted_mean(
        acceleration_element,
        frame_weight[:, 2:] * acceleration_mask.to(frame_weight.dtype),
    )

    edges = torch.as_tensor(C.SKELETON_EDGES, device=predicted.device)
    pred_bones = predicted[:, :, edges[:, 1]] - predicted[:, :, edges[:, 0]]
    target_bones = target[:, :, edges[:, 1]] - target[:, :, edges[:, 0]]
    pred_length = torch.linalg.vector_norm(pred_bones, dim=-1)
    target_length = torch.linalg.vector_norm(target_bones, dim=-1)
    bone_length = _weighted_mean(
        F.smooth_l1_loss(
            pred_length, target_length, reduction="none", beta=0.02
        ), frame_weight[..., None]
    )
    cosine = (
        F.normalize(pred_bones, dim=-1) * F.normalize(target_bones, dim=-1)
    ).sum(-1).clamp(-1.0, 1.0)
    bone_direction = _weighted_mean(1.0 - cosine, frame_weight[..., None])

    static = valid & (speed < 0.08)
    residual_size = torch.linalg.vector_norm(output["pose_delta"], dim=-1).mean(-1)
    static_residual = _weighted_mean(
        residual_size, static.to(residual_size.dtype) * quality[:, None]
    )
    endpoint_mask = torch.zeros_like(valid)
    for item in range(len(valid)):
        if int(risk[item]) != 2:
            continue
        frames = torch.nonzero(valid[item], as_tuple=False).flatten()
        endpoint_mask[item, frames[-15:]] = True
    endpoint = _weighted_mean(
        coordinate,
        endpoint_mask.to(coordinate.dtype)[:, :, None]
        * quality[:, None, None],
    )
    total = (
        position
        + args.lambda_distal * distal
        + args.lambda_velocity * velocity
        + args.lambda_aux_velocity * auxiliary_velocity
        + args.lambda_acceleration * acceleration
        + args.lambda_bone_length * bone_length
        + args.lambda_bone_direction * bone_direction
        + args.lambda_static * static_residual
        + args.lambda_endpoint * endpoint
    )
    return total, {
        "total": float(total.detach()),
        "position": float(position.detach()),
        "distal": float(distal.detach()),
        "velocity": float(velocity.detach()),
        "auxiliary_velocity": float(auxiliary_velocity.detach()),
        "acceleration": float(acceleration.detach()),
        "bone_length": float(bone_length.detach()),
        "bone_direction": float(bone_direction.detach()),
        "static_residual": float(static_residual.detach()),
        "danger_endpoint": float(endpoint.detach()),
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if len(left) < 3:
        return math.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator) if denominator > 1e-8 else math.nan


def _pose_rows(predicted: torch.Tensor, batch: dict) -> list[dict]:
    valid = batch["valid"].bool()
    predicted = smooth_valid(predicted.float().cpu(), valid, 5)
    target = batch["pose_rel"].float()
    rows = []
    head = C.JOINT_INDEX["head"]
    for item in range(len(predicted)):
        mask = valid[item]
        frames = torch.nonzero(mask, as_tuple=False).flatten()
        if len(frames) == 0:
            continue
        error = torch.linalg.vector_norm(predicted[item] - target[item], dim=-1)
        pair = mask[1:] & mask[:-1]
        pred_speed = torch.linalg.vector_norm(
            predicted[item, 1:] - predicted[item, :-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        target_speed = torch.linalg.vector_norm(
            target[item, 1:] - target[item, :-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        dynamic = pair & (target_speed > 0.25)
        high_motion = torch.zeros_like(pair)
        if pair.any():
            threshold = torch.quantile(target_speed[pair], 0.75)
            high_motion = pair & (target_speed >= threshold) & (target_speed > 0.08)
        predicted_motion = float(pred_speed[pair].mean()) if pair.any() else math.nan
        target_motion = float(target_speed[pair].mean()) if pair.any() else math.nan
        danger = int(batch["risk_id"][item]) == 2
        torso_angle = math.nan
        if danger:
            predicted_head = F.normalize(predicted[item, :, head], dim=-1)
            target_head = F.normalize(target[item, :, head], dim=-1)
            cosine = (predicted_head * target_head).sum(-1).clamp(-1.0, 1.0)
            torso_angle = float(torch.rad2deg(torch.acos(cosine[mask])).mean())
        endpoint = frames[-15:]
        rows.append({
            "mpjpe_m": float(error[mask].mean()),
            "distal_mpjpe_m": float(error[mask][:, DISTAL_JOINTS].mean()),
            "dynamic_mpjpe_m": (
                float(error[1:][dynamic].mean()) if dynamic.any() else math.nan
            ),
            "high_motion_mpjpe_m": (
                float(error[1:][high_motion].mean()) if high_motion.any() else math.nan
            ),
            "pose_speed_ratio": (
                predicted_motion / max(target_motion, 1e-6) if pair.any() else math.nan
            ),
            "speed_correlation": _correlation(pred_speed[pair], target_speed[pair]),
            "danger_pose_mpjpe_m": float(error[mask].mean()) if danger else math.nan,
            "danger_distal_mpjpe_m": (
                float(error[mask][:, DISTAL_JOINTS].mean()) if danger else math.nan
            ),
            "danger_endpoint_mpjpe_m": (
                float(error[endpoint].mean()) if danger else math.nan
            ),
            "danger_high_motion_mpjpe_m": (
                float(error[1:][high_motion].mean())
                if danger and high_motion.any() else math.nan
            ),
            "danger_torso_angle_deg": torso_angle,
            "danger_speed_correlation": (
                _correlation(pred_speed[pair], target_speed[pair]) if danger else math.nan
            ),
            "danger_trial": float(danger),
        })
    return rows


def _aggregate_rows(rows: list[dict]) -> dict:
    metrics = {
        key: finite_mean([row[key] for row in rows])
        for key in rows[0] if key != "danger_trial"
    }
    metrics["trials"] = len(rows)
    metrics["danger_trials"] = int(sum(row["danger_trial"] for row in rows))
    return metrics


@torch.no_grad()
def evaluate_strengths(model: KineticPoseResidual, loader: DataLoader,
                       strengths: list[float], device: str,
                       coarse_store: CoarsePoseStore) -> dict[float, dict]:
    model.eval()
    model.set_residual_strength(1.0)
    rows = {float(strength): [] for strength in strengths}
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            coarse_pose=coarse_store.lookup(batch["row"], device),
        )
        coarse = output["pose_coarse"].float().cpu()
        delta = output["pose_delta"].float().cpu()
        for strength in rows:
            rows[strength].extend(_pose_rows(coarse + strength * delta, batch))
    return {strength: _aggregate_rows(items) for strength, items in rows.items()}


def pose_selection_score(metrics: dict) -> float:
    speed_ratio = max(float(metrics["pose_speed_ratio"]), 1e-4)
    return (
        float(metrics["mpjpe_m"])
        + 0.50 * float(metrics["distal_mpjpe_m"])
        + 0.75 * float(metrics["dynamic_mpjpe_m"])
        + 0.75 * float(metrics["high_motion_mpjpe_m"])
        + 1.25 * float(metrics["danger_pose_mpjpe_m"])
        + 0.75 * float(metrics["danger_distal_mpjpe_m"])
        + 0.50 * float(metrics["danger_endpoint_mpjpe_m"])
        + 0.25 * float(metrics["danger_high_motion_mpjpe_m"])
        + 0.002 * float(metrics["danger_torso_angle_deg"])
        + 0.03 * abs(math.log(speed_ratio))
    )


def make_loaders(args, device: str):
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
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
    weights = weights * torch.where(
        danger, torch.tensor(args.danger_weight, dtype=weights.dtype),
        torch.tensor(1.0, dtype=weights.dtype),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return (train, validation, test), {
        "train": DataLoader(
            train, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=device == "cuda",
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


def train_epoch(model: KineticPoseResidual, loader: DataLoader,
                optimizer: torch.optim.Optimizer, scaler, device: str,
                args, coarse_store: CoarsePoseStore) -> dict:
    model.train()
    model.set_residual_strength(1.0)
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
                coarse_pose=coarse_store.lookup(batch["row"].cpu(), device),
            )
            loss, parts = kinetic_pose_loss(output, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        for key, value in parts.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def build_components(args, device: str):
    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    baseline, baseline_config = build_locked_model(
        args, device, root_lock, class_lock
    )
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    p2_model = make_model(p2_checkpoint, device)
    normalizer = copy.deepcopy(p2_model.norm).cpu()
    del p2_model
    return baseline, normalizer, baseline_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
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
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument(
        "--condition-on-coarse", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--activity-floor", type=float, default=0.0)
    parser.add_argument("--experiment-name", default="KP1-EXP02")
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_exp02_signal_gated",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, loaders = make_loaders(args, device)
    train, validation, test = datasets
    baseline, normalizer, baseline_config = build_components(args, device)
    coarse_store = load_or_create_coarse_store(
        baseline, datasets, args.coarse_cache, device, args.batch_size, args.exp
    )
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()
    model = KineticPoseResidual(
        None, normalizer, hidden=args.hidden,
        temporal_layers=args.temporal_layers, heads=args.heads,
        dropout=args.dropout, max_delta=args.max_delta,
        condition_on_coarse=args.condition_on_coarse,
        activity_floor=args.activity_floor,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    best = {"score": math.inf, "epoch": 0, "strength": 0.0, "state": None}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, scaler, device, args, coarse_store
        )
        validation_candidates = evaluate_strengths(
            model, loaders["val"], list(args.strengths), device, coarse_store
        )
        scores = {
            strength: pose_selection_score(metrics)
            for strength, metrics in validation_candidates.items()
        }
        selected_strength = min(scores, key=scores.get)
        selected_score = scores[selected_strength]
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation_score": selected_score,
            "validation_strength": selected_strength,
            "validation": validation_candidates[selected_strength],
        })
        print(json.dumps(history[-1], ensure_ascii=False))
        if selected_score < best["score"] - 1e-5:
            best = {
                "score": selected_score,
                "epoch": epoch,
                "strength": selected_strength,
                "state": copy.deepcopy(model.trainable_state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best["state"] is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_trainable_state_dict(best["state"])
    selected_strength = float(best["strength"])
    validation_metrics = evaluate_strengths(
        model, loaders["val"], [0.0, selected_strength], device, coarse_store
    )
    test_metrics = evaluate_strengths(
        model, loaders["test"], [0.0, selected_strength], device, coarse_store
    )
    result = {
        "run": args.experiment_name,
        "model_family": "NotiFi-KineticPose",
        "candidate_version": "KP-1",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "objective": "pelvis_relative_pose_only",
        "device": device,
        "seed": args.seed,
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
            "quality": quality_summary(train),
        },
        "architecture": {
            "dynamic_inputs": ["delta_1", "delta_3", "delta_7", "high_pass_15"],
            "static_csi_visible_to_residual": False,
            "baseline_frozen": True,
            "root_trained": False,
            "classification_trained": False,
            "hidden": args.hidden,
            "temporal_layers": args.temporal_layers,
            "max_delta_m": args.max_delta,
            "condition_on_coarse": args.condition_on_coarse,
            "activity_floor": args.activity_floor,
        },
        "baseline_configuration": baseline_config,
        "coarse_pose_cache": report_path(args.coarse_cache),
        "selection": {
            "epoch": int(best["epoch"]),
            "score": float(best["score"]),
            "residual_strength": selected_strength,
        },
        "validation": {
            "v13s_strength_0": validation_metrics[0.0],
            "kp1_selected": validation_metrics[selected_strength],
        },
        "test": {
            "v13s_strength_0": test_metrics[0.0],
            "kp1_selected": test_metrics[selected_strength],
        },
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report = args.run_dir / "result.json"
    report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": args.experiment_name,
        "protocol": args.exp,
        "trainable_model": best["state"],
        "residual_strength": selected_strength,
        "architecture": result["architecture"],
        "source": {
            "p2_checkpoint": report_path(args.p2_checkpoint),
            "root_calibration": report_path(args.root_calibration),
            "classification_calibration": report_path(args.classification_calibration),
        },
        "selection": result["selection"],
        "validation": result["validation"],
        "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
