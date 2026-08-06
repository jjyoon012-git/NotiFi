"""Train KP11-DYNAMIC-MOTION without opening the fixed test by default."""

from __future__ import annotations

import argparse
import copy
import json
import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..dynamic_motion import DISTAL_JOINTS, DynamicMotionPoseNet
from ..motion_tokens import pose_to_bones, trial_bone_lengths
from ..quality import QualityWeightedDataset, protocol_audit_path, quality_summary
from ..seen_v2 import injury_targets
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .train_conditioned_contact_pose import fall_phase_targets
from .train_hierarchical_pose import banded_velocity_alignment
from .train_kinetic_pose import (
    _aggregate_rows,
    _pose_rows,
    _weighted_mean,
    pose_selection_score,
    relative_pose_speed,
)
from .audit_kp10_paired_bootstrap import kp10_prediction
from .calibrate_part_motion_profile_reranking import prepare as prepare_kp10


KP10_VALIDATION = {
    "score": 0.969946978528323,
    "mpjpe_m": 0.11206,
    "danger_pose_mpjpe_m": 0.15682,
    "danger_distal_mpjpe_m": 0.22898,
}


class PoseAnchorDataset(Dataset):
    """Attach an immutable CSI-only pose anchor without exposing GT to inference."""

    def __init__(self, source: QualityWeightedDataset, anchors: torch.Tensor):
        if len(source) != len(anchors):
            raise ValueError("pose anchor count does not match dataset")
        self.source = source
        self.target = source.target
        self.index = source.index
        self.rows = source.rows
        self.weights = source.weights
        self.anchors = anchors.float().cpu()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        sample["pose_anchor"] = self.anchors[index]
        return sample

    def sampler_weights(self):
        return self.source.sampler_weights()


def _kp10_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        exp=args.exp,
        selector_checkpoint=args.kp10_selector_checkpoint,
        reranker_checkpoint=args.kp10_reranker_checkpoint,
        scalar_profile_checkpoint=args.kp10_scalar_profile_checkpoint,
        part_profile_checkpoint=args.kp10_part_profile_checkpoint,
        adaptive_calibration=args.kp10_adaptive_calibration,
        classifier_checkpoint=args.kp10_classifier_checkpoint,
        profile_ranker_checkpoint=args.kp10_profile_ranker_checkpoint,
        strength_calibration=args.kp10_strength_calibration,
        candidate_action_penalty=0.05,
    )


@torch.no_grad()
def attach_kp10_anchors(dataset: QualityWeightedDataset, split: str,
                        args, device: str) -> PoseAnchorDataset:
    pose_source = QualityWeightedDataset(
        pose_only(dataset.target), protocol_audit_path(args.exp)
    )
    trial_ids = pose_source.index.trial_id.astype(str).tolist()
    cache_path = args.anchor_cache_dir / f"{args.exp}_{split}_kp10.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("trial_ids") != trial_ids:
            raise RuntimeError(f"stale KP10 anchor order in {cache_path}")
        anchors = cached["pose_rel"].float()
    else:
        data = prepare_kp10(_kp10_args(args), split, device)
        anchors = kp10_prediction(data, _kp10_args(args), device).float().cpu()
        if len(anchors) != len(pose_source):
            raise RuntimeError("KP10 anchor generation changed dataset cardinality")
        source_pose, source_valid, _, _ = _load_pose_arrays(pose_source)
        if not torch.equal(data["target_valid"].bool(), source_valid.bool()):
            raise RuntimeError("KP10 anchor order failed valid-mask audit")
        if not torch.allclose(data["target_pose"], source_pose, atol=1e-6, rtol=0):
            raise RuntimeError("KP10 anchor order failed pose-row audit")
        args.anchor_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "protocol": args.exp,
            "split": split,
            "inference_inputs": ["CSI", "link_mask", "train_only_motion_bank"],
            "test_target_used": False,
            "trial_ids": trial_ids,
            "pose_rel": anchors,
        }, cache_path)
    anchor_by_trial = dict(zip(trial_ids, anchors))
    full_anchors = torch.zeros(
        len(dataset), C.CACHE_FRAMES, C.N_JOINTS, 3, dtype=torch.float32
    )
    for index, trial_id in enumerate(dataset.index.trial_id.astype(str)):
        if trial_id in anchor_by_trial:
            full_anchors[index] = anchor_by_trial[trial_id]
    return PoseAnchorDataset(dataset, full_anchors)


def build_pose_priors(dataset) -> tuple[torch.Tensor, torch.Tensor, dict]:
    pose, valid, _, _ = _load_pose_arrays(dataset)
    directions, _ = pose_to_bones(pose)
    weight = valid[..., None, None].to(pose.dtype)
    prior = (directions * weight).sum((0, 1))
    prior = F.normalize(prior, dim=-1)
    prior[C.ROOT_JOINT] = 0.0
    if not torch.isfinite(prior).all():
        raise RuntimeError("non-finite train direction prior")
    lengths = trial_bone_lengths(pose, valid)
    length_prior = lengths.median(0).values
    length_prior[C.ROOT_JOINT] = 0.0
    return prior, length_prior, {
        "source": "train_only_pose_frames",
        "trials": len(pose),
        "bone_length_mean_m": float(length_prior[1:].mean()),
        "bone_length_min_m": float(length_prior[1:].min()),
        "bone_length_max_m": float(length_prior[1:].max()),
    }


def _class_weights(labels: torch.Tensor, classes: int,
                   device: str) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=classes).float().clamp_min(1)
    weights = counts.sum() / (counts * classes)
    return weights.to(device)


def make_loaders(args, device: str):
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(datasets["train"], audit)
    validation = QualityWeightedDataset(datasets["val"], audit)
    test = QualityWeightedDataset(datasets["test"], audit)
    train_pose = QualityWeightedDataset(pose_only(datasets["train"]), audit)

    weights = train.sampler_weights()
    risk = torch.tensor(train.index.risk_id.to_numpy(dtype=np.int64))
    action = torch.tensor(train.index.class_id.to_numpy(dtype=np.int64))
    weights *= torch.where(
        risk == 2,
        torch.tensor(args.danger_sample_weight, dtype=weights.dtype),
        torch.tensor(1.0, dtype=weights.dtype),
    )
    weights *= torch.where(
        action == 6,
        torch.tensor(args.absence_sample_weight, dtype=weights.dtype),
        torch.tensor(1.0, dtype=weights.dtype),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.pose_anchor == "kp10":
        train = attach_kp10_anchors(train, "train", args, device)
        validation = attach_kp10_anchors(validation, "val", args, device)
    loaders = {
        "train": DataLoader(
            train, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=device == "cuda",
        ),
        "val": DataLoader(
            validation, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=device == "cuda",
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=device == "cuda",
        ),
    }
    return (train, validation, test, train_pose), loaders


def motion_profiles(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    velocity = torch.zeros_like(pose)
    velocity[:, 1:] = (pose[:, 1:] - pose[:, :-1]) * C.TARGET_FPS
    joint_speed = torch.linalg.vector_norm(velocity, dim=-1)
    profiles = [joint_speed.mean(-1)]
    profiles.extend(
        joint_speed[:, :, list(joints)].mean(-1)
        for joints in C.JOINT_GROUPS.values()
    )
    result = torch.stack(profiles, dim=-1)
    return result * valid[..., None].to(result.dtype)


def timestamp_alignment(predicted: torch.Tensor, target: torch.Tensor,
                        valid: torch.Tensor, risk: torch.Tensor,
                        exact: torch.Tensor) -> torch.Tensor:
    speed = relative_pose_speed(target, valid)
    frame_weight = 1.0 + 2.0 * (speed / 0.50).clamp(0.0, 2.0)
    frame_weight *= 1.0 + 0.75 * (risk == 2).float()[:, None]
    terms, counts = [], []
    for selected, radius in ((exact.bool(), 1), (~exact.bool(), 3)):
        if selected.any():
            terms.append(banded_velocity_alignment(
                predicted[selected], target[selected], valid[selected],
                frame_weight[selected], radius=radius,
                temperature=0.05, lag_penalty=0.004,
            ))
            counts.append(float(selected.sum()))
    if not terms:
        return predicted.new_zeros(())
    return sum(term * count for term, count in zip(terms, counts)) / sum(counts)


def dynamic_motion_loss(output: dict, batch: dict, args,
                        class_weight: torch.Tensor,
                        risk_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    predicted = output["pose_rel"]
    target = batch["pose_rel"]
    valid = batch["valid"].bool()
    risk = batch["risk_id"].long()
    quality = batch["quality_weight"].to(predicted.dtype)
    speed = relative_pose_speed(target, valid)
    frame_weight = 1.0 + args.motion_weight * (speed / 0.50).clamp(0.0, 2.0)
    frame_weight *= 1.0 + args.danger_frame_boost * (risk == 2).float()[:, None]
    frame_weight *= quality[:, None] * valid.to(predicted.dtype)

    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.04
    ).mean(-1)
    joint_weight = coordinate.new_ones(C.N_JOINTS)
    joint_weight[list(DISTAL_JOINTS)] = args.distal_joint_weight
    position = _weighted_mean(
        coordinate, frame_weight[..., None] * joint_weight
    )
    distal = _weighted_mean(
        coordinate[:, :, list(DISTAL_JOINTS)], frame_weight[..., None]
    )

    target_direction, _ = pose_to_bones(target)
    direction_error = 1.0 - (
        output["bone_direction"] * target_direction
    ).sum(-1).clamp(-1.0, 1.0)
    direction = _weighted_mean(direction_error[:, :, 1:], frame_weight[..., None])

    target_lengths = trial_bone_lengths(target, valid)
    shape_mask = valid.any(1).to(predicted.dtype)
    shape = _weighted_mean(
        F.smooth_l1_loss(
            output["bone_lengths"][:, 1:], target_lengths[:, 1:],
            reduction="none", beta=0.02,
        ), shape_mask[:, None] * quality[:, None],
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
    target_joint_velocity = torch.zeros_like(target)
    target_joint_velocity[:, 1:] = (
        target[:, 1:] - target[:, :-1]
    ) * C.TARGET_FPS
    auxiliary_velocity = _weighted_mean(
        F.smooth_l1_loss(
            output["kinetic_velocity"], target_joint_velocity,
            reduction="none", beta=0.20,
        ).mean((-1, -2)), frame_weight,
    )

    acceleration_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    predicted_acceleration = (
        predicted[:, 2:] - 2.0 * predicted[:, 1:-1] + predicted[:, :-2]
    ) * (C.TARGET_FPS ** 2)
    target_acceleration = (
        target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    ) * (C.TARGET_FPS ** 2)
    acceleration = _weighted_mean(
        F.smooth_l1_loss(
            predicted_acceleration, target_acceleration,
            reduction="none", beta=1.0,
        ).mean((-1, -2)),
        frame_weight[:, 2:] * acceleration_mask.to(frame_weight.dtype),
    )
    alignment = timestamp_alignment(
        predicted, target, valid, risk, batch["timestamp_exact"].bool()
    )

    endpoint_mask = torch.zeros_like(valid)
    for item in range(len(valid)):
        if int(risk[item]) != 2:
            continue
        frames = torch.nonzero(valid[item], as_tuple=False).flatten()
        endpoint_mask[item, frames[-15:]] = True
    endpoint = _weighted_mean(
        coordinate,
        endpoint_mask[..., None].to(coordinate.dtype) * quality[:, None, None],
    )

    phase_target, phase_mask = fall_phase_targets(
        target, batch["root"], valid, risk
    )
    if phase_mask.any():
        phase = F.cross_entropy(
            output["phase_logits"][phase_mask], phase_target[phase_mask],
            weight=predicted.new_tensor((0.25, 1.0, 2.0, 1.0)),
        )
    else:
        phase = predicted.new_zeros(())

    contact_target = injury_targets(
        target, batch["root"], valid, risk
    )["injury_contact"].to(predicted.dtype)
    contact_mask = valid[..., None].expand_as(contact_target)
    positives = contact_target[contact_mask].sum()
    negatives = contact_mask.sum() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 15.0)
    contact = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output["contact_logits"], contact_target,
            reduction="none", pos_weight=positive_weight,
        ), contact_mask.to(predicted.dtype),
    )
    profile = _weighted_mean(
        F.smooth_l1_loss(
            output["motion_profile"], motion_profiles(target, valid),
            reduction="none", beta=0.15,
        ), valid[..., None].to(predicted.dtype),
    )
    if "pose_anchor" in batch:
        anchor = _weighted_mean(
            F.smooth_l1_loss(
                predicted, batch["pose_anchor"], reduction="none", beta=0.04
            ).mean(-1),
            valid[..., None].to(predicted.dtype) * quality[:, None, None],
        )
    else:
        anchor = predicted.new_zeros(())

    action = F.cross_entropy(
        output["class_logits"], batch["class_id"].long(),
        weight=class_weight, label_smoothing=0.02,
    )
    risk_class = F.cross_entropy(
        output["risk_logits"], risk,
        weight=risk_weight, label_smoothing=0.01,
    )
    action_probability = torch.softmax(output["class_logits"], dim=-1)
    action_risk = torch.stack((
        action_probability[:, :9].sum(-1),
        action_probability[:, 9:12].sum(-1),
        action_probability[:, 12:].sum(-1),
    ), dim=-1)
    hierarchy = F.nll_loss(
        action_risk.clamp_min(1e-8).log(), risk, weight=risk_weight
    )

    total = (
        position
        + args.lambda_distal * distal
        + args.lambda_direction * direction
        + args.lambda_shape * shape
        + args.lambda_velocity * velocity
        + args.lambda_aux_velocity * auxiliary_velocity
        + args.lambda_acceleration * acceleration
        + args.lambda_alignment * alignment
        + args.lambda_endpoint * endpoint
        + args.lambda_phase * phase
        + args.lambda_contact * contact
        + args.lambda_profile * profile
        + args.lambda_anchor * anchor
        + args.lambda_action * action
        + args.lambda_risk * risk_class
        + args.lambda_hierarchy * hierarchy
    )
    return total, {
        "total": float(total.detach()),
        "position": float(position.detach()),
        "distal": float(distal.detach()),
        "direction": float(direction.detach()),
        "shape": float(shape.detach()),
        "velocity": float(velocity.detach()),
        "aux_velocity": float(auxiliary_velocity.detach()),
        "acceleration": float(acceleration.detach()),
        "alignment": float(alignment.detach()),
        "endpoint": float(endpoint.detach()),
        "phase": float(phase.detach()),
        "contact": float(contact.detach()),
        "profile": float(profile.detach()),
        "anchor": float(anchor.detach()),
        "action": float(action.detach()),
        "risk": float(risk_class.detach()),
        "hierarchy": float(hierarchy.detach()),
    }


def _macro_f1(predicted: torch.Tensor, target: torch.Tensor,
              classes: int) -> float:
    values = []
    for class_id in range(classes):
        tp = ((predicted == class_id) & (target == class_id)).sum().float()
        fp = ((predicted == class_id) & (target != class_id)).sum().float()
        fn = ((predicted != class_id) & (target == class_id)).sum().float()
        values.append(2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0))
    return float(torch.stack(values).mean())


def classification_metrics(action_logits: torch.Tensor,
                           risk_logits: torch.Tensor,
                           action: torch.Tensor, risk: torch.Tensor) -> dict:
    action_prediction = action_logits.argmax(-1)
    risk_prediction = risk_logits.argmax(-1)
    danger = risk == 2
    safe = risk == 0
    return {
        "action_accuracy": float((action_prediction == action).float().mean()),
        "action_macro_f1": _macro_f1(action_prediction, action, C.N_CLASSES),
        "risk_accuracy": float((risk_prediction == risk).float().mean()),
        "risk_macro_f1": _macro_f1(risk_prediction, risk, C.N_RISK),
        "danger_recall": float((risk_prediction[danger] == 2).float().mean()),
        "danger_correct": int((risk_prediction[danger] == 2).sum()),
        "danger_total": int(danger.sum()),
        "danger_action_accuracy": float(
            (action_prediction[danger] == action[danger]).float().mean()
        ),
        "safe_to_danger": int((safe & (risk_prediction == 2)).sum()),
        "safe_total": int(safe.sum()),
        "trials": len(action),
    }


@torch.no_grad()
def evaluate(model: DynamicMotionPoseNet, loader: DataLoader,
             device: str) -> dict:
    model.eval()
    pose_rows, action_logits, risk_logits, actions, risks = [], [], [], [], []
    gate_values = []
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            batch.get("pose_anchor", None).to(device)
            if "pose_anchor" in batch else None,
        )
        pose_rows.extend(_pose_rows(output["pose_rel"].float().cpu(), batch))
        action_logits.append(output["class_logits"].float().cpu())
        risk_logits.append(output["risk_logits"].float().cpu())
        actions.append(batch["class_id"].long())
        risks.append(batch["risk_id"].long())
        mask = batch["link_mask"].any(-1).to(device)
        gate_values.append(output["fusion_gate"][mask].float().cpu())
    pose = _aggregate_rows(pose_rows)
    classification = classification_metrics(
        torch.cat(action_logits), torch.cat(risk_logits),
        torch.cat(actions), torch.cat(risks),
    )
    pose["fusion_static_gate_mean"] = float(torch.cat(gate_values).mean())
    return {"pose": pose, "classification": classification}


def selection_score(result: dict) -> float:
    pose = result["pose"]
    classification = result["classification"]
    return (
        pose_selection_score(pose)
        + 0.15 * (1.0 - classification["action_accuracy"])
        + 0.20 * (1.0 - classification["risk_accuracy"])
        + 0.30 * (1.0 - classification["danger_recall"])
    )


def promotion_audit(validation: dict) -> dict:
    pose = validation["pose"]
    classification = validation["classification"]
    checks = {
        "overall_not_worse": pose["mpjpe_m"] <= KP10_VALIDATION["mpjpe_m"] + 0.003,
        "danger_pose_better": (
            pose["danger_pose_mpjpe_m"] < KP10_VALIDATION["danger_pose_mpjpe_m"]
        ),
        "danger_distal_better": (
            pose["danger_distal_mpjpe_m"]
            < KP10_VALIDATION["danger_distal_mpjpe_m"]
        ),
        "speed_correlation": pose["speed_correlation"] >= 0.55,
        "action_accuracy": classification["action_accuracy"] >= 0.90,
        "risk_accuracy": classification["risk_accuracy"] >= 0.95,
        "danger_recall": classification["danger_recall"] >= 0.93,
    }
    return {"passed": all(checks.values()), "checks": checks}


@torch.no_grad()
def calibrate_danger_bias(model, loader, device: str) -> dict:
    model.eval()
    logits, labels = [], []
    model.set_danger_logit_bias(0.0)
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            batch.get("pose_anchor", None).to(device)
            if "pose_anchor" in batch else None,
        )
        logits.append(output["risk_logits"].float().cpu())
        labels.append(batch["risk_id"].long())
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    candidates = []
    dummy_action = torch.zeros(len(labels), C.N_CLASSES)
    dummy_action[:, 0] = 1.0
    for bias in torch.linspace(0.0, 1.0, 41).tolist():
        adjusted = logits.clone()
        adjusted[:, 2] += bias
        metrics = classification_metrics(dummy_action, adjusted, labels * 0, labels)
        if metrics["risk_accuracy"] >= 0.96 and metrics["safe_to_danger"] <= 4:
            candidates.append((
                metrics["danger_recall"], metrics["risk_accuracy"], -bias,
                bias, metrics,
            ))
    selected = max(candidates) if candidates else (
        0.0, 0.0, 0.0, 0.0,
        classification_metrics(dummy_action, logits, labels * 0, labels),
    )
    model.set_danger_logit_bias(selected[3])
    return {"bias": selected[3], "metrics": selected[4]}


def train_epoch(model, loader, optimizer, scaler, scheduler, device,
                args, class_weight, risk_weight):
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
                batch["csi"], batch["link_mask"], batch.get("pose_anchor")
            )
            loss, parts = dynamic_motion_loss(
                output, batch, args, class_weight, risk_weight
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for key, value in parts.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune"
        / "best_model.pt",
    )
    parser.add_argument("--pose-anchor", choices=("p2", "kp10"), default="kp10")
    parser.add_argument(
        "--anchor-cache-dir", type=Path,
        default=C.PROJECT_ROOT / "work_v2" / "kp11_anchor_cache",
    )
    parser.add_argument(
        "--kp10-selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--kp10-reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--kp10-scalar-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed79" / "best_model.pt",
    )
    parser.add_argument(
        "--kp10-part-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed101" / "best_model.pt",
    )
    parser.add_argument(
        "--kp10-adaptive-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend" / "calibration.json",
    )
    parser.add_argument(
        "--kp10-classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--kp10-profile-ranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker_seed127"
        / "best_model.pt",
    )
    parser.add_argument(
        "--kp10-strength-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength" / "calibration.json",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--classification-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--danger-sample-weight", type=float, default=3.0)
    parser.add_argument("--absence-sample-weight", type=float, default=2.0)
    parser.add_argument("--danger-frame-boost", type=float, default=0.75)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--distal-joint-weight", type=float, default=2.0)
    parser.add_argument("--lambda-distal", type=float, default=0.50)
    parser.add_argument("--lambda-direction", type=float, default=0.15)
    parser.add_argument("--lambda-shape", type=float, default=0.05)
    parser.add_argument("--lambda-velocity", type=float, default=0.30)
    parser.add_argument("--lambda-aux-velocity", type=float, default=0.05)
    parser.add_argument("--lambda-acceleration", type=float, default=0.005)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    parser.add_argument("--lambda-endpoint", type=float, default=0.35)
    parser.add_argument("--lambda-phase", type=float, default=0.03)
    parser.add_argument("--lambda-contact", type=float, default=0.03)
    parser.add_argument("--lambda-profile", type=float, default=0.03)
    parser.add_argument("--lambda-anchor", type=float, default=0.20)
    parser.add_argument("--lambda-action", type=float, default=0.0)
    parser.add_argument("--lambda-risk", type=float, default=0.0)
    parser.add_argument("--lambda-hierarchy", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dynamic-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--endpoint-scale", type=float, default=0.05)
    parser.add_argument("--shape-scale", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp11_dynamic_motion_seed311",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, loaders = make_loaders(args, device)
    train, validation, test, train_pose = datasets
    prior, lengths, prior_audit = build_pose_priors(train_pose)
    checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    base = make_model(checkpoint, device)
    model = DynamicMotionPoseNet(
        base, prior, lengths, hidden=args.hidden,
        dynamic_layers=args.dynamic_layers, heads=args.heads,
        dropout=args.dropout, endpoint_scale=args.endpoint_scale,
        shape_scale=args.shape_scale,
    ).to(device)

    action_labels = torch.tensor(train.index.class_id.to_numpy(dtype=np.int64))
    risk_labels = torch.tensor(train.index.risk_id.to_numpy(dtype=np.int64))
    class_weight = _class_weights(action_labels, C.N_CLASSES, device)
    risk_weight = _class_weights(risk_labels, C.N_RISK, device)
    risk_weight[2] *= 1.5
    optimizer = torch.optim.AdamW([
        {"params": list(model.pose_parameters()), "lr": args.learning_rate},
        {
            "params": list(model.classification_parameters()),
            "lr": args.classification_learning_rate,
        },
    ], weight_decay=args.weight_decay)
    steps_per_epoch = (
        args.max_train_batches or math.ceil(len(train) / args.batch_size)
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    def schedule(step):
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 0.05)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history, stale = [], 0
    best = {"score": math.inf, "epoch": 0, "state": None, "validation": None}
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            model, loaders["train"], optimizer, scaler, scheduler,
            device, args, class_weight, risk_weight,
        )
        current = evaluate(model, loaders["val"], device)
        score = selection_score(current)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": training,
            "validation_score": score,
            "validation": current,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score < best["score"] - 1e-5:
            best = {
                "score": score,
                "epoch": epoch,
                "state": copy.deepcopy(model.trainable_state_dict()),
                "validation": current,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best["state"] is None:
        raise RuntimeError("KP11 training produced no checkpoint")
    model.load_trainable_state_dict(best["state"])
    danger_calibration = calibrate_danger_bias(model, loaders["val"], device)
    validation_result = evaluate(model, loaders["val"], device)
    promotion = promotion_audit(validation_result)
    test_result = None
    if args.evaluate_test:
        if not promotion["passed"]:
            raise RuntimeError(
                "fixed test remains sealed because validation promotion failed"
            )
        if args.pose_anchor == "kp10":
            test = attach_kp10_anchors(test, "test", args, device)
            test_loader = DataLoader(
                test, batch_size=args.batch_size * 2, shuffle=False,
                num_workers=0, pin_memory=device == "cuda",
            )
        else:
            test_loader = loaders["test"]
        test_result = evaluate(model, test_loader, device)

    result = {
        "run": "KP11-DYNAMIC-MOTION-EXP01",
        "model_family": "NotiFi-KP11",
        "candidate_version": "KP11-DYNAMIC-MOTION",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "fixed_test_opened": test_result is not None,
        "inference_inputs": ["CSI", "link_mask"],
        "device": device,
        "seed": args.seed,
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
            "pose_train": train_pose.target.describe(),
            "quality": quality_summary(train),
        },
        "architecture": {
            "pose_anchor": args.pose_anchor,
            "static_context": "frozen P2 temporal features and soft class probabilities",
            "dynamic_inputs": ["delta_1", "delta_3", "delta_7", "high_pass_15"],
            "temporal_scales": ["short_d1_d2", "medium_d4_d8", "global_transformer"],
            "decoder": "anchor-relative bone rotation + direct FK reconstruction",
            "coarse_pose_blend": False,
            "retrieval_pose_blend": False,
            "endpoint_scale_m": args.endpoint_scale,
            "shape_scale": args.shape_scale,
            "classification_gradient_to_pose": False,
            "classification_heads_trained": args.classification_learning_rate > 0,
            "auxiliary_heads": ["phase", "relative_contact", "seven_part_motion"],
            "hidden": args.hidden,
            "trainable_parameters": model.n_trainable_parameters(),
        },
        "pose_prior": prior_audit,
        "selection": {"epoch": best["epoch"], "score": best["score"]},
        "danger_bias_calibration": danger_calibration,
        "validation": validation_result,
        "promotion": promotion,
        "test": test_result,
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"],
        "protocol": args.exp,
        "architecture": result["architecture"],
        "pose_prior": {"directions": prior, "bone_lengths": lengths},
        "trainable_model": model.trainable_state_dict(),
        "source": {"p2_checkpoint": report_path(args.p2_checkpoint)},
        "selection": result["selection"],
        "validation": validation_result,
        "promotion": promotion,
        "test": test_result,
    }, args.run_dir / "best_model.pt")
    print(json.dumps({
        "run_dir": str(args.run_dir),
        "selection": result["selection"],
        "validation": validation_result,
        "promotion": promotion,
        "fixed_test_opened": test_result is not None,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
