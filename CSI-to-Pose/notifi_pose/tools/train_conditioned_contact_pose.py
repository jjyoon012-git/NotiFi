"""Train KP4-DCC: directional, conditioned, contact-aware seen pose."""

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
from ..conditioned_contact_pose import (
    CONTACT_JOINTS,
    DirectionalConditionedContactPose,
)
from ..external_pretraining import transplant_external_encoder
from ..seen_v2 import injury_targets
from ..trainer import set_seed
from .diagnose_observability import report_path
from .train_geometry_phase_pose import (
    ParameterEMA,
    build_model as build_source_model,
    warmup_cosine_factor,
)
from .train_hierarchical_pose import banded_velocity_alignment
from .train_kinetic_pose import (
    _weighted_mean,
    build_components,
    evaluate_strengths,
    kinetic_pose_loss,
    load_or_create_coarse_store,
    make_loaders,
    pose_selection_score,
    relative_pose_speed,
)


def fall_phase_targets(pose: torch.Tensor, root: torch.Tensor,
                       valid: torch.Tensor,
                       risk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split danger motion into rest, descent, ground transition, and settled.

    This target uses sustained body-to-floor proximity, not an acceleration
    peak or a guessed collision score. It intentionally does not label which
    joint touched first.
    """
    phase = torch.zeros_like(valid, dtype=torch.long)
    mask = valid & (risk == 2)[:, None]
    speed = relative_pose_speed(pose, valid)
    contact = injury_targets(pose, root, valid, risk)["injury_contact"].any(-1)
    absolute = pose[:, :, list(CONTACT_JOINTS)] + root[:, :, None]
    mean_height = absolute[..., 1].mean(-1)
    for item in range(len(pose)):
        frames = torch.nonzero(mask[item], as_tuple=False).flatten()
        if len(frames) < 2:
            continue
        local_speed = speed[item, frames]
        threshold = max(0.12, 0.20 * float(local_speed.max()))
        active = frames[local_speed > threshold]
        start = int(active[0]) if len(active) else int(frames[0])
        contacts = torch.nonzero(
            contact[item] & valid[item]
            & (torch.arange(valid.shape[1], device=valid.device) >= start),
            as_tuple=False,
        ).flatten()
        if len(contacts):
            ground = int(contacts[0])
        else:
            eligible = frames[frames >= start]
            ground = int(eligible[torch.argmin(mean_height[item, eligible])])
        transition_end = min(ground + 5, int(frames[-1]) + 1)
        phase[item, start:ground] = 1
        phase[item, ground:transition_end] = 2
        phase[item, transition_end:int(frames[-1]) + 1] = 3
    return phase, mask


def timestamp_aware_alignment(predicted: torch.Tensor, target: torch.Tensor,
                              valid: torch.Tensor, risk: torch.Tensor,
                              exact: torch.Tensor) -> torch.Tensor:
    """Use a one-frame band for measured time and three for inferred time."""
    speed = relative_pose_speed(target, valid)
    frame_weight = 1.0 + 2.0 * (speed / 0.5).clamp(0.0, 2.0)
    frame_weight *= 1.0 + 0.75 * (risk == 2).float()[:, None]
    terms = []
    weights = []
    for selected, radius in ((exact.bool(), 1), (~exact.bool(), 3)):
        if not selected.any():
            continue
        terms.append(banded_velocity_alignment(
            predicted[selected], target[selected], valid[selected],
            frame_weight[selected], radius=radius,
            temperature=0.05, lag_penalty=0.004,
        ))
        weights.append(float(selected.sum()))
    if not terms:
        return predicted.new_zeros(())
    return sum(term * weight for term, weight in zip(terms, weights)) / sum(weights)


def auxiliary_loss(output: dict, batch: dict, class_weight: torch.Tensor,
                   risk_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    valid = batch["valid"].bool()
    risk = batch["risk_id"]
    action = F.cross_entropy(
        output["action_logits"], batch["class_id"], weight=class_weight
    )
    risk_class = F.cross_entropy(
        output["risk_logits"], risk, weight=risk_weight
    )

    phase_target, phase_mask = fall_phase_targets(
        batch["pose_rel"], batch["root"], valid, risk
    )
    if phase_mask.any():
        phase = F.cross_entropy(
            output["phase_logits"][phase_mask], phase_target[phase_mask],
            weight=output["phase_logits"].new_tensor((0.25, 1.0, 2.0, 1.0)),
        )
    else:
        phase = output["pose_rel"].new_zeros(())

    injury = injury_targets(
        batch["pose_rel"], batch["root"], valid, risk
    )
    contact_target = injury["injury_contact"].to(output["contact_logits"].dtype)
    contact_mask = valid[..., None].expand_as(contact_target)
    positives = contact_target[contact_mask].sum()
    negatives = contact_mask.sum() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 15.0)
    contact_element = F.binary_cross_entropy_with_logits(
        output["contact_logits"], contact_target, reduction="none",
        pos_weight=positive_weight,
    )
    contact = _weighted_mean(contact_element, contact_mask.to(contact_element.dtype))

    predicted_absolute = (
        output["pose_rel"][:, :, list(CONTACT_JOINTS)]
        + batch["root"][:, :, None]
    )
    predicted_height = (
        predicted_absolute[..., 1] - injury["floor_height"][:, None, None]
    )
    geometric_contact = torch.sigmoid((0.12 - predicted_height) / 0.03)
    contact_probability = torch.sigmoid(output["contact_logits"])
    contact_consistency = _weighted_mean(
        (contact_probability - geometric_contact).square(),
        contact_mask.to(contact_probability.dtype),
    )

    alignment = timestamp_aware_alignment(
        output["pose_rel"], batch["pose_rel"], valid, risk,
        batch["timestamp_exact"],
    )
    contact_frames = injury["injury_contact"].any(-1) & valid & (risk == 2)[:, None]
    coordinate = F.smooth_l1_loss(
        output["pose_rel"], batch["pose_rel"], reduction="none", beta=0.04
    ).mean(-1)
    contact_pose = _weighted_mean(
        coordinate,
        contact_frames[..., None].to(coordinate.dtype),
    )

    speed = relative_pose_speed(batch["pose_rel"], valid)
    static = valid & (speed < 0.08)
    gate_regularization = _weighted_mean(
        (output["joint_confidence_gate"] - 0.30).square(),
        static[..., None].to(output["joint_confidence_gate"].dtype),
    )
    gate_pair = valid[:, 1:] & valid[:, :-1]
    gate_temporal = _weighted_mean(
        (output["joint_confidence_gate"][:, 1:]
         - output["joint_confidence_gate"][:, :-1]).square(),
        gate_pair[..., None].to(output["joint_confidence_gate"].dtype),
    )
    total = (
        0.04 * action
        + 0.06 * risk_class
        + 0.03 * phase
        + 0.04 * contact
        + 0.02 * contact_consistency
        + 0.03 * alignment
        + 0.20 * contact_pose
        + 0.10 * gate_regularization
        + 0.02 * gate_temporal
    )
    return total, {
        "auxiliary": float(total.detach()),
        "action": float(action.detach()),
        "risk": float(risk_class.detach()),
        "phase": float(phase.detach()),
        "contact": float(contact.detach()),
        "contact_consistency": float(contact_consistency.detach()),
        "timestamp_alignment": float(alignment.detach()),
        "contact_pose": float(contact_pose.detach()),
        "gate_regularization": float(gate_regularization.detach()),
        "gate_temporal": float(gate_temporal.detach()),
    }


def _macro_f1(prediction: torch.Tensor, target: torch.Tensor,
              classes: int) -> float:
    scores = []
    for value in range(classes):
        predicted = prediction == value
        actual = target == value
        tp = (predicted & actual).sum().float()
        fp = (predicted & ~actual).sum().float()
        fn = (~predicted & actual).sum().float()
        scores.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1.0)))
    return float(np.mean(scores))


@torch.no_grad()
def evaluate_auxiliary(model, loader, coarse_store, device: str) -> dict:
    model.eval()
    action_pred, action_true = [], []
    risk_pred, risk_true = [], []
    phase_pred, phase_true = [], []
    contact_pred, contact_true = [], []
    gates = []
    for batch in loader:
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(
            device_batch["csi"], device_batch["link_mask"],
            coarse_store.lookup(batch["row"], device),
        )
        valid = device_batch["valid"].bool()
        phase, phase_mask = fall_phase_targets(
            device_batch["pose_rel"], device_batch["root"], valid,
            device_batch["risk_id"],
        )
        injury = injury_targets(
            device_batch["pose_rel"], device_batch["root"], valid,
            device_batch["risk_id"],
        )
        action_pred.append(output["action_logits"].argmax(-1).cpu())
        action_true.append(device_batch["class_id"].cpu())
        risk_pred.append(output["risk_logits"].argmax(-1).cpu())
        risk_true.append(device_batch["risk_id"].cpu())
        phase_pred.append(output["phase_logits"].argmax(-1)[phase_mask].cpu())
        phase_true.append(phase[phase_mask].cpu())
        expanded = valid[..., None].expand_as(injury["injury_contact"])
        contact_pred.append(
            (torch.sigmoid(output["contact_logits"])[expanded] >= 0.5).cpu()
        )
        contact_true.append(injury["injury_contact"][expanded].cpu())
        gates.append(output["joint_confidence_gate"][valid].mean().cpu())
    action_pred = torch.cat(action_pred)
    action_true = torch.cat(action_true)
    risk_pred = torch.cat(risk_pred)
    risk_true = torch.cat(risk_true)
    phase_pred = torch.cat(phase_pred) if phase_pred else torch.empty(0, dtype=torch.long)
    phase_true = torch.cat(phase_true) if phase_true else torch.empty(0, dtype=torch.long)
    contact_pred = torch.cat(contact_pred)
    contact_true = torch.cat(contact_true)
    danger = risk_true == 2
    tp = (contact_pred & contact_true).sum().float()
    fp = (contact_pred & ~contact_true).sum().float()
    fn = (~contact_pred & contact_true).sum().float()
    return {
        "action_accuracy": float((action_pred == action_true).float().mean()),
        "action_macro_f1": _macro_f1(action_pred, action_true, C.N_CLASSES),
        "risk_accuracy": float((risk_pred == risk_true).float().mean()),
        "risk_macro_f1": _macro_f1(risk_pred, risk_true, C.N_RISK),
        "danger_recall": float((risk_pred[danger] == 2).float().mean()),
        "phase_macro_f1": _macro_f1(phase_pred, phase_true, 4),
        "contact_f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1.0)),
        "mean_joint_gate": float(torch.stack(gates).mean()),
    }


def train_epoch(model, loader, optimizer, scaler, ema, coarse_store,
                class_weight: torch.Tensor, risk_weight: torch.Tensor,
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
            output = model(
                batch["csi"], batch["link_mask"],
                coarse_store.lookup(batch["row"].cpu(), device),
            )
            pose_loss, pose_parts = kinetic_pose_loss(output, batch, args)
            auxiliary, auxiliary_parts = auxiliary_loss(
                output, batch, class_weight, risk_weight
            )
            loss = pose_loss + auxiliary
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        parts = {**pose_parts, **auxiliary_parts, "total_with_aux": float(loss.detach())}
        for key, value in parts.items():
            if math.isfinite(value):
                totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def _class_weights(index, column: str, classes: int,
                   device: str) -> torch.Tensor:
    counts = np.bincount(index[column].to_numpy(dtype=np.int64), minlength=classes)
    weight = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weight = weight / weight.mean()
    return torch.tensor(weight, device=device, dtype=torch.float32)


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
    parser.add_argument(
        "--external-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "mmfi_e01_external_pretrain_v1" / "best_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--aux-warmup-epochs", type=int, default=2)
    parser.add_argument("--encoder-warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aux-learning-rate", type=float, default=2e-4)
    parser.add_argument("--pose-learning-rate", type=float, default=5e-5)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--danger-frame-boost", type=float, default=0.75)
    parser.add_argument("--motion-weight", type=float, default=2.5)
    parser.add_argument("--distal-joint-weight", type=float, default=2.75)
    parser.add_argument("--lambda-distal", type=float, default=0.50)
    parser.add_argument("--lambda-velocity", type=float, default=0.25)
    parser.add_argument("--lambda-aux-velocity", type=float, default=0.05)
    parser.add_argument("--lambda-acceleration", type=float, default=0.005)
    parser.add_argument("--lambda-bone-length", type=float, default=0.12)
    parser.add_argument("--lambda-bone-direction", type=float, default=0.08)
    parser.add_argument("--lambda-static", type=float, default=0.08)
    parser.add_argument("--lambda-endpoint", type=float, default=0.30)
    parser.add_argument("--minimum-score-improvement", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--experiment-name", default="KP4-DCC-EXP01")
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp4_dcc_seed17",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_checkpoint = torch.load(
        args.hierarchical_checkpoint, map_location="cpu", weights_only=False
    )
    if source_checkpoint.get("protocol") != args.exp:
        raise RuntimeError("KP2-DH checkpoint protocol mismatch")
    source_model, _, source_architecture, _ = build_source_model(
        source_checkpoint, device
    )
    model = DirectionalConditionedContactPose(
        source_model, dropout=float(source_architecture.get("dropout", 0.08))
    ).to(device)
    external_report = None
    if args.external_checkpoint and args.external_checkpoint.exists():
        external = torch.load(
            args.external_checkpoint, map_location="cpu", weights_only=False
        )
        external_report = transplant_external_encoder(
            model.motion_encoder, external["shared"]
        )

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

    pose_prefixes = (
        "condition_projection.", "condition_blocks.",
        "joint_gate_head.", "direction_head.", "cartesian_head.",
        "velocity_head.",
    )
    pose_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(pose_prefixes)
    ]
    auxiliary_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith(("pose_model.", "motion_encoder.", *pose_prefixes))
    ]
    # Per-link normalizer parameters are frozen by the source contract and
    # must stay frozen when the copied motion encoder is enabled.
    encoder_parameters = [
        parameter for parameter in model.motion_encoder.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW([
        {"params": auxiliary_parameters, "lr": args.aux_learning_rate, "name": "auxiliary"},
        {"params": pose_parameters, "lr": args.pose_learning_rate, "name": "pose"},
        {"params": encoder_parameters, "lr": args.encoder_learning_rate, "name": "encoder"},
    ], weight_decay=args.weight_decay)
    for parameter in pose_parameters:
        parameter.requires_grad_(False)
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if not name.startswith("pose_model.") and (
            not name.startswith("motion_encoder.")
            or id(parameter) in encoder_ids
        )
    }
    ema = ParameterEMA(trainable, args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    class_weight = _class_weights(train.index, "class_id", C.N_CLASSES, device)
    risk_weight = _class_weights(train.index, "risk_id", C.N_RISK, device)
    risk_weight[2] *= 1.5

    initial_metrics = evaluate_strengths(
        model, loaders["val"], [0.0, 1.0], device, coarse_store
    )
    initial_aux = evaluate_auxiliary(model, loaders["val"], coarse_store, device)
    initial_score = pose_selection_score(initial_metrics[1.0])
    best = {
        "epoch": 0, "score": initial_score,
        "state": copy.deepcopy(model.trainable_state_dict()),
        "metrics": initial_metrics[1.0], "auxiliary": initial_aux,
    }
    history = [{
        "epoch": 0, "stage": "locked_KP2_DH_30_percent",
        "validation_score": initial_score,
        "validation": initial_metrics[1.0],
        "validation_auxiliary": initial_aux,
    }]
    print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        pose_enabled = epoch > args.aux_warmup_epochs
        encoder_enabled = epoch > args.encoder_warmup_epochs
        for parameter in pose_parameters:
            parameter.requires_grad_(pose_enabled)
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder_enabled)
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, scaler, ema, coarse_store,
            class_weight, risk_weight, device, args,
        )
        live = copy.deepcopy(model.trainable_state_dict())
        model.load_trainable_state_dict(ema.model_state(model))
        validation_metrics = evaluate_strengths(
            model, loaders["val"], [1.0], device, coarse_store
        )[1.0]
        validation_aux = evaluate_auxiliary(
            model, loaders["val"], coarse_store, device
        )
        model.load_trainable_state_dict(live)
        score = pose_selection_score(validation_metrics)
        record = {
            "epoch": epoch,
            "stage": (
                "joint_finetune" if encoder_enabled else
                "pose_warmup" if pose_enabled else "auxiliary_warmup"
            ),
            "train": train_metrics,
            "validation_score": score,
            "validation": validation_metrics,
            "validation_auxiliary": validation_aux,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score < best["score"] - args.minimum_score_improvement:
            best = {
                "epoch": epoch, "score": score,
                "state": copy.deepcopy(ema.model_state(model)),
                "metrics": validation_metrics, "auxiliary": validation_aux,
            }
            stale = 0
        elif encoder_enabled:
            stale += 1
        if stale >= args.patience:
            break

    model.load_trainable_state_dict(best["state"])
    test_baseline = evaluate_strengths(
        model, loaders["test"], [0.0], device, coarse_store
    )[0.0]
    test_metrics = evaluate_strengths(
        model, loaders["test"], [1.0], device, coarse_store
    )[1.0]
    test_aux = evaluate_auxiliary(model, loaders["test"], coarse_store, device)
    critical = (
        "mpjpe_m", "danger_pose_mpjpe_m", "danger_distal_mpjpe_m",
        "danger_high_motion_mpjpe_m",
    )
    deployment_gate = best["epoch"] > 0 and all(
        test_metrics[key] <= test_baseline[key] for key in critical
    )
    result = {
        "run": args.experiment_name,
        "model_family": "NotiFi-KP4",
        "candidate_version": "KP4-DCC",
        "promotion_status": "deployment_gate_passed" if deployment_gate else "experimental",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "inference_inputs": ["csi", "link_mask", "frozen_v13s_pose"],
        "config": {
            key: report_path(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
            "exact_timestamp_train": int(train.index.time_method.eq("timestamps").sum()),
        },
        "architecture": {
            "proposal": "frozen V13S + 0.30 * frozen KP2-DH delta",
            "motion_encoder": "trainable copied 4-link directional Doppler encoder",
            "external_initialization": external_report,
            "conditioning": ["17_action", "3_risk", "4_fall_phase", "8_body_contact"],
            "adaptive_gate": "frame_and_joint CSI gate initialized at 0.30",
            "pose_decoder": "bone-length-preserving direction residual plus bounded Cartesian residual",
            "alignment": "timestamp exact radius 1; inferred 30fps radius 3",
            "contact_target": "GT body-to-floor proximity; no acceleration collision heuristic",
        },
        "selection": {
            "epoch": best["epoch"], "score": best["score"],
            "initial_score": initial_score,
            "validation_improved": best["epoch"] > 0,
            "test_deployment_gate_passed": deployment_gate,
            "critical_test_metrics": list(critical),
        },
        "validation": best["metrics"],
        "validation_auxiliary": best["auxiliary"],
        "test_baseline": test_baseline,
        "test": test_metrics,
        "test_auxiliary": test_aux,
        "history": history,
        "baseline_configuration": baseline_config,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "trainable_model": best["state"],
        "architecture": result["architecture"],
        "source": {
            "hierarchical_checkpoint": report_path(args.hierarchical_checkpoint),
            "external_checkpoint": report_path(args.external_checkpoint),
        },
        "selection": result["selection"],
        "validation": result["validation"],
        "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
