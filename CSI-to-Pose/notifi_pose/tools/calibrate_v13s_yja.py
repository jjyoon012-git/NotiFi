"""Calibrate the locked V13S model on yja/E02 before held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import PoseDataset, build_datasets
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_v11_final import evaluate_pa_mpjpe
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import (
    DISTAL_JOINTS,
    classification_metrics,
    evaluate_classification,
    evaluate_trajectory,
)

BASIC_POSE_IDENTITY_PENALTY = 0.08


def _subset(dataset: PoseDataset, positions: np.ndarray) -> PoseDataset:
    rows = dataset.rows[np.asarray(positions, dtype=np.int64)]
    return PoseDataset(
        rows,
        dataset.cache,
        dataset.link_ok,
        train=False,
        seed=dataset.seed,
        baseline=dataset.baseline,
    )


def stratified_calibration_split(
    dataset: PoseDataset,
    calibration_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, PoseDataset]:
    if calibration_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("calibration and validation ratios must be positive")
    if calibration_ratio + validation_ratio >= 1:
        raise ValueError("calibration and validation must leave held-out test data")

    rng = np.random.default_rng(seed)
    labels = dataset.index.class_id.to_numpy(dtype=np.int64)
    positions = {"calibration": [], "validation": [], "test": []}
    for class_id in sorted(np.unique(labels)):
        group = np.flatnonzero(labels == class_id)
        rng.shuffle(group)
        n_calibration = max(1, int(round(len(group) * calibration_ratio)))
        n_validation = max(1, int(round(len(group) * validation_ratio)))
        if n_calibration + n_validation >= len(group):
            n_validation = max(1, len(group) - n_calibration - 1)
        positions["calibration"].extend(group[:n_calibration])
        positions["validation"].extend(
            group[n_calibration:n_calibration + n_validation]
        )
        positions["test"].extend(group[n_calibration + n_validation:])

    result = {
        name: _subset(dataset, np.sort(np.asarray(selected)))
        for name, selected in positions.items()
    }
    trial_sets = [set(item.index.trial_id.astype(str)) for item in result.values()]
    if any(trial_sets[i] & trial_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("calibration split contains overlapping trials")
    return result


def basic_pose_calibration_split(
    dataset: PoseDataset,
    calibration_per_pose: int,
    validation_per_pose: int,
    seed: int,
    basic_class_ids: tuple[int, ...] = (1, 2, 3),
) -> dict[str, PoseDataset]:
    """Use only standing, sitting and lying trials for target calibration."""
    rng = np.random.default_rng(seed)
    labels = dataset.index.class_id.to_numpy(dtype=np.int64)
    calibration = []
    validation = []
    used = set()
    for class_id in basic_class_ids:
        group = np.flatnonzero(labels == class_id)
        required = calibration_per_pose + validation_per_pose
        if len(group) < required:
            raise RuntimeError(
                f"basic class {class_id} has {len(group)} trials, needs {required}"
            )
        rng.shuffle(group)
        calibration.extend(group[:calibration_per_pose])
        validation.extend(group[calibration_per_pose:required])
        used.update(int(item) for item in group[:required])
    test = [index for index in range(len(dataset)) if index not in used]
    return {
        "calibration": _subset(dataset, np.sort(np.asarray(calibration))),
        "validation": _subset(dataset, np.sort(np.asarray(validation))),
        "test": _subset(dataset, np.asarray(test)),
    }


def _trial_digest(dataset: PoseDataset) -> str:
    text = "\n".join(sorted(dataset.index.trial_id.astype(str)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MomentInputCalibration(nn.Module):
    """Map target amp/phase moments to the immutable source reference."""

    def __init__(self, base: nn.Module, source_mu: torch.Tensor,
                 source_sigma: torch.Tensor, target_mu: torch.Tensor,
                 target_sigma: torch.Tensor):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("source_mu", source_mu.float())
        self.register_buffer("source_sigma", source_sigma.float())
        self.register_buffer("target_mu", target_mu.float())
        minimum_sigma = torch.maximum(
            source_sigma.float() * 0.10,
            torch.full_like(source_sigma.float(), 1e-4),
        )
        self.register_buffer(
            "target_sigma", torch.maximum(target_sigma.float(), minimum_sigma)
        )
        self.strength = 0.0

    def set_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("moment calibration strength must be in [0, 1]")
        self.strength = float(strength)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        if not self.strength:
            return self.base(csi, link_mask)
        aligned = (
            (csi - self.target_mu[None, None])
            / self.target_sigma[None, None]
            * self.source_sigma[None, None]
            + self.source_mu[None, None]
        )
        calibrated = torch.lerp(csi, aligned, self.strength)
        observed = link_mask[..., None, None]
        calibrated = torch.where(observed, calibrated, csi)
        return self.base(calibrated, link_mask)


class OutputCalibration(nn.Module):
    """Apply low-dimensional target geometry and logit calibration."""

    def __init__(self, base: nn.Module, pose_matrix: torch.Tensor,
                 pose_bias: torch.Tensor, joint_bias: torch.Tensor,
                 root_matrix: torch.Tensor, root_bias: torch.Tensor,
                 class_scale: torch.Tensor, class_bias: torch.Tensor,
                 risk_scale: torch.Tensor, risk_bias: torch.Tensor,
                 raw_logit_base: nn.Module | None = None,
                 preserve_raw_root: bool = False):
        super().__init__()
        self.base = base
        self.raw_logit_base = raw_logit_base
        self.preserve_raw_root = bool(preserve_raw_root)
        for name, value in {
            "pose_matrix": pose_matrix,
            "pose_bias": pose_bias,
            "joint_bias": joint_bias,
            "root_matrix": root_matrix,
            "root_bias": root_bias,
            "class_scale": class_scale,
            "class_bias": class_bias,
            "risk_scale": risk_scale,
            "risk_bias": risk_bias,
        }.items():
            self.register_buffer(name, value.float())

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = dict(self.base(csi, link_mask))
        if self.raw_logit_base is not None:
            raw_output = self.raw_logit_base(csi, link_mask)
            output["class_logits"] = raw_output["class_logits"]
            output["risk_logits"] = raw_output["risk_logits"]
            if self.preserve_raw_root:
                output["root"] = raw_output["root"]
        pose = (
            output["pose_rel"] @ self.pose_matrix
            + self.pose_bias
            + self.joint_bias[None, None]
        )
        pelvis = C.JOINT_INDEX["pelvis"]
        output["pose_rel"] = pose - pose[:, :, pelvis:pelvis + 1]
        output["root"] = output["root"] @ self.root_matrix + self.root_bias
        output["class_logits"] = (
            output["class_logits"] * self.class_scale + self.class_bias
        )
        output["risk_logits"] = (
            output["risk_logits"] * self.risk_scale + self.risk_bias
        )
        return output


@torch.no_grad()
def estimate_moments(dataset: PoseDataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    total = None
    square = None
    count = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        csi = batch["csi"].double()
        weight = batch["link_mask"][..., None, None].double()
        batch_total = (csi * weight).sum((0, 1))
        batch_square = (csi.square() * weight).sum((0, 1))
        batch_count = weight.sum((0, 1))
        total = batch_total if total is None else total + batch_total
        square = batch_square if square is None else square + batch_square
        count = batch_count if count is None else count + batch_count
    if total is None:
        raise RuntimeError("cannot estimate moments from an empty dataset")
    mean = total / count.clamp_min(1.0)
    variance = square / count.clamp_min(1.0) - mean.square()
    return mean.float(), variance.clamp_min(1e-8).sqrt().float()


def _loader(dataset: PoseDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model: nn.Module, dataset: PoseDataset, batch_size: int,
             device: str, max_shift: int) -> dict:
    loader = _loader(dataset, batch_size)
    trajectory = evaluate_trajectory(model, loader, device, max_shift)
    trajectory["pa_mpjpe_m"] = evaluate_pa_mpjpe(model, loader, device)
    classification = evaluate_classification(model, loader, device, 0.0)
    return {"trajectory": trajectory, "classification": classification}


def _selection_score(result: dict) -> float:
    trajectory = result["trajectory"]
    classification = result["classification"]
    score = float(
        trajectory["mpjpe_m"]
        + 0.50 * trajectory["root_error_m"]
        + 0.08 * (1.0 - classification["class"]["macro_f1"])
        + 0.08 * (1.0 - classification["risk"]["macro_f1"])
    )
    danger_pose = trajectory.get("danger_pose_mpjpe_m")
    if danger_pose is not None and np.isfinite(danger_pose):
        score += float(
            0.50 * danger_pose
            + 0.25 * trajectory["danger_pose_distal_mpjpe_m"]
            + 0.12 * (1.0 - classification["risk"]["danger_recall"])
        )
    return score


@torch.no_grad()
def collect_predictions(model: nn.Module, dataset: PoseDataset,
                        batch_size: int, device: str) -> dict[str, torch.Tensor]:
    values = {key: [] for key in (
        "pose", "root", "class_logits", "risk_logits", "pose_target",
        "root_target", "valid", "class_target", "risk_target",
    )}
    for batch in _loader(dataset, batch_size):
        output = model(batch["csi"].to(device), batch["link_mask"].to(device))
        values["pose"].append(output["pose_rel"].float().cpu())
        values["root"].append(output["root"].float().cpu())
        values["class_logits"].append(output["class_logits"].float().cpu())
        values["risk_logits"].append(output["risk_logits"].float().cpu())
        values["pose_target"].append(batch["pose_rel"].float())
        values["root_target"].append(batch["root"].float())
        values["valid"].append(batch["valid"].bool())
        values["class_target"].append(batch["class_id"].long())
        values["risk_target"].append(batch["risk_id"].long())
    return {key: torch.cat(parts) for key, parts in values.items()}


def pose_prototypes(data: dict[str, torch.Tensor],
                    class_ids: tuple[int, ...] = (1, 2, 3)) -> dict[int, torch.Tensor]:
    """Average frozen source predictions for each instructed static pose."""
    prototypes = {}
    for class_id in class_ids:
        selected = data["class_target"] == class_id
        valid = data["valid"] & selected[:, None]
        if not valid.any():
            raise RuntimeError(f"source prototype class {class_id} is empty")
        prototypes[class_id] = data["pose"][valid].mean(0)
    return prototypes


def with_prototype_targets(data: dict[str, torch.Tensor],
                           prototypes: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace target pose with the known-label source prototype, never target GT."""
    targets = torch.empty_like(data["pose_target"])
    covered = torch.zeros(len(targets), dtype=torch.bool)
    for class_id, prototype in prototypes.items():
        selected = data["class_target"] == class_id
        targets[selected] = prototype[None, None]
        covered |= selected
    if not covered.all():
        missing = sorted(set(data["class_target"][~covered].tolist()))
        raise RuntimeError(f"missing source prototypes for classes {missing}")
    result = dict(data)
    result["pose_target"] = targets
    return result


def fit_affine(x: torch.Tensor, y: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.double().reshape(-1, 3)
    y = y.double().reshape(-1, 3)
    ones = torch.ones(len(x), 1, dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    prior = torch.zeros(4, 3, dtype=x.dtype)
    prior[:3] = torch.eye(3, dtype=x.dtype)
    covariance = design.T @ design / max(len(x), 1)
    cross = design.T @ y / max(len(x), 1)
    regularizer = torch.eye(4, dtype=x.dtype) * float(ridge)
    regularizer[-1, -1] *= 0.10
    solution = torch.linalg.solve(
        covariance + regularizer + torch.eye(4, dtype=x.dtype) * 1e-8,
        cross + regularizer @ prior,
    )
    return solution[:3].float(), solution[3].float()


def _apply_pose(data: dict, matrix: torch.Tensor, bias: torch.Tensor,
                joint_bias: torch.Tensor) -> torch.Tensor:
    pose = data["pose"] @ matrix + bias + joint_bias[None, None]
    pelvis = C.JOINT_INDEX["pelvis"]
    return pose - pose[:, :, pelvis:pelvis + 1]


def _pose_score(predicted: torch.Tensor, data: dict) -> float:
    distance = torch.linalg.vector_norm(predicted - data["pose_target"], dim=-1)
    valid = data["valid"]
    all_error = distance[valid].mean()
    danger = data["risk_target"] == 2
    danger_valid = valid & danger[:, None]
    if not danger.any():
        return float(all_error)
    danger_error = distance[danger_valid].mean()
    distal = distance[:, :, list(DISTAL_JOINTS)]
    danger_distal = distal[danger_valid].mean()
    return float(all_error + 0.50 * danger_error + 0.25 * danger_distal)


def _root_score(predicted: torch.Tensor, data: dict) -> float:
    distance = torch.linalg.vector_norm(predicted - data["root_target"], dim=-1)
    valid = data["valid"]
    danger = data["risk_target"] == 2
    score = distance[valid].mean()
    if danger.any():
        score = score + 0.75 * distance[valid & danger[:, None]].mean()
    return float(score)


def select_pose_geometry(train: dict, validation: dict,
                         identity_penalty: float = 0.0) -> tuple[dict, list[dict]]:
    valid_train = train["valid"]
    pose_x = train["pose"][valid_train]
    pose_y = train["pose_target"][valid_train]
    identity = torch.eye(3)
    zero = torch.zeros(3)
    zero_joint = torch.zeros(C.N_JOINTS, 3)

    identity_metric = _pose_score(validation["pose"], validation)
    pose_candidates = [{
        "name": "identity", "ridge": None, "joint_bias_strength": 0.0,
        "matrix": identity, "bias": zero, "joint_bias": zero_joint,
        "metric_score": identity_metric, "complexity": 0.0,
        "score": identity_metric,
    }]
    for ridge in (0.001, 0.01, 0.1, 1.0):
        pose_matrix, pose_bias = fit_affine(pose_x, pose_y, ridge)
        calibrated_train = train["pose"] @ pose_matrix + pose_bias
        residual = train["pose_target"] - calibrated_train
        residual = residual * train["valid"][..., None, None]
        denominator = train["valid"].sum().clamp_min(1)
        joint_mean = residual.sum((0, 1)) / denominator
        for strength in (0.0, 0.25, 0.5, 1.0):
            joint_bias = joint_mean * strength
            predicted = _apply_pose(
                validation, pose_matrix, pose_bias, joint_bias
            )
            metric_score = _pose_score(predicted, validation)
            complexity = float(
                (pose_matrix - identity).square().sum()
                + 4.0 * pose_bias.square().sum()
                + 4.0 * joint_bias.square().sum(-1).mean()
            )
            pose_candidates.append({
                "name": "affine", "ridge": ridge,
                "joint_bias_strength": strength,
                "matrix": pose_matrix, "bias": pose_bias,
                "joint_bias": joint_bias,
                "metric_score": metric_score,
                "complexity": complexity,
                "score": metric_score + float(identity_penalty) * complexity,
            })

    return min(pose_candidates, key=lambda item: item["score"]), pose_candidates


def select_geometry(train: dict, validation: dict) -> tuple[dict, list[dict]]:
    selected_pose, pose_candidates = select_pose_geometry(train, validation)
    valid_train = train["valid"]
    root_x = train["root"][valid_train]
    root_y = train["root_target"][valid_train]
    identity = torch.eye(3)
    zero = torch.zeros(3)
    root_candidates = [{
        "name": "identity", "ridge": None,
        "matrix": identity, "bias": zero,
        "score": _root_score(validation["root"], validation),
    }]
    for ridge in (0.001, 0.01, 0.1, 1.0):

        root_matrix, root_bias = fit_affine(root_x, root_y, ridge)
        predicted_root = validation["root"] @ root_matrix + root_bias
        root_candidates.append({
            "name": "affine", "ridge": ridge,
            "matrix": root_matrix, "bias": root_bias,
            "score": _root_score(predicted_root, validation),
        })

    selected_root = min(root_candidates, key=lambda item: item["score"])
    return {
        "pose": selected_pose,
        "root": selected_root,
    }, pose_candidates + root_candidates


def fit_logit_calibration(logits: torch.Tensor, targets: torch.Tensor,
                          l2: float, danger_boost: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    classes = logits.shape[1]
    log_scale = torch.zeros(classes, requires_grad=True)
    bias = torch.zeros(classes, requires_grad=True)
    optimizer = torch.optim.Adam([log_scale, bias], lr=0.05)
    weight = torch.ones(classes)
    weight[-1] = float(danger_boost)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        calibrated = logits * log_scale.exp() + bias
        loss = F.cross_entropy(calibrated, targets, weight=weight)
        loss = loss + float(l2) * (log_scale.square().mean() + bias.square().mean())
        loss.backward()
        optimizer.step()
    return log_scale.detach().exp(), bias.detach()


def _logit_metrics(class_logits: torch.Tensor, risk_logits: torch.Tensor,
                   data: dict) -> dict:
    return classification_metrics({
        "class_logits": class_logits,
        "risk_logits": risk_logits,
        "class_target": data["class_target"],
        "risk_target": data["risk_target"],
    })


def select_logits(train: dict, validation: dict) -> tuple[dict, list[dict]]:
    class_candidates = [{
        "l2": None,
        "scale": torch.ones(C.N_CLASSES),
        "bias": torch.zeros(C.N_CLASSES),
    }]
    risk_candidates = [{
        "l2": None,
        "scale": torch.ones(C.N_RISK),
        "bias": torch.zeros(C.N_RISK),
    }]
    for l2 in (0.01, 0.1, 1.0, 10.0):
        scale, bias = fit_logit_calibration(
            train["class_logits"], train["class_target"], l2
        )
        class_candidates.append({"l2": l2, "scale": scale, "bias": bias})
        scale, bias = fit_logit_calibration(
            train["risk_logits"], train["risk_target"], l2,
            danger_boost=2.0,
        )
        risk_candidates.append({"l2": l2, "scale": scale, "bias": bias})

    for candidate in class_candidates:
        metrics = _logit_metrics(
            validation["class_logits"] * candidate["scale"] + candidate["bias"],
            validation["risk_logits"], validation,
        )["class"]
        candidate["metrics"] = metrics
        candidate["score"] = float(
            1.0 - metrics["accuracy"] + 0.50 * (1.0 - metrics["macro_f1"])
        )
    for candidate in risk_candidates:
        metrics = _logit_metrics(
            validation["class_logits"],
            validation["risk_logits"] * candidate["scale"] + candidate["bias"],
            validation,
        )["risk"]
        candidate["metrics"] = metrics
        candidate["score"] = float(
            1.0 - metrics["macro_f1"]
            + 0.75 * (1.0 - metrics["danger_recall"])
            + 0.10 * metrics["safe_to_danger_rate"]
        )
    return {
        "class": min(class_candidates, key=lambda item: item["score"]),
        "risk": min(risk_candidates, key=lambda item: item["score"]),
    }, class_candidates + risk_candidates


def _serialize_candidate(candidate: dict) -> dict:
    return {
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in candidate.items()
        if key not in {"matrix", "bias", "joint_bias", "scale"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=Path("work_v2/runs/p2_sub_single_clean_finetune/best_model.pt"),
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=Path("docs/results/v13s_pruned_pose_root_ensemble.json"),
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=Path("work_v2/runs/p2_v12w_robust_classification_ensemble/validation.json"),
    )
    parser.add_argument("--source-exp", default="single_split_lmh_e01")
    parser.add_argument("--sealed-fold", default="yja_E02")
    parser.add_argument(
        "--calibration-mode",
        choices=("basic_poses", "stratified_actions"),
        default="basic_poses",
    )
    parser.add_argument("--basic-calibration-per-pose", type=int, default=4)
    parser.add_argument("--basic-validation-per-pose", type=int, default=2)
    parser.add_argument("--calibration-ratio", type=float, default=0.30)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/v13s_yja_basic_pose_calibration.json"),
    )
    parser.add_argument(
        "--calibration-state", type=Path,
        default=Path("work_v2/runs/v13s_yja_basic_pose_calibration/calibration.pt"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sealed = build_datasets(
        exp="sealed", fold=args.sealed_fold, baseline="sub", seed=args.seed
    )["test"]
    absence = sealed.index.task.to_numpy() == C.TASK_CLS
    if int(absence.sum()) != 12:
        raise RuntimeError(f"expected 12 absence trials, found {int(absence.sum())}")
    if not (sealed.index.n_alive.to_numpy(dtype=np.int64) == 3).all():
        raise RuntimeError("V13S clean calibration expects all three yja links")
    pose_dataset = pose_only(sealed)
    if args.calibration_mode == "basic_poses":
        splits = basic_pose_calibration_split(
            pose_dataset,
            args.basic_calibration_per_pose,
            args.basic_validation_per_pose,
            args.seed,
        )
    else:
        splits = stratified_calibration_split(
            pose_dataset, args.calibration_ratio, args.validation_ratio, args.seed
        )

    root_lock = _read_locked(args.root_calibration, args.source_exp)
    class_lock = _read_locked(args.classification_calibration, args.source_exp)
    model_args = argparse.Namespace(
        p2_checkpoint=args.p2_checkpoint, exp=args.source_exp
    )
    base, configuration = build_locked_model(
        model_args, device, root_lock, class_lock
    )
    base.eval()
    p2 = torch.load(args.p2_checkpoint, map_location="cpu", weights_only=False)
    source_prototype = None
    source_reference = {"kind": "checkpoint_train_moments"}
    if args.calibration_mode == "basic_poses":
        source_train = pose_only(build_datasets(
            exp=args.source_exp, baseline="sub", seed=args.seed
        )["train"])
        basic_source_positions = np.flatnonzero(
            np.isin(source_train.index.class_id.to_numpy(dtype=np.int64), (1, 2, 3))
        )
        source_basic = _subset(source_train, basic_source_positions)
        source_mu, source_sigma = estimate_moments(source_basic, args.batch_size)
        source_predictions = collect_predictions(
            base, source_basic, args.batch_size, device
        )
        source_prototype = pose_prototypes(source_predictions)
        source_reference = {
            "kind": "source_basic_pose_csi_and_frozen_prediction_prototypes",
            "trials": len(source_basic),
            "classes": [1, 2, 3],
            "target_gt_required": False,
        }
    else:
        source_mu = p2["model"]["norm.mu"]
        source_sigma = p2["model"]["norm.sigma"]
    target_mu, target_sigma = estimate_moments(
        splits["calibration"], args.batch_size
    )
    moment_model = MomentInputCalibration(
        base, source_mu, source_sigma, target_mu, target_sigma
    ).to(device).eval()

    moment_candidates = []
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        moment_model.set_strength(strength)
        if args.calibration_mode == "basic_poses":
            predictions = collect_predictions(
                moment_model, splits["validation"], args.batch_size, device
            )
            prototype_validation = with_prototype_targets(
                predictions, source_prototype
            )
            prototype_error = _pose_score(
                prototype_validation["pose"], prototype_validation
            )
            validation_result = {
                "source_prototype_pose_error_m": prototype_error,
                "target_gt_used": False,
            }
            selection_score = prototype_error
        else:
            validation_result = evaluate(
                moment_model, splits["validation"], args.batch_size,
                device, args.max_shift,
            )
            selection_score = _selection_score(validation_result)
        moment_candidates.append({
            "strength": strength,
            "score": selection_score,
            "validation": validation_result,
        })
    selected_moment = min(moment_candidates, key=lambda item: item["score"])
    moment_model.set_strength(float(selected_moment["strength"]))

    calibration_predictions = collect_predictions(
        moment_model, splits["calibration"], args.batch_size, device
    )
    validation_predictions = collect_predictions(
        moment_model, splits["validation"], args.batch_size, device
    )
    if args.calibration_mode == "basic_poses":
        prototype_calibration = with_prototype_targets(
            calibration_predictions, source_prototype
        )
        prototype_validation = with_prototype_targets(
            validation_predictions, source_prototype
        )
        selected_pose, geometry_candidates = select_pose_geometry(
            prototype_calibration, prototype_validation,
            identity_penalty=BASIC_POSE_IDENTITY_PENALTY,
        )
        selected_geometry = {
            "pose": selected_pose,
            "root": {
                "name": "identity",
                "ridge": None,
                "matrix": torch.eye(3),
                "bias": torch.zeros(3),
                "decision": "identity: instructed pose labels do not identify world root",
            },
        }
        selected_logits = {
            "class": {
                "l2": None,
                "scale": torch.ones(C.N_CLASSES),
                "bias": torch.zeros(C.N_CLASSES),
                "decision": "identity: basic safe poses cannot calibrate 17 classes",
            },
            "risk": {
                "l2": None,
                "scale": torch.ones(C.N_RISK),
                "bias": torch.zeros(C.N_RISK),
                "decision": "identity: basic safe poses cannot calibrate danger logits",
            },
        }
        logit_candidates = []
    else:
        selected_geometry, geometry_candidates = select_geometry(
            calibration_predictions, validation_predictions
        )
        selected_logits, logit_candidates = select_logits(
            calibration_predictions, validation_predictions
        )

    pose_choice = selected_geometry["pose"]
    root_choice = selected_geometry["root"]
    class_choice = selected_logits["class"]
    risk_choice = selected_logits["risk"]
    calibrated = OutputCalibration(
        moment_model,
        pose_choice["matrix"].to(device),
        pose_choice["bias"].to(device),
        pose_choice["joint_bias"].to(device),
        root_choice["matrix"].to(device),
        root_choice["bias"].to(device),
        class_choice["scale"].to(device),
        class_choice["bias"].to(device),
        risk_choice["scale"].to(device),
        risk_choice["bias"].to(device),
        raw_logit_base=base if args.calibration_mode == "basic_poses" else None,
        preserve_raw_root=args.calibration_mode == "basic_poses",
    ).to(device).eval()

    # The held-out split is opened only after every candidate is fixed above.
    moment_model.set_strength(0.0)
    baseline_test = evaluate(
        moment_model, splits["test"], args.batch_size, device, args.max_shift
    )
    moment_model.set_strength(float(selected_moment["strength"]))
    calibrated_test = evaluate(
        calibrated, splits["test"], args.batch_size, device, args.max_shift
    )

    split_report = {
        name: {
            "trials": len(dataset),
            "danger_trials": int((dataset.index.risk_id == 2).sum()),
            "sha256": _trial_digest(dataset),
            "class_counts": {
                str(key): int(value)
                for key, value in dataset.index.class_id.value_counts().sort_index().items()
            },
        }
        for name, dataset in splits.items()
    }
    report = {
        "run": (
            "v13s_yja_e02_basic_pose_calibrated"
            if args.calibration_mode == "basic_poses"
            else "v13s_yja_e02_action_calibrated"
        ),
        "source_protocol": args.source_exp,
        "target_protocol": f"sealed/{args.sealed_fold}",
        "calibration_mode": args.calibration_mode,
        "basic_pose_classes": {
            "1": "standing_still",
            "2": "sitting_still",
            "3": "lying_still",
        } if args.calibration_mode == "basic_poses" else None,
        "target_gt_usage": (
            "none for calibration fitting or selection; target GT is opened only "
            "for final held-out evaluation"
            if args.calibration_mode == "basic_poses"
            else "stratified action calibration trials"
        ),
        "source_reference": source_reference,
        "deployment_constraints": ({
            "pose_identity_penalty": BASIC_POSE_IDENTITY_PENALTY,
            "root_passthrough": True,
            "classification_passthrough": True,
        } if args.calibration_mode == "basic_poses" else None),
        "selection_data": "yja calibration + calibration-validation only",
        "test_used_for_selection": False,
        "held_out_test_opened_after_lock": True,
        "absence_trials": 12,
        "absence_usage": "site baseline subtraction only; excluded from pose metrics",
        "classification_path": (
            "raw frozen V13S CSI path; basic-pose moment calibration is pose-only"
            if args.calibration_mode == "basic_poses"
            else "target-calibrated CSI path"
        ),
        "root_path": (
            "raw frozen V13S CSI path; instructed pose labels do not identify world root"
            if args.calibration_mode == "basic_poses"
            else "target-calibrated CSI path"
        ),
        "split": split_report,
        "base_configuration": configuration,
        "selected": {
            "moment_strength": selected_moment["strength"],
            "pose": _serialize_candidate(pose_choice),
            "root": _serialize_candidate(root_choice),
            "class": _serialize_candidate(class_choice),
            "risk": _serialize_candidate(risk_choice),
        },
        "calibration_validation": {
            "moment_candidates": moment_candidates,
            "geometry_candidates": [
                _serialize_candidate(item) for item in geometry_candidates
            ],
            "logit_candidates": [
                _serialize_candidate(item) for item in logit_candidates
            ],
        },
        "held_out_test": {
            "frozen_v13s": baseline_test,
            "calibrated_v13s": calibrated_test,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.calibration_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "calibration_mode": args.calibration_mode,
        "source_protocol": args.source_exp,
        "target_protocol": f"sealed/{args.sealed_fold}",
        "source_reference": source_reference,
        "pose_identity_penalty": (
            BASIC_POSE_IDENTITY_PENALTY
            if args.calibration_mode == "basic_poses" else 0.0
        ),
        "preserve_raw_root": args.calibration_mode == "basic_poses",
        "preserve_raw_logits": args.calibration_mode == "basic_poses",
        "split": split_report,
        "moment_strength": selected_moment["strength"],
        "source_mu": source_mu,
        "source_sigma": source_sigma,
        "target_mu": target_mu,
        "target_sigma": target_sigma,
        "pose_matrix": pose_choice["matrix"],
        "pose_bias": pose_choice["bias"],
        "joint_bias": pose_choice["joint_bias"],
        "root_matrix": root_choice["matrix"],
        "root_bias": root_choice["bias"],
        "class_scale": class_choice["scale"],
        "class_bias": class_choice["bias"],
        "risk_scale": risk_choice["scale"],
        "risk_bias": risk_choice["bias"],
    }, args.calibration_state)
    print(json.dumps({
        "output": str(args.output),
        "calibration_state": str(args.calibration_state),
        "split": split_report,
        "selected": report["selected"],
        "held_out_test": report["held_out_test"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
