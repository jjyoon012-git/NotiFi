"""Shared observability metrics and probe helpers for source-only experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import PoseDataset
from ..nets import GraphPoseNet
from ..trainer import fit_norm
from .evaluate_sealed import smooth_valid


def pose_only(dataset: PoseDataset) -> PoseDataset:
    keep = dataset.index.task.to_numpy() == C.TASK_POSE
    valid = np.asarray(dataset.cache.arrays["valid"][dataset.rows]).any(1)
    rows = dataset.rows[np.flatnonzero(keep & valid)]
    return PoseDataset(
        rows, dataset.cache, dataset.link_ok, train=False, seed=dataset.seed,
        baseline=dataset.baseline,
    )


class ShuffledSignalDataset(Dataset):
    def __init__(self, target: PoseDataset, seed: int):
        self.target = target
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(target))
        if len(target) > 1 and np.all(permutation == np.arange(len(target))):
            permutation = np.roll(permutation, 1)
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict:
        target = self.target[index]
        signal = self.target[int(self.permutation[index])]
        target["csi"] = signal["csi"]
        target["link_mask"] = signal["link_mask"]
        return target


def finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(C.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


@torch.no_grad()
def evaluate_predictions(pred_pose: torch.Tensor, pred_root: torch.Tensor,
                         batch: dict, smooth_window: int) -> list[dict]:
    valid = batch["valid"].bool()
    pred_pose = smooth_valid(pred_pose.float().cpu(), valid, smooth_window)
    pred_root = smooth_valid(pred_root.float().cpu(), valid, smooth_window)
    target_pose = batch["pose_rel"].float()
    target_root = batch["root"].float()
    rows = []
    for item in range(len(pred_pose)):
        mask = valid[item]
        pair = mask[1:] & mask[:-1]
        pose_error = torch.linalg.vector_norm(
            pred_pose[item] - target_pose[item], dim=-1
        )
        root_error = torch.linalg.vector_norm(
            pred_root[item] - target_root[item], dim=-1
        )
        target_speed = torch.linalg.vector_norm(
            target_pose[item, 1:] - target_pose[item, :-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        predicted_speed = torch.linalg.vector_norm(
            pred_pose[item, 1:] - pred_pose[item, :-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        dynamic = pair & (target_speed > 0.25)
        target_motion = float(target_speed[pair].mean()) if pair.any() else math.nan
        predicted_motion = float(predicted_speed[pair].mean()) if pair.any() else math.nan
        rows.append({
            "mpjpe_m": float(pose_error[mask].mean()),
            "dynamic_mpjpe_m": (
                float(pose_error[1:][dynamic].mean()) if dynamic.any() else math.nan
            ),
            "root_error_m": float(root_error[mask].mean()),
            "pose_speed_ratio": (
                predicted_motion / max(target_motion, 1e-6) if pair.any() else math.nan
            ),
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    return {
        key: finite_mean([row[key] for row in rows])
        for key in rows[0]
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, dataset: Dataset, device: str,
                   batch_size: int, smooth_window: int = 5) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        rows.extend(evaluate_predictions(
            output["pose_rel"], output["root"], batch, smooth_window
        ))
    return aggregate(rows)


def mean_pose_baseline(train: PoseDataset, test: PoseDataset,
                       batch_size: int, smooth_window: int) -> dict:
    arrays = train.cache.arrays
    pose_sum = np.zeros((C.N_JOINTS, 3), dtype=np.float64)
    root_sum = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(train.rows), 64):
        rows = train.rows[start:start + 64]
        valid = np.asarray(arrays["valid"][rows], dtype=bool)
        pose = np.asarray(arrays["pose_rel"][rows], dtype=np.float32)
        root = np.asarray(arrays["root"][rows], dtype=np.float32)
        pose_sum += pose[valid].sum(0)
        root_sum += root[valid].sum(0)
        count += int(valid.sum())
    mean_pose = torch.tensor(pose_sum / max(count, 1), dtype=torch.float32)
    mean_root = torch.tensor(root_sum / max(count, 1), dtype=torch.float32)

    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = []
    for batch in loader:
        batch_size_now, frames = batch["pose_rel"].shape[:2]
        predicted_pose = mean_pose[None, None].expand(batch_size_now, frames, -1, -1)
        predicted_root = mean_root[None, None].expand(batch_size_now, frames, -1)
        metrics.extend(evaluate_predictions(
            predicted_pose, predicted_root, batch, smooth_window
        ))
    return aggregate(metrics)


@dataclass
class ProbeFrames:
    feature: torch.Tensor
    speed: torch.Tensor
    moving: torch.Tensor
    phase: torch.Tensor
    phase_valid: torch.Tensor
    impact: torch.Tensor
    danger: torch.Tensor
    trial: torch.Tensor
    frame: torch.Tensor


@torch.no_grad()
def extract_probe_frames(model: nn.Module, dataset: PoseDataset, device: str,
                         batch_size: int, max_frames_per_trial: int | None,
                         seed: int) -> ProbeFrames:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rng = np.random.default_rng(seed)
    values: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "feature", "speed", "moving", "phase", "phase_valid",
            "impact", "danger", "trial", "frame",
        )
    }
    trial_offset = 0
    model.eval()
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        feature = output["temporal_features"].float().cpu()
        valid = batch["valid"].bool()
        speed, _ = L.target_motion(
            batch["pose_rel"].float(), batch["root"].float(), valid
        )
        phase, phase_valid = L.phase_targets(speed, valid, batch["risk_id"])
        impact = L.impact_window(
            batch["pose_rel"].float(), batch["root"].float(), valid,
            batch["risk_id"],
        )
        for item in range(len(feature)):
            positions = torch.nonzero(valid[item], as_tuple=False).flatten()
            if max_frames_per_trial and len(positions) > max_frames_per_trial:
                chosen = rng.choice(
                    positions.numpy(), size=max_frames_per_trial, replace=False
                )
                positions = torch.from_numpy(np.sort(chosen)).long()
            n_frames = len(positions)
            values["feature"].append(feature[item, positions])
            values["speed"].append(speed[item, positions])
            values["moving"].append((speed[item, positions] > 0.25).float())
            values["phase"].append(phase[item, positions])
            values["phase_valid"].append(phase_valid[item, positions])
            values["impact"].append(impact[item, positions].float())
            values["danger"].append(
                torch.full((n_frames,), int(batch["risk_id"][item]) == 2,
                           dtype=torch.bool)
            )
            values["trial"].append(
                torch.full((n_frames,), trial_offset + item, dtype=torch.long)
            )
            values["frame"].append(positions)
        trial_offset += len(feature)
    return ProbeFrames(**{key: torch.cat(items) for key, items in values.items()})


class MotionObservabilityProbe(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.1)
        )
        self.speed = nn.Linear(64, 1)
        self.moving = nn.Linear(64, 1)
        self.phase = nn.Linear(64, 4)
        self.impact = nn.Linear(64, 1)

    def forward(self, feature: torch.Tensor) -> dict:
        hidden = self.shared(feature)
        return {
            "speed": self.speed(hidden).squeeze(-1),
            "moving": self.moving(hidden).squeeze(-1),
            "phase": self.phase(hidden),
            "impact": self.impact(hidden).squeeze(-1),
        }


def positive_weight(target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    selected = target if mask is None else target[mask]
    positives = float(selected.sum())
    negatives = float(len(selected) - positives)
    return min(20.0, negatives / max(positives, 1.0))


def train_probe(frames: ProbeFrames, hidden: int, device: str,
                epochs: int, batch_size: int) -> MotionObservabilityProbe:
    model = MotionObservabilityProbe(hidden).to(device)
    dataset = TensorDataset(
        frames.feature, frames.speed, frames.moving, frames.phase,
        frames.phase_valid, frames.impact, frames.danger,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    moving_weight = torch.tensor(
        positive_weight(frames.moving), device=device
    )
    impact_weight = torch.tensor(
        positive_weight(frames.impact, frames.danger), device=device
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running = count = 0
        for feature, speed, moving, phase, phase_valid, impact, danger in loader:
            feature = feature.to(device)
            speed = speed.to(device)
            moving = moving.to(device)
            phase = phase.to(device)
            phase_valid = phase_valid.to(device)
            impact = impact.to(device)
            danger = danger.to(device)
            output = model(feature)
            speed_target = torch.log1p(speed * 10.0)
            loss = F.smooth_l1_loss(output["speed"], speed_target, beta=0.10)
            loss = loss + F.binary_cross_entropy_with_logits(
                output["moving"], moving, pos_weight=moving_weight
            )
            if phase_valid.any():
                loss = loss + 0.5 * F.cross_entropy(
                    output["phase"][phase_valid], phase[phase_valid]
                )
            if danger.any():
                loss = loss + F.binary_cross_entropy_with_logits(
                    output["impact"][danger], impact[danger],
                    pos_weight=impact_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(feature)
            count += len(feature)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  probe ep {epoch:2d} loss={running / max(count, 1):.4f}")
    return model


def binary_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict:
    predicted = probability >= 0.5
    target = target.bool()
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
    }


def macro_f1(predicted: torch.Tensor, target: torch.Tensor, classes: int) -> float:
    scores = []
    for label in range(classes):
        scores.append(binary_metrics(
            (predicted == label).float(), (target == label).float()
        )["f1"])
    return float(np.mean(scores))


@torch.no_grad()
def evaluate_probe(model: MotionObservabilityProbe, frames: ProbeFrames,
                   device: str, batch_size: int) -> dict:
    model.eval()
    outputs = {key: [] for key in ("speed", "moving", "phase", "impact")}
    loader = DataLoader(frames.feature, batch_size=batch_size, shuffle=False)
    for feature in loader:
        prediction = model(feature.to(device))
        for key, value in prediction.items():
            outputs[key].append(value.float().cpu())
    outputs = {key: torch.cat(value) for key, value in outputs.items()}
    predicted_speed = torch.expm1(outputs["speed"]).clamp_min(0.0) / 10.0
    speed_target = frames.speed
    residual = ((predicted_speed - speed_target) ** 2).sum()
    total = ((speed_target - speed_target.mean()) ** 2).sum().clamp_min(1e-8)
    correlation = torch.corrcoef(torch.stack((predicted_speed, speed_target)))[0, 1]
    moving = binary_metrics(torch.sigmoid(outputs["moving"]), frames.moving)
    phase_mask = frames.phase_valid
    phase_f1 = macro_f1(
        outputs["phase"][phase_mask].argmax(-1), frames.phase[phase_mask], 4
    ) if phase_mask.any() else math.nan
    danger = frames.danger
    impact_metrics = binary_metrics(
        torch.sigmoid(outputs["impact"])[danger], frames.impact[danger]
    ) if danger.any() else {}

    timing_errors = []
    impact_probability = torch.sigmoid(outputs["impact"])
    for trial in torch.unique(frames.trial):
        selected = frames.trial == trial
        target_impact = frames.impact[selected].bool()
        if not target_impact.any():
            continue
        local_probability = impact_probability[selected]
        local_frames = frames.frame[selected]
        predicted_frame = int(local_frames[torch.argmax(local_probability)])
        target_frame = float(local_frames[target_impact].float().mean())
        timing_errors.append(abs(predicted_frame - target_frame))
    return {
        "n_frames": len(frames.feature),
        "speed_mae_mps": float((predicted_speed - speed_target).abs().mean()),
        "speed_r2": float(1.0 - residual / total),
        "speed_correlation": float(correlation),
        "moving": moving,
        "phase_macro_f1": phase_f1,
        "impact": impact_metrics,
        "impact_timing_mae_frames": finite_mean(timing_errors),
    }


def dynamic_score_for_rows(dataset: PoseDataset) -> np.ndarray:
    arrays = dataset.cache.arrays
    scores = np.zeros(len(dataset.rows), dtype=np.float32)
    for index, row in enumerate(dataset.rows):
        valid = np.asarray(arrays["valid"][row], dtype=bool)
        pose = np.asarray(arrays["pose_rel"][row], dtype=np.float32)
        root = np.asarray(arrays["root"][row], dtype=np.float32)
        absolute = pose + root[:, None]
        pair = valid[1:] & valid[:-1]
        if pair.any():
            speed = np.linalg.norm(absolute[1:] - absolute[:-1], axis=-1).mean(-1)
            scores[index] = float(speed[pair].mean() * C.TARGET_FPS)
    return scores


def overfit_probe(source: PoseDataset, trials: int, steps: int,
                  device: str, smooth_window: int, seed: int) -> dict:
    score = dynamic_score_for_rows(source)
    selected = np.argsort(-score)[:trials]
    rows = source.rows[selected]
    subset = PoseDataset(
        rows, source.cache, source.link_ok, train=False, seed=seed,
        baseline=source.baseline,
    )
    loader = DataLoader(
        subset, batch_size=min(trials, 5), shuffle=True, num_workers=0
    )
    model = GraphPoseNet(
        hidden=64, n_blocks=1, heads=4, graph_blocks=1,
        decoder="hybrid", dropout=0.0,
    ).to(device)
    fit_norm(model, loader, device, max_batches=max(1, len(loader)))
    criterion = L.PoseLoss(
        lambda_root=1.0, lambda_bone=0.1, lambda_cls=0.0, lambda_risk=0.0,
        lambda_velocity=0.1, lambda_acceleration=0.001,
        lambda_displacement=0.1, motion_weight=3.0, device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    iterator = iter(loader)
    model.train()
    last_loss = math.nan
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(batch["csi"], batch["link_mask"])
        loss, _ = criterion(output, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach())
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"  overfit n={trials:2d} step={step:3d} loss={last_loss:.4f}")
    metrics = evaluate_model(
        model, subset, device, batch_size=min(trials, 5),
        smooth_window=smooth_window,
    )
    metrics.update({
        "trials": trials, "steps": steps, "last_train_loss": last_loss,
        "trial_ids": subset.index.trial_id.tolist(),
    })
    return metrics

# Sealed-test execution intentionally lives only in evaluate_sealed.py.
