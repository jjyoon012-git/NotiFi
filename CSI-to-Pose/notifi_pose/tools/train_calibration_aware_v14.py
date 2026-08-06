"""Train V14 with deployment-style absence/basic-pose support episodes."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .. import contract as C
from ..calibration_aware import CalibrationAwareV14
from ..dataio.dataset import DropoutConfig, PoseDataset, build_datasets
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_v11_final import evaluate_pa_mpjpe
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import (
    DISTAL_JOINTS,
    evaluate_classification,
    evaluate_trajectory,
)


BASIC_CLASSES = (1, 2, 3)
META_VALIDATION_SITES = ("ajh_E03", "mhw_E03")


def subset_dataset(dataset: PoseDataset, positions: np.ndarray,
                   train: bool = False) -> PoseDataset:
    rows = dataset.rows[np.asarray(positions, dtype=np.int64)]
    return PoseDataset(
        rows,
        dataset.cache,
        dataset.link_ok,
        train=train,
        dropout=DropoutConfig(p=0.10, max_drop=1),
        seed=dataset.seed,
        baseline=dataset.baseline,
    )


def site_name(index) -> np.ndarray:
    return (index.subject.astype(str) + "_" + index.environment.astype(str)).to_numpy()


def split_support_queries(dataset: PoseDataset, per_pose: int,
                          seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    sites = site_name(dataset.index)
    labels = dataset.index.class_id.to_numpy(dtype=np.int64)
    support: dict[str, np.ndarray] = {}
    query: dict[str, np.ndarray] = {}
    for site in sorted(set(sites)):
        selected: list[int] = []
        site_positions = np.flatnonzero(sites == site)
        for class_id in BASIC_CLASSES:
            candidates = site_positions[labels[site_positions] == class_id].copy()
            rng.shuffle(candidates)
            if len(candidates) < per_pose:
                raise RuntimeError(
                    f"{site} class {class_id} has {len(candidates)}, needs {per_pose}"
                )
            selected.extend(int(item) for item in candidates[:per_pose])
        support[site] = np.sort(np.asarray(selected, dtype=np.int64))
        query[site] = np.asarray(
            [item for item in site_positions if item not in set(selected)],
            dtype=np.int64,
        )
    return support, query


@torch.no_grad()
def _moments(dataset: PoseDataset, positions: np.ndarray,
             class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    selected = positions[
        dataset.index.iloc[positions].class_id.to_numpy(dtype=np.int64) == class_id
    ]
    loader = DataLoader(subset_dataset(dataset, selected), batch_size=3, shuffle=False)
    total = square = count = None
    for batch in loader:
        csi = batch["csi"].double()
        weight = batch["link_mask"][..., None, None].double()
        batch_total = (csi * weight).sum((0, 1))
        batch_square = (csi.square() * weight).sum((0, 1))
        batch_count = weight.sum((0, 1))
        total = batch_total if total is None else total + batch_total
        square = batch_square if square is None else square + batch_square
        count = batch_count if count is None else count + batch_count
    mean = total / count.clamp_min(1.0)
    variance = square / count.clamp_min(1.0) - mean.square()
    return mean.float(), variance.clamp_min(1e-8).sqrt().float()


@torch.no_grad()
def support_profile(dataset: PoseDataset, positions: np.ndarray,
                    site: str) -> torch.Tensor:
    baseline = dataset.baseline.table.get(site)
    if baseline is None:
        raise RuntimeError(f"absence baseline is missing for {site}")
    absence = torch.from_numpy(np.asarray(baseline[0], dtype=np.float32))
    channels = [absence[..., 0], absence[..., 1]]
    for class_id in BASIC_CLASSES:
        mean, std = _moments(dataset, positions, class_id)
        channels.extend((mean[..., 0], mean[..., 1], std[..., 0], std[..., 1]))
    return torch.stack(channels, dim=-1)


def build_profiles(dataset: PoseDataset, support: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        site: support_profile(dataset, positions, site)
        for site, positions in support.items()
    }


def profile_normalization(profiles: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack(list(profiles.values()))
    mean = values.mean((0, 1, 2))
    std = values.std((0, 1, 2), unbiased=False).clamp_min(1e-4)
    return mean, std


def episode_augment(csi: torch.Tensor, profile: torch.Tensor,
                    probability: float = 0.70) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand((), device=csi.device) >= probability:
        return csi, profile
    links = csi.shape[2]
    gain = torch.exp(torch.randn(links, device=csi.device) * 0.20)
    curvature_strength = torch.randn(links, device=csi.device) * 0.20
    frequency = torch.linspace(-1.0, 1.0, csi.shape[3], device=csi.device)
    curvature = frequency.square() - frequency.square().mean()
    ripple = curvature_strength[:, None] * curvature[None]
    augmented = csi.clone()
    augmented[..., 0] *= gain[None, None, :, None]
    augmented[..., 1] += ripple[None, None]

    adjusted = profile.clone()
    amplitude_features = (0, 2, 4, 6, 8, 10, 12)
    phase_mean_features = (1, 3, 7, 11)
    adjusted[..., list(amplitude_features)] *= gain[:, None, None]
    adjusted[..., list(phase_mean_features)] += ripple[..., None]
    return augmented, adjusted


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.expand_as(values).sum().clamp_min(1.0)


def calibration_loss(output: dict, batch: dict,
                     class_weight: torch.Tensor,
                     risk_weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid"].bool()
    danger = batch["risk_id"].eq(C.N_RISK - 1)
    frame_weight = 1.0 + 2.0 * danger[:, None].to(output["pose_rel"].dtype)
    pose_distance = torch.linalg.vector_norm(
        output["pose_rel"] - batch["pose_rel"], dim=-1
    )
    pose = _masked_mean(pose_distance * frame_weight[..., None], valid)
    distal = _masked_mean(
        pose_distance[..., list(DISTAL_JOINTS)] * frame_weight[..., None], valid
    )
    root_distance = torch.linalg.vector_norm(output["root"] - batch["root"], dim=-1)
    root = _masked_mean(root_distance * frame_weight, valid)

    pair = valid[:, 1:] & valid[:, :-1]
    predicted_velocity = output["pose_rel"][:, 1:] - output["pose_rel"][:, :-1]
    target_velocity = batch["pose_rel"][:, 1:] - batch["pose_rel"][:, :-1]
    velocity = _masked_mean(
        torch.linalg.vector_norm(predicted_velocity - target_velocity, dim=-1),
        pair,
    )
    class_loss = F.cross_entropy(
        output["class_logits"], batch["class_id"], weight=class_weight
    )
    risk_loss = F.cross_entropy(
        output["risk_logits"], batch["risk_id"], weight=risk_weight
    )
    total = (
        pose + 0.25 * distal + 0.55 * root + 0.15 * velocity
        + 0.08 * class_loss + 0.12 * risk_loss
    )
    return total, {
        "loss": float(total.detach()),
        "pose": float(pose.detach()),
        "distal": float(distal.detach()),
        "root": float(root.detach()),
        "velocity": float(velocity.detach()),
        "class": float(class_loss.detach()),
        "risk": float(risk_loss.detach()),
    }


class ActiveSupportModel(nn.Module):
    def __init__(self, model: CalibrationAwareV14):
        super().__init__()
        self.model = model
        self.register_buffer("profile", torch.empty(0), persistent=False)

    def set_profile(self, profile: torch.Tensor) -> None:
        self.profile = profile.to(next(self.model.parameters()).device)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        return self.model(csi, link_mask, self.profile)


def _classification_summary(confusion: torch.Tensor) -> dict:
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(0).clamp(min=1).float()
    recall = true_positive / confusion.sum(1).clamp(min=1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    present = confusion.sum(1) > 0
    return {
        "accuracy": float(true_positive.sum() / confusion.sum().clamp(min=1)),
        "macro_f1": float(f1[present].mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": confusion.sum(1).tolist(),
        "confusion": confusion.tolist(),
    }


def combine_classification(results: list[dict]) -> dict:
    class_confusion = sum(
        (torch.tensor(item["class"]["confusion"], dtype=torch.long) for item in results),
        torch.zeros(C.N_CLASSES, C.N_CLASSES, dtype=torch.long),
    )
    risk_confusion = sum(
        (torch.tensor(item["risk"]["confusion"], dtype=torch.long) for item in results),
        torch.zeros(C.N_RISK, C.N_RISK, dtype=torch.long),
    )
    class_result = _classification_summary(class_confusion)
    risk_result = _classification_summary(risk_confusion)
    danger = C.N_RISK - 1
    risk_result.update({
        "danger_recall": risk_result["recall"][danger],
        "danger_precision": risk_result["precision"][danger],
        "danger_f1": risk_result["f1"][danger],
        "danger_tp": int(risk_confusion[danger, danger]),
        "danger_support": int(risk_confusion[danger].sum()),
        "safe_to_danger": int(risk_confusion[0, danger]),
        "safe_to_danger_rate": float(
            risk_confusion[0, danger] / risk_confusion[0].sum().clamp(min=1)
        ),
    })
    return {"class": class_result, "risk": risk_result, "danger_logit_bias": 0.0}


def combine_trajectory(results: list[tuple[int, dict]]) -> dict:
    total_trials = sum(count for count, _ in results)
    danger_trials = sum(int(item.get("danger_trials", 0)) for _, item in results)
    combined = {}
    ignored = {"danger_by_class", "danger_trials"}
    keys = set.intersection(*(set(item) for _, item in results)) - ignored
    for key in keys:
        if not isinstance(results[0][1][key], (float, int)):
            continue
        weights = [
            int(item.get("danger_trials", 0)) if key.startswith("danger_") else count
            for count, item in results
        ]
        denominator = sum(weights)
        if denominator:
            combined[key] = sum(
                weight * float(item[key])
                for weight, (_, item) in zip(weights, results)
                if math.isfinite(float(item[key]))
            ) / denominator
    combined["danger_trials"] = danger_trials
    combined["trials"] = total_trials
    return combined


@torch.no_grad()
def evaluate_sites(model: CalibrationAwareV14, dataset: PoseDataset,
                   profiles: dict[str, torch.Tensor], sites: tuple[str, ...],
                   batch_size: int, device: str, max_shift: int) -> dict:
    active = ActiveSupportModel(model).to(device).eval()
    names = site_name(dataset.index)
    trajectory = []
    classifications = []
    for site in sites:
        positions = np.flatnonzero(names == site)
        site_dataset = subset_dataset(dataset, positions)
        loader = DataLoader(site_dataset, batch_size=batch_size, shuffle=False)
        active.set_profile(profiles[site])
        trajectory.append((len(site_dataset), evaluate_trajectory(
            active, loader, device, max_shift
        )))
        classifications.append(evaluate_classification(active, loader, device, 0.0))
    return {
        "trajectory": combine_trajectory(trajectory),
        "classification": combine_classification(classifications),
    }


def selection_score(result: dict) -> float:
    t = result["trajectory"]
    c = result["classification"]
    return float(
        t["mpjpe_m"] + 0.50 * t["root_error_m"]
        + 0.75 * t["danger_mpjpe_m"]
        + 0.30 * t["danger_pose_endpoint_mpjpe_m"]
        + 0.12 * (1.0 - c["risk"]["danger_recall"])
        + 0.05 * (1.0 - c["risk"]["macro_f1"])
        + 0.10 * abs(math.log(max(t["pose_speed_ratio"], 1e-3)))
    )


def _weights(dataset: PoseDataset, column: str, classes: int,
             device: str, danger_boost: float = 1.0) -> torch.Tensor:
    labels = dataset.index[column].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    values = counts.sum() / (classes * np.maximum(counts, 1.0))
    if column == "risk_id":
        values[-1] *= danger_boost
    return torch.tensor(values, dtype=torch.float32, device=device)


def adapter_state(model: CalibrationAwareV14) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("base.")
    }


def load_adapter_state(model: CalibrationAwareV14, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid = [key for key in missing if not key.startswith("base.")]
    if invalid or unexpected:
        raise RuntimeError(f"invalid adapter state: missing={invalid}, unexpected={unexpected}")


def build_model(args, device: str, support_mean: torch.Tensor,
                support_std: torch.Tensor) -> tuple[CalibrationAwareV14, dict]:
    root_lock = _read_locked(args.root_calibration, args.source_exp)
    class_lock = _read_locked(args.classification_calibration, args.source_exp)
    base_args = argparse.Namespace(
        p2_checkpoint=args.p2_checkpoint, exp=args.source_exp
    )
    base, configuration = build_locked_model(
        base_args, device, root_lock, class_lock
    )
    model = CalibrationAwareV14(base, hidden=args.hidden).to(device)
    model.set_support_normalization(
        support_mean.to(device), support_std.to(device)
    )
    return model, configuration


def train_adapter(model: CalibrationAwareV14, train_dataset: PoseDataset,
                  queries: dict[str, np.ndarray], profiles: dict[str, torch.Tensor],
                  sites: tuple[str, ...], epochs: int, args, device: str,
                  validation_dataset: PoseDataset | None = None,
                  validation_sites: tuple[str, ...] = ()) -> tuple[dict, list[dict]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    combined_positions = np.concatenate([queries[site] for site in sites])
    weight_dataset = subset_dataset(train_dataset, combined_positions)
    class_weight = _weights(weight_dataset, "class_id", C.N_CLASSES, device)
    risk_weight = _weights(
        weight_dataset, "risk_id", C.N_RISK, device, danger_boost=2.0
    )
    best = adapter_state(model)
    best_score = math.inf
    best_epoch = 0
    history = []
    if validation_dataset is not None:
        baseline = evaluate_sites(
            model, validation_dataset, profiles, validation_sites,
            args.batch_size, device, args.max_shift,
        )
        best_score = selection_score(baseline)
        history.append({"epoch": 0, "validation": baseline, "score": best_score})
    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        rng = np.random.default_rng(args.seed + epoch * 1009)
        ordered_sites = list(sites)
        rng.shuffle(ordered_sites)
        for site in ordered_sites:
            query_dataset = subset_dataset(train_dataset, queries[site], train=True)
            query_dataset.set_epoch(epoch)
            loader = DataLoader(
                query_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            raw_profile = profiles[site].to(device)
            for batch in loader:
                csi = batch["csi"].to(device, non_blocking=True)
                mask = batch["link_mask"].to(device, non_blocking=True)
                profile = raw_profile
                csi, profile = episode_augment(csi, profile)
                moved = {
                    key: value.to(device, non_blocking=True)
                    if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                output = model(csi, mask, profile)
                loss, parts = calibration_loss(
                    output, moved, class_weight, risk_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                batches += 1
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + value
        row = {
            "epoch": epoch,
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
        }
        if validation_dataset is not None:
            validation = evaluate_sites(
                model, validation_dataset, profiles, validation_sites,
                args.batch_size, device, args.max_shift,
            )
            score = selection_score(validation)
            row.update({"validation": validation, "score": score})
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best = adapter_state(model)
        else:
            best = adapter_state(model)
            best_epoch = epoch
        history.append(row)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": row["train"].get("loss"),
            "validation_score": row.get("score"),
            "best_epoch": best_epoch,
        }, ensure_ascii=False), flush=True)
    load_adapter_state(model, best)
    return {"state": best, "epoch": best_epoch, "score": best_score}, history


def select_strengths(model: CalibrationAwareV14, validation: PoseDataset,
                     profiles: dict[str, torch.Tensor], sites: tuple[str, ...],
                     args, device: str) -> tuple[dict, list[dict]]:
    candidates = []
    selected = {"pose": 0.0, "root": 0.0, "classification": 0.0, "risk": 0.0}

    def evaluate(values: dict) -> dict:
        model.set_strengths(**values)
        return evaluate_sites(
            model, validation, profiles, sites,
            args.batch_size, device, args.max_shift,
        )

    for component in ("pose", "root", "classification", "risk"):
        options = []
        for strength in (0.0, 0.5, 1.0):
            values = dict(selected)
            values[component] = strength
            result = evaluate(values)
            if component == "pose":
                t = result["trajectory"]
                score = (
                    t["mpjpe_m"] + 0.75 * t["danger_pose_mpjpe_m"]
                    + 0.35 * t["danger_pose_distal_mpjpe_m"]
                    + 0.30 * t["danger_pose_endpoint_mpjpe_m"]
                    + 0.10 * abs(math.log(max(t["pose_speed_ratio"], 1e-3)))
                )
            elif component == "root":
                t = result["trajectory"]
                score = (
                    t["root_error_m"] + 0.75 * t["danger_root_error_m"]
                    + 0.35 * t["danger_endpoint_mpjpe_m"]
                )
            elif component == "classification":
                score = 1.0 - result["classification"]["class"]["macro_f1"]
            else:
                risk = result["classification"]["risk"]
                score = (
                    1.0 - risk["macro_f1"]
                    + 0.75 * (1.0 - risk["danger_recall"])
                    + 0.10 * risk["safe_to_danger_rate"]
                )
            options.append({
                "component": component, "strength": strength,
                "score": float(score), "result": result,
            })
        choice = min(options, key=lambda item: item["score"])
        selected[component] = choice["strength"]
        candidates.extend(options)
    model.set_strengths(**selected)
    return selected, candidates


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
    parser.add_argument("--target-fold", default="yja_E02")
    parser.add_argument("--support-per-pose", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("work_v2/runs/v14_calibration_aware"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("docs/results/v14_calibration_aware_yja.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = build_datasets(
        exp=args.source_exp, baseline="sub", seed=args.seed
    )
    source_train = pose_only(source["train"])
    source_val = pose_only(source["val"])
    source_support, source_query = split_support_queries(
        source_train, args.support_per_pose, args.seed
    )
    source_profiles = build_profiles(source_train, source_support)
    support_mean, support_std = profile_normalization(source_profiles)
    all_sites = tuple(sorted(source_profiles))
    meta_val_sites = tuple(site for site in META_VALIDATION_SITES if site in all_sites)
    meta_train_sites = tuple(site for site in all_sites if site not in meta_val_sites)
    if len(meta_val_sites) != len(META_VALIDATION_SITES):
        raise RuntimeError(f"meta-validation sites are missing: {meta_val_sites}")

    model, base_configuration = build_model(
        args, device, support_mean, support_std
    )
    meta_best, meta_history = train_adapter(
        model, source_train, source_query, source_profiles,
        meta_train_sites, args.epochs, args, device,
        validation_dataset=source_val,
        validation_sites=meta_val_sites,
    )
    selected_strengths, strength_candidates = select_strengths(
        model, source_val, source_profiles, meta_val_sites, args, device
    )
    selected_epoch = int(meta_best["epoch"])

    del model
    torch.cuda.empty_cache()
    set_seed(args.seed + 1)
    final_model, _ = build_model(args, device, support_mean, support_std)
    final_best, final_history = train_adapter(
        final_model, source_train, source_query, source_profiles,
        all_sites, selected_epoch, args, device,
    )
    final_model.set_strengths(**selected_strengths)
    source_validation = evaluate_sites(
        final_model, source_val, source_profiles, all_sites,
        args.batch_size, device, args.max_shift,
    )

    target = pose_only(build_datasets(
        exp="sealed", fold=args.target_fold, baseline="sub", seed=args.seed
    )["test"])
    target_support, target_query = split_support_queries(
        target, args.support_per_pose, args.seed
    )
    if set(target_support) != {"yja_E02"}:
        raise RuntimeError(f"unexpected target sites: {set(target_support)}")
    target_profile = support_profile(
        target, target_support["yja_E02"], "yja_E02"
    )
    target_test = subset_dataset(target, target_query["yja_E02"])
    target_loader = DataLoader(
        target_test, batch_size=args.batch_size, shuffle=False
    )
    active = ActiveSupportModel(final_model).to(device).eval()
    active.set_profile(target_profile)
    calibrated_target = {
        "trajectory": evaluate_trajectory(
            active, target_loader, device, args.max_shift
        ),
        "classification": evaluate_classification(
            active, target_loader, device, 0.0
        ),
    }
    active.model.set_strengths(0.0, 0.0, 0.0, 0.0)
    frozen_target = {
        "trajectory": evaluate_trajectory(
            active, target_loader, device, args.max_shift
        ),
        "classification": evaluate_classification(
            active, target_loader, device, 0.0
        ),
    }
    active.model.set_strengths(**selected_strengths)
    calibrated_target["trajectory"]["pa_mpjpe_m"] = evaluate_pa_mpjpe(
        active, target_loader, device
    )
    active.model.set_strengths(0.0, 0.0, 0.0, 0.0)
    frozen_target["trajectory"]["pa_mpjpe_m"] = evaluate_pa_mpjpe(
        active, target_loader, device
    )
    active.model.set_strengths(**selected_strengths)

    report = {
        "run": "v14_calibration_aware",
        "source_protocol": args.source_exp,
        "target_protocol": f"sealed/{args.target_fold}",
        "selection_protocol": {
            "meta_train_sites": list(meta_train_sites),
            "meta_validation_sites": list(meta_val_sites),
            "target_used_for_selection": False,
        },
        "support_protocol": {
            "classes": {"1": "standing", "2": "sitting", "3": "lying"},
            "trials_per_pose": args.support_per_pose,
            "warning_or_danger_support": 0,
            "target_gt_used_for_calibration": False,
            "target_support_trials": len(target_support["yja_E02"]),
            "target_test_trials": len(target_test),
            "target_danger_test_trials": int((target_test.index.risk_id == 2).sum()),
        },
        "base_configuration": base_configuration,
        "selected_epoch": selected_epoch,
        "selected_strengths": selected_strengths,
        "meta_history": meta_history,
        "strength_candidates": strength_candidates,
        "final_history": final_history,
        "source_validation": source_validation,
        "target_test": {
            "frozen_v13s": frozen_target,
            "calibration_aware_v14": calibrated_target,
        },
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "run": report["run"],
        "source_protocol": args.source_exp,
        "target_protocol": f"sealed/{args.target_fold}",
        "base_configuration": base_configuration,
        "support_mean": support_mean,
        "support_std": support_std,
        "selected_epoch": selected_epoch,
        "selected_strengths": selected_strengths,
        "adapter": adapter_state(final_model),
    }
    torch.save(checkpoint, args.run_dir / "best_model.pt")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "report": str(args.report),
        "checkpoint": str(args.run_dir / "best_model.pt"),
        "selected_epoch": selected_epoch,
        "selected_strengths": selected_strengths,
        "source_validation": source_validation,
        "target_test": report["target_test"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
