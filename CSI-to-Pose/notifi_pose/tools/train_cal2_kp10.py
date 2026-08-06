"""Train CAL2-KP10 raw-link calibration with held-site selection."""

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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..cal2_kp10 import (
    RawCalibratedKP4,
    RawLinkCanonicalizer,
    support_statistics,
    support_trial_descriptors,
)
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_kp10_paired_bootstrap import kp10_prediction
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
)
from .diagnose_observability import pose_only, report_path
from .evaluate_motion_retrieval_pose import _load_model
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_cal1_kp10 import (
    META_TRAIN_SITES,
    META_VALIDATION_SITES,
    add_paths,
    classifier_evaluation,
    configure_work_root,
    prepare_custom,
    site_names,
    slice_cache,
    split_support_query,
)
from .train_calibration_aware_v14 import subset_dataset
from .train_csi_motion_profile import speed_targets
from .train_csi_part_motion_profile import part_speed_targets
from .train_dynamic_motion import classification_metrics
from .train_kinetic_pose import (
    CoarsePoseStore,
    DISTAL_JOINTS,
    load_or_create_coarse_store,
    pose_selection_score,
)
from .train_motion_retrieval_selector import (
    motion_descriptor,
    project_motion,
)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float):
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


class SiteDiscriminator(nn.Module):
    def __init__(self, input_dim: int, sites: int, hidden: int = 96):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden, sites),
        )

    def forward(self, features: torch.Tensor,
                reversal_strength: float) -> torch.Tensor:
        reversed_features = _GradientReversal.apply(
            features, reversal_strength
        )
        return self.classifier(reversed_features)


def support_summary(dataset, support: dict[str, np.ndarray],
                    sites: tuple[str, ...]) -> dict[str, torch.Tensor]:
    statistics = {"mean": [], "std": [], "dynamic": []}
    descriptors, labels = [], []
    for site in sites:
        samples = [dataset[int(position)] for position in support[site]]
        csi = torch.stack([sample["csi"] for sample in samples]).float()
        mask = torch.stack([sample["link_mask"] for sample in samples]).bool()
        current = support_statistics(csi, mask)
        for key in statistics:
            statistics[key].append(current[key])
        descriptors.append(support_trial_descriptors(csi[None], mask[None])[0])
        labels.append(torch.tensor(
            dataset.index.iloc[support[site]].class_id.to_numpy(dtype=np.int64)
        ))
    return {
        **{key: torch.stack(values) for key, values in statistics.items()},
        "descriptors": torch.stack(descriptors),
        "classes": torch.stack(labels).long(),
    }


def fit_reference(summary: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mean = summary["mean"].median(0).values
    std = summary["std"].median(0).values.clamp_min(1e-4)
    dynamic = summary["dynamic"].median(0).values.clamp_min(1e-4)
    return {"mean": mean, "std": std, "dynamic": dynamic}


def move_summary(summary: dict[str, torch.Tensor], device: str) -> dict:
    return {key: value.to(device) for key, value in summary.items()}


def context_from_summary(canonicalizer, summary, device: str) -> dict:
    moved = move_summary(summary, device)
    statistics = {key: moved[key] for key in ("mean", "std", "dynamic")}
    return canonicalizer.encode_summary(
        statistics, moved["descriptors"], moved["classes"]
    )


def select_context(context: dict[str, torch.Tensor],
                   site_id: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        key: value.index_select(0, site_id)
        for key, value in context.items()
    }


def row_site_ids(dataset, sites: tuple[str, ...]) -> dict[int, int]:
    lookup = {site: index for index, site in enumerate(sites)}
    names = site_names(dataset.index)
    return {
        int(row): lookup[names[position]]
        for position, row in enumerate(dataset.rows)
        if names[position] in lookup
    }


def site_ids_for_rows(rows: torch.Tensor, mapping: dict[int, int],
                      device: str) -> torch.Tensor:
    return torch.tensor(
        [mapping[int(row)] for row in rows.tolist()],
        dtype=torch.long, device=device,
    )


def build_coarse_model(work_root: Path, publish_root: Path,
                       exp: str, device: str):
    project_root = work_root.parent
    def resolved(value: str | Path) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else project_root / path)

    model_args = argparse.Namespace(
        p2_checkpoint=work_root / "runs/p2_sub_single_clean_finetune/best_model.pt",
        exp=exp,
    )
    root_lock = _read_locked(
        publish_root / "docs/results/v13s_pruned_pose_root_ensemble.json", exp
    )
    class_lock = _read_locked(
        work_root / "runs/p2_v12w_robust_classification_ensemble/validation.json",
        exp,
    )
    root_source = root_lock["source"]
    root_source["p2_checkpoint"] = resolved(root_source["p2_checkpoint"])
    root_source["pose_checkpoints"] = [
        resolved(path) for path in root_source["pose_checkpoints"]
    ]
    root_source["root_checkpoints"] = [
        resolved(path) for path in root_source["root_checkpoints"]
    ]
    class_source = class_lock["source"]
    class_source["p2_checkpoint"] = resolved(class_source["p2_checkpoint"])
    if "classification_expert_checkpoints" in class_source:
        class_source["classification_expert_checkpoints"] = [
            resolved(path)
            for path in class_source["classification_expert_checkpoints"]
        ]
    elif "classification_expert_checkpoint" in class_source:
        class_source["classification_expert_checkpoint"] = resolved(
            class_source["classification_expert_checkpoint"]
        )
    model, _ = build_locked_model(
        model_args, device, root_lock, class_lock
    )
    return model.eval()


def load_source_coarse(work_root: Path, datasets: dict, device: str,
                       batch_size: int, protocol: str) -> CoarsePoseStore:
    pose_sets = tuple(pose_only(datasets[name]) for name in ("train", "val", "test"))
    path = work_root / f"runs/kp1_v13s_coarse_{protocol}.pt"
    return load_or_create_coarse_store(
        None, pose_sets, path, device, batch_size, protocol
    )


def build_motion_targets(dataset, selector_checkpoint: dict) -> dict:
    pose, valid, action, risk = _load_pose_arrays(dataset)
    bank = torch.stack([
        _canonicalize(value, mask, C.CACHE_FRAMES)
        for value, mask in zip(pose, valid)
    ])
    return {
        "embedding": project_motion(
            motion_descriptor(bank), selector_checkpoint["pca"]
        ),
        "rows": torch.from_numpy(dataset.rows).long(),
    }


class MotionTargetStore:
    def __init__(self, target: dict):
        self.embedding = target["embedding"].float()
        self.position = {
            int(row): index for index, row in enumerate(target["rows"].tolist())
        }

    def lookup(self, rows: torch.Tensor, device: str) -> torch.Tensor:
        positions = torch.tensor([
            self.position[int(row)] for row in rows.tolist()
        ], dtype=torch.long)
        return self.embedding.index_select(0, positions).to(device)


def load_frozen_heads(args, device: str):
    selector_checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(
        **selector_checkpoint["model_config"]
    ).to(device)
    selector.load_state_dict(selector_checkpoint["model"])
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    classifier = TemporalMotionSelector(
        **classifier_checkpoint["model_config"]
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    scalar_checkpoint = torch.load(
        args.scalar_profile_checkpoint, map_location="cpu", weights_only=False
    )
    part_checkpoint = torch.load(
        args.part_profile_checkpoint, map_location="cpu", weights_only=False
    )
    from ..motion_retrieval import MotionProfileHead, PartMotionProfileHead
    scalar = MotionProfileHead(**scalar_checkpoint["model_config"]).to(device)
    scalar.load_state_dict(scalar_checkpoint["model"])
    part = PartMotionProfileHead(**part_checkpoint["model_config"]).to(device)
    part.load_state_dict(part_checkpoint["model"])
    for model in (selector, classifier, scalar, part):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return selector_checkpoint, selector, classifier, scalar, part


def masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[..., None].to(features.dtype)
    return (features * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def cross_site_motion_loss(features: torch.Tensor, action: torch.Tensor,
                           site_id: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features, dim=-1)
    similarity = normalized @ normalized.T
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = (action[:, None] == action[None]) & (
        site_id[:, None] != site_id[None]
    ) & ~eye
    negative = (action[:, None] != action[None]) & ~eye
    pull = (1.0 - similarity[positive]).mean() if positive.any() else similarity.sum() * 0
    push = F.relu(similarity[negative] - 0.25).mean() if negative.any() else similarity.sum() * 0
    return pull + 0.25 * push


def pose_losses(predicted: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor, risk: torch.Tensor) -> dict[str, torch.Tensor]:
    error = torch.linalg.vector_norm(predicted - target, dim=-1)
    weight = valid[..., None].to(error.dtype)
    trial_scale = torch.where(risk == 2, 2.0, 1.0)[:, None, None]
    pose = (error * weight * trial_scale).sum() / (
        weight * trial_scale
    ).sum().clamp_min(1.0) / C.N_JOINTS
    distal = (
        error[..., list(DISTAL_JOINTS)]
        * valid[..., None].to(error.dtype)
        * trial_scale
    ).sum() / (
        valid.sum() * len(DISTAL_JOINTS)
    ).clamp_min(1.0)
    pair = valid[:, 1:] & valid[:, :-1]
    predicted_velocity = predicted[:, 1:] - predicted[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_error = torch.linalg.vector_norm(
        predicted_velocity - target_velocity, dim=-1
    )
    velocity = (
        velocity_error * pair[..., None]
    ).sum() / (pair.sum() * C.N_JOINTS).clamp_min(1.0)
    return {"pose": pose, "distal": distal, "velocity": velocity}


def profile_input(output: dict) -> torch.Tensor:
    return torch.cat((
        output["conditioned_features"],
        output["motion_activity"][..., None],
        torch.softmax(output["phase_logits"], dim=-1),
    ), dim=-1)


def train_epoch(model, discriminator, loader, coarse, target_store,
                summary, site_mapping, frozen, optimizer, scaler,
                epoch: int, args, device: str) -> dict[str, float]:
    _, selector, classifier, scalar, part = frozen
    model.train()
    discriminator.train()
    totals: dict[str, list[float]] = {}
    domain_strength = min(1.0, epoch / max(args.domain_warmup_epochs, 1))
    for batch in loader:
        csi = batch["csi"].to(device, non_blocking=True)
        mask = batch["link_mask"].to(device, non_blocking=True)
        rows = batch["row"].long()
        site_id = site_ids_for_rows(rows, site_mapping, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            all_context = context_from_summary(
                model.canonicalizer, summary, device
            )
            context = select_context(all_context, site_id)
            output = model(
                csi, mask, coarse.lookup(rows, device), context, strength=1.0
            )
            features = output["conditioned_features"]
            frame_mask = mask.any(-1)
            selected = selector(features, frame_mask)
            classified = classifier(features, frame_mask)
            profiles = profile_input(output)
            scalar_speed = scalar(profiles, frame_mask)["speed"]
            part_speed = part(profiles, frame_mask)["part_speed"]
            action = batch["class_id"].to(device).long()
            risk = batch["risk_id"].to(device).long()
            target_pose_cpu = batch["pose_rel"].float()
            valid_cpu = batch["valid"].bool()
            speed_target = speed_targets(target_pose_cpu, valid_cpu).to(device)
            part_target = part_speed_targets(target_pose_cpu, valid_cpu).to(device)
            target_pose = target_pose_cpu.to(device)
            valid = valid_cpu.to(device)
            motion_target = target_store.lookup(rows, device)
            losses = pose_losses(output["pose_rel"], target_pose, valid, risk)
            action_loss = F.cross_entropy(
                1.50 * output["action_logits"]
                + 0.75 * classified["action_logits"],
                action, label_smoothing=0.03,
            )
            risk_each = F.cross_entropy(
                output["risk_logits"] + selected["risk_logits"],
                risk, reduction="none", label_smoothing=0.02,
            )
            risk_loss = (
                risk_each * torch.where(risk == 2, 2.5, 1.0)
            ).mean()
            motion_loss = F.smooth_l1_loss(
                selected["motion_embedding"], motion_target, beta=0.5
            ) + 0.20 * (
                1.0 - F.cosine_similarity(
                    selected["motion_embedding"], motion_target
                )
            ).mean()
            speed_loss = F.smooth_l1_loss(
                scalar_speed[valid], speed_target[valid], beta=0.15
            )
            part_loss = F.smooth_l1_loss(
                part_speed[valid], part_target[valid], beta=0.15
            )
            pooled = masked_pool(features, frame_mask)
            domain_loss = F.cross_entropy(
                discriminator(pooled, domain_strength), site_id
            )
            invariant = cross_site_motion_loss(pooled, action, site_id)
            input_scale = csi.detach().square().mean().sqrt().clamp_min(1e-3)
            calibration = (
                output["calibrated_csi"] - csi
            ).square().mean() / input_scale.square()
            loss = (
                2.0 * losses["pose"] + 0.75 * losses["distal"]
                + 0.75 * losses["velocity"] + motion_loss
                + 0.45 * action_loss + 0.40 * risk_loss
                + 0.20 * speed_loss + 0.20 * part_loss
                + args.domain_weight * domain_loss
                + args.invariant_weight * invariant
                + args.calibration_weight * calibration
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.canonicalizer.parameters())
            + list(discriminator.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        values = {
            "loss": loss, **losses, "action": action_loss,
            "risk": risk_loss, "motion": motion_loss,
            "speed": speed_loss, "part": part_loss,
            "domain": domain_loss, "invariant": invariant,
            "calibration": calibration,
        }
        for key, value in values.items():
            totals.setdefault(key, []).append(float(value.detach()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def validation_objective(model, dataset, coarse, summary, sites,
                         site_mapping, frozen, target_store,
                         args, device: str) -> dict[str, float]:
    _, selector, classifier, scalar, part = frozen
    model.eval()
    context_all = context_from_summary(model.canonicalizer, summary, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    pose_values, motion_values, action_values, risk_values = [], [], [], []
    for batch in loader:
        csi = batch["csi"].to(device)
        mask = batch["link_mask"].to(device)
        rows = batch["row"].long()
        site_id = site_ids_for_rows(rows, site_mapping, device)
        output = model(
            csi, mask, coarse.lookup(rows, device),
            select_context(context_all, site_id), strength=1.0,
        )
        frame_mask = mask.any(-1)
        selected = selector(output["conditioned_features"], frame_mask)
        classified = classifier(output["conditioned_features"], frame_mask)
        action = batch["class_id"].to(device).long()
        risk = batch["risk_id"].to(device).long()
        target_pose = batch["pose_rel"].to(device)
        valid = batch["valid"].to(device).bool()
        current = pose_losses(output["pose_rel"], target_pose, valid, risk)
        pose_values.append(float(
            2.0 * current["pose"] + 0.75 * current["distal"]
            + 0.75 * current["velocity"]
        ))
        target_embedding = target_store.lookup(rows, device)
        motion_values.append(float(F.smooth_l1_loss(
            selected["motion_embedding"], target_embedding, beta=0.5
        )))
        action_values.append(float(F.cross_entropy(
            1.50 * output["action_logits"]
            + 0.75 * classified["action_logits"], action
        )))
        risk_each = F.cross_entropy(
            output["risk_logits"] + selected["risk_logits"],
            risk, reduction="none"
        )
        risk_values.append(float((
            risk_each * torch.where(risk == 2, 2.5, 1.0)
        ).mean()))
    result = {
        "pose": float(np.mean(pose_values)),
        "motion": float(np.mean(motion_values)),
        "action": float(np.mean(action_values)),
        "risk": float(np.mean(risk_values)),
    }
    result["objective"] = (
        result["pose"] + result["motion"]
        + 0.45 * result["action"] + 0.40 * result["risk"]
    )
    return result


def train_calibrator(args, model, discriminator, train_dataset,
                     validation_dataset, train_summary, validation_summary,
                     train_sites, validation_sites, coarse, frozen,
                     device: str) -> tuple[dict, list[dict]]:
    train_mapping = row_site_ids(train_dataset, train_sites)
    validation_mapping = row_site_ids(validation_dataset, validation_sites)
    selector_checkpoint = frozen[0]
    train_targets = MotionTargetStore(
        build_motion_targets(train_dataset, selector_checkpoint)
    )
    validation_targets = MotionTargetStore(
        build_motion_targets(validation_dataset, selector_checkpoint)
    )
    labels = torch.tensor(
        train_dataset.index.class_id.to_numpy(dtype=np.int64)
    )
    count = torch.bincount(labels, minlength=C.N_CLASSES).float()
    weights = 1.0 / torch.sqrt(count[labels].clamp_min(1.0))
    sampler = WeightedRandomSampler(weights.double(), len(weights), replacement=True)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    parameters = list(model.canonicalizer.parameters()) + list(discriminator.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    best_state = copy.deepcopy(model.canonicalizer.state_dict())
    best_discriminator = copy.deepcopy(discriminator.state_dict())
    best = math.inf
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_result = train_epoch(
            model, discriminator, loader, coarse, train_targets,
            train_summary, train_mapping, frozen, optimizer, scaler,
            epoch, args, device,
        )
        validation = validation_objective(
            model, validation_dataset, coarse, validation_summary,
            validation_sites, validation_mapping, frozen,
            validation_targets, args, device,
        )
        row = {"epoch": epoch, "train": train_result, "validation": validation}
        history.append(row)
        if validation["objective"] < best - 1e-4:
            best = validation["objective"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.canonicalizer.state_dict())
            best_discriminator = copy.deepcopy(discriminator.state_dict())
            stale = 0
        else:
            stale += 1
        print(json.dumps({
            "epoch": epoch, "train": train_result["loss"],
            "validation": validation["objective"],
            "best_epoch": best_epoch,
        }), flush=True)
        if stale >= args.patience:
            break
    model.canonicalizer.load_state_dict(best_state)
    discriminator.load_state_dict(best_discriminator)
    return {"epoch": best_epoch, "objective": best}, history


@torch.no_grad()
def extract_features(model, dataset, coarse, summary, sites,
                     strength: float, device: str,
                     protocol: str) -> dict[str, torch.Tensor]:
    model.eval()
    context_all = context_from_summary(model.canonicalizer, summary, device)
    mapping = row_site_ids(dataset, sites)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    values = {
        "features": [], "frame_mask": [], "baseline_pose": [],
        "base_action_logits": [], "base_risk_logits": [],
        "contact_logits": [], "phase_logits": [],
        "motion_activity": [], "rows": [],
    }
    for batch in loader:
        csi = batch["csi"].to(device)
        mask = batch["link_mask"].to(device)
        rows = batch["row"].long()
        site_id = site_ids_for_rows(rows, mapping, device)
        output = model(
            csi, mask, coarse.lookup(rows, device),
            select_context(context_all, site_id), strength=strength,
        )
        values["features"].append(output["conditioned_features"].cpu().half())
        values["frame_mask"].append(mask.any(-1).cpu())
        values["baseline_pose"].append(output["pose_rel"].cpu().half())
        values["base_action_logits"].append(output["action_logits"].cpu().half())
        values["base_risk_logits"].append(output["risk_logits"].cpu().half())
        values["contact_logits"].append(output["contact_logits"].cpu().half())
        values["phase_logits"].append(output["phase_logits"].cpu().half())
        values["motion_activity"].append(output["motion_activity"].cpu().half())
        values["rows"].append(rows)
    result = {key: torch.cat(items) for key, items in values.items()}
    result.update({
        "protocol": protocol,
        "source": "CAL2 raw-link canonicalizer + frozen KP4-DCC",
        "cal2_strength": float(strength),
    })
    return result


def evaluate_source_strengths(args, model, validation_dataset,
                              validation_summary, coarse,
                              run_dir: Path, device: str) -> tuple[float, dict]:
    metrics = {}
    for strength in args.strengths:
        cache = extract_features(
            model, validation_dataset, coarse, validation_summary,
            META_VALIDATION_SITES, strength, device,
            f"CAL2 source meta-validation strength {strength}",
        )
        data = prepare_custom(
            args, validation_dataset, cache,
            run_dir / "source_validation_exact_distance.pt", device,
        )
        prediction = kp10_prediction(data, args, device).cpu()
        pose = _metric_batch(
            prediction, data["target_pose"], data["target_valid"],
            data["target_risk"],
        )
        classification = classifier_evaluation(
            args, cache, data["target_class"], data["target_risk"], device
        )
        metrics[f"strength_{strength:.2f}"] = {
            "selection_score": pose_selection_score(pose),
            "pose": pose, "classification": classification,
        }
    baseline = metrics["strength_0.00"]
    eligible = []
    for strength in args.strengths:
        current = metrics[f"strength_{strength:.2f}"]
        if (
            current["selection_score"] <= baseline["selection_score"] + 0.002
            and current["pose"]["mpjpe_m"]
            <= baseline["pose"]["mpjpe_m"] + 0.001
        ):
            eligible.append(strength)
    selected = min(
        eligible,
        key=lambda value: (
            metrics[f"strength_{value:.2f}"]["pose"]["danger_pose_mpjpe_m"]
            + 0.75 * metrics[f"strength_{value:.2f}"]["pose"][
                "danger_distal_mpjpe_m"
            ]
            + 0.25 * metrics[f"strength_{value:.2f}"]["pose"][
                "danger_endpoint_mpjpe_m"
            ]
        ),
    )
    return float(selected), metrics


def evaluate_yja(args, model, selected_strength: float,
                 coarse_model, run_dir: Path, device: str) -> dict:
    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline="sub", seed=args.seed
    )["test"]
    full = QualityWeightedDataset(sealed, None)
    support, query = split_support_query(
        sealed.index, ("yja_E02",), args.support_per_class, args.seed + 31
    )
    summary = support_summary(full, support, ("yja_E02",))
    coarse = load_or_create_coarse_store(
        coarse_model, (full,), run_dir / "yja_e02_v13s_coarse.pt",
        device, args.batch_size, "sealed_yja_E02_CAL2",
    )
    all_query = QualityWeightedDataset(
        subset_dataset(sealed, query), None
    )
    adapted_all = extract_features(
        model, all_query, coarse, summary, ("yja_E02",),
        selected_strength, device, "CAL2 yja/E02 target query",
    )
    base_full_cache = torch.load(
        args.yja_feature_cache, map_location="cpu", weights_only=False
    )
    base_all = slice_cache(base_full_cache, torch.from_numpy(query).long())
    action = torch.tensor(
        sealed.index.iloc[query].class_id.to_numpy(dtype=np.int64)
    )
    risk = torch.tensor(
        sealed.index.iloc[query].risk_id.to_numpy(dtype=np.int64)
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[query].task.to_numpy() == C.TASK_POSE
    )
    pose_positions = query[pose_local]
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, pose_positions), None
    )
    local = torch.from_numpy(pose_local).long()
    variants = {}
    for name, cache in (
        ("kp10_no_calibration", slice_cache(base_all, local)),
        ("cal2_kp10", slice_cache(adapted_all, local)),
    ):
        data = prepare_custom(
            args, pose_target, cache,
            run_dir / "yja_e02_query_exact_distance.pt", device,
        )
        prediction = kp10_prediction(data, args, device).cpu()
        variants[name] = {"pose": _metric_batch(
            prediction, data["target_pose"], data["target_valid"],
            data["target_risk"],
        )}
    variants["kp10_no_calibration"]["classification"] = classifier_evaluation(
        args, base_all, action, risk, device
    )
    variants["cal2_kp10"]["classification"] = classifier_evaluation(
        args, adapted_all, action, risk, device
    )
    return {
        "target": "yja/E02",
        "support_trials": 16,
        "query_trials": int(len(query)),
        "pose_query_trials": int(len(pose_positions)),
        "danger_query_trials": int((risk == 2).sum()),
        "target_support_pose_gt_used": False,
        "target_query_used_for_training_or_selection": False,
        "selected_strength": selected_strength,
        "variants": variants,
    }


def main() -> None:
    publish_root = Path(__file__).resolve().parents[2]
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--domain-weight", type=float, default=0.06)
    parser.add_argument("--invariant-weight", type=float, default=0.08)
    parser.add_argument("--calibration-weight", type=float, default=0.003)
    parser.add_argument("--domain-warmup-epochs", type=int, default=4)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=(0.0, 0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--yja-feature-cache", type=Path,
        default=Path(r"C:\Users\jjeong\Documents\Playground"
                     r"\kp10_yja_e02_zero_shot_local\yja_e02_features.pt"),
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal2_kp10_seed223"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train_all = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation_all = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_support, train_query = split_support_query(
        train_all.index, META_TRAIN_SITES,
        args.support_per_class, args.seed,
    )
    validation_support, validation_query = split_support_query(
        validation_all.index, META_VALIDATION_SITES,
        args.support_per_class, args.seed + 17,
    )
    train_summary = support_summary(
        train_all, train_support, META_TRAIN_SITES
    )
    validation_summary = support_summary(
        validation_all, validation_support, META_VALIDATION_SITES
    )
    train_dataset = QualityWeightedDataset(
        subset_dataset(train_all.target, train_query, train=True), audit
    )
    validation_dataset = QualityWeightedDataset(
        subset_dataset(validation_all.target, validation_query), audit
    )

    coarse = load_source_coarse(
        args.work_root, datasets, device, args.batch_size, args.exp
    )
    kp4, _ = _load_model(
        args.work_root / "runs/kp4_dcc_staged_seed17/deployment_model.pt",
        device,
    )
    canonicalizer = RawLinkCanonicalizer().to(device)
    reference = fit_reference(train_summary)
    canonicalizer.set_reference(**{
        key: value.to(device) for key, value in reference.items()
    })
    model = RawCalibratedKP4(kp4, canonicalizer).to(device)
    discriminator = SiteDiscriminator(
        input_dim=128, sites=len(META_TRAIN_SITES)
    ).to(device)
    frozen = load_frozen_heads(args, device)
    if args.evaluation_only:
        if args.resume_checkpoint is None:
            raise ValueError("--evaluation-only requires --resume-checkpoint")
        resumed = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        model.canonicalizer.load_state_dict(resumed["canonicalizer"])
        training = resumed.get("training", {"source": "resumed"})
        history = []
    else:
        training, history = train_calibrator(
            args, model, discriminator, train_dataset, validation_dataset,
            train_summary, validation_summary, META_TRAIN_SITES,
            META_VALIDATION_SITES, coarse, frozen, device,
        )
    selected_strength, source_metrics = evaluate_source_strengths(
        args, model, validation_dataset, validation_summary,
        coarse, args.run_dir, device,
    )

    coarse_model = build_coarse_model(
        args.work_root, publish_root, args.exp, device
    )
    yja = evaluate_yja(
        args, model, selected_strength, coarse_model, args.run_dir, device
    )
    checkpoint = {
        "run": "CAL2-KP10",
        "canonicalizer": model.canonicalizer.state_dict(),
        "canonicalizer_config": {
            "token_dim": 96, "basis_rank": 8, "lowpass_window": 31,
        },
        "reference": reference,
        "selected_strength": selected_strength,
        "training": training,
        "meta_train_sites": META_TRAIN_SITES,
        "meta_validation_sites": META_VALIDATION_SITES,
    }
    torch.save(checkpoint, args.run_dir / "best_model.pt")
    result = {
        "run": "CAL2-KP10",
        "status": "completed_target_support_query_evaluation",
        "protocol": args.exp,
        "device": device,
        "contract": {
            "raw_link_calibration": True,
            "support_per_class": args.support_per_class,
            "support_classes": [0, 1, 2, 3, 4, 5, 7, 8],
            "target_support_pose_gt_used": False,
            "target_query_used_for_training_or_selection": False,
            "strength_selected_on": list(META_VALIDATION_SITES),
            "yja_opened_after_source_selection": True,
        },
        "training": training,
        "history": history,
        "source_meta_validation": {
            "selected_strength": selected_strength,
            "metrics": source_metrics,
        },
        "yja_e02": yja,
        "artifacts": {
            "checkpoint": report_path(args.run_dir / "best_model.pt"),
        },
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "run": result["run"],
        "selected_strength": selected_strength,
        "source": source_metrics[f"strength_{selected_strength:.2f}"],
        "yja": yja["variants"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
