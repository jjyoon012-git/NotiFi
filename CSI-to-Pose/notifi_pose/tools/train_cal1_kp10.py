"""Train and audit CAL1-KP10 safe-support calibration."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..cal1_kp10 import Cal1KP10Adapter, SAFE_SUPPORT_CLASSES
from ..dataio.dataset import build_datasets
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_kp10_paired_bootstrap import kp10_prediction
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
)
from .calibrate_motion_profile_reranking import (
    candidate_speed_profiles,
    predict_profile,
    profile_distance,
    standardize,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .calibrate_part_motion_profile_reranking import (
    candidate_part_speed_profiles,
    load_models,
    part_profile_distance,
)
from .diagnose_observability import pose_only, report_path
from .train_calibration_aware_v14 import subset_dataset
from .train_csi_motion_profile import profile_features, speed_targets
from .train_csi_part_motion_profile import part_speed_targets, predict_part_profile
from .train_dynamic_motion import classification_metrics
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import (
    motion_descriptor,
    predict_selector,
    project_motion,
)


META_TRAIN_SITES = ("ajh_E01", "ajh_E02", "lmh_E01", "mhw_E01", "mhw_E02")
META_VALIDATION_SITES = ("ajh_E03", "mhw_E03")


def configure_work_root(root: Path) -> None:
    C.PROJECT_ROOT = root.parent
    C.WORK_ROOT = root
    C.INDEX_DIR = root / "index"
    C.CACHE_DIR = root / "cache"
    C.REPORT_DIR = root / "reports"
    C.SPLIT_DIR = root / "splits"


def site_names(index) -> np.ndarray:
    return (
        index.subject.astype(str) + "_" + index.environment.astype(str)
    ).to_numpy()


def split_support_query(index, sites: tuple[str, ...], per_class: int,
                        seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    names = site_names(index)
    labels = index.class_id.to_numpy(dtype=np.int64)
    support: dict[str, np.ndarray] = {}
    query: list[int] = []
    for site in sites:
        site_positions = np.flatnonzero(names == site)
        selected: list[int] = []
        for class_id in SAFE_SUPPORT_CLASSES:
            candidates = site_positions[labels[site_positions] == class_id].copy()
            rng.shuffle(candidates)
            if len(candidates) < per_class:
                raise RuntimeError(
                    f"{site} class {class_id} has {len(candidates)}, needs {per_class}"
                )
            selected.extend(int(value) for value in candidates[:per_class])
        chosen = np.sort(np.asarray(selected, dtype=np.int64))
        support[site] = chosen
        query.extend(int(value) for value in site_positions if value not in set(chosen))
    return support, np.sort(np.asarray(query, dtype=np.int64))


def slice_cache(cache: dict, positions: torch.Tensor) -> dict:
    count = len(cache["rows"])
    result = {}
    for key, value in cache.items():
        result[key] = (
            value.index_select(0, positions)
            if torch.is_tensor(value) and value.ndim and len(value) == count
            else value
        )
    return result


def audit_cache(cache: dict, dataset, name: str) -> None:
    expected = torch.from_numpy(dataset.rows).long()
    if not torch.equal(cache["rows"].long(), expected):
        raise RuntimeError(f"{name} feature cache row order does not match dataset")


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
    _, _, _, scalar, part, _ = load_models(args, device)
    for model in (selector, classifier, scalar, part):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return selector_checkpoint, selector, classifier, scalar, part


def build_targets(dataset, selector_checkpoint: dict) -> dict[str, torch.Tensor]:
    pose, valid, action, risk = _load_pose_arrays(dataset)
    bank = torch.stack([
        _canonicalize(value, mask, C.CACHE_FRAMES)
        for value, mask in zip(pose, valid)
    ])
    embedding = project_motion(
        motion_descriptor(bank), selector_checkpoint["pca"]
    )
    return {
        "pose": pose, "valid": valid, "action": action, "risk": risk,
        "embedding": embedding,
        "speed": speed_targets(pose, valid),
        "part_speed": part_speed_targets(pose, valid),
    }


def support_tensors(cache: dict, index, support: dict[str, np.ndarray],
                    sites: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    positions = torch.tensor(
        np.stack([support[site] for site in sites]), dtype=torch.long
    )
    labels = torch.tensor(
        np.stack([
            index.iloc[support[site]].class_id.to_numpy(dtype=np.int64)
            for site in sites
        ]),
        dtype=torch.long,
    )
    flat = positions.flatten()
    shape = tuple(positions.shape)
    return (
        cache["features"].index_select(0, flat).float().reshape(
            *shape, cache["features"].shape[1], cache["features"].shape[2]
        ),
        cache["frame_mask"].index_select(0, flat).bool().reshape(
            *shape, cache["frame_mask"].shape[1]
        ),
        labels,
    )


def class_weights(labels: torch.Tensor, classes: int,
                  device: str) -> torch.Tensor:
    count = torch.bincount(labels, minlength=classes).float()
    weight = count.sum() / count.clamp_min(1.0)
    return (weight / weight.mean()).clamp(0.35, 3.0).to(device)


def profile_input(cache: dict, indices: torch.Tensor,
                  adapted: torch.Tensor) -> torch.Tensor:
    return torch.cat((
        adapted,
        cache["motion_activity"].index_select(0, indices).float()[..., None],
        torch.softmax(
            cache["phase_logits"].index_select(0, indices).float(), dim=-1
        ),
    ), dim=-1)


@torch.no_grad()
def validation_head_objective(adapter, cache, dataset, support, query,
                              sites, frozen, device: str) -> dict[str, float]:
    selector_checkpoint, selector, classifier, scalar, part = frozen
    targets = build_targets(dataset, selector_checkpoint)
    support_features, support_mask, support_class = support_tensors(
        cache, dataset.index, support, sites
    )
    token = adapter.encode_support(
        support_features.to(device), support_mask.to(device),
        support_class.to(device),
    )
    names = site_names(dataset.index)
    lookup = {site: number for number, site in enumerate(sites)}
    query_tensor = torch.from_numpy(query).long()
    query_sites = torch.tensor(
        [lookup[names[int(position)]] for position in query], dtype=torch.long
    )
    totals = {
        "motion": 0.0, "action": 0.0, "risk": 0.0,
        "speed": 0.0, "part": 0.0,
    }
    count = 0
    adapter.eval()
    for start in range(0, len(query_tensor), 64):
        positions = query_tensor[start:start + 64]
        site_id = query_sites[start:start + 64].to(device)
        features = cache["features"].index_select(0, positions).to(device).float()
        mask = cache["frame_mask"].index_select(0, positions).to(device).bool()
        adapted = adapter.adapt(features, mask, token[site_id])
        selector_output = selector(adapted, mask)
        classifier_output = classifier(adapted, mask)
        extras = torch.cat((
            cache["motion_activity"].index_select(
                0, positions
            ).to(device).float()[..., None],
            torch.softmax(cache["phase_logits"].index_select(
                0, positions
            ).to(device).float(), dim=-1),
        ), dim=-1)
        profile = torch.cat((adapted, extras), dim=-1)
        scalar_output = scalar(profile, mask)["speed"]
        part_output = part(profile, mask)["part_speed"]
        action = targets["action"].index_select(0, positions).to(device)
        risk = targets["risk"].index_select(0, positions).to(device)
        embedding = targets["embedding"].index_select(0, positions).to(device)
        base_action = cache["base_action_logits"].index_select(
            0, positions
        ).to(device).float()
        base_risk = cache["base_risk_logits"].index_select(
            0, positions
        ).to(device).float()
        valid = targets["valid"].index_select(0, positions).to(device)
        speed = targets["speed"].index_select(0, positions).to(device)
        part_speed = targets["part_speed"].index_select(0, positions).to(device)
        batch = len(positions)
        totals["motion"] += batch * float(F.smooth_l1_loss(
            selector_output["motion_embedding"], embedding, beta=0.5
        ))
        totals["action"] += batch * float(F.cross_entropy(
            1.50 * base_action + 0.75 * classifier_output["action_logits"], action
        ))
        risk_loss = F.cross_entropy(
            base_risk + selector_output["risk_logits"], risk, reduction="none"
        )
        risk_scale = torch.where(risk == 2, 2.25, 1.0)
        totals["risk"] += batch * float((risk_loss * risk_scale).mean())
        totals["speed"] += batch * float(F.smooth_l1_loss(
            scalar_output[valid], speed[valid], beta=0.15
        ))
        totals["part"] += batch * float(F.smooth_l1_loss(
            part_output[valid], part_speed[valid], beta=0.15
        ))
        count += batch
    values = {key: value / max(count, 1) for key, value in totals.items()}
    values["objective"] = (
        values["motion"] + 0.45 * values["action"]
        + 0.35 * values["risk"] + 0.20 * values["speed"]
        + 0.20 * values["part"]
    )
    return values


def train_adapter(args, adapter, train_cache, train_dataset,
                  support, query, frozen, validation,
                  device: str) -> tuple[dict, dict]:
    selector_checkpoint, selector, classifier, scalar, part = frozen
    targets = build_targets(train_dataset, selector_checkpoint)
    site_lookup = {site: number for number, site in enumerate(META_TRAIN_SITES)}
    names = site_names(train_dataset.index)
    query_tensor = torch.from_numpy(query).long()
    query_site = torch.tensor(
        [site_lookup[names[int(position)]] for position in query], dtype=torch.long
    )
    sampler_count = torch.bincount(
        targets["action"].index_select(0, query_tensor), minlength=C.N_CLASSES
    ).float()
    sampler_weight = 1.0 / torch.sqrt(
        sampler_count[targets["action"].index_select(0, query_tensor)].clamp_min(1)
    )
    sampler = WeightedRandomSampler(
        sampler_weight.double(), len(query_tensor), replacement=True
    )
    loader = DataLoader(
        TensorDataset(query_tensor, query_site), batch_size=args.batch_size,
        sampler=sampler, num_workers=0,
    )
    support_features, support_mask, support_class = support_tensors(
        train_cache, train_dataset.index, support, META_TRAIN_SITES
    )
    support_features = support_features.to(device)
    support_mask = support_mask.to(device)
    support_class = support_class.to(device)
    action_weight = class_weights(
        targets["action"].index_select(0, query_tensor), C.N_CLASSES, device
    )
    risk_weight = class_weights(
        targets["risk"].index_select(0, query_tensor), C.N_RISK, device
    )
    risk_weight[-1] *= 2.25

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1), eta_min=args.learning_rate * 0.08
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    best_state = copy.deepcopy(adapter.state_dict())
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        totals: dict[str, list[float]] = {}
        for positions, site_id in loader:
            positions = positions.long()
            site_id = site_id.to(device)
            features = train_cache["features"].index_select(
                0, positions
            ).to(device).float()
            mask = train_cache["frame_mask"].index_select(
                0, positions
            ).to(device).bool()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                tokens = adapter.encode_support(
                    support_features, support_mask, support_class
                )
                adapted = adapter.adapt(features, mask, tokens[site_id])
                selector_output = selector(adapted, mask)
                classifier_output = classifier(adapted, mask)
                if device == "cpu":
                    profile = profile_input(train_cache, positions, adapted)
                else:
                    extras = torch.cat((
                        train_cache["motion_activity"].index_select(
                            0, positions
                        ).to(device).float()[..., None],
                        torch.softmax(train_cache["phase_logits"].index_select(
                            0, positions
                        ).to(device).float(), dim=-1),
                    ), dim=-1)
                    profile = torch.cat((adapted, extras), dim=-1)
                scalar_output = scalar(profile, mask)["speed"]
                part_output = part(profile, mask)["part_speed"]
                action = targets["action"].index_select(0, positions).to(device)
                risk = targets["risk"].index_select(0, positions).to(device)
                embedding = targets["embedding"].index_select(
                    0, positions
                ).to(device)
                base_action = train_cache["base_action_logits"].index_select(
                    0, positions
                ).to(device).float()
                base_risk = train_cache["base_risk_logits"].index_select(
                    0, positions
                ).to(device).float()
                action_loss = F.cross_entropy(
                    1.50 * base_action + 0.75 * classifier_output["action_logits"],
                    action, weight=action_weight, label_smoothing=0.03,
                )
                selector_action = F.cross_entropy(
                    base_action + selector_output["action_logits"], action,
                    weight=action_weight, label_smoothing=0.03,
                )
                risk_loss = F.cross_entropy(
                    base_risk + selector_output["risk_logits"], risk,
                    weight=risk_weight, label_smoothing=0.02,
                )
                motion_loss = F.smooth_l1_loss(
                    selector_output["motion_embedding"], embedding, beta=0.5
                )
                motion_cosine = (
                    1.0 - F.cosine_similarity(
                        selector_output["motion_embedding"], embedding
                    )
                ).mean()
                valid = targets["valid"].index_select(0, positions).to(device)
                speed = targets["speed"].index_select(0, positions).to(device)
                part_speed = targets["part_speed"].index_select(0, positions).to(device)
                speed_loss = F.smooth_l1_loss(
                    scalar_output[valid], speed[valid], beta=0.15
                )
                part_loss = F.smooth_l1_loss(
                    part_output[valid], part_speed[valid], beta=0.15
                )
                identity = ((adapted - features).square() * mask[..., None]).mean()
                loss = (
                    motion_loss + 0.20 * motion_cosine
                    + 0.45 * action_loss + 0.20 * selector_action
                    + 0.35 * risk_loss + 0.20 * speed_loss
                    + 0.20 * part_loss + 0.002 * identity
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            values = {
                "loss": loss, "motion": motion_loss, "action": action_loss,
                "risk": risk_loss, "speed": speed_loss,
                "part": part_loss, "identity": identity,
            }
            for key, value in values.items():
                totals.setdefault(key, []).append(float(value.detach()))
        current = {key: float(np.mean(value)) for key, value in totals.items()}
        validation_cache, validation_dataset, validation_support, validation_query = (
            validation
        )
        validation_metrics = validation_head_objective(
            adapter, validation_cache, validation_dataset,
            validation_support, validation_query,
            META_VALIDATION_SITES, frozen, device,
        )
        current["validation_objective"] = validation_metrics["objective"]
        current["epoch"] = epoch
        history.append(current)
        if current["validation_objective"] < best_loss - 1e-4:
            best_loss = current["validation_objective"]
            best_state = copy.deepcopy(adapter.state_dict())
            stale = 0
        else:
            stale += 1
        print(json.dumps(current), flush=True)
        if stale >= args.patience:
            break
    adapter.load_state_dict(best_state)
    return {"best_validation_objective": best_loss, "epochs": len(history)}, {
        "history": history,
        "support_trials": sum(len(value) for value in support.values()),
        "query_trials": len(query),
    }


@torch.no_grad()
def adapted_cache(adapter, cache: dict, support_features: torch.Tensor,
                  support_mask: torch.Tensor, support_class: torch.Tensor,
                  positions: torch.Tensor, site_ids: torch.Tensor,
                  strength: float, device: str) -> dict:
    adapter.eval()
    token = adapter.encode_support(
        support_features.to(device), support_mask.to(device),
        support_class.to(device),
    )
    selected = slice_cache(cache, positions)
    values = []
    for start in range(0, len(positions), 64):
        stop = min(start + 64, len(positions))
        original = selected["features"][start:stop].to(device).float()
        mask = selected["frame_mask"][start:stop].to(device).bool()
        values.append(adapter.adapt(
            original, mask, token[site_ids[start:stop].to(device)], strength
        ).float().cpu().half())
    selected["features"] = torch.cat(values)
    selected["cal1_strength"] = float(strength)
    return selected


@torch.no_grad()
def prepare_custom(args, target_set, cache: dict,
                   distance_path: Path, device: str) -> dict:
    checkpoint, selector, reranker, scalar, part, part_checkpoint = load_models(
        args, device
    )
    selector_output = predict_selector(selector, cache, 64, device)
    target_pose, target_valid, target_class, target_risk = _load_pose_arrays(target_set)
    inference_valid = cache["frame_mask"].bool()
    baseline = cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, inference_valid)
    ])
    train_bank = checkpoint["train_bank"].float()
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    distance = exact_pose_distance(baseline_bank, train_bank, distance_path)
    pool = make_candidate_pool(
        baseline_bank, torch.zeros_like(baseline_bank),
        torch.zeros_like(target_risk), train_bank,
        checkpoint["train_class"].long(), fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
        action_penalty=args.candidate_action_penalty,
    )
    logits = []
    for start in range(0, len(target_set), 64):
        indices = torch.arange(start, min(start + 64, len(target_set)))
        inputs = tuple(value.to(device) for value in model_inputs(
            pool, selector_output, checkpoint, risk_probability, indices
        ))
        logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    predicted_scalar = predict_profile(scalar, cache, inference_valid, device)
    candidate_scalar = candidate_speed_profiles(train_bank, pool, inference_valid)
    scalar_distance = standardize(profile_distance(
        predicted_scalar, candidate_scalar, inference_valid
    ))
    predicted_part = predict_part_profile(part, cache, inference_valid, device)
    candidate_part = candidate_part_speed_profiles(train_bank, pool, inference_valid)
    part_distance = part_profile_distance(
        predicted_part, candidate_part, inference_valid
    )
    return {
        "checkpoint": checkpoint, "part_checkpoint": part_checkpoint,
        "cache": cache, "baseline": baseline, "baseline_bank": baseline_bank,
        "train_bank": train_bank, "fused_action": fused_action,
        "risk_probability": risk_probability,
        "target_pose": target_pose, "target_valid": target_valid,
        "inference_valid": inference_valid,
        "target_class": target_class, "target_risk": target_risk,
        "base_action_logits": cache["base_action_logits"].float(),
        "selector_embedding": selector_output["embedding"].float(),
        "selector_action_logits": selector_output["action_logits"].float(),
        "base_risk_logits": cache["base_risk_logits"].float(),
        "selector_risk_logits": selector_output["risk_logits"].float(),
        "pool": pool, "logits": logits,
        "scalar_distance": scalar_distance, "part_distance": part_distance,
        "predicted_scalar_profile": predicted_scalar,
        "predicted_part_profile": predicted_part,
        "candidate_scalar_profiles": candidate_scalar,
        "candidate_part_profiles": candidate_part,
    }


@torch.no_grad()
def classifier_evaluation(args, cache, action, risk, device: str) -> dict:
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    classifier = TemporalMotionSelector(
        **classifier_checkpoint["model_config"]
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    selector_checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(
        **selector_checkpoint["model_config"]
    ).to(device)
    selector.load_state_dict(selector_checkpoint["model"])
    extra = predict_selector(classifier, cache, 64, device)
    selected = predict_selector(selector, cache, 64, device)
    action_logits = 1.50 * cache["base_action_logits"].float() + 0.75 * extra["action_logits"]
    risk_logits = cache["base_risk_logits"].float() + selected["risk_logits"]
    return classification_metrics(action_logits, risk_logits, action, risk)


def evaluate_strengths(args, adapter, cache, dataset, support, query,
                       sites, run_dir: Path, device: str) -> tuple[float, dict]:
    support_features, support_mask, support_class = support_tensors(
        cache, dataset.index, support, sites
    )
    names = site_names(dataset.index)
    lookup = {site: value for value, site in enumerate(sites)}
    query_tensor = torch.from_numpy(query).long()
    query_sites = torch.tensor(
        [lookup[names[int(position)]] for position in query], dtype=torch.long
    )
    query_target = QualityWeightedDataset(
        subset_dataset(dataset.target, query), protocol_audit_path(args.exp)
    )
    all_metrics = {}
    for strength in args.strengths:
        current_cache = adapted_cache(
            adapter, cache, support_features, support_mask, support_class,
            query_tensor, query_sites, strength, device,
        )
        data = prepare_custom(
            args, query_target, current_cache,
            run_dir / "source_meta_validation_exact_distance.pt", device,
        )
        prediction = kp10_prediction(data, args, device).cpu()
        pose = _metric_batch(
            prediction, data["target_pose"], data["target_valid"],
            data["target_risk"],
        )
        classification = classifier_evaluation(
            args, current_cache, data["target_class"], data["target_risk"], device
        )
        all_metrics[f"strength_{strength:.2f}"] = {
            "selection_score": pose_selection_score(pose),
            "pose": pose, "classification": classification,
        }
    best = min(
        args.strengths,
        key=lambda value: all_metrics[f"strength_{value:.2f}"]["selection_score"],
    )
    return float(best), all_metrics


def evaluate_yja(args, adapter, selected_strength: float,
                 run_dir: Path, device: str) -> dict:
    full_cache = torch.load(
        args.yja_feature_cache, map_location="cpu", weights_only=False
    )
    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline="sub", seed=args.seed
    )["test"]
    audit_cache(full_cache, sealed, "yja/E02")
    support, query = split_support_query(
        sealed.index, ("yja_E02",), args.support_per_class, args.seed + 31
    )
    support_features, support_mask, support_class = support_tensors(
        full_cache, sealed.index, support, ("yja_E02",)
    )
    all_query = torch.from_numpy(query).long()
    all_site = torch.zeros(len(all_query), dtype=torch.long)
    adapted_all = adapted_cache(
        adapter, full_cache, support_features, support_mask, support_class,
        all_query, all_site, selected_strength, device,
    )
    base_all = slice_cache(full_cache, all_query)
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
    pose_local_tensor = torch.from_numpy(pose_local).long()
    base_pose_cache = slice_cache(base_all, pose_local_tensor)
    adapted_pose_cache = slice_cache(adapted_all, pose_local_tensor)

    variants = {}
    for name, current_cache in (
        ("kp10_no_calibration", base_pose_cache),
        ("cal1_kp10", adapted_pose_cache),
    ):
        data = prepare_custom(
            args, pose_target, current_cache,
            run_dir / "yja_e02_query_exact_distance.pt", device,
        )
        predicted = kp10_prediction(data, args, device).cpu()
        variants[name] = {
            "pose": _metric_batch(
                predicted, data["target_pose"], data["target_valid"],
                data["target_risk"],
            )
        }
    variants["kp10_no_calibration"]["classification"] = classifier_evaluation(
        args, base_all, action, risk, device
    )
    variants["cal1_kp10"]["classification"] = classifier_evaluation(
        args, adapted_all, action, risk, device
    )
    return {
        "target": "yja/E02",
        "support": {
            "trials": int(sum(len(value) for value in support.values())),
            "per_class": args.support_per_class,
            "class_ids": list(SAFE_SUPPORT_CLASSES),
            "uses_csi": True, "uses_known_prompt_labels": True,
            "uses_pose_gt": False,
        },
        "query": {
            "all_trials": int(len(query)),
            "pose_trials": int(len(pose_positions)),
            "absence_trials": int(len(query) - len(pose_positions)),
            "danger_trials": int((risk == 2).sum()),
        },
        "selected_strength": selected_strength,
        "target_query_used_for_training_or_selection": False,
        "variants": variants,
    }


def add_paths(parser: argparse.ArgumentParser, work_root: Path) -> None:
    runs = work_root / "runs"
    parser.add_argument("--selector-checkpoint", type=Path,
        default=runs / "kp5_mpr_selector_seed17/best_model.pt")
    parser.add_argument("--reranker-checkpoint", type=Path,
        default=runs / "kp5_mpr_reranker_seed17/best_model.pt")
    parser.add_argument("--scalar-profile-checkpoint", type=Path,
        default=runs / "kp5_motion_profile_seed79/best_model.pt")
    parser.add_argument("--part-profile-checkpoint", type=Path,
        default=runs / "kp5_part_motion_profile_seed101/best_model.pt")
    parser.add_argument("--adaptive-calibration", type=Path,
        default=runs / "kp6_risk_adaptive_blend/calibration.json")
    parser.add_argument("--classifier-checkpoint", type=Path,
        default=runs / "kp10_action_classifier_seed181/best_model.pt")
    parser.add_argument("--profile-ranker-checkpoint", type=Path,
        default=runs / "kp8_profile_candidate_ranker_seed127/best_model.pt")
    parser.add_argument("--strength-calibration", type=Path,
        default=runs / "kp10_action_strength/calibration.json")


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=(0.0, 0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--yja-feature-cache", type=Path,
        default=Path(r"C:\Users\jjeong\Documents\Playground"
                     r"\kp10_yja_e02_zero_shot_local\yja_e02_features.pt"),
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    args.run_dir = args.run_dir or args.work_root / "runs/cal1_kp10_seed211"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train_dataset = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    val_dataset = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    feature_root = args.selector_checkpoint.parent
    train_cache = torch.load(
        feature_root / "train_features.pt", map_location="cpu", weights_only=False
    )
    val_cache = torch.load(
        feature_root / "val_features.pt", map_location="cpu", weights_only=False
    )
    audit_cache(train_cache, train_dataset, "train")
    audit_cache(val_cache, val_dataset, "validation")
    train_support, train_query = split_support_query(
        train_dataset.index, META_TRAIN_SITES,
        args.support_per_class, args.seed,
    )
    val_support, val_query = split_support_query(
        val_dataset.index, META_VALIDATION_SITES,
        args.support_per_class, args.seed + 17,
    )
    adapter = Cal1KP10Adapter(
        feature_dim=train_cache["features"].shape[-1]
    ).to(device)
    frozen = load_frozen_heads(args, device)
    training, audit_result = train_adapter(
        args, adapter, train_cache, train_dataset,
        train_support, train_query, frozen,
        (val_cache, val_dataset, val_support, val_query), device,
    )
    selected_strength, source_validation = evaluate_strengths(
        args, adapter, val_cache, val_dataset,
        val_support, val_query, META_VALIDATION_SITES,
        args.run_dir, device,
    )
    yja = evaluate_yja(args, adapter, selected_strength, args.run_dir, device)
    checkpoint = {
        "run": "CAL1-KP10",
        "model": adapter.state_dict(),
        "model_config": {
            "feature_dim": train_cache["features"].shape[-1],
            "token_dim": 96, "rank": 64,
        },
        "selected_strength": selected_strength,
        "support_classes": SAFE_SUPPORT_CLASSES,
        "meta_train_sites": META_TRAIN_SITES,
        "meta_validation_sites": META_VALIDATION_SITES,
        "training": training,
    }
    torch.save(checkpoint, args.run_dir / "best_model.pt")
    result = {
        "run": "CAL1-KP10",
        "status": "completed_target_support_query_evaluation",
        "protocol": args.exp,
        "device": device,
        "calibration_contract": {
            "support_classes": list(SAFE_SUPPORT_CLASSES),
            "support_per_class": args.support_per_class,
            "target_support_pose_gt_used": False,
            "target_query_labels_or_gt_used_for_adaptation": False,
            "strength_selected_on": list(META_VALIDATION_SITES),
            "yja_opened_after_source_selection": True,
        },
        "training": training,
        "training_audit": audit_result,
        "source_meta_validation": {
            "selected_strength": selected_strength,
            "support_trials": sum(len(value) for value in val_support.values()),
            "query_trials": len(val_query),
            "metrics": source_validation,
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
        "yja": yja["variants"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
