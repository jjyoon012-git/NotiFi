"""Fit CAL3-KP10 site adapters from prompted safe support actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import contract as C
from ..cal3_kp10 import SafeSupportFeatureAdapter
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_kp10_paired_bootstrap import kp10_prediction
from .audit_motion_retrieval_oracle import _load_pose_arrays, _metric_batch
from .diagnose_observability import pose_only, report_path
from .train_cal1_kp10 import (
    META_VALIDATION_SITES,
    add_paths,
    configure_work_root,
    prepare_custom,
    site_names,
    slice_cache,
    split_support_query,
)
from .train_calibration_aware_v14 import subset_dataset
from .train_dynamic_motion import classification_metrics
from .train_kinetic_pose import pose_selection_score
from .train_motion_retrieval_selector import predict_selector


def load_heads(args, device: str):
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
    for model in (classifier, selector):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return classifier, selector


@torch.no_grad()
def class_prototypes(classifier, cache: dict, labels: torch.Tensor,
                     device: str) -> torch.Tensor:
    pooled = predict_selector(classifier, cache, 64, device)["pooled_features"]
    prototypes = []
    for class_id in range(C.N_CLASSES):
        selected = pooled[labels == class_id]
        if not len(selected):
            prototypes.append(torch.zeros(pooled.shape[-1]))
        else:
            prototypes.append(F.normalize(selected.mean(0), dim=-1))
    return torch.stack(prototypes).to(device)


def relation_loss(pooled: torch.Tensor, labels: torch.Tensor,
                  prototypes: torch.Tensor) -> torch.Tensor:
    classes = labels.unique(sorted=True)
    centroids = torch.stack([
        F.normalize(pooled[labels == class_id].mean(0), dim=-1)
        for class_id in classes
    ])
    source = prototypes.index_select(0, classes)
    return F.mse_loss(centroids @ centroids.T, source @ source.T)


@torch.no_grad()
def support_accuracy(classifier, selector, adapter, cache,
                     labels, device: str) -> dict:
    features = cache["features"].to(device).float()
    mask = cache["frame_mask"].to(device).bool()
    adapted = adapter(features, mask)
    action = classifier(adapted, mask)["action_logits"].argmax(-1).cpu()
    selector_action = selector(adapted, mask)["action_logits"].argmax(-1).cpu()
    return {
        "classifier": float((action == labels).float().mean()),
        "selector": float((selector_action == labels).float().mean()),
    }


def fit_site_adapter(args, support_cache: dict, support_labels: torch.Tensor,
                     prototypes: torch.Tensor, classifier, selector,
                     device: str) -> tuple[SafeSupportFeatureAdapter, dict]:
    adapter = SafeSupportFeatureAdapter(
        feature_dim=support_cache["features"].shape[-1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    features = support_cache["features"].to(device).float()
    mask = support_cache["frame_mask"].to(device).bool()
    labels = support_labels.to(device)
    with torch.no_grad():
        base_classifier = classifier(features, mask)
        base_selector = selector(features, mask)
    initial = support_accuracy(
        classifier, selector, adapter, support_cache, support_labels, device
    )
    history = []
    for step in range(1, args.steps + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        augmented = features + args.feature_noise * torch.randn_like(features)
        if args.temporal_drop > 0:
            drop = (
                torch.rand(mask.shape, device=device) < args.temporal_drop
            ) & mask
            augmented = augmented.masked_fill(drop[..., None], 0.0)
        adapted = adapter(augmented, mask)
        classified = classifier(adapted, mask)
        selected = selector(adapted, mask)
        classifier_ce = F.cross_entropy(
            classified["action_logits"], labels, label_smoothing=0.02
        )
        selector_ce = F.cross_entropy(
            selected["action_logits"], labels, label_smoothing=0.02
        )
        target = prototypes.index_select(0, labels)
        prototype = (
            1.0 - F.cosine_similarity(
                F.normalize(classified["pooled_features"], dim=-1), target
            )
        ).mean()
        relation = relation_loss(
            classified["pooled_features"], labels, prototypes
        )
        risk_anchor = F.mse_loss(
            selected["risk_logits"], base_selector["risk_logits"]
        )
        motion_anchor = F.smooth_l1_loss(
            selected["motion_embedding"], base_selector["motion_embedding"],
            beta=0.25,
        )
        feature_anchor = (
            (adapted - augmented).square() * mask[..., None]
        ).mean()
        loss = (
            classifier_ce + 0.35 * selector_ce
            + 0.55 * prototype + 0.20 * relation
            + 0.30 * risk_anchor + 0.15 * motion_anchor
            + args.feature_anchor * feature_anchor
            + args.parameter_anchor * adapter.parameter_penalty()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step in (1, args.steps // 2, args.steps):
            history.append({
                "step": step, "loss": float(loss.detach()),
                "classifier_ce": float(classifier_ce.detach()),
                "prototype": float(prototype.detach()),
                "risk_anchor": float(risk_anchor.detach()),
                "feature_anchor": float(feature_anchor.detach()),
            })
    adapter.eval()
    final = support_accuracy(
        classifier, selector, adapter, support_cache, support_labels, device
    )
    return adapter, {"initial": initial, "final": final, "history": history}


@torch.no_grad()
def apply_adapter(adapter, cache: dict, device: str) -> dict:
    result = dict(cache)
    features = []
    for start in range(0, len(cache["features"]), 64):
        stop = min(start + 64, len(cache["features"]))
        value = cache["features"][start:stop].to(device).float()
        mask = cache["frame_mask"][start:stop].to(device).bool()
        features.append(adapter(value, mask).cpu().half())
    result["features"] = torch.cat(features)
    result["source"] = "CAL3 prompted safe-support feature adaptation"
    return result


def adapt_sites(args, cache: dict, dataset, support: dict[str, np.ndarray],
                query: np.ndarray, sites: tuple[str, ...], prototypes,
                classifier, selector, device: str) -> tuple[dict, dict]:
    names = site_names(dataset.index)
    query_tensor = torch.from_numpy(query).long()
    output = slice_cache(cache, query_tensor)
    output_features = output["features"].clone()
    query_position = {
        int(source): local for local, source in enumerate(query.tolist())
    }
    audit = {}
    for site in sites:
        support_positions = torch.from_numpy(support[site]).long()
        support_cache = slice_cache(cache, support_positions)
        labels = torch.tensor(
            dataset.index.iloc[support[site]].class_id.to_numpy(dtype=np.int64)
        )
        adapter, fit_audit = fit_site_adapter(
            args, support_cache, labels, prototypes,
            classifier, selector, device,
        )
        site_query = [
            int(position) for position in query
            if names[int(position)] == site
        ]
        local = torch.tensor([
            query_position[position] for position in site_query
        ], dtype=torch.long)
        adapted = apply_adapter(
            adapter,
            slice_cache(cache, torch.tensor(site_query, dtype=torch.long)),
            device,
        )
        output_features.index_copy_(0, local, adapted["features"])
        audit[site] = fit_audit
    output["features"] = output_features
    output["source"] = "CAL3 site-wise prompted support adaptation"
    return output, audit


@torch.no_grad()
def classification_with_preserved_risk(args, adapted_cache, base_cache,
                                       action, risk, device: str) -> dict:
    classifier, selector = load_heads(args, device)
    adapted_action = predict_selector(
        classifier, adapted_cache, 64, device
    )["action_logits"]
    base_risk = predict_selector(
        selector, base_cache, 64, device
    )["risk_logits"]
    action_logits = (
        1.50 * adapted_cache["base_action_logits"].float()
        + 0.75 * adapted_action
    )
    risk_logits = base_cache["base_risk_logits"].float() + base_risk
    return classification_metrics(action_logits, risk_logits, action, risk)


def evaluate_pose(args, target, base_cache, adapted_cache,
                  distance_path: Path, device: str) -> tuple[dict, dict]:
    base_data = prepare_custom(
        args, target, base_cache, distance_path, device
    )
    adapted_data = prepare_custom(
        args, target, adapted_cache, distance_path, device
    )
    # Safe support does not identify danger boundaries, so retain the original
    # risk evidence while adapting action and motion evidence.
    adapted_data["risk_probability"] = base_data["risk_probability"]
    adapted_data["selector_risk_logits"] = base_data["selector_risk_logits"]
    base_pose = kp10_prediction(base_data, args, device).cpu()
    adapted_pose = kp10_prediction(adapted_data, args, device).cpu()
    return (
        _metric_batch(
            base_pose, base_data["target_pose"], base_data["target_valid"],
            base_data["target_risk"],
        ),
        _metric_batch(
            adapted_pose, adapted_data["target_pose"],
            adapted_data["target_valid"], adapted_data["target_risk"],
        ),
    )


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--feature-noise", type=float, default=0.01)
    parser.add_argument("--temporal-drop", type=float, default=0.02)
    parser.add_argument("--feature-anchor", type=float, default=0.20)
    parser.add_argument("--parameter-anchor", type=float, default=0.02)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=239)
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
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal3_kp10_seed239"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit_path = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit_path)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit_path)
    feature_root = args.selector_checkpoint.parent
    train_cache = torch.load(
        feature_root / "train_features.pt", map_location="cpu", weights_only=False
    )
    validation_cache = torch.load(
        feature_root / "val_features.pt", map_location="cpu", weights_only=False
    )
    _, _, train_action, _ = _load_pose_arrays(train)
    classifier, selector = load_heads(args, device)
    prototypes = class_prototypes(
        classifier, train_cache, train_action, device
    )

    source_support, source_query = split_support_query(
        validation.index, META_VALIDATION_SITES,
        args.support_per_class, args.seed + 17,
    )
    source_adapted, source_audit = adapt_sites(
        args, validation_cache, validation, source_support, source_query,
        META_VALIDATION_SITES, prototypes, classifier, selector, device,
    )
    source_base = slice_cache(
        validation_cache, torch.from_numpy(source_query).long()
    )
    source_target = QualityWeightedDataset(
        subset_dataset(validation.target, source_query), audit_path
    )
    source_base_pose, source_adapted_pose = evaluate_pose(
        args, source_target, source_base, source_adapted,
        args.run_dir / "source_validation_exact_distance.pt", device,
    )
    source_action = torch.tensor(
        validation.index.iloc[source_query].class_id.to_numpy(dtype=np.int64)
    )
    source_risk = torch.tensor(
        validation.index.iloc[source_query].risk_id.to_numpy(dtype=np.int64)
    )
    source_classification = {
        "base": classification_with_preserved_risk(
            args, source_base, source_base, source_action, source_risk, device
        ),
        "cal3": classification_with_preserved_risk(
            args, source_adapted, source_base, source_action, source_risk, device
        ),
    }
    source_promoted = (
        pose_selection_score(source_adapted_pose)
        < pose_selection_score(source_base_pose)
        and source_adapted_pose["danger_pose_mpjpe_m"]
        <= source_base_pose["danger_pose_mpjpe_m"]
        and source_adapted_pose["danger_distal_mpjpe_m"]
        <= source_base_pose["danger_distal_mpjpe_m"]
    )

    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline="sub", seed=args.seed
    )["test"]
    yja_full = QualityWeightedDataset(sealed, None)
    yja_cache = torch.load(
        args.yja_feature_cache, map_location="cpu", weights_only=False
    )
    yja_support, yja_query = split_support_query(
        sealed.index, ("yja_E02",),
        args.support_per_class, args.seed + 15,
    )
    yja_adapted, yja_audit = adapt_sites(
        args, yja_cache, yja_full, yja_support, yja_query,
        ("yja_E02",), prototypes, classifier, selector, device,
    )
    yja_base = slice_cache(yja_cache, torch.from_numpy(yja_query).long())
    yja_action = torch.tensor(
        sealed.index.iloc[yja_query].class_id.to_numpy(dtype=np.int64)
    )
    yja_risk = torch.tensor(
        sealed.index.iloc[yja_query].risk_id.to_numpy(dtype=np.int64)
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[yja_query].task.to_numpy() == C.TASK_POSE
    )
    pose_positions = yja_query[pose_local]
    yja_target = QualityWeightedDataset(
        subset_dataset(sealed, pose_positions), None
    )
    local = torch.from_numpy(pose_local).long()
    yja_base_pose, yja_adapted_pose = evaluate_pose(
        args, yja_target, slice_cache(yja_base, local),
        slice_cache(yja_adapted, local),
        args.run_dir / "yja_e02_exact_distance.pt", device,
    )
    yja_classification = {
        "base": classification_with_preserved_risk(
            args, yja_base, yja_base, yja_action, yja_risk, device
        ),
        "cal3": classification_with_preserved_risk(
            args, yja_adapted, yja_base, yja_action, yja_risk, device
        ),
    }
    result = {
        "run": "CAL3-KP10",
        "status": "completed",
        "protocol": args.exp,
        "contract": {
            "support_uses_csi_and_known_safe_action_only": True,
            "support_pose_gt_used": False,
            "query_labels_or_gt_used_for_adaptation": False,
            "risk_evidence_preserved_from_kp10": True,
        },
        "source_meta_validation": {
            "promoted": source_promoted,
            "base_pose": source_base_pose,
            "cal3_pose": source_adapted_pose,
            "classification": source_classification,
            "support_fit": source_audit,
        },
        "yja_e02": {
            "query_trials": int(len(yja_query)),
            "pose_trials": int(len(pose_positions)),
            "danger_trials": int((yja_risk == 2).sum()),
            "base_pose": yja_base_pose,
            "cal3_pose": yja_adapted_pose,
            "classification": yja_classification,
            "support_fit": yja_audit,
        },
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
