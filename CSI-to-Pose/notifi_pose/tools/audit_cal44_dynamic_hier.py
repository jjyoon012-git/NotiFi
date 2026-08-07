"""Audit CAL44 with safe+warning calibration and no target danger support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..cal23_kp10 import DynamicMotionClassifier
from ..cal27_kp10 import action_risk_consistency
from ..cal27_kp10 import (
    apply_local_prototype as apply_safe_prototypes,
    fit_local_prototype as fit_safe_prototypes,
)
from ..cal42_kp10 import guarded_phase_blend
from ..cal44_kp10 import (
    DYNAMIC_CALIBRATION_CLASSES,
    apply_dynamic_prototypes,
    apply_hierarchical_risk,
    class_prototypes,
    fit_dynamic_prototypes,
    fit_hierarchical_risk,
    preserve_control_danger,
)
from ..dataio.dataset import PoseDataset, SiteBaseline, build_datasets
from ..calibration_quality import SAFE_CALIBRATION_CLASSES
from ..quality import QualityWeightedDataset
from .audit_cal33_meta_risk import summarize, take
from .train_cal1_kp10 import add_paths, configure_work_root
from .train_cal23_dynamic_meta_kp10 import predict
from .train_dynamic_motion import classification_metrics


def load_encoder(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DynamicMotionClassifier(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    hierarchy = float(
        checkpoint.get("hierarchy", {}).get("selected", {}).get("weight", 0.0)
    )
    return model, hierarchy


def split_dynamic_support(index, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = index.class_id.to_numpy(dtype=np.int64)
    support = []
    for class_id in DYNAMIC_CALIBRATION_CLASSES:
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) <= repeats:
            raise ValueError(f"class {class_id} has no query after support split")
        support.extend(rng.choice(candidates, repeats, replace=False).tolist())
    support = np.sort(np.asarray(support, dtype=np.int64))
    query = np.setdiff1d(np.arange(len(index)), support)
    return support, query


def calibrated_action(full, direct, support, query, prototypes):
    support_view = take(full, support)
    support_direct = direct.index_select(0, torch.as_tensor(support).long())
    calibration = fit_dynamic_prototypes(
        support_view["embedding"], support_direct, support_view["class_id"],
        prototypes,
    )
    query_view = take(full, query)
    query_direct = direct.index_select(0, torch.as_tensor(query).long())
    query_logits = apply_dynamic_prototypes(
        query_view["embedding"], query_direct,
        support_view["embedding"], support_view["class_id"],
        prototypes, calibration,
    )
    support_logits = apply_dynamic_prototypes(
        support_view["embedding"], support_direct,
        support_view["embedding"], support_view["class_id"],
        prototypes, calibration,
    )
    return query_logits, support_logits, calibration


def calibrated_safe_action(full, direct, support, query, prototypes, safe_mean):
    support_view = take(full, support)
    support_direct = direct.index_select(0, torch.as_tensor(support).long())
    query_view = take(full, query)
    query_direct = direct.index_select(0, torch.as_tensor(query).long())
    calibration = fit_safe_prototypes(
        support_view["embedding"], support_direct, support_view["class_id"],
        prototypes, safe_mean,
    )
    return apply_safe_prototypes(
        query_view["embedding"], query_direct,
        support_view["embedding"], support_view["class_id"],
        prototypes, safe_mean, calibration,
    )


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--loso-fold", default=None)
    parser.add_argument("--target-environment", default="E01")
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--phase-weight", type=float, default=0.15)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=tuple(range(272, 288)))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--energy-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.loso_fold:
        datasets = build_datasets(
            exp="loso", fold=args.loso_fold, baseline=args.baseline, seed=17
        )
        source = datasets["train"]
        subject = args.loso_fold.removeprefix("test_")
        cache = source.cache
        rows = np.flatnonzero(((
            cache.index.subject == subject
        ) & (
            cache.index.environment == args.target_environment
        ) & cache.index.cache_ok.astype(bool)).to_numpy())
        target = PoseDataset(
            rows, cache, source.link_ok, train=False, seed=503,
            baseline=SiteBaseline(args.baseline),
        )
        target_site = f"{subject}_{args.target_environment}"
    else:
        source = build_datasets(
            exp="single_split_lmh_e01", baseline=args.baseline, seed=17
        )["train"]
        target = build_datasets(
            exp="sealed", fold="yja_E02", baseline=args.baseline, seed=503
        )["test"]
        target_site = "yja_E02"

    energy_model, energy_hierarchy = load_encoder(args.energy_checkpoint, device)
    phase_model, phase_hierarchy = load_encoder(args.phase_checkpoint, device)
    source_energy = predict(
        energy_model, QualityWeightedDataset(source, None), args.batch_size, device
    )
    source_phase = predict(
        phase_model, QualityWeightedDataset(source, None), args.batch_size, device
    )
    energy_prototypes = class_prototypes(
        source_energy["embedding"], source_energy["class_id"]
    )
    phase_prototypes = class_prototypes(
        source_phase["embedding"], source_phase["class_id"]
    )
    source_energy_safe = torch.zeros_like(source_energy["class_id"], dtype=torch.bool)
    source_phase_safe = torch.zeros_like(source_phase["class_id"], dtype=torch.bool)
    for class_id in SAFE_CALIBRATION_CLASSES:
        source_energy_safe |= source_energy["class_id"] == class_id
        source_phase_safe |= source_phase["class_id"] == class_id
    energy_safe_mean = source_energy["embedding"][source_energy_safe].mean(0)
    phase_safe_mean = source_phase["embedding"][source_phase_safe].mean(0)
    energy = predict(
        energy_model, QualityWeightedDataset(target, None), args.batch_size, device
    )
    phase = predict(
        phase_model, QualityWeightedDataset(target, None), args.batch_size, device
    )
    energy_direct = action_risk_consistency(
        energy["action_logits"], energy["risk_logits"], energy_hierarchy
    )
    phase_direct = action_risk_consistency(
        phase["action_logits"], phase["risk_logits"], phase_hierarchy
    )

    rows = []
    for split_seed in args.split_seeds:
        support, query = split_dynamic_support(
            target.index, args.support_per_class, split_seed
        )
        energy_query, energy_support, energy_cal = calibrated_action(
            energy, energy_direct, support, query, energy_prototypes
        )
        phase_query, phase_support, phase_cal = calibrated_action(
            phase, phase_direct, support, query, phase_prototypes
        )
        action_query = guarded_phase_blend(
            energy_query, phase_query, args.phase_weight
        )
        action_support = guarded_phase_blend(
            energy_support, phase_support, args.phase_weight
        )
        support_view = take(energy, support)
        query_view = take(energy, query)
        support_tensor = torch.as_tensor(support).long()
        direct_support = guarded_phase_blend(
            energy_direct.index_select(0, support_tensor),
            phase_direct.index_select(0, support_tensor),
            args.phase_weight,
        )
        risk_cal = fit_hierarchical_risk(
            direct_support, support_view["risk_logits"], support_view["risk_id"]
        )
        target_labels = target.index.class_id.to_numpy(dtype=np.int64)
        safe_support = support[np.isin(
            target_labels[support], np.asarray(SAFE_CALIBRATION_CLASSES)
        )]
        control_energy = calibrated_safe_action(
            energy, energy_direct, safe_support, query,
            energy_prototypes, energy_safe_mean,
        )
        control_phase = calibrated_safe_action(
            phase, phase_direct, safe_support, query,
            phase_prototypes, phase_safe_mean,
        )
        control_action = guarded_phase_blend(
            control_energy, control_phase, args.phase_weight
        )
        action_query = preserve_control_danger(
            control_action, action_query, danger_start=12
        )
        # The support-selected hierarchy is audited above, but CAL42 risk is
        # kept unchanged until a multi-site risk improvement is demonstrated.
        risk_query = query_view["risk_logits"]
        metrics = classification_metrics(
            action_query, risk_query,
            query_view["class_id"], query_view["risk_id"],
        )
        control = classification_metrics(
            control_action, query_view["risk_logits"],
            query_view["class_id"], query_view["risk_id"],
        )
        rows.append({
            "split_seed": split_seed,
            "energy_prototype_weight": energy_cal["selected"]["weight"],
            "phase_prototype_weight": phase_cal["selected"]["weight"],
            "risk_hierarchy_weight": risk_cal["selected"]["weight"],
            **metrics,
            **{f"control_{key}": value for key, value in control.items()},
            "action_accuracy_delta": (
                metrics["action_accuracy"] - control["action_accuracy"]
            ),
            "risk_accuracy_delta": (
                metrics["risk_accuracy"] - control["risk_accuracy"]
            ),
        })
    keys = (
        "action_accuracy", "action_macro_f1", "danger_action_accuracy",
        "risk_accuracy", "risk_macro_f1", "danger_recall", "safe_to_danger",
        "energy_prototype_weight", "phase_prototype_weight",
        "risk_hierarchy_weight",
        "control_action_accuracy", "control_action_macro_f1",
        "control_risk_accuracy", "control_risk_macro_f1",
        "control_danger_recall", "control_safe_to_danger",
        "action_accuracy_delta", "risk_accuracy_delta",
    )
    result = {
        "run": "CAL44-DYNAMIC-ACTION-CAL-KP10",
        "status": "EXPERIMENTAL_AUDIT_ONLY",
        "contract": {
            "target_site": target_site,
            "safe_warning_support_per_action": args.support_per_class,
            "danger_support_trials": 0,
            "target_query_used_for_calibration_or_selection": False,
            "phase_weight_fixed_before_target_audit": args.phase_weight,
            "risk_branch": "CAL42_ENERGY_UNCHANGED",
            "hierarchical_risk_candidate_accepted": False,
        },
        "summary": {key: summarize(rows, key) for key in keys},
        "splits": rows,
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
