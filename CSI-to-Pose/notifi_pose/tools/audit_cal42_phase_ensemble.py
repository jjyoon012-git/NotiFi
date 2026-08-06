"""Audit a fixed low-weight physical-phase ensemble without target selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import contract as C
from ..cal23_kp10 import DynamicMotionClassifier
from ..cal27_kp10 import (
    apply_local_prototype,
    class_prototypes,
    fit_local_prototype,
)
from ..cal42_kp10 import guarded_phase_blend, risk_logits_from_action
from ..calibration_quality import SAFE_CALIBRATION_CLASSES
from ..dataio.dataset import PoseDataset, SiteBaseline, build_datasets
from ..quality import QualityWeightedDataset
from .audit_cal33_meta_risk import summarize, take
from .train_cal1_kp10 import add_paths, configure_work_root, split_support_query
from .train_cal23_dynamic_meta_kp10 import action_risk_consistency, predict
from .train_dynamic_motion import classification_metrics


def load_encoder(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DynamicMotionClassifier(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    hierarchy = float(
        checkpoint.get("hierarchy", {}).get("selected", {}).get("weight", 0.0)
    )
    return model, hierarchy


def source_reference(model, hierarchy, source, batch_size, device):
    prediction = predict(
        model, QualityWeightedDataset(source, None), batch_size, device
    )
    prototypes = class_prototypes(prediction["embedding"], prediction["class_id"])
    mask = torch.zeros_like(prediction["class_id"], dtype=torch.bool)
    for class_id in SAFE_CALIBRATION_CLASSES:
        mask |= prediction["class_id"] == class_id
    return {
        "prototypes": prototypes,
        "safe_mean": prediction["embedding"][mask].mean(0),
        "hierarchy": hierarchy,
    }


def calibrated_logits(full, direct, support_positions, query_positions, reference):
    support = take(full, support_positions)
    support_direct = direct.index_select(0, torch.as_tensor(support_positions).long())
    query = take(full, query_positions)
    query_direct = direct.index_select(0, torch.as_tensor(query_positions).long())
    calibration = fit_local_prototype(
        support["embedding"], support_direct, support["class_id"],
        reference["prototypes"], reference["safe_mean"],
    )
    return apply_local_prototype(
        query["embedding"], query_direct,
        support["embedding"], support["class_id"],
        reference["prototypes"], reference["safe_mean"], calibration,
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
    parser.add_argument("--support-per-class", type=int, default=8)
    parser.add_argument(
        "--common-query-reserve-per-class", type=int, default=None
    )
    parser.add_argument("--phase-weight", type=float, default=0.15)
    parser.add_argument("--preserve-energy-danger", action="store_true")
    parser.add_argument(
        "--split-seeds", type=int, nargs="+", default=tuple(range(272, 288))
    )
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
        rows = np.flatnonzero((
            (cache.index.subject == subject)
            & (cache.index.environment == args.target_environment)
            & cache.index.cache_ok.astype(bool)
        ).to_numpy())
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
    energy_reference = source_reference(
        energy_model, energy_hierarchy, source, args.batch_size, device
    )
    phase_reference = source_reference(
        phase_model, phase_hierarchy, source, args.batch_size, device
    )
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
        query_reserve = args.common_query_reserve_per_class or args.support_per_class
        if query_reserve < args.support_per_class:
            raise ValueError("common query reserve cannot be smaller than support")
        pool, query = split_support_query(
            target.index, (target_site,), query_reserve, split_seed
        )
        support_pool = np.asarray(pool[target_site])
        if query_reserve == args.support_per_class:
            support = support_pool
        else:
            labels = target.index.class_id.to_numpy(dtype=np.int64)
            support = np.sort(np.concatenate([
                support_pool[labels[support_pool] == class_id][
                    :args.support_per_class
                ]
                for class_id in SAFE_CALIBRATION_CLASSES
            ]))
        query = np.asarray(query)
        energy_logits = calibrated_logits(
            energy, energy_direct, support, query, energy_reference
        )
        phase_logits = calibrated_logits(
            phase, phase_direct, support, query, phase_reference
        )
        alpha = float(args.phase_weight)
        if args.preserve_energy_danger:
            blended = guarded_phase_blend(energy_logits, phase_logits, alpha)
        else:
            blended = (
                (1.0 - alpha) * F.log_softmax(energy_logits, dim=-1)
                + alpha * F.log_softmax(phase_logits, dim=-1)
            )
        target_query = take(energy, query)
        metrics = classification_metrics(
            blended, target_query["risk_logits"],
            target_query["class_id"], target_query["risk_id"],
        )
        grouped = classification_metrics(
            blended, risk_logits_from_action(blended),
            target_query["class_id"], target_query["risk_id"],
        )
        rows.append({
            "split_seed": split_seed,
            **metrics,
            **{f"grouped_{key}": value for key, value in grouped.items()},
        })
    keys = (
        "action_accuracy", "action_macro_f1", "danger_action_accuracy",
        "danger_recall", "safe_to_danger",
        "grouped_risk_accuracy", "grouped_risk_macro_f1",
        "grouped_danger_recall", "grouped_safe_to_danger",
    )
    if abs(float(args.phase_weight) - 0.25) < 1e-9:
        run_name = "CAL43-GUARDED-PHYSICAL-PHASE-KP10"
    else:
        run_name = "CAL42-GUARDED-PHYSICAL-PHASE-KP10"
    result = {
        "run": run_name,
        "contract": {
            "target_site": target_site,
            "phase_weight_fixed_before_target_audit": args.phase_weight,
            "support_per_safe_action": args.support_per_class,
            "common_query_reserve_per_safe_action": query_reserve,
            "target_query_used_for_calibration_or_selection": False,
            "risk_branch": "energy_only",
            "energy_danger_predictions_are_immutable": bool(
                args.preserve_energy_danger
            ),
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
