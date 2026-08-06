"""Audit CAL33 on sealed unseen subjects with support-only calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..cal23_kp10 import DynamicMotionClassifier
from ..cal27_kp10 import (
    apply_local_prototype,
    class_prototypes,
    fit_local_prototype,
)
from ..cal33_kp10 import (
    MetaRiskHead,
    apply_risk_group_gate,
    build_safe_context,
    meta_risk_features,
)
from ..calibration_quality import SAFE_CALIBRATION_CLASSES
from ..dataio.dataset import PoseDataset, SiteBaseline, build_datasets
from ..quality import QualityWeightedDataset
from ..trainer import set_seed
from .train_cal1_kp10 import add_paths, configure_work_root, split_support_query
from .train_cal23_dynamic_meta_kp10 import (
    action_risk_consistency,
    conformal_safe_threshold,
    danger_score,
    predict,
    safe_location_scale,
    standardize_score,
    threshold_risk,
)
from .train_dynamic_motion import classification_metrics


def take(values: dict, positions: np.ndarray) -> dict:
    index = torch.as_tensor(positions).long()
    return {key: value.index_select(0, index) for key, value in values.items()}


def summarize(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def _risk_logits(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    output = reference.new_full((len(prediction), 3), -20.0)
    output.scatter_(1, prediction[:, None], 20.0)
    return output


def _split_prompt_query(index, site: str, safe_repeats: int,
                        warning_repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    names = (index.subject.astype(str) + "_" + index.environment.astype(str)).to_numpy()
    labels = index.class_id.to_numpy(dtype=np.int64)
    site_positions = np.flatnonzero(names == site)
    selected = []
    for class_id in SAFE_CALIBRATION_CLASSES:
        candidates = site_positions[labels[site_positions] == class_id]
        selected.extend(generator.choice(
            candidates, safe_repeats, replace=False
        ).tolist())
    if warning_repeats:
        for class_id in (9, 10, 11):
            candidates = site_positions[labels[site_positions] == class_id]
            selected.extend(generator.choice(
                candidates, warning_repeats, replace=False
            ).tolist())
    support = np.sort(np.asarray(selected, dtype=np.int64))
    query = np.setdiff1d(site_positions, support, assume_unique=False)
    return support, query


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=439)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-reserve-per-class", type=int, default=8)
    parser.add_argument("--warning-reserve-per-class", type=int, default=4)
    parser.add_argument(
        "--prompt-mode", choices=("safe", "safe_warning"), default="safe"
    )
    parser.add_argument("--loso-fold", default=None)
    parser.add_argument("--target-environment", default="E01")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=tuple(range(272, 288)))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cal23-checkpoint", type=Path, required=True)
    parser.add_argument("--cal33-checkpoint", type=Path, nargs="+", required=True)
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_checkpoint = torch.load(
        args.cal23_checkpoint, map_location="cpu", weights_only=False
    )
    encoder = DynamicMotionClassifier(
        **encoder_checkpoint.get("model_config", {})
    ).to(device)
    encoder.load_state_dict(encoder_checkpoint["model_state_dict"])
    risk_models = []
    for path in args.cal33_checkpoint:
        risk_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        risk_model = MetaRiskHead(**risk_checkpoint.get("model_config", {})).to(device)
        risk_model.load_state_dict(risk_checkpoint["model_state_dict"])
        risk_model.eval()
        risk_models.append(risk_model)
    hierarchy_weight = float(
        encoder_checkpoint.get("hierarchy", {}).get("selected", {}).get("weight", 0.0)
    )
    if args.loso_fold:
        source_sets = build_datasets(
            exp="loso", fold=args.loso_fold, baseline=args.baseline, seed=17
        )
        source = source_sets["train"]
        held_subject = args.loso_fold.removeprefix("test_")
        cache = source.cache
        target_rows = np.flatnonzero(((
            cache.index.subject == held_subject
        ) & (
            cache.index.environment == args.target_environment
        ) & cache.index.cache_ok.astype(bool)).to_numpy())
        sealed = PoseDataset(
            target_rows, cache, source.link_ok, train=False, seed=args.seed,
            baseline=SiteBaseline(args.baseline),
        )
        target_site = f"{held_subject}_{args.target_environment}"
    else:
        source = build_datasets(
            exp="single_split_lmh_e01", baseline=args.baseline, seed=17
        )["train"]
        sealed = build_datasets(
            exp="sealed", fold="yja_E02", baseline=args.baseline, seed=args.seed
        )["test"]
        target_site = "yja_E02"
    source_prediction = predict(
        encoder, QualityWeightedDataset(source, None), args.batch_size, device
    )
    prototypes = class_prototypes(
        source_prediction["embedding"], source_prediction["class_id"]
    )
    safe = torch.zeros_like(source_prediction["class_id"], dtype=torch.bool)
    for class_id in SAFE_CALIBRATION_CLASSES:
        safe |= source_prediction["class_id"] == class_id
    source_safe_mean = source_prediction["embedding"][safe].mean(0)
    full = predict(
        encoder, QualityWeightedDataset(sealed, None), args.batch_size, device
    )
    full_direct = action_risk_consistency(
        full["action_logits"], full["risk_logits"], hierarchy_weight
    )
    rows = []
    support_classes = tuple(SAFE_CALIBRATION_CLASSES)
    if args.prompt_mode == "safe_warning":
        support_classes += (9, 10, 11)
    for split_seed in args.split_seeds:
        if args.prompt_mode == "safe_warning":
            support_positions, query_positions = _split_prompt_query(
                sealed.index, target_site, args.target_reserve_per_class,
                args.warning_reserve_per_class, split_seed,
            )
        else:
            pool, query_positions = split_support_query(
                sealed.index, (target_site,), args.target_reserve_per_class, split_seed
            )
            support_positions = np.asarray(pool[target_site])
        support = take(full, support_positions)
        query = take(full, np.asarray(query_positions))
        support_direct = full_direct.index_select(
            0, torch.as_tensor(support_positions).long()
        )
        query_direct = full_direct.index_select(
            0, torch.as_tensor(query_positions).long()
        )
        local = fit_local_prototype(
            support["embedding"], support_direct, support["class_id"],
            prototypes, source_safe_mean,
        )
        support_action = apply_local_prototype(
            support["embedding"], support_direct,
            support["embedding"], support["class_id"],
            prototypes, source_safe_mean, local,
        )
        query_action = apply_local_prototype(
            query["embedding"], query_direct,
            support["embedding"], support["class_id"],
            prototypes, source_safe_mean, local,
        )
        context = build_safe_context(
            support["embedding"], support["risk_logits"], support["class_id"],
            support_classes,
        )
        with torch.no_grad():
            support_feature = meta_risk_features(
                support["embedding"], support["risk_logits"], context
            ).to(device)
            query_feature = meta_risk_features(
                query["embedding"], query["risk_logits"], context
            ).to(device)
            support_meta = torch.stack([
                model(support_feature).cpu() for model in risk_models
            ]).mean(0)
            query_meta = torch.stack([
                model(query_feature).cpu() for model in risk_models
            ]).mean(0)
        statistics = safe_location_scale(
            danger_score(support_meta), support["risk_id"]
        )
        support_score = standardize_score(danger_score(support_meta), statistics)
        query_score = standardize_score(danger_score(query_meta), statistics)
        threshold = conformal_safe_threshold(
            support_score[support["risk_id"] == 0], 0.10
        )
        risk_prediction = threshold_risk(query_meta, query_score, threshold)
        gated_action = apply_risk_group_gate(query_action, risk_prediction)
        metrics = classification_metrics(
            gated_action, _risk_logits(risk_prediction, query_meta),
            query["class_id"], query["risk_id"],
        )
        raw = classification_metrics(
            query_action, query_meta, query["class_id"], query["risk_id"]
        )
        rows.append({
            "split_seed": split_seed,
            "support_false_danger": int((support_score >= threshold).sum()),
            "local_support_accuracy": float((
                support_action.argmax(-1) == support["class_id"]
            ).float().mean()),
            "raw_meta_risk_accuracy": raw["risk_accuracy"],
            "raw_meta_danger_recall": raw["danger_recall"],
            **{f"cal33_{key}": value for key, value in metrics.items()},
        })
    summary_keys = [
        "local_support_accuracy", "raw_meta_risk_accuracy",
        "raw_meta_danger_recall", "cal33_action_accuracy",
        "cal33_action_macro_f1", "cal33_risk_accuracy",
        "cal33_risk_macro_f1", "cal33_danger_recall",
        "cal33_danger_action_accuracy", "cal33_safe_to_danger",
    ]
    result = {
        "run": "CAL33-EPISODIC-META-RISK-AUDIT",
        "contract": {
            "target_site": target_site,
            "support_per_safe_action": args.target_reserve_per_class,
            "support_per_warning_action": (
                args.warning_reserve_per_class
                if args.prompt_mode == "safe_warning" else 0
            ),
            "prompt_mode": args.prompt_mode,
            "target_query_used_for_calibration_or_selection": False,
            "risk_certified": False,
            "meta_risk_ensemble_size": len(risk_models),
        },
        "draws": len(rows),
        "summary": {key: summarize(rows, key) for key in summary_keys},
        "splits": rows,
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
