"""Evaluate CAL33 risk-gated action evidence in the frozen KP10 pose path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..cal16_kp10 import TARGET_CALIBRATION_SPLIT_SEED
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
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from ..trainer import set_seed
from .audit_cal33_meta_risk import _risk_logits
from .evaluate_cal16_identity_spectrum_kp10 import _cache_for_model, _evaluate_pair
from .evaluate_cal4_linkmap_kp10 import load_coarse
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import add_paths, configure_work_root, slice_cache, split_support_query
from .train_cal23_dynamic_meta_kp10 import (
    action_risk_consistency,
    calibrated_cache,
    conformal_safe_threshold,
    danger_score,
    predict,
    safe_location_scale,
    standardize_score,
    threshold_risk,
)
from .train_calibration_aware_v14 import subset_dataset
from .train_dynamic_motion import classification_metrics


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=449)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-reserve-per-class", type=int, default=8)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cal23-checkpoint", type=Path, required=True)
    parser.add_argument("--cal33-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--kp4-checkpoint", type=Path,
        default=default_work / "runs/kp4_dcc_staged_seed17/deployment_model.pt",
    )
    parser.add_argument(
        "--yja-coarse", type=Path,
        default=default_work / "runs/cal2_kp10_seed223_danger_gate/yja_e02_v13s_coarse.pt",
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.cal23_checkpoint, map_location="cpu", weights_only=False)
    encoder = DynamicMotionClassifier(**checkpoint.get("model_config", {})).to(device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    meta_checkpoint = torch.load(
        args.cal33_checkpoint, map_location="cpu", weights_only=False
    )
    risk_model = MetaRiskHead(**meta_checkpoint.get("model_config", {})).to(device)
    risk_model.load_state_dict(meta_checkpoint["model_state_dict"])
    risk_model.eval()
    hierarchy_weight = float(
        checkpoint.get("hierarchy", {}).get("selected", {}).get("weight", 0.0)
    )
    source = build_datasets(
        exp="single_split_lmh_e01", baseline=args.baseline, seed=17
    )["train"]
    source_prediction = predict(
        encoder, QualityWeightedDataset(source, None), args.batch_size, device
    )
    prototypes = class_prototypes(
        source_prediction["embedding"], source_prediction["class_id"]
    )
    source_safe = torch.zeros_like(source_prediction["class_id"], dtype=torch.bool)
    for class_id in SAFE_CALIBRATION_CLASSES:
        source_safe |= source_prediction["class_id"] == class_id
    source_safe_mean = source_prediction["embedding"][source_safe].mean(0)
    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline=args.baseline, seed=args.seed
    )["test"]
    pool, query_positions = split_support_query(
        sealed.index, ("yja_E02",), args.target_reserve_per_class,
        TARGET_CALIBRATION_SPLIT_SEED,
    )
    support_positions = np.asarray(pool["yja_E02"])
    full = predict(
        encoder, QualityWeightedDataset(sealed, None), args.batch_size, device
    )
    support_index = torch.as_tensor(support_positions).long()
    query_index = torch.as_tensor(query_positions).long()
    support = {key: value.index_select(0, support_index) for key, value in full.items()}
    query = {key: value.index_select(0, query_index) for key, value in full.items()}
    full_direct = action_risk_consistency(
        full["action_logits"], full["risk_logits"], hierarchy_weight
    )
    support_direct = full_direct.index_select(0, support_index)
    query_direct = full_direct.index_select(0, query_index)
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
        support["embedding"], support["risk_logits"], support["class_id"]
    )
    with torch.no_grad():
        support_meta = risk_model(meta_risk_features(
            support["embedding"], support["risk_logits"], context
        ).to(device)).cpu()
        query_meta = risk_model(meta_risk_features(
            query["embedding"], query["risk_logits"], context
        ).to(device)).cpu()
    statistics = safe_location_scale(danger_score(support_meta), support["risk_id"])
    support_score = standardize_score(danger_score(support_meta), statistics)
    query_score = standardize_score(danger_score(query_meta), statistics)
    threshold = conformal_safe_threshold(support_score, 0.10)
    risk_prediction = threshold_risk(query_meta, query_score, threshold)
    gated_action = apply_risk_group_gate(query_action, risk_prediction)
    classification = classification_metrics(
        gated_action, _risk_logits(risk_prediction, query_meta),
        query["class_id"], query["risk_id"],
    )

    base_model, _ = _load_model(args.kp4_checkpoint, device)
    identity = _cache_for_model(
        base_model, QualityWeightedDataset(
            subset_dataset(sealed, query_positions), None
        ), load_coarse(args.yja_coarse), "yja_E02", device,
        "CAL33 fixed yja query",
    )
    fusion = checkpoint["fusion"]
    local_cache = calibrated_cache(
        identity, query_action, fusion["weight"], fusion["temperature"]
    )
    gated_cache = calibrated_cache(
        identity, gated_action, fusion["weight"], fusion["temperature"]
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[query_positions].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, query_positions[pose_local]), None
    )
    local_index = torch.as_tensor(pose_local).long()
    base_pose, local_pose = _evaluate_pair(
        args, pose_target, slice_cache(identity, local_index),
        slice_cache(local_cache, local_index), args.run_dir / "cal27", device,
    )
    _, gated_pose = _evaluate_pair(
        args, pose_target, slice_cache(identity, local_index),
        slice_cache(gated_cache, local_index), args.run_dir / "cal33", device,
    )
    result = {
        "run": "CAL33-META-RISK-GATED-KP10",
        "status": "EXPERIMENTAL",
        "contract": {
            "target_query_used_for_calibration_or_selection": False,
            "fixed_target_split_seed": TARGET_CALIBRATION_SPLIT_SEED,
            "risk_certified": False,
        },
        "support": {
            "trials": len(support_positions),
            "local_action_accuracy": float((
                support_action.argmax(-1) == support["class_id"]
            ).float().mean()),
            "false_danger": int((support_score >= threshold).sum()),
        },
        "query_classification": classification,
        "kp10_base": base_pose,
        "kp10_plus_cal27": local_pose,
        "kp10_plus_cal33": gated_pose,
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
