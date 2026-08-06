"""Evaluate deployable target-local safe prototypes with CAL23 and KP10."""

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
from ..calibration_quality import SAFE_CALIBRATION_CLASSES
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from ..trainer import set_seed
from .evaluate_cal16_identity_spectrum_kp10 import _cache_for_model, _evaluate_pair
from .evaluate_cal4_linkmap_kp10 import load_coarse
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import add_paths, configure_work_root, slice_cache, split_support_query
from .train_cal23_dynamic_meta_kp10 import (
    action_risk_consistency,
    calibrated_cache,
    conformal_safe_threshold,
    danger_score,
    metrics_with_threshold,
    predict,
    safe_location_scale,
    standardize_score,
)
from .train_calibration_aware_v14 import subset_dataset


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-reserve-per-class", type=int, default=8)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--cal23-checkpoint", type=Path,
        default=default_work / "runs/cal23_dynamic_meta_kp10_v2/calibration_candidate.pt",
    )
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
    args.run_dir = args.run_dir or args.work_root / "runs/cal27_local_prototype_kp10"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        args.cal23_checkpoint, map_location="cpu", weights_only=False
    )
    model = DynamicMotionClassifier(
        **checkpoint.get("model_config", {})
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    hierarchy_weight = float(
        checkpoint.get("hierarchy", {}).get("selected", {}).get("weight", 0.0)
    )
    source = build_datasets(
        exp="single_split_lmh_e01", baseline=args.baseline, seed=17
    )["train"]
    source_prediction = predict(
        model, QualityWeightedDataset(source, None), args.batch_size, device
    )
    prototypes = class_prototypes(
        source_prediction["embedding"], source_prediction["class_id"]
    )
    safe_mask = torch.zeros_like(source_prediction["class_id"], dtype=torch.bool)
    for class_id in SAFE_CALIBRATION_CLASSES:
        safe_mask |= source_prediction["class_id"] == class_id
    source_safe_mean = source_prediction["embedding"][safe_mask].mean(0)

    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline=args.baseline, seed=args.seed
    )["test"]
    pool, query_positions = split_support_query(
        sealed.index, ("yja_E02",), args.target_reserve_per_class,
        TARGET_CALIBRATION_SPLIT_SEED,
    )
    support_positions = pool["yja_E02"]
    support_set = QualityWeightedDataset(
        subset_dataset(sealed, support_positions), None
    )
    query_set = QualityWeightedDataset(
        subset_dataset(sealed, query_positions), None
    )
    support = predict(model, support_set, args.batch_size, device)
    query = predict(model, query_set, args.batch_size, device)
    support_direct = action_risk_consistency(
        support["action_logits"], support["risk_logits"], hierarchy_weight
    )
    query_direct = action_risk_consistency(
        query["action_logits"], query["risk_logits"], hierarchy_weight
    )
    calibration = fit_local_prototype(
        support["embedding"], support_direct, support["class_id"],
        prototypes, source_safe_mean,
    )
    support_action = apply_local_prototype(
        support["embedding"], support_direct,
        support["embedding"], support["class_id"],
        prototypes, source_safe_mean, calibration,
    )
    query_action = apply_local_prototype(
        query["embedding"], query_direct,
        support["embedding"], support["class_id"],
        prototypes, source_safe_mean, calibration,
    )
    target_statistics = safe_location_scale(
        danger_score(support["risk_logits"]), support["risk_id"]
    )
    support_score = standardize_score(
        danger_score(support["risk_logits"]), target_statistics
    )
    query_score = standardize_score(
        danger_score(query["risk_logits"]), target_statistics
    )
    threshold = conformal_safe_threshold(support_score, 0.10)
    support_false = int((support_score >= threshold).sum())
    classification = metrics_with_threshold(
        query_action, query["risk_logits"], query_score, threshold,
        query["class_id"], query["risk_id"],
    )
    support_accuracy = float((
        support_action.argmax(-1) == support["class_id"]
    ).float().mean())
    experimental_action = bool(
        calibration["selected"]["accuracy"] >= 0.35
        and support_accuracy >= 0.55
    )
    action_ready = bool(
        calibration["selected"]["accuracy"] >= 0.60
        and support_accuracy >= 0.70
    )
    # The prompt contains no warning/danger trials.  It can constrain safe
    # false alarms, but it cannot validate target-site danger direction.
    risk_ready = False

    base_model, _ = _load_model(args.kp4_checkpoint, device)
    identity = _cache_for_model(
        base_model, query_set, load_coarse(args.yja_coarse),
        "yja_E02", device, "CAL27 fixed yja query",
    )
    fusion = checkpoint["fusion"]
    fused = calibrated_cache(
        identity, query_action, fusion["weight"], fusion["temperature"]
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[query_positions].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, query_positions[pose_local]), None
    )
    local = torch.as_tensor(pose_local).long()
    base_pose, calibrated_pose = _evaluate_pair(
        args, pose_target, slice_cache(identity, local), slice_cache(fused, local),
        args.run_dir / "yja_fixed_query", device,
    )
    partial = experimental_action
    result = {
        "run": "CAL27-LOCAL-PROTOTYPE-CAL-KP10",
        "status": "EXPERIMENTAL" if partial else "REJECT",
        "contract": {
            "fixed_physical_link_order": ["TX1_South", "TX2_West", "TX3_East"],
            "support_per_safe_action": args.target_reserve_per_class,
            "target_query_used_for_calibration_or_selection": False,
            "safe_prototypes_are_target_local": True,
            "warning_and_danger_prototypes_are_source_only": True,
        },
        "calibration_quality": {
            "accepted_for_normal_inference": False,
            "accepted_for_action_pose_inference": action_ready,
            "experimental_action_pose_candidate": experimental_action,
            "action_pose_ready": action_ready, "risk_ready": risk_ready,
            "risk_reason": "safe_only_calibration_cannot_validate_danger_direction",
            "held_repeat_accuracy": calibration["selected"]["accuracy"],
            "full_support_accuracy": support_accuracy,
            "support_false_danger": support_false,
            "selected": calibration["selected"],
        },
        "yja_e02": {
            "query_trials": len(query_positions),
            "operational_classification": classification,
            "kp10_base": base_pose,
            "kp10_plus_cal27": calibrated_pose,
        },
    }
    torch.save({
        "run": result["run"],
        "deployable": False,
        "action_pose_deployable": action_ready,
        "experimental_action_pose": experimental_action,
        "calibration_quality": result["calibration_quality"],
        "dynamic_model_state_dict": checkpoint["model_state_dict"],
        "dynamic_model_config": checkpoint.get("model_config", {}),
        "source_prototypes": prototypes, "source_safe_mean": source_safe_mean,
        "support_embedding": support["embedding"],
        "support_labels": support["class_id"],
        "prototype_calibration": calibration,
        "risk_statistics": target_statistics, "risk_threshold": threshold,
        "hierarchy_weight": hierarchy_weight, "kp10_fusion": fusion,
        "support_rows": sealed.rows[support_positions].tolist(),
    }, args.run_dir / "deployment_candidate.pt")
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
