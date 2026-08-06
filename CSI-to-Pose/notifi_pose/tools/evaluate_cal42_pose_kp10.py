"""Evaluate fixed CAL42 action evidence in the CSI-only KP10 pose pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..cal42_kp10 import guarded_phase_blend
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from .audit_cal33_meta_risk import take
from .audit_cal42_phase_ensemble import (
    calibrated_logits,
    load_encoder,
    source_reference,
)
from .diagnose_observability import pose_only
from .evaluate_cal16_identity_spectrum_kp10 import _cache_for_model, _evaluate_pair
from .evaluate_cal4_linkmap_kp10 import load_coarse
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import (
    add_paths,
    configure_work_root,
    slice_cache,
    split_support_query,
)
from .train_cal23_dynamic_meta_kp10 import (
    action_risk_consistency,
    calibrated_cache,
    predict,
)
from .train_calibration_aware_v14 import subset_dataset
from .train_dynamic_motion import classification_metrics


def main() -> None:
    default_work = Path(r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--split-seed", type=int, default=272)
    parser.add_argument("--support-per-class", type=int, default=8)
    parser.add_argument("--phase-weight", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--energy-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = build_datasets(
        exp="single_split_lmh_e01", baseline=args.baseline, seed=17
    )["train"]
    target = build_datasets(
        exp="sealed", fold="yja_E02", baseline=args.baseline, seed=args.seed
    )["test"]
    pool, query_positions = split_support_query(
        target.index, ("yja_E02",), args.support_per_class, args.split_seed
    )
    support_positions = np.asarray(pool["yja_E02"])
    query_positions = np.asarray(query_positions)
    energy_model, energy_hierarchy = load_encoder(args.energy_checkpoint, device)
    phase_model, phase_hierarchy = load_encoder(args.phase_checkpoint, device)
    energy_reference = source_reference(
        energy_model, energy_hierarchy, source, args.batch_size, device
    )
    phase_reference = source_reference(
        phase_model, phase_hierarchy, source, args.batch_size, device
    )
    full_set = QualityWeightedDataset(target, None)
    energy = predict(energy_model, full_set, args.batch_size, device)
    phase = predict(phase_model, full_set, args.batch_size, device)
    energy_direct = action_risk_consistency(
        energy["action_logits"], energy["risk_logits"], energy_hierarchy
    )
    phase_direct = action_risk_consistency(
        phase["action_logits"], phase["risk_logits"], phase_hierarchy
    )
    energy_logits = calibrated_logits(
        energy, energy_direct, support_positions, query_positions, energy_reference
    )
    phase_logits = calibrated_logits(
        phase, phase_direct, support_positions, query_positions, phase_reference
    )
    alpha = float(args.phase_weight)
    blended = guarded_phase_blend(energy_logits, phase_logits, alpha)
    query = take(energy, query_positions)
    classification = classification_metrics(
        blended, query["risk_logits"], query["class_id"], query["risk_id"]
    )
    query_set = QualityWeightedDataset(
        subset_dataset(target, query_positions), None
    )
    base_model, _ = _load_model(args.kp4_checkpoint, device)
    identity = _cache_for_model(
        base_model, query_set, load_coarse(args.yja_coarse),
        "yja_E02", device, "CAL42 fixed yja query",
    )
    checkpoint = torch.load(
        args.energy_checkpoint, map_location="cpu", weights_only=False
    )
    fusion = checkpoint["fusion"]
    adapted = calibrated_cache(
        identity, blended, fusion["weight"], fusion["temperature"]
    )
    pose_positions = np.flatnonzero(
        target.index.iloc[query_positions].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        pose_only(subset_dataset(target, query_positions[pose_positions])), None
    )
    local = torch.as_tensor(pose_positions).long()
    base_pose, calibrated_pose = _evaluate_pair(
        args, pose_target, slice_cache(identity, local), slice_cache(adapted, local),
        args.run_dir, device,
    )
    version = "CAL43" if abs(alpha - 0.25) < 1e-9 else "CAL42"
    result = {
        "run": f"{version}-GUARDED-PHYSICAL-PHASE-KP10-POSE",
        "contract": {
            "target_query_used_for_calibration_or_selection": False,
            "phase_weight_fixed": args.phase_weight,
            "energy_danger_predictions_are_immutable": True,
        },
        "classification": classification,
        "kp10_base": base_pose,
        f"kp10_plus_{version.lower()}": calibrated_pose,
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
