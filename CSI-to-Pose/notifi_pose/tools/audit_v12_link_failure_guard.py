"""Audit the validation-only V12 missing-link fallback without opening test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..hybrid_v10 import (
    ConditionalLinkFailureLogitBlend,
    ConditionalLinkFailurePoseBlend,
    ConditionalLinkFailureRootBlend,
    SequenceBoneCalibration,
    SharedBackboneExecution,
)
from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset, _summary
from .calibrate_v11_residual_temporal import ResidualTemporalCalibration
from .evaluate_v12_final import _load_hybrid, _read_locked, build_locked_model
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--pose-calibration", type=Path, required=True)
    parser.add_argument("--failure-root-calibration", type=Path, required=True)
    parser.add_argument("--failure-class-calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--minimum-link-coverage", type=float, default=0.0)
    parser.add_argument(
        "--classification-minimum-link-coverage", type=float, default=0.0,
        help=(
            "classification fallback threshold; keep at zero when the "
            "specialist was trained for complete link loss"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    pose_lock = json.loads(args.pose_calibration.read_text(encoding="utf-8"))
    failure_class_lock = json.loads(
        args.failure_class_calibration.read_text(encoding="utf-8")
    )
    failure_root_lock = json.loads(
        args.failure_root_calibration.read_text(encoding="utf-8")
    )
    if any(
        lock.get("protocol") != args.exp
        or lock.get("test_used_for_selection") is not False
        for lock in (pose_lock, failure_root_lock, failure_class_lock)
    ):
        raise RuntimeError("link-failure calibration protocol/sealing mismatch")
    if pose_lock.get("selection_split") != "validation_drop_one_link":
        raise RuntimeError("pose expert was not selected on link-failure validation")
    if failure_class_lock.get("selection_split") not in {
        "validation_drop_one_link", "validation_drop_each_link"
    }:
        raise RuntimeError("classifier was not selected on link-failure validation")
    if failure_root_lock.get("selection_split") != "validation_drop_one_link":
        raise RuntimeError("root expert was not selected on link-failure validation")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    primary_execution, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    if not isinstance(primary_execution, SharedBackboneExecution):
        raise RuntimeError("V12G requires the verified shared V12 backbone")
    shared_backbone = primary_execution.backbone
    primary = primary_execution.model
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)

    pose_path = Path(pose_lock["expert_checkpoint"])
    pose_expert, pose_checkpoint = _load_hybrid(
        p2, pose_path, args.exp, device, 1.0, 0.0, shared_backbone
    )
    if pose_checkpoint.get("objective") != "pose_only":
        raise RuntimeError("failure pose checkpoint is not pose-only")
    temporal = ResidualTemporalCalibration(pose_expert).to(device)
    temporal.set_calibration(31, 1.0, "probability", 0.0)
    pose_expert = SequenceBoneCalibration(
        temporal, blend=0.25, symmetric=True
    ).to(device)
    pose_guard = ConditionalLinkFailurePoseBlend(
        primary, pose_expert,
        strength=float(pose_lock["selected"]["strength"]),
        minimum_link_coverage=args.minimum_link_coverage,
    ).to(device)

    root_path = Path(failure_root_lock["expert_checkpoint"])
    root_expert, root_checkpoint = _load_hybrid(
        p2, root_path, args.exp, device, 0.0, 1.0, shared_backbone
    )
    if root_checkpoint.get("objective") != "root_only":
        raise RuntimeError("failure root checkpoint is not root-only")
    root_guard = ConditionalLinkFailureRootBlend(
        pose_guard,
        root_expert,
        strength=float(failure_root_lock["selected"]["strength"]),
        minimum_link_coverage=args.minimum_link_coverage,
    ).to(device)

    class_path = Path(failure_class_lock["expert_checkpoint"])
    class_expert, class_checkpoint = _load_hybrid(
        p2, class_path, args.exp, device, 0.0, 0.0, shared_backbone
    )
    if class_checkpoint.get("objective") != "classification_only":
        raise RuntimeError("failure classifier is not classification-only")
    class_expert.set_calibration(0.0, 0.0, 1.0, 1.0)
    if "selected" in failure_class_lock:
        selected = failure_class_lock["selected"]
        class_strength = list(selected["class_strengths"])
        risk_strength = list(selected["risk_strengths"])
        danger_bias = list(selected["danger_biases"])
    else:
        selected_class = failure_class_lock["selected_class"]
        selected_risk = failure_class_lock["selected_risk"]
        class_strength = float(selected_class["strength"])
        risk_strength = float(selected_risk["strength"])
        danger_bias = float(selected_risk["danger_bias"])
    guarded = ConditionalLinkFailureLogitBlend(
        root_guard,
        class_expert,
        class_strength=class_strength,
        risk_strength=risk_strength,
        danger_logit_bias=danger_bias,
        minimum_link_coverage=args.classification_minimum_link_coverage,
    ).to(device)
    model = SharedBackboneExecution(
        guarded, shared_backbone
    ).to(device).eval()

    modes = (
        "clean", "time_jitter_2", "drop_one_link",
        "drop_link_0", "drop_link_1", "drop_link_2", "drop_link_burst",
        "drop_link_burst_early", "drop_link_burst_late",
        "drop_link_burst_shifted",
        "subcarrier_band",
        "gain_phase", "gain_phase_trial",
    )
    results = {}
    for mode in modes:
        pose_loader = DataLoader(
            PerturbedDataset(loaders["val"].dataset, mode),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        class_loader = DataLoader(
            PerturbedDataset(loaders["val_class"].dataset, mode),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        trajectory = evaluate_trajectory(
            model, pose_loader, device, args.max_shift
        )
        classification = evaluate_classification(
            model, class_loader, device, 0.0
        )
        results[mode] = {
            "trajectory": _summary(trajectory),
            "class_accuracy": classification["class"]["accuracy"],
            "class_macro_f1": classification["class"]["macro_f1"],
            "risk_accuracy": classification["risk"]["accuracy"],
            "risk_macro_f1": classification["risk"]["macro_f1"],
            "danger_recall": classification["risk"]["danger_recall"],
            "danger_precision": classification["risk"]["danger_precision"],
            "safe_to_danger": classification["risk"]["safe_to_danger"],
        }
    report = {
        "run": "p2_v12_link_failure_guard_audit",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "base_configuration": configuration,
        "shared_guard_backbone": True,
        "minimum_link_coverage": float(args.minimum_link_coverage),
        "classification_minimum_link_coverage": float(
            args.classification_minimum_link_coverage
        ),
        "pose_calibration": str(args.pose_calibration),
        "root_calibration": str(args.failure_root_calibration),
        "classification_calibration": str(args.failure_class_calibration),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
