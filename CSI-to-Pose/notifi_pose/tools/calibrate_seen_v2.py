"""Calibrate rotation, high-pose, and root strengths on validation only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from ..seen_v2 import SeenReconstructionV2Net
from ..trainer import evaluate
from .diagnose_observability import evaluate_model, pose_only, report_path
from .train_seen_v2 import load_motion, make_final_seen_backbone


def make_model(args, device: str) -> SeenReconstructionV2Net:
    backbone = make_final_seen_backbone(
        args.baseline_checkpoint, args.motion_checkpoint,
        args.pose_residual_checkpoint, args.root_residual_checkpoint,
        0.5, device,
    )
    motion = load_motion(args.motion_checkpoint, device)
    model = SeenReconstructionV2Net(
        backbone, motion, hidden=motion.hidden
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.set_partial_finetune(False)
    return model


def measure(model, dataset, loader, criterion, device: str,
            batch_size: int) -> dict:
    return {
        **evaluate(model, loader, criterion, device),
        **evaluate_model(model, dataset, device, batch_size, 5),
    }


def pose_score(metrics: dict) -> float:
    ratio = max(float(metrics["pose_speed_ratio"]), 1e-3)
    return (
        float(metrics["mpjpe_m"])
        + 0.10 * float(metrics["dynamic_mpjpe_m"])
        + 0.10 * float(metrics["distal_mpjpe"])
        + 0.10 * float(metrics["impact_mpjpe"])
        + 0.20 * abs(math.log(ratio))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "best_model.pt",
    )
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
    )
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--pose-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--root-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--rotation-strengths", type=float, nargs="+",
        default=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--high-pose-strengths", type=float, nargs="+",
        default=(0.0, 0.5, 1.0),
    )
    parser.add_argument(
        "--root-strengths", type=float, nargs="+", default=(0.0, 0.5, 1.0)
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibration.json",
    )
    parser.add_argument(
        "--output-model", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibrated_model.pt",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp="single_split", baseline="sub")
    validation = QualityWeightedDataset(pose_only(datasets["val"]))
    test = QualityWeightedDataset(pose_only(datasets["test"]))
    loaders = {
        "val": DataLoader(validation, batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=args.batch_size, shuffle=False),
    }
    model = make_model(args, device)
    criterion = L.PoseLoss(
        lambda_root=1.0, lambda_bone=0.1, lambda_cls=0.0, lambda_risk=0.0,
        lambda_velocity=0.1, lambda_impact=0.2,
        lambda_displacement=0.1, motion_weight=3.0, device=device,
    ).to(device)

    pose_candidates = []
    for rotation in args.rotation_strengths:
        for high_pose in args.high_pose_strengths:
            model.set_calibration(rotation, high_pose, 0.0)
            metrics = measure(
                model, validation, loaders["val"], criterion,
                device, args.batch_size,
            )
            score = pose_score(metrics)
            feasible = 0.80 <= metrics["pose_speed_ratio"] <= 1.20
            pose_candidates.append({
                "rotation_strength": rotation,
                "high_pose_strength": high_pose,
                "feasible_speed": feasible,
                "score": score,
                "validation": metrics,
            })
            print(
                f"rotation={rotation:.2f} high={high_pose:.2f} "
                f"mpjpe={metrics['mpjpe_m'] * 100:.2f}cm "
                f"speed={metrics['pose_speed_ratio']:.3f} feasible={feasible}"
            )
    feasible = [item for item in pose_candidates if item["feasible_speed"]]
    selected_pose = min(feasible or pose_candidates, key=lambda item: item["score"])

    root_candidates = []
    for root in args.root_strengths:
        model.set_calibration(
            selected_pose["rotation_strength"],
            selected_pose["high_pose_strength"], root,
        )
        metrics = measure(
            model, validation, loaders["val"], criterion,
            device, args.batch_size,
        )
        score = float(metrics["root_err"]) + 0.25 * float(metrics["impact_mpjpe"])
        root_candidates.append({
            "root_strength": root, "score": score, "validation": metrics,
        })
        print(
            f"root={root:.2f} root_err={metrics['root_err'] * 100:.2f}cm "
            f"impact={metrics['impact_mpjpe'] * 100:.2f}cm"
        )
    selected_root = min(root_candidates, key=lambda item: item["score"])
    calibration = {
        "rotation_strength": selected_pose["rotation_strength"],
        "high_pose_strength": selected_pose["high_pose_strength"],
        "root_strength": selected_root["root_strength"],
    }
    model.set_calibration(**{
        "rotation": calibration["rotation_strength"],
        "high_pose": calibration["high_pose_strength"],
        "root": calibration["root_strength"],
    })
    test_metrics = measure(
        model, test, loaders["test"], criterion, device, args.batch_size
    )
    result = {
        "run": "seen_reconstruction_v2_validation_calibration",
        "protocol": "single_split",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "speed_gate": [0.8, 1.2],
        "selected": calibration,
        "selected_pose_validation": selected_pose["validation"],
        "selected_root_validation": selected_root["validation"],
        "test": test_metrics,
        "pose_candidates": pose_candidates,
        "root_candidates": root_candidates,
        "checkpoint": report_path(args.checkpoint),
        "calibrated_checkpoint": report_path(args.output_model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "calibration": calibration,
        "source_checkpoint": report_path(args.checkpoint),
        "validation": selected_root["validation"],
        "test": test_metrics,
    }, args.output_model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
