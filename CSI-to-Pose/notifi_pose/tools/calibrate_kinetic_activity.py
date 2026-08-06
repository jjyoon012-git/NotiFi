"""Validation-lock the KP1 activity threshold and residual strength."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..kinetic_pose import KineticPoseResidual
from ..quality import QualityWeightedDataset
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .train_kinetic_pose import (
    CoarsePoseStore,
    evaluate_strengths,
    pose_selection_score,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_exp03_velocity_aux" / "best_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_exp03_activity_calibrated",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_path = Path(checkpoint["source"]["p2_checkpoint"])
    if not source_path.is_absolute():
        source_path = C.PROJECT_ROOT / source_path
    p2_checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    p2_model = make_model(p2_checkpoint, device)
    architecture = checkpoint["architecture"]
    model = KineticPoseResidual(
        None, p2_model.norm,
        hidden=int(architecture["hidden"]),
        temporal_layers=int(architecture["temporal_layers"]),
        max_delta=float(architecture["max_delta_m"]),
        condition_on_coarse=bool(architecture.get("condition_on_coarse", True)),
        activity_floor=float(architecture.get("activity_floor", 0.15)),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    del p2_model

    cached = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    store = CoarsePoseStore(cached["rows"], cached["pose"])
    datasets = build_datasets(exp=args.exp, baseline="sub")
    validation = QualityWeightedDataset(pose_only(datasets["val"]))
    test = QualityWeightedDataset(pose_only(datasets["test"]))
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    candidates = []
    for threshold in args.thresholds:
        model.set_activity_threshold(threshold)
        metrics_by_strength = evaluate_strengths(
            model, validation_loader, list(args.strengths), device, store
        )
        for strength, metrics in metrics_by_strength.items():
            candidates.append({
                "activity_threshold": float(threshold),
                "residual_strength": float(strength),
                "score": pose_selection_score(metrics),
                "metrics": metrics,
            })
    selected = min(candidates, key=lambda item: item["score"])
    model.set_activity_threshold(selected["activity_threshold"])
    test_metrics = evaluate_strengths(
        model, test_loader,
        [0.0, selected["residual_strength"]], device, store,
    )
    result = {
        "run": f"{checkpoint['run']}-activity-calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_checkpoint": report_path(args.checkpoint),
        "selected": selected,
        "test": {
            "v13s_strength_0": test_metrics[0.0],
            "kp1_selected": test_metrics[selected["residual_strength"]],
        },
        "candidates": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calibrated = dict(checkpoint)
    calibrated.update({
        "activity_threshold": selected["activity_threshold"],
        "residual_strength": selected["residual_strength"],
        "activity_calibration": {
            "selection_split": "validation",
            "test_used_for_selection": False,
            "score": selected["score"],
            "source": report_path(args.checkpoint),
        },
        "validation": selected["metrics"],
        "test": result["test"],
    })
    torch.save(calibrated, args.output_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
