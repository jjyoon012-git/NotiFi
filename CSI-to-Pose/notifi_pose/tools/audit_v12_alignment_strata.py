"""Compare locked V12 validation metrics by CSI-to-GT time alignment source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..trainer import set_seed
from .audit_v11_input_robustness import _summary
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


def _loader_for_method(loader: DataLoader, method: str,
                       batch_size: int) -> DataLoader:
    index = loader.dataset.index.reset_index(drop=True)
    positions = index.index[index["time_method"].eq(method)].tolist()
    return DataLoader(
        Subset(loader.dataset, positions), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model, _ = build_locked_model(args, device, root_lock, class_lock)
    model.eval()

    results = {}
    for method in ("timestamps", "uniform_30fps"):
        pose_loader = _loader_for_method(
            loaders["val"], method, args.batch_size * 2
        )
        class_loader = _loader_for_method(
            loaders["val_class"], method, args.batch_size * 2
        )
        trajectory = evaluate_trajectory(
            model, pose_loader, device, args.max_shift
        )
        classification = evaluate_classification(model, class_loader, device)
        results[method] = {
            "pose_trials": len(pose_loader.dataset),
            "classification_trials": len(class_loader.dataset),
            "trajectory": _summary(trajectory),
            "danger_trials": trajectory["danger_trials"],
            "class_accuracy": classification["class"]["accuracy"],
            "risk_accuracy": classification["risk"]["accuracy"],
            "danger_recall": classification["risk"]["danger_recall"],
            "safe_to_danger": classification["risk"]["safe_to_danger"],
        }
    exact = results["timestamps"]["trajectory"]
    assumed = results["uniform_30fps"]["trajectory"]
    report = {
        "run": "p2_v12_alignment_source_audit",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "results": results,
        "uniform_minus_timestamps": {
            key: assumed[key] - exact[key]
            for key in (
                "mpjpe_m", "dynamic_mpjpe_m", "root_error_m",
                "danger_mpjpe_m", "danger_endpoint_mpjpe_m",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
