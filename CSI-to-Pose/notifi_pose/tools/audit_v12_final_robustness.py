"""Validation-only perturbation audit for the locked V12 final model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset, _summary
from .evaluate_v12_final import _read_locked, build_locked_model
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
    model, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    model.eval()

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
            batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
        )
        class_loader = DataLoader(
            PerturbedDataset(loaders["val_class"].dataset, mode),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
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
            "safe_to_danger": classification["risk"]["safe_to_danger"],
        }
    clean = results["clean"]["trajectory"]
    for mode in modes[1:]:
        metrics = results[mode]["trajectory"]
        results[mode]["delta"] = {
            "mpjpe_m": metrics["mpjpe_m"] - clean["mpjpe_m"],
            "danger_mpjpe_m": (
                metrics["danger_mpjpe_m"] - clean["danger_mpjpe_m"]
            ),
        }
    report = {
        "run": "p2_v12_final_input_robustness_audit",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "configuration": configuration,
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
