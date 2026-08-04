"""Recalibrate a trained P2+V9 hybrid without retraining or test selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..hybrid_v10 import P2V9HybridNet
from ..trainer import set_seed
from .evaluate_sealed import make_model
from .train_p2_v9_hybrid import select_calibration
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--exp", default="single_split_lmh_e01",
        choices=("single_split", "single_split_lmh_e01"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--minimum-danger-recall", type=float, default=0.97)
    parser.add_argument("--maximum-danger-bias", type=float, default=4.0)
    parser.add_argument("--minimum-root-gain", type=float, default=0.005)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pose-strengths", type=float, nargs="+",
        default=(0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--root-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--logit-strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    model = P2V9HybridNet(make_model(p2_checkpoint, device)).to(device)
    hybrid_checkpoint = torch.load(
        args.hybrid_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(hybrid_checkpoint["model"])

    model.set_calibration(0.0, 0.0, 0.0, 0.0)
    baseline_test = evaluate_trajectory(
        model, loaders["test"], device, args.max_shift
    )
    baseline_test_classification = evaluate_classification(
        model, loaders["test_class"], device
    )

    calibration = select_calibration(model, loaders, device, args)
    selected = {
        "pose_strength": calibration["pose"]["strength"],
        "root_strength": calibration["root"]["strength"],
        "class_strength": calibration["classification"]["strength"],
        "risk_strength": calibration["risk"]["strength"],
        "danger_logit_bias": calibration["risk"]["danger_logit_bias"],
    }
    model.set_calibration(
        selected["pose_strength"], selected["root_strength"],
        selected["class_strength"], selected["risk_strength"],
    )
    test = evaluate_trajectory(model, loaders["test"], device, args.max_shift)
    raw_test_classification = evaluate_classification(
        model, loaders["test_class"], device
    )
    test_classification = evaluate_classification(
        model, loaders["test_class"], device, selected["danger_logit_bias"]
    )
    result = {
        "run": "p2_v9_hybrid_v10_recalibrated",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selected": selected,
        "baseline_test": baseline_test,
        "baseline_test_classification": baseline_test_classification,
        "selected_validation": calibration["root"]["validation"],
        "validation_classification": {
            "class": calibration["classification"]["validation"],
            "risk": calibration["risk"]["validation"],
        },
        "test": test,
        "raw_test_classification": raw_test_classification,
        "test_classification": test_classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
