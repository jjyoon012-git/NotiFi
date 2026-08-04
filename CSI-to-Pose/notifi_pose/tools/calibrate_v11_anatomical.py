"""Validation-only calibration of symmetric torso, arm, and leg residuals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .. import contract as C
from ..hybrid_v10 import AnatomicalResidualCalibration, SequenceBoneCalibration
from ..trainer import set_seed
from .calibrate_v11_temporal import _load_model
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def selection_score(metrics: dict) -> float:
    speed = max(float(metrics["pose_speed_ratio"]), 1e-3)
    return (
        float(metrics["mpjpe_m"])
        + 0.20 * float(metrics["dynamic_mpjpe_m"])
        + 0.15 * float(metrics["danger_mpjpe_m"])
        + 0.10 * float(metrics["danger_distal_mpjpe_m"])
        + 0.05 * float(metrics["danger_endpoint_mpjpe_m"])
        + 0.05 * abs(math.log(speed))
    )


def feasible(metrics: dict, baseline: dict) -> bool:
    return (
        metrics["mpjpe_m"] <= baseline["mpjpe_m"] * 1.002
        and metrics["dynamic_mpjpe_m"] <= baseline["dynamic_mpjpe_m"] * 1.005
        and metrics["danger_mpjpe_m"] <= baseline["danger_mpjpe_m"] * 1.005
        and metrics["danger_endpoint_mpjpe_m"]
        <= baseline["danger_endpoint_mpjpe_m"] * 1.005
        and 0.90 <= metrics["pose_speed_ratio"] <= 1.10
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-calibration", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--root-expert-kind", choices=("v9", "p2_hybrid"),
        default="p2_hybrid",
    )
    parser.add_argument("--allow-unverified-root-protocol", action="store_true")
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument("--v2-checkpoint", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--motion-checkpoint", type=Path)
    parser.add_argument("--pose-residual-checkpoint", type=Path)
    parser.add_argument("--root-residual-checkpoint", type=Path)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--root-expert-strength", type=float, default=1.0)
    parser.add_argument("--pose-strength-override", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.20, 0.35, 0.50, 0.75),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    base, source = _load_model(args, device)
    model = AnatomicalResidualCalibration(base, 0.35, 0.35, 0.35).to(device)
    baseline = evaluate_trajectory(model, loaders["val"], device, args.max_shift)

    names = ("torso", "arms", "legs")
    selected = [0.35, 0.35, 0.35]
    candidates = []
    for round_index in range(args.rounds):
        changed = False
        for group_index, name in enumerate(names):
            group_candidates = []
            for strength in args.strengths:
                values = list(selected)
                values[group_index] = float(strength)
                model.set_calibration(*values)
                metrics = evaluate_trajectory(
                    model, loaders["val"], device, args.max_shift
                )
                item = {
                    "round": round_index + 1,
                    "group": name,
                    "strengths": dict(zip(names, values)),
                    "feasible": feasible(metrics, baseline),
                    "score": selection_score(metrics),
                    "validation": metrics,
                }
                candidates.append(item)
                group_candidates.append(item)
            accepted = [item for item in group_candidates if item["feasible"]]
            choice = min(accepted or group_candidates, key=lambda item: item["score"])
            values = [choice["strengths"][key] for key in names]
            changed = changed or values[group_index] != selected[group_index]
            selected = values
        if not changed:
            break

    model.set_calibration(*selected)
    anatomical_validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    bone_model = SequenceBoneCalibration(model).to(device)
    bone_candidates = []
    for symmetric in (False, True):
        for blend in (0.0, 0.25, 0.50, 0.75, 1.0):
            bone_model.set_calibration(blend, symmetric)
            metrics = evaluate_trajectory(
                bone_model, loaders["val"], device, args.max_shift
            )
            bone_candidates.append({
                "blend": blend,
                "symmetric": symmetric,
                "feasible": feasible(metrics, anatomical_validation),
                "score": selection_score(metrics),
                "validation": metrics,
            })
    accepted = [item for item in bone_candidates if item["feasible"]]
    bone = min(accepted or bone_candidates, key=lambda item: item["score"])

    result = {
        "run": "p2_v11_anatomical_residual_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_calibration": source,
        "selected": {
            **dict(zip(names, selected)),
            "bone_blend": bone["blend"],
            "bone_symmetric": bone["symmetric"],
        },
        "baseline_validation": baseline,
        "anatomical_validation": anatomical_validation,
        "selected_validation": bone["validation"],
        "anatomical_candidates": candidates,
        "bone_candidates": bone_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
