"""Select a link-failure pose expert on synthetic validation corruption only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..hybrid_v10 import (
    ConditionalLinkFailurePoseBlend,
    SequenceBoneCalibration,
)
from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset, _summary
from .calibrate_v11_residual_temporal import ResidualTemporalCalibration
from .evaluate_v12_final import _load_hybrid, _read_locked, build_locked_model
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def _score(metrics: dict) -> float:
    speed = max(float(metrics["pose_speed_ratio"]), 1e-6)
    return (
        float(metrics["mpjpe_m"])
        + 0.35 * float(metrics["danger_mpjpe_m"])
        + 0.10 * float(metrics["danger_endpoint_mpjpe_m"])
        + 0.10 * abs(math.log(speed))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    primary, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)
    expert, checkpoint = _load_hybrid(
        p2, args.expert_checkpoint, args.exp, device, 1.0, 0.0
    )
    if checkpoint.get("objective") != "pose_only":
        raise RuntimeError("link-failure expert must be pose-only")
    temporal = ResidualTemporalCalibration(expert).to(device)
    temporal.set_calibration(31, 1.0, "probability", 0.0)
    expert = SequenceBoneCalibration(
        temporal, blend=0.25, symmetric=True
    ).to(device)
    model = ConditionalLinkFailurePoseBlend(primary, expert).to(device).eval()

    clean_loader = loaders["val"]
    failure_loader = DataLoader(
        PerturbedDataset(clean_loader.dataset, "drop_one_link"),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    candidates = []
    baseline_clean = None
    for strength in args.strengths:
        model.set_strength(float(strength))
        clean = evaluate_trajectory(
            model, clean_loader, device, args.max_shift
        )
        failure = evaluate_trajectory(
            model, failure_loader, device, args.max_shift
        )
        if baseline_clean is None:
            baseline_clean = clean
        feasible = (
            clean["mpjpe_m"] <= baseline_clean["mpjpe_m"] + 0.001
            and clean["danger_mpjpe_m"]
            <= baseline_clean["danger_mpjpe_m"] + 0.002
        )
        candidates.append({
            "strength": float(strength),
            "feasible": bool(feasible),
            "failure_score": _score(failure),
            "clean": _summary(clean),
            "drop_one_link": _summary(failure),
        })
        print(
            f"strength={strength:.2f} clean={clean['mpjpe_m'] * 100:.2f}cm "
            f"drop={failure['mpjpe_m'] * 100:.2f}cm "
            f"danger={failure['danger_mpjpe_m'] * 100:.2f}cm "
            f"speed={failure['pose_speed_ratio']:.3f} feasible={feasible}"
        )

    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = min(feasible or candidates, key=lambda item: item["failure_score"])
    report = {
        "run": "p2_v12_link_failure_pose_calibration",
        "protocol": args.exp,
        "selection_split": "validation_drop_one_link",
        "test_used_for_selection": False,
        "source_configuration": configuration,
        "expert_checkpoint": str(args.expert_checkpoint),
        "expert_training_config": checkpoint.get("training_config", {}),
        "selection_rule": {
            "clean_mpjpe_tolerance_m": 0.001,
            "clean_danger_tolerance_m": 0.002,
        },
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
