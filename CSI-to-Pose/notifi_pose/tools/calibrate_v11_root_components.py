"""Validation-only calibration of clean root anchor and step components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..hybrid_v10 import RootComponentBlend, RootExpertBlend, SequenceBoneCalibration
from ..trainer import set_seed
from .calibrate_v11_temporal import _load_model
from .train_p2_v9_hybrid import root_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-calibration", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--root-expert-kind", choices=("p2_hybrid",), default="p2_hybrid"
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
    parser.add_argument("--pose-strength-override", type=float, default=0.35)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    loaded, source = _load_model(args, device)
    if not isinstance(loaded, RootExpertBlend):
        raise TypeError("expected a primary/root-expert pair")
    components = RootComponentBlend(loaded.primary, loaded.root_expert).to(device)
    model = SequenceBoneCalibration(
        components, blend=args.bone_blend, symmetric=args.bone_symmetric
    ).to(device)

    candidates = []
    for anchor in args.strengths:
        for step in args.strengths:
            components.set_calibration(anchor, step)
            metrics = evaluate_trajectory(
                model, loaders["val"], device, args.max_shift
            )
            candidates.append({
                "anchor_strength": anchor,
                "step_strength": step,
                "score": root_selection_score(metrics),
                "validation": metrics,
            })
    baseline = next(
        item for item in candidates
        if item["anchor_strength"] == 0.0 and item["step_strength"] == 0.0
    )
    selected = min(candidates, key=lambda item: item["score"])
    if (
        baseline["validation"]["root_error_m"]
        - selected["validation"]["root_error_m"] < 0.001
    ):
        selected = baseline

    result = {
        "run": "p2_v11_clean_root_component_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_calibration": source,
        "selected": {
            "anchor_strength": selected["anchor_strength"],
            "step_strength": selected["step_strength"],
            "bone_blend": args.bone_blend,
            "bone_symmetric": args.bone_symmetric,
        },
        "baseline_validation": baseline["validation"],
        "selected_validation": selected["validation"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
