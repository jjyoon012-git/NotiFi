"""Validation-only consistency calibration between class and risk heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..trainer import set_seed
from .calibrate_v11_classification_expert import _blended, _build_expert
from .evaluate_sealed import make_model
from .train_seen_v4_trajectory import (
    classification_metrics,
    collect_classification_logits,
    make_loaders,
)


CLASS_RANGES = ((0, 9), (9, 12), (12, 17))


def _hierarchical(logits: dict, class_weight: float) -> dict:
    class_probability = torch.softmax(logits["class_logits"], dim=-1)
    class_risk = torch.stack([
        class_probability[:, start:stop].sum(-1)
        for start, stop in CLASS_RANGES
    ], dim=-1)
    risk_probability = torch.softmax(logits["risk_logits"], dim=-1)
    mixed = (
        (1.0 - class_weight) * risk_probability
        + class_weight * class_risk
    ).clamp_min(1e-8)
    return {
        **logits,
        "risk_logits": mixed.log(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--classification-expert-checkpoint", type=Path, required=True
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument(
        "--class-weights", type=float, nargs="+",
        default=(0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    primary_model = make_model(p2_checkpoint, device).eval()
    expert_model = _build_expert(args, device).eval()
    primary = collect_classification_logits(
        primary_model, loaders["val_class"], device
    )
    expert = collect_classification_logits(
        expert_model, loaders["val_class"], device
    )
    if not torch.equal(primary["class_target"], expert["class_target"]):
        raise RuntimeError("classification expert validation order mismatch")

    logits = _blended(primary, expert, class_strength=1.0, risk_strength=0.0)
    baseline = classification_metrics(logits, 1.1)
    minimum_recall = float(baseline["risk"]["danger_recall"])
    maximum_false_alarms = int(baseline["risk"]["safe_to_danger"])
    candidates = []
    for class_weight in args.class_weights:
        hierarchical = _hierarchical(logits, class_weight)
        for step in range(61):
            danger_bias = step * 0.05
            metrics = classification_metrics(hierarchical, danger_bias)
            risk = metrics["risk"]
            feasible = (
                risk["danger_recall"] >= minimum_recall
                and risk["safe_to_danger"] <= maximum_false_alarms
            )
            score = (
                risk["macro_f1"] + 0.35 * risk["danger_f1"]
                - 0.75 * risk["safe_to_danger_rate"]
            )
            candidates.append({
                "class_weight": class_weight,
                "danger_logit_bias": danger_bias,
                "feasible": feasible,
                "score": score,
                "validation": risk,
            })
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    selected = max(feasible, key=lambda candidate: candidate["score"])
    report = {
        "run": "p2_v12_hierarchical_risk_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "p2_checkpoint": str(args.p2_checkpoint),
            "classification_expert_checkpoint": str(
                args.classification_expert_checkpoint
            ),
            "class_expert_strength": 1.0,
        },
        "constraints": {
            "minimum_danger_recall": minimum_recall,
            "maximum_safe_to_danger": maximum_false_alarms,
        },
        "baseline": baseline,
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "baseline": baseline["risk"],
        "selected": selected,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
