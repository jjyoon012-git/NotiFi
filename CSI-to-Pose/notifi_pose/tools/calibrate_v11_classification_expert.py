"""Blend an independently trained classification expert on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..trainer import set_seed
from .calibrate_v11_residual_temporal import _build_model, _checked_checkpoint
from .evaluate_sealed import make_model
from .train_p2_v9_hybrid import build_residual_hybrid
from .train_seen_v4_trajectory import (
    classification_metrics,
    collect_classification_logits,
    make_loaders,
)


def _configure_primary(args, locked: dict, device: str):
    source = locked["source"]
    args.pose_strength = float(source["pose_strength"])
    args.root_strength = float(source["root_strength"])
    args.bone_blend = float(source["bone_blend"])
    args.bone_symmetric = bool(source["bone_symmetric"])
    model = _build_model(args, device)
    model.base.set_calibration(
        int(locked["selected"]["window"]),
        float(locked["selected"]["blend"]),
        source.get("risk_adaptive", "none"),
        float(source.get("danger_logit_bias", 0.0)),
    )
    model.base.set_root_calibration(
        int(locked["selected"].get("root_window", 1)),
        float(locked["selected"].get("root_blend", 0.0)),
    )
    return model


def _build_expert(args, device: str):
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False,
    )
    checkpoint = _checked_checkpoint(
        args.classification_expert_checkpoint, device, args.exp,
    )
    if checkpoint.get("objective") != "classification_only":
        raise RuntimeError("classification expert has the wrong objective")
    model = build_residual_hybrid(
        make_model(p2_checkpoint, device),
        checkpoint.get("residual_decoder", "subcarrier"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.set_calibration(0.0, 0.0, 1.0, 1.0)
    return model


def _blended(primary: dict, expert: dict, class_strength: float,
             risk_strength: float) -> dict:
    return {
        "class_logits": primary["class_logits"] + class_strength * (
            expert["class_logits"] - primary["class_logits"]
        ),
        "risk_logits": primary["risk_logits"] + risk_strength * (
            expert["risk_logits"] - primary["risk_logits"]
        ),
        "class_target": primary["class_target"],
        "risk_target": primary["risk_target"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--classification-expert-checkpoint", type=Path, required=True,
    )
    parser.add_argument("--calibration", type=Path, required=True)
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

    locked = json.loads(args.calibration.read_text(encoding="utf-8"))
    if locked.get("protocol") != args.exp:
        raise RuntimeError("calibration protocol mismatch")
    if locked.get("selection_split") != "validation":
        raise RuntimeError("classification calibration must use validation")
    if locked.get("test_used_for_selection") is not False:
        raise RuntimeError("test split was not proven sealed")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    primary_model = _configure_primary(args, locked, device).eval()
    expert_model = _build_expert(args, device).eval()
    primary = collect_classification_logits(
        primary_model, loaders["val_class"], device,
    )
    expert = collect_classification_logits(
        expert_model, loaders["val_class"], device,
    )
    if not torch.equal(primary["class_target"], expert["class_target"]):
        raise RuntimeError("classification expert validation order mismatch")

    base_bias = float(locked["source"].get("danger_logit_bias", 0.0))
    baseline = classification_metrics(primary, base_bias)
    class_candidates = []
    for strength in args.strengths:
        metrics = classification_metrics(
            _blended(primary, expert, strength, 0.0), base_bias,
        )
        class_candidates.append({
            "strength": strength,
            "macro_f1": metrics["class"]["macro_f1"],
            "accuracy": metrics["class"]["accuracy"],
            "validation": metrics["class"],
        })
    selected_class = max(
        class_candidates, key=lambda item: (item["macro_f1"], item["accuracy"]),
    )

    minimum_recall = float(baseline["risk"]["danger_recall"])
    maximum_false_alarms = int(baseline["risk"]["safe_to_danger"])
    risk_candidates = []
    for strength in args.strengths:
        mixed = _blended(primary, expert, selected_class["strength"], strength)
        for step in range(41):
            bias = step * 0.05
            metrics = classification_metrics(mixed, bias)
            risk = metrics["risk"]
            feasible = (
                risk["danger_recall"] >= minimum_recall
                and risk["safe_to_danger"] <= maximum_false_alarms
            )
            score = (
                risk["macro_f1"] + 0.25 * risk["danger_f1"]
                - 0.50 * risk["safe_to_danger_rate"]
            )
            risk_candidates.append({
                "strength": strength,
                "danger_logit_bias": bias,
                "feasible": feasible,
                "score": score,
                "validation": risk,
            })
    feasible = [item for item in risk_candidates if item["feasible"]]
    selected_risk = max(feasible, key=lambda item: item["score"])
    selected_logits = _blended(
        primary, expert, selected_class["strength"], selected_risk["strength"],
    )
    selected_metrics = classification_metrics(
        selected_logits, selected_risk["danger_logit_bias"],
    )
    report = {
        "run": "p2_v11_classification_expert_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "p2_checkpoint": str(args.p2_checkpoint),
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "root_expert_checkpoint": str(args.root_expert_checkpoint),
            "classification_expert_checkpoint": str(
                args.classification_expert_checkpoint
            ),
            "pose_calibration": str(args.calibration),
        },
        "constraints": {
            "minimum_danger_recall": minimum_recall,
            "maximum_safe_to_danger": maximum_false_alarms,
        },
        "baseline": baseline,
        "selected": {
            "class_strength": selected_class["strength"],
            "risk_strength": selected_risk["strength"],
            "danger_logit_bias": selected_risk["danger_logit_bias"],
            "validation": selected_metrics,
        },
        "class_candidates": class_candidates,
        "risk_candidates": risk_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "baseline": baseline,
        "selected": report["selected"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
