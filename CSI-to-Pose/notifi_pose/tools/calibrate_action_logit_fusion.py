"""Validation-only calibration of the two CSI action heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .diagnose_observability import report_path


def calibrated_action_logits(data, config):
    return (
        config["base_weight"] * data["base_action_logits"]
        + config["selector_weight"] * data["selector_action_logits"]
    ) / config["temperature"]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_action_logit_fusion"
        / "calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "val", device)
    results = {}
    for base_weight in (0.50, 0.75, 1.00, 1.25, 1.50):
        for selector_weight in (0.50, 0.75, 1.00, 1.25, 1.50):
            for temperature in (0.75, 1.00, 1.25, 1.50):
                config = {
                    "base_weight": base_weight,
                    "selector_weight": selector_weight,
                    "temperature": temperature,
                }
                logits = calibrated_action_logits(data, config)
                accuracy = float(
                    (logits.argmax(-1) == data["target_class"]).float().mean()
                )
                nll = float(F.cross_entropy(logits, data["target_class"]))
                name = (
                    f"b{int(base_weight * 100):03d}"
                    f"_s{int(selector_weight * 100):03d}"
                    f"_t{int(temperature * 100):03d}"
                )
                results[name] = {
                    **config, "accuracy": accuracy, "nll": nll,
                    "selection_score": (1.0 - accuracy) + 0.05 * nll,
                }
    best = min(results, key=lambda name: results[name]["selection_score"])
    result = {
        "status": "validation_selected_action_logit_fusion",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best, **results[best]},
        "results": results,
        "selector_checkpoint": report_path(args.selector_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
