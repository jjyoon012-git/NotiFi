"""Select a clean-constrained classification ensemble on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset
from .calibrate_v11_classification_expert import _build_expert
from .calibrate_v12_hierarchical_risk import _hierarchical
from .evaluate_sealed import make_model
from .train_seen_v4_trajectory import (
    classification_metrics,
    collect_classification_logits,
    make_loaders,
)


def _expert(args, checkpoint: Path, device: str):
    values = vars(args).copy()
    values["classification_expert_checkpoint"] = checkpoint
    return _build_expert(SimpleNamespace(**values), device).eval()


def _mix(first: dict, second: dict, second_weight: float) -> dict:
    return {
        "class_logits": (1.0 - second_weight) * first["class_logits"]
        + second_weight * second["class_logits"],
        "risk_logits": (1.0 - second_weight) * first["risk_logits"]
        + second_weight * second["risk_logits"],
        "class_target": first["class_target"],
        "risk_target": first["risk_target"],
    }


def _collect(model, dataset, batch_size: int, device: str) -> dict:
    loader = DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )
    return collect_classification_logits(model, loader, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--robust-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument(
        "--robust-weights", type=float, nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )
    parser.add_argument(
        "--class-weights", type=float, nargs="+",
        default=(0.3, 0.4, 0.5, 0.6),
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
    clean_model = _expert(args, args.clean_expert_checkpoint, device)
    robust_model = _expert(args, args.robust_expert_checkpoint, device)

    datasets = {
        "clean": loaders["val_class"].dataset,
        "drop_one_link": PerturbedDataset(
            loaders["val_class"].dataset, "drop_one_link"
        ),
    }
    logits = {}
    for mode, dataset in datasets.items():
        logits[mode] = {
            "primary": _collect(
                primary_model, dataset, args.batch_size, device
            ),
            "clean_expert": _collect(
                clean_model, dataset, args.batch_size, device
            ),
            "robust_expert": _collect(
                robust_model, dataset, args.batch_size, device
            ),
        }
        targets = logits[mode]
        if not torch.equal(
            targets["primary"]["class_target"],
            targets["clean_expert"]["class_target"],
        ) or not torch.equal(
            targets["primary"]["class_target"],
            targets["robust_expert"]["class_target"],
        ):
            raise RuntimeError(f"classification order mismatch for {mode}")

    def evaluate(mode: str, robust_weight: float, class_weight: float,
                 danger_bias: float) -> dict:
        values = logits[mode]
        expert = _mix(
            values["clean_expert"], values["robust_expert"], robust_weight
        )
        combined = {
            **expert,
            "risk_logits": values["primary"]["risk_logits"],
        }
        return classification_metrics(
            _hierarchical(combined, class_weight), danger_bias
        )

    baseline = evaluate("clean", 0.0, 0.5, 0.5)
    baseline_drop = evaluate("drop_one_link", 0.0, 0.5, 0.5)
    minimum_class_f1 = float(baseline["class"]["macro_f1"] - 1e-8)
    minimum_risk_f1 = float(baseline["risk"]["macro_f1"] - 1e-8)
    minimum_recall = float(baseline["risk"]["danger_recall"])
    maximum_false_alarms = int(baseline["risk"]["safe_to_danger"])
    candidates = []
    for robust_weight in args.robust_weights:
        for class_weight in args.class_weights:
            for step in range(21):
                danger_bias = step * 0.05
                clean = evaluate(
                    "clean", robust_weight, class_weight, danger_bias
                )
                drop = evaluate(
                    "drop_one_link", robust_weight, class_weight, danger_bias
                )
                feasible = (
                    clean["class"]["macro_f1"] >= minimum_class_f1
                    and clean["risk"]["macro_f1"] >= minimum_risk_f1
                    and clean["risk"]["danger_recall"] >= minimum_recall
                    and clean["risk"]["safe_to_danger"] <= maximum_false_alarms
                )
                score = (
                    drop["class"]["macro_f1"]
                    + drop["risk"]["macro_f1"]
                    + 0.5 * drop["risk"]["danger_recall"]
                    - 0.5 * drop["risk"]["safe_to_danger_rate"]
                )
                candidates.append({
                    "robust_expert_weight": robust_weight,
                    "class_weight": class_weight,
                    "danger_logit_bias": danger_bias,
                    "feasible": feasible,
                    "score": score,
                    "clean": clean,
                    "drop_one_link": drop,
                })
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        raise RuntimeError("no robust ensemble met the clean non-regression guard")
    selected = max(feasible, key=lambda candidate: candidate["score"])
    robust_weight = float(selected["robust_expert_weight"])
    report = {
        "run": "p2_v12_robust_classification_ensemble",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source": {
            "p2_checkpoint": str(args.p2_checkpoint),
            "classification_expert_checkpoints": [
                str(args.clean_expert_checkpoint),
                str(args.robust_expert_checkpoint),
            ],
            "classification_expert_weights": [
                1.0 - robust_weight, robust_weight
            ],
            "class_expert_strength": 1.0,
        },
        "constraints": {
            "minimum_clean_class_macro_f1": minimum_class_f1,
            "minimum_clean_risk_macro_f1": minimum_risk_f1,
            "minimum_clean_danger_recall": minimum_recall,
            "maximum_clean_safe_to_danger": maximum_false_alarms,
        },
        "baseline": {"clean": baseline, "drop_one_link": baseline_drop},
        "selected": {
            "class_weight": selected["class_weight"],
            "danger_logit_bias": selected["danger_logit_bias"],
            "robust_expert_weight": robust_weight,
            "validation": selected["clean"]["risk"],
            "clean": selected["clean"],
            "drop_one_link": selected["drop_one_link"],
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "baseline": report["baseline"],
        "selected": report["selected"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
