"""Calibrate missing-link classification separately for each physical link."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset
from .calibrate_v12_link_failure_classification import _collect_pair, _metrics
from .evaluate_v12_final import _load_hybrid, _read_locked, build_locked_model
from .train_seen_v4_trajectory import classification_metrics, make_loaders


def _mixed_metrics(pair: dict, class_strengths: list[float],
                   risk_strengths: list[float],
                   danger_biases: list[float]) -> dict:
    class_logits = pair["primary_class"].clone()
    risk_logits = pair["primary_risk"].clone()
    for link in range(C.N_LINKS):
        selected = pair["failed"] & pair["missing_link"].eq(link)
        if not selected.any():
            continue
        class_amount = float(class_strengths[link])
        if class_amount:
            probability = (
                (1.0 - class_amount)
                * torch.softmax(pair["primary_class"][selected], dim=-1)
                + class_amount
                * torch.softmax(pair["expert_class"][selected], dim=-1)
            )
            class_logits[selected] = probability.clamp_min(1e-8).log()
        risk_amount = float(risk_strengths[link])
        if risk_amount:
            probability = (
                (1.0 - risk_amount)
                * torch.softmax(pair["primary_risk"][selected], dim=-1)
                + risk_amount
                * torch.softmax(pair["expert_risk"][selected], dim=-1)
            )
            risk_logits[selected] = probability.clamp_min(1e-8).log()
        risk_logits[selected, C.N_RISK - 1] += float(danger_biases[link])
    return classification_metrics({
        "class_logits": class_logits,
        "risk_logits": risk_logits,
        "class_target": pair["class_target"],
        "risk_target": pair["risk_target"],
    })


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
    parser.add_argument(
        "--danger-biases", type=float, nargs="+",
        default=(-0.5, -0.25, 0.0, 0.25, 0.5),
    )
    parser.add_argument(
        "--selection-profile", choices=("balanced", "danger_recall"),
        default="balanced",
    )
    parser.add_argument("--max-accuracy-drop", type=float, default=0.03)
    parser.add_argument(
        "--max-safe-to-danger-increase", type=int, default=10
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
        p2, args.expert_checkpoint, args.exp, device, 0.0, 0.0
    )
    if checkpoint.get("objective") != "classification_only":
        raise RuntimeError("link-failure classifier must be classification-only")
    expert.set_calibration(0.0, 0.0, 1.0, 1.0)

    clean_pair = _collect_pair(primary, expert, loaders["val_class"], device)
    cyclic_loader = DataLoader(
        PerturbedDataset(loaders["val_class"].dataset, "drop_one_link"),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
    )
    cyclic_pair = _collect_pair(primary, expert, cyclic_loader, device)
    class_selected = []
    risk_selected = []
    per_link = {}
    for link in range(C.N_LINKS):
        loader = DataLoader(
            PerturbedDataset(
                loaders["val_class"].dataset, f"drop_link_{link}"
            ),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
        )
        pair = _collect_pair(primary, expert, loader, device)
        baseline = _metrics(pair, 0.0, 0.0)
        class_candidates = []
        for strength in args.strengths:
            metrics = _metrics(pair, float(strength), 0.0)
            class_metrics = metrics["class"]
            feasible = (
                class_metrics["accuracy"]
                >= baseline["class"]["accuracy"] - 0.005
            )
            class_candidates.append({
                "strength": float(strength),
                "feasible": bool(feasible),
                "score": float(class_metrics["macro_f1"]),
                "metrics": class_metrics,
            })
        selected_class = max(
            [item for item in class_candidates if item["feasible"]]
            or class_candidates,
            key=lambda item: item["score"],
        )
        class_selected.append(selected_class["strength"])

        risk_candidates = []
        for strength in args.strengths:
            for bias in args.danger_biases:
                metrics = _metrics(
                    pair, 0.0, float(strength), float(bias)
                )["risk"]
                accuracy_drop = (
                    args.max_accuracy_drop
                    if args.selection_profile == "danger_recall"
                    else 0.005
                )
                false_alarm_increase = (
                    args.max_safe_to_danger_increase
                    if args.selection_profile == "danger_recall"
                    else 2
                )
                feasible = (
                    metrics["accuracy"]
                    >= baseline["risk"]["accuracy"] - accuracy_drop
                    and metrics["danger_recall"]
                    >= baseline["risk"]["danger_recall"]
                    and metrics["safe_to_danger"]
                    <= baseline["risk"]["safe_to_danger"]
                    + false_alarm_increase
                )
                score = (
                    metrics["macro_f1"]
                    + 0.50 * metrics["danger_recall"]
                    - 0.50 * metrics["safe_to_danger_rate"]
                )
                risk_candidates.append({
                    "strength": float(strength),
                    "danger_bias": float(bias),
                    "feasible": bool(feasible),
                    "score": float(score),
                    "metrics": metrics,
                })
        feasible_risk = (
            [item for item in risk_candidates if item["feasible"]]
            or risk_candidates
        )
        if args.selection_profile == "danger_recall":
            selected_risk = max(
                feasible_risk,
                key=lambda item: (
                    item["metrics"]["danger_recall"],
                    item["metrics"]["macro_f1"],
                    item["metrics"]["accuracy"],
                    -item["metrics"]["safe_to_danger"],
                    -abs(item["danger_bias"]),
                ),
            )
        else:
            selected_risk = max(
                feasible_risk, key=lambda item: item["score"]
            )
        risk_selected.append((
            selected_risk["strength"], selected_risk["danger_bias"]
        ))
        per_link[str(link)] = {
            "baseline": baseline,
            "selected_class": selected_class,
            "selected_risk": selected_risk,
            "class_candidates": class_candidates,
            "risk_candidates": risk_candidates,
        }

    risk_strengths = [item[0] for item in risk_selected]
    danger_biases = [item[1] for item in risk_selected]
    clean = _mixed_metrics(
        clean_pair, class_selected, risk_strengths, danger_biases
    )
    cyclic = _mixed_metrics(
        cyclic_pair, class_selected, risk_strengths, danger_biases
    )
    report = {
        "run": "p2_v12_link_specific_classification_calibration",
        "protocol": args.exp,
        "selection_split": "validation_drop_each_link",
        "test_used_for_selection": False,
        "selection_profile": args.selection_profile,
        "max_accuracy_drop": float(args.max_accuracy_drop),
        "max_safe_to_danger_increase": int(
            args.max_safe_to_danger_increase
        ),
        "source_configuration": configuration,
        "expert_checkpoint": str(args.expert_checkpoint),
        "expert_training_config": checkpoint.get("training_config", {}),
        "selected": {
            "class_strengths": class_selected,
            "risk_strengths": risk_strengths,
            "danger_biases": danger_biases,
        },
        "combined_clean": clean,
        "combined_drop_one_link": cyclic,
        "per_link": per_link,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selected": report["selected"],
        "combined_clean": clean,
        "combined_drop_one_link": cyclic,
        "per_link": {
            key: {
                "class": value["selected_class"],
                "risk": value["selected_risk"],
            }
            for key, value in per_link.items()
        },
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
