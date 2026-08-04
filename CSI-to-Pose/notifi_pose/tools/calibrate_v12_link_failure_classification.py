"""Calibrate a missing-link classifier expert on validation logits only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset
from .evaluate_v12_final import _load_hybrid, _read_locked, build_locked_model
from .train_seen_v4_trajectory import classification_metrics, make_loaders
from torch.utils.data import DataLoader


@torch.no_grad()
def _collect_pair(primary, expert, loader, device: str) -> dict:
    values = {key: [] for key in (
        "primary_class", "primary_risk", "expert_class", "expert_risk",
        "class_target", "risk_target", "failed", "missing_link",
    )}
    primary.eval()
    expert.eval()
    for batch in loader:
        csi = batch["csi"].to(device)
        mask = batch["link_mask"].to(device)
        first = primary(csi, mask)
        second = (
            expert.forward_logits(csi, mask)
            if hasattr(expert, "forward_logits") else expert(csi, mask)
        )
        values["primary_class"].append(first["class_logits"].float().cpu())
        values["primary_risk"].append(first["risk_logits"].float().cpu())
        values["expert_class"].append(second["class_logits"].float().cpu())
        values["expert_risk"].append(second["risk_logits"].float().cpu())
        values["class_target"].append(batch["class_id"].long())
        values["risk_target"].append(batch["risk_id"].long())
        alive = mask.any(dim=1)
        failed = alive.sum(dim=-1).lt(C.N_LINKS)
        coverage = mask.float().mean(dim=1)
        missing = coverage.argmin(dim=-1)
        values["failed"].append(failed.cpu())
        values["missing_link"].append(
            torch.where(failed, missing, torch.full_like(missing, -1)).cpu()
        )
    return {key: torch.cat(items) for key, items in values.items()}


def _mix(primary: torch.Tensor, expert: torch.Tensor, failed: torch.Tensor,
         strength: float) -> torch.Tensor:
    output = primary.clone()
    if strength and failed.any():
        probability = (
            (1.0 - strength) * torch.softmax(primary[failed], dim=-1)
            + strength * torch.softmax(expert[failed], dim=-1)
        )
        output[failed] = probability.clamp_min(1e-8).log()
    return output


def _metrics(pair: dict, class_strength: float, risk_strength: float,
             danger_bias: float = 0.0) -> dict:
    risk_logits = _mix(
        pair["primary_risk"], pair["expert_risk"], pair["failed"],
        risk_strength,
    )
    if danger_bias and pair["failed"].any():
        risk_logits[pair["failed"], C.N_RISK - 1] += danger_bias
    logits = {
        "class_logits": _mix(
            pair["primary_class"], pair["expert_class"], pair["failed"],
            class_strength,
        ),
        "risk_logits": risk_logits,
        "class_target": pair["class_target"],
        "risk_target": pair["risk_target"],
    }
    return classification_metrics(logits, 0.0)


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
        default=(-0.25, 0.0, 0.25, 0.5),
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
    failure_loader = DataLoader(
        PerturbedDataset(loaders["val_class"].dataset, "drop_one_link"),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    failure_pair = _collect_pair(primary, expert, failure_loader, device)
    baseline_clean = _metrics(clean_pair, 0.0, 0.0)
    baseline_failure = _metrics(failure_pair, 0.0, 0.0)

    class_candidates = []
    for strength in args.strengths:
        clean = _metrics(clean_pair, float(strength), 0.0)
        failure = _metrics(failure_pair, float(strength), 0.0)
        feasible = (
            clean["class"]["macro_f1"]
            >= baseline_clean["class"]["macro_f1"] - 0.005
        )
        class_candidates.append({
            "strength": float(strength),
            "feasible": bool(feasible),
            "score": float(failure["class"]["macro_f1"]),
            "clean_class": clean["class"],
            "failure_class": failure["class"],
        })
    feasible_class = [item for item in class_candidates if item["feasible"]]
    selected_class = max(
        feasible_class or class_candidates, key=lambda item: item["score"]
    )

    risk_candidates = []
    for strength in args.strengths:
        for bias in args.danger_biases:
            clean = _metrics(clean_pair, 0.0, float(strength), float(bias))
            failure = _metrics(
                failure_pair, 0.0, float(strength), float(bias)
            )
            clean_risk = clean["risk"]
            risk = failure["risk"]
            feasible = (
                clean_risk["macro_f1"]
                >= baseline_clean["risk"]["macro_f1"] - 0.005
                and clean_risk["safe_to_danger"]
                <= baseline_clean["risk"]["safe_to_danger"] + 1
                and risk["accuracy"]
                >= baseline_failure["risk"]["accuracy"] - 0.005
                and risk["safe_to_danger"]
                <= baseline_failure["risk"]["safe_to_danger"] + 2
            )
            score = (
                risk["macro_f1"]
                + 0.50 * risk["danger_recall"]
                - 0.50 * risk["safe_to_danger_rate"]
            )
            risk_candidates.append({
                "strength": float(strength),
                "danger_bias": float(bias),
                "feasible": bool(feasible),
                "score": float(score),
                "clean_risk": clean_risk,
                "failure_risk": risk,
            })
    feasible_risk = [item for item in risk_candidates if item["feasible"]]
    selected_risk = max(
        feasible_risk or risk_candidates, key=lambda item: item["score"]
    )
    combined_clean = _metrics(
        clean_pair, selected_class["strength"], selected_risk["strength"],
        selected_risk["danger_bias"],
    )
    combined_failure = _metrics(
        failure_pair, selected_class["strength"], selected_risk["strength"],
        selected_risk["danger_bias"],
    )
    report = {
        "run": "p2_v12_link_failure_classification_calibration",
        "protocol": args.exp,
        "selection_split": "validation_drop_one_link",
        "test_used_for_selection": False,
        "source_configuration": configuration,
        "expert_checkpoint": str(args.expert_checkpoint),
        "expert_training_config": checkpoint.get("training_config", {}),
        "natural_clean_failed_samples": int(clean_pair["failed"].sum()),
        "synthetic_failed_samples": int(failure_pair["failed"].sum()),
        "baseline_clean": baseline_clean,
        "baseline_drop_one_link": baseline_failure,
        "selected_class": selected_class,
        "selected_risk": selected_risk,
        "combined_clean": combined_clean,
        "combined_drop_one_link": combined_failure,
        "class_candidates": class_candidates,
        "risk_candidates": risk_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selected_class": selected_class,
        "selected_risk": selected_risk,
        "combined_clean": combined_clean,
        "combined_drop_one_link": combined_failure,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
