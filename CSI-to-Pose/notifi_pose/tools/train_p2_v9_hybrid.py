"""Train and validate a P2 coarse model plus bounded V9 residual decoder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..hybrid_v10 import P2V9HybridNet
from ..seen_v4 import trajectory_reconstruction_loss
from ..trainer import set_seed
from .diagnose_observability import ShuffledSignalDataset, evaluate_model, report_path
from .evaluate_sealed import make_model
from .train_seen_v4_trajectory import (
    _balanced_weight,
    classification_metrics,
    collect_classification_logits,
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
    move_batch,
)


def pose_selection_score(metrics: dict) -> float:
    speed = max(float(metrics["pose_speed_ratio"]), 1e-3)
    return (
        float(metrics["mpjpe_m"])
        + 0.35 * float(metrics["danger_mpjpe_m"])
        + 0.10 * float(metrics["danger_endpoint_mpjpe_m"])
        + 0.15 * abs(math.log(speed))
    )


def root_selection_score(metrics: dict) -> float:
    return (
        float(metrics["root_error_m"])
        + 0.40 * float(metrics["danger_root_error_m"])
        + 0.20 * float(metrics["danger_root_drop_mae_m"])
    )


def _candidate(model, loaders, device: str, max_shift: int,
               pose: float, root: float, classification: float,
               risk: float) -> dict:
    model.set_calibration(pose, root, classification, risk)
    return evaluate_trajectory(model, loaders["val"], device, max_shift)


def select_calibration(model, loaders, device: str, args) -> dict:
    pose_candidates = []
    for strength in args.pose_strengths:
        metrics = _candidate(
            model, loaders, device, args.max_shift,
            strength, 0.0, 0.0, 0.0,
        )
        pose_candidates.append({
            "strength": strength,
            "feasible_speed": 0.85 <= metrics["pose_speed_ratio"] <= 1.20,
            "score": pose_selection_score(metrics),
            "validation": metrics,
        })
    feasible_pose = [item for item in pose_candidates if item["feasible_speed"]]
    pose = min(feasible_pose or pose_candidates, key=lambda item: item["score"])

    root_candidates = []
    for strength in args.root_strengths:
        metrics = _candidate(
            model, loaders, device, args.max_shift,
            pose["strength"], strength, 0.0, 0.0,
        )
        root_candidates.append({
            "strength": strength,
            "score": root_selection_score(metrics),
            "validation": metrics,
        })
    root = min(root_candidates, key=lambda item: item["score"])
    root_baseline = next(
        item for item in root_candidates if float(item["strength"]) == 0.0
    )
    root_gain = (
        root_baseline["validation"]["root_error_m"]
        - root["validation"]["root_error_m"]
    )
    if root_gain < args.minimum_root_gain:
        root = root_baseline

    class_candidates = []
    for strength in args.logit_strengths:
        model.set_calibration(
            pose["strength"], root["strength"], strength, 0.0
        )
        metrics = evaluate_classification(model, loaders["val_class"], device)
        class_candidates.append({
            "strength": strength,
            "score": metrics["class"]["macro_f1"] + 0.25 * metrics["class"]["accuracy"],
            "validation": metrics["class"],
        })
    classification = max(class_candidates, key=lambda item: item["score"])

    risk_candidates = []
    for strength in args.logit_strengths:
        model.set_calibration(
            pose["strength"], root["strength"],
            classification["strength"], strength,
        )
        logits = collect_classification_logits(model, loaders["val_class"], device)
        bias_steps = int(round(args.maximum_danger_bias / 0.05)) + 1
        for bias in np.linspace(0.0, args.maximum_danger_bias, bias_steps):
            metrics = classification_metrics(logits, float(bias))["risk"]
            risk_candidates.append({
                "strength": strength,
                "danger_logit_bias": float(bias),
                "feasible": metrics["danger_recall"] >= args.minimum_danger_recall,
                "score": (
                    metrics["macro_f1"] + 0.25 * metrics["accuracy"]
                    - 0.50 * metrics["safe_to_danger_rate"]
                ),
                "validation": metrics,
            })
    feasible_risk = [item for item in risk_candidates if item["feasible"]]
    risk = max(feasible_risk or risk_candidates, key=lambda item: item["score"])
    return {
        "pose": pose,
        "root": root,
        "classification": classification,
        "risk": risk,
        "pose_candidates": pose_candidates,
        "root_candidates": root_candidates,
        "class_candidates": class_candidates,
        "risk_candidates": risk_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--exp", default="single_split_lmh_e01",
        choices=("single_split", "single_split_lmh_e01"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--lambda-class", type=float, default=0.10)
    parser.add_argument("--lambda-risk", type=float, default=0.10)
    parser.add_argument("--risk-danger-boost", type=float, default=1.25)
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--minimum-danger-recall", type=float, default=0.94)
    parser.add_argument("--maximum-danger-bias", type=float, default=4.0)
    parser.add_argument(
        "--minimum-root-gain", type=float, default=0.005,
        help="minimum validation root-error gain in metres before enabling the residual",
    )
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
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v9_hybrid_v10",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, loaders = make_loaders(args, device)
    train, _, test = datasets
    checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    base = make_model(checkpoint, device)
    model = P2V9HybridNet(base).to(device)
    class_weight = _balanced_weight(train.index, "class_id", C.N_CLASSES, device)
    risk_weight = _balanced_weight(
        train.index, "risk_id", C.N_RISK, device, args.risk_danger_boost
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"

    model.set_calibration(0.0, 0.0, 0.0, 0.0)
    baseline_validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    baseline_classification = evaluate_classification(
        model, loaders["val_class"], device
    )
    # Root, danger trajectory, and logits are gated independently after
    # training. Preserve the checkpoint with the best validation MPJPE, then
    # let the calibration grid reject residual strengths that hurt a subgroup.
    best_score = float(baseline_validation["mpjpe_m"])
    torch.save({
        "model": model.state_dict(),
        "epoch": 0,
        "validation": baseline_validation,
        "validation_classification": baseline_classification,
    }, checkpoint_path)

    model.set_calibration(1.0, 1.0, 1.0, 1.0)
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(batch["csi"], batch["link_mask"])
                loss, parts = trajectory_reconstruction_loss(
                    output, batch,
                    alignment_weight=args.alignment_weight,
                    max_shift=args.max_shift,
                    class_weight=class_weight,
                    risk_weight=risk_weight,
                    lambda_class=args.lambda_class,
                    lambda_risk=args.lambda_risk,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["class_id"])
            examples += count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * count

        train_metrics = {
            key: value / max(examples, 1) for key, value in totals.items()
        }
        validation = evaluate_trajectory(
            model, loaders["val"], device, args.max_shift
        )
        validation_classification = evaluate_classification(
            model, loaders["val_class"], device
        )
        score = float(validation["mpjpe_m"])
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "selection_score": score,
            "validation": validation,
            "validation_classification": validation_classification,
        })
        print(
            f"epoch={epoch:02d} loss={train_metrics['total']:.4f} "
            f"mpjpe={validation['mpjpe_m'] * 100:.2f}cm "
            f"danger={validation['danger_mpjpe_m'] * 100:.2f}cm "
            f"root={validation['root_error_m'] * 100:.2f}cm "
            f"class={validation_classification['class']['accuracy']:.3f} "
            f"danger-R={validation_classification['risk']['danger_recall']:.3f}"
        )
        if score < best_score:
            best_score = score
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "validation": validation,
                "validation_classification": validation_classification,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    selected_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(selected_checkpoint["model"])
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
    test_metrics = evaluate_trajectory(
        model, loaders["test"], device, args.max_shift
    )
    raw_test_classification = evaluate_classification(
        model, loaders["test_class"], device
    )
    test_classification = evaluate_classification(
        model, loaders["test_class"], device, selected["danger_logit_bias"]
    )
    result = {
        "run": "p2_v9_hybrid_v10",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_p2": report_path(args.p2_checkpoint),
        "best_epoch": selected_checkpoint["epoch"],
        "selected": selected,
        "baseline_validation": baseline_validation,
        "baseline_validation_classification": baseline_classification,
        "selected_validation": calibration["root"]["validation"],
        "validation_classification": {
            "class": calibration["classification"]["validation"],
            "risk": calibration["risk"]["validation"],
        },
        "test": test_metrics,
        "raw_test_classification": raw_test_classification,
        "test_classification": test_classification,
        "shuffled_test": evaluate_model(
            model, ShuffledSignalDataset(test, args.seed),
            device, args.batch_size * 2, 5,
        ),
        "calibration_candidates": {
            "pose": calibration["pose_candidates"],
            "root": calibration["root_candidates"],
            "class": calibration["class_candidates"],
            "risk": calibration["risk_candidates"],
        },
        "history": history,
    }
    (args.run_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "source_p2": report_path(args.p2_checkpoint),
        "selected": selected,
        "validation": result["selected_validation"],
        "test": test_metrics,
        "test_classification": test_classification,
    }, args.run_dir / "calibrated_model.pt")
    print(json.dumps({
        "selected": selected,
        "test": test_metrics,
        "test_classification": test_classification,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
