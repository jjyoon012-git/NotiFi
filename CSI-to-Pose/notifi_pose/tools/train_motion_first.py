"""Train and evaluate the motion-first CSI encoder on source and sealed yja E02."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..motion_first import MotionFirstEncoder
from ..trainer import fit_norm, set_seed
from .diagnose_observability import (
    ShuffledSignalDataset,
    binary_metrics,
    finite_mean,
    macro_f1,
    pose_only,
    report_path,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.new_zeros(())


def motion_targets(batch: dict) -> dict:
    valid = batch["valid"].bool()
    speed, pair_valid = L.target_motion(
        batch["pose_rel"].float(), batch["root"].float(), valid
    )
    speed_valid = torch.zeros_like(valid)
    speed_valid[:, 1:] = pair_valid
    phase, phase_valid = L.phase_targets(speed, valid, batch["risk_id"])
    impact = L.impact_window(
        batch["pose_rel"].float(), batch["root"].float(), valid,
        batch["risk_id"],
    )
    return {
        "speed": speed,
        "speed_log": torch.log1p(speed * 10.0),
        "speed_valid": speed_valid,
        "moving": (speed > 0.25).float(),
        "phase": phase,
        "phase_valid": phase_valid,
        "impact": impact.float(),
        "danger": batch["risk_id"] == 2,
    }


def trajectory_correlation_loss(prediction: torch.Tensor, target: torch.Tensor,
                                valid: torch.Tensor) -> torch.Tensor:
    losses = []
    for item in range(len(prediction)):
        selected = valid[item]
        if selected.sum() < 12:
            continue
        predicted_values = prediction[item, selected]
        target_values = target[item, selected]
        predicted_values = predicted_values - predicted_values.mean()
        target_values = target_values - target_values.mean()
        denominator = predicted_values.norm() * target_values.norm()
        if denominator > 1e-6:
            losses.append(1.0 - (predicted_values * target_values).sum() / denominator)
    return torch.stack(losses).mean() if losses else prediction.new_zeros(())


def compute_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    target = motion_targets(batch)
    speed_loss = masked_mean(
        F.smooth_l1_loss(
            output["speed_log"], target["speed_log"], reduction="none", beta=0.10
        ),
        target["speed_valid"],
    )
    correlation_loss = trajectory_correlation_loss(
        output["speed_log"], target["speed_log"], target["speed_valid"]
    )
    moving_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            output["moving_logits"], target["moving"], reduction="none",
            pos_weight=output["moving_logits"].new_tensor(2.0),
        ),
        target["speed_valid"],
    )
    if target["phase_valid"].any():
        phase_loss = F.cross_entropy(
            output["phase_logits"][target["phase_valid"]],
            target["phase"][target["phase_valid"]],
            weight=output["phase_logits"].new_tensor((0.25, 1.0, 3.0, 1.0)),
        )
    else:
        phase_loss = output["phase_logits"].new_zeros(())
    danger_frames = target["danger"][:, None] & batch["valid"].bool()
    impact_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            output["impact_logits"], target["impact"], reduction="none",
            pos_weight=output["impact_logits"].new_tensor(20.0),
        ),
        danger_frames,
    )
    class_loss = F.cross_entropy(output["class_logits"], batch["class_id"])
    risk_loss = F.cross_entropy(output["risk_logits"], batch["risk_id"])
    contrastive = L.cross_domain_supervised_contrastive(
        output["embedding"], batch["class_id"], batch["domain_id"]
    )
    total = (
        speed_loss + 0.5 * correlation_loss + moving_loss
        + 0.5 * phase_loss + impact_loss
        + 0.2 * class_loss + 0.2 * risk_loss + 0.1 * contrastive
    )
    return total, {
        "speed": float(speed_loss.detach()),
        "correlation": float(correlation_loss.detach()),
        "moving": float(moving_loss.detach()),
        "phase": float(phase_loss.detach()),
        "impact": float(impact_loss.detach()),
        "class": float(class_loss.detach()),
        "risk": float(risk_loss.detach()),
        "contrastive": float(contrastive.detach()),
    }


@torch.no_grad()
def evaluate(model: MotionFirstEncoder, dataset: Dataset, device: str,
             batch_size: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    values = {key: [] for key in (
        "speed", "speed_target", "moving", "moving_target", "phase",
        "phase_target", "phase_valid", "impact", "impact_target", "danger",
    )}
    class_ok = risk_ok = trials = 0
    timing_errors = []
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(batch["csi"], batch["link_mask"])
        target = motion_targets(batch)
        speed = torch.expm1(output["speed_log"]).clamp_min(0.0) / 10.0
        frame_valid = target["speed_valid"]
        values["speed"].append(speed[frame_valid].cpu())
        values["speed_target"].append(target["speed"][frame_valid].cpu())
        values["moving"].append(torch.sigmoid(output["moving_logits"])[frame_valid].cpu())
        values["moving_target"].append(target["moving"][frame_valid].cpu())
        values["phase"].append(output["phase_logits"].argmax(-1).cpu())
        values["phase_target"].append(target["phase"].cpu())
        values["phase_valid"].append(target["phase_valid"].cpu())
        danger_frames = target["danger"][:, None] & batch["valid"].bool()
        values["impact"].append(torch.sigmoid(output["impact_logits"])[danger_frames].cpu())
        values["impact_target"].append(target["impact"][danger_frames].cpu())
        values["danger"].append(target["danger"].cpu())
        class_ok += int((output["class_logits"].argmax(-1) == batch["class_id"]).sum())
        risk_ok += int((output["risk_logits"].argmax(-1) == batch["risk_id"]).sum())
        trials += len(batch["class_id"])
        impact_probability = torch.sigmoid(output["impact_logits"])
        for item in range(len(batch["risk_id"])):
            target_impact = target["impact"][item].bool()
            if not target_impact.any():
                continue
            valid_positions = torch.nonzero(batch["valid"][item], as_tuple=False).flatten()
            predicted_frame = int(valid_positions[
                torch.argmax(impact_probability[item, valid_positions])
            ])
            target_frame = float(torch.nonzero(
                target_impact, as_tuple=False
            ).float().mean())
            timing_errors.append(abs(predicted_frame - target_frame))
    values = {
        key: torch.cat(parts) if parts else torch.empty(0)
        for key, parts in values.items()
    }
    speed = values["speed"]
    speed_target = values["speed_target"]
    residual = ((speed - speed_target) ** 2).sum()
    total = ((speed_target - speed_target.mean()) ** 2).sum().clamp_min(1e-8)
    correlation = torch.corrcoef(torch.stack((speed, speed_target)))[0, 1]
    phase_mask = values["phase_valid"].bool()
    metrics = {
        "n_trials": trials,
        "n_speed_frames": len(speed),
        "speed_mae_mps": float((speed - speed_target).abs().mean()),
        "speed_r2": float(1.0 - residual / total),
        "speed_correlation": float(correlation),
        "moving": binary_metrics(values["moving"], values["moving_target"]),
        "phase_macro_f1": macro_f1(
            values["phase"][phase_mask], values["phase_target"][phase_mask], 4
        ) if phase_mask.any() else math.nan,
        "impact": binary_metrics(values["impact"], values["impact_target"]),
        "impact_timing_mae_frames": finite_mean(timing_errors),
        "action_accuracy": class_ok / max(trials, 1),
        "risk_accuracy": risk_ok / max(trials, 1),
    }
    metrics["selection_score"] = (
        metrics["speed_r2"] + metrics["moving"]["f1"]
        + 0.5 * metrics["phase_macro_f1"] + metrics["impact"]["f1"]
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--baseline", default="sub", choices=("none", "sub", "sub_z"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exp", default="yja_holdout", choices=("single_split", "yja_holdout")
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_yja_e02",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dropout = DropoutConfig(rf_augment=True)
    datasets = build_datasets(
        exp=args.exp, baseline=args.baseline, dropout=dropout, seed=args.seed
    )
    train = pose_only(datasets["train"])
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    train.train = True
    train.dropout = dropout
    loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device == "cuda",
    )
    model = MotionFirstEncoder(
        hidden=args.hidden, temporal_layers=args.temporal_layers,
        heads=args.heads,
    ).to(device)
    fit_norm(model, loader, device, max_batches=20)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    history = []
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(batch["csi"], batch["link_mask"])
                loss, parts = compute_loss(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["class_id"])
            totals["total"] = totals.get("total", 0.0) + float(loss.detach()) * count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * count
            examples += count
        validation_metrics = evaluate(
            model, validation, device, args.batch_size * 2
        )
        epoch_result = {
            "epoch": epoch,
            "train": {key: value / max(examples, 1) for key, value in totals.items()},
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch:02d} loss={epoch_result['train']['total']:.4f} "
            f"val_R2={validation_metrics['speed_r2']:.3f} "
            f"move_F1={validation_metrics['moving']['f1']:.3f} "
            f"impact_F1={validation_metrics['impact']['f1']:.3f} "
            f"score={validation_metrics['selection_score']:.3f}"
        )
        if validation_metrics["selection_score"] > best_score:
            best_score = validation_metrics["selection_score"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "validation": validation_metrics,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "run": f"motion_first_{args.exp}",
        "protocol": args.exp,
        "device": device,
        "parameters": model.n_params(),
        "pose_trials": {
            "train": len(train), "validation": len(validation), "test": len(test),
        },
        "best_epoch": checkpoint["epoch"],
        "source_validation": evaluate(
            model, validation, device, args.batch_size * 2
        ),
        "test": evaluate(model, test, device, args.batch_size * 2),
        "shuffled_test": evaluate(
            model, ShuffledSignalDataset(test, args.seed),
            device, args.batch_size * 2,
        ),
        "checkpoint": report_path(checkpoint_path),
        "history": history,
    }
    output_path = args.run_dir / "results.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
