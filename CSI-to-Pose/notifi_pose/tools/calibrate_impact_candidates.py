"""Calibrate CSI motion-peak candidate filtering for impact timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import DropoutConfig, build_datasets
from ..impact_event import ImpactEventLocalizer, physical_impact_targets
from ..quality import QualityWeightedDataset
from .diagnose_observability import pose_only, report_path
from .train_impact_event import load_v3, move_batch


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        v3_checkpoint=C.WORK_ROOT / "runs" / "seen_v3_contact_root" / "calibrated_model.pt",
        v2_checkpoint=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibrated_model.pt",
        baseline_checkpoint=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
        motion_checkpoint=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
        pose_residual_checkpoint=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
        root_residual_checkpoint=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
    )


@torch.no_grad()
def collect(model, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    values = {key: [] for key in ("base", "raw", "valid", "target")}
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["csi"], batch["link_mask"])
        target = physical_impact_targets(
            batch["pose_rel"], batch["root"], batch["valid"].bool(),
            batch["risk_id"],
        )
        selected = target["event_valid"]
        if not selected.any():
            continue
        values["base"].append(output["impact_logits"][selected].cpu().numpy())
        values["raw"].append(
            output["raw_csi_event_features_v8"][selected].cpu().numpy()
        )
        values["valid"].append(
            batch["link_mask"].any(-1)[selected].cpu().numpy()
        )
        values["target"].append(target["event_frame"][selected].cpu().numpy())
    return {key: np.concatenate(chunks) for key, chunks in values.items()}


def evaluate(data: dict[str, np.ndarray], channel: int,
             fraction: float, alpha: float) -> dict:
    raw = data["raw"][..., channel]
    valid = data["valid"]
    logits = data["base"] + alpha * raw
    candidate = valid.copy()
    if fraction < 1.0:
        candidate[:] = False
        for item in range(len(raw)):
            count = max(1, int(valid[item].sum() * fraction))
            score = np.where(valid[item], raw[item], -np.inf)
            indices = np.argpartition(score, -count)[-count:]
            candidate[item, indices] = True
    prediction = np.where(candidate, logits, -np.inf).argmax(1)
    error = np.abs(prediction - data["target"])
    nearest = []
    for item in range(len(raw)):
        nearest.append(np.abs(
            np.flatnonzero(candidate[item]) - data["target"][item]
        ).min())
    nearest = np.asarray(nearest)
    return {
        "channel": channel,
        "candidate_fraction": fraction,
        "energy_alpha": alpha,
        "timing_mae_frames": float(error.mean()),
        "timing_median_frames": float(np.median(error)),
        "timing_hit_at_2": float((error <= 2).mean()),
        "timing_hit_at_5": float((error <= 5).mean()),
        "candidate_oracle_hit_at_2": float((nearest <= 2).mean()),
        "candidate_oracle_hit_at_5": float((nearest <= 5).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "impact_event_v8c_raw" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "impact_event_v8c_raw" / "candidate_calibration.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImpactEventLocalizer(load_v3(make_args(), device)).to(device)
    checkpoint = torch.load(
        args.event_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.set_calibration(event=0, joint=0, contact=0, speed=0)
    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=7,
    )
    loaders = {
        split: DataLoader(
            QualityWeightedDataset(pose_only(datasets[split])),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        for split in ("val", "test")
    }
    validation = collect(model, loaders["val"], device)
    candidates = []
    for channel in range(4):
        for fraction in (0.05, 0.10, 0.20, 1.0):
            for alpha in (0.0, 0.25, 0.50, 1.0, 2.0):
                candidates.append(evaluate(
                    validation, channel, fraction, alpha
                ))
    selected = min(
        candidates,
        key=lambda item: (
            item["timing_mae_frames"], -item["timing_hit_at_5"]
        ),
    )
    test = collect(model, loaders["test"], device)
    result = {
        "run": "impact_candidate_calibration",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_checkpoint": report_path(args.event_checkpoint),
        "selected": selected,
        "test": evaluate(
            test, selected["channel"], selected["candidate_fraction"],
            selected["energy_alpha"],
        ),
        "baseline_validation": evaluate(validation, 0, 1.0, 0.0),
        "baseline_test": evaluate(test, 0, 1.0, 0.0),
        "candidates": candidates,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "run", "selected", "test", "baseline_validation", "baseline_test"
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
