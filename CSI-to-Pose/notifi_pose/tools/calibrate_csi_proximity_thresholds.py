"""Calibrate body-to-floor proximity thresholds on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import ContactProfileHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from .diagnose_observability import pose_only, report_path
from .train_csi_contact_profile import (
    PROXIMITY_JOINTS, contact_targets, predict_contact,
)


def _f1(predicted, target, mask):
    predicted = predicted[mask]
    target = target[mask]
    tp = (predicted & target).sum().float()
    fp = (predicted & ~target).sum().float()
    fn = (~predicted & target).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def proximity_metrics(probability, target, valid, risk, thresholds):
    predicted = probability >= thresholds[None, None]
    target = target.bool()
    mask = valid[..., None].expand_as(target)
    danger = mask & (risk[:, None, None] == 2)
    per_contact = []
    danger_per_contact = []
    for contact in range(target.shape[-1]):
        per_contact.append(_f1(
            predicted[..., contact], target[..., contact], valid
        ))
        danger_per_contact.append(_f1(
            predicted[..., contact], target[..., contact],
            danger[..., contact],
        ))
    overall = _f1(predicted, target, mask)
    danger_f1 = _f1(predicted, target, danger)
    return {
        "f1": overall,
        "danger_f1": danger_f1,
        "per_contact_f1": per_contact,
        "danger_per_contact_f1": danger_per_contact,
        "selection_score": 1.0 - 0.35 * overall - 0.65 * danger_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp13_relative_proximity_seed241"
        / "best_model.pt",
    )
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp13_relative_proximity_seed241"
        / "thresholds.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("target_mode") != "relative":
        raise ValueError("Threshold calibration requires a relative-proximity head")
    model = ContactProfileHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    cache = torch.load(
        args.feature_root / "val_features.pt", map_location="cpu",
        weights_only=False,
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    dataset = QualityWeightedDataset(
        pose_only(datasets["val"]), protocol_audit_path(args.exp)
    )
    target, valid, risk = contact_targets(dataset, "relative")
    inference_valid = cache["frame_mask"].bool()
    probability = torch.sigmoid(predict_contact(
        model, cache, inference_valid, device
    ))
    valid = valid & inference_valid
    thresholds = torch.full((len(PROXIMITY_JOINTS),), 0.50)
    grid = torch.arange(0.20, 0.81, 0.025)
    best = proximity_metrics(probability, target, valid, risk, thresholds)
    for _ in range(3):
        changed = False
        for contact in range(len(thresholds)):
            local = None
            for value in grid:
                candidate = thresholds.clone()
                candidate[contact] = value
                metrics = proximity_metrics(
                    probability, target, valid, risk, candidate
                )
                if local is None or metrics["selection_score"] < local[0]:
                    local = (metrics["selection_score"], float(value), metrics)
            if local[0] < best["selection_score"] - 1e-7:
                thresholds[contact] = local[1]
                best = local[2]
                changed = True
        if not changed:
            break
    result = {
        "status": "validation_locked_relative_proximity_thresholds",
        "protocol": args.exp,
        "thresholds": thresholds.tolist(),
        "contact_order": list(PROXIMITY_JOINTS),
        "validation": best,
        "checkpoint": report_path(args.checkpoint),
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "semantic_limit": "relative floor proximity, not measured collision",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
