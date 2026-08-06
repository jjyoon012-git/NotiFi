"""One-shot fixed-test evaluation of CSI-only relative floor proximity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import ContactProfileHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from .calibrate_csi_proximity_thresholds import proximity_metrics
from .diagnose_observability import pose_only, report_path
from .train_csi_contact_profile import contact_targets, predict_contact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp13_relative_proximity_seed241"
        / "best_model.pt",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp13_relative_proximity_seed241"
        / "thresholds.json",
    )
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp13_relative_proximity_seed241"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    model = ContactProfileHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    cache = torch.load(
        args.feature_root / "test_features.pt", map_location="cpu",
        weights_only=False,
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    dataset = QualityWeightedDataset(
        pose_only(datasets["test"]), protocol_audit_path(args.exp)
    )
    target, valid, risk = contact_targets(dataset, "relative")
    inference_valid = cache["frame_mask"].bool()
    probability = torch.sigmoid(predict_contact(
        model, cache, inference_valid, device
    ))
    valid = valid & inference_valid
    thresholds = torch.tensor(calibration["thresholds"])
    result = {
        "status": "fixed_test_evaluation",
        "protocol": args.exp,
        "metrics": proximity_metrics(
            probability, target, valid, risk, thresholds
        ),
        "thresholds": calibration["thresholds"],
        "contact_order": calibration["contact_order"],
        "checkpoint": report_path(args.checkpoint),
        "calibration": report_path(args.calibration),
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
