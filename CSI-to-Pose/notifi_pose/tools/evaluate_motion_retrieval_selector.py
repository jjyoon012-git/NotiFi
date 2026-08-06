"""Confirm the validation-locked KP5-MPR-S deployment on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance, select_indices
from .diagnose_observability import pose_only, report_path
from .evaluate_motion_retrieval_pose import _load_model
from .train_kinetic_pose import CoarsePoseStore, pose_selection_score
from .train_motion_retrieval_selector import extract_features, predict_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "best_model.pt",
    )
    parser.add_argument(
        "--source-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp4_dcc_staged_seed17"
        / "deployment_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "test_selected.json",
    )
    args = parser.parse_args()
    if not args.allow_test:
        raise RuntimeError("test confirmation requires explicit --allow-test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    test = QualityWeightedDataset(pose_only(datasets["test"]), audit)
    target_pose, target_valid, target_class, target_risk = _load_pose_arrays(test)
    frozen, source = _load_model(args.source_checkpoint, device)
    raw_cache = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    coarse_store = CoarsePoseStore(raw_cache["rows"], raw_cache["pose"])
    feature_cache = extract_features(
        frozen, test, coarse_store,
        args.checkpoint.parent / "test_features.pt",
        device, args.batch_size, args.exp,
    )
    del frozen
    output = predict_selector(selector, feature_cache, args.batch_size * 2, device)
    baseline = feature_cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, target_valid)
    ])
    train_bank = checkpoint["train_bank"].float()
    pose_distance = exact_pose_distance(
        baseline_bank, train_bank,
        args.checkpoint.parent / "test_exact_pose_distance.pt",
    )
    latent_distance = torch.cdist(
        output["embedding"], checkpoint["train_embedding"].float()
    )
    fused_logits = (
        feature_cache["base_action_logits"].float()
        + output["action_logits"]
    )
    indices = select_indices(
        pose_distance, latent_distance, checkpoint["train_class"].long(),
        fused_logits, "soft_05",
    )
    candidate = torch.stack([
        _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
        for index, valid in zip(indices, target_valid)
    ])
    selected_pose = 0.50 * baseline + 0.50 * candidate
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "soft_fused_05_cartesian_500": _metric_batch(
            selected_pose, target_pose, target_valid, target_risk
        ),
    }
    result = {
        "status": "promoted_candidate_test_confirmation",
        "protocol": args.exp,
        "selection_source": "validation deployment_calibration.json",
        "test_used_for_selection": False,
        "selected_configuration": "soft_fused_05_cartesian_500",
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "metrics": metrics,
        "classification": {
            "base_action_accuracy": float(
                (feature_cache["base_action_logits"].argmax(-1) == target_class).float().mean()
            ),
            "selector_action_accuracy": float(
                (output["action_logits"].argmax(-1) == target_class).float().mean()
            ),
            "fused_action_accuracy": float(
                (fused_logits.argmax(-1) == target_class).float().mean()
            ),
            "selector_risk_accuracy": float(
                (output["risk_logits"].argmax(-1) == target_risk).float().mean()
            ),
        },
        "checkpoint": report_path(args.checkpoint),
        "source_run": source.get("run"),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
