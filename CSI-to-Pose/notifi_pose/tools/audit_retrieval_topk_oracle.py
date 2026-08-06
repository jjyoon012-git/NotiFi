"""Audit whether KP5's top-K train motions contain a better fall hypothesis."""

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
    PARTS,
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only
from .train_motion_retrieval_selector import predict_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "best_model.pt",
    )
    parser.add_argument(
        "--feature-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "val_features.pt",
    )
    parser.add_argument(
        "--distance-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "val_exact_pose_distance.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "topk_oracle.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    output = predict_selector(model, cache, 48, device)

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    validation = QualityWeightedDataset(
        pose_only(datasets["val"]), protocol_audit_path(args.exp)
    )
    target_pose, target_valid, _, target_risk = _load_pose_arrays(validation)
    target_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(target_pose, target_valid)
    ])
    baseline = cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, target_valid)
    ])
    train_bank = checkpoint["train_bank"].float()
    train_class = checkpoint["train_class"].long()
    pose_distance = exact_pose_distance(
        baseline_bank, train_bank, args.distance_cache
    )
    fused_logits = cache["base_action_logits"].float() + output["action_logits"]
    probability = torch.softmax(fused_logits, dim=-1).clamp_min(1e-6)
    score = []
    for item, distance in enumerate(pose_distance):
        scale = torch.quantile(distance, 0.50).clamp_min(1e-5)
        score.append(
            distance / scale - 0.05 * probability[item, train_class].log()
        )
    score = torch.stack(score)

    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    recall_distance = {}
    for top_k in (1, 3, 5, 10, 20, 50):
        whole_predictions, part_predictions = [], []
        selected_distance = []
        for item, valid in enumerate(target_valid):
            indices = score[item].topk(top_k, largest=False).indices
            candidates = train_bank.index_select(0, indices)
            error = torch.linalg.vector_norm(
                candidates - target_bank[item][None], dim=-1
            )
            whole_error = error.mean((1, 2))
            whole = candidates[whole_error.argmin()]
            part = whole.clone()
            for joints in PARTS.values():
                part_error = error[:, :, joints].mean((1, 2))
                part[:, joints] = candidates[part_error.argmin(), :, joints]
            whole_predictions.append(_render(whole, valid, C.CACHE_FRAMES))
            part_predictions.append(_render(part, valid, C.CACHE_FRAMES))
            selected_distance.append(float(whole_error.min()))
        whole = torch.stack(whole_predictions)
        part = torch.stack(part_predictions)
        metrics[f"top{top_k}_whole_oracle_blend_500"] = _metric_batch(
            0.5 * baseline + 0.5 * whole,
            target_pose, target_valid, target_risk,
        )
        metrics[f"top{top_k}_part_oracle_blend_500"] = _metric_batch(
            0.5 * baseline + 0.5 * part,
            target_pose, target_valid, target_risk,
        )
        recall_distance[str(top_k)] = {
            "mean_m": float(torch.tensor(selected_distance).mean()),
            "median_m": float(torch.tensor(selected_distance).median()),
        }
    result = {
        "status": "validation_only_topk_oracle_diagnostic",
        "protocol": args.exp,
        "test_split_touched": False,
        "metrics": metrics,
        "topk_best_normalized_motion_distance": recall_distance,
        "interpretation": (
            "Oracle entries use validation GT only to select within the deployable "
            "KP5 score's top-K train candidates; they are upper bounds."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
