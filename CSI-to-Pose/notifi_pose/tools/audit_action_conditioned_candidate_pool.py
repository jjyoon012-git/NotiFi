"""Validation-only audit of action conditioning in the retrieval candidate pool."""

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
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool
from .train_motion_retrieval_selector import predict_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "action_conditioned_pool_audit.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])
    root = args.selector_checkpoint.parent
    cache = torch.load(
        root / "val_features.pt", map_location="cpu", weights_only=False
    )
    output = predict_selector(selector, cache, 64, device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    _, _, train_class, _ = _load_pose_arrays(train)
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
    fused_action = cache["base_action_logits"].float() + output["action_logits"]
    distance = exact_pose_distance(
        baseline_bank, train_bank, root / "val_exact_pose_distance.pt"
    )
    results = {}
    for penalty in (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        for top_k in (5, 10, 20, 50):
            pool = make_candidate_pool(
                baseline_bank, target_bank, target_risk,
                train_bank, train_class, fused_action,
                top_k=top_k, shortlist=100,
                exact_distance_matrix=distance,
                action_penalty=penalty,
            )
            local = pool["target_cost"].argmin(-1)
            selected = pool["indices"].gather(1, local[:, None]).squeeze(1)
            candidate = torch.stack([
                _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
                for index, valid in zip(selected, target_valid)
            ])
            metric = _metric_batch(
                0.375 * baseline + 0.625 * candidate,
                target_pose, target_valid, target_risk,
            )
            name = f"a{int(penalty * 1000):04d}_top{top_k}"
            results[name] = {
                "score": pose_selection_score(metric),
                "mean_oracle_cost": float(
                    pool["target_cost"].gather(1, local[:, None]).mean()
                ),
                "metrics": metric,
            }
    best = min(results, key=lambda name: results[name]["score"])
    result = {
        "status": "validation_only_candidate_pool_audit",
        "test_used_for_selection": False,
        "selection": {"name": best, **results[best]},
        "results": results,
        "selector_checkpoint": report_path(args.selector_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
