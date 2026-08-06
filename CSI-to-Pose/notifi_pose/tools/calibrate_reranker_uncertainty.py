"""Validation-only uncertainty and multi-hypothesis calibration for KP5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import CandidateMotionReranker, TemporalMotionSelector
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
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "uncertainty_calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selector_checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**selector_checkpoint["model_config"]).to(device)
    selector.load_state_dict(selector_checkpoint["model"])
    reranker_checkpoint = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    reranker = CandidateMotionReranker(
        **reranker_checkpoint["model_config"]
    ).to(device)
    reranker.load_state_dict(reranker_checkpoint["model"])
    reranker.eval()
    root = args.selector_checkpoint.parent
    cache = torch.load(root / "val_features.pt", map_location="cpu", weights_only=False)
    selector_output = predict_selector(selector, cache, 64, device)
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
    train_bank = selector_checkpoint["train_bank"].float()
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    distance = exact_pose_distance(
        baseline_bank, train_bank, root / "val_exact_pose_distance.pt"
    )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, selector_checkpoint["train_class"].long(), fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
    )
    logits = []
    with torch.no_grad():
        for start in range(0, len(validation), 64):
            indices = torch.arange(start, min(start + 64, len(validation)))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, selector_checkpoint,
                risk_probability, indices,
            ))
            logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    winner = logits.argmax(-1)
    winner_indices = pool["indices"].gather(1, winner[:, None]).squeeze(1)
    winner_pose = torch.stack([
        _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
        for index, valid in zip(winner_indices, target_valid)
    ])
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "argmax_fixed_500": _metric_batch(
            0.5 * baseline + 0.5 * winner_pose,
            target_pose, target_valid, target_risk,
        ),
    }

    for temperature in (0.10, 0.20, 0.50, 1.0):
        probability = torch.softmax(logits / temperature, dim=-1)
        confidence = probability.max(-1).values
        confidence = (
            (confidence - torch.quantile(confidence, 0.10))
            / (torch.quantile(confidence, 0.90) - torch.quantile(confidence, 0.10)).clamp_min(1e-5)
        ).clamp(0.0, 1.0)
        for gain in (0.125, 0.25):
            strength = (0.50 + gain * (confidence - 0.50)).clamp(0.25, 0.75)
            pose = (
                (1.0 - strength)[:, None, None, None] * baseline
                + strength[:, None, None, None] * winner_pose
            )
            key = f"confidence_t{int(temperature * 100):03d}_g{int(gain * 1000):03d}"
            metrics[key] = _metric_batch(
                pose, target_pose, target_valid, target_risk
            )
        for top_k in (2, 3, 5):
            top = logits.topk(top_k, dim=-1).indices
            top_probability = probability.gather(1, top)
            top_probability = top_probability / top_probability.sum(1, keepdim=True)
            motions = []
            for item, valid in enumerate(target_valid):
                bank_indices = pool["indices"][item].gather(0, top[item])
                canonical = (
                    train_bank.index_select(0, bank_indices)
                    * top_probability[item, :, None, None, None]
                ).sum(0)
                motions.append(_render(canonical, valid, C.CACHE_FRAMES))
            candidate = torch.stack(motions)
            for strength in (0.375, 0.50, 0.625):
                key = (
                    f"mixture_t{int(temperature * 100):03d}_top{top_k}"
                    f"_{int(strength * 1000):03d}"
                )
                metrics[key] = _metric_batch(
                    (1.0 - strength) * baseline + strength * candidate,
                    target_pose, target_valid, target_risk,
                )
    danger_probability = risk_probability[:, 2]
    for gain in (0.0625, 0.125, 0.1875):
        strength = (0.50 + gain * danger_probability).clamp_max(0.75)
        pose = (
            (1.0 - strength)[:, None, None, None] * baseline
            + strength[:, None, None, None] * winner_pose
        )
        key = f"danger_adaptive_{int(gain * 1000):03d}"
        metrics[key] = _metric_batch(
            pose, target_pose, target_valid, target_risk
        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_uncertainty_calibration",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
