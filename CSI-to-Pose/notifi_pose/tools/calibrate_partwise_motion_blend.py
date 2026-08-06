"""Validation-only body-part blend calibration for the promoted KP5 mixture."""

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


def build_candidate(selector_checkpoint, reranker, selector_output, cache,
                    pool, risk_probability, target_valid, device):
    logits = []
    with torch.no_grad():
        for start in range(0, len(target_valid), 64):
            indices = torch.arange(start, min(start + 64, len(target_valid)))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, selector_checkpoint,
                risk_probability, indices,
            ))
            logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    top = logits.topk(3, dim=-1).indices
    probability = torch.softmax(logits / 0.50, dim=-1)
    weight = probability.gather(1, top)
    weight = weight / weight.sum(1, keepdim=True)
    train_bank = selector_checkpoint["train_bank"].float()
    motions = []
    for item, valid in enumerate(target_valid):
        bank_indices = pool["indices"][item].gather(0, top[item])
        canonical = (
            train_bank.index_select(0, bank_indices)
            * weight[item, :, None, None, None]
        ).sum(0)
        motions.append(_render(canonical, valid, C.CACHE_FRAMES))
    return torch.stack(motions)


def apply_strength(baseline, candidate, strengths):
    predicted = baseline.clone()
    for name, joints in C.JOINT_GROUPS.items():
        strength = float(strengths[name])
        predicted[:, :, list(joints)] = (
            (1.0 - strength) * baseline[:, :, list(joints)]
            + strength * candidate[:, :, list(joints)]
        )
    return predicted


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
        / "partwise_blend_calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])
    reranker_checkpoint = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    reranker = CandidateMotionReranker(**reranker_checkpoint["model_config"]).to(device)
    reranker.load_state_dict(reranker_checkpoint["model"])
    reranker.eval()
    root = args.selector_checkpoint.parent
    cache = torch.load(root / "val_features.pt", map_location="cpu", weights_only=False)
    selector_output = predict_selector(selector, cache, 64, device)
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
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    distance = exact_pose_distance(
        baseline_bank, train_bank, root / "val_exact_pose_distance.pt"
    )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, train_class, fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
    )
    candidate = build_candidate(
        checkpoint, reranker, selector_output, cache, pool,
        risk_probability, target_valid, device,
    )
    options = (0.375, 0.50, 0.625, 0.75)
    strengths = {name: 0.625 for name in C.JOINT_GROUPS}
    trace = []
    for sweep in range(2):
        for name in C.JOINT_GROUPS:
            choices = {}
            for option in options:
                trial = strengths | {name: option}
                metrics = _metric_batch(
                    apply_strength(baseline, candidate, trial),
                    target_pose, target_valid, target_risk,
                )
                choices[str(option)] = {
                    "score": pose_selection_score(metrics),
                    "metrics": metrics,
                }
            selected = min(choices, key=lambda key: choices[key]["score"])
            strengths[name] = float(selected)
            trace.append({
                "sweep": sweep + 1, "part": name,
                "selected": float(selected), "choices": choices,
            })
    final_metrics = _metric_batch(
        apply_strength(baseline, candidate, strengths),
        target_pose, target_valid, target_risk,
    )
    uniform_metrics = _metric_batch(
        0.375 * baseline + 0.625 * candidate,
        target_pose, target_valid, target_risk,
    )
    result = {
        "status": "validation_selected_partwise_blend",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "fixed_candidate": "KP5-MPR-R top3 temperature0.50",
        "selection": {
            "strengths": strengths,
            "score": pose_selection_score(final_metrics),
        },
        "metrics": {
            "uniform_625": uniform_metrics,
            "partwise": final_metrics,
        },
        "scores": {
            "uniform_625": pose_selection_score(uniform_metrics),
            "partwise": pose_selection_score(final_metrics),
        },
        "trace": trace,
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"],
        "scores": result["scores"],
        "metrics": result["metrics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
