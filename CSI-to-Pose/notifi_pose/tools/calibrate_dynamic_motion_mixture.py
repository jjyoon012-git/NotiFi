"""Restore high-motion detail in the KP5 top-3 motion mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

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


def frame_activity(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    speed = pose.new_zeros(pose.shape[:2])
    delta = torch.linalg.vector_norm(pose[:, 1:] - pose[:, :-1], dim=-1).mean(-1)
    speed[:, 1:] = delta
    speed = F.avg_pool1d(speed[:, None], 9, stride=1, padding=4).squeeze(1)
    rows = []
    for values, mask in zip(speed, valid):
        scale = torch.quantile(values[mask], 0.80).clamp_min(1e-5)
        rows.append((values / scale).clamp(0.0, 1.0) * mask)
    return torch.stack(rows)


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
        / "dynamic_mixture_calibration.json",
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
    probability = torch.softmax(logits / 0.50, dim=-1)
    top3 = logits.topk(3, dim=-1).indices
    top3_probability = probability.gather(1, top3)
    top3_probability = top3_probability / top3_probability.sum(1, keepdim=True)
    top1_candidates, mixture_candidates = [], []
    for item, valid in enumerate(target_valid):
        local = top3[item]
        bank_indices = pool["indices"][item].gather(0, local)
        motions = train_bank.index_select(0, bank_indices)
        top1_candidates.append(_render(motions[0], valid, C.CACHE_FRAMES))
        mixture = (
            motions * top3_probability[item, :, None, None, None]
        ).sum(0)
        mixture_candidates.append(_render(mixture, valid, C.CACHE_FRAMES))
    top1 = torch.stack(top1_candidates)
    mixture = torch.stack(mixture_candidates)
    activity = frame_activity(baseline, target_valid)[..., None, None]
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "mixture_fixed": _metric_batch(
            0.375 * baseline + 0.625 * mixture,
            target_pose, target_valid, target_risk,
        ),
    }
    for gain in (0.125, 0.25, 0.375, 0.50):
        candidate = mixture + gain * activity * (top1 - mixture)
        pose = 0.375 * baseline + 0.625 * candidate
        metrics[f"activity_top1_{int(gain * 1000):03d}"] = _metric_batch(
            pose, target_pose, target_valid, target_risk
        )
    for reduction in (0.0625, 0.125, 0.1875, 0.25):
        strength = (0.625 - reduction * activity).clamp(0.25, 0.75)
        pose = (1.0 - strength) * baseline + strength * mixture
        metrics[f"activity_baseline_{int(reduction * 1000):03d}"] = _metric_batch(
            pose, target_pose, target_valid, target_risk
        )
    disagreement = torch.linalg.vector_norm(top1 - mixture, dim=-1).mean(-1)
    rows = []
    for values, valid in zip(disagreement, target_valid):
        scale = torch.quantile(values[valid], 0.80).clamp_min(1e-5)
        rows.append((values / scale).clamp(0.0, 1.0) * valid)
    agreement = (1.0 - torch.stack(rows))[..., None, None]
    for gain in (0.125, 0.25, 0.375):
        candidate = mixture + gain * activity * agreement * (top1 - mixture)
        pose = 0.375 * baseline + 0.625 * candidate
        metrics[f"agreement_top1_{int(gain * 1000):03d}"] = _metric_batch(
            pose, target_pose, target_valid, target_risk
        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_dynamic_mixture",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "activity_source": "locked CSI pose speed only",
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
