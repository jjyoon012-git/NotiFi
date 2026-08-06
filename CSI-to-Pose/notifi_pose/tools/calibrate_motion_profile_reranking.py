"""Validation-only motion-profile consistency reranking for KP5 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import (
    CandidateMotionReranker,
    MotionProfileHead,
    TemporalMotionSelector,
)
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_csi_motion_profile import profile_features
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


@torch.no_grad()
def predict_profile(model, cache, valid, device):
    values = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(model(
            profile_features(cache, indices).to(device),
            valid.index_select(0, indices).to(device),
        )["speed"].float().cpu())
    return torch.cat(values)


def candidate_speed_profiles(train_bank, pool, target_valid):
    profiles = []
    for item, valid in enumerate(target_valid):
        poses = torch.stack([
            _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            for index in pool["indices"][item]
        ])
        speed = torch.zeros(poses.shape[:2])
        speed[:, 1:] = torch.linalg.vector_norm(
            poses[:, 1:] - poses[:, :-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        profiles.append(speed)
    return torch.stack(profiles)


def profile_distance(predicted, candidates, valid):
    mask = valid[:, None]
    predicted = F.avg_pool1d(
        predicted[:, None], 9, stride=1, padding=4
    )[:, 0]
    candidates = F.avg_pool1d(
        candidates.flatten(0, 1)[:, None], 9, stride=1, padding=4
    )[:, 0].reshape_as(candidates)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    query_mean = (predicted[:, None] * mask).sum(-1, keepdim=True) / count
    candidate_mean = (candidates * mask).sum(-1, keepdim=True) / count
    query_centered = (predicted[:, None] - query_mean) * mask
    candidate_centered = (candidates - candidate_mean) * mask
    correlation = (query_centered * candidate_centered).sum(-1) / (
        torch.linalg.vector_norm(query_centered, dim=-1)
        * torch.linalg.vector_norm(candidate_centered, dim=-1)
    ).clamp_min(1e-6)
    log_error = (
        torch.log1p(candidates) - torch.log1p(predicted[:, None])
    ).abs()
    log_error = (log_error * mask).sum(-1) / count.squeeze(-1)
    query_peak = predicted.masked_fill(~valid, -1).argmax(-1)
    candidate_peak = candidates.masked_fill(~mask, -1).argmax(-1)
    peak = (candidate_peak - query_peak[:, None]).abs() / count.squeeze(-1)
    return (1.0 - correlation) + 0.50 * log_error + 0.20 * peak


def standardize(values):
    return (values - values.mean(-1, keepdim=True)) / values.std(
        -1, keepdim=True
    ).clamp_min(1e-5)


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
        "--profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71"
        / "reranking_calibration.json",
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
    profile_checkpoint = torch.load(
        args.profile_checkpoint, map_location="cpu", weights_only=False
    )
    profile_model = MotionProfileHead(**profile_checkpoint["model_config"]).to(device)
    profile_model.load_state_dict(profile_checkpoint["model"])
    profile_model.eval()
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
    logits = []
    with torch.no_grad():
        for start in range(0, len(validation), 64):
            indices = torch.arange(start, min(start + 64, len(validation)))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, checkpoint,
                risk_probability, indices,
            ))
            logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    predicted_profile = predict_profile(
        profile_model, cache, target_valid, device
    )
    candidate_profiles = candidate_speed_profiles(
        train_bank, pool, target_valid
    )
    motion_distance = standardize(profile_distance(
        predicted_profile, candidate_profiles, target_valid
    ))
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for motion_weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        adjusted = logits - motion_weight * motion_distance
        for temperature in (0.20, 0.35, 0.50, 0.75, 1.0):
            probability = torch.softmax(adjusted / temperature, dim=-1)
            for top_k in (2, 3, 5):
                top = adjusted.topk(top_k, dim=-1).indices
                weight = probability.gather(1, top)
                weight = weight / weight.sum(1, keepdim=True)
                motions = []
                for item, valid in enumerate(target_valid):
                    bank_indices = pool["indices"][item].gather(0, top[item])
                    canonical = (
                        train_bank.index_select(0, bank_indices)
                        * weight[item, :, None, None, None]
                    ).sum(0)
                    motions.append(_render(canonical, valid, C.CACHE_FRAMES))
                candidate = torch.stack(motions)
                key = (
                    f"m{int(motion_weight * 1000):04d}"
                    f"_t{int(temperature * 100):03d}_top{top_k}_625"
                )
                metrics[key] = _metric_batch(
                    0.375 * baseline + 0.625 * candidate,
                    target_pose, target_valid, target_risk,
                )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_motion_profile_reranking",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "profile_validation": profile_checkpoint["selection"],
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
        "profile_checkpoint": report_path(args.profile_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"],
        "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
