"""Validation-only anatomical motion-profile reranking for KP5 candidates."""

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
    PartMotionProfileHead,
    TemporalMotionSelector,
)
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_profile_reranking import (
    candidate_speed_profiles,
    predict_profile,
    profile_distance,
    standardize,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_csi_part_motion_profile import PART_NAMES, predict_part_profile
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


def candidate_part_speed_profiles(train_bank, pool, target_valid):
    profiles = []
    for item, valid in enumerate(target_valid):
        poses = torch.stack([
            _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            for index in pool["indices"][item]
        ])
        velocity = torch.zeros_like(poses)
        velocity[:, 1:] = (poses[:, 1:] - poses[:, :-1]) * C.TARGET_FPS
        parts = []
        for joints in C.JOINT_GROUPS.values():
            parts.append(torch.linalg.vector_norm(
                velocity[:, :, list(joints)], dim=-1
            ).mean(-1))
        profiles.append(torch.stack(parts, dim=-1))
    return torch.stack(profiles)


def part_profile_distance(predicted, candidates, valid):
    """Return scale and timing mismatch independently for each body region."""
    batch, choices, frames, parts = candidates.shape
    predicted = F.avg_pool1d(
        predicted.permute(0, 2, 1).reshape(batch * parts, 1, frames),
        9, stride=1, padding=4,
    ).reshape(batch, parts, frames).permute(0, 2, 1)
    candidates = F.avg_pool1d(
        candidates.permute(0, 1, 3, 2).reshape(batch * choices * parts, 1, frames),
        9, stride=1, padding=4,
    ).reshape(batch, choices, parts, frames).permute(0, 1, 3, 2)
    mask = valid[:, None, :, None]
    count = mask.sum(2).clamp_min(1)
    query_mean = (predicted[:, None] * mask).sum(2) / count
    candidate_mean = (candidates * mask).sum(2) / count
    query_centered = (predicted[:, None] - query_mean[:, :, None]) * mask
    candidate_centered = (candidates - candidate_mean[:, :, None]) * mask
    correlation = (query_centered * candidate_centered).sum(2) / (
        torch.linalg.vector_norm(query_centered, dim=2)
        * torch.linalg.vector_norm(candidate_centered, dim=2)
    ).clamp_min(1e-6)
    log_error = (
        torch.log1p(candidates) - torch.log1p(predicted[:, None])
    ).abs()
    log_error = (log_error * mask).sum(2) / count
    query_peak = predicted.masked_fill(~valid[..., None], -1).argmax(1)
    candidate_peak = candidates.masked_fill(~mask, -1).argmax(2)
    peak = (candidate_peak - query_peak[:, None]).abs() / count
    return (1.0 - correlation) + 0.45 * log_error + 0.15 * peak


def weighted_part_distance(values, pattern):
    weights = values.new_tensor(pattern)
    weights = weights / weights.sum()
    combined = (values * weights).sum(-1)
    return standardize(combined)


def load_models(args, device):
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
    scalar_checkpoint = torch.load(
        args.scalar_profile_checkpoint, map_location="cpu", weights_only=False
    )
    scalar = MotionProfileHead(**scalar_checkpoint["model_config"]).to(device)
    scalar.load_state_dict(scalar_checkpoint["model"])
    part_checkpoint = torch.load(
        args.part_profile_checkpoint, map_location="cpu", weights_only=False
    )
    part = PartMotionProfileHead(**part_checkpoint["model_config"]).to(device)
    part.load_state_dict(part_checkpoint["model"])
    for model in (selector, reranker, scalar, part):
        model.eval()
    return checkpoint, selector, reranker, scalar, part, part_checkpoint


@torch.no_grad()
def prepare(args, split, device, pool_top_k: int = 20):
    checkpoint, selector, reranker, scalar, part, part_checkpoint = load_models(
        args, device
    )
    root = args.selector_checkpoint.parent
    cache = torch.load(
        root / f"{split}_features.pt", map_location="cpu", weights_only=False
    )
    selector_output = predict_selector(selector, cache, 64, device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    target_set = QualityWeightedDataset(pose_only(datasets[split]), audit)
    _, _, train_class, _ = _load_pose_arrays(train)
    target_pose, target_valid, target_class, target_risk = _load_pose_arrays(target_set)
    inference_valid = cache["frame_mask"].bool()
    target_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(target_pose, target_valid)
    ])
    baseline = cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, inference_valid)
    ])
    train_bank = checkpoint["train_bank"].float()
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    self_indices = None
    if split == "train":
        # Training priors must never retrieve the trial whose GT trajectory is
        # the target. The approximate shortlist keeps this leave-self-out path
        # tractable without building a quadratic full-pose distance cache.
        distance = None
        self_indices = torch.arange(len(train_bank))
    else:
        distance = exact_pose_distance(
            baseline_bank, train_bank, root / f"{split}_exact_pose_distance.pt"
        )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, train_class, fused_action,
        top_k=pool_top_k, shortlist=max(100, pool_top_k),
        self_indices=self_indices,
        exact_distance_matrix=distance,
        action_penalty=float(getattr(args, "candidate_action_penalty", 0.05)),
    )
    logits = []
    for start in range(0, len(target_set), 64):
        indices = torch.arange(start, min(start + 64, len(target_set)))
        inputs = tuple(value.to(device) for value in model_inputs(
            pool, selector_output, checkpoint, risk_probability, indices,
        ))
        logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    predicted_scalar_profile = predict_profile(
        scalar, cache, inference_valid, device
    )
    candidate_scalar_profiles = candidate_speed_profiles(
        train_bank, pool, inference_valid
    )
    scalar_distance = standardize(profile_distance(
        predicted_scalar_profile,
        candidate_scalar_profiles,
        inference_valid,
    ))
    predicted_part_profile = predict_part_profile(
        part, cache, inference_valid, device
    )
    candidate_part_profiles = candidate_part_speed_profiles(
        train_bank, pool, inference_valid
    )
    part_distance = part_profile_distance(
        predicted_part_profile,
        candidate_part_profiles,
        inference_valid,
    )
    return {
        "checkpoint": checkpoint, "part_checkpoint": part_checkpoint,
        "cache": cache, "baseline": baseline, "baseline_bank": baseline_bank,
        "train_bank": train_bank, "fused_action": fused_action,
        "risk_probability": risk_probability,
        "target_pose": target_pose, "target_valid": target_valid,
        "inference_valid": inference_valid,
        "target_class": target_class, "target_risk": target_risk,
        "base_action_logits": cache["base_action_logits"].float(),
        "selector_embedding": selector_output["embedding"].float(),
        "selector_action_logits": selector_output["action_logits"].float(),
        "base_risk_logits": cache["base_risk_logits"].float(),
        "selector_risk_logits": selector_output["risk_logits"].float(),
        "pool": pool, "logits": logits,
        "scalar_distance": scalar_distance, "part_distance": part_distance,
        "predicted_scalar_profile": predicted_scalar_profile,
        "predicted_part_profile": predicted_part_profile,
        "candidate_scalar_profiles": candidate_scalar_profiles,
        "candidate_part_profiles": candidate_part_profiles,
    }


def render_mixture(data, adjusted, temperature, top_k):
    top = adjusted.topk(top_k, dim=-1).indices
    probability = torch.softmax(adjusted / temperature, dim=-1)
    weight = probability.gather(1, top)
    weight = weight / weight.sum(1, keepdim=True)
    motions = []
    inference_valid = data["inference_valid"]
    for item, valid in enumerate(inference_valid):
        bank_indices = data["pool"]["indices"][item].gather(0, top[item])
        canonical = (
            data["train_bank"].index_select(0, bank_indices)
            * weight[item, :, None, None, None]
        ).sum(0)
        motions.append(_render(canonical, valid, C.CACHE_FRAMES))
    return torch.stack(motions)


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
        "--scalar-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71" / "best_model.pt",
    )
    parser.add_argument(
        "--part-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed83"
        / "reranking_calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "val", device)
    patterns = {
        "uniform": (1, 1, 1, 1, 1, 1),
        "distal": (1.2, 0.7, 1.4, 1.4, 1.4, 1.4),
        "limbs": (0.8, 0.6, 1.7, 1.7, 1.7, 1.7),
    }
    metrics = {
        "locked_kp2_dh": _metric_batch(
            data["baseline"], data["target_pose"],
            data["target_valid"], data["target_risk"],
        )
    }
    base_adjusted = data["logits"] - 0.20 * data["scalar_distance"]
    for pattern_name, pattern in patterns.items():
        part_distance = weighted_part_distance(data["part_distance"], pattern)
        for part_weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
            if part_weight == 0.0 and pattern_name != "uniform":
                continue
            adjusted = base_adjusted - part_weight * part_distance
            for temperature in (0.35, 0.50, 0.75):
                for top_k in (3, 5):
                    candidate = render_mixture(data, adjusted, temperature, top_k)
                    for strength in (0.50, 0.625, 0.75):
                        key = (
                            f"p{int(part_weight * 1000):04d}_{pattern_name}"
                            f"_t{int(temperature * 100):03d}_top{top_k}"
                            f"_s{int(strength * 1000):03d}"
                        )
                        metrics[key] = _metric_batch(
                            (1.0 - strength) * data["baseline"] + strength * candidate,
                            data["target_pose"], data["target_valid"],
                            data["target_risk"],
                        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_part_motion_profile_reranking",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "part_order": PART_NAMES,
        "selection": {"name": best, "score": scores[best]},
        "scores": scores,
        "metrics": metrics,
        "part_profile_validation": data["part_checkpoint"]["selection"],
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
        "scalar_profile_checkpoint": report_path(args.scalar_profile_checkpoint),
        "part_profile_checkpoint": report_path(args.part_profile_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "selection": result["selection"], "metrics": metrics[best],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
