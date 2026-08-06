"""Validation-only calibration of exact KP5 motion retrieval and blending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from ..motion_tokens import forward_kinematics, pose_to_bones, trial_bone_lengths
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_retrieval_selector import predict_selector


def exact_pose_distance(queries: torch.Tensor, bank: torch.Tensor,
                        path: Path) -> torch.Tensor:
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(cached.shape) == (len(queries), len(bank)):
            return cached.float()
    distances = []
    for query in queries:
        distances.append(
            torch.linalg.vector_norm(bank - query[None], dim=-1).mean((1, 2))
        )
    result = torch.stack(distances)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.half(), path)
    return result


def select_indices(pose_distance: torch.Tensor, latent_distance: torch.Tensor,
                   bank_class: torch.Tensor, logits: torch.Tensor,
                   mode: str) -> torch.Tensor:
    selected = []
    for item in range(len(pose_distance)):
        pose = pose_distance[item]
        latent = latent_distance[item]
        probability = torch.softmax(logits[item], dim=-1).clamp_min(1e-6)
        keep = torch.ones_like(bank_class, dtype=torch.bool)
        if "top" in mode:
            top_k = int(mode.rsplit("top", 1)[-1])
            classes = logits[item].topk(top_k).indices
            keep = (bank_class[:, None] == classes[None]).any(-1)
            if not keep.any():
                keep = torch.ones_like(bank_class, dtype=torch.bool)
        pose_scale = torch.quantile(pose[keep], 0.50).clamp_min(1e-5)
        if mode.startswith("latent"):
            score = latent
        elif mode.startswith("hybrid"):
            alpha = 0.25
            latent_scale = torch.quantile(latent[keep], 0.50).clamp_min(1e-5)
            score = (1.0 - alpha) * pose / pose_scale + alpha * latent / latent_scale
        elif mode.startswith("soft"):
            penalty = float(mode.rsplit("_", 1)[-1]) / 100.0
            score = pose / pose_scale - penalty * probability[bank_class].log()
        else:
            score = pose
        selected.append(score.masked_fill(~keep, float("inf")).argmin())
    return torch.stack(selected)


def kinematic_blend(baseline: torch.Tensor, candidate: torch.Tensor,
                    valid: torch.Tensor, strength: float) -> torch.Tensor:
    baseline_direction, _ = pose_to_bones(baseline)
    candidate_direction, _ = pose_to_bones(candidate)
    direction = F.normalize(
        (1.0 - strength) * baseline_direction
        + strength * candidate_direction,
        dim=-1,
    )
    direction[:, :, C.ROOT_JOINT] = 0.0
    lengths = trial_bone_lengths(baseline, valid)
    pose = forward_kinematics(direction, lengths)
    return pose * valid[..., None, None].to(pose.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=48)
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
        / "deployment_calibration.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    feature_cache = torch.load(
        args.feature_cache, map_location="cpu", weights_only=False
    )
    output = predict_selector(model, feature_cache, args.batch_size, device)

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    target_pose, target_valid, target_class, target_risk = _load_pose_arrays(validation)
    baseline = feature_cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, target_valid)
    ])
    train_bank = checkpoint["train_bank"].float()
    train_class = checkpoint["train_class"].long()
    train_embedding = checkpoint["train_embedding"].float()
    pose_distance = exact_pose_distance(
        baseline_bank, train_bank, args.distance_cache
    )
    latent_distance = torch.cdist(output["embedding"], train_embedding)
    base_logits = feature_cache["base_action_logits"].float()
    selector_logits = output["action_logits"]
    fused_logits = base_logits + selector_logits

    configurations = {
        "pose_global": ("pose", base_logits),
        "pose_base_top1": ("pose_top1", base_logits),
        "pose_base_top2": ("pose_top2", base_logits),
        "pose_selector_top1": ("pose_top1", selector_logits),
        "pose_selector_top2": ("pose_top2", selector_logits),
        "pose_fused_top1": ("pose_top1", fused_logits),
        "pose_fused_top2": ("pose_top2", fused_logits),
        "pose_fused_top3": ("pose_top3", fused_logits),
        "latent_fused_top2": ("latent_top2", fused_logits),
        "hybrid_fused_top2": ("hybrid_top2", fused_logits),
        "soft_fused_05": ("soft_05", fused_logits),
        "soft_fused_10": ("soft_10", fused_logits),
        "soft_fused_20": ("soft_20", fused_logits),
    }
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    selected = {}
    for name, (mode, logits) in configurations.items():
        indices = select_indices(
            pose_distance, latent_distance, train_class, logits, mode
        )
        selected[name] = indices
        candidate = torch.stack([
            _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            for index, valid in zip(indices, target_valid)
        ])
        for strength in (0.25, 0.375, 0.50, 0.625, 0.75):
            suffix = f"{int(strength * 1000):03d}"
            cartesian = (1.0 - strength) * baseline + strength * candidate
            metrics[f"{name}_cartesian_{suffix}"] = _metric_batch(
                cartesian, target_pose, target_valid, target_risk
            )
            kinematic = kinematic_blend(
                baseline, candidate, target_valid, strength
            )
            metrics[f"{name}_kinematic_{suffix}"] = _metric_batch(
                kinematic, target_pose, target_valid, target_risk
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    best_config = best_name.rsplit("_", 2)[0]
    result = {
        "status": "validation_selected_deployment_calibration",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best_name, "score": scores[best_name]},
        "metrics": metrics,
        "scores": scores,
        "classification": {
            "base_action_accuracy": float(
                (base_logits.argmax(-1) == target_class).float().mean()
            ),
            "selector_action_accuracy": float(
                (selector_logits.argmax(-1) == target_class).float().mean()
            ),
            "fused_action_accuracy": float(
                (fused_logits.argmax(-1) == target_class).float().mean()
            ),
        },
        "checkpoint": report_path(args.checkpoint),
        "selected_train_rows": checkpoint["train_rows"].index_select(
            0, selected[best_config]
        ).tolist(),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
