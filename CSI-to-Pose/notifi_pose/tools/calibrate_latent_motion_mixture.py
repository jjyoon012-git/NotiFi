"""Decode KP5's top-3 hypotheses through the frozen continuous motion prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import CandidateMotionReranker, TemporalMotionSelector
from ..motion_tokens import trial_bone_lengths
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_continuous_pose import load_teacher
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
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2b_continuous_motion_autoencoder"
        / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "latent_mixture_calibration.json",
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
    teacher, motion_architecture = load_teacher(args.motion_checkpoint, device)
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
    top_probability = probability.gather(1, top3)
    top_probability = top_probability / top_probability.sum(1, keepdim=True)
    raw_rows = []
    latent_rows = {0.0: [], 0.25: [], 0.50: []}
    with torch.no_grad():
        for item, valid in enumerate(target_valid):
            bank_indices = pool["indices"][item].gather(0, top3[item])
            canonical = train_bank.index_select(0, bank_indices)
            rendered = torch.stack([
                _render(motion, valid, C.CACHE_FRAMES) for motion in canonical
            ])
            weights = top_probability[item]
            raw_rows.append(
                (rendered * weights[:, None, None, None]).sum(0)
            )
            valid_batch = valid[None].expand(3, -1).to(device)
            encoded = teacher.encode(rendered.to(device), valid_batch)
            latent = encoded["latent"]
            weighted = (
                latent * weights.to(device)[:, None, None]
            ).sum(0, keepdim=True)
            lengths = trial_bone_lengths(
                baseline[item:item + 1].to(device), valid[None].to(device)
            )
            for top1_strength in latent_rows:
                mixed = (
                    (1.0 - top1_strength) * weighted
                    + top1_strength * latent[:1]
                )
                decoded = teacher.decode(
                    mixed, lengths, C.CACHE_FRAMES, valid[None].to(device)
                )["pose_rel"][0]
                latent_rows[top1_strength].append(decoded.cpu())
    raw = torch.stack(raw_rows)
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "raw_mixture_625": _metric_batch(
            0.375 * baseline + 0.625 * raw,
            target_pose, target_valid, target_risk,
        ),
    }
    for top1_strength, rows in latent_rows.items():
        candidate = torch.stack(rows)
        label = int(top1_strength * 1000)
        for strength in (0.375, 0.50, 0.625, 0.75):
            key = f"latent_top1_{label:03d}_blend_{int(strength * 1000):03d}"
            metrics[key] = _metric_batch(
                (1.0 - strength) * baseline + strength * candidate,
                target_pose, target_valid, target_risk,
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_latent_mixture",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "motion_prior": {
            "checkpoint": report_path(args.motion_checkpoint),
            "architecture": motion_architecture,
        },
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
