"""Train KP5-MPR-DM with the final blended 3D motion as its direct target."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import MotionMixturePredictor, TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import DISTAL_JOINTS, pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


def direct_motion_loss(predicted: torch.Tensor, target: torch.Tensor,
                       risk: torch.Tensor, probability: torch.Tensor,
                       entropy_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    distance = torch.linalg.vector_norm(predicted - target, dim=-1)
    overall = distance.mean((1, 2))
    distal = distance[:, :, list(DISTAL_JOINTS)].mean((1, 2))

    target_speed = torch.linalg.vector_norm(
        target[:, 1:] - target[:, :-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    threshold = torch.quantile(target_speed, 0.75, dim=1, keepdim=True)
    high_mask = (target_speed >= threshold) & (target_speed > 0.08)
    high_distance = distance[:, 1:].mean(-1)
    high = (high_distance * high_mask).sum(1) / high_mask.sum(1).clamp_min(1)

    predicted_velocity = (predicted[:, 1:] - predicted[:, :-1]) * C.TARGET_FPS
    target_velocity = (target[:, 1:] - target[:, :-1]) * C.TARGET_FPS
    velocity = torch.linalg.vector_norm(
        predicted_velocity - target_velocity, dim=-1
    ).mean((1, 2))

    edges = torch.as_tensor(C.SKELETON_EDGES, device=predicted.device)
    predicted_bones = torch.linalg.vector_norm(
        predicted[:, :, edges[:, 1]] - predicted[:, :, edges[:, 0]], dim=-1
    )
    target_bones = torch.linalg.vector_norm(
        target[:, :, edges[:, 1]] - target[:, :, edges[:, 0]], dim=-1
    )
    bone = (predicted_bones - target_bones).abs().mean((1, 2))

    sample = overall + 0.65 * distal + 0.35 * high + 0.08 * velocity + 0.20 * bone
    sample_weight = torch.where(
        risk == 2, torch.full_like(sample, 2.5), torch.ones_like(sample)
    )
    reconstruction = (sample * sample_weight).sum() / sample_weight.sum()
    entropy = -(probability.clamp_min(1e-7).log() * probability).sum(-1).mean()
    loss = reconstruction + float(entropy_weight) * entropy
    stats = {
        "loss": float(loss.detach()),
        "reconstruction": float(reconstruction.detach()),
        "overall": float(overall.mean().detach()),
        "distal": float(distal.mean().detach()),
        "high_motion": float(high.mean().detach()),
        "velocity": float(velocity.mean().detach()),
        "bone": float(bone.mean().detach()),
        "entropy": float(entropy.detach()),
    }
    return loss, stats


def predict_outputs(model, pool, selector_output, checkpoint,
                    risk_probability, device: str, batch_size: int = 64):
    model.eval()
    logits, strengths = [], []
    with torch.no_grad():
        for start in range(0, len(pool["indices"]), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(pool["indices"])))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, checkpoint, risk_probability, indices
            ))
            output = model(*inputs)
            logits.append(output["candidate_logits"].float().cpu())
            strengths.append(output["blend_strength"].float().cpu())
    return torch.cat(logits), torch.cat(strengths)


@torch.no_grad()
def evaluate(model, pool, selector_output, checkpoint, risk_probability,
             baseline: torch.Tensor, target_pose: torch.Tensor,
             target_valid: torch.Tensor, target_risk: torch.Tensor,
             device: str) -> dict:
    logits, learned_strength = predict_outputs(
        model, pool, selector_output, checkpoint, risk_probability, device
    )
    train_bank = checkpoint["train_bank"].float()
    candidate_bank = train_bank[pool["indices"]]
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for temperature in (0.20, 0.35, 0.50, 0.75, 1.0):
        probability = torch.softmax(logits / temperature, dim=-1)
        canonical = (probability[..., None, None, None] * candidate_bank).sum(1)
        candidate = torch.stack([
            _render(motion, valid, C.CACHE_FRAMES)
            for motion, valid in zip(canonical, target_valid)
        ])
        for mode, strength in (
            ("learned", learned_strength),
            ("fixed500", torch.full_like(learned_strength, 0.50)),
            ("fixed625", torch.full_like(learned_strength, 0.625)),
        ):
            predicted = (
                (1.0 - strength)[:, None, None, None] * baseline
                + strength[:, None, None, None] * candidate
            )
            key = f"t{int(temperature * 100):03d}_{mode}"
            metrics[key] = _metric_batch(
                predicted, target_pose, target_valid, target_risk
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "mean_learned_strength": float(learned_strength.mean()),
        "danger_learned_strength": float(learned_strength[target_risk == 2].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--initial-reranker", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_direct_mixture_seed43",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])
    feature_root = args.selector_checkpoint.parent
    train_cache = torch.load(
        feature_root / "train_features.pt", map_location="cpu", weights_only=False
    )
    val_cache = torch.load(
        feature_root / "val_features.pt", map_location="cpu", weights_only=False
    )
    train_output = predict_selector(selector, train_cache, 64, device)
    val_output = predict_selector(selector, val_cache, 64, device)
    del selector

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, train_class, train_risk = _load_pose_arrays(train)
    val_pose, val_valid, _, val_risk = _load_pose_arrays(validation)
    train_bank = checkpoint["train_bank"].float()
    val_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(val_pose, val_valid)
    ])
    train_baseline = train_cache["baseline_pose"].float()
    train_baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(train_baseline, train_valid)
    ])
    val_baseline = val_cache["baseline_pose"].float()
    val_baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(val_baseline, val_valid)
    ])
    train_fused_action = (
        train_cache["base_action_logits"].float() + train_output["action_logits"]
    )
    val_fused_action = (
        val_cache["base_action_logits"].float() + val_output["action_logits"]
    )
    train_risk_probability = torch.softmax(
        train_cache["base_risk_logits"].float() + train_output["risk_logits"], dim=-1
    )
    val_risk_probability = torch.softmax(
        val_cache["base_risk_logits"].float() + val_output["risk_logits"], dim=-1
    )
    train_pool = make_candidate_pool(
        train_baseline_bank, train_bank, train_risk,
        train_bank, train_class, train_fused_action,
        args.top_k, args.shortlist, self_indices=torch.arange(len(train_bank)),
    )
    val_distance = exact_pose_distance(
        val_baseline_bank, train_bank, feature_root / "val_exact_pose_distance.pt"
    )
    val_pool = make_candidate_pool(
        val_baseline_bank, val_bank, val_risk,
        train_bank, train_class, val_fused_action,
        args.top_k, args.shortlist, exact_distance_matrix=val_distance,
    )

    model_config = {
        "query_dim": train_output["pooled_features"].shape[-1],
        "embedding_dim": checkpoint["train_embedding"].shape[-1],
    }
    model = MotionMixturePredictor(**model_config).to(device)
    initial = torch.load(
        args.initial_reranker, map_location="cpu", weights_only=False
    )
    loaded = model.load_state_dict(initial["model"], strict=False)
    print(json.dumps({
        "initialized_from": report_path(args.initial_reranker),
        "missing_keys": loaded.missing_keys,
        "unexpected_keys": loaded.unexpected_keys,
    }), flush=True)

    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(2.5, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))), batch_size=args.batch_size,
        sampler=sampler, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (
            1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_stats = []
        for (indices,) in loader:
            indices = indices.long()
            inputs = tuple(value.to(device) for value in model_inputs(
                train_pool, train_output, checkpoint,
                train_risk_probability, indices,
            ))
            candidate = train_bank[train_pool["indices"].index_select(0, indices)].to(device)
            baseline = train_baseline_bank.index_select(0, indices).to(device)
            target = train_bank.index_select(0, indices).to(device)
            risk = train_risk.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                output = model(*inputs)
                probability = torch.softmax(output["candidate_logits"] / 0.50, dim=-1)
                mixture = (probability[..., None, None, None] * candidate).sum(1)
                strength = output["blend_strength"][:, None, None, None]
                predicted = (1.0 - strength) * baseline + strength * mixture
                loss, stats = direct_motion_loss(
                    predicted, target, risk, probability, args.entropy_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_stats.append(stats)
        validation_result = evaluate(
            model, val_pool, val_output, checkpoint, val_risk_probability,
            val_baseline, val_pose, val_valid, val_risk, device,
        )
        train_stats = {
            key: float(np.mean([row[key] for row in epoch_stats]))
            for key in epoch_stats[0]
        }
        row = {
            "epoch": epoch,
            "train": train_stats,
            "validation_selection": validation_result["selection"],
            "mean_strength": validation_result["mean_learned_strength"],
            "danger_strength": validation_result["danger_learned_strength"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = validation_result["selection"]["score"]
        if best is None or score < best["score"] - 1e-5:
            best = {
                "epoch": epoch,
                "score": score,
                "state": copy.deepcopy(model.state_dict()),
                "validation": validation_result,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best is not None
    result = {
        "run": "KP5-MPR-DM-EXP01",
        "candidate_version": "KP5-MPR-DM",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "selector_checkpoint": report_path(args.selector_checkpoint),
            "initial_reranker": report_path(args.initial_reranker),
            "run_dir": report_path(args.run_dir),
        },
        "architecture": {
            "candidate_pool": f"leave-self-out top-{args.top_k}",
            "objective": "final blended canonical pose, distal, high-motion, velocity, bone",
            "mixture": "CSI-conditioned candidate softmax and bounded blend strength",
        },
        "selection": {
            "epoch": best["epoch"],
            **best["validation"]["selection"],
        },
        "validation": best["validation"],
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"],
        "protocol": args.exp,
        "model": best["state"],
        "model_config": model_config,
        "selection": result["selection"],
        "selector_checkpoint": report_path(args.selector_checkpoint),
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
