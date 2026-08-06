"""Train KP5-MPR-R: a listwise CSI-conditioned top-K motion reranker."""

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
from ..motion_retrieval import CandidateMotionReranker, TemporalMotionSelector
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
from .train_motion_retrieval_selector import predict_selector


def compact_position(bank: torch.Tensor, bins: int = 19) -> torch.Tensor:
    return F.adaptive_avg_pool1d(
        bank.flatten(2).transpose(1, 2), bins
    ).flatten(1)


def candidate_cost(candidates: torch.Tensor, target: torch.Tensor,
                   danger: bool, baseline: torch.Tensor | None = None,
                   blend_strength: float | None = None) -> torch.Tensor:
    predicted = candidates
    if baseline is not None and blend_strength is not None:
        predicted = (
            (1.0 - float(blend_strength)) * baseline[None]
            + float(blend_strength) * candidates
        )
    error = torch.linalg.vector_norm(predicted - target[None], dim=-1)
    overall = error.mean((1, 2))
    distal = error[:, :, DISTAL_JOINTS].mean((1, 2))
    speed = torch.linalg.vector_norm(
        target[1:] - target[:-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    high = speed >= torch.quantile(speed, 0.75)
    high &= speed > 0.08
    high_error = (
        error[:, 1:, :][:, high].mean((1, 2)) if high.any() else overall
    )
    cost = overall + 0.45 * distal + 0.30 * high_error
    if danger:
        cost = cost + 0.65 * overall + 0.55 * distal + 0.45 * high_error
    return cost


def make_candidate_pool(
    baseline_bank: torch.Tensor,
    target_bank: torch.Tensor,
    risk: torch.Tensor,
    train_bank: torch.Tensor,
    train_class: torch.Tensor,
    fused_logits: torch.Tensor,
    top_k: int,
    shortlist: int,
    self_indices: torch.Tensor | None = None,
    exact_distance_matrix: torch.Tensor | None = None,
    target_blend_strength: float | None = None,
    action_penalty: float = 0.05,
) -> dict[str, torch.Tensor]:
    probability = torch.softmax(fused_logits, dim=-1).clamp_min(1e-6)
    if exact_distance_matrix is None:
        approximate = torch.cdist(
            compact_position(baseline_bank), compact_position(train_bank)
        )
    else:
        approximate = None
    all_indices, all_scores, all_log_probability, all_cost = [], [], [], []
    for item in range(len(baseline_bank)):
        if exact_distance_matrix is not None:
            candidate_indices = torch.arange(len(train_bank))
            exact = exact_distance_matrix[item].clone()
        else:
            approx = approximate[item].clone()
            if self_indices is not None:
                approx[int(self_indices[item])] = float("inf")
            candidate_indices = approx.topk(
                min(shortlist, len(train_bank) - 1), largest=False
            ).indices
            exact = torch.linalg.vector_norm(
                train_bank.index_select(0, candidate_indices)
                - baseline_bank[item][None], dim=-1,
            ).mean((1, 2))
        if self_indices is not None:
            exact[candidate_indices == int(self_indices[item])] = float("inf")
        finite = torch.isfinite(exact)
        scale = torch.quantile(exact[finite], 0.50).clamp_min(1e-5)
        action_log = probability[item, train_class.index_select(0, candidate_indices)].log()
        score = exact / scale - float(action_penalty) * action_log
        local = score.topk(top_k, largest=False).indices
        indices = candidate_indices.index_select(0, local)
        all_indices.append(indices)
        all_scores.append(score.index_select(0, local))
        all_log_probability.append(action_log.index_select(0, local))
        all_cost.append(candidate_cost(
            train_bank.index_select(0, indices), target_bank[item],
            int(risk[item]) == 2,
            baseline=baseline_bank[item],
            blend_strength=target_blend_strength,
        ))
    return {
        "indices": torch.stack(all_indices),
        "retrieval_score": torch.stack(all_scores),
        "action_log_probability": torch.stack(all_log_probability),
        "target_cost": torch.stack(all_cost),
    }


def model_inputs(pool: dict, selector_output: dict, checkpoint: dict,
                 risk_probability: torch.Tensor,
                 indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    candidate_indices = pool["indices"].index_select(0, indices)
    return (
        selector_output["pooled_features"].index_select(0, indices),
        selector_output["embedding"].index_select(0, indices),
        checkpoint["train_embedding"][candidate_indices],
        checkpoint["train_class"][candidate_indices],
        risk_probability.index_select(0, indices),
        pool["retrieval_score"].index_select(0, indices),
        pool["action_log_probability"].index_select(0, indices),
    )


def train_epoch(model, loader, pool, selector_output, checkpoint,
                risk_probability, optimizer, scaler, scheduler,
                device: str) -> dict:
    model.train()
    losses = []
    for (indices,) in loader:
        indices = indices.long()
        inputs = tuple(value.to(device) for value in model_inputs(
            pool, selector_output, checkpoint, risk_probability, indices
        ))
        target_cost = pool["target_cost"].index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(*inputs)
            temperature = 0.020
            target_probability = torch.softmax(-target_cost / temperature, dim=-1)
            listwise = -(target_probability * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            hard = F.cross_entropy(logits, target_cost.argmin(-1))
            best = logits.gather(1, target_cost.argmin(-1, keepdim=True)).squeeze(1)
            worst = logits.gather(1, target_cost.argmax(-1, keepdim=True)).squeeze(1)
            margin = F.relu(0.30 - best + worst).mean()
            loss = listwise + 0.25 * hard + 0.10 * margin
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append((float(loss), float(listwise), float(hard), float(margin)))
    values = np.asarray(losses)
    return {
        "total": float(values[:, 0].mean()),
        "listwise": float(values[:, 1].mean()),
        "hard": float(values[:, 2].mean()),
        "margin": float(values[:, 3].mean()),
    }


@torch.no_grad()
def evaluate(model, pool, selector_output, checkpoint, risk_probability,
             baseline: torch.Tensor, target_pose: torch.Tensor,
             target_valid: torch.Tensor, target_risk: torch.Tensor,
             device: str) -> dict:
    model.eval()
    logits = []
    batch_size = 64
    for start in range(0, len(baseline), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(baseline)))
        inputs = tuple(value.to(device) for value in model_inputs(
            pool, selector_output, checkpoint, risk_probability, indices
        ))
        logits.append(model(*inputs).float().cpu())
    logits = torch.cat(logits)
    local = logits.argmax(-1)
    selected = pool["indices"].gather(1, local[:, None]).squeeze(1)
    candidate = torch.stack([
        _render(checkpoint["train_bank"][int(index)].float(), valid, C.CACHE_FRAMES)
        for index, valid in zip(selected, target_valid)
    ])
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for strength in (0.25, 0.375, 0.50, 0.625):
        name = f"reranked_cartesian_{int(strength * 1000):03d}"
        metrics[name] = _metric_batch(
            (1.0 - strength) * baseline + strength * candidate,
            target_pose, target_valid, target_risk,
        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    target_cost = pool["target_cost"]
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "top1_oracle_accuracy": float(
            (local == target_cost.argmin(-1)).float().mean()
        ),
        "mean_selected_cost": float(
            target_cost.gather(1, local[:, None]).mean()
        ),
        "mean_retrieval_top1_cost": float(target_cost[:, 0].mean()),
        "selected_bank_indices": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--target-blend-strength", type=float, default=None)
    parser.add_argument("--action-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17",
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

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
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
    train_fused_action = train_cache["base_action_logits"].float() + train_output["action_logits"]
    val_fused_action = val_cache["base_action_logits"].float() + val_output["action_logits"]
    train_risk_probability = torch.softmax(
        train_cache["base_risk_logits"].float() + train_output["risk_logits"], dim=-1
    )
    val_risk_probability = torch.softmax(
        val_cache["base_risk_logits"].float() + val_output["risk_logits"], dim=-1
    )
    train_pool = make_candidate_pool(
        train_baseline_bank, train_bank, train_risk,
        train_bank, train_class, train_fused_action,
        args.top_k, args.shortlist,
        self_indices=torch.arange(len(train_bank)),
        target_blend_strength=args.target_blend_strength,
        action_penalty=args.action_penalty,
    )
    val_distance = exact_pose_distance(
        val_baseline_bank, train_bank,
        feature_root / "val_exact_pose_distance.pt",
    )
    val_pool = make_candidate_pool(
        val_baseline_bank, val_bank, val_risk,
        train_bank, train_class, val_fused_action,
        args.top_k, args.shortlist,
        exact_distance_matrix=val_distance,
        target_blend_strength=args.target_blend_strength,
        action_penalty=args.action_penalty,
    )

    model = CandidateMotionReranker(
        query_dim=train_output["pooled_features"].shape[-1],
        embedding_dim=checkpoint["train_embedding"].shape[-1],
    ).to(device)
    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(args.danger_weight, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))),
        batch_size=args.batch_size, sampler=sampler, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        trained = train_epoch(
            model, loader, train_pool, train_output, checkpoint,
            train_risk_probability, optimizer, scaler, scheduler, device,
        )
        validation_result = evaluate(
            model, val_pool, val_output, checkpoint, val_risk_probability,
            val_baseline, val_pose, val_valid, val_risk, device,
        )
        row = {
            "epoch": epoch, "train": trained,
            "validation_selection": validation_result["selection"],
            "top1_oracle_accuracy": validation_result["top1_oracle_accuracy"],
            "selected_cost": validation_result["mean_selected_cost"],
            "retrieval_top1_cost": validation_result["mean_retrieval_top1_cost"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = validation_result["selection"]["score"]
        if best is None or score < best["score"] - 1e-5:
            best = {
                "epoch": epoch, "score": score,
                "state": copy.deepcopy(model.state_dict()),
                "validation": validation_result,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best is not None
    selected = best["validation"].pop("selected_bank_indices")
    result = {
        "run": "KP5-MPR-R-EXP01",
        "model_family": "NotiFi-KP5",
        "candidate_version": "KP5-MPR-R",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": vars(args) | {
            "selector_checkpoint": report_path(args.selector_checkpoint),
            "run_dir": report_path(args.run_dir),
        },
        "architecture": {
            "candidate_pool": f"leave-self-out top-{args.top_k}",
            "reranker": "CSI query plus motion embedding listwise MLP",
            "target": "whole, distal, high-motion candidate cost",
            "target_blend_strength": args.target_blend_strength,
        },
        "selection": {
            "epoch": best["epoch"], "score": best["score"],
            "name": best["validation"]["selection"]["name"],
        },
        "validation": best["validation"],
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "model": best["state"],
        "model_config": {
            "query_dim": train_output["pooled_features"].shape[-1],
            "embedding_dim": checkpoint["train_embedding"].shape[-1],
        },
        "selection": result["selection"],
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "selected_validation_bank_indices": selected,
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
