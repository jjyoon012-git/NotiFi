"""Train KP5-MPR-X: temporal CSI-to-candidate cross-matching reranker."""

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
from ..motion_retrieval import TemporalCandidateMotionReranker, TemporalMotionSelector
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
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool
from .train_motion_retrieval_selector import predict_selector


def motion_sequence(bank: torch.Tensor, bins: int = 38) -> torch.Tensor:
    position = F.adaptive_avg_pool1d(
        bank.flatten(2).transpose(1, 2), bins
    ).transpose(1, 2)
    velocity = torch.zeros_like(position)
    velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) * (bins - 1)
    return torch.cat((position, velocity), dim=-1)


def temporal_inputs(pool: dict, cache: dict, bank_motion: torch.Tensor,
                    baseline_motion: torch.Tensor,
                    train_class: torch.Tensor,
                    risk_probability: torch.Tensor,
                    indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    candidates = pool["indices"].index_select(0, indices)
    return (
        cache["features"].index_select(0, indices),
        cache["frame_mask"].index_select(0, indices),
        baseline_motion.index_select(0, indices),
        bank_motion[candidates],
        train_class[candidates],
        risk_probability.index_select(0, indices),
        pool["retrieval_score"].index_select(0, indices),
        pool["action_log_probability"].index_select(0, indices),
    )


def train_epoch(model, loader, pool, cache, bank_motion, baseline_motion,
                train_class, risk_probability, optimizer, scaler,
                scheduler, device: str) -> dict:
    model.train()
    rows = []
    for (indices,) in loader:
        indices = indices.long()
        inputs = tuple(value.to(device).float() if value.dtype.is_floating_point
                       else value.to(device) for value in temporal_inputs(
                           pool, cache, bank_motion, baseline_motion,
                           train_class, risk_probability, indices,
                       ))
        target_cost = pool["target_cost"].index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(*inputs)
            target_probability = torch.softmax(-target_cost / 0.018, dim=-1)
            listwise = -(
                target_probability * F.log_softmax(logits, dim=-1)
            ).sum(-1).mean()
            hard = F.cross_entropy(logits, target_cost.argmin(-1))
            best = logits.gather(
                1, target_cost.argmin(-1, keepdim=True)
            ).squeeze(1)
            worst = logits.gather(
                1, target_cost.argmax(-1, keepdim=True)
            ).squeeze(1)
            margin = F.relu(0.30 - best + worst).mean()
            loss = listwise + 0.25 * hard + 0.10 * margin
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        rows.append((
            float(loss.detach()), float(listwise.detach()),
            float(hard.detach()), float(margin.detach()),
        ))
    values = np.asarray(rows)
    return {
        "total": float(values[:, 0].mean()),
        "listwise": float(values[:, 1].mean()),
        "hard": float(values[:, 2].mean()),
        "margin": float(values[:, 3].mean()),
    }


@torch.no_grad()
def predict_logits(model, pool, cache, bank_motion, baseline_motion,
                   train_class, risk_probability, batch_size: int,
                   device: str) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(cache["features"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(cache["features"])))
        inputs = tuple(value.to(device).float() if value.dtype.is_floating_point
                       else value.to(device) for value in temporal_inputs(
                           pool, cache, bank_motion, baseline_motion,
                           train_class, risk_probability, indices,
                       ))
        outputs.append(model(*inputs).float().cpu())
    return torch.cat(outputs)


@torch.no_grad()
def evaluate(model, pool, cache, bank_motion, baseline_motion,
             train_class, risk_probability, train_bank,
             baseline, target_pose, target_valid, target_risk,
             device: str) -> dict:
    logits = predict_logits(
        model, pool, cache, bank_motion, baseline_motion,
        train_class, risk_probability, 24, device,
    )
    local = logits.argmax(-1)
    selected = pool["indices"].gather(1, local[:, None]).squeeze(1)
    candidate = torch.stack([
        _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
        for index, valid in zip(selected, target_valid)
    ])
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for strength in (0.25, 0.375, 0.50, 0.625):
        name = f"temporal_cartesian_{int(strength * 1000):03d}"
        metrics[name] = _metric_batch(
            (1.0 - strength) * baseline + strength * candidate,
            target_pose, target_valid, target_risk,
        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "top1_oracle_accuracy": float(
            (local == pool["target_cost"].argmin(-1)).float().mean()
        ),
        "logits": logits,
        "selected_bank_indices": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=6e-4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_temporal_seed31",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])
    root = args.selector_checkpoint.parent
    train_cache = torch.load(root / "train_features.pt", map_location="cpu", weights_only=False)
    val_cache = torch.load(root / "val_features.pt", map_location="cpu", weights_only=False)
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
    train_action = train_cache["base_action_logits"].float() + train_output["action_logits"]
    val_action = val_cache["base_action_logits"].float() + val_output["action_logits"]
    train_risk_probability = torch.softmax(
        train_cache["base_risk_logits"].float() + train_output["risk_logits"], dim=-1
    )
    val_risk_probability = torch.softmax(
        val_cache["base_risk_logits"].float() + val_output["risk_logits"], dim=-1
    )
    train_pool = make_candidate_pool(
        train_baseline_bank, train_bank, train_risk,
        train_bank, train_class, train_action,
        args.top_k, args.shortlist,
        self_indices=torch.arange(len(train_bank)),
    )
    val_distance = exact_pose_distance(
        val_baseline_bank, train_bank, root / "val_exact_pose_distance.pt"
    )
    val_pool = make_candidate_pool(
        val_baseline_bank, val_bank, val_risk,
        train_bank, train_class, val_action,
        args.top_k, args.shortlist,
        exact_distance_matrix=val_distance,
    )
    bank_motion = motion_sequence(train_bank)
    train_baseline_motion = motion_sequence(train_baseline_bank)
    val_baseline_motion = motion_sequence(val_baseline_bank)
    model = TemporalCandidateMotionReranker(
        csi_dim=train_cache["features"].shape[-1],
        motion_dim=bank_motion.shape[-1],
    ).to(device)
    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(args.danger_weight, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))), batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights, len(train), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ), num_workers=0,
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
    best, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        trained = train_epoch(
            model, loader, train_pool, train_cache,
            bank_motion, train_baseline_motion, train_class,
            train_risk_probability, optimizer, scaler, scheduler, device,
        )
        validation_result = evaluate(
            model, val_pool, val_cache, bank_motion,
            val_baseline_motion, train_class, val_risk_probability,
            train_bank, val_baseline, val_pose, val_valid, val_risk, device,
        )
        row = {
            "epoch": epoch, "train": trained,
            "validation_selection": validation_result["selection"],
            "top1_oracle_accuracy": validation_result["top1_oracle_accuracy"],
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
    logits = best["validation"].pop("logits")
    selected = best["validation"].pop("selected_bank_indices")
    result = {
        "run": "KP5-MPR-X-EXP01",
        "model_family": "NotiFi-KP5",
        "candidate_version": "KP5-MPR-X",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": vars(args) | {
            "selector_checkpoint": report_path(args.selector_checkpoint),
            "run_dir": report_path(args.run_dir),
        },
        "architecture": {
            "query": "38-bin frozen CSI semantic sequence",
            "candidate": "38-bin joint position and velocity",
            "cross_match": "temporal pair encoder with dilated local blocks",
            "training_pool": "leave-self-out top-10",
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
            "csi_dim": train_cache["features"].shape[-1],
            "motion_dim": bank_motion.shape[-1],
        },
        "selection": result["selection"],
        "validation_logits": logits,
        "selected_validation_bank_indices": selected,
        "selector_checkpoint": report_path(args.selector_checkpoint),
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
