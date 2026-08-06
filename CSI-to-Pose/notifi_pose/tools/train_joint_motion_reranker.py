"""Jointly fine-tune the temporal CSI selector and motion reranker."""

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
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool
from .train_motion_retrieval_selector import _augment_features, predict_selector


def reranker_inputs(output, checkpoint, cache, pool, indices, device):
    candidate_indices = pool["indices"].index_select(0, indices)
    risk_probability = torch.softmax(
        cache["base_risk_logits"].index_select(0, indices).to(device).float()
        + output["risk_logits"], dim=-1,
    )
    return (
        output["pooled_features"],
        output["motion_embedding"],
        checkpoint["train_embedding"][candidate_indices].to(device),
        checkpoint["train_class"][candidate_indices].to(device),
        risk_probability,
        pool["retrieval_score"].index_select(0, indices).to(device),
        pool["action_log_probability"].index_select(0, indices).to(device),
    )


@torch.no_grad()
def evaluate(selector, reranker, cache, pool, checkpoint,
             baseline, target_pose, target_valid, target_risk, device):
    selector.eval()
    reranker.eval()
    logits = []
    fused_action = []
    for start in range(0, len(baseline), 64):
        indices = torch.arange(start, min(start + 64, len(baseline)))
        output = selector(
            cache["features"].index_select(0, indices).to(device).float(),
            cache["frame_mask"].index_select(0, indices).to(device),
        )
        logits.append(reranker(*reranker_inputs(
            output, checkpoint, cache, pool, indices, device,
        )).float().cpu())
        fused_action.append(
            cache["base_action_logits"].index_select(0, indices).float()
            + output["action_logits"].float().cpu()
        )
    logits = torch.cat(logits)
    fused_action = torch.cat(fused_action)
    train_bank = checkpoint["train_bank"].float()
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for temperature in (0.20, 0.35, 0.50, 0.75, 1.0):
        probability = torch.softmax(logits / temperature, dim=-1)
        for top_k in (1, 2, 3, 5):
            top = logits.topk(top_k, dim=-1).indices
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
            for strength in (0.50, 0.625):
                key = (
                    f"t{int(temperature * 100):03d}_top{top_k}"
                    f"_{int(strength * 1000):03d}"
                )
                metrics[key] = _metric_batch(
                    (1.0 - strength) * baseline + strength * candidate,
                    target_pose, target_valid, target_risk,
                )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "action_accuracy": float(
            (fused_action.argmax(-1) == cache["class_id"]).float().mean()
        ) if "class_id" in cache else None,
        "top1_oracle_accuracy": float(
            (logits.argmax(-1) == pool["target_cost"].argmin(-1)).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--selector-lr", type=float, default=4e-5)
    parser.add_argument("--reranker-lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--action-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_joint_seed61",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])
    reranker_checkpoint = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    reranker = CandidateMotionReranker(
        **reranker_checkpoint["model_config"]
    ).to(device)
    reranker.load_state_dict(reranker_checkpoint["model"])
    root = args.selector_checkpoint.parent
    train_cache = torch.load(
        root / "train_features.pt", map_location="cpu", weights_only=False
    )
    val_cache = torch.load(
        root / "val_features.pt", map_location="cpu", weights_only=False
    )
    with torch.no_grad():
        initial_train = predict_selector(selector, train_cache, 64, device)
        initial_val = predict_selector(selector, val_cache, 64, device)

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, train_class, train_risk = _load_pose_arrays(train)
    val_pose, val_valid, val_class, val_risk = _load_pose_arrays(validation)
    train_cache["class_id"] = train_class
    train_cache["risk_id"] = train_risk
    val_cache["class_id"] = val_class
    val_cache["risk_id"] = val_risk
    train_bank = checkpoint["train_bank"].float()
    val_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(val_pose, val_valid)
    ])
    train_baseline = train_cache["baseline_pose"].float()
    val_baseline = val_cache["baseline_pose"].float()
    train_baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(train_baseline, train_valid)
    ])
    val_baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(val_baseline, val_valid)
    ])
    train_action = train_cache["base_action_logits"].float() + initial_train["action_logits"]
    val_action = val_cache["base_action_logits"].float() + initial_val["action_logits"]
    train_pool = make_candidate_pool(
        train_baseline_bank, train_bank, train_risk,
        train_bank, train_class, train_action,
        args.top_k, args.shortlist, self_indices=torch.arange(len(train_bank)),
        action_penalty=args.action_penalty,
    )
    distance = exact_pose_distance(
        val_baseline_bank, train_bank, root / "val_exact_pose_distance.pt"
    )
    val_pool = make_candidate_pool(
        val_baseline_bank, val_bank, val_risk,
        train_bank, train_class, val_action,
        args.top_k, args.shortlist, exact_distance_matrix=distance,
        action_penalty=args.action_penalty,
    )

    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(3.0, dtype=torch.double),
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
    optimizer = torch.optim.AdamW([
        {"params": selector.parameters(), "lr": args.selector_lr},
        {"params": reranker.parameters(), "lr": args.reranker_lr},
    ], weight_decay=args.weight_decay)
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
        selector.train()
        reranker.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            features = train_cache["features"].index_select(0, indices).to(device).float()
            mask = train_cache["frame_mask"].index_select(0, indices).to(device)
            class_id = train_class.index_select(0, indices).to(device)
            risk_id = train_risk.index_select(0, indices).to(device)
            target_cost = train_pool["target_cost"].index_select(0, indices).to(device)
            anchor = initial_train["embedding"].index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                output = selector(_augment_features(features, mask), mask)
                logits = reranker(*reranker_inputs(
                    output, checkpoint, train_cache, train_pool, indices, device,
                ))
                target_probability = torch.softmax(-target_cost / 0.020, dim=-1)
                listwise = -(
                    target_probability * F.log_softmax(logits, dim=-1)
                ).sum(-1).mean()
                hard = F.cross_entropy(logits, target_cost.argmin(-1))
                action = F.cross_entropy(output["action_logits"], class_id)
                risk = F.cross_entropy(output["risk_logits"], risk_id)
                anchor_loss = F.smooth_l1_loss(
                    output["motion_embedding"], anchor, beta=0.5
                )
                loss = (
                    listwise + 0.20 * hard + 0.15 * action
                    + 0.10 * risk + 0.15 * anchor_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(selector.parameters()) + list(reranker.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append((
                float(loss.detach()), float(listwise.detach()),
                float(action.detach()), float(risk.detach()),
            ))
        validation_result = evaluate(
            selector, reranker, val_cache, val_pool, checkpoint,
            val_baseline, val_pose, val_valid, val_risk, device,
        )
        values = np.asarray(losses)
        row = {
            "epoch": epoch,
            "train": {
                "loss": float(values[:, 0].mean()),
                "listwise": float(values[:, 1].mean()),
                "action": float(values[:, 2].mean()),
                "risk": float(values[:, 3].mean()),
            },
            "validation_selection": validation_result["selection"],
            "action_accuracy": validation_result["action_accuracy"],
            "top1_oracle_accuracy": validation_result["top1_oracle_accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = validation_result["selection"]["score"]
        if best is None or score < best["score"] - 1e-5:
            best = {
                "epoch": epoch, "score": score,
                "selector": copy.deepcopy(selector.state_dict()),
                "reranker": copy.deepcopy(reranker.state_dict()),
                "validation": validation_result,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best is not None
    result = {
        "run": "KP5-MPR-JOINT-EXP01",
        "candidate_version": "KP5-MPR-JOINT",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "selector_checkpoint": report_path(args.selector_checkpoint),
            "reranker_checkpoint": report_path(args.reranker_checkpoint),
            "run_dir": report_path(args.run_dir),
        },
        "architecture": {
            "training": "joint temporal CSI selector and candidate reranker",
            "regularization": "feature augmentation plus embedding/action/risk anchors",
            "candidate_pool": f"leave-self-out top-{args.top_k}",
        },
        "selection": {"epoch": best["epoch"], **best["validation"]["selection"]},
        "validation": best["validation"],
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "selector_model": best["selector"],
        "selector_config": checkpoint["model_config"],
        "reranker_model": best["reranker"],
        "reranker_config": reranker_checkpoint["model_config"],
        "selection": result["selection"],
        "source_selector_checkpoint": report_path(args.selector_checkpoint),
        "source_reranker_checkpoint": report_path(args.reranker_checkpoint),
        "action_penalty": args.action_penalty,
        "top_k": args.top_k,
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
