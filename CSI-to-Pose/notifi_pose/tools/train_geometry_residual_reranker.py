"""Train a geometry-aware residual correction over the frozen KP5 reranker."""

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
from ..motion_retrieval import (
    GeometryResidualReranker,
    TemporalMotionSelector,
    geometric_pair_features,
)
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
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


def geometry_inputs(pool, selector_output, checkpoint, risk_probability,
                    pair_features, indices, device):
    base = model_inputs(
        pool, selector_output, checkpoint, risk_probability, indices
    )
    return tuple(value.to(device) for value in base) + (
        pair_features.index_select(0, indices).to(device),
    )


@torch.no_grad()
def evaluate(model, pool, selector_output, checkpoint, risk_probability,
             pair_features, baseline, target_pose, target_valid, target_risk,
             device):
    model.eval()
    logits = []
    for start in range(0, len(baseline), 64):
        indices = torch.arange(start, min(start + 64, len(baseline)))
        logits.append(model(*geometry_inputs(
            pool, selector_output, checkpoint, risk_probability,
            pair_features, indices, device,
        )).float().cpu())
    logits = torch.cat(logits)
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
            predicted = 0.375 * baseline + 0.625 * candidate
            key = f"t{int(temperature * 100):03d}_top{top_k}_625"
            metrics[key] = _metric_batch(
                predicted, target_pose, target_valid, target_risk
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    target_cost = pool["target_cost"]
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "top1_oracle_accuracy": float(
            (logits.argmax(-1) == target_cost.argmin(-1)).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--action-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--base-reranker", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_geometry_seed47",
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
    train_cache = torch.load(
        root / "train_features.pt", map_location="cpu", weights_only=False
    )
    val_cache = torch.load(
        root / "val_features.pt", map_location="cpu", weights_only=False
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
    val_baseline = val_cache["baseline_pose"].float()
    train_baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(train_baseline, train_valid)
    ])
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
    train_pairs = geometric_pair_features(
        train_baseline_bank, train_bank[train_pool["indices"]]
    )
    val_pairs = geometric_pair_features(
        val_baseline_bank, train_bank[val_pool["indices"]]
    )

    model_config = {
        "query_dim": train_output["pooled_features"].shape[-1],
        "embedding_dim": checkpoint["train_embedding"].shape[-1],
        "pair_dim": train_pairs.shape[-1],
    }
    model = GeometryResidualReranker(**model_config).to(device)
    base_checkpoint = torch.load(
        args.base_reranker, map_location="cpu", weights_only=False
    )
    model.base.load_state_dict(base_checkpoint["model"])

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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
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
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            target_cost = train_pool["target_cost"].index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(*geometry_inputs(
                    train_pool, train_output, checkpoint,
                    train_risk_probability, train_pairs, indices, device,
                ))
                target_probability = torch.softmax(-target_cost / 0.020, dim=-1)
                listwise = -(
                    target_probability * F.log_softmax(logits, dim=-1)
                ).sum(-1).mean()
                hard = F.cross_entropy(logits, target_cost.argmin(-1))
                loss = listwise + 0.20 * hard
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append((float(loss.detach()), float(listwise.detach()), float(hard.detach())))
        validation_result = evaluate(
            model, val_pool, val_output, checkpoint, val_risk_probability,
            val_pairs, val_baseline, val_pose, val_valid, val_risk, device,
        )
        loss_array = np.asarray(losses)
        row = {
            "epoch": epoch,
            "train": {
                "loss": float(loss_array[:, 0].mean()),
                "listwise": float(loss_array[:, 1].mean()),
                "hard": float(loss_array[:, 2].mean()),
            },
            "validation_selection": validation_result["selection"],
            "top1_oracle_accuracy": validation_result["top1_oracle_accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
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
    result = {
        "run": "KP5-MPR-GR-EXP01",
        "candidate_version": "KP5-MPR-GR",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "selector_checkpoint": report_path(args.selector_checkpoint),
            "base_reranker": report_path(args.base_reranker),
            "run_dir": report_path(args.run_dir),
        },
        "architecture": {
            "base": "frozen KP5-MPR-R",
            "correction": "pose-part-velocity-acceleration-height-contact residual",
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
        "model": best["state"], "model_config": model_config,
        "selection": result["selection"],
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "base_reranker": report_path(args.base_reranker),
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
