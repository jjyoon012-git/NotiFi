"""Train KP5-MPR-P: part-specific listwise motion candidate reranking."""

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
from ..motion_retrieval import PartCandidateMotionReranker, TemporalMotionSelector
from ..motion_tokens import forward_kinematics, pose_to_bones, trial_bone_lengths
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import (
    PARTS,
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


PART_ITEMS = tuple(PARTS.items())


def attach_part_cost(pool: dict, target_bank: torch.Tensor,
                     target_risk: torch.Tensor,
                     train_bank: torch.Tensor) -> None:
    rows = []
    for item, indices in enumerate(pool["indices"]):
        candidates = train_bank.index_select(0, indices)
        target = target_bank[item]
        error = torch.linalg.vector_norm(candidates - target[None], dim=-1)
        speed = torch.linalg.vector_norm(
            target[1:] - target[:-1], dim=-1
        ).mean(-1) * C.TARGET_FPS
        high = (speed >= torch.quantile(speed, 0.75)) & (speed > 0.08)
        costs = []
        for _, joints in PART_ITEMS:
            part = error[:, :, joints].mean((1, 2))
            endpoint = error[:, :, joints[-1]].mean(1)
            high_error = (
                error[:, 1:, :][:, high][:, :, joints].mean((1, 2))
                if high.any() else part
            )
            cost = part + 0.35 * endpoint + 0.35 * high_error
            if int(target_risk[item]) == 2:
                cost = cost + 0.45 * part + 0.40 * high_error
            costs.append(cost)
        rows.append(torch.stack(costs, dim=-1))
    pool["target_part_cost"] = torch.stack(rows)


def train_epoch(model, loader, pool, selector_output, checkpoint,
                risk_probability, optimizer, scaler, scheduler,
                device: str) -> dict:
    model.train()
    values = []
    for (indices,) in loader:
        indices = indices.long()
        inputs = tuple(value.to(device) for value in model_inputs(
            pool, selector_output, checkpoint, risk_probability, indices
        ))
        cost = pool["target_part_cost"].index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(*inputs)
            target_probability = torch.softmax(-cost / 0.018, dim=1)
            listwise = -(
                target_probability * F.log_softmax(logits, dim=1)
            ).sum(1).mean()
            hard_logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[1])
            hard_target = cost.argmin(1).reshape(-1)
            hard = F.cross_entropy(hard_logits, hard_target)
            best_index = cost.argmin(1, keepdim=True)
            worst_index = cost.argmax(1, keepdim=True)
            best = logits.gather(1, best_index).squeeze(1)
            worst = logits.gather(1, worst_index).squeeze(1)
            margin = F.relu(0.30 - best + worst).mean()
            loss = listwise + 0.25 * hard + 0.10 * margin
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        values.append((
            float(loss.detach()), float(listwise.detach()),
            float(hard.detach()), float(margin.detach()),
        ))
    values = np.asarray(values)
    return {
        "total": float(values[:, 0].mean()),
        "listwise": float(values[:, 1].mean()),
        "hard": float(values[:, 2].mean()),
        "margin": float(values[:, 3].mean()),
    }


def assemble_candidates(indices: torch.Tensor, pool: dict,
                        train_bank: torch.Tensor,
                        target_valid: torch.Tensor,
                        baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    direct_rows, kinematic_rows = [], []
    for item, valid in enumerate(target_valid):
        direct = baseline.new_zeros(baseline[item].shape)
        local_bones = baseline.new_zeros(baseline[item].shape)
        for part_index, (_, joints) in enumerate(PART_ITEMS):
            local = int(indices[item, part_index])
            bank_index = int(pool["indices"][item, local])
            source = _render(train_bank[bank_index], valid, C.CACHE_FRAMES)
            direct[:, joints] = source[:, joints]
            source_bones, _ = pose_to_bones(source[None])
            local_bones[:, joints] = source_bones[0, :, joints]
        direct_rows.append(direct)
        direction = F.normalize(local_bones[None], dim=-1)
        direction[:, :, C.ROOT_JOINT] = 0.0
        lengths = trial_bone_lengths(baseline[item:item + 1], valid[None])
        pose = forward_kinematics(direction, lengths)[0]
        kinematic_rows.append(
            pose * valid[:, None, None].to(pose.dtype)
        )
    return torch.stack(direct_rows), torch.stack(kinematic_rows)


@torch.no_grad()
def evaluate(model, pool, selector_output, checkpoint, risk_probability,
             baseline, target_pose, target_valid, target_risk,
             device: str) -> dict:
    model.eval()
    logits = []
    for start in range(0, len(baseline), 64):
        indices = torch.arange(start, min(start + 64, len(baseline)))
        inputs = tuple(value.to(device) for value in model_inputs(
            pool, selector_output, checkpoint, risk_probability, indices
        ))
        logits.append(model(*inputs).float().cpu())
    logits = torch.cat(logits)
    local = logits.argmax(1)
    direct, kinematic = assemble_candidates(
        local, pool, checkpoint["train_bank"].float(),
        target_valid, baseline,
    )
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for strength in (0.25, 0.375, 0.50, 0.625):
        suffix = f"{int(strength * 1000):03d}"
        metrics[f"part_cartesian_{suffix}"] = _metric_batch(
            (1.0 - strength) * baseline + strength * direct,
            target_pose, target_valid, target_risk,
        )
        metrics[f"part_kinematic_{suffix}"] = _metric_batch(
            (1.0 - strength) * baseline + strength * kinematic,
            target_pose, target_valid, target_risk,
        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    target = pool["target_part_cost"].argmin(1)
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "part_oracle_accuracy": (
            local == target
        ).float().mean(0).tolist(),
        "selected_local_indices": local,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=7e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_part_seed23",
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
    train_risk_prob = torch.softmax(
        train_cache["base_risk_logits"].float() + train_output["risk_logits"], dim=-1
    )
    val_risk_prob = torch.softmax(
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
    attach_part_cost(train_pool, train_bank, train_risk, train_bank)
    attach_part_cost(val_pool, val_bank, val_risk, train_bank)
    model = PartCandidateMotionReranker(
        query_dim=train_output["pooled_features"].shape[-1],
        embedding_dim=checkpoint["train_embedding"].shape[-1],
        parts=len(PART_ITEMS),
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
            model, loader, train_pool, train_output, checkpoint,
            train_risk_prob, optimizer, scaler, scheduler, device,
        )
        validation_result = evaluate(
            model, val_pool, val_output, checkpoint, val_risk_prob,
            val_baseline, val_pose, val_valid, val_risk, device,
        )
        row = {
            "epoch": epoch, "train": trained,
            "validation_selection": validation_result["selection"],
            "part_oracle_accuracy": validation_result["part_oracle_accuracy"],
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
    selected = best["validation"].pop("selected_local_indices")
    result = {
        "run": "KP5-MPR-P-EXP01",
        "model_family": "NotiFi-KP5",
        "candidate_version": "KP5-MPR-P",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "config": vars(args) | {
            "selector_checkpoint": report_path(args.selector_checkpoint),
            "run_dir": report_path(args.run_dir),
        },
        "architecture": {
            "parts": [name for name, _ in PART_ITEMS],
            "outputs": ["direct part assembly", "bone-direction FK assembly"],
            "training_pool": "leave-self-out top-20",
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
            "parts": len(PART_ITEMS),
        },
        "selection": result["selection"],
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "selected_validation_local_indices": selected,
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
