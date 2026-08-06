"""Train a CSI temporal action/risk classifier without pose-regression conflict."""

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
from ..motion_retrieval import TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _canonicalize, _load_pose_arrays
from .diagnose_observability import pose_only, report_path
from .train_motion_retrieval_selector import _augment_features, _class_weights


@torch.no_grad()
def evaluate(model, cache, action, risk, device):
    model.eval()
    action_logits, risk_logits = [], []
    for start in range(0, len(action), 64):
        stop = min(start + 64, len(action))
        output = model(
            cache["features"][start:stop].to(device).float(),
            cache["frame_mask"][start:stop].to(device),
        )
        action_logits.append(output["action_logits"].float().cpu())
        risk_logits.append(output["risk_logits"].float().cpu())
    action_logits = torch.cat(action_logits)
    risk_logits = torch.cat(risk_logits)
    action_accuracy = float((action_logits.argmax(-1) == action).float().mean())
    risk_accuracy = float((risk_logits.argmax(-1) == risk).float().mean())
    action_nll = float(F.cross_entropy(action_logits, action))
    risk_nll = float(F.cross_entropy(risk_logits, risk))
    return {
        "action_accuracy": action_accuracy,
        "risk_accuracy": risk_accuracy,
        "action_nll": action_nll,
        "risk_nll": risk_nll,
        "selection_score": (
            (1.0 - action_accuracy) + 0.20 * (1.0 - risk_accuracy)
            + 0.03 * action_nll + 0.01 * risk_nll
        ),
    }


def motion_soft_label_table(pose, valid, class_id, weight):
    bank = torch.stack([
        _canonicalize(value, mask, C.CACHE_FRAMES)
        for value, mask in zip(pose, valid)
    ])
    class_risk = torch.tensor(
        [0] * 9 + [1] * 3 + [2] * 5, dtype=torch.long
    )
    prototypes = []
    for index in range(C.N_CLASSES):
        members = bank[class_id == index]
        if not len(members):
            members = bank[class_risk.index_select(0, class_id) == class_risk[index]]
        prototypes.append(members.mean(0))
    prototypes = torch.stack(prototypes)
    distance = torch.linalg.vector_norm(
        prototypes[:, None] - prototypes[None], dim=-1
    ).mean((2, 3))
    finite = distance[~torch.eye(C.N_CLASSES, dtype=torch.bool)]
    scale = finite.median().clamp_min(1e-5)
    cross_risk = class_risk[:, None] != class_risk[None]
    distance = distance + cross_risk * (2.0 * scale)
    similarity = torch.softmax(-distance / scale, dim=-1)
    one_hot = torch.eye(C.N_CLASSES)
    table = (1.0 - float(weight)) * one_hot + float(weight) * similarity
    if not torch.isfinite(table).all() or not torch.allclose(
        table.sum(-1), torch.ones(C.N_CLASSES), atol=1e-5
    ):
        raise RuntimeError("Invalid train-only motion soft-label table")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--hierarchy-weight", type=float, default=0.0)
    parser.add_argument("--motion-soft-label-weight", type=float, default=0.0)
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cache = torch.load(
        args.feature_root / "train_features.pt", map_location="cpu",
        weights_only=False,
    )
    val_cache = torch.load(
        args.feature_root / "val_features.pt", map_location="cpu",
        weights_only=False,
    )
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, train_action, train_risk = _load_pose_arrays(train)
    _, _, val_action, val_risk = _load_pose_arrays(validation)
    model_config = {
        "input_dim": train_cache["features"].shape[-1],
        "embedding_dim": 16, "width": 192, "layers": 2, "heads": 6,
    }
    model = TemporalMotionSelector(**model_config).to(device)
    action_weight = _class_weights(train_action, C.N_CLASSES, device)
    risk_weight = _class_weights(train_risk, C.N_RISK, device)
    risk_weight[2] *= 1.5
    soft_label_table = motion_soft_label_table(
        train_pose, train_valid, train_action, args.motion_soft_label_weight
    ).to(device)
    sample_weight = train.sampler_weights()
    sample_weight *= torch.where(
        train_risk == 2, torch.tensor(2.5, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))), batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sample_weight, len(train), replacement=True,
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
        lambda step: 0.5 * (
            1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)
        ),
    )
    history, best, stale = [], None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            feature = train_cache["features"].index_select(0, indices).to(device).float()
            mask = train_cache["frame_mask"].index_select(0, indices).to(device)
            action = train_action.index_select(0, indices).to(device)
            risk = train_risk.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(_augment_features(feature, mask), mask)
            if args.motion_soft_label_weight > 0:
                soft_target = soft_label_table.index_select(0, action)
                row_weight = action_weight.index_select(0, action)
                action_loss = -(
                    soft_target * F.log_softmax(output["action_logits"], dim=-1)
                ).sum(-1)
                action_loss = (action_loss * row_weight).sum() / row_weight.sum()
            else:
                action_loss = F.cross_entropy(
                    output["action_logits"], action, weight=action_weight,
                    label_smoothing=0.03,
                )
            risk_loss = F.cross_entropy(
                output["risk_logits"], risk, weight=risk_weight,
                label_smoothing=0.02,
            )
            action_probability = torch.softmax(output["action_logits"], dim=-1)
            action_risk = torch.stack((
                action_probability[:, :9].sum(-1),
                action_probability[:, 9:12].sum(-1),
                action_probability[:, 12:].sum(-1),
            ), dim=-1).clamp_min(1e-7)
            hierarchy_risk = F.nll_loss(
                action_risk.log(), risk, weight=risk_weight,
            )
            direct_risk = torch.softmax(
                output["risk_logits"], dim=-1
            ).clamp_min(1e-7)
            consistency = 0.50 * (
                F.kl_div(direct_risk.log(), action_risk, reduction="batchmean")
                + F.kl_div(action_risk.log(), direct_risk, reduction="batchmean")
            )
            if args.hierarchy_weight > 0:
                loss = (
                    action_loss + 0.25 * risk_loss
                    + args.hierarchy_weight * (
                        0.25 * hierarchy_risk + 0.10 * consistency
                    )
                )
            else:
                loss = action_loss + 0.50 * risk_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append((
                float(loss.detach()), float(action_loss.detach()),
                float(risk_loss.detach()), float(hierarchy_risk.detach()),
                float(consistency.detach()),
            ))
        validation_result = evaluate(
            model, val_cache, val_action, val_risk, device
        )
        values = np.asarray(losses)
        row = {
            "epoch": epoch,
            "train": {
                "loss": float(values[:, 0].mean()),
                "action": float(values[:, 1].mean()),
                "risk": float(values[:, 2].mean()),
                "hierarchy_risk": float(values[:, 3].mean()),
                "consistency": float(values[:, 4].mean()),
            },
            "validation": validation_result,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = validation_result["selection_score"]
        if best is None or score < best["score"] - 1e-6:
            best = {
                "score": score, "epoch": epoch,
                "validation": validation_result,
                "model": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best["model"], "model_config": model_config,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "protocol": args.exp, "seed": args.seed,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "feature_root": report_path(args.feature_root),
        "hierarchy_weight": args.hierarchy_weight,
        "motion_soft_label_weight": args.motion_soft_label_weight,
    }, args.run_dir / "best_model.pt")
    result = {
        "status": "validation_selected_csi_action_classifier",
        "protocol": args.exp,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "history": history, "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "feature_root": report_path(args.feature_root),
    }
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
