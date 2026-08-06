"""Train a CSI-only temporal profile for eight body-to-floor contacts."""

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
from ..motion_retrieval import ContactProfileHead
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..seen_v2 import N_INJURY_JOINTS, injury_targets
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _load_pose_arrays
from .diagnose_observability import pose_only, report_path


PROXIMITY_JOINTS = tuple(C.JOINT_INDEX[name] for name in (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_foot", "right_foot", "head", "left_wrist", "right_wrist",
))


def contact_features(cache, indices=None):
    def take(value):
        return value if indices is None else value.index_select(0, indices)
    return torch.cat((
        take(cache["features"]).float(),
        take(cache["motion_activity"]).float()[..., None],
        torch.softmax(take(cache["phase_logits"]).float(), dim=-1),
        take(cache["contact_logits"]).float(),
    ), dim=-1)


def roots(dataset):
    arrays = dataset.target.cache.arrays
    return torch.from_numpy(
        np.asarray(arrays["root"][dataset.target.rows])
    ).float()


def contact_targets(dataset, mode="absolute"):
    pose, valid, _, risk = _load_pose_arrays(dataset)
    if mode == "relative":
        floor = pose[..., C.UP_AXIS].amin(-1)
        height = (
            pose[:, :, list(PROXIMITY_JOINTS), C.UP_AXIS]
            - floor[..., None]
        )
        return (height < 0.12).float() * valid[..., None], valid, risk
    if mode != "absolute":
        raise ValueError(f"Unsupported contact target mode: {mode}")
    return injury_targets(
        pose, roots(dataset), valid, risk
    )["injury_contact"].float(), valid, risk


@torch.no_grad()
def predict_contact(model, cache, valid, device):
    model.eval()
    values = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(model(
            contact_features(cache, indices).to(device),
            valid.index_select(0, indices).to(device),
        )["contact_logits"].float().cpu())
    return torch.cat(values)


def _f1(predicted, target, mask):
    predicted = predicted[mask]
    target = target[mask]
    tp = (predicted & target).sum().float()
    fp = (predicted & ~target).sum().float()
    fn = (~predicted & target).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


@torch.no_grad()
def evaluate(model, cache, target, valid, risk, device):
    inference_valid = cache["frame_mask"].bool()
    probability = torch.sigmoid(predict_contact(
        model, cache, inference_valid, device
    ))
    predicted = probability >= 0.50
    target = target.bool()
    supervision_valid = valid & inference_valid
    mask = supervision_valid[..., None].expand_as(target)
    danger = mask & (risk[:, None, None] == 2)
    f1 = _f1(predicted, target, mask)
    danger_f1 = _f1(predicted, target, danger)
    per_contact = []
    for contact in range(target.shape[-1]):
        per_contact.append(_f1(
            predicted[..., contact], target[..., contact], supervision_valid
        ))
    return {
        "f1": f1, "danger_f1": danger_f1,
        "per_contact_f1": per_contact,
        "selection_score": 1.0 - 0.35 * f1 - 0.65 * danger_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument(
        "--target-mode", choices=("absolute", "relative"), default="absolute"
    )
    parser.add_argument(
        "--feature-root", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp13_contact_profile_seed239",
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
    train_target, train_valid, train_risk = contact_targets(train, args.target_mode)
    val_target, val_valid, val_risk = contact_targets(validation, args.target_mode)
    input_dim = contact_features(train_cache, torch.arange(1)).shape[-1]
    model_config = {
        "input_dim": input_dim, "contacts": train_target.shape[-1],
    }
    model = ContactProfileHead(**model_config).to(device)
    mask = train_valid[..., None].expand_as(train_target).bool()
    positives = (train_target * mask).sum((0, 1))
    negatives = mask.sum((0, 1)) - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 15).to(device)
    sample_weight = train.sampler_weights()
    sample_weight *= torch.where(
        train_risk == 2, torch.tensor(3.0, dtype=torch.double),
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
    best, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            feature = contact_features(train_cache, indices).to(device)
            target_valid = train_valid.index_select(0, indices).to(device)
            valid = train_cache["frame_mask"].index_select(
                0, indices
            ).to(device).bool()
            supervision_valid = valid & target_valid
            target = train_target.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature, valid)["contact_logits"]
            element = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none", pos_weight=positive_weight,
            )
            weight = supervision_valid[..., None] * torch.where(
                train_risk.index_select(0, indices).to(device)[:, None, None] == 2,
                1.75, 1.0,
            )
            contact = (element * weight).sum() / (
                weight.sum() * target.shape[-1]
            ).clamp_min(1)
            temporal_mask = (
                supervision_valid[:, 1:] & supervision_valid[:, :-1]
            )[..., None]
            temporal = F.smooth_l1_loss(
                torch.sigmoid(logits[:, 1:]) - torch.sigmoid(logits[:, :-1]),
                target[:, 1:] - target[:, :-1], reduction="none", beta=0.10,
            )
            temporal = (temporal * temporal_mask).sum() / (
                temporal_mask.sum() * target.shape[-1]
            ).clamp_min(1)
            loss = contact + 0.10 * temporal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        validation_result = evaluate(
            model, val_cache, val_target, val_valid, val_risk, device
        )
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation": validation_result,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = validation_result["selection_score"]
        if best is None or score < best["score"] - 1e-5:
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
        "contact_order": (
            list(PROXIMITY_JOINTS) if args.target_mode == "relative"
            else list(range(N_INJURY_JOINTS))
        ),
        "target_mode": args.target_mode,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "feature_root": report_path(args.feature_root),
    }, args.run_dir / "best_model.pt")
    summary = {
        "status": "validation_selected_csi_contact_profile",
        "protocol": args.exp,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "history": history, "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
