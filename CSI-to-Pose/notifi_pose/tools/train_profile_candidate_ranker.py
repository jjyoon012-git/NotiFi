"""Train a leave-self-out profile ranker for action-conditioned motion retrieval."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..motion_retrieval import (
    ContactProfileHead, ProfileCandidateRanker, TemporalMotionSelector,
)
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _canonicalize, _metric_batch, _render
from .calibrate_action_logit_fusion import calibrated_action_logits
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_kinetic_pose import DISTAL_JOINTS, pose_selection_score
from .train_csi_contact_profile import predict_contact


@torch.no_grad()
def _external_action_logits(model, cache, device):
    values = []
    model.eval()
    for start in range(0, len(cache["features"]), 64):
        stop = min(start + 64, len(cache["features"]))
        values.append(model(
            cache["features"][start:stop].to(device).float(),
            cache["frame_mask"][start:stop].to(device),
        )["action_logits"].float().cpu())
    return torch.cat(values)


def group_candidate_values(group):
    values = [
        group["pose"][:, None], group["scalar"][:, None],
        group["part_values"],
    ]
    if group.get("contact_values") is not None:
        values.append(group["contact_values"])
    if group.get("selector_values") is not None:
        values.append(group["selector_values"][:, None])
    return torch.cat(values, dim=-1)


def group_tensors(data, features, leave_self_out: bool):
    """Pad variable action groups and build train-only listwise targets."""
    target_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(data["target_pose"], data["target_valid"])
    ])
    rows = []
    max_candidates = max(
        len(group["indices"]) for groups in features for group in groups
    )
    feature_dim = group_candidate_values(features[0][0]).shape[-1]
    for item, groups in enumerate(features):
        for group in groups:
            values = group_candidate_values(group)
            candidates = data["train_bank"].index_select(0, group["indices"])
            error = torch.linalg.vector_norm(
                candidates - target_bank[item][None], dim=-1
            )
            overall = error.mean((1, 2))
            distal = error[:, :, DISTAL_JOINTS].mean((1, 2))
            speed = torch.linalg.vector_norm(
                target_bank[item, 1:] - target_bank[item, :-1], dim=-1
            ).mean(-1) * C.TARGET_FPS
            high = (speed >= torch.quantile(speed, 0.75)) & (speed > 0.08)
            high_error = (
                error[:, 1:][:, high].mean((1, 2)) if high.any() else overall
            )
            cost = overall + 0.45 * distal + 0.30 * high_error
            if int(data["target_risk"][item]) == 2:
                cost = cost + 0.65 * overall + 0.55 * distal + 0.45 * high_error
            cost = (cost - cost.mean()) / cost.std().clamp_min(1e-5)
            count = len(values)
            padded_features = values.new_zeros(max_candidates, feature_dim)
            padded_cost = values.new_zeros(max_candidates)
            padded_mask = torch.zeros(max_candidates, dtype=torch.bool)
            padded_features[:count] = values
            padded_cost[:count] = cost
            padded_mask[:count] = True
            rows.append((
                padded_features, padded_cost, padded_mask,
                int(group["class_id"]), int(data["target_risk"][item]), item,
                group["context"],
            ))
    return {
        "features": torch.stack([row[0] for row in rows]),
        "cost": torch.stack([row[1] for row in rows]),
        "mask": torch.stack([row[2] for row in rows]),
        "class_id": torch.tensor([row[3] for row in rows]),
        "risk": torch.tensor([row[4] for row in rows]),
        "query": torch.tensor([row[5] for row in rows]),
        "context": torch.stack([row[6] for row in rows]),
        "leave_self_out": leave_self_out,
    }


def train_epoch(model, groups, loader, optimizer, device):
    model.train()
    losses = []
    for (indices,) in loader:
        feature = groups["features"].index_select(0, indices).to(device)
        class_id = groups["class_id"].index_select(0, indices).to(device)
        class_id = class_id[:, None].expand(-1, feature.shape[1])
        mask = groups["mask"].index_select(0, indices).to(device)
        cost = groups["cost"].index_select(0, indices).to(device)
        context = groups["context"].index_select(0, indices).to(device)
        context = context[:, None].expand(-1, feature.shape[1], -1)
        optimizer.zero_grad(set_to_none=True)
        score = model(
            feature, class_id,
            context if model.context_dim else None,
        ).masked_fill(~mask, -1e4)
        target = torch.softmax(
            (-cost / 0.50).masked_fill(~mask, -1e4), dim=-1
        )
        listwise = -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()
        best = cost.masked_fill(~mask, float("inf")).argmin(-1)
        hard = F.cross_entropy(score, best)
        loss = listwise + 0.20 * hard
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append((float(loss), float(listwise), float(hard)))
    values = np.asarray(losses)
    return {
        "total": float(values[:, 0].mean()),
        "listwise": float(values[:, 1].mean()),
        "hard": float(values[:, 2].mean()),
    }


@torch.no_grad()
def render_ranked_action(model, data, features, device,
                         inner_top_k=2, inner_temperature=1.0):
    models = model if isinstance(model, (list, tuple)) else (model,)
    for member in models:
        member.eval()
    motions = []
    inference_valid = data["inference_valid"]
    for groups, valid in zip(features, inference_valid):
        candidates, probabilities = [], []
        for group in groups:
            values = group_candidate_values(group).to(device)
            class_id = torch.full(
                (len(values),), int(group["class_id"]),
                dtype=torch.long, device=device,
            )
            context = group["context"].to(device)
            context = context[None].expand(len(values), -1)
            score = torch.stack([
                member(
                    values, class_id,
                    context if member.context_dim else None,
                ) for member in models
            ]).mean(0).cpu()
            count = min(int(inner_top_k), len(score))
            local = score.topk(count).indices
            bank_indices = group["indices"].index_select(0, local)
            inner_weight = torch.softmax(
                score.index_select(0, local) / float(inner_temperature), dim=0
            )
            candidate = (
                data["train_bank"].index_select(0, bank_indices)
                * inner_weight[:, None, None, None]
            ).sum(0)
            candidates.append(candidate)
            probabilities.append(group["probability"])
        probability = torch.stack(probabilities)
        probability = probability / probability.sum().clamp_min(1e-6)
        canonical = (
            torch.stack(candidates) * probability[:, None, None, None]
        ).sum(0)
        motions.append(_render(canonical, valid, C.CACHE_FRAMES))
    return torch.stack(motions)


@torch.no_grad()
def evaluate_pose(model, data, features, current, device,
                  blend_strength=0.35):
    inference_valid = data["inference_valid"]
    action = render_ranked_action(model, data, features, device)
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    action = monotonic_energy_warp(
        action, activity, inference_valid, 0.50, 0.30
    )
    low = smooth_valid_delta(action - current, inference_valid, 17)
    predicted = current + float(blend_strength) * low
    predicted = predicted - predicted[:, :, :1]
    metrics = _metric_batch(
        predicted, data["target_pose"], data["target_valid"],
        data["target_risk"],
    )
    return predicted, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--danger-weight", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--context-dim", type=int, default=0)
    parser.add_argument(
        "--blend-strength", type=float, default=0.35,
        help="Validation-locked action-prior blend used to select the ranker.",
    )
    parser.add_argument(
        "--include-selector-distance", action="store_true",
        help="Add the frozen CSI-to-motion embedding distance per candidate.",
    )
    parser.add_argument(
        "--external-action-checkpoint", type=Path, default=None,
        help="Train ranker groups from a validation-locked independent CSI head.",
    )
    parser.add_argument(
        "--contact-profile-checkpoint", type=Path, default=None,
        help="Add CSI-predicted relative body-floor proximity to ranking.",
    )
    parser.add_argument(
        "--action-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp7_action_logit_fusion"
        / "calibration.json",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_config = json.loads(
        args.action_calibration.read_text(encoding="utf-8")
    )["selection"]
    adaptive_config = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]

    train_data = prepare(args, "train", device)
    val_data = prepare(args, "val", device)
    for data in (train_data, val_data):
        data["include_selector_distance"] = args.include_selector_distance
    if args.contact_profile_checkpoint is not None:
        contact_checkpoint = torch.load(
            args.contact_profile_checkpoint, map_location="cpu",
            weights_only=False,
        )
        contact_model = ContactProfileHead(
            **contact_checkpoint["model_config"]
        ).to(device)
        contact_model.load_state_dict(contact_checkpoint["model"])
        for data in (train_data, val_data):
            data["predicted_proximity"] = torch.sigmoid(predict_contact(
                contact_model, data["cache"], data["inference_valid"], device
            ))
    if args.external_action_checkpoint is not None:
        action_checkpoint = torch.load(
            args.external_action_checkpoint, map_location="cpu",
            weights_only=False,
        )
        action_model = TemporalMotionSelector(
            **action_checkpoint["model_config"]
        ).to(device)
        action_model.load_state_dict(action_checkpoint["model"])
        for data in (train_data, val_data):
            extra = _external_action_logits(action_model, data["cache"], device)
            data["fused_action"] = (
                1.50 * data["base_action_logits"] + 0.75 * extra
            )
    else:
        for data in (train_data, val_data):
            data["fused_action"] = calibrated_action_logits(data, action_config)
    train_features = retrieval_features(
        train_data, 3, 1.0, self_indices=torch.arange(len(train_data["train_bank"]))
    )
    val_features = retrieval_features(val_data, 3, 1.0)
    train_groups = group_tensors(train_data, train_features, True)

    model_config = {
        "feature_dim": group_candidate_values(train_features[0][0]).shape[-1],
        "context_dim": args.context_dim,
    }
    model = ProfileCandidateRanker(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    sample_weight = torch.where(
        train_groups["risk"] == 2,
        torch.tensor(args.danger_weight, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    sampler = WeightedRandomSampler(
        sample_weight, len(sample_weight), replacement=True
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(sample_weight))),
        batch_size=args.batch_size, sampler=sampler,
    )
    current = predict_locked(val_data, adaptive_config)
    history, best = [], None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_groups, loader, optimizer, device)
        _, metrics = evaluate_pose(
            model, val_data, val_features, current, device,
            blend_strength=args.blend_strength,
        )
        score = pose_selection_score(metrics)
        row = {"epoch": epoch, "loss": loss, "score": score, "metrics": metrics}
        history.append(row)
        print(json.dumps({
            "epoch": epoch, "loss": loss["total"], "score": score,
            "danger_pose_m": metrics["danger_pose_mpjpe_m"],
            "danger_distal_m": metrics["danger_distal_mpjpe_m"],
        }))
        if best is None or score < best["score"] - 1e-6:
            best = {
                "score": score, "epoch": epoch,
                "metrics": metrics, "model": copy.deepcopy(model.state_dict()),
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best["model"], "model_config": model_config,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "protocol": args.exp, "seed": args.seed,
        "train_leave_self_out": True,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "external_action_checkpoint": (
            report_path(args.external_action_checkpoint)
            if args.external_action_checkpoint is not None else None
        ),
        "contact_profile_checkpoint": (
            report_path(args.contact_profile_checkpoint)
            if args.contact_profile_checkpoint is not None else None
        ),
        "blend_strength": args.blend_strength,
        "include_selector_distance": args.include_selector_distance,
    }, args.run_dir / "best_model.pt")
    result = {
        "status": "validation_selected_profile_candidate_ranker",
        "protocol": args.exp,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "history": history,
        "train_leave_self_out": True,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "action_calibration": report_path(args.action_calibration),
    }
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
