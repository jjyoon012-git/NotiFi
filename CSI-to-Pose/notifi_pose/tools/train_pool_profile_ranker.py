"""Train a listwise ranker over the frozen CSI top-20 candidate pool."""

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
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _canonicalize, _metric_batch, _render
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_independent_risk_fusion import classifier_outputs
from .calibrate_motion_profile_reranking import standardize
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .diagnose_observability import report_path
from .train_kinetic_pose import DISTAL_JOINTS, pose_selection_score


def class_risk_map(train_data):
    mapping = torch.zeros(C.N_CLASSES, dtype=torch.long)
    for class_id in range(C.N_CLASSES):
        keep = train_data["target_class"] == class_id
        if keep.any():
            mapping[class_id] = torch.mode(train_data["target_risk"][keep]).values
    return mapping


def candidate_features(data, risk_map):
    indices = data["pool"]["indices"]
    batch, candidates = indices.shape
    bank = data["train_bank"].index_select(0, indices.flatten()).reshape(
        batch, candidates, C.CACHE_FRAMES, C.N_JOINTS, 3
    )
    pose = torch.linalg.vector_norm(
        bank - data["baseline_bank"][:, None], dim=-1
    ).mean((2, 3))
    scalar = data["scalar_distance"]
    parts = data["part_distance"]
    parts = (parts - parts.mean(1, keepdim=True)) / parts.std(
        1, keepdim=True
    ).clamp_min(1e-5)
    reranker = standardize(-data["logits"])
    train_embedding = data["checkpoint"]["train_embedding"].index_select(
        0, indices.flatten()
    ).reshape(batch, candidates, -1)
    selector = torch.linalg.vector_norm(
        train_embedding - data["selector_embedding"][:, None], dim=-1
    )
    candidate_class = data["checkpoint"]["train_class"].index_select(
        0, indices.flatten()
    ).reshape(batch, candidates)
    action_probability = torch.softmax(data["fused_action"], dim=-1)
    action = -torch.log(action_probability.gather(
        1, candidate_class
    ).clamp_min(1e-6))
    candidate_risk = risk_map.index_select(
        0, candidate_class.flatten()
    ).reshape(batch, candidates)
    risk = -torch.log(data["risk_probability"].gather(
        1, candidate_risk
    ).clamp_min(1e-6))
    values = torch.cat((
        standardize(pose)[..., None],
        standardize(scalar)[..., None],
        parts,
        reranker[..., None],
        standardize(selector)[..., None],
        standardize(action)[..., None],
        standardize(risk)[..., None],
    ), dim=-1)
    return values, candidate_class, bank


def candidate_cost(data, bank):
    target = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(data["target_pose"], data["target_valid"])
    ])
    error = torch.linalg.vector_norm(bank - target[:, None], dim=-1)
    overall = error.mean((2, 3))
    distal = error[..., list(DISTAL_JOINTS)].mean((2, 3))
    speed = torch.linalg.vector_norm(
        target[:, 1:] - target[:, :-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    high = speed >= torch.quantile(speed, 0.75, dim=1, keepdim=True)
    high_error = (error[:, :, 1:].mean(-1) * high[:, None]).sum(2) / high.sum(
        1
    ).clamp_min(1)[:, None]
    cost = overall + 0.45 * distal + 0.30 * high_error
    danger = data["target_risk"] == 2
    cost[danger] += (
        0.65 * overall[danger] + 0.55 * distal[danger]
        + 0.45 * high_error[danger]
    )
    return (cost - cost.mean(1, keepdim=True)) / cost.std(
        1, keepdim=True
    ).clamp_min(1e-5)


def train_epoch(model, features, classes, cost, loader, optimizer, device):
    model.train()
    losses = []
    for (indices,) in loader:
        score = model(
            features.index_select(0, indices).to(device),
            classes.index_select(0, indices).to(device),
        )
        target_cost = cost.index_select(0, indices).to(device)
        target = torch.softmax(-target_cost / 0.50, dim=-1)
        listwise = -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()
        hard = F.cross_entropy(score, target_cost.argmin(-1))
        loss = listwise + 0.20 * hard
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def render(model, data, features, classes, device, top_k=2):
    model.eval()
    score = []
    for start in range(0, len(features), 64):
        stop = min(start + 64, len(features))
        score.append(model(
            features[start:stop].to(device), classes[start:stop].to(device)
        ).float().cpu())
    score = torch.cat(score)
    top = score.topk(top_k, dim=-1).indices
    weight = torch.softmax(score.gather(1, top), dim=-1)
    motions = []
    inference_valid = data["inference_valid"]
    for item, valid in enumerate(inference_valid):
        bank_indices = data["pool"]["indices"][item].gather(0, top[item])
        canonical = (
            data["train_bank"].index_select(0, bank_indices)
            * weight[item, :, None, None, None]
        ).sum(0)
        motions.append(_render(canonical, valid, C.CACHE_FRAMES))
    return torch.stack(motions)


@torch.no_grad()
def evaluate_pose(model, data, features, classes, current, device):
    inference_valid = data["inference_valid"]
    prior = render(model, data, features, classes, device)
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    prior = monotonic_energy_warp(
        prior, activity, inference_valid, 0.50, 0.30
    )
    low = smooth_valid_delta(prior - current, inference_valid, 17)
    predicted = current + 0.45 * low
    return _metric_batch(
        predicted, data["target_pose"], data["target_valid"],
        data["target_risk"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=293)
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp18_pool_profile_ranker_seed293",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptive = json.loads(
        args.adaptive_calibration.read_text(encoding="utf-8")
    )["selection"]
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    classifier = TemporalMotionSelector(
        **classifier_checkpoint["model_config"]
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    train = prepare(args, "train", device)
    validation = prepare(args, "val", device)
    for data in (train, validation):
        action, _ = classifier_outputs(classifier, data["cache"], device)
        data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * action
    risk_map = class_risk_map(train)
    train_features, train_classes, train_bank = candidate_features(train, risk_map)
    val_features, val_classes, _ = candidate_features(validation, risk_map)
    cost = candidate_cost(train, train_bank)
    model_config = {"feature_dim": train_features.shape[-1]}
    model = ProfileCandidateRanker(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    sample_weight = torch.where(
        train["target_risk"] == 2,
        torch.tensor(2.5, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_features))),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sample_weight, len(sample_weight), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ), num_workers=0,
    )
    current = predict_locked(validation, adaptive)
    best, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_features, train_classes, cost,
            loader, optimizer, device,
        )
        metrics = evaluate_pose(
            model, validation, val_features, val_classes, current, device
        )
        score = pose_selection_score(metrics)
        row = {"epoch": epoch, "loss": loss, "score": score, "metrics": metrics}
        history.append(row)
        print(json.dumps({
            "epoch": epoch, "loss": loss, "score": score,
            "danger_pose_m": metrics["danger_pose_mpjpe_m"],
            "danger_distal_m": metrics["danger_distal_mpjpe_m"],
        }), flush=True)
        if best is None or score < best["score"] - 1e-6:
            best = {
                "epoch": epoch, "score": score, "metrics": metrics,
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
        "pool_top_k": 20, "train_leave_self_out": True,
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }, args.run_dir / "best_model.pt")
    result = {
        "status": "validation_selected_pool_profile_ranker",
        "protocol": args.exp,
        "selection": {k: v for k, v in best.items() if k != "model"},
        "history": history,
        "train_leave_self_out": True,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
