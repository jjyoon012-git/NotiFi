"""Train a conservative zero-init residual around the locked KP10 pose."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..motion_retrieval import (
    MotionPriorResidualRefiner, ProfileCandidateRanker, TemporalMotionSelector,
)
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_core_seed_selection import predict_locked
from .diagnose_observability import report_path
from .train_action_conditioned_strength import build_components
from .train_csi_contact_profile import contact_features
from .train_kinetic_pose import pose_selection_score
from .train_motion_prior_refiner import training_loss


def pack(data, current, low):
    return {
        "features": contact_features(data["cache"]).float(),
        "base": (current + 0.45 * low).float(),
        "target": data["target_pose"].float(),
        "valid": data["target_valid"].bool(),
        "risk": data["target_risk"].long(),
    }


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    predicted = []
    for start in range(0, len(data["valid"]), 32):
        indices = torch.arange(start, min(start + 32, len(data["valid"])))
        predicted.append(model(
            data["features"].index_select(0, indices).to(device),
            data["base"].index_select(0, indices).to(device),
            data["valid"].index_select(0, indices).to(device),
        )["pose"].float().cpu())
    pose = torch.cat(predicted)
    return _metric_batch(
        pose, data["target"], data["valid"], data["risk"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--max-residual", type=float, default=0.05)
    parser.add_argument(
        "--minimum-score-gain", type=float, default=0.001,
        help="Reject changes smaller than the validation noise floor.",
    )
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--ranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker_seed127"
        / "best_model.pt",
    )
    parser.add_argument(
        "--adaptive-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp6_risk_adaptive_blend"
        / "calibration.json",
    )
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17"
        / "best_model.pt",
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "best_model.pt",
    )
    parser.add_argument(
        "--scalar-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed79"
        / "best_model.pt",
    )
    parser.add_argument(
        "--part-profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_part_motion_profile_seed101"
        / "best_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp19_conservative_refiner_seed307",
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
    ranker_checkpoint = torch.load(
        args.ranker_checkpoint, map_location="cpu", weights_only=False
    )
    ranker = ProfileCandidateRanker(
        **ranker_checkpoint["model_config"]
    ).to(device)
    ranker.load_state_dict(ranker_checkpoint["model"])
    train_data, _, train_current, train_low = build_components(
        args, "train", adaptive, classifier, ranker, device
    )
    val_data, _, val_current, val_low = build_components(
        args, "val", adaptive, classifier, ranker, device
    )
    train = pack(train_data, train_current, train_low)
    validation = pack(val_data, val_current, val_low)
    del train_data, val_data
    model_config = {
        "feature_dim": train["features"].shape[-1],
        "hidden": args.hidden, "layers": args.layers,
        "dropout": args.dropout, "max_residual": args.max_residual,
    }
    model = MotionPriorResidualRefiner(**model_config).to(device)
    identity_metrics = _metric_batch(
        validation["base"], validation["target"],
        validation["valid"], validation["risk"],
    )
    identity_score = pose_selection_score(identity_metrics)
    best = {
        "epoch": 0, "score": identity_score,
        "metrics": identity_metrics,
        "state": copy.deepcopy(model.state_dict()),
    }
    sample_weight = torch.where(
        train["risk"] == 2,
        torch.tensor(2.5, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train["risk"]))),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sample_weight, len(sample_weight), replacement=True,
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
    history, stale = [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (indices,) in loader:
            indices = indices.long()
            feature = train["features"].index_select(0, indices).to(device)
            base = train["base"].index_select(0, indices).to(device)
            target = train["target"].index_select(0, indices).to(device)
            valid = train["valid"].index_select(0, indices).to(device)
            risk = train["risk"].index_select(0, indices).to(device)
            feature = feature + 0.006 * torch.randn_like(feature)
            optimizer.zero_grad(set_to_none=True)
            output = model(feature, base, valid)
            values = training_loss(output, target, valid, risk)
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append([float(value.detach()) for value in values])
        metrics = evaluate(model, validation, device)
        score = pose_selection_score(metrics)
        row = {
            "epoch": epoch,
            "train_loss": float(np.asarray(losses)[:, 0].mean()),
            "score": score, "metrics": metrics,
        }
        history.append(row)
        print(json.dumps({
            "epoch": epoch, "loss": row["train_loss"], "score": score,
            "danger_pose_m": metrics["danger_pose_mpjpe_m"],
            "danger_distal_m": metrics["danger_distal_mpjpe_m"],
        }), flush=True)
        danger_safe = (
            metrics["danger_pose_mpjpe_m"]
            <= identity_metrics["danger_pose_mpjpe_m"]
            and metrics["danger_distal_mpjpe_m"]
            <= identity_metrics["danger_distal_mpjpe_m"]
        )
        if (
            score < identity_score - args.minimum_score_gain
            and danger_safe
            and score < best["score"] - 1e-5
        ):
            best = {
                "epoch": epoch, "score": score, "metrics": metrics,
                "state": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best["state"], "model_config": model_config,
        "selection": {k: v for k, v in best.items() if k != "state"},
        "protocol": args.exp, "seed": args.seed,
        "base": "KP10 strength 0.45, train leave-self-out",
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }, args.run_dir / "best_model.pt")
    result = {
        "status": (
            "validation_promoted" if best["epoch"] > 0
            else "validation_rejected_identity_retained"
        ),
        "protocol": args.exp,
        "selection": {k: v for k, v in best.items() if k != "state"},
        "identity": {"score": identity_score, "metrics": identity_metrics},
        "history": history,
        "train_leave_self_out": True,
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "ranker_checkpoint": report_path(args.ranker_checkpoint),
    }
    (args.run_dir / "train_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"], "selection": result["selection"],
        "identity_score": identity_score,
    }, indent=2))


if __name__ == "__main__":
    main()
