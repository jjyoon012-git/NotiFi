"""Learn train-only action-conditioned motion-prior strengths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .. import contract as C
from ..motion_retrieval import ProfileCandidateRanker, TemporalMotionSelector
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import _metric_batch
from .calibrate_core_seed_selection import predict_locked
from .calibrate_frequency_preserving_action_prior import smooth_valid_delta
from .calibrate_independent_risk_fusion import classifier_outputs
from .calibrate_motion_profile_warping import monotonic_energy_warp
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .calibrate_profile_action_retrieval import retrieval_features
from .diagnose_observability import report_path
from .train_kinetic_pose import DISTAL_JOINTS, pose_selection_score
from .train_profile_candidate_ranker import render_ranked_action


def action_strength(probability, raw, minimum=0.20, maximum=0.70):
    per_class = minimum + (maximum - minimum) * torch.sigmoid(raw)
    return probability @ per_class, per_class


@torch.no_grad()
def build_components(args, split, adaptive, classifier, ranker, device):
    data = prepare(args, split, device)
    extra, _ = classifier_outputs(classifier, data["cache"], device)
    data["fused_action"] = 1.50 * data["base_action_logits"] + 0.75 * extra
    inference_valid = data["inference_valid"]
    probability = torch.softmax(data["fused_action"], dim=-1)
    current = predict_locked(data, adaptive)
    self_indices = (
        torch.arange(len(data["train_bank"])) if split == "train" else None
    )
    prior = render_ranked_action(
        ranker, data,
        retrieval_features(data, 3, 1.0, self_indices=self_indices),
        device,
    )
    activity = (
        0.50 * data["predicted_scalar_profile"]
        + 0.50 * data["predicted_part_profile"][..., 2:].mean(-1)
    )
    prior = monotonic_energy_warp(
        prior, activity, inference_valid, 0.50, 0.30
    )
    low = smooth_valid_delta(prior - current, inference_valid, 17)
    return data, probability, current, low


def pose_loss(predicted, target, valid, risk):
    error = torch.linalg.vector_norm(predicted - target, dim=-1)
    speed = torch.zeros_like(valid, dtype=target.dtype)
    speed[:, 1:] = torch.linalg.vector_norm(
        target[:, 1:] - target[:, :-1], dim=-1
    ).mean(-1) * C.TARGET_FPS
    motion = 1.0 + 1.25 * (speed / 0.35).clamp_max(2.0)
    danger = torch.where(risk[:, None] == 2, 1.8, 1.0)
    frame_weight = valid * motion * danger
    joint_weight = target.new_ones(C.N_JOINTS)
    joint_weight[list(DISTAL_JOINTS)] = 1.55
    weight = frame_weight[..., None] * joint_weight
    return (error * weight).sum() / weight.sum().clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_action_arguments(parser)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--regularization", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=277)
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
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp16_action_conditioned_strength"
        / "calibration.json",
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
    train, train_probability, train_current, train_low = build_components(
        args, "train", adaptive, classifier, ranker, device
    )
    validation, val_probability, val_current, val_low = build_components(
        args, "val", adaptive, classifier, ranker, device
    )
    initial = (0.45 - 0.20) / (0.70 - 0.20)
    raw = torch.nn.Parameter(torch.full(
        (C.N_CLASSES,), math.log(initial / (1.0 - initial)), device=device
    ))
    optimizer = torch.optim.AdamW([raw], lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train["target_valid"]), (args.batch_size,),
            generator=generator,
        )
        gpu_indices = indices.to(device)
        probability = train_probability.index_select(0, indices).to(device)
        strength, per_class = action_strength(probability, raw)
        current = train_current.index_select(0, indices).to(device)
        low = train_low.index_select(0, indices).to(device)
        predicted = current + strength[:, None, None, None] * low
        loss = pose_loss(
            predicted,
            train["target_pose"].index_select(0, indices).to(device),
            train["target_valid"].index_select(0, indices).to(device),
            train["target_risk"].index_select(0, indices).to(device),
        )
        regularization = (per_class - 0.45).square().mean()
        total = loss + args.regularization * regularization
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            history.append({
                "step": step, "pose_loss": float(loss.detach()),
                "regularization": float(regularization.detach()),
                "strengths": per_class.detach().cpu().tolist(),
            })
    with torch.no_grad():
        val_strength, per_class = action_strength(
            val_probability.to(device), raw
        )
        learned = val_current + val_strength.cpu()[:, None, None, None] * val_low
        fixed = val_current + 0.45 * val_low
    metrics = {
        "fixed_045": _metric_batch(
            fixed, validation["target_pose"], validation["target_valid"],
            validation["target_risk"],
        ),
        "action_conditioned": _metric_batch(
            learned, validation["target_pose"], validation["target_valid"],
            validation["target_risk"],
        ),
    }
    scores = {
        name: pose_selection_score(value) for name, value in metrics.items()
    }
    selected = min(scores, key=scores.get)
    result = {
        "status": (
            "validation_promoted" if selected == "action_conditioned"
            else "validation_rejected"
        ),
        "protocol": args.exp,
        "selection": {"name": selected, "score": scores[selected]},
        "strengths": per_class.detach().cpu().tolist(),
        "metrics": metrics,
        "scores": scores,
        "history": history,
        "training": {
            "leave_self_out": True,
            "steps": args.steps,
            "regularization": args.regularization,
        },
        "classifier_checkpoint": report_path(args.classifier_checkpoint),
        "ranker_checkpoint": report_path(args.ranker_checkpoint),
        "test_used_for_selection": False,
        "inference_inputs": "CSI and link mask only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"], "selection": result["selection"],
        "scores": scores, "metrics": metrics[selected],
        "strengths": result["strengths"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
