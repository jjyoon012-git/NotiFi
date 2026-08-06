"""Validation-only kinematic projection of the promoted KP5 motion mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import CandidateMotionReranker, TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import _canonicalize, _load_pose_arrays, _metric_batch
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .calibrate_partwise_motion_blend import build_candidate
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool
from .train_motion_retrieval_selector import predict_selector


def kinematic_projection(predicted: torch.Tensor, baseline: torch.Tensor,
                         valid: torch.Tensor) -> torch.Tensor:
    bones = torch.zeros_like(predicted)
    baseline_bones = torch.zeros_like(baseline)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            bones[:, :, child] = predicted[:, :, child] - predicted[:, :, parent]
            baseline_bones[:, :, child] = baseline[:, :, child] - baseline[:, :, parent]
    directions = F.normalize(bones, dim=-1)
    lengths = torch.linalg.vector_norm(baseline_bones, dim=-1)
    trial_lengths = []
    for item, mask in enumerate(valid):
        trial_lengths.append(lengths[item, mask].median(0).values)
    lengths = torch.stack(trial_lengths)[:, None, :, None]
    projected_bones = directions * lengths
    joints = []
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent < 0:
            joints.append(torch.zeros_like(projected_bones[:, :, child]))
        else:
            joints.append(joints[parent] + projected_bones[:, :, child])
    return torch.stack(joints, dim=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "kinematic_projection_calibration.json",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.split == "test" and not args.output.name.startswith("test_"):
        raise RuntimeError("test output must be explicitly named test_*")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    selector.load_state_dict(checkpoint["model"])
    reranker_checkpoint = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    reranker = CandidateMotionReranker(**reranker_checkpoint["model_config"]).to(device)
    reranker.load_state_dict(reranker_checkpoint["model"])
    reranker.eval()
    root = args.selector_checkpoint.parent
    cache = torch.load(
        root / f"{args.split}_features.pt", map_location="cpu", weights_only=False
    )
    selector_output = predict_selector(selector, cache, 64, device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    target_dataset = QualityWeightedDataset(pose_only(datasets[args.split]), audit)
    _, _, train_class, _ = _load_pose_arrays(train)
    target_pose, target_valid, _, target_risk = _load_pose_arrays(target_dataset)
    target_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(target_pose, target_valid)
    ])
    baseline = cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, target_valid)
    ])
    train_bank = checkpoint["train_bank"].float()
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    distance = exact_pose_distance(
        baseline_bank, train_bank, root / f"{args.split}_exact_pose_distance.pt"
    )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, train_class, fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
    )
    candidate = build_candidate(
        checkpoint, reranker, selector_output, cache, pool,
        risk_probability, target_valid, device,
    )
    promoted = 0.375 * baseline + 0.625 * candidate
    projected = kinematic_projection(promoted, baseline, target_valid)
    metrics = {
        "uniform_625": _metric_batch(
            promoted, target_pose, target_valid, target_risk
        )
    }
    strengths = (0.0, 0.25, 0.50, 0.75, 1.0) if args.split == "val" else ()
    if args.split == "test":
        calibration = json.loads(
            (root / "kinematic_projection_calibration.json").read_text(encoding="utf-8")
        )
        strengths = (float(calibration["selection"]["strength"]),)
    for strength in strengths:
        metrics[f"projection_{int(strength * 1000):04d}"] = _metric_batch(
            promoted + strength * (projected - promoted),
            target_pose, target_valid, target_risk,
        )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best = min(scores, key=scores.get)
    result = {
        "status": (
            "validation_selected_kinematic_projection" if args.split == "val"
            else "fixed_test_complete_no_test_tuning"
        ),
        "protocol": args.exp,
        "split": args.split,
        "test_used_for_selection": False,
        "selection": {
            "name": best,
            "strength": float(best.split("_")[-1]) / 1000 if best.startswith("projection") else 0.0,
            "score": scores[best],
        },
        "scores": scores,
        "metrics": metrics,
        "length_source": "per-trial median frozen-baseline bone length",
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
