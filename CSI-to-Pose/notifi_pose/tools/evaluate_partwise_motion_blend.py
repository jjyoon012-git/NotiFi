"""Fixed-test evaluation of validation-selected partwise KP5 blending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import CandidateMotionReranker, TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import _canonicalize, _load_pose_arrays, _metric_batch
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .calibrate_partwise_motion_blend import apply_strength, build_candidate
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool
from .train_motion_retrieval_selector import predict_selector


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
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "partwise_blend_calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17"
        / "partwise_test_fixed.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    strengths = calibration["selection"]["strengths"]
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
    cache = torch.load(root / "test_features.pt", map_location="cpu", weights_only=False)
    selector_output = predict_selector(selector, cache, 64, device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    test = QualityWeightedDataset(pose_only(datasets["test"]), audit)
    _, _, train_class, _ = _load_pose_arrays(train)
    target_pose, target_valid, _, target_risk = _load_pose_arrays(test)
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
        baseline_bank, train_bank, root / "test_exact_pose_distance.pt"
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
    uniform = 0.375 * baseline + 0.625 * candidate
    partwise = apply_strength(baseline, candidate, strengths)
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "kp5_mpr_rm": _metric_batch(
            uniform, target_pose, target_valid, target_risk
        ),
        "kp5_mpr_partwise": _metric_batch(
            partwise, target_pose, target_valid, target_risk
        ),
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation",
        "fixed_strengths": strengths,
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
        "calibration": report_path(args.calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
