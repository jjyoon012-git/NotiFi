"""One-shot fixed-test evaluation of motion-profile-aware KP5 reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import (
    CandidateMotionReranker,
    MotionProfileHead,
    TemporalMotionSelector,
)
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_profile_reranking import (
    candidate_speed_profiles,
    predict_profile,
    profile_distance,
    standardize,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
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
        "--profile-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71" / "best_model.pt",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71"
        / "reranking_calibration.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_profile_seed71"
        / "test_fixed.json",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration["selection"]["name"] != "m0200_t050_top3_625":
        raise RuntimeError("calibration is not the locked validation selection")
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
    profile_checkpoint = torch.load(
        args.profile_checkpoint, map_location="cpu", weights_only=False
    )
    profile_model = MotionProfileHead(**profile_checkpoint["model_config"]).to(device)
    profile_model.load_state_dict(profile_checkpoint["model"])
    profile_model.eval()
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
    logits = []
    with torch.no_grad():
        for start in range(0, len(test), 64):
            indices = torch.arange(start, min(start + 64, len(test)))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, checkpoint,
                risk_probability, indices,
            ))
            logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    predicted_profile = predict_profile(
        profile_model, cache, target_valid, device
    )
    candidate_profiles = candidate_speed_profiles(
        train_bank, pool, target_valid
    )
    motion_distance = standardize(profile_distance(
        predicted_profile, candidate_profiles, target_valid
    ))

    adjusted = logits - 0.20 * motion_distance
    top = adjusted.topk(3, dim=-1).indices
    probability = torch.softmax(adjusted / 0.50, dim=-1)
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
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "kp5_motion_profile": _metric_batch(
            predicted, target_pose, target_valid, target_risk
        ),
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation",
        "fixed_configuration": calibration["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
        "profile_checkpoint": report_path(args.profile_checkpoint),
        "calibration": report_path(args.calibration),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
