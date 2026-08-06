"""Test confirmation for the validation-locked KP5-MPR-R reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import CandidateMotionReranker, TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .calibrate_motion_time_alignment import align_affine
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--reranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--selector-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "best_model.pt",
    )
    parser.add_argument(
        "--feature-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17" / "test_features.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_reranker_seed17" / "test_selected.json",
    )
    args = parser.parse_args()
    if not args.allow_test:
        raise RuntimeError("test confirmation requires explicit --allow-test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selector_checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**selector_checkpoint["model_config"]).to(device)
    selector.load_state_dict(selector_checkpoint["model"])
    reranker_checkpoint = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    reranker = CandidateMotionReranker(
        **reranker_checkpoint["model_config"]
    ).to(device)
    reranker.load_state_dict(reranker_checkpoint["model"])
    reranker.eval()
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    selector_output = predict_selector(selector, cache, 64, device)

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    test = QualityWeightedDataset(
        pose_only(datasets["test"]), protocol_audit_path(args.exp)
    )
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
    train_bank = selector_checkpoint["train_bank"].float()
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    distance = exact_pose_distance(
        baseline_bank, train_bank,
        args.selector_checkpoint.parent / "test_exact_pose_distance.pt",
    )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, selector_checkpoint["train_class"].long(), fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
    )
    logits = []
    with torch.no_grad():
        for start in range(0, len(test), 64):
            indices = torch.arange(start, min(start + 64, len(test)))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, selector_checkpoint,
                risk_probability, indices,
            ))
            logits.append(reranker(*inputs).float().cpu())
    local = torch.cat(logits).argmax(-1)
    reranker_logits = torch.cat(logits)
    reranked_indices = pool["indices"].gather(1, local[:, None]).squeeze(1)
    retrieval_indices = pool["indices"][:, 0]

    def render(indices: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            for index, valid in zip(indices, target_valid)
        ])

    retrieval_pose = 0.5 * baseline + 0.5 * render(retrieval_indices)
    reranked_pose = 0.5 * baseline + 0.5 * render(reranked_indices)
    reranked_candidate = render(reranked_indices)
    aligned_candidates = []
    for candidate, base, valid in zip(
        reranked_candidate, baseline, target_valid
    ):
        frames = torch.nonzero(valid, as_tuple=False).flatten()
        aligned = candidate.clone()
        if len(frames) >= 3:
            aligned[frames] = align_affine(
                candidate[frames], base[frames], 15, (1.0,)
            )
        aligned_candidates.append(aligned)
    aligned_candidate = torch.stack(aligned_candidates)
    aligned_pose = 0.375 * baseline + 0.625 * aligned_candidate
    probability = torch.softmax(reranker_logits / 0.50, dim=-1)
    top3 = reranker_logits.topk(3, dim=-1).indices
    top3_probability = probability.gather(1, top3)
    top3_probability = top3_probability / top3_probability.sum(1, keepdim=True)
    mixture_candidates = []
    for item, valid in enumerate(target_valid):
        bank_indices = pool["indices"][item].gather(0, top3[item])
        canonical = (
            train_bank.index_select(0, bank_indices)
            * top3_probability[item, :, None, None, None]
        ).sum(0)
        mixture_candidates.append(_render(canonical, valid, C.CACHE_FRAMES))
    mixture_candidate = torch.stack(mixture_candidates)
    mixture_pose = 0.375 * baseline + 0.625 * mixture_candidate
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        ),
        "kp5_mpr_s": _metric_batch(
            retrieval_pose, target_pose, target_valid, target_risk
        ),
        "kp5_mpr_r": _metric_batch(
            reranked_pose, target_pose, target_valid, target_risk
        ),
        "kp5_mpr_rt": _metric_batch(
            aligned_pose, target_pose, target_valid, target_risk
        ),
        "kp5_mpr_rm": _metric_batch(
            mixture_pose, target_pose, target_valid, target_risk
        ),
    }
    result = {
        "status": "promoted_candidate_test_confirmation",
        "protocol": args.exp,
        "selection_source": "reranker validation epoch 1, Cartesian 0.50",
        "test_used_for_selection": False,
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "reranker_top1_oracle_accuracy_diagnostic": float(
            (local == pool["target_cost"].argmin(-1)).float().mean()
        ),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
