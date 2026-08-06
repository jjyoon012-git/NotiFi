"""One-shot fixed-test evaluation of validation-selected KP5-MPR-GR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import (
    GeometryResidualReranker,
    TemporalMotionSelector,
    geometric_pair_features,
)
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .calibrate_motion_retrieval_selector import exact_pose_distance
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool
from .train_motion_retrieval_selector import predict_selector
from .train_geometry_residual_reranker import geometry_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_geometry_seed47" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_geometry_seed47" / "test_fixed.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geometry_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if geometry_checkpoint["selection"]["name"] != "t075_top2_625":
        raise RuntimeError("checkpoint is not the locked validation selection")
    selector_path = Path(geometry_checkpoint["selector_checkpoint"])
    selector_checkpoint = torch.load(
        selector_path, map_location="cpu", weights_only=False
    )
    selector = TemporalMotionSelector(**selector_checkpoint["model_config"]).to(device)
    selector.load_state_dict(selector_checkpoint["model"])
    model = GeometryResidualReranker(**geometry_checkpoint["model_config"]).to(device)
    model.load_state_dict(geometry_checkpoint["model"])
    model.eval()

    root = selector_path.parent
    cache = torch.load(
        root / "test_features.pt", map_location="cpu", weights_only=False
    )
    selector_output = predict_selector(selector, cache, 64, device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    test = QualityWeightedDataset(
        pose_only(datasets["test"]), protocol_audit_path(args.exp)
    )
    target_pose, target_valid, _, target_risk = _load_pose_arrays(test)
    train = QualityWeightedDataset(
        pose_only(datasets["train"]), protocol_audit_path(args.exp)
    )
    _, _, train_class, _ = _load_pose_arrays(train)
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
        baseline_bank, train_bank, root / "test_exact_pose_distance.pt"
    )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, train_class, fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
    )
    pairs = geometric_pair_features(
        baseline_bank, train_bank[pool["indices"]]
    )
    logits = []
    with torch.no_grad():
        for start in range(0, len(test), 64):
            indices = torch.arange(start, min(start + 64, len(test)))
            logits.append(model(*geometry_inputs(
                pool, selector_output, selector_checkpoint,
                risk_probability, pairs, indices, device,
            )).float().cpu())
    logits = torch.cat(logits)
    top = logits.topk(2, dim=-1).indices
    probability = torch.softmax(logits / 0.75, dim=-1)
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
        "kp5_mpr_gr": _metric_batch(
            predicted, target_pose, target_valid, target_risk
        ),
    }
    result = {
        "status": "fixed_test_complete_no_test_tuning",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection_source": "validation",
        "fixed_configuration": geometry_checkpoint["selection"],
        "metrics": metrics,
        "scores": {
            name: pose_selection_score(value) for name, value in metrics.items()
        },
        "checkpoint": report_path(args.checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
