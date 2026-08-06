"""Validation-only contact-consistency reranking for KP5 motion hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from ..conditioned_contact_pose import CONTACT_JOINTS
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
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import pose_selection_score
from .train_motion_candidate_reranker import make_candidate_pool, model_inputs
from .train_motion_retrieval_selector import predict_selector


def pseudo_contact(pose: torch.Tensor) -> torch.Tensor:
    floor = pose[..., C.UP_AXIS].amin(-1, keepdim=True)
    height = pose[..., list(CONTACT_JOINTS), C.UP_AXIS]
    distance = height - floor
    contact = torch.sigmoid((0.12 - distance) / 0.04)
    return F.avg_pool1d(
        contact.transpose(1, 2), 9, stride=1, padding=4
    ).transpose(1, 2)


def standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(-1, keepdim=True)) / values.std(
        -1, keepdim=True
    ).clamp_min(1e-5)


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
        / "contact_reranking_calibration.json",
    )
    args = parser.parse_args()
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
    root = args.selector_checkpoint.parent
    cache = torch.load(root / "val_features.pt", map_location="cpu", weights_only=False)
    selector_output = predict_selector(selector, cache, 64, device)
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    validation = QualityWeightedDataset(
        pose_only(datasets["val"]), protocol_audit_path(args.exp)
    )
    target_pose, target_valid, _, target_risk = _load_pose_arrays(validation)
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
    train_class = selector_checkpoint["train_class"].long()
    fused_action = cache["base_action_logits"].float() + selector_output["action_logits"]
    risk_probability = torch.softmax(
        cache["base_risk_logits"].float() + selector_output["risk_logits"], dim=-1
    )
    distance = exact_pose_distance(
        baseline_bank, train_bank, root / "val_exact_pose_distance.pt"
    )
    pool = make_candidate_pool(
        baseline_bank, target_bank, target_risk,
        train_bank, train_class, fused_action,
        top_k=20, shortlist=100, exact_distance_matrix=distance,
    )
    logits = []
    with torch.no_grad():
        for start in range(0, len(validation), 64):
            indices = torch.arange(start, min(start + 64, len(validation)))
            inputs = tuple(value.to(device) for value in model_inputs(
                pool, selector_output, selector_checkpoint,
                risk_probability, indices,
            ))
            logits.append(reranker(*inputs).float().cpu())
    logits = torch.cat(logits)
    predicted_contact = torch.sigmoid(cache["contact_logits"].float())
    predicted_contact = F.avg_pool1d(
        predicted_contact.transpose(1, 2), 9, stride=1, padding=4
    ).transpose(1, 2)
    contact_scores = []
    for item, valid in enumerate(target_valid):
        candidate_rows = []
        for index in pool["indices"][item]:
            candidate_rows.append(
                _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            )
        candidate = torch.stack(candidate_rows)
        contact = pseudo_contact(candidate)
        mask = valid[None, :, None].to(contact.dtype)
        frame_error = (
            (contact - predicted_contact[item][None]).square() * mask
        ).sum((1, 2)) / mask.sum().clamp_min(1.0) / len(CONTACT_JOINTS)
        peak_error = (
            contact.amax(1) - predicted_contact[item].amax(0)[None]
        ).square().mean(-1)
        contact_scores.append(-(frame_error + 0.50 * peak_error))
    contact_scores = torch.stack(contact_scores)
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    for contact_weight in (0.0, 0.10, 0.25, 0.50, 1.0):
        adjusted = standardize(logits) + (
            contact_weight * risk_probability[:, 2:3]
            * standardize(contact_scores)
        )
        for temperature in (0.20, 0.50, 1.0):
            probability = torch.softmax(adjusted / temperature, dim=-1)
            for top_k in (3, 5):
                top = adjusted.topk(top_k, dim=-1).indices
                weight = probability.gather(1, top)
                weight = weight / weight.sum(1, keepdim=True)
                motions = []
                for item, valid in enumerate(target_valid):
                    indices = pool["indices"][item].gather(0, top[item])
                    canonical = (
                        train_bank.index_select(0, indices)
                        * weight[item, :, None, None, None]
                    ).sum(0)
                    motions.append(_render(canonical, valid, C.CACHE_FRAMES))
                candidate = torch.stack(motions)
                key = (
                    f"c{int(contact_weight * 100):03d}_t{int(temperature * 100):03d}"
                    f"_top{top_k}_625"
                )
                metrics[key] = _metric_batch(
                    0.375 * baseline + 0.625 * candidate,
                    target_pose, target_valid, target_risk,
                )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    result = {
        "status": "validation_selected_contact_reranking",
        "protocol": args.exp,
        "test_used_for_selection": False,
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "contact_definition": (
            "candidate joint height within 0.12m of its per-frame lowest joint"
        ),
        "contact_joints": list(CONTACT_JOINTS),
        "selector_checkpoint": report_path(args.selector_checkpoint),
        "reranker_checkpoint": report_path(args.reranker_checkpoint),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
