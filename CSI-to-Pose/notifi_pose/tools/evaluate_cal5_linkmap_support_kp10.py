"""Combine support-selected TX mapping with gated CAL3 feature adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from .diagnose_observability import pose_only
from .evaluate_cal4_linkmap_kp10 import (
    extract_features,
    load_classifier,
    load_coarse,
    select_permutation,
)
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import (
    add_paths,
    configure_work_root,
    split_support_query,
)
from .train_cal3_kp10 import (
    apply_adapter,
    class_prototypes,
    classification_with_preserved_risk,
    evaluate_pose,
    fit_site_adapter,
    load_heads,
)
from .train_calibration_aware_v14 import subset_dataset
from .audit_motion_retrieval_oracle import _load_pose_arrays


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--feature-noise", type=float, default=0.01)
    parser.add_argument("--temporal-drop", type=float, default=0.02)
    parser.add_argument("--feature-anchor", type=float, default=0.20)
    parser.add_argument("--parameter-anchor", type=float, default=0.02)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--adapter-gate", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--yja-coarse", type=Path,
        default=default_work / "runs/cal2_kp10_seed223_danger_gate"
        / "yja_e02_v13s_coarse.pt",
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal5_linkmap_support_kp10"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = build_datasets(exp=args.exp, baseline="sub", seed=17)
    train = QualityWeightedDataset(pose_only(source["train"]), None)
    train_cache = torch.load(
        args.selector_checkpoint.parent / "train_features.pt",
        map_location="cpu", weights_only=False,
    )
    _, _, train_action, _ = _load_pose_arrays(train)
    classifier, selector = load_heads(args, device)
    prototypes = class_prototypes(classifier, train_cache, train_action, device)

    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline="sub", seed=args.seed
    )["test"]
    full = QualityWeightedDataset(sealed, None)
    support, query = split_support_query(
        sealed.index, ("yja_E02",),
        args.support_per_class, args.seed + 15,
    )
    coarse = load_coarse(args.yja_coarse)
    kp4, _ = _load_model(
        args.work_root / "runs/kp4_dcc_staged_seed17/deployment_model.pt", device
    )
    permutation_classifier = load_classifier(args, device)
    selected, candidates = select_permutation(
        kp4, permutation_classifier, full, support["yja_E02"], coarse, device
    )
    mapping = {"yja_E02": selected}
    support_dataset = QualityWeightedDataset(
        subset_dataset(sealed, support["yja_E02"]), None
    )
    support_cache = extract_features(
        kp4, support_dataset, coarse, mapping,
        device, "CAL5 yja mapped support",
    )
    labels = torch.tensor(
        sealed.index.iloc[support["yja_E02"]].class_id.to_numpy(dtype=np.int64)
    )
    best_support_accuracy = float(candidates[0]["accuracy"])
    adapter_used = best_support_accuracy < args.adapter_gate
    if adapter_used:
        adapter, fit_audit = fit_site_adapter(
            args, support_cache, labels, prototypes,
            classifier, selector, device,
        )
    else:
        adapter = None
        fit_audit = {
            "skipped": True,
            "reason": "mapped support accuracy passed adapter gate",
        }

    query_dataset = QualityWeightedDataset(
        subset_dataset(sealed, query), None
    )
    mapped_query = extract_features(
        kp4, query_dataset, coarse, mapping,
        device, "CAL5 yja mapped query",
    )
    adapted_query = (
        apply_adapter(adapter, mapped_query, device)
        if adapter is not None else mapped_query
    )
    action = torch.tensor(
        sealed.index.iloc[query].class_id.to_numpy(dtype=np.int64)
    )
    risk = torch.tensor(
        sealed.index.iloc[query].risk_id.to_numpy(dtype=np.int64)
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[query].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, query[pose_local]), None
    )
    local = torch.from_numpy(pose_local).long()
    mapped_pose, cal5_pose = evaluate_pose(
        args, pose_target,
        {key: value.index_select(0, local)
         if torch.is_tensor(value) and value.ndim and len(value) == len(query)
         else value for key, value in mapped_query.items()},
        {key: value.index_select(0, local)
         if torch.is_tensor(value) and value.ndim and len(value) == len(query)
         else value for key, value in adapted_query.items()},
        args.run_dir / "yja_exact_distance.pt", device,
    )
    mapped_classification = classification_with_preserved_risk(
        args, mapped_query, mapped_query, action, risk, device
    )
    cal5_classification = classification_with_preserved_risk(
        args, adapted_query, mapped_query, action, risk, device
    )
    cal4_result = json.loads((
        args.work_root / "runs/cal4_linkmap_kp10/result.json"
    ).read_text(encoding="utf-8"))
    source_identity = all(
        value == [0, 1, 2]
        for value in cal4_result["source_meta_validation"]["mapping"].values()
    )
    result = {
        "run": "CAL5-LINKMAP-SUPPORT-KP10",
        "contract": {
            "link_mapping_selected_from_support_action_only": True,
            "feature_adapter_uses_support_action_only": True,
            "support_pose_gt_used": False,
            "query_labels_or_gt_used_for_adaptation": False,
            "adapter_gate": args.adapter_gate,
        },
        "source_audit": {
            "all_held_sites_selected_identity": source_identity,
            "adapter_policy": "skip when mapped support accuracy >= gate",
        },
        "yja_e02": {
            "selected_mapping": list(selected),
            "mapped_support_accuracy": best_support_accuracy,
            "adapter_used": adapter_used,
            "support_fit": fit_audit,
            "mapped_pose": mapped_pose,
            "cal5_pose": cal5_pose,
            "mapped_classification": mapped_classification,
            "cal5_classification": cal5_classification,
        },
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
