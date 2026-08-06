"""Evaluate support-selected TX mapping before frozen KP4 and KP10."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_kp10_paired_bootstrap import kp10_prediction
from .audit_motion_retrieval_oracle import _metric_batch
from .diagnose_observability import pose_only
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import (
    META_VALIDATION_SITES,
    add_paths,
    classifier_evaluation,
    configure_work_root,
    prepare_custom,
    site_names,
    slice_cache,
    split_support_query,
)
from .train_calibration_aware_v14 import subset_dataset
from .train_kinetic_pose import CoarsePoseStore, pose_selection_score


def load_coarse(path: Path) -> CoarsePoseStore:
    cached = torch.load(path, map_location="cpu", weights_only=False)
    return CoarsePoseStore(cached["rows"], cached["pose"])


def load_classifier(args, device: str):
    checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    model = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def select_permutation(kp4, classifier, dataset, positions,
                       coarse, device: str) -> tuple[tuple[int, ...], list[dict]]:
    samples = [dataset[int(position)] for position in positions]
    csi = torch.stack([sample["csi"] for sample in samples]).to(device)
    mask = torch.stack([sample["link_mask"] for sample in samples]).to(device)
    rows = torch.stack([sample["row"] for sample in samples]).long()
    labels = torch.stack([sample["class_id"] for sample in samples]).long().to(device)
    coarse_pose = coarse.lookup(rows, device)
    candidates = []
    for permutation in itertools.permutations(range(C.N_LINKS)):
        order = torch.tensor(permutation, device=device)
        current_mask = mask.index_select(2, order)
        output = kp4(
            csi.index_select(2, order), current_mask, coarse_pose
        )
        classified = classifier(
            output["conditioned_features"], current_mask.any(-1)
        )
        logits = (
            1.50 * output["action_logits"]
            + 0.75 * classified["action_logits"]
        )
        candidates.append({
            "permutation": tuple(int(value) for value in permutation),
            "accuracy": float((logits.argmax(-1) == labels).float().mean()),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
        })
    candidates.sort(key=lambda value: (-value["accuracy"], value["cross_entropy"]))
    return candidates[0]["permutation"], candidates


@torch.no_grad()
def extract_features(kp4, dataset, coarse, permutation_by_site: dict,
                     device: str, protocol: str) -> dict:
    names = site_names(dataset.index)
    row_site = {
        int(row): names[position]
        for position, row in enumerate(dataset.rows)
    }
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    keys = (
        "features", "frame_mask", "baseline_pose", "base_action_logits",
        "base_risk_logits", "contact_logits", "phase_logits",
        "motion_activity", "rows",
    )
    values = {key: [] for key in keys}
    for batch in loader:
        rows = batch["row"].long()
        sites = [row_site[int(row)] for row in rows.tolist()]
        batch_output: dict[str, torch.Tensor] = {}
        for site in sorted(set(sites)):
            local = torch.tensor([
                index for index, value in enumerate(sites) if value == site
            ], dtype=torch.long)
            permutation = torch.tensor(
                permutation_by_site[site], dtype=torch.long, device=device
            )
            csi = batch["csi"].index_select(0, local).to(device).index_select(
                2, permutation
            )
            mask = batch["link_mask"].index_select(0, local).to(device).index_select(
                2, permutation
            )
            selected_rows = rows.index_select(0, local)
            output = kp4(csi, mask, coarse.lookup(selected_rows, device))
            current = {
                "features": output["conditioned_features"],
                "frame_mask": mask.any(-1),
                "baseline_pose": output["pose_rel"],
                "base_action_logits": output["action_logits"],
                "base_risk_logits": output["risk_logits"],
                "contact_logits": output["contact_logits"],
                "phase_logits": output["phase_logits"],
                "motion_activity": output["motion_activity"],
            }
            for key, value in current.items():
                if key not in batch_output:
                    batch_output[key] = torch.empty(
                        (len(rows),) + tuple(value.shape[1:]),
                        device=value.device, dtype=value.dtype,
                    )
                batch_output[key].index_copy_(0, local.to(device), value)
        values["features"].append(batch_output["features"].cpu().half())
        values["frame_mask"].append(batch_output["frame_mask"].cpu())
        values["baseline_pose"].append(batch_output["baseline_pose"].cpu().half())
        values["base_action_logits"].append(
            batch_output["base_action_logits"].cpu().half()
        )
        values["base_risk_logits"].append(
            batch_output["base_risk_logits"].cpu().half()
        )
        values["contact_logits"].append(batch_output["contact_logits"].cpu().half())
        values["phase_logits"].append(batch_output["phase_logits"].cpu().half())
        values["motion_activity"].append(
            batch_output["motion_activity"].cpu().half()
        )
        values["rows"].append(rows)
    result = {key: torch.cat(items) for key, items in values.items()}
    result.update({"protocol": protocol, "source": "CAL4 support-selected TX mapping"})
    return result


def evaluate_variant(args, target, cache, distance_path, device: str) -> tuple[dict, dict]:
    data = prepare_custom(args, target, cache, distance_path, device)
    pose = kp10_prediction(data, args, device).cpu()
    return (
        _metric_batch(
            pose, data["target_pose"], data["target_valid"], data["target_risk"]
        ),
        classifier_evaluation(
            args, cache, data["target_class"], data["target_risk"], device
        ),
    )


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--source-coarse", type=Path,
        default=default_work / "runs/kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--yja-coarse", type=Path,
        default=default_work / "runs/cal2_kp10_seed223_danger_gate"
        / "yja_e02_v13s_coarse.pt",
    )
    parser.add_argument(
        "--yja-feature-cache", type=Path,
        default=Path(r"C:\Users\jjeong\Documents\Playground"
                     r"\kp10_yja_e02_zero_shot_local\yja_e02_features.pt"),
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal4_linkmap_kp10"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kp4, _ = _load_model(
        args.work_root / "runs/kp4_dcc_staged_seed17/deployment_model.pt", device
    )
    classifier = load_classifier(args, device)

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    audit = protocol_audit_path(args.exp)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    source_support, source_query = split_support_query(
        validation.index, META_VALIDATION_SITES,
        args.support_per_class, args.seed + 17,
    )
    source_coarse = load_coarse(args.source_coarse)
    source_mapping, source_candidates = {}, {}
    for site in META_VALIDATION_SITES:
        selected, candidates = select_permutation(
            kp4, classifier, validation, source_support[site],
            source_coarse, device,
        )
        source_mapping[site] = selected
        source_candidates[site] = candidates
    source_target = QualityWeightedDataset(
        subset_dataset(validation.target, source_query), audit
    )
    source_cache = extract_features(
        kp4, source_target, source_coarse, source_mapping,
        device, "CAL4 source held-site query",
    )
    source_pose, source_classification = evaluate_variant(
        args, source_target, source_cache,
        args.run_dir / "source_exact_distance.pt", device,
    )

    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline="sub", seed=args.seed
    )["test"]
    full = QualityWeightedDataset(sealed, None)
    yja_support, yja_query = split_support_query(
        sealed.index, ("yja_E02",),
        args.support_per_class, args.seed + 15,
    )
    yja_coarse = load_coarse(args.yja_coarse)
    yja_selected, yja_candidates = select_permutation(
        kp4, classifier, full, yja_support["yja_E02"], yja_coarse, device
    )
    query_all = QualityWeightedDataset(
        subset_dataset(sealed, yja_query), None
    )
    mapped_all = extract_features(
        kp4, query_all, yja_coarse, {"yja_E02": yja_selected},
        device, "CAL4 yja/E02 support-selected mapping",
    )
    base_full = torch.load(
        args.yja_feature_cache, map_location="cpu", weights_only=False
    )
    base_all = slice_cache(base_full, torch.from_numpy(yja_query).long())
    action = torch.tensor(
        sealed.index.iloc[yja_query].class_id.to_numpy(dtype=np.int64)
    )
    risk = torch.tensor(
        sealed.index.iloc[yja_query].risk_id.to_numpy(dtype=np.int64)
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[yja_query].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, yja_query[pose_local]), None
    )
    local = torch.from_numpy(pose_local).long()
    base_pose, _ = evaluate_variant(
        args, pose_target, slice_cache(base_all, local),
        args.run_dir / "yja_exact_distance.pt", device,
    )
    mapped_pose, _ = evaluate_variant(
        args, pose_target, slice_cache(mapped_all, local),
        args.run_dir / "yja_exact_distance.pt", device,
    )
    base_classification = classifier_evaluation(
        args, base_all, action, risk, device
    )
    mapped_classification = classifier_evaluation(
        args, mapped_all, action, risk, device
    )
    result = {
        "run": "CAL4-LINKMAP-KP10",
        "contract": {
            "permutation_selected_from_support_action_only": True,
            "support_pose_gt_used": False,
            "query_labels_or_gt_used_for_selection": False,
        },
        "source_meta_validation": {
            "mapping": {key: list(value) for key, value in source_mapping.items()},
            "candidates": source_candidates,
            "pose": source_pose,
            "classification": source_classification,
        },
        "yja_e02": {
            "selected_mapping": list(yja_selected),
            "support_candidates": yja_candidates,
            "base_pose": base_pose,
            "mapped_pose": mapped_pose,
            "base_classification": base_classification,
            "mapped_classification": mapped_classification,
        },
    }
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "source_mapping": result["source_meta_validation"]["mapping"],
        "yja_mapping": result["yja_e02"]["selected_mapping"],
        "yja_base_pose": base_pose,
        "yja_mapped_pose": mapped_pose,
        "yja_base_classification": base_classification,
        "yja_mapped_classification": mapped_classification,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
