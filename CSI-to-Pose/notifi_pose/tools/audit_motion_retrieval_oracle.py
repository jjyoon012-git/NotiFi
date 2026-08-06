"""Measure the train-only motion-bank ceiling without touching the test split.

This is a diagnostic, not a deployable predictor.  The oracle variants use the
validation pose only to choose among motions that already exist in training.
Their gap to the frozen CSI baseline tells us whether retrieval/multi-hypothesis
decoding is worth implementing before a heavier generative model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import _aggregate_rows, _pose_rows


PARTS = {
    "head": C.JOINT_GROUPS["head"],
    "torso": C.JOINT_GROUPS["torso"],
    "left_arm": C.JOINT_GROUPS["left_arm"],
    "right_arm": C.JOINT_GROUPS["right_arm"],
    "left_leg": C.JOINT_GROUPS["left_leg"],
    "right_leg": C.JOINT_GROUPS["right_leg"],
}


def _canonicalize(pose: torch.Tensor, valid: torch.Tensor,
                  frames: int) -> torch.Tensor:
    """Resample the valid part of one trajectory to normalized action phase."""
    selected = pose[valid]
    if len(selected) == 0:
        return pose.new_zeros(frames, C.N_JOINTS, 3)
    if len(selected) == 1:
        return selected.expand(frames, -1, -1).clone()
    flat = selected.flatten(1).T[None]
    result = F.interpolate(
        flat, size=frames, mode="linear", align_corners=True
    )[0].T
    return result.reshape(frames, C.N_JOINTS, 3)


def _render(canonical: torch.Tensor, valid: torch.Tensor,
            frames: int) -> torch.Tensor:
    """Render a normalized-phase trajectory on the observed valid frame span."""
    output = canonical.new_zeros(frames, C.N_JOINTS, 3)
    positions = torch.nonzero(valid, as_tuple=False).flatten()
    if len(positions) == 0:
        return output
    if len(positions) == 1:
        output[positions] = canonical[0]
        return output
    flat = canonical.flatten(1).T[None]
    values = F.interpolate(
        flat, size=len(positions), mode="linear", align_corners=True
    )[0].T.reshape(len(positions), C.N_JOINTS, 3)
    output[positions] = values
    return output


def _load_pose_arrays(dataset: QualityWeightedDataset) -> tuple[torch.Tensor, ...]:
    target = dataset.target
    arrays = target.cache.arrays
    rows = target.rows
    pose = torch.from_numpy(np.asarray(arrays["pose_rel"][rows])).float()
    valid = torch.from_numpy(np.asarray(arrays["valid"][rows])).bool()
    classes = torch.from_numpy(
        target.index.class_id.to_numpy(dtype=np.int64)
    )
    risks = torch.from_numpy(target.index.risk_id.to_numpy(dtype=np.int64))
    return pose, valid, classes, risks


def _load_coarse(path: Path, rows: np.ndarray) -> torch.Tensor:
    cached = torch.load(path, map_location="cpu", weights_only=False)
    position = {
        int(row): index for index, row in enumerate(cached["rows"].tolist())
    }
    missing = [int(row) for row in rows if int(row) not in position]
    if missing:
        raise RuntimeError(f"coarse cache lacks {len(missing)} validation rows")
    indices = torch.tensor([position[int(row)] for row in rows])
    return cached["pose"].index_select(0, indices).float()


def _metric_batch(predicted: torch.Tensor, target: torch.Tensor,
                  valid: torch.Tensor, risks: torch.Tensor) -> dict:
    rows = []
    batch_size = 16
    for start in range(0, len(predicted), batch_size):
        stop = min(start + batch_size, len(predicted))
        batch = {
            "pose_rel": target[start:stop],
            "valid": valid[start:stop],
            "risk_id": risks[start:stop],
        }
        rows.extend(_pose_rows(predicted[start:stop], batch))
    return _aggregate_rows(rows)


def _mean_and_medoid(bank: torch.Tensor, classes: torch.Tensor,
                     class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    members = bank[classes == class_id]
    mean = members.mean(0)
    distance = torch.linalg.vector_norm(members - mean[None], dim=-1).mean((1, 2))
    return mean, members[distance.argmin()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase-frames", type=int, default=C.CACHE_FRAMES)
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_prior_oracle_seen"
        / "validation_oracle.json",
    )
    args = parser.parse_args()

    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, train_class, _ = _load_pose_arrays(train)
    val_pose, val_valid, val_class, val_risk = _load_pose_arrays(validation)

    train_bank = torch.stack([
        _canonicalize(pose, valid, args.phase_frames)
        for pose, valid in zip(train_pose, train_valid)
    ])
    val_bank = torch.stack([
        _canonicalize(pose, valid, args.phase_frames)
        for pose, valid in zip(val_pose, val_valid)
    ])

    means: dict[int, torch.Tensor] = {}
    medoids: dict[int, torch.Tensor] = {}
    for class_id in sorted(set(train_class.tolist())):
        means[class_id], medoids[class_id] = _mean_and_medoid(
            train_bank, train_class, class_id
        )

    predictions = {
        "class_mean": [],
        "class_medoid": [],
        "same_class_oracle": [],
        "part_oracle": [],
    }
    selected_distances = []
    for target, valid, class_id in zip(val_bank, val_valid, val_class.tolist()):
        members = train_bank[train_class == class_id]
        joint_error = torch.linalg.vector_norm(members - target[None], dim=-1)
        whole_error = joint_error.mean((1, 2))
        selected_distances.append(float(whole_error.min()))
        whole = members[whole_error.argmin()]
        part = whole.clone()
        for joints in PARTS.values():
            part_error = joint_error[:, :, joints].mean((1, 2))
            part[:, joints] = members[part_error.argmin(), :, joints]
        candidates = {
            "class_mean": means[class_id],
            "class_medoid": medoids[class_id],
            "same_class_oracle": whole,
            "part_oracle": part,
        }
        for name, candidate in candidates.items():
            predictions[name].append(_render(candidate, valid, C.CACHE_FRAMES))

    coarse = _load_coarse(args.coarse_cache, validation.target.rows)
    coarse_bank = torch.stack([
        _canonicalize(pose, valid, args.phase_frames)
        for pose, valid in zip(coarse, val_valid)
    ])
    retrievals = {
        "csi_query_global_retrieval": [],
        "csi_query_label_retrieval": [],
        "csi_query_part_retrieval": [],
    }
    for query, valid, class_id in zip(
        coarse_bank, val_valid, val_class.tolist()
    ):
        global_error = torch.linalg.vector_norm(
            train_bank - query[None], dim=-1
        )
        global_motion = train_bank[global_error.mean((1, 2)).argmin()]
        class_members = train_bank[train_class == class_id]
        class_error = torch.linalg.vector_norm(
            class_members - query[None], dim=-1
        )
        label_motion = class_members[class_error.mean((1, 2)).argmin()]
        part_motion = label_motion.clone()
        for joints in PARTS.values():
            part_error = class_error[:, :, joints].mean((1, 2))
            part_motion[:, joints] = class_members[
                part_error.argmin(), :, joints
            ]
        retrievals["csi_query_global_retrieval"].append(
            _render(global_motion, valid, C.CACHE_FRAMES)
        )
        retrievals["csi_query_label_retrieval"].append(
            _render(label_motion, valid, C.CACHE_FRAMES)
        )
        retrievals["csi_query_part_retrieval"].append(
            _render(part_motion, valid, C.CACHE_FRAMES)
        )
    predictions.update(retrievals)
    metrics = {
        "frozen_csi_baseline": _metric_batch(
            coarse, val_pose, val_valid, val_risk
        )
    }
    for name, values in predictions.items():
        candidate = torch.stack(values)
        metrics[name] = _metric_batch(candidate, val_pose, val_valid, val_risk)
        if name.startswith("csi_query_"):
            for strength in (0.25, 0.50, 0.75):
                blend_name = f"{name}_blend_{int(strength * 100):02d}"
                metrics[blend_name] = _metric_batch(
                    (1.0 - strength) * coarse + strength * candidate,
                    val_pose, val_valid, val_risk,
                )

    baseline = metrics["frozen_csi_baseline"]
    keys = (
        "mpjpe_m", "distal_mpjpe_m", "dynamic_mpjpe_m",
        "high_motion_mpjpe_m", "danger_pose_mpjpe_m",
        "danger_distal_mpjpe_m", "danger_high_motion_mpjpe_m",
    )
    deltas = {
        name: {key: float(value[key] - baseline[key]) for key in keys}
        for name, value in metrics.items() if name != "frozen_csi_baseline"
    }
    result = {
        "status": "validation_only_oracle_diagnostic",
        "protocol": args.exp,
        "test_split_touched": False,
        "train_trials": len(train),
        "validation_trials": len(validation),
        "phase_frames": args.phase_frames,
        "metrics": metrics,
        "delta_vs_frozen_csi_baseline": deltas,
        "same_class_oracle_normalized_distance_m": {
            "mean": float(np.mean(selected_distances)),
            "median": float(np.median(selected_distances)),
            "p90": float(np.quantile(selected_distances, 0.90)),
        },
        "interpretation": (
            "Oracle rows are an upper-bound diagnostic only. They use validation "
            "GT to select from train-only trajectories and are not deployable."
        ),
        "coarse_cache": report_path(args.coarse_cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
