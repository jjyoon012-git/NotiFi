"""CAL60 학습 sampler가 16개 query action을 실제로 균형화하는지 감사한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from train_cal20_source_folds import ACTION_CLASSES, SOURCE_SITES  # noqa: E402


def main() -> None:
    """7개 source site와 5개 seed의 재표집 class count를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    options = parser.parse_args()

    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja entered sampler audit")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    if set(sites.tolist()) != SOURCE_SITES:
        raise RuntimeError("unexpected source site contract")

    raw_counts = np.bincount(
        selected.class_id.to_numpy(dtype=np.int64), minlength=C.N_CLASSES
    )
    raw_active = raw_counts[list(ACTION_CLASSES)].astype(np.float64)
    episodes = []
    pooled = np.zeros(C.N_CLASSES, dtype=np.int64)
    for seed in options.seeds:
        for site in sorted(SOURCE_SITES):
            rows = base.site_rows(selected_rows, sites, site)
            batches = base.balanced_batches(
                rows, index, options.batch_size, seed
            )
            sampled = np.concatenate(batches)
            labels = index.class_id.iloc[sampled].to_numpy(dtype=np.int64)
            counts = np.bincount(labels, minlength=C.N_CLASSES)
            pooled += counts
            active = counts[list(ACTION_CLASSES)].astype(np.float64)
            episodes.append({
                "support_seed": int(seed),
                "site": site,
                "draws": int(len(sampled)),
                "counts": {
                    C.ACTION_NAMES[class_id]: int(counts[class_id])
                    for class_id in ACTION_CLASSES
                },
                "coefficient_of_variation": float(
                    active.std() / max(active.mean(), 1e-8)
                ),
            })
    active_pooled = pooled[list(ACTION_CLASSES)].astype(np.float64)
    result = {
        "run": "A59-CAL60-TRAINING-SAMPLER-BALANCE-AUDIT",
        "source_sites": sorted(SOURCE_SITES),
        "seeds": [int(seed) for seed in options.seeds],
        "episodes": episodes,
        "raw_counts": {
            C.ACTION_NAMES[class_id]: int(raw_counts[class_id])
            for class_id in ACTION_CLASSES
        },
        "raw_max_to_min_ratio": float(
            raw_active.max() / max(raw_active.min(), 1.0)
        ),
        "pooled_counts": {
            C.ACTION_NAMES[class_id]: int(pooled[class_id])
            for class_id in ACTION_CLASSES
        },
        "pooled_minimum": int(active_pooled.min()),
        "pooled_maximum": int(active_pooled.max()),
        "pooled_max_to_min_ratio": float(
            active_pooled.max() / max(active_pooled.min(), 1.0)
        ),
        "mean_episode_coefficient_of_variation": float(np.mean([
            episode["coefficient_of_variation"] for episode in episodes
        ])),
        "absence_in_query_sampler": int(pooled[C.ACTION_TO_ID["absence"]]),
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "pooled_minimum": result["pooled_minimum"],
        "pooled_maximum": result["pooled_maximum"],
        "raw_max_to_min_ratio": result["raw_max_to_min_ratio"],
        "pooled_max_to_min_ratio": result["pooled_max_to_min_ratio"],
        "mean_episode_cv": result["mean_episode_coefficient_of_variation"],
        "absence": result["absence_in_query_sampler"],
    }, indent=2))


if __name__ == "__main__":
    main()
