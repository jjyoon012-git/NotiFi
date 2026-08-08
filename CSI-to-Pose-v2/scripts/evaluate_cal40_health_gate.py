"""CAL40 두 encoder의 calibration geometry health gate를 source support로 검증한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import select_support_shots  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.deployment import CAL20Deployment  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def summarize(values: list[float], passed: list[bool]) -> dict:
    """geometry error의 요약 통계와 health gate 통과율을 반환한다."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(array.max()),
        "pass_rate": float(np.mean(passed)),
        "false_rejections": int(len(passed) - sum(passed)),
        "episodes": int(len(passed)),
    }


def main() -> None:
    """7 source site의 support seed를 바꿔 두 geometry threshold를 독립 검사한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    options = parser.parse_args()
    runtime = CAL20Deployment.load(str(options.bundle))
    if runtime.secondary_model is None:
        raise RuntimeError("CAL40 health audit requires a secondary encoder")
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter health gate audit")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    if set(sites.tolist()) != SOURCE_SITES:
        raise RuntimeError("unexpected source site contract")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS)
            & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in sorted(SOURCE_SITES)
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    episodes = []
    for seed in options.support_seeds:
        for site in sorted(SOURCE_SITES):
            rows = base.site_rows(selected_rows, sites, site)
            support = select_support_shots(rows, index, seed, 2)
            absence = base.select_absence(
                site, index, seed + 1, trials=options.absence_trials
            )
            support_csi, support_mask = store.get(support, runtime.device)
            absence_csi, absence_mask = store.get(absence, runtime.device)
            labels = index.class_id.iloc[support].to_numpy(dtype=np.int64)
            try:
                calibration = runtime.calibrate(
                    support_csi, support_mask,
                    torch.tensor(labels, device=runtime.device).long(),
                    absence_csi, absence_mask,
                )
            except ValueError as error:
                episodes.append({
                    "support_seed": int(seed),
                    "site": site,
                    "input_quality_pass": False,
                    "input_quality_error": str(error),
                })
                continue
            episodes.append({
                "support_seed": int(seed),
                "site": site,
                "input_quality_pass": True,
                "primary_error": float(calibration.geometry_error),
                "primary_pass": bool(calibration.domain_pass),
                "secondary_error": float(calibration.secondary_geometry_error),
                "secondary_pass": bool(calibration.secondary_domain_pass),
            })
    valid = [row for row in episodes if row["input_quality_pass"]]
    result = {
        "run": "A52-CAL40-DUAL-GEOMETRY-HEALTH-GATE",
        "primary_threshold": runtime.geometry_threshold,
        "secondary_threshold": runtime.secondary_geometry_threshold,
        "primary": summarize(
            [row["primary_error"] for row in valid],
            [row["primary_pass"] for row in valid],
        ),
        "secondary": summarize(
            [row["secondary_error"] for row in valid],
            [row["secondary_pass"] for row in valid],
        ),
        "input_quality": {
            "pass_rate": len(valid) / max(len(episodes), 1),
            "rejections": len(episodes) - len(valid),
            "episodes": len(episodes),
        },
        "episodes": episodes,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "primary": result["primary"],
        "secondary": result["secondary"],
    }, indent=2))


if __name__ == "__main__":
    main()
