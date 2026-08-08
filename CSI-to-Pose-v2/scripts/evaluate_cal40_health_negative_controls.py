"""CAL40 calibration health gate를 source-only 음성대조군으로 검사한다."""

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


def corrupt(
    name: str,
    support_csi: torch.Tensor,
    support_mask: torch.Tensor,
    labels: torch.Tensor,
    absence_csi: torch.Tensor,
    absence_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """고정 음성대조군을 만들어 calibration 입력 계약 위반을 모사한다."""
    if name == "clean":
        return support_csi, support_mask, labels, absence_csi, absence_mask
    if name == "tx12_swapped":
        order = torch.tensor([1, 0, 2], device=support_csi.device)
        return (
            support_csi.index_select(2, order),
            support_mask.index_select(2, order),
            labels,
            absence_csi.index_select(2, order),
            absence_mask.index_select(2, order),
        )
    if name == "time_reversed":
        return (
            support_csi.flip(1), support_mask.flip(1), labels,
            absence_csi.flip(1), absence_mask.flip(1),
        )
    if name == "prompt_labels_rolled":
        return (
            support_csi, support_mask, labels.roll(2),
            absence_csi, absence_mask,
        )
    if name == "one_link_only":
        support_one = torch.zeros_like(support_mask)
        absence_one = torch.zeros_like(absence_mask)
        support_one[..., 0] = support_mask[..., 0]
        absence_one[..., 0] = absence_mask[..., 0]
        return support_csi, support_one, labels, absence_csi, absence_one
    raise ValueError(f"unknown corruption {name!r}")


def main() -> None:
    """7개 source site의 고정 support에 calibration 오염을 적용해 gate를 감사한다."""
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
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja entered negative-control audit")
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
    conditions = (
        "clean", "tx12_swapped", "time_reversed",
        "prompt_labels_rolled", "one_link_only",
    )
    episodes = []
    for seed in options.support_seeds:
        for site in sorted(SOURCE_SITES):
            rows = base.site_rows(selected_rows, sites, site)
            support = select_support_shots(rows, index, seed, 2)
            absence = base.select_absence(
                site, index, seed + 1, trials=options.absence_trials,
            )
            support_csi, support_mask = store.get(support, runtime.device)
            absence_csi, absence_mask = store.get(absence, runtime.device)
            labels = torch.tensor(
                index.class_id.iloc[support].to_numpy(dtype=np.int64),
                device=runtime.device,
            ).long()
            for condition in conditions:
                inputs = corrupt(
                    condition, support_csi, support_mask, labels,
                    absence_csi, absence_mask,
                )
                try:
                    calibration = runtime.calibrate(*inputs)
                except ValueError as error:
                    episodes.append({
                        "support_seed": int(seed),
                        "site": site,
                        "condition": condition,
                        "accepted": False,
                        "input_rejection": str(error),
                    })
                    continue
                episodes.append({
                    "support_seed": int(seed),
                    "site": site,
                    "condition": condition,
                    "accepted": bool(
                        calibration.domain_pass
                        and calibration.secondary_domain_pass
                    ),
                    "primary_error": float(calibration.geometry_error),
                    "secondary_error": float(calibration.secondary_geometry_error),
                })

    summary = {}
    for condition in conditions:
        rows = [row for row in episodes if row["condition"] == condition]
        summary[condition] = {
            "accepted": int(sum(row["accepted"] for row in rows)),
            "episodes": int(len(rows)),
            "accept_rate": float(np.mean([row["accepted"] for row in rows])),
        }
    result = {
        "run": "A54-CAL40-HEALTH-NEGATIVE-CONTROLS",
        "support_seeds": [int(seed) for seed in options.support_seeds],
        "summary": summary,
        "episodes": episodes,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
