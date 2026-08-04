"""Audit CSI motion alignment against physical danger-event targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import contract as C
from ..dataio.dataset import DropoutConfig, build_datasets
from ..impact_event import physical_impact_targets
from .audit_motion_alignment import (
    csi_motion_energy, lag_correlations, masked_smooth,
)
from .diagnose_observability import pose_only, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--smooth-width", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=C.REPORT_DIR / "impact_alignment_audit.json",
    )
    args = parser.parse_args()
    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=7,
    )
    rows = []
    for split in ("train", "val"):
        dataset = pose_only(datasets[split])
        metadata = dataset.index.reset_index(drop=True)
        danger_indices = np.flatnonzero(metadata.risk_id.to_numpy() == 2)
        for number, index in enumerate(danger_indices, start=1):
            item = dataset[int(index)]
            valid = item["valid"].numpy().astype(bool)
            energy, energy_valid = csi_motion_energy(
                item["csi"].numpy(),
                item["link_mask"].numpy().astype(bool),
                args.smooth_width,
            )
            target = physical_impact_targets(
                item["pose_rel"][None], item["root"][None],
                item["valid"][None].bool(), item["risk_id"][None],
            )
            event_frame = int(target["event_frame"][0])
            physical = target["physical_score"][0].amax(-1).numpy()
            physical = masked_smooth(physical, valid, args.smooth_width)
            comparable = valid & energy_valid
            best, lag, zero = lag_correlations(
                energy, physical, comparable, args.max_lag
            )
            masked_energy = np.where(comparable, energy, -np.inf)
            peak = int(np.argmax(masked_energy))
            top_count = max(1, int(comparable.sum() * 0.05))
            top_frames = np.argpartition(masked_energy, -top_count)[-top_count:]
            top_distance = int(np.abs(top_frames - event_frame).min())
            meta = metadata.iloc[int(index)]
            rows.append({
                "split": split,
                "trial_id": str(meta.trial_id),
                "label": str(meta.detail_label),
                "event_frame": event_frame,
                "csi_peak_frame": peak,
                "csi_peak_error_frames": abs(peak - event_frame),
                "top5pct_distance_frames": top_distance,
                "zero_lag_correlation": zero,
                "best_correlation": best,
                "best_lag_frames": lag,
            })
            if number % 50 == 0 or number == len(danger_indices):
                print(f"{split}: {number}/{len(danger_indices)}", flush=True)
    frame = pd.DataFrame(rows)
    summary = {
        "protocol": "single_split",
        "danger_trials": int(len(frame)),
        "by_split": [],
        "by_label": [],
        "rows": rows,
    }
    for columns, key in ((["split"], "by_split"), (["split", "label"], "by_label")):
        for values, group in frame.groupby(columns):
            if not isinstance(values, tuple):
                values = (values,)
            record = dict(zip(columns, values))
            record.update({
                "trials": int(len(group)),
                "median_csi_peak_error_frames": float(
                    group.csi_peak_error_frames.median()
                ),
                "median_top5pct_distance_frames": float(
                    group.top5pct_distance_frames.median()
                ),
                "median_zero_lag_correlation": float(
                    group.zero_lag_correlation.median()
                ),
                "median_best_correlation": float(group.best_correlation.median()),
                "median_abs_best_lag_frames": float(
                    group.best_lag_frames.abs().median()
                ),
            })
            summary[key].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "protocol": summary["protocol"],
        "danger_trials": summary["danger_trials"],
        "by_split": summary["by_split"],
        "by_label": summary["by_label"],
        "output": report_path(args.output),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
