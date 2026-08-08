"""봉인 대상 없이 source CSI만으로 link-coherence 임계값을 감사한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from notifi_pose.linkqc import (  # noqa: E402
    MAX_ALLZERO_RATIO,
    MIN_FRAME_CORR,
    MIN_PKT_MAX,
)


SOURCE_SITES = {
    "ajh_E01", "ajh_E02", "ajh_E03",
    "mhw_E01", "mhw_E02", "mhw_E03",
    "lmh_E01",
}


def main() -> None:
    """원지표에서 source-only quantile과 임계값 아래 링크 수를 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    frame = pd.read_csv(options.link_quality)
    sites = frame.subject.astype(str) + "_" + frame.environment.astype(str)
    source = frame.loc[sites.isin(SOURCE_SITES)].copy()
    selected_sites = set(
        source.subject.astype(str) + "_" + source.environment.astype(str)
    )
    if selected_sites != SOURCE_SITES:
        raise RuntimeError(f"source site mismatch: {sorted(selected_sites)}")
    if "yja" in set(source.subject.astype(str)):
        raise RuntimeError("sealed subject entered link-threshold audit")

    valid = source.error.isna()
    basic = (
        valid
        & (source.live_sc > 0)
        & (source.pkt_max_med > MIN_PKT_MAX)
        & (source.allzero_ratio.fillna(0.0) <= MAX_ALLZERO_RATIO)
        & source.frame_corr.notna()
    )
    coherence = source.loc[basic, "frame_corr"].to_numpy(dtype=np.float64)
    below = source.loc[basic & (source.frame_corr < MIN_FRAME_CORR)].copy()
    below_sites = (
        below.subject.astype(str) + "_" + below.environment.astype(str)
    ).value_counts().sort_index()

    result = {
        "run": "A53-SOURCE-ONLY-LINK-THRESHOLD-AUDIT",
        "source_sites": sorted(SOURCE_SITES),
        "basic_quality_links": int(len(coherence)),
        "frame_corr_threshold": float(MIN_FRAME_CORR),
        "quantiles": {
            "q01": float(np.quantile(coherence, 0.01)),
            "q03": float(np.quantile(coherence, 0.03)),
            "q05": float(np.quantile(coherence, 0.05)),
            "median": float(np.median(coherence)),
        },
        "below_threshold": int(len(below)),
        "below_threshold_rate": float(len(below) / len(coherence)),
        "below_threshold_by_site": {
            str(site): int(count) for site, count in below_sites.items()
        },
        "interpretation": (
            "source-only lower-tail quality guardrail; not a supervised "
            "hardware-failure label"
        ),
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
