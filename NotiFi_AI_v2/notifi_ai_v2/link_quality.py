"""고정 cache의 trial별 CSI 링크 품질 표를 읽는다."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_link_quality(cache_root: str | Path) -> dict[str, np.ndarray]:
    """품질 임계를 통과한 TX 링크를 trial별 boolean mask로 반환한다."""
    report = Path(cache_root).resolve().parent / "reports" / "link_quality.csv"
    if not report.exists():
        return {}
    temporary: dict[str, dict[str, bool]] = {}
    with report.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error", "").strip():
                continue
            try:
                dead = (
                    float(row["live_sc"]) <= 0
                    or float(row["pkt_max_med"]) <= 1.0
                    or float(row.get("allzero_ratio") or 0.0) > 0.9
                    or float(row.get("frame_corr") or 0.0) < 0.65
                )
            except (KeyError, ValueError):
                dead = True
            temporary.setdefault(row["trial_id"], {})[row["tx"]] = not dead
    return {
        trial_id: np.asarray(
            [links.get(name, False) for name in ("TX1", "TX2", "TX3")],
            dtype=bool,
        )
        for trial_id, links in temporary.items()
    }
