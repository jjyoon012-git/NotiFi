"""Render the compact validation progression chart used by the README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.summary.read_text(encoding="utf-8"))
    progress = report["validation_progress"]
    names = [item["name"] for item in progress]
    metrics = (
        ("MPJPE", "mpjpe_cm", "#2563eb"),
        ("Root", "root_cm", "#0f766e"),
        ("Danger", "danger_cm", "#dc2626"),
        ("Danger endpoint", "endpoint_cm", "#7c3aed"),
    )
    x = np.arange(len(names))
    width = 0.19
    fig, axis = plt.subplots(figsize=(10, 4.8), dpi=160)
    for index, (label, key, color) in enumerate(metrics):
        values = [item[key] for item in progress]
        offset = (index - 1.5) * width
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axis.set_ylabel("Validation error (cm, lower is better)")
    axis.set_xticks(x, names)
    axis.set_ylim(0, 62)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncols=4, loc="upper center", frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
