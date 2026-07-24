from __future__ import annotations

import ast
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TX_ORDER = ("TX1", "TX2", "TX3")


def parse_csi_array(payload: str) -> np.ndarray | None:
    payload = payload.strip()
    if not payload:
        return None
    try:
        values = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(values, list) or len(values) < 2:
        return None
    try:
        return np.asarray(values, dtype=np.float32)
    except ValueError:
        return None


def mean_amplitude(values: np.ndarray) -> float:
    i_values = values[0::2]
    q_values = values[1::2]
    amplitude = np.sqrt(i_values**2 + q_values**2)
    amplitude = amplitude[amplitude > 0]
    return float(amplitude.mean()) if len(amplitude) else 0.0


def load_amplitude_series(csv_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sender = row.get("sender_id", "UNKNOWN") or "UNKNOWN"
            if sender == "UNKNOWN":
                continue
            payload = row.get("csi_data", "")
            values = parse_csi_array(payload)
            if values is None:
                continue
            try:
                elapsed_s = float(row["pc_elapsed_s"])
            except (KeyError, ValueError):
                continue
            series[sender].append((elapsed_s, mean_amplitude(values)))

    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sender, rows in series.items():
        if not rows:
            continue
        rows.sort(key=lambda item: item[0])
        t = np.asarray([item[0] for item in rows], dtype=np.float64)
        a = np.asarray([item[1] for item in rows], dtype=np.float64)
        output[sender] = (t, a)
    return output


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def save_csi_visualization(
    csv_path: Path,
    out_path: Path | None = None,
    *,
    title: str | None = None,
    smooth_window: int = 5,
) -> dict[str, int | str]:
    series = load_amplitude_series(csv_path)
    if not series:
        raise ValueError(f"No valid CSI frames found: {csv_path}")

    out_path = out_path or csv_path.with_name(f"{csv_path.stem}_visualization.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(TX_ORDER) + 1,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 1.0, 1.0]},
    )

    colors = {
        "TX1": "#1f77b4",
        "TX2": "#2ca02c",
        "TX3": "#d62728",
    }
    counts: dict[str, int] = {}

    overview = axes[0]
    for sender in TX_ORDER:
        if sender not in series:
            counts[sender] = 0
            continue
        t, a = series[sender]
        counts[sender] = int(len(a))
        overview.plot(t, smooth(a, smooth_window), label=f"{sender} ({len(a)})", linewidth=1.0, color=colors[sender])
    overview.set_ylabel("Mean amp")
    overview.set_title(title or csv_path.stem)
    overview.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    overview.legend(loc="upper right", ncols=3, fontsize=9)

    for ax, sender in zip(axes[1:], TX_ORDER):
        if sender in series:
            t, a = series[sender]
            ax.plot(t, smooth(a, smooth_window), linewidth=0.9, color=colors[sender])
            ax.text(
                0.01,
                0.88,
                f"{sender} frames={len(a)}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "#dddddd"},
            )
        else:
            ax.text(0.5, 0.5, f"{sender}: no frames", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Mean amp")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)

    axes[-1].set_xlabel("Elapsed time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "path": str(out_path),
        "tx1_frames": counts.get("TX1", 0),
        "tx2_frames": counts.get("TX2", 0),
        "tx3_frames": counts.get("TX3", 0),
    }
