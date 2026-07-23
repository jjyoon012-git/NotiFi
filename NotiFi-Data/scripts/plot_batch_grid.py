"""
Batch grid visualization for CSI amplitude.
Usage:
    python scripts/plot_batch_grid.py <folder_path> [--cols N] [--smooth N] [--out filename.png]
"""
import argparse
import ast
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_amplitude(path: Path, smooth: int = 5) -> tuple[np.ndarray, np.ndarray]:
    times, amps = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("raw_line", "")
            payload = row.get("csi_data", "") or raw
            if not payload:
                continue
            m = re.search(r'"?\[(.+?)\]"?', payload)
            if not m:
                continue
            try:
                vals = np.array(ast.literal_eval(f"[{m.group(1)}]"), dtype=np.float32)
            except Exception:
                continue
            I, Q = vals[0::2], vals[1::2]
            amp = np.sqrt(I**2 + Q**2)
            amp = amp[amp > 0]
            if row.get("pc_monotonic_ns"):
                times.append(int(row["pc_monotonic_ns"]) / 1_000_000)
            elif row.get("pc_time_ms"):
                times.append(int(row["pc_time_ms"]))
            else:
                continue
            amps.append(float(amp.mean()) if len(amp) else 0)

    if not times:
        return np.array([]), np.array([])
    t = (np.array(times, dtype=np.float64) - times[0]) / 1000.0
    a = np.array(amps, dtype=np.float64)
    if smooth > 1:
        a = np.convolve(a, np.ones(smooth) / smooth, mode="same")
    return t, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="CSV 파일이 있는 폴더 경로")
    parser.add_argument("--cols",   type=int, default=5,  help="열 수 (기본 5)")
    parser.add_argument("--smooth", type=int, default=5,  help="이동평균 window (기본 5)")
    parser.add_argument("--out",    default=None,          help="출력 PNG 파일명")
    args = parser.parse_args()

    folder = Path(args.folder)
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] CSV 없음: {folder}")
        sys.exit(1)

    n = len(csvs)
    cols = args.cols
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    folder_label = folder.name
    fig.suptitle(folder_label, fontsize=13, fontweight="bold", y=1.01)

    for i, (ax, csv_path) in enumerate(zip(axes_flat, csvs)):
        t, a = read_amplitude(csv_path, smooth=args.smooth)
        trial = csv_path.stem.split("_")[-1]  # e.g. t001
        if len(t):
            ax.plot(t, a, linewidth=0.9, color="#1f77b4")
            ax.set_xlim(0, t[-1] + 0.5)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        ax.set_title(trial, fontsize=9)
        ax.set_xlabel("s", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    # 빈 subplot 숨기기
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    out_name = args.out or f"{folder_label}_grid.png"
    out_path = folder / out_name
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
