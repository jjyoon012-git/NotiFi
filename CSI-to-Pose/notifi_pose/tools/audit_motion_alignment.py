"""Audit trial-level CSI motion observability and temporal alignment against GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio.dataset import PoseDataset, build_datasets
from .diagnose_observability import pose_only, report_path


def masked_smooth(values: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64)
    numerator = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    denominator = np.convolve(valid.astype(np.float64), kernel, mode="same")
    return (numerator / np.maximum(denominator, 1.0)).astype(np.float32)


def csi_motion_energy(csi: np.ndarray, link_mask: np.ndarray,
                      smooth_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Scale-invariant amplitude/phase change energy for each aligned CSI frame."""
    signal = csi[..., 0].astype(np.float64) + 1j * csi[..., 1].astype(np.float64)
    amplitude = np.abs(signal)
    floor = np.maximum(np.nanmedian(amplitude[amplitude > 0]) * 1e-3, 1e-6)
    log_amplitude = np.log(amplitude + floor)
    amplitude_delta = log_amplitude[1:] - log_amplitude[:-1]
    phase_delta = np.angle(signal[1:] * np.conj(signal[:-1]))
    feature_energy = np.sqrt(amplitude_delta ** 2 + phase_delta ** 2)

    pair_links = link_mask[1:] & link_mask[:-1]
    link_energy = np.nanmedian(feature_energy, axis=-1)
    link_energy = np.where(pair_links, link_energy, np.nan)
    finite = np.isfinite(link_energy)
    energy = np.zeros(len(csi), dtype=np.float32)
    energy[1:] = (
        np.nansum(link_energy, axis=-1)
        / np.maximum(finite.sum(axis=-1), 1)
    ).astype(np.float32)
    valid = np.zeros(len(csi), dtype=bool)
    valid[1:] = pair_links.any(axis=-1)
    if valid.any():
        ceiling = float(np.nanpercentile(energy[valid], 99.5))
        energy = np.clip(energy, 0.0, max(ceiling, 1e-6))
    return masked_smooth(energy, valid, smooth_width), valid


def gt_body_speed(pose_rel: np.ndarray, root: np.ndarray, valid: np.ndarray,
                  smooth_width: int) -> tuple[np.ndarray, np.ndarray]:
    absolute = pose_rel + root[:, None]
    speed = np.zeros(len(pose_rel), dtype=np.float32)
    speed[1:] = (
        np.linalg.norm(absolute[1:] - absolute[:-1], axis=-1).mean(-1)
        * C.TARGET_FPS
    )
    pair_valid = np.zeros_like(valid)
    pair_valid[1:] = valid[1:] & valid[:-1]
    return masked_smooth(speed, pair_valid, smooth_width), pair_valid


def pearson(values: np.ndarray, target: np.ndarray) -> float:
    if len(values) < 12 or np.std(values) < 1e-8 or np.std(target) < 1e-8:
        return float("nan")
    return float(np.corrcoef(values, target)[0, 1])


def lag_correlations(csi_energy: np.ndarray, gt_speed: np.ndarray,
                     valid: np.ndarray, max_lag: int) -> tuple[float, int, float]:
    """Return best correlation, lag, and zero-lag correlation.

    A positive lag means the GT motion occurs later than the CSI energy.
    """
    candidates: list[tuple[float, int]] = []
    zero = float("nan")
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            energy = csi_energy[-lag:]
            speed = gt_speed[:lag]
            mask = valid[-lag:] & valid[:lag]
        elif lag > 0:
            energy = csi_energy[:-lag]
            speed = gt_speed[lag:]
            mask = valid[:-lag] & valid[lag:]
        else:
            energy, speed, mask = csi_energy, gt_speed, valid
        correlation = pearson(energy[mask], speed[mask])
        if lag == 0:
            zero = correlation
        if np.isfinite(correlation):
            candidates.append((correlation, lag))
    if not candidates:
        return float("nan"), 0, zero
    best, lag = max(candidates, key=lambda item: item[0])
    return best, lag, zero


def audit_trial(item: dict, max_lag: int, smooth_width: int) -> dict:
    csi = item["csi"].numpy()
    link_mask = item["link_mask"].numpy().astype(bool)
    valid = item["valid"].numpy().astype(bool)
    energy, energy_valid = csi_motion_energy(csi, link_mask, smooth_width)
    speed, speed_valid = gt_body_speed(
        item["pose_rel"].numpy(), item["root"].numpy(), valid, smooth_width
    )
    comparable = energy_valid & speed_valid
    best, lag, zero = lag_correlations(energy, speed, comparable, max_lag)
    if not np.isfinite(best):
        status = "invalid"
    elif best < 0.30:
        status = "low_observability"
    elif abs(lag) > 6:
        status = "lag_candidate"
    else:
        status = "aligned_observable"
    return {
        "n_comparable_frames": int(comparable.sum()),
        "zero_lag_correlation": zero,
        "best_correlation": best,
        "best_lag_frames": int(lag),
        "best_lag_ms": 1000.0 * lag / C.TARGET_FPS,
        "mean_csi_motion_energy": (
            float(energy[comparable].mean()) if comparable.any() else float("nan")
        ),
        "mean_gt_speed_mps": (
            float(speed[comparable].mean()) if comparable.any() else float("nan")
        ),
        "status": status,
    }


def audit_dataset(dataset: PoseDataset, split: str, max_lag: int,
                  smooth_width: int) -> list[dict]:
    rows = []
    metadata = dataset.index.reset_index(drop=True)
    for index in range(len(dataset)):
        result = audit_trial(dataset[index], max_lag, smooth_width)
        meta = metadata.iloc[index]
        rows.append({
            "split": split,
            "trial_id": str(meta.trial_id),
            "subject": str(meta.subject),
            "environment": str(meta.environment),
            "risk": str(meta.risk),
            "label": str(meta.detail_label),
            **result,
        })
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(f"  {split}: {index + 1}/{len(dataset)}")
    return rows


def grouped_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    records = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(columns, keys))
        record.update({
            "trials": int(len(group)),
            "median_zero_lag_correlation": float(group.zero_lag_correlation.median()),
            "median_best_correlation": float(group.best_correlation.median()),
            "median_abs_lag_frames": float(group.best_lag_frames.abs().median()),
            "aligned_observable_rate": float(
                (group.status == "aligned_observable").mean()
            ),
            "lag_candidate_rate": float((group.status == "lag_candidate").mean()),
            "low_observability_rate": float(
                (group.status == "low_observability").mean()
            ),
        })
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="yja_holdout", choices=("yja_holdout",))
    parser.add_argument("--baseline", default="none", choices=("none", "sub", "sub_z"))
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--smooth-width", type=int, default=7)
    parser.add_argument(
        "--output-csv", type=Path,
        default=C.REPORT_DIR / "motion_alignment_audit.csv",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=C.REPORT_DIR / "motion_alignment_audit.json",
    )
    args = parser.parse_args()

    datasets = build_datasets(exp=args.exp, baseline=args.baseline)
    rows = []
    for split in ("train", "val", "test"):
        rows.extend(audit_dataset(
            pose_only(datasets[split]), split, args.max_lag, args.smooth_width
        ))
    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    summary = {
        "protocol": args.exp,
        "baseline": args.baseline,
        "max_lag_frames": args.max_lag,
        "smooth_width": args.smooth_width,
        "thresholds": {
            "observable_correlation": 0.30,
            "aligned_abs_lag_frames": 6,
        },
        "status_counts": frame.status.value_counts().to_dict(),
        "by_split": grouped_summary(frame, ["split"]),
        "by_subject_environment": grouped_summary(
            frame, ["split", "subject", "environment"]
        ),
        "by_risk": grouped_summary(frame, ["split", "risk"]),
        "csv": report_path(args.output_csv),
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
