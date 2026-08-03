"""Aggregate final Protocol A and fixed LOSO evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .. import contract as C


RUNS = (
    ("yja_e02", "robust_gf_yja_e02", "eval_yja_holdout_yja_E02_test"),
    ("loso_ajh", "robust_gf_loso_ajh", "eval_loso_test_ajh_test"),
    ("loso_lmh", "robust_gf_loso_lmh", "eval_loso_test_lmh_test"),
    ("loso_mhw", "robust_gf_loso_mhw", "eval_loso_test_mhw_test"),
)


def load_summary(run: str, evaluation: str, smooth: int) -> dict:
    path = C.WORK_ROOT / "runs" / run / f"{evaluation}_smooth{smooth}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for protocol, run, evaluation in RUNS:
        raw = load_summary(run, evaluation, 1)
        smooth = load_summary(run, evaluation, 5)
        row = {
            "protocol": protocol,
            "run": run,
            "pose_trials": smooth["pose_trials"],
            "mpjpe_cm": smooth["overall"]["mpjpe_m"] * 100,
            "dynamic_mpjpe_cm": smooth["overall"]["dynamic_mpjpe_m"] * 100,
            "root_error_cm": smooth["overall"]["root_error_m"] * 100,
            "class_accuracy": smooth["overall"]["class_accuracy"],
            "risk_accuracy": smooth["overall"]["risk_accuracy"],
            "raw_pose_speed_ratio": raw["overall"]["pose_speed_ratio"],
            "smooth5_pose_speed_ratio": smooth["overall"]["pose_speed_ratio"],
            "raw_root_speed_ratio": raw["overall"]["root_speed_ratio"],
            "smooth5_root_speed_ratio": smooth["overall"]["root_speed_ratio"],
        }
        rows.append(row)

    loso = rows[1:]
    numeric = [key for key in rows[0] if key not in {"protocol", "run"}]
    loso_mean = {"protocol": "loso_mean", "run": "three_fold_mean"}
    for key in numeric:
        loso_mean[key] = float(np.mean([row[key] for row in loso]))

    baseline = {
        "loso_mean_mpjpe_cm": 30.51,
        "loso_mean_root_error_cm": 52.39,
        "yja_e02_mpjpe_cm": 29.08,
        "yja_e02_root_error_cm": 62.77,
        "source": "docs/graphformer_gvhmr_v2_experiment.md",
    }
    payload = {
        "model": "robust_graphformer",
        "protocol_a": rows[0],
        "loso_folds": loso,
        "loso_mean": loso_mean,
        "previous_graphformer": baseline,
        "delta": {
            "loso_mpjpe_cm": loso_mean["mpjpe_cm"] - baseline["loso_mean_mpjpe_cm"],
            "loso_root_error_cm": (
                loso_mean["root_error_cm"] - baseline["loso_mean_root_error_cm"]
            ),
            "yja_mpjpe_cm": rows[0]["mpjpe_cm"] - baseline["yja_e02_mpjpe_cm"],
            "yja_root_error_cm": (
                rows[0]["root_error_cm"] - baseline["yja_e02_root_error_cm"]
            ),
        },
    }
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = C.REPORT_DIR / "robust_protocol_results.json"
    csv_path = C.REPORT_DIR / "robust_protocol_results.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [loso_mean])
    print(json.dumps(payload["delta"], indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
