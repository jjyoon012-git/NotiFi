"""Summarize repeated source-only calibration seeds with provenance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = (
    "action_accuracy",
    "action_macro_f1",
    "risk_accuracy",
    "risk_macro_f1",
    "danger_recall",
    "danger_action_accuracy",
)


def summarize(files: list[Path]) -> dict:
    """Aggregate each seed over outer sites and then report seed variation."""
    if not files:
        raise ValueError("at least one calibration result is required")
    seeds = []
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("target_subject_used") is not False:
            raise RuntimeError(f"target subject contamination in {path}")
        if result.get("sealed_yja_used") is not False:
            raise RuntimeError(f"sealed target contamination in {path}")
        if result.get("query_labels_or_pose_gt_at_inference") is not False:
            raise RuntimeError(f"query supervision contamination in {path}")
        outer = []
        for subject in ("ajh", "mhw", "lmh"):
            fold = result["folds"][subject]
            if fold.get("outer_used_for_selection") is not False:
                raise RuntimeError(f"outer selection contamination in {path}")
            outer.extend(fold["outer_metrics"])
        weights = np.asarray([row["trials"] for row in outer], dtype=np.float64)
        metrics = {
            key: float(np.average(
                [row[key] for row in outer], weights=weights
            ))
            for key in METRIC_KEYS
        }
        safe_total = int(sum(row["safe_total"] for row in outer))
        metrics.update({
            "safe_to_danger_rate": (
                sum(row["safe_to_danger"] for row in outer)
                / max(safe_total, 1)
            ),
            "worst_site_action": float(min(
                row["action_accuracy"] for row in outer
            )),
        })
        seeds.append({
            "support_seed": int(result["support_seed"]),
            "metrics": metrics,
            "path": str(path.resolve()),
        })
    keys = (*METRIC_KEYS, "safe_to_danger_rate", "worst_site_action")
    aggregate = {
        key: {
            "mean": float(np.mean([row["metrics"][key] for row in seeds])),
            "std": float(np.std([row["metrics"][key] for row in seeds])),
        }
        for key in keys
    }
    return {
        "protocol": "source_nested_loso_repeated_support_v1",
        "seed_count": len(seeds),
        "aggregate": aggregate,
        "seeds": seeds,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }


def main() -> None:
    """Resolve a glob, save the report if requested, and print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--pattern", default="cal17_abs12_support_seed*.json"
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    files = sorted(options.run_dir.glob(options.pattern))
    report = summarize(files)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
