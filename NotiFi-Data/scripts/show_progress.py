from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.labels import ENVIRONMENTS, LABEL_SPECS, SUBJECTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Show NotiFi v2.0 collection progress.")
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--environment", required=True, choices=ENVIRONMENTS)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "collection_data" / "v2" / "manifests" / "trials.csv",
    )
    args = parser.parse_args()

    rows = []
    if args.manifest.exists():
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("subject_id") == args.subject
                and row.get("environment_id") == args.environment
            ]
    counts = Counter(
        row["scenario_label"]
        for row in rows
        if row.get("automatic_qc") != "AUTO_REJECT"
        and row.get("manual_qc") != "REJECT"
    )
    print(f"\nProgress: subject={args.subject} environment={args.environment}")
    print(f"{'ID':<4} {'risk':<8} {'label':<28} {'done':>5} {'target':>6} {'remain':>7}")
    print("-" * 66)
    completed_total = 0
    for spec in LABEL_SPECS:
        done = counts.get(spec.label, 0)
        target = spec.repeats_per_environment
        remain = max(0, target - done)
        completed_total += min(done, target)
        print(
            f"{spec.id:<4} {spec.risk:<8} {spec.label:<28} "
            f"{done:>5} {target:>6} {remain:>7}"
        )
    print("-" * 66)
    print(f"Total: {completed_total}/275, remaining={275 - completed_total}")


if __name__ == "__main__":
    main()
