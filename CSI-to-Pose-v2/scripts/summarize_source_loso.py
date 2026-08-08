"""Aggregate source nested-LOSO checkpoints without opening sealed target data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


SITE_MACRO_KEYS = (
    "action_accuracy",
    "action_macro_f1",
    "risk_accuracy",
    "risk_macro_f1",
    "danger_recall",
    "danger_action_accuracy",
)


def summarize(run_dir: Path) -> dict:
    """Read the three clean source folds and return one auditable summary."""
    sites: list[dict] = []
    folds = {}
    for subject in ("ajh", "mhw", "lmh"):
        path = run_dir / f"selection_{subject}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required_false = (
            "outer_holdout_used_for_selection",
            "target_subject_used",
            "sealed_yja_used",
        )
        contaminated = [
            key for key in required_false if checkpoint.get(key) is not False
        ]
        if contaminated:
            raise RuntimeError(f"unclean checkpoint {path}: {contaminated}")
        if checkpoint.get("protocol_version") != "nested_source_v1":
            raise RuntimeError(f"unexpected protocol in {path}")
        outer = checkpoint["outer_test"]
        expected_prefix = f"{subject}_"
        if not outer or any(
            not str(site).startswith(expected_prefix) for site in outer
        ):
            raise RuntimeError(f"invalid outer sites in {path}: {list(outer)}")
        current = list(outer.values())
        sites.extend(current)
        folds[subject] = {
            key: float(np.mean([row[key] for row in current]))
            for key in SITE_MACRO_KEYS
        }

    site_macro = {
        key: float(np.mean([row[key] for row in sites]))
        for key in SITE_MACRO_KEYS
    }
    safe_total = int(sum(row["safe_total"] for row in sites))
    safe_to_danger = int(sum(row["safe_to_danger"] for row in sites))
    danger_total = int(sum(row["danger_total"] for row in sites))
    danger_correct = int(sum(row["danger_correct"] for row in sites))
    return {
        "run_dir": str(run_dir.resolve()),
        "protocol": "nested_source_subject_loso_v1",
        "sites": len(sites),
        "site_macro": site_macro,
        "safe_to_danger_rate": safe_to_danger / max(safe_total, 1),
        "safe_to_danger": safe_to_danger,
        "safe_total": safe_total,
        "danger_recall_pooled": danger_correct / max(danger_total, 1),
        "danger_correct": danger_correct,
        "danger_total": danger_total,
        "worst_site_action": float(min(
            row["action_accuracy"] for row in sites
        )),
        "fold_macro": folds,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }


def main() -> None:
    """Parse paths, print JSON, and optionally save the same report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = summarize(options.run_dir)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
