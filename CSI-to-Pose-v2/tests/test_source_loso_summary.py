"""Tests for leakage-guarded source LOSO aggregation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from scripts.summarize_source_loso import summarize


def _metrics(site: str, action: float) -> dict:
    """Build one compact site result for aggregation tests."""
    return {
        "site": site,
        "action_accuracy": action,
        "action_macro_f1": action - 0.1,
        "risk_accuracy": 0.5,
        "risk_macro_f1": 0.4,
        "danger_recall": 0.6,
        "danger_action_accuracy": 0.2,
        "danger_correct": 6,
        "danger_total": 10,
        "safe_to_danger": 2,
        "safe_total": 20,
    }


class SourceLosoSummaryTests(unittest.TestCase):
    """Ensure aggregation checks provenance before reporting metrics."""

    def test_clean_folds_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for number, subject in enumerate(("ajh", "mhw", "lmh")):
                torch.save({
                    "protocol_version": "nested_source_v1",
                    "outer_holdout_used_for_selection": False,
                    "target_subject_used": False,
                    "sealed_yja_used": False,
                    "outer_test": {
                        f"{subject}_E01": _metrics(
                            f"{subject}_E01", 0.3 + number * 0.1
                        )
                    },
                }, run / f"selection_{subject}.pt")
            report = summarize(run)
            self.assertAlmostEqual(
                report["site_macro"]["action_accuracy"], 0.4
            )
            self.assertAlmostEqual(report["safe_to_danger_rate"], 0.1)
            self.assertFalse(report["sealed_yja_used"])

    def test_target_contamination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for subject in ("ajh", "mhw", "lmh"):
                torch.save({
                    "protocol_version": "nested_source_v1",
                    "outer_holdout_used_for_selection": False,
                    "target_subject_used": subject == "mhw",
                    "sealed_yja_used": False,
                    "outer_test": {
                        f"{subject}_E01": _metrics(f"{subject}_E01", 0.4)
                    },
                }, run / f"selection_{subject}.pt")
            with self.assertRaisesRegex(RuntimeError, "unclean checkpoint"):
                summarize(run)


if __name__ == "__main__":
    unittest.main()
