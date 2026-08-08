"""Tests for repeated calibration aggregation and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_calibration_seeds import summarize


def _result(seed: int, action: float) -> dict:
    """Build a minimal clean three-fold calibration result."""
    folds = {}
    for subject in ("ajh", "mhw", "lmh"):
        folds[subject] = {
            "outer_used_for_selection": False,
            "outer_metrics": [{
                "action_accuracy": action,
                "action_macro_f1": action - 0.1,
                "risk_accuracy": 0.5,
                "risk_macro_f1": 0.4,
                "danger_recall": 0.6,
                "danger_action_accuracy": 0.2,
                "safe_to_danger": 1,
                "safe_total": 20,
                "trials": 40,
            }],
        }
    return {
        "support_seed": seed,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "folds": folds,
    }


class CalibrationSeedSummaryTests(unittest.TestCase):
    """Check seed statistics and target contamination rejection."""

    def test_seed_mean_and_population_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for seed, action in ((1, 0.3), (2, 0.5)):
                path = root / f"seed{seed}.json"
                path.write_text(
                    json.dumps(_result(seed, action)), encoding="utf-8"
                )
                files.append(path)
            report = summarize(files)
            self.assertAlmostEqual(
                report["aggregate"]["action_accuracy"]["mean"], 0.4
            )
            self.assertAlmostEqual(
                report["aggregate"]["action_accuracy"]["std"], 0.1
            )

    def test_query_supervision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            result = _result(1, 0.4)
            result["query_labels_or_pose_gt_at_inference"] = True
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "query supervision"):
                summarize([path])


if __name__ == "__main__":
    unittest.main()
