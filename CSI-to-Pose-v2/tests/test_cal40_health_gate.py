"""CAL40 calibration health gate summary tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_cal40_health_gate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_cal40_health_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CAL40HealthGateTests(unittest.TestCase):
    """Verify deterministic geometry-error health statistics."""

    def test_summarize_reports_pass_rate_and_false_rejections(self) -> None:
        summary = MODULE.summarize([0.01, 0.02, 0.04], [True, False, True])

        self.assertAlmostEqual(summary["mean"], 0.07 / 3.0)
        self.assertAlmostEqual(summary["median"], 0.02)
        self.assertAlmostEqual(summary["maximum"], 0.04)
        self.assertAlmostEqual(summary["pass_rate"], 2.0 / 3.0)
        self.assertEqual(summary["false_rejections"], 1)
        self.assertEqual(summary["episodes"], 3)


if __name__ == "__main__":
    unittest.main()
