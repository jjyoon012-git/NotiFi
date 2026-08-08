"""CAL43 partial safe-anchor shrinkage tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "evaluate_cal43_safe_anchor_shrinkage.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_cal43_safe_anchor_shrinkage", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CAL43ShrinkageTests(unittest.TestCase):
    """Verify danger-bias shrinkage toward an inner-source reference."""

    def test_shrinkage_adjustment_uses_only_safe_margin_difference(self) -> None:
        payload = {
            "calibration_safe_risk": torch.tensor([
                [0.0, -1.0, 1.0],
                [0.0, -1.0, 3.0],
            ]),
            "risk": torch.tensor([[0.0, -1.0, 2.0]]),
            "action": torch.zeros(1, 17),
            "labels": torch.tensor([0]),
            "risks": torch.tensor([0]),
        }

        metrics = MODULE.shrinkage_metrics(
            payload,
            quantile=0.5,
            base_bias=1.5,
            shrinkage=0.5,
            reference_margin=1.0,
        )

        self.assertAlmostEqual(metrics["observed_safe_margin"], 2.0)
        self.assertAlmostEqual(metrics["danger_adjustment"], 1.0)
        torch.testing.assert_close(
            payload["risk"], torch.tensor([[0.0, -1.0, 2.0]])
        )


if __name__ == "__main__":
    unittest.main()
