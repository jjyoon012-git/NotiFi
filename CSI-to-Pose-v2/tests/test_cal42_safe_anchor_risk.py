"""CAL42 safe-anchor risk calibration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "evaluate_cal42_safe_anchor_risk.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_cal42_safe_anchor_risk", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CAL42SafeAnchorRiskTests(unittest.TestCase):
    """Verify that safe anchors set the declared danger-margin quantile."""

    def test_safe_anchor_quantile_is_shifted_to_target_margin(self) -> None:
        payload = {
            "calibration_safe_risk": torch.tensor([
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 2.0],
            ]),
            "risk": torch.tensor([[0.0, -1.0, 3.0]]),
        }

        risk, adjustment, observed = MODULE.safe_anchor_risk(
            payload, quantile=0.5, target_margin=-0.5
        )

        self.assertAlmostEqual(observed, 1.0)
        self.assertAlmostEqual(adjustment, -1.5)
        torch.testing.assert_close(risk, torch.tensor([[0.0, -1.0, 1.5]]))
        torch.testing.assert_close(
            payload["risk"], torch.tensor([[0.0, -1.0, 3.0]])
        )


if __name__ == "__main__":
    unittest.main()
