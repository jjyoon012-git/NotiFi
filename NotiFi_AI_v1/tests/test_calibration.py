from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from notifi_ai.calibration import CalibrationProfile


class CalibrationTest(unittest.TestCase):
    def test_absence_baseline_round_trip(self):
        csi = np.zeros((304, 3, 114, 2), np.float32)
        csi[..., 0] = 10.0
        csi[..., 1] = 0.25
        mask = np.ones((304, 3), bool)
        profile = CalibrationProfile.fit_absence(
            "unit-device", [(csi, mask), (csi + 0.1, mask)]
        )
        calibrated, output_mask = profile.apply_csi(csi, mask)
        self.assertEqual(calibrated.shape, csi.shape)
        self.assertTrue(output_mask.all())
        self.assertLess(float(np.abs(calibrated).mean()), 0.06)
        with tempfile.TemporaryDirectory() as folder:
            path = profile.save(Path(folder) / "profile.pt")
            restored = CalibrationProfile.load(path)
        np.testing.assert_allclose(
            restored.absence_baseline, profile.absence_baseline
        )

    def test_logit_bias_is_bounded(self):
        profile = CalibrationProfile.fit_absence(
            "unit-device",
            [
                (
                    np.ones((304, 3, 114, 2), np.float32),
                    np.ones((304, 3), bool),
                )
            ],
        )
        action_logits = torch.zeros(4, 17)
        risk_logits = torch.zeros(4, 3)
        actions = torch.tensor([0, 1, 2, 3])
        risks = torch.zeros(4, dtype=torch.long)
        profile.fit_logit_biases(
            action_logits,
            action_logits,
            risk_logits,
            risk_logits,
            actions,
            risks,
        )
        self.assertLessEqual(float(np.abs(profile.display_action_bias).max()), 0.75)
        self.assertLessEqual(float(np.abs(profile.display_risk_bias).max()), 0.75)

    def test_link_without_absence_coverage_is_disabled(self):
        csi = np.ones((304, 3, 114, 2), np.float32)
        mask = np.ones((304, 3), bool)
        absence_mask = mask.copy()
        absence_mask[:, 2] = False
        profile = CalibrationProfile.fit_absence(
            "unit-device", [(csi, absence_mask)]
        )
        calibrated, output_mask = profile.apply_csi(csi, mask)
        self.assertFalse(output_mask[:, 2].any())
        self.assertEqual(float(calibrated[:, 2].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
