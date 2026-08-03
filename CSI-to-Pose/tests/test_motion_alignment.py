import unittest

import numpy as np

from notifi_pose.tools.audit_motion_alignment import (
    lag_correlations,
    masked_smooth,
)


class MotionAlignmentTests(unittest.TestCase):
    def test_positive_lag_means_gt_occurs_later(self) -> None:
        rng = np.random.default_rng(7)
        energy = rng.normal(size=120).astype(np.float32)
        speed = np.zeros_like(energy)
        speed[5:] = energy[:-5]
        valid = np.ones(120, dtype=bool)
        best, lag, _ = lag_correlations(energy, speed, valid, max_lag=10)
        self.assertGreater(best, 0.99)
        self.assertEqual(lag, 5)

    def test_masked_smoothing_ignores_invalid_values(self) -> None:
        values = np.array([1.0, 1000.0, 3.0], dtype=np.float32)
        valid = np.array([True, False, True])
        smoothed = masked_smooth(values, valid, width=3)
        np.testing.assert_allclose(smoothed, np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
