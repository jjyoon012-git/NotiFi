import unittest

import numpy as np

from notifi_pose.dataio.dataset import SiteBaseline


class SiteBaselineTests(unittest.TestCase):
    def test_invalid_calibration_link_keeps_original_signal(self) -> None:
        baseline = SiteBaseline("none")
        baseline.mode = "sub"
        mean = np.full((3, 4, 2), 2.0, dtype=np.float32)
        scale = np.ones_like(mean)
        baseline.table = {"subject_E01": (mean, scale, np.array([True, False, True]))}
        csi = np.full((5, 3, 4, 2), 3.0, dtype=np.float32)
        mask = np.ones((5, 3), dtype=bool)

        output, output_mask = baseline.apply(csi, mask, "subject_E01")

        np.testing.assert_allclose(output[:, 0], 1.0)
        np.testing.assert_allclose(output[:, 1], 3.0)
        self.assertTrue(output_mask.all())
