import unittest

import torch

from notifi_pose.tools.diagnose_v13_motion_probe import (
    _regression_metrics,
    _ridge_fit,
    _ridge_predict,
)


class V13MotionProbeTests(unittest.TestCase):
    def test_ridge_recovers_linear_targets(self):
        torch.manual_seed(3)
        feature = torch.randn(200, 5)
        target = feature @ torch.randn(5, 3) + 0.2
        probe = _ridge_fit(feature, target, ridge=1e-4)
        predicted = _ridge_predict(probe, feature)
        metrics = _regression_metrics(predicted, target)
        self.assertGreater(metrics["r2_mean"], 0.999)


if __name__ == "__main__":
    unittest.main()
