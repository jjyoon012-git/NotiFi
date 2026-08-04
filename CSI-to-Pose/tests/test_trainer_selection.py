import unittest

from notifi_pose.trainer import _checkpoint_score


class TrainerSelectionTest(unittest.TestCase):
    def test_mpjpe_mode_ignores_secondary_metrics(self):
        metrics = {
            "mpjpe": 0.15,
            "impact_mpjpe": 1.0,
            "distal_mpjpe": 1.0,
            "root_err": 1.0,
        }
        self.assertEqual(_checkpoint_score(metrics, "mpjpe"), 0.15)
        self.assertGreater(_checkpoint_score(metrics, "composite"), 0.15)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _checkpoint_score({"mpjpe": 0.1}, "other")


if __name__ == "__main__":
    unittest.main()
