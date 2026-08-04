import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.tools.train_seen_v4_trajectory import classification_metrics


class V9MultitaskMetricsTests(unittest.TestCase):
    def test_positive_danger_bias_can_recover_a_missed_fall(self) -> None:
        class_target = torch.tensor([0, 12, 13])
        class_logits = torch.zeros(3, C.N_CLASSES)
        class_logits[torch.arange(3), class_target] = 1.0
        risk_target = torch.tensor([0, 2, 2])
        risk_logits = torch.tensor([
            [2.0, 0.0, -1.0],
            [0.0, 1.0, 0.9],
            [0.0, 0.2, 1.0],
        ])
        logits = {
            "class_logits": class_logits,
            "risk_logits": risk_logits,
            "class_target": class_target,
            "risk_target": risk_target,
        }

        raw = classification_metrics(logits)
        calibrated = classification_metrics(logits, danger_bias=0.2)

        self.assertEqual(raw["risk"]["danger_recall"], 0.5)
        self.assertEqual(calibrated["risk"]["danger_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
