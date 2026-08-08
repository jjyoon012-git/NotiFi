"""source-clean deep ensemble의 확률·위험 결합 테스트."""

import math
import unittest

import torch

from scripts.calibrate_cal62_deep_ensemble import (
    ensemble_risk_logits,
    probability_ensemble,
    target_components,
)


class CAL62EnsembleTest(unittest.TestCase):
    def test_probability_ensemble_averages_probabilities(self) -> None:
        left = torch.tensor([[4.0, 0.0]])
        right = torch.tensor([[0.0, 4.0]])

        combined = probability_ensemble(left, right).exp()

        torch.testing.assert_close(combined, torch.full((1, 2), 0.5))

    def test_probability_ensemble_rejects_broadcast_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same"):
            probability_ensemble(torch.zeros(1, 17), torch.zeros(2, 17))


    def test_target_components_preserve_labels_and_average_margin(self) -> None:
        labels = torch.tensor([3])
        risks = torch.tensor([1])
        left = {
            "embedding": torch.tensor([[1.0, 0.0]]),
            "anchors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "direct_risk": torch.tensor([[2.0, 1.0, 0.0]]),
            "labels": labels,
            "risks": risks,
        }
        right = {
            "embedding": torch.tensor([[0.0, 1.0]]),
            "anchors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "direct_risk": torch.tensor([[0.0, 1.0, 2.0]]),
            "labels": labels,
            "risks": risks,
        }
        action = torch.zeros(1, 17)

        components = target_components(left, right, action, action)

        self.assertIs(components["labels"], labels)
        self.assertAlmostEqual(float(components["safe_margin"]), 1.0)
        self.assertTrue(torch.isfinite(components["direct"]).all())

    def test_target_components_reject_misaligned_queries(self) -> None:
        common = {
            "embedding": torch.tensor([[1.0, 0.0]]),
            "anchors": torch.eye(2),
            "direct_risk": torch.zeros(1, 3),
            "risks": torch.tensor([0]),
        }
        left = {**common, "labels": torch.tensor([1])}
        right = {**common, "labels": torch.tensor([2])}

        with self.assertRaisesRegex(RuntimeError, "label order"):
            target_components(
                left, right, torch.zeros(1, 17), torch.zeros(1, 17)
            )

    def test_risk_logits_follow_hierarchical_action_mass(self) -> None:
        action = torch.full((1, 17), -20.0)
        action[:, 12:] = 0.0
        target = {
            "action": action,
            "direct": torch.zeros(1, 3),
            "safe_margin": torch.zeros(1),
        }

        risk = ensemble_risk_logits(target, (0.0, 1.0, 0.0))

        self.assertEqual(int(risk.argmax(-1)), 2)
        self.assertAlmostEqual(float(risk[0, 2]), math.log(5.0), places=5)


if __name__ == "__main__":
    unittest.main()
