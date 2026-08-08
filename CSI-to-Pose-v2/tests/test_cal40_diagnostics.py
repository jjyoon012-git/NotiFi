"""CAL40 계층 action 진단의 oracle/실제 routing 계산 테스트."""

from pathlib import Path
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_cal40_fixed_deep_action import hierarchy_diagnostics  # noqa: E402


class CAL40HierarchyDiagnosticsTests(unittest.TestCase):
    def test_oracle_and_danger_top2_counts_are_explicit(self) -> None:
        action = torch.full((3, 17), -5.0)
        action[0, 0] = 5.0
        action[1, 13] = 4.0
        action[1, 12] = 3.0
        action[2, 16] = 4.0
        action[2, 15] = 3.0
        labels = torch.tensor((0, 12, 15))
        risks = torch.tensor((0, 2, 2))
        risk = torch.tensor(((5.0, 0.0, 0.0),
                             (0.0, 0.0, 5.0),
                             (0.0, 5.0, 0.0)))

        result = hierarchy_diagnostics(action, risk, labels, risks)

        self.assertEqual(result["oracle_risk_action_correct"], 1)
        self.assertEqual(result["predicted_risk_action_correct"], 1)
        self.assertEqual(result["danger_within_group_correct"], 0)
        self.assertEqual(result["danger_within_group_top2_correct"], 2)
        self.assertEqual(result["danger_within_group_total"], 2)


if __name__ == "__main__":
    unittest.main()
