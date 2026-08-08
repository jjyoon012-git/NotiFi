"""CAL23 pose 병목 진단용 oracle action logit 테스트."""

from pathlib import Path
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_cal23_pose_ensemble import oracle_action_logits  # noqa: E402


class CAL23OracleDiagnosticTests(unittest.TestCase):
    def test_oracle_logits_select_exact_labels(self) -> None:
        labels = torch.tensor((0, 12, 16))

        logits = oracle_action_logits(labels)

        self.assertEqual(tuple(logits.shape), (3, 17))
        torch.testing.assert_close(logits.argmax(-1), labels)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
