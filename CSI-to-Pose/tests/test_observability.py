import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from notifi_pose.tools.diagnose_observability import (
    ShuffledSignalDataset,
    binary_metrics,
    macro_f1,
    report_path,
)


class _SignalDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "csi": torch.tensor([index], dtype=torch.float32),
            "link_mask": torch.tensor([index], dtype=torch.long),
            "target_id": torch.tensor(index),
        }


class ObservabilityDiagnosticTests(unittest.TestCase):
    def test_shuffle_replaces_signal_but_preserves_target(self) -> None:
        shuffled = ShuffledSignalDataset(_SignalDataset(), seed=7)
        for index in range(len(shuffled)):
            item = shuffled[index]
            self.assertEqual(int(item["target_id"]), index)
            self.assertEqual(
                int(item["csi"].item()), int(shuffled.permutation[index])
            )
        self.assertFalse(torch.equal(
            torch.as_tensor(shuffled.permutation), torch.arange(len(shuffled))
        ))

    def test_binary_metrics(self) -> None:
        metrics = binary_metrics(
            torch.tensor([0.9, 0.8, 0.7, 0.1]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)

    def test_macro_f1_is_one_for_perfect_labels(self) -> None:
        target = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        self.assertAlmostEqual(macro_f1(target, target, classes=4), 1.0)

    def test_project_path_is_reported_without_private_prefix(self) -> None:
        path = Path(__file__).resolve().parents[1] / "work_v2" / "model.pt"
        self.assertEqual(report_path(path), "work_v2/model.pt")


if __name__ == "__main__":
    unittest.main()
