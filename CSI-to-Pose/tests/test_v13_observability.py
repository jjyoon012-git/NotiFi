import unittest

import pandas as pd
import torch
from torch.utils.data import Dataset

from notifi_pose.tools.audit_v13_observability import ObservabilityPerturbation


class _ToyDataset(Dataset):
    def __init__(self):
        self.index = pd.DataFrame({
            "subject": ["a", "a", "b"],
            "environment": ["E01", "E01", "E02"],
            "class_id": [3, 3, 3],
        })

    def __len__(self):
        return len(self.index)

    def __getitem__(self, position):
        csi = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
        return {
            "csi": csi + 10.0 * position,
            "link_mask": torch.ones(4, 1, dtype=torch.bool),
            "class_id": torch.tensor(3),
        }


class V13ObservabilityTests(unittest.TestCase):
    def test_same_site_class_shuffle_preserves_target(self):
        dataset = ObservabilityPerturbation(
            _ToyDataset(), "same_site_class_shuffle", seed=17
        )
        sample = dataset[0]
        self.assertEqual(int(sample["class_id"]), 3)
        self.assertTrue(torch.equal(
            sample["csi"], _ToyDataset()[1]["csi"]
        ))

    def test_temporal_perturbations_preserve_source_dataset(self):
        source = _ToyDataset()
        reversed_sample = ObservabilityPerturbation(source, "time_reverse")[0]
        mean_sample = ObservabilityPerturbation(source, "time_mean")[0]
        self.assertEqual(reversed_sample["csi"].flatten().tolist(), [3, 2, 1, 0])
        self.assertTrue(torch.allclose(
            mean_sample["csi"], torch.full_like(mean_sample["csi"], 1.5)
        ))
        self.assertEqual(source[0]["csi"].flatten().tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
