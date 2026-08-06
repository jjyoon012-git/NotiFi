import unittest

import torch
from torch import nn

from notifi_pose.hybrid_v10 import P2V13StateRootHybridNet


class _Base(nn.Module):
    hidden = 96


class V13StateRootTests(unittest.TestCase):
    def test_new_state_path_starts_as_identity(self):
        model = P2V13StateRootHybridNet(_Base())
        feature = torch.randn(2, 12, 128)
        root = torch.randn(2, 12, 3)
        valid = torch.ones(2, 12, dtype=torch.bool)
        pooled = feature.mean(1)
        adjusted, auxiliary = model.root_candidate(
            feature, root, valid, pooled
        )
        self.assertTrue(torch.allclose(adjusted, root, atol=1e-6))
        self.assertTrue(torch.allclose(
            auxiliary["root_velocity_delta_v13"],
            torch.zeros_like(auxiliary["root_velocity_delta_v13"]),
        ))


if __name__ == "__main__":
    unittest.main()
