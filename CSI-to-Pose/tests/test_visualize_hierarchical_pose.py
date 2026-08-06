import unittest

import torch
from torch import nn

from notifi_pose.tools.visualize_hierarchical_pose import HierarchicalPoseBlend


class _FixedModel(nn.Module):
    def __init__(self, pose, root=None):
        super().__init__()
        self.register_buffer("pose", pose)
        self.register_buffer(
            "root", torch.zeros(pose.shape[:2] + (3,)) if root is None else root
        )

    def forward(self, csi, link_mask):
        batch = len(csi)
        return {
            "pose_rel": self.pose.expand(batch, -1, -1, -1),
            "root": self.root.expand(batch, -1, -1),
            "class_logits": torch.tensor(((0.0, 1.0),)).expand(batch, -1),
            "risk_logits": torch.tensor(((0.0, 0.0, 1.0),)).expand(batch, -1),
        }


class HierarchicalPoseBlendTests(unittest.TestCase):
    def test_locked_blend_preserves_coarse_non_pose_outputs(self):
        coarse_pose = torch.zeros(1, 2, 22, 3)
        candidate_pose = torch.full_like(coarse_pose, 2.0)
        coarse_root = torch.full((1, 2, 3), 7.0)
        model = HierarchicalPoseBlend(
            _FixedModel(coarse_pose, coarse_root),
            _FixedModel(candidate_pose),
            strength=0.30,
        )
        output = model(torch.zeros(1, 2, 3), torch.ones(1, 2, 3, dtype=torch.bool))
        self.assertTrue(torch.allclose(output["pose_rel"], torch.full_like(coarse_pose, 0.6)))
        self.assertTrue(torch.equal(output["root"], coarse_root))
        self.assertEqual(int(output["class_logits"].argmax(-1)), 1)
        self.assertEqual(int(output["risk_logits"].argmax(-1)), 2)

    def test_rejects_out_of_range_strength(self):
        pose = torch.zeros(1, 2, 22, 3)
        with self.assertRaises(ValueError):
            HierarchicalPoseBlend(_FixedModel(pose), _FixedModel(pose), 1.01)


if __name__ == "__main__":
    unittest.main()
