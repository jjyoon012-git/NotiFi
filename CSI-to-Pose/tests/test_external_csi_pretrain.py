import unittest

import torch

from notifi_pose.tools.pretrain_external_csi import (
    _drop_links,
    _pose_velocity_target,
)


class ExternalPretrainingTargetTests(unittest.TestCase):
    def test_pose_velocity_is_translation_and_scale_invariant(self) -> None:
        pose = torch.randn(2, 12, 17, 3)
        baseline = _pose_velocity_target(pose)
        transformed = _pose_velocity_target(pose * 2.5 + 7.0)
        self.assertTrue(torch.allclose(baseline, transformed, atol=1e-5))

    def test_link_dropout_keeps_one_link(self) -> None:
        torch.manual_seed(1)
        mask = torch.ones(16, 20, 3, dtype=torch.bool)
        output = _drop_links(mask, 1.0)
        self.assertTrue((output.any(1).sum(-1) >= 1).all())
        self.assertTrue((output.any(1).sum(-1) == 2).all())


if __name__ == "__main__":
    unittest.main()
