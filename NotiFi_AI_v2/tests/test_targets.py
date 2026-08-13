import unittest

import torch

from notifi_ai_v2.targets import motion_targets


class MotionTargetTests(unittest.TestCase):
    def test_translation_offset_does_not_change_motion_target(self):
        torch.manual_seed(23)
        pose = torch.randn(2, 12, 22, 3) * 0.1
        root = torch.randn(2, 12, 3) * 0.1
        valid = torch.ones(2, 12, dtype=torch.bool)
        first = motion_targets(pose, root, valid)
        second = motion_targets(
            pose, root + torch.tensor([3.0, -2.0, 5.0]), valid
        )
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))

    def test_invalid_frames_are_zero(self):
        pose = torch.randn(1, 10, 22, 3)
        root = torch.randn(1, 10, 3)
        valid = torch.ones(1, 10, dtype=torch.bool)
        valid[:, 4:7] = False
        target = motion_targets(pose, root, valid)
        self.assertEqual(torch.count_nonzero(target[:, 4:7]), 0)
        self.assertTrue(torch.isfinite(target).all())


if __name__ == "__main__":
    unittest.main()
