import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.motion_prior_v9 import TemporalMotionDenoiser, corrupt_motion


class MotionPriorV9Tests(unittest.TestCase):
    def test_identity_initialization_preserves_pose(self):
        model = TemporalMotionDenoiser(hidden=16)
        pose = torch.randn(2, 16, C.N_JOINTS, 3)
        pose = pose - pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        valid = torch.ones(2, 16, dtype=torch.bool)
        output = model(pose, valid)
        self.assertTrue(torch.allclose(output["pose_rel"], pose, atol=1e-6))

    def test_corruption_preserves_shape_and_valid_padding(self):
        pose = torch.randn(2, 16, C.N_JOINTS, 3)
        valid = torch.ones(2, 16, dtype=torch.bool)
        valid[:, -2:] = False
        corrupted, observed = corrupt_motion(
            pose, valid, noise_std=0.02, frame_drop=0.2, joint_drop=0.2
        )
        self.assertEqual(corrupted.shape, pose.shape)
        self.assertEqual(observed.shape, pose.shape[:-1])
        self.assertTrue(torch.equal(
            corrupted[:, -2:], torch.zeros_like(corrupted[:, -2:])
        ))


if __name__ == "__main__":
    unittest.main()
