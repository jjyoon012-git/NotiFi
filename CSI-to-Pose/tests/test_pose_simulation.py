import unittest

import torch

from notifi_pose.pose_simulation import fill_pose_gaps


class FillPoseGapsTest(unittest.TestCase):
    def test_interpolates_middle_and_clamps_edges(self) -> None:
        pose = torch.tensor([0.0, 0.0, 2.0, 0.0, 4.0, 0.0]).view(1, 6, 1, 1)
        pose = pose.expand(-1, -1, 1, 3).clone()
        valid = torch.tensor([[False, False, True, False, True, False]])

        filled = fill_pose_gaps(pose, valid)

        self.assertTrue(torch.allclose(
            filled[0, :, 0, 0], torch.tensor([2.0, 2.0, 2.0, 3.0, 4.0, 4.0])
        ))

    def test_treats_nonfinite_frame_as_missing(self) -> None:
        pose = torch.zeros(1, 3, 1, 3)
        pose[0, 0] = 1.0
        pose[0, 1] = torch.nan
        pose[0, 2] = 3.0
        valid = torch.ones(1, 3, dtype=torch.bool)

        filled = fill_pose_gaps(pose, valid)

        self.assertTrue(torch.isfinite(filled).all())
        self.assertTrue(torch.allclose(filled[0, 1], torch.full((1, 3), 2.0)))

    def test_empty_sequence_becomes_zero(self) -> None:
        pose = torch.full((1, 2, 1, 3), torch.nan)
        valid = torch.zeros(1, 2, dtype=torch.bool)

        filled = fill_pose_gaps(pose, valid)

        self.assertEqual(float(filled.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
