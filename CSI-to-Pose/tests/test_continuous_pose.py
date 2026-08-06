from __future__ import annotations

import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.continuous_pose import CSILatentPoseRegressor
from notifi_pose.motion_tokens import KinematicMotionTokenizer
from notifi_pose.nets import PoseNet


class ContinuousPoseTests(unittest.TestCase):
    def test_csi_latent_pose_is_csi_only_and_freezes_teachers(self):
        base = PoseNet(hidden=32, dilations=(1, 2), n_blocks=1, dropout=0.0)
        tokenizer = KinematicMotionTokenizer(
            hidden=32, code_dim=16, downsample=2, continuous=True
        )
        model = CSILatentPoseRegressor(
            base, tokenizer.decoder, torch.zeros(16), torch.ones(16),
            torch.full((C.N_JOINTS,), 0.2), hidden=32, code_dim=16,
            temporal_layers=1, heads=4, dropout=0.0,
        )
        csi = torch.randn(2, 68, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 68, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["motion_latent"].shape, (2, 34, 16))
        self.assertEqual(output["pose_rel"].shape, (2, 68, C.N_JOINTS, 3))
        self.assertFalse(any(p.requires_grad for p in model.base.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.decoder.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.dynamic.parameters()))

    def test_missing_frames_are_zeroed(self):
        base = PoseNet(hidden=32, dilations=(1,), n_blocks=1, dropout=0.0)
        tokenizer = KinematicMotionTokenizer(
            hidden=32, code_dim=16, downsample=2, continuous=True
        )
        lengths = torch.full((C.N_JOINTS,), 0.2)
        lengths[C.ROOT_JOINT] = 0.0
        model = CSILatentPoseRegressor(
            base, tokenizer.decoder, torch.zeros(16), torch.ones(16), lengths,
            hidden=32, code_dim=16, temporal_layers=1, heads=4, dropout=0.0,
        ).eval()
        csi = torch.randn(1, 68, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 68, C.N_LINKS, dtype=torch.bool)
        mask[:, 50:] = False
        output = model(csi, mask)
        self.assertEqual(torch.count_nonzero(output["pose_rel"][:, 50:]), 0)


if __name__ == "__main__":
    unittest.main()
