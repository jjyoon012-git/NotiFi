from __future__ import annotations

import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.motion_tokens import (
    FactorizedMotionTokenizer,
    KinematicMotionTokenizer,
    forward_kinematics,
    pose_to_bones,
    trial_bone_lengths,
)


class MotionTokenTests(unittest.TestCase):
    def test_continuous_bottleneck_preserves_encoder_gradient(self):
        model = KinematicMotionTokenizer(
            hidden=32, code_dim=16, downsample=2, continuous=True
        )
        pose = torch.randn(2, 8, C.N_JOINTS, 3)
        pose[:, :, C.ROOT_JOINT] = 0.0
        valid = torch.ones(2, 8, dtype=torch.bool)
        output = model(pose, valid)
        output["quantized"].square().mean().backward()
        self.assertIsNotNone(model.encoder.input[0].weight.grad)
        self.assertGreater(float(model.encoder.input[0].weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(output["latent"], output["quantized"]))

    def test_bone_conversion_round_trip(self):
        torch.manual_seed(2)
        directions = torch.randn(2, 20, C.N_JOINTS, 3)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        directions[:, :, C.ROOT_JOINT] = 0.0
        lengths = 0.1 + 0.3 * torch.rand(2, C.N_JOINTS)
        lengths[:, C.ROOT_JOINT] = 0.0
        pose = forward_kinematics(directions, lengths)
        recovered_direction, _ = pose_to_bones(pose)
        recovered_length = trial_bone_lengths(
            pose, torch.ones(2, 20, dtype=torch.bool)
        )
        torch.testing.assert_close(
            recovered_direction[:, :, 1:], directions[:, :, 1:],
            atol=2e-6, rtol=2e-6,
        )
        torch.testing.assert_close(recovered_length, lengths, atol=2e-6, rtol=2e-6)

    def test_tokenizer_downsamples_by_four_and_reconstructs_shape(self):
        model = KinematicMotionTokenizer(
            hidden=32, code_dim=16, codes=32, dropout=0.0
        )
        pose = torch.randn(2, 68, C.N_JOINTS, 3)
        pose -= pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        valid = torch.ones(2, 68, dtype=torch.bool)
        output = model(pose, valid)
        self.assertEqual(output["token_ids"].shape, (2, 17))
        self.assertEqual(output["pose_rel"].shape, pose.shape)
        self.assertTrue(torch.isfinite(output["pose_rel"]).all())
        self.assertTrue(torch.isfinite(output["codebook_loss"]))

    def test_padded_frames_stay_zero(self):
        model = KinematicMotionTokenizer(
            hidden=32, code_dim=16, codes=32, dropout=0.0
        ).eval()
        pose = torch.randn(1, 68, C.N_JOINTS, 3)
        pose -= pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        valid = torch.ones(1, 68, dtype=torch.bool)
        valid[:, 52:] = False
        output = model(pose, valid)
        self.assertEqual(torch.count_nonzero(output["pose_rel"][:, 52:]), 0)

    def test_residual_tokens_keep_level_axis(self):
        model = KinematicMotionTokenizer(
            hidden=32, code_dim=16, codes=32, dropout=0.0,
            downsample=2, quantizer_levels=2,
        )
        pose = torch.randn(2, 68, C.N_JOINTS, 3)
        pose -= pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        valid = torch.ones(2, 68, dtype=torch.bool)
        output = model(pose, valid)
        self.assertEqual(output["token_ids"].shape, (2, 34, 2))
        self.assertEqual(output["pose_rel"].shape, pose.shape)

    def test_factorized_tokens_cover_five_body_parts(self):
        model = FactorizedMotionTokenizer(
            hidden=32, code_dim=16, codes=32, dropout=0.0,
            downsample=2, part_hidden=24,
        )
        pose = torch.randn(2, 68, C.N_JOINTS, 3)
        pose -= pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        valid = torch.ones(2, 68, dtype=torch.bool)
        output = model(pose, valid)
        self.assertEqual(output["token_ids"].shape, (2, 34, 5))
        self.assertEqual(output["pose_rel"].shape, pose.shape)


if __name__ == "__main__":
    unittest.main()
