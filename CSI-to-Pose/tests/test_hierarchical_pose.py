from __future__ import annotations

import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.continuous_pose import CSILatentPoseRegressor
from notifi_pose.hierarchical_pose import (
    DISTAL_JOINTS,
    HierarchicalCSIPoseRegressor,
    JointConfidencePoseGate,
)
from notifi_pose.motion_tokens import KinematicMotionTokenizer
from notifi_pose.nets import PoseNet
from notifi_pose.tools.train_hierarchical_pose import (
    banded_velocity_alignment,
    danger_keyframe_loss,
)


class HierarchicalPoseTests(unittest.TestCase):
    @staticmethod
    def make_model() -> HierarchicalCSIPoseRegressor:
        base = PoseNet(hidden=32, dilations=(1,), n_blocks=1, dropout=0.0)
        tokenizer = KinematicMotionTokenizer(
            hidden=32, code_dim=16, downsample=2, continuous=True
        )
        lengths = torch.full((C.N_JOINTS,), 0.2)
        lengths[C.ROOT_JOINT] = 0.0
        backbone = CSILatentPoseRegressor(
            base, tokenizer.decoder, torch.zeros(16), torch.ones(16), lengths,
            hidden=32, code_dim=16, temporal_layers=1, heads=4, dropout=0.0,
        )
        return HierarchicalCSIPoseRegressor(backbone, dropout=0.0)

    def test_zero_initialized_heads_preserve_kp2c_pose(self):
        torch.manual_seed(3)
        model = self.make_model().eval()
        csi = torch.randn(2, 68, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 68, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            expected = model.backbone(csi, mask)["pose_rel"]
            actual = model(csi, mask)["pose_rel"]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_outputs_explicit_hierarchy_and_masks_missing_frames(self):
        model = self.make_model().eval()
        csi = torch.randn(1, 68, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 68, C.N_LINKS, dtype=torch.bool)
        mask[:, 50:] = False
        output = model(csi, mask)
        self.assertEqual(output["pose_rel"].shape, (1, 68, C.N_JOINTS, 3))
        self.assertEqual(
            output["endpoint_delta"].shape, (1, 68, len(DISTAL_JOINTS), 3)
        )
        self.assertEqual(
            output["kinetic_velocity"].shape, (1, 68, C.N_JOINTS, 3)
        )
        self.assertEqual(torch.count_nonzero(output["pose_rel"][:, 50:]), 0)

    def test_all_hierarchical_heads_receive_gradients(self):
        model = self.make_model().train()
        csi = torch.randn(1, 36, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 36, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        loss = (
            output["pose_rel"].square().mean()
            + (output["kinetic_velocity"] - 1.0).square().mean()
        )
        loss.backward()
        for name in (
            "torso_direction_head", "limb_direction_head",
            "endpoint_head", "velocity_head",
        ):
            gradients = [
                parameter.grad for parameter in getattr(model, name).parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(any(
                gradient is not None and torch.count_nonzero(gradient)
                for gradient in gradients
            ), name)

    def test_joint_gate_starts_at_requested_blend_and_freezes_pose_model(self):
        pose_model = self.make_model().eval()
        model = JointConfidencePoseGate(
            pose_model, initial_strength=0.3, hidden=16, dropout=0.0
        ).eval()
        csi = torch.randn(1, 36, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 36, C.N_LINKS, dtype=torch.bool)
        coarse = torch.randn(1, 36, C.N_JOINTS, 3)
        output = model(csi, mask, coarse)
        expected = coarse + 0.3 * (output["pose_candidate"] - coarse)
        torch.testing.assert_close(
            output["pose_rel"], expected, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            output["joint_confidence_gate"],
            torch.full_like(output["joint_confidence_gate"], 0.3),
        )
        self.assertFalse(any(
            parameter.requires_grad for parameter in pose_model.parameters()
        ))
        self.assertTrue(any(
            parameter.requires_grad for parameter in model.gate_head.parameters()
        ))

    def test_banded_alignment_is_differentiable_for_shifted_motion(self):
        torch.manual_seed(9)
        velocity = 0.03 * torch.randn(1, 20, C.N_JOINTS, 3)
        target = velocity.cumsum(1)
        predicted = torch.zeros_like(target)
        predicted[:, 1:] = target[:, :-1]
        predicted.requires_grad_(True)
        valid = torch.ones(1, 20, dtype=torch.bool)
        weight = torch.ones(1, 20)
        loss = banded_velocity_alignment(
            predicted, target, valid, weight,
            radius=2, temperature=0.01, lag_penalty=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(torch.count_nonzero(predicted.grad), 0)

    def test_danger_keyframes_select_only_fast_danger_frames(self):
        predicted = torch.zeros(2, 20, C.N_JOINTS, 3, requires_grad=True)
        target = torch.randn_like(predicted)
        valid = torch.ones(2, 20, dtype=torch.bool)
        risk = torch.tensor([2, 0])
        speed = torch.arange(20, dtype=torch.float32).repeat(2, 1)
        loss, selected = danger_keyframe_loss(
            predicted, target, valid, risk, speed,
            frames=5, distal_scale=0.75,
        )
        self.assertEqual(int(selected[0].sum()), 5)
        self.assertEqual(int(selected[1].sum()), 0)
        self.assertEqual(
            torch.nonzero(selected[0], as_tuple=False).flatten().tolist(),
            [15, 16, 17, 18, 19],
        )
        loss.backward()
        self.assertGreater(torch.count_nonzero(predicted.grad[0]), 0)
        self.assertEqual(torch.count_nonzero(predicted.grad[1]), 0)


if __name__ == "__main__":
    unittest.main()
