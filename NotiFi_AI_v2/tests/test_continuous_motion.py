"""Contract tests for direct continuous CSI-to-motion generation."""

from __future__ import annotations

import unittest

import torch

from notifi_ai_v2.continuous_motion import (
    ContinuousMotionGenerator,
    continuous_motion_loss,
)
from notifi_pose import contract as C


class ContinuousMotionGeneratorTest(unittest.TestCase):
    """Verify shapes, masks, and kinematic root constraints."""

    def model(self) -> ContinuousMotionGenerator:
        lengths = torch.full((C.N_JOINTS,), 0.2)
        lengths[C.ROOT_JOINT] = 0.0
        directions = torch.randn(C.N_JOINTS, 3)
        directions[C.ROOT_JOINT] = torch.tensor((0.0, 1.0, 0.0))
        return ContinuousMotionGenerator(
            16, lengths, directions, hidden=48, temporal_layers=2,
            attention_layers=1,
        )

    def test_output_shape_and_root(self) -> None:
        """Generated motion follows the body-22 root-relative contract."""
        model = self.model()
        features = torch.randn(2, 12, 16)
        mask = torch.ones(2, 12, dtype=torch.bool)
        action = torch.softmax(torch.randn(2, C.N_CLASSES), dim=-1)
        risk = torch.softmax(torch.randn(2, C.N_RISK), dim=-1)
        output = model(features, mask, action, risk)
        self.assertEqual(output["pose_rel"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertTrue(torch.equal(
            output["pose_rel"][:, :, C.ROOT_JOINT],
            torch.zeros_like(output["pose_rel"][:, :, C.ROOT_JOINT]),
        ))

    def test_padding_is_zero(self) -> None:
        """Padded frames cannot emit a synthetic pose."""
        model = self.model()
        features = torch.randn(1, 8, 16)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        action = torch.full((1, C.N_CLASSES), 1.0 / C.N_CLASSES)
        risk = torch.full((1, C.N_RISK), 1.0 / C.N_RISK)
        pose = model(features, mask, action, risk)["pose_rel"]
        self.assertTrue(torch.equal(pose[:, 5:], torch.zeros_like(pose[:, 5:])))

    def test_backward_updates_parameters(self) -> None:
        """A real pose loss produces finite gradients and changes a head weight."""
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        features = torch.randn(2, 8, 16)
        mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
        action = torch.softmax(torch.randn(2, C.N_CLASSES), dim=-1)
        risk_probability = torch.softmax(torch.randn(2, C.N_RISK), dim=-1)
        target = torch.randn(2, 8, C.N_JOINTS, 3) * 0.2
        target[:, :, C.ROOT_JOINT] = 0.0
        before = model.direction_head[-1].weight.detach().clone()
        output = model(features, mask, action, risk_probability)
        loss, _ = continuous_motion_loss(
            output, target, mask, torch.tensor((0, 2)), (7, 8, 10, 11, 20, 21),
        )
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.direction_head[-1].weight))


if __name__ == "__main__":
    unittest.main()
