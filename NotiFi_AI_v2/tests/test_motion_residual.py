"""Shape and identity tests for the continuous motion residual decoder."""

from __future__ import annotations

import unittest

import torch

from notifi_ai_v2.motion_residual import MotionResidualDecoder, local_bones
from notifi_pose import contract as C


class MotionResidualDecoderTest(unittest.TestCase):
    """Verify that the decoder starts as an exact retrieval identity."""

    def test_initial_prediction_equals_coarse_pose(self) -> None:
        """Zero-initialized residual heads must not alter deployment output."""
        model = MotionResidualDecoder(feature_dim=16, hidden=32, layers=2)
        features = torch.randn(2, 12, 16)
        coarse = torch.randn(2, 12, C.N_JOINTS, 3)
        coarse[:, :, C.ROOT_JOINT] = 0.0
        mask = torch.ones(2, 12, dtype=torch.bool)
        action = torch.softmax(torch.randn(2, C.N_CLASSES), dim=-1)
        risk = torch.softmax(torch.randn(2, C.N_RISK), dim=-1)
        output = model(features, coarse, mask, action, risk)
        self.assertTrue(torch.allclose(output["pose_rel"], coarse, atol=1e-6))

    def test_padding_keeps_coarse_pose(self) -> None:
        """Invalid padded frames must stay byte-equivalent to retrieval."""
        model = MotionResidualDecoder(
            feature_dim=8, hidden=32, layers=1, bone_length_blend=1.0
        )
        features = torch.randn(1, 9, 8)
        coarse = torch.randn(1, 9, C.N_JOINTS, 3)
        coarse[:, :, C.ROOT_JOINT] = 0.0
        mask = torch.tensor([[True] * 5 + [False] * 4])
        action = torch.full((1, C.N_CLASSES), 1.0 / C.N_CLASSES)
        risk = torch.full((1, C.N_RISK), 1.0 / C.N_RISK)
        output = model(features, coarse, mask, action, risk)
        self.assertTrue(torch.equal(output["pose_rel"][:, 5:], coarse[:, 5:]))

    def test_refinement_preserves_coarse_bone_lengths(self) -> None:
        """The decoder may rotate bones but must not resize them per frame."""
        model = MotionResidualDecoder(
            feature_dim=8, hidden=32, layers=1, bone_length_blend=1.0
        )
        with torch.no_grad():
            model.delta_head[-1].bias.fill_(0.5)
            model.gate_head[-1].bias.fill_(2.0)
        features = torch.randn(1, 9, 8)
        coarse = torch.randn(1, 9, C.N_JOINTS, 3)
        coarse[:, :, C.ROOT_JOINT] = 0.0
        mask = torch.ones(1, 9, dtype=torch.bool)
        action = torch.full((1, C.N_CLASSES), 1.0 / C.N_CLASSES)
        risk = torch.full((1, C.N_RISK), 1.0 / C.N_RISK)
        refined = model(features, coarse, mask, action, risk)["pose_rel"]
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(local_bones(coarse), dim=-1),
            torch.linalg.vector_norm(local_bones(refined), dim=-1),
            atol=1e-5,
        ))


if __name__ == "__main__":
    unittest.main()
