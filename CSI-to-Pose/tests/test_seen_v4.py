import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.seen_v4 import (
    AlignmentRobustTrajectoryNet,
    bounded_piecewise_alignment_loss,
    trajectory_reconstruction_loss,
)


def make_pose(batch: int, frames: int, device=None) -> torch.Tensor:
    pose = torch.zeros(batch, frames, C.N_JOINTS, 3, device=device)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            offset = pose.new_tensor((0.03 * ((child % 3) - 1), 0.08, 0.02))
            pose[:, :, child] = pose[:, :, parent] + offset
    return pose


class DummyV3(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.hidden = hidden

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        device = csi.device
        pose = make_pose(batch, frames, device)
        feature = torch.zeros(batch, frames, self.hidden, device=device)
        return {
            "pose_rel": pose,
            "root": torch.zeros(batch, frames, 3, device=device),
            "temporal_features": feature,
            "motion_features": feature,
            "temporal_features_v3": feature,
            "class_logits": torch.zeros(batch, C.N_CLASSES, device=device),
            "risk_logits": torch.zeros(batch, C.N_RISK, device=device),
        }


class SeenV4Tests(unittest.TestCase):
    def make_inputs(self):
        csi = torch.randn(2, 24, 3, 8, 2)
        link_mask = torch.ones(2, 24, 3, dtype=torch.bool)
        return csi, link_mask

    def test_zero_strength_exactly_preserves_v7_pose_and_root(self):
        model = AlignmentRobustTrajectoryNet(DummyV3(), hidden=16)
        model.set_calibration(0.0, 0.0)
        output = model(*self.make_inputs())
        self.assertTrue(torch.allclose(output["pose_rel"], output["pose_v7"], atol=1e-6))
        self.assertTrue(torch.equal(output["root"], output["root_v7"]))

    def test_piecewise_alignment_recovers_small_shift(self):
        target = torch.zeros(1, 32, 2)
        target[:, 10:18, 0] = 1.0
        predicted = torch.zeros_like(target)
        predicted[:, 13:21, 0] = 1.0
        valid = torch.ones(1, 32, dtype=torch.bool)
        fixed = bounded_piecewise_alignment_loss(
            predicted, target, valid, max_shift=0, segments=4
        )
        aligned = bounded_piecewise_alignment_loss(
            predicted, target, valid, max_shift=5, segments=4,
            shift_penalty=0.0, transition_penalty=0.0,
        )
        self.assertLess(float(aligned), float(fixed))

    def test_trajectory_loss_is_finite_and_backward(self):
        model = AlignmentRobustTrajectoryNet(DummyV3(), hidden=16)
        csi, link_mask = self.make_inputs()
        output = model(csi, link_mask)
        target_pose = make_pose(2, 24)
        target_pose[:, 8:18, C.JOINT_INDEX["left_wrist"], 0] += 0.10
        target_root = torch.zeros(2, 24, 3)
        target_root[:, :, C.UP_AXIS] = torch.linspace(0.8, 0.1, 24)
        batch = {
            "pose_rel": target_pose,
            "root": target_root,
            "valid": torch.ones(2, 24, dtype=torch.bool),
            "risk_id": torch.tensor([0, 2]),
            "quality_weight": torch.ones(2),
        }
        loss, parts = trajectory_reconstruction_loss(
            output, batch, alignment_weight=0.1, max_shift=3
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("alignment", parts)
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)


if __name__ == "__main__":
    unittest.main()
