import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.conditioned_contact_pose import (
    CONTACT_JOINTS,
    DirectionalConditionedContactPose,
)
from notifi_pose.tools.train_conditioned_contact_pose import (
    fall_phase_targets,
    timestamp_aware_alignment,
)


class _FakeMotion(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.projection = nn.Linear(2, hidden)

    def forward(self, csi, link_mask):
        pooled = csi.mean((-3, -2))
        return self.projection(pooled), pooled.square().mean(-1)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.latent_head = nn.Conv1d(hidden, 8, 1)
        self.dynamic = _FakeMotion(hidden)


class _FakePoseModel(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.backbone = _FakeBackbone(hidden)
        self.feature = nn.Linear(2, hidden)

    def forward(self, csi, link_mask):
        pooled = csi.mean((-3, -2))
        features = self.feature(pooled)
        pose = torch.zeros(
            csi.shape[0], csi.shape[1], C.N_JOINTS, 3,
            device=csi.device,
        )
        for joint, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                pose[:, :, joint] = pose[:, :, parent]
                pose[:, :, joint, joint % 3] += 0.05 + 0.002 * joint
        return {
            "pose_rel": pose,
            "csi_pose_features": features,
            "kinetic_velocity": torch.zeros_like(pose),
        }


class ConditionedContactPoseTests(unittest.TestCase):
    def test_zero_initialized_model_reproduces_locked_thirty_percent(self):
        torch.manual_seed(4)
        model = DirectionalConditionedContactPose(_FakePoseModel())
        csi = torch.randn(2, 12, C.N_LINKS, 7, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        coarse = torch.zeros(2, 12, C.N_JOINTS, 3)
        output = model(csi, mask, coarse)
        expected = coarse + 0.30 * (output["pose_candidate"] - coarse)
        self.assertTrue(torch.allclose(output["pose_rel"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(output["pose_delta"], torch.zeros_like(expected), atol=1e-6))
        self.assertTrue(torch.allclose(
            output["joint_confidence_gate"],
            torch.full_like(output["joint_confidence_gate"], 0.30),
            atol=1e-6,
        ))

    def test_fall_phase_uses_floor_contact_transition(self):
        frames = 16
        pose = torch.zeros(1, frames, C.N_JOINTS, 3)
        root = torch.zeros(1, frames, 3)
        valid = torch.ones(1, frames, dtype=torch.bool)
        risk = torch.tensor([2])
        pose[..., 1] = 1.0
        for name in ("left_ankle", "right_ankle", "left_foot", "right_foot"):
            pose[:, :, C.JOINT_INDEX[name], 1] = 0.0
        heights = (0.80, 0.50, 0.20, 0.05)
        for offset, height in enumerate(heights, start=3):
            pose[:, offset:, list(CONTACT_JOINTS), 1] = height
        phase, mask = fall_phase_targets(pose, root, valid, risk)
        self.assertTrue(mask.all())
        self.assertTrue((phase == 1).any())
        self.assertTrue((phase == 2).any())
        self.assertTrue((phase == 3).any())
        self.assertLess(int(torch.nonzero(phase[0] == 1)[0]), int(torch.nonzero(phase[0] == 2)[0]))

    def test_timestamp_alignment_is_zero_for_identical_motion(self):
        torch.manual_seed(2)
        pose = torch.randn(2, 20, C.N_JOINTS, 3) * 0.05
        valid = torch.ones(2, 20, dtype=torch.bool)
        risk = torch.tensor([0, 2])
        exact = torch.tensor([True, False])
        loss = timestamp_aware_alignment(pose, pose, valid, risk, exact)
        self.assertLess(float(loss), 1e-6)


if __name__ == "__main__":
    unittest.main()
