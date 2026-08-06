import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.dynamic_motion import DynamicMotionPoseNet
from notifi_pose.nets import PerLinkNorm


class _FrozenBase(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.norm = PerLinkNorm()
        self.projection = nn.Linear(2, hidden)
        self.class_head = nn.Linear(hidden, C.N_CLASSES)
        self.risk_head = nn.Linear(hidden, C.N_RISK)
        self.root_head = nn.Linear(hidden, 3)

    def forward(self, csi, link_mask):
        features = self.projection(csi.mean((2, 3)))
        mask = link_mask.any(-1)
        pooled = (features * mask[..., None]).sum(1)
        pooled = pooled / mask.sum(1, keepdim=True).clamp_min(1)
        pose = torch.zeros(
            csi.shape[0], csi.shape[1], C.N_JOINTS, 3,
            device=csi.device, dtype=csi.dtype,
        )
        for joint, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                pose[:, :, joint] = pose[:, :, parent]
                pose[:, :, joint, 1] += 0.2
        return {
            "temporal_features": features,
            "pose_rel": pose,
            "class_logits": self.class_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "root": self.root_head(features),
        }


def _priors():
    directions = torch.zeros(C.N_JOINTS, 3)
    lengths = torch.zeros(C.N_JOINTS)
    for joint, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            directions[joint, 1] = 1.0
            lengths[joint] = 0.2
    return directions, lengths


class DynamicMotionPoseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        directions, lengths = _priors()
        self.base = _FrozenBase()
        self.model = DynamicMotionPoseNet(
            self.base, directions, lengths, hidden=32,
            dynamic_layers=1, heads=4, dropout=0.0,
        )
        self.csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        self.mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        self.mask[1, -2:] = False

    def test_output_contract_and_masks(self):
        output = self.model(self.csi, self.mask)
        self.assertEqual(output["pose_rel"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertEqual(output["pose_anchor"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertEqual(output["bone_direction"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertEqual(output["bone_lengths"].shape, (2, C.N_JOINTS))
        self.assertEqual(output["phase_logits"].shape, (2, 12, 4))
        self.assertEqual(output["contact_logits"].shape[:2], (2, 12))
        self.assertEqual(output["motion_profile"].shape, (2, 12, 7))
        self.assertEqual(output["class_logits"].shape, (2, C.N_CLASSES))
        self.assertEqual(output["risk_logits"].shape, (2, C.N_RISK))
        self.assertEqual(output["action_probability"].shape, (2, C.N_CLASSES))
        self.assertEqual(output["risk_probability"].shape, (2, C.N_RISK))
        self.assertEqual(output["semantic_gate"].shape, (2, 12, 32))
        self.assertTrue(torch.isfinite(output["pose_rel"]).all())
        self.assertTrue(torch.equal(
            output["pose_rel"][1, -2:], torch.zeros_like(output["pose_rel"][1, -2:])
        ))

    def test_zero_initialized_heads_preserve_base_classification(self):
        self.model.eval()
        with torch.no_grad():
            base = self.base(self.csi, self.mask)
            output = self.model(self.csi, self.mask)
        torch.testing.assert_close(output["class_logits"], base["class_logits"])
        torch.testing.assert_close(output["risk_logits"], base["risk_logits"])

    def test_pose_semantics_come_from_frozen_csi_classifier(self):
        self.model.eval()
        with torch.no_grad():
            base = self.base(self.csi, self.mask)
            output = self.model(self.csi, self.mask)
        torch.testing.assert_close(
            output["action_probability"],
            torch.softmax(base["class_logits"], dim=-1),
        )
        torch.testing.assert_close(
            output["risk_probability"],
            torch.softmax(base["risk_logits"], dim=-1),
        )

    def test_external_csi_only_anchor_is_preserved_at_initialization(self):
        external = self.base(self.csi, self.mask)["pose_rel"].clone()
        external[:, :, 1:, 0] += 0.05
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.csi, self.mask, external)
        valid = self.mask.any(-1)[..., None, None]
        torch.testing.assert_close(
            output["pose_rel"], external * valid,
            atol=1e-5, rtol=1e-5,
        )

    def test_zero_initialized_pose_heads_preserve_base_bone_directions(self):
        self.model.eval()
        with torch.no_grad():
            base = self.base(self.csi, self.mask)
            output = self.model(self.csi, self.mask)
        base_bones = torch.zeros_like(base["pose_rel"])
        output_bones = torch.zeros_like(output["pose_rel"])
        for joint, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                base_bones[:, :, joint] = (
                    base["pose_rel"][:, :, joint]
                    - base["pose_rel"][:, :, parent]
                )
                output_bones[:, :, joint] = (
                    output["pose_rel"][:, :, joint]
                    - output["pose_rel"][:, :, parent]
                )
        valid = self.mask.any(-1)[..., None, None]
        base_directions = torch.nn.functional.normalize(base_bones, dim=-1)
        output_directions = torch.nn.functional.normalize(output_bones, dim=-1)
        torch.testing.assert_close(
            output_directions * valid, base_directions * valid,
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            output["pose_rel"], output["pose_anchor"],
            atol=1e-5, rtol=1e-5,
        )

    def test_classification_gradient_is_decoupled_from_pose_encoder(self):
        output = self.model(self.csi, self.mask)
        (output["class_logits"].sum() + output["risk_logits"].sum()).backward()
        class_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.classification_parameters()
            if parameter.grad is not None
        )
        pose_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.pose_parameters()
            if parameter.grad is not None
        )
        self.assertGreater(class_gradient, 0.0)
        self.assertEqual(pose_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
