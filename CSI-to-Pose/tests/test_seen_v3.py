import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.seen_v3 import ContactGuidedRootNet, contact_guided_root_loss


class DummyV2(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.hidden = hidden

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        device = csi.device
        return {
            "pose_rel": torch.zeros(batch, frames, C.N_JOINTS, 3, device=device),
            "root": torch.zeros(batch, frames, 3, device=device),
            "contact_logits": torch.zeros(batch, frames, 4, device=device),
            "phase_logits": torch.zeros(batch, frames, 4, device=device),
            "impact_logits": torch.zeros(batch, frames, device=device),
            "temporal_features": torch.zeros(batch, frames, self.hidden, device=device),
            "class_logits": torch.zeros(batch, C.N_CLASSES, device=device),
            "risk_logits": torch.zeros(batch, C.N_RISK, device=device),
        }


class SeenV3Tests(unittest.TestCase):
    def make_inputs(self):
        csi = torch.randn(2, 12, 3, 8, 2)
        link_mask = torch.ones(2, 12, 3, dtype=torch.bool)
        return csi, link_mask

    def test_zero_strength_preserves_v2_root_and_pose(self):
        model = ContactGuidedRootNet(DummyV2(), hidden=16)
        model.set_root_strength(0.0)
        csi, link_mask = self.make_inputs()
        output = model(csi, link_mask)
        self.assertTrue(torch.equal(output["root"], output["root_v2"]))
        self.assertEqual(tuple(output["pose_rel"].shape), (2, 12, C.N_JOINTS, 3))

    def test_contact_guided_outputs_are_finite(self):
        model = ContactGuidedRootNet(DummyV2(), hidden=16)
        csi, link_mask = self.make_inputs()
        output = model(csi, link_mask)
        self.assertTrue(torch.isfinite(output["root"]).all())
        self.assertEqual(tuple(output["contact_logits"].shape), (2, 12, 4))
        self.assertEqual(tuple(output["root_velocity_v3"].shape), (2, 12, 3))

    def test_root_loss_is_finite_and_backward(self):
        model = ContactGuidedRootNet(DummyV2(), hidden=16)
        csi, link_mask = self.make_inputs()
        output = model(csi, link_mask)
        batch = {
            "pose_rel": torch.zeros(2, 12, C.N_JOINTS, 3),
            "root": torch.zeros(2, 12, 3),
            "valid": torch.ones(2, 12, dtype=torch.bool),
            "risk_id": torch.tensor([0, 2]),
            "quality_weight": torch.ones(2),
        }
        loss, parts = contact_guided_root_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("foot_slip", parts)
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)


if __name__ == "__main__":
    unittest.main()
