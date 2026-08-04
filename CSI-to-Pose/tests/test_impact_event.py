import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.impact_event import (
    ImpactEventLocalizer,
    N_IMPACT_REGIONS,
    impact_event_loss,
    physical_impact_targets,
    raw_csi_event_features,
)
from notifi_pose.seen_v2 import N_INJURY_JOINTS


class DummyV3(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.hidden = hidden

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        device = csi.device
        return {
            "pose_rel": torch.zeros(batch, frames, C.N_JOINTS, 3, device=device),
            "root": torch.zeros(batch, frames, 3, device=device),
            "phase_logits": torch.zeros(batch, frames, 4, device=device),
            "impact_logits": torch.zeros(batch, frames, device=device),
            "injury_contact_logits": torch.zeros(
                batch, frames, N_INJURY_JOINTS, device=device
            ),
            "first_contact_logits": torch.zeros(
                batch, N_INJURY_JOINTS, device=device
            ),
            "joint_impact_speed": torch.zeros(
                batch, frames, N_INJURY_JOINTS, device=device
            ),
            "temporal_features": torch.zeros(
                batch, frames, self.hidden, device=device
            ),
        }


class ImpactEventTests(unittest.TestCase):
    def make_batch(self):
        batch, frames = 2, 20
        pose = torch.zeros(batch, frames, C.N_JOINTS, 3)
        root = torch.zeros(batch, frames, 3)
        root[0, :, 1] = torch.linspace(1.0, 0.1, frames)
        return {
            "pose_rel": pose,
            "root": root,
            "valid": torch.ones(batch, frames, dtype=torch.bool),
            "risk_id": torch.tensor([2, 0]),
            "quality_weight": torch.ones(batch),
        }

    def test_physical_target_has_one_danger_event(self):
        batch = self.make_batch()
        target = physical_impact_targets(
            batch["pose_rel"], batch["root"],
            batch["valid"], batch["risk_id"],
        )
        self.assertTrue(target["event_valid"][0])
        self.assertFalse(target["event_valid"][1])
        self.assertAlmostEqual(float(target["event_soft"][0].sum()), 1.0, places=5)
        self.assertEqual(float(target["event_soft"][1].sum()), 0.0)

    def test_event_output_shapes_and_finite_loss(self):
        model = ImpactEventLocalizer(DummyV3(), hidden=16)
        csi = torch.randn(2, 20, 3, 8, 2)
        link_mask = torch.ones(2, 20, 3, dtype=torch.bool)
        output = model(csi, link_mask)
        self.assertEqual(tuple(output["event_logits_v8"].shape), (2, 20))
        self.assertEqual(
            tuple(output["joint_time_logits_v8"].shape),
            (2, 20, N_INJURY_JOINTS),
        )
        self.assertEqual(
            tuple(output["region_time_logits_v8"].shape),
            (2, 20, N_IMPACT_REGIONS),
        )
        loss, parts = impact_event_loss(output, self.make_batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("joint_time", parts)
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
            if parameter.requires_grad
        ))

    def test_raw_csi_event_features_are_finite(self):
        csi = torch.randn(2, 20, 3, 8, 2)
        link_mask = torch.ones(2, 20, 3, dtype=torch.bool)
        feature = raw_csi_event_features(csi, link_mask)
        self.assertEqual(tuple(feature.shape), (2, 20, 4))
        self.assertTrue(torch.isfinite(feature).all())

    def test_zero_calibration_preserves_legacy_event_heads(self):
        model = ImpactEventLocalizer(DummyV3(), hidden=16)
        model.set_calibration(event=0, joint=0, contact=0, speed=0)
        csi = torch.randn(2, 20, 3, 8, 2)
        link_mask = torch.ones(2, 20, 3, dtype=torch.bool)
        output = model(csi, link_mask)
        self.assertTrue(torch.equal(
            output["event_logits_v8"], output["impact_logits"]
        ))
        self.assertTrue(torch.equal(
            output["injury_contact_logits"],
            torch.zeros_like(output["injury_contact_logits"]),
        ))


if __name__ == "__main__":
    unittest.main()
