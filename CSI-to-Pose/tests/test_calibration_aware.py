import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.calibration_aware import (
    CalibrationAwareV14,
    CalibrationSupportEncoder,
    MomentAlignedSupportConditionedP2,
    SupportConditionedP2,
)
from notifi_pose.nets import PoseNet


class FrozenFeatureStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        device = csi.device
        pose = torch.zeros(batch, frames, C.N_JOINTS, 3, device=device)
        root = torch.zeros(batch, frames, 3, device=device)
        return {
            "pose_rel": pose,
            "root": root,
            "class_logits": torch.zeros(batch, C.N_CLASSES, device=device),
            "risk_logits": torch.zeros(batch, C.N_RISK, device=device),
            "temporal_features": torch.zeros(batch, frames, 96, device=device),
            "temporal_features_v10": torch.zeros(batch, frames, 128, device=device),
        }


class CalibrationAwareTests(unittest.TestCase):
    def test_support_encoder_shape_is_trial_independent(self):
        encoder = CalibrationSupportEncoder(hidden=32)
        profile = torch.randn(4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 14)
        token = encoder(profile)
        self.assertEqual(token.shape, (4, 32))
        self.assertTrue(torch.isfinite(token).all())

    def test_zero_initialized_adapter_exactly_preserves_base(self):
        base = FrozenFeatureStub()
        model = CalibrationAwareV14(base, hidden=32)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        profile = torch.randn(C.N_LINKS, C.N_LIVE_SUBCARRIERS, 14)
        expected = base(csi, mask)
        output = model(csi, mask, profile)
        for key in ("pose_rel", "root", "class_logits", "risk_logits"):
            self.assertTrue(torch.equal(output[key], expected[key]))

    def test_frozen_backbone_receives_no_gradient(self):
        base = FrozenFeatureStub()
        model = CalibrationAwareV14(base, hidden=32)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        profile = torch.randn(C.N_LINKS, C.N_LIVE_SUBCARRIERS, 14)
        output = model(csi, mask, profile)
        output["calibration_rotation_delta"].square().mean().backward()
        self.assertIsNone(base.anchor.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.rotation_head.parameters()
        ))

    def test_pre_encoder_conditioner_has_exact_identity_fallback(self):
        base = PoseNet(hidden=32, dilations=(1, 2), fusion="concat", film=True)
        base.eval()
        model = SupportConditionedP2(base, support_hidden=32).eval()
        model.set_strength(0.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        profile = torch.randn(C.N_LINKS, C.N_LIVE_SUBCARRIERS, 14)
        with torch.no_grad():
            expected = base(csi, mask)
            output = model(csi, mask, profile)
        for key in ("pose_rel", "root", "class_logits", "risk_logits"):
            self.assertTrue(torch.equal(output[key], expected[key]))

    def test_moment_alignment_has_exact_identity_fallback(self):
        base = PoseNet(hidden=32, dilations=(1, 2), fusion="concat", film=True)
        base.eval()
        model = MomentAlignedSupportConditionedP2(
            base, support_hidden=32
        ).eval()
        model.set_alignment_strength(0.0)
        model.set_strength(0.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        profile = torch.randn(C.N_LINKS, C.N_LIVE_SUBCARRIERS, 14)
        with torch.no_grad():
            expected = base(csi, mask)
            output = model(csi, mask, profile)
        for key in ("pose_rel", "root", "class_logits", "risk_logits"):
            self.assertTrue(torch.equal(output[key], expected[key]))

    def test_profile_moments_include_between_pose_variation(self):
        profile = torch.zeros(C.N_LINKS, C.N_LIVE_SUBCARRIERS, 14)
        profile[..., 2:4] = 0.0
        profile[..., 6:8] = 2.0
        profile[..., 10:12] = 4.0
        profile[..., 4:6] = 1.0
        profile[..., 8:10] = 1.0
        profile[..., 12:14] = 1.0
        mean, std = MomentAlignedSupportConditionedP2.profile_moments(profile)
        self.assertTrue(torch.allclose(mean, torch.full_like(mean, 2.0)))
        self.assertTrue(torch.allclose(
            std, torch.full_like(std, (11.0 / 3.0) ** 0.5), atol=1e-6
        ))


if __name__ == "__main__":
    unittest.main()
