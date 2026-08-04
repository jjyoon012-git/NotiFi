import unittest

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.hybrid_v10 import (
    P2V9HybridNet,
    RootExpertBlend,
    p2_motion_features,
)
from notifi_pose.tools.train_p2_v9_hybrid import (
    pose_selection_score,
    root_selection_score,
)


class DummyP2(nn.Module):
    hidden = 8

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        pose = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
        pose[:, :, C.ROOT_JOINT, 0] = 0.02
        for joint in range(1, C.N_JOINTS):
            pose[:, :, joint, 1] = joint * 0.01
        return {
            "pose_rel": pose,
            "root": csi.new_zeros(batch, frames, 3),
            "class_logits": csi.new_zeros(batch, C.N_CLASSES),
            "risk_logits": csi.new_zeros(batch, C.N_RISK),
            "temporal_features": csi.new_zeros(batch, frames, self.hidden),
        }


class HybridV10Test(unittest.TestCase):
    def test_zero_calibration_recovers_p2(self):
        model = P2V9HybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        model.set_calibration(0.0, 0.0, 0.0, 0.0)
        csi = torch.ones(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.equal(output["pose_rel"], output["pose_p2"]))
        self.assertTrue(torch.equal(output["root"], output["root_p2"]))
        self.assertTrue(torch.equal(output["class_logits"], output["class_logits_p2"]))
        self.assertTrue(torch.equal(output["risk_logits"], output["risk_logits_p2"]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.base.parameters()))

    def test_output_shapes(self):
        model = P2V9HybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        csi = torch.ones(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["pose_rel"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertEqual(output["root"].shape, (2, 12, 3))
        self.assertEqual(output["class_logits"].shape, (2, C.N_CLASSES))
        self.assertEqual(output["risk_logits"].shape, (2, C.N_RISK))

    def test_motion_features_are_finite_and_masked(self):
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        mask[0, 4] = False
        feature = p2_motion_features(csi, mask)
        self.assertEqual(feature.shape, (2, 12, 4))
        self.assertTrue(torch.isfinite(feature).all())
        self.assertTrue(torch.equal(feature[0, 4], torch.zeros(4)))

    def test_pose_and_root_selection_are_decoupled(self):
        metrics = {
            "mpjpe_m": 0.15,
            "danger_mpjpe_m": 0.45,
            "danger_endpoint_mpjpe_m": 0.60,
            "pose_speed_ratio": 1.0,
            "root_error_m": 0.35,
            "danger_root_error_m": 0.40,
            "danger_root_drop_mae_m": 0.10,
        }
        pose_score = pose_selection_score(metrics)
        metrics["root_error_m"] = 9.0
        self.assertEqual(pose_selection_score(metrics), pose_score)
        self.assertGreater(root_selection_score(metrics), pose_score)

    def test_root_expert_blend_changes_only_root(self):
        class ConstantModel(nn.Module):
            def __init__(self, root_value):
                super().__init__()
                self.root_value = root_value

            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                return {
                    "pose_rel": csi.new_ones(batch, frames, C.N_JOINTS, 3),
                    "root": csi.new_full((batch, frames, 3), self.root_value),
                    "class_logits": csi.new_zeros(batch, C.N_CLASSES),
                    "risk_logits": csi.new_zeros(batch, C.N_RISK),
                }

        model = RootExpertBlend(ConstantModel(1.0), ConstantModel(3.0))
        model.set_root_strength(0.25)
        csi = torch.ones(1, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 4, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.equal(output["pose_rel"], torch.ones_like(output["pose_rel"])))
        self.assertTrue(torch.allclose(output["root"], torch.full_like(output["root"], 1.5)))


if __name__ == "__main__":
    unittest.main()
