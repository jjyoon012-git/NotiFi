from __future__ import annotations

import unittest

import pandas as pd
import torch

from notifi_pose import contract as C
from notifi_pose.doppler_pose import (
    DopplerFilterBank,
    DopplerMotionEncoder,
    DopplerPoseResidual,
    correspondence_loss,
)
from notifi_pose.nets import PerLinkNorm
from notifi_pose.tools.train_doppler_pose import CorrespondenceBatchSampler


def fitted_normalizer() -> PerLinkNorm:
    normalizer = PerLinkNorm()
    normalizer.mu.copy_(torch.randn_like(normalizer.mu))
    normalizer.sigma.copy_(0.5 + torch.rand_like(normalizer.sigma))
    normalizer.fitted.fill_(True)
    return normalizer


class DopplerPoseTests(unittest.TestCase):
    def test_correspondence_sampler_places_same_class_site_pairs_together(self):
        index = pd.DataFrame({
            "subject": ["a", "a", "a", "a"],
            "environment": ["E01", "E01", "E02", "E02"],
            "class_id": [3, 3, 4, 4],
        })
        sampler = CorrespondenceBatchSampler(
            index, torch.ones(4), batch_size=4, seed=9
        )
        batch = next(iter(sampler))
        for left, right in zip(batch[::2], batch[1::2]):
            self.assertNotEqual(left, right)
            self.assertEqual(index.iloc[left].to_dict(), index.iloc[right].to_dict())

    def test_doppler_filter_bank_preserves_frame_and_link_shape(self):
        bank = DopplerFilterBank(channels=8, output=16, dropout=0.0)
        values = torch.randn(2, 67, C.N_LINKS, 8)
        output = bank(values)
        self.assertEqual(output.shape, (2, 67, C.N_LINKS, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_doppler_motion_encoder_rejects_static_link_offsets(self):
        torch.manual_seed(11)
        encoder = DopplerMotionEncoder(
            fitted_normalizer(), hidden=32, temporal_layers=1,
            heads=4, dropout=0.0,
        ).eval()
        csi = torch.randn(1, 67, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 67, C.N_LINKS, dtype=torch.bool)
        offset = torch.randn(1, 1, C.N_LINKS, 1, 2)
        original, original_activity = encoder(csi, mask)
        shifted, shifted_activity = encoder(csi + offset, mask)
        torch.testing.assert_close(original, shifted, atol=4e-5, rtol=4e-5)
        torch.testing.assert_close(
            original_activity, shifted_activity, atol=2e-5, rtol=2e-5
        )

    def test_correspondence_loss_supports_duplicate_sampler_rows(self):
        csi = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        pose = csi.clone()
        rows = torch.tensor([10, 10, 20, 30])
        classes = torch.tensor([2, 2, 2, 2])
        domains = torch.tensor([1, 1, 1, 1])
        loss, metrics = correspondence_loss(csi, pose, rows, classes, domains)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(0.0 <= metrics["retrieval_at_1"] <= 1.0)
        self.assertGreater(metrics["hard_negative_pairs"], 0)

    def test_kp2_compact_checkpoint_contains_cross_modal_heads(self):
        model = DopplerPoseResidual(
            None, fitted_normalizer(), hidden=32, temporal_layers=1,
            heads=4, dropout=0.0,
        )
        state = model.trainable_state_dict()
        self.assertTrue(any(key.startswith("dynamic.doppler.") for key in state))
        self.assertTrue(any(key.startswith("csi_motion_embedding.") for key in state))
        self.assertTrue(any(key.startswith("pose_motion_embedding.") for key in state))
        self.assertFalse(any(key.startswith("baseline.") for key in state))

    def test_constant_csi_keeps_exact_frozen_fallback(self):
        model = DopplerPoseResidual(
            None, fitted_normalizer(), hidden=32, temporal_layers=1,
            heads=4, dropout=0.0, condition_on_coarse=True,
            activity_floor=0.0,
        ).eval()
        csi = torch.randn(1, 1, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        csi = csi.expand(1, 67, -1, -1, -1).clone()
        mask = torch.ones(1, 67, C.N_LINKS, dtype=torch.bool)
        coarse = torch.randn(1, 67, C.N_JOINTS, 3)
        coarse -= coarse[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        output = model(csi, mask, coarse_pose=coarse)
        self.assertEqual(torch.count_nonzero(output["kinetic_activity"]), 0)
        torch.testing.assert_close(output["pose_rel"], coarse, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
