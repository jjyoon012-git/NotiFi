import unittest

import torch

from notifi_pose.doppler_pose import DopplerMotionEncoder
from notifi_pose.external_pretraining import (
    MultiSourceCSIPretrainer,
    UniversalDopplerEncoder,
    transplant_external_encoder,
)
from notifi_pose.nets import PerLinkNorm


class UniversalDopplerEncoderTests(unittest.TestCase):
    def test_variable_link_counts_share_one_encoder(self) -> None:
        model = UniversalDopplerEncoder(
            hidden=32, temporal_layers=1, heads=4, dropout=0.0
        ).eval()
        for links in (3, 4, 9):
            values = torch.randn(2, 20, links, 32, 2)
            mask = torch.ones(2, 20, links, dtype=torch.bool)
            with torch.no_grad():
                encoded, frame_mask = model(values, mask)
            self.assertEqual(encoded.shape, (2, 20, 32))
            self.assertEqual(frame_mask.shape, (2, 20))
            self.assertTrue(torch.isfinite(encoded).all())

    def test_multisource_heads_are_not_shared(self) -> None:
        model = MultiSourceCSIPretrainer(
            {"mmfi": 27, "csi_bench_fall": 2},
            hidden=32, temporal_layers=1, heads=4, motion_dim=16, dropout=0.0,
        ).eval()
        values = torch.randn(2, 20, 3, 32, 2)
        mask = torch.ones(2, 20, 3, dtype=torch.bool)
        with torch.no_grad():
            mmfi = model(values, mask, "mmfi")
            fall = model(values, mask, "csi_bench_fall")
        self.assertEqual(mmfi["class_logits"].shape, (2, 27))
        self.assertEqual(fall["class_logits"].shape, (2, 2))
        self.assertEqual(mmfi["impact_logits"].shape, (2, 20))
        self.assertEqual(mmfi["motion_embedding"].shape, (2, 20, 16))

    def test_transfer_preserves_phase_input_channels(self) -> None:
        source = MultiSourceCSIPretrainer(
            {"mmfi": 27}, hidden=32, temporal_layers=1,
            heads=4, dropout=0.0,
        )
        with torch.no_grad():
            source.encoder.subcarrier[0].weight.fill_(0.75)
        target = DopplerMotionEncoder(
            PerLinkNorm(n_links=3, n_sc=114), hidden=32,
            temporal_layers=1, heads=4, dropout=0.0,
        )
        before = target.doppler.subcarrier[0].weight.detach().clone()
        report = transplant_external_encoder(target, source.shared_checkpoint())
        after = target.doppler.subcarrier[0].weight.detach()
        self.assertTrue(torch.allclose(after[:, 0], torch.full_like(after[:, 0], 0.75)))
        self.assertTrue(torch.allclose(after[:, 2], torch.full_like(after[:, 2], 0.75)))
        self.assertTrue(torch.equal(after[:, 1], before[:, 1]))
        self.assertTrue(torch.equal(after[:, 3], before[:, 3]))
        self.assertEqual(report["first_conv_amplitude_channels"], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_transfer_accepts_cpu_checkpoint_for_cuda_target(self) -> None:
        source = MultiSourceCSIPretrainer({"mmfi": 27}, hidden=32)
        target = DopplerMotionEncoder(
            PerLinkNorm(n_links=3, n_sc=114), hidden=32,
            temporal_layers=2, heads=4,
        ).cuda()
        report = transplant_external_encoder(target, source.shared_checkpoint())
        self.assertGreater(report["tensors_copied"], 0)


if __name__ == "__main__":
    unittest.main()
