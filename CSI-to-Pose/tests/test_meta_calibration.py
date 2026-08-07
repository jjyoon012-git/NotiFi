import unittest

import torch

from notifi_pose.meta_calibration import (
    LinkAwareCSIEncoder,
    MOTION_PROMPT_CLASSES,
    RawSupportConditionedModel,
    SupportBaselineCanonicalizer,
    SupportConditionedCalibrator,
    masked_moments,
)


class MetaCalibrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = SupportConditionedCalibrator(
            feature_dim=8, token_dim=16, width=32, heads=4, layers=1, domains=3
        )
        self.model.eval()
        self.support = torch.randn(8, 12, 8)
        self.support_mask = torch.ones(8, 12, dtype=torch.bool)
        self.support_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.query = torch.randn(5, 12, 8)
        self.query_mask = torch.ones(5, 12, dtype=torch.bool)

    def test_output_contract(self):
        output = self.model(
            self.query, self.query_mask,
            self.support, self.support_mask, self.support_labels,
        )
        self.assertEqual(output["calibrated_features"].shape, self.query.shape)
        self.assertEqual(output["action_logits"].shape, (5, 17))
        self.assertEqual(output["risk_logits"].shape, (5, 3))
        self.assertEqual(output["support_attention"].shape, (5, 4))

    def test_support_order_is_irrelevant(self):
        permutation = torch.tensor([7, 0, 5, 2, 1, 6, 3, 4])
        first = self.model(
            self.query, self.query_mask,
            self.support, self.support_mask, self.support_labels,
        )
        second = self.model(
            self.query, self.query_mask,
            self.support[permutation], self.support_mask[permutation],
            self.support_labels[permutation],
        )
        self.assertTrue(torch.allclose(
            first["calibrated_features"], second["calibrated_features"], atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            first["action_logits"], second["action_logits"], atol=1e-6
        ))

    def test_padding_does_not_change_moments(self):
        padded = torch.cat((self.query, torch.full((5, 3, 8), 99.0)), dim=1)
        padded_mask = torch.cat((
            self.query_mask, torch.zeros(5, 3, dtype=torch.bool)
        ), dim=1)
        first = masked_moments(self.query, self.query_mask)
        second = masked_moments(padded, padded_mask)
        for left, right in zip(first, second):
            self.assertTrue(torch.allclose(left, right, atol=1e-6))

    def test_missing_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing calibration prompt"):
            self.model.encode_support(
                self.support[:-2], self.support_mask[:-2], self.support_labels[:-2]
            )

    def test_raw_encoder_contract(self):
        encoder = LinkAwareCSIEncoder(hidden=16, temporal_layers=2, dropout=0.0)
        csi = torch.randn(2, 20, 3, 12, 2)
        mask = torch.ones(2, 20, 3, dtype=torch.bool)
        mask[0, :, 1] = False
        output, frame_mask, link_weight = encoder(csi, mask)
        self.assertEqual(output.shape, (2, 20, 16))
        self.assertEqual(frame_mask.shape, (2, 20))
        self.assertEqual(link_weight.shape, (2, 20, 3))
        self.assertTrue(torch.allclose(link_weight[0, :, 1], torch.zeros(20)))
        self.assertTrue(torch.allclose(link_weight.sum(-1), torch.ones(2, 20)))

    def test_raw_support_model_contract(self):
        model = RawSupportConditionedModel(
            hidden=16, token_dim=16, width=32, domains=3, dropout=0.0,
            prompt_classes=MOTION_PROMPT_CLASSES,
        )
        query = torch.randn(3, 20, 3, 12, 2)
        support = torch.randn(12, 20, 3, 12, 2)
        absence = torch.randn(2, 20, 3, 12, 2)
        query[..., 0].abs_().add_(1.0)
        support[..., 0].abs_().add_(1.0)
        absence[..., 0].abs_().add_(1.0)
        query_mask = torch.ones(3, 20, 3, dtype=torch.bool)
        support_mask = torch.ones(12, 20, 3, dtype=torch.bool)
        absence_mask = torch.ones(2, 20, 3, dtype=torch.bool)
        raw_support_labels = torch.tensor([
            0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 7, 8,
        ])
        output = model(
            query, query_mask, support, support_mask, raw_support_labels,
            absence, absence_mask,
        )
        self.assertEqual(output["action_logits"].shape, (3, 17))
        self.assertEqual(output["risk_logits"].shape, (3, 3))
        self.assertEqual(output["query_link_weight"].shape, (3, 20, 3))
        self.assertEqual(output["baseline_valid_links"].shape, (3,))

    def test_baseline_canonicalization_removes_link_gain_and_phase_offset(self):
        canonicalizer = SupportBaselineCanonicalizer()
        absence = torch.randn(2, 20, 3, 12, 2)
        query = torch.randn(3, 20, 3, 12, 2)
        absence[..., 0].abs_().add_(20.0)
        query[..., 0].mul_(0.2).add_(absence[..., 0].mean((0, 1)))
        mask = torch.ones(3, 20, 3, dtype=torch.bool)
        absence_mask = torch.ones(2, 20, 3, dtype=torch.bool)
        first, _, _ = canonicalizer(query, mask, absence, absence_mask)

        gain = torch.tensor([0.7, 1.8, 1.2])[None, None, :, None]
        phase_offset = torch.tensor([0.8, -1.1, 0.3])[None, None, :, None]
        shifted_query = query.clone()
        shifted_absence = absence.clone()
        shifted_query[..., 0] *= gain
        shifted_absence[..., 0] *= gain
        shifted_query[..., 1] += phase_offset
        shifted_absence[..., 1] += phase_offset
        second, _, _ = canonicalizer(
            shifted_query, mask, shifted_absence, absence_mask
        )
        self.assertTrue(torch.allclose(first, second, atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
    unittest.main()
