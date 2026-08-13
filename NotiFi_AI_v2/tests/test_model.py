import unittest

import torch

from notifi_ai_v2.model import MotionCalibratedEncoder, MotionEncoderConfig


class ModelTests(unittest.TestCase):
    def test_model_output_contract_and_link_masking(self):
        torch.manual_seed(13)
        csi = torch.randn(2, 40, 3, 114, 2)
        mask = torch.ones(2, 40, 3, dtype=torch.bool)
        mask[0, 8:12] = False
        mask[1, :, 2] = False
        model = MotionCalibratedEncoder(
            MotionEncoderConfig(hidden=48, temporal_layers=2, motion_targets=8)
        ).eval()
        with torch.no_grad():
            output = model(csi, mask)
        self.assertEqual(output["action_logits"].shape, (2, 17))
        self.assertEqual(output["risk_logits"].shape, (2, 3))
        self.assertEqual(output["motion"].shape, (2, 40, 8))
        self.assertEqual(output["link_weight"].shape, (2, 40, 3))
        self.assertEqual(torch.count_nonzero(output["motion"][0, 8:12]), 0)
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_shared_link_encoder_has_no_link_specific_parameters(self):
        model = MotionCalibratedEncoder(MotionEncoderConfig(hidden=48))
        names = [name for name, _ in model.link_encoder.named_parameters()]
        self.assertTrue(names)
        self.assertFalse(any(
            "link_0" in name or "link_1" in name or "link_2" in name
            for name in names
        ))


if __name__ == "__main__":
    unittest.main()
