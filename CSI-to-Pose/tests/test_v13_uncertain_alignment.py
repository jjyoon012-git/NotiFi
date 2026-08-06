import unittest

import torch

from notifi_pose.tools.train_p2_v9_hybrid import (
    global_shift_root_soft_loss,
    root_only_reconstruction_loss,
)


class V13UncertainAlignmentTests(unittest.TestCase):
    def test_soft_shift_matches_exact_loss_at_zero_radius(self):
        target = torch.randn(2, 8, 3)
        valid = torch.ones(2, 8, dtype=torch.bool)
        loss = global_shift_root_soft_loss(
            target, target, valid, max_shift=0, temperature=0.01
        )
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-7))

    def test_uncertain_shift_reduces_shifted_sequence_loss(self):
        target = torch.zeros(1, 6, 3)
        target[0, :, 0] = torch.arange(6, dtype=torch.float32)
        predicted = torch.roll(target, shifts=-1, dims=1)
        output = {"root": predicted}
        batch = {
            "root": target,
            "valid": torch.ones(1, 6, dtype=torch.bool),
            "risk_id": torch.zeros(1, dtype=torch.long),
            "timestamp_exact": torch.zeros(1, dtype=torch.bool),
        }
        plain, _ = root_only_reconstruction_loss(
            output, batch, velocity_weight=0.0, displacement_weight=0.0,
            endpoint_weight=0.0, velocity_lags=(1,), max_shift=1,
        )
        aligned, _ = root_only_reconstruction_loss(
            output, batch, velocity_weight=0.0, displacement_weight=0.0,
            endpoint_weight=0.0, velocity_lags=(1,), max_shift=1,
            uncertain_shift_weight=1.0, uncertain_shift_temperature=1e-4,
        )
        self.assertLess(float(aligned), float(plain))


if __name__ == "__main__":
    unittest.main()
