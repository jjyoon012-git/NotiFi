import math
import unittest

import torch

from notifi_ai_v2.frontend import PhysicsMotionFrontend


def make_input(batch: int = 2, frames: int = 32):
    torch.manual_seed(11)
    csi = torch.randn(batch, frames, 3, 114, 2)
    mask = torch.ones(batch, frames, 3, dtype=torch.bool)
    return csi, mask


class FrontendTests(unittest.TestCase):
    def test_frontend_is_invariant_to_constant_link_gain(self):
        csi, mask = make_input()
        gain = torch.tensor([2.0, 0.5, 4.0])[None, None, :, None, None]
        frontend = PhysicsMotionFrontend()
        first = frontend(csi, mask).features[..., :4]
        second = frontend(csi * gain, mask).features[..., :4]
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))

    def test_differential_phase_rejects_constant_link_rotation(self):
        csi, mask = make_input()
        angle = torch.tensor([0.7, -1.2, 2.4])[None, None, :, None]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated = csi.clone()
        real, imag = csi[..., 0], csi[..., 1]
        rotated[..., 0] = real * cosine - imag * sine
        rotated[..., 1] = real * sine + imag * cosine
        frontend = PhysicsMotionFrontend()
        first = frontend(csi, mask).features[..., 4:]
        second = frontend(rotated, mask).features[..., 4:]
        self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5))

    def test_missing_links_are_zero_and_finite(self):
        csi, mask = make_input()
        mask[:, :, 1] = False
        output = PhysicsMotionFrontend()(csi, mask)
        self.assertEqual(torch.count_nonzero(output.features[:, :, 1]), 0)
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.isfinite(output.phase_quality).all())

    def test_frontend_detects_temporal_motion(self):
        frames = 32
        time = torch.linspace(0, 2 * math.pi, frames)
        csi = torch.ones(1, frames, 3, 114, 2)
        csi[..., 1] = 0.0
        motion = (1.0 + 0.5 * torch.sin(time)).view(1, frames, 1)
        csi[:, :, 0, :, 0] *= motion
        mask = torch.ones(1, frames, 3, dtype=torch.bool)
        output = PhysicsMotionFrontend()(csi, mask)
        self.assertGreater(output.activity[0].max(), 0.01)
        self.assertGreater(output.activity[0].sum(), 0.1)


if __name__ == "__main__":
    unittest.main()
