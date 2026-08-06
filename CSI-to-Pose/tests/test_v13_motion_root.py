import unittest

import torch
from torch import nn

from notifi_pose.hybrid_v10 import P2V13MotionRootHybridNet
from notifi_pose.tools.train_p2_v9_hybrid import root_only_reconstruction_loss
from notifi_pose.tools.train_p2_v9_hybrid import (
    _motion_regression_metrics,
    motion_observation_per_sample,
    motion_only_loss,
)


class _Base(nn.Module):
    hidden = 96


class V13MotionRootTests(unittest.TestCase):
    def test_auxiliary_head_does_not_change_initial_root(self):
        model = P2V13MotionRootHybridNet(_Base())
        feature = torch.randn(2, 12, 128)
        root = torch.randn(2, 12, 3)
        valid = torch.ones(2, 12, dtype=torch.bool)
        adjusted, auxiliary = model.root_candidate(
            feature, root, valid, feature.mean(1)
        )
        self.assertTrue(torch.allclose(adjusted, root, atol=1e-7))
        self.assertEqual(auxiliary["root_velocity_observation_v13"].shape, root.shape)

    def test_motion_auxiliary_loss_is_finite(self):
        frames = 8
        output = {
            "root": torch.zeros(2, frames, 3),
            "root_velocity_observation_v13": torch.zeros(2, frames, 3),
            "pose_speed_observation_v13": torch.zeros(2, frames),
        }
        batch = {
            "root": torch.randn(2, frames, 3) * 0.01,
            "pose_rel": torch.randn(2, frames, 22, 3) * 0.01,
            "valid": torch.ones(2, frames, dtype=torch.bool),
            "risk_id": torch.tensor([0, 2]),
        }
        loss, parts = root_only_reconstruction_loss(
            output, batch, velocity_weight=0.0, displacement_weight=0.0,
            endpoint_weight=0.0, velocity_lags=(1,), motion_aux_weight=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(parts["motion_aux"], 0.0)

    def test_perfect_motion_observation_has_zero_loss(self):
        frames = 10
        lag = 2
        root = torch.randn(2, frames, 3) * 0.02
        pose = torch.randn(2, frames, 22, 3) * 0.02
        scale = 30.0 / lag
        root_velocity = torch.zeros_like(root)
        root_velocity[:, lag:] = (root[:, lag:] - root[:, :-lag]) * scale
        pose_speed = torch.zeros(2, frames)
        pose_speed[:, lag:] = torch.linalg.vector_norm(
            (pose[:, lag:] - pose[:, :-lag]) * scale, dim=-1
        ).mean(-1)
        output = {
            "root_velocity_observation_v13": root_velocity,
            "pose_speed_observation_v13": pose_speed,
        }
        batch = {
            "root": root,
            "pose_rel": pose,
            "valid": torch.ones(2, frames, dtype=torch.bool),
        }
        per_sample, _ = motion_observation_per_sample(output, batch, lag)
        loss, parts = motion_only_loss(output, batch, lag)
        self.assertTrue(torch.allclose(per_sample, torch.zeros_like(per_sample)))
        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertAlmostEqual(parts["root_motion"], 0.0, places=7)

    def test_motion_regression_metrics_reports_perfect_fit(self):
        target = torch.randn(32, 3)
        metrics = _motion_regression_metrics(target, target)
        self.assertAlmostEqual(metrics["r2_mean"], 1.0, places=7)
        self.assertAlmostEqual(metrics["correlation_mean"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
