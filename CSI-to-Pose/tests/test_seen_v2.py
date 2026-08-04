import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from notifi_pose import contract as C
from notifi_pose.motion_first import MotionFirstEncoder
from notifi_pose.nets import GraphPoseNet
from notifi_pose.quality import quality_scores
from notifi_pose.seen_v2 import (
    N_INJURY_JOINTS,
    SeenReconstructionV2Net,
    injury_targets,
    weighted_seen_v2_loss,
)


class SeenV2Tests(unittest.TestCase):
    def make_model(self) -> SeenReconstructionV2Net:
        baseline = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", dropout=0.0,
        )
        motion = MotionFirstEncoder(
            hidden=64, temporal_layers=1, heads=4, dropout=0.0
        )
        return SeenReconstructionV2Net(
            baseline, motion, hidden=64, dropout=0.0
        )

    def test_identity_initialization_preserves_pose_and_root(self) -> None:
        model = self.make_model().eval()
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            baseline = model.baseline(csi, mask)
            output = model(csi, mask)
        torch.testing.assert_close(output["pose_rel"], baseline["pose_rel"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output["root"], baseline["root"], atol=1e-5, rtol=1e-5)
        self.assertEqual(output["injury_contact_logits"].shape, (2, 12, N_INJURY_JOINTS))

    def test_rotation_path_preserves_bone_lengths(self) -> None:
        model = self.make_model().eval()
        with torch.no_grad():
            model.rotation_head[-1].bias.normal_(std=0.2)
        csi = torch.randn(1, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 12, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            baseline = model.baseline(csi, mask)["pose_rel"]
            output = model(csi, mask)["pose_low"]
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent < 0:
                continue
            before = torch.linalg.vector_norm(baseline[:, :, child] - baseline[:, :, parent], dim=-1)
            after = torch.linalg.vector_norm(output[:, :, child] - output[:, :, parent], dim=-1)
            torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_partial_finetune_opens_only_last_backbone_blocks(self) -> None:
        model = self.make_model()
        model.set_partial_finetune(True)
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in model.baseline.temporal.transformer.layers[-1].parameters()
        ))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.baseline.encoder.parameters()
        ))

    def test_zero_calibration_preserves_baseline_after_nonzero_heads(self) -> None:
        model = self.make_model().eval()
        with torch.no_grad():
            model.rotation_head[-1].bias.normal_(std=0.2)
            model.high_pose_head[-1].bias.normal_(std=0.2)
            model.root_anchor_head[-1].bias.normal_(std=0.2)
            model.root_step_head[-1].bias.normal_(std=0.2)
        model.set_calibration(0.0, 0.0, 0.0)
        csi = torch.randn(1, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 12, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            baseline = model.baseline(csi, mask)
            output = model(csi, mask)
        torch.testing.assert_close(output["pose_rel"], baseline["pose_rel"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output["root"], baseline["root"], atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            model.set_calibration(1.1, 0.0, 0.0)

    def test_targets_and_loss_are_finite(self) -> None:
        model = self.make_model()
        batch_size, frames = 2, 12
        csi = torch.randn(batch_size, frames, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(batch_size, frames, C.N_LINKS, dtype=torch.bool)
        valid = torch.ones(batch_size, frames, dtype=torch.bool)
        pose = torch.randn(batch_size, frames, C.N_JOINTS, 3) * 0.1
        pose = pose - pose[:, :, :1]
        root = torch.randn(batch_size, frames, 3) * 0.05
        batch = {
            "csi": csi, "link_mask": mask, "pose_rel": pose, "root": root,
            "valid": valid, "class_id": torch.tensor([0, 12]),
            "risk_id": torch.tensor([0, 2]),
            "quality_weight": torch.tensor([1.0, 0.5]),
        }
        output = model(csi, mask)
        targets = injury_targets(pose, root, valid, batch["risk_id"])
        self.assertEqual(targets["injury_contact"].shape, (2, frames, N_INJURY_JOINTS))
        loss, parts = weighted_seen_v2_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn("injury_contact", parts)

    def test_quality_prefers_recorded_timestamp(self) -> None:
        index = pd.DataFrame({
            "trial_id": ["exact", "assumed"],
            "time_method": ["timestamps", "uniform_30fps"],
            "n_alive": [3, 3],
        })
        audit = pd.DataFrame({
            "trial_id": ["exact", "assumed"],
            "mean_gt_speed_mps": [0.3, 0.3],
            "zero_lag_correlation": [0.5, 0.5],
            "status": ["aligned_observable", "aligned_observable"],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            audit.to_csv(path, index=False)
            scores = quality_scores(index, path)
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(np.all((scores >= 0.35) & (scores <= 1.0)))


if __name__ == "__main__":
    unittest.main()
