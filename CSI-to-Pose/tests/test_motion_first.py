import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.motion_first import (
    ActionMotionResidualPoseNet,
    KeyframeRootResidualNet,
    MotionFirstEncoder,
    MotionFirstPoseNet,
    MotionResidualPoseNet,
    masked_temporal_average,
    temporal_keyframes,
    temporal_difference,
)


class MotionFirstTests(unittest.TestCase):
    def test_temporal_difference_respects_link_pairs(self) -> None:
        values = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
        mask = torch.tensor([[[True], [True], [False], [True]]])
        difference, pair = temporal_difference(values, mask)
        torch.testing.assert_close(
            difference.flatten(), torch.tensor([0.0, 1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            pair.flatten(), torch.tensor([False, True, False, False])
        )

    def test_output_shapes(self) -> None:
        model = MotionFirstEncoder(hidden=64, temporal_layers=1, heads=4, dropout=0.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["speed_log"].shape, (2, 12))
        self.assertEqual(output["moving_logits"].shape, (2, 12))
        self.assertEqual(output["phase_logits"].shape, (2, 12, 4))
        self.assertEqual(output["impact_logits"].shape, (2, 12))
        self.assertEqual(output["class_logits"].shape, (2, C.N_CLASSES))
        self.assertEqual(output["risk_logits"].shape, (2, C.N_RISK))
        self.assertEqual(output["temporal_features"].shape, (2, 12, 64))

    def test_pose_model_preserves_motion_heads(self) -> None:
        model = MotionFirstPoseNet(
            hidden=64, temporal_layers=1, heads=4, graph_blocks=1, dropout=0.0
        )
        csi = torch.randn(1, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 8, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["pose_rel"].shape, (1, 8, C.N_JOINTS, 3))
        self.assertEqual(output["root"].shape, (1, 8, 3))
        self.assertEqual(output["motion"].shape, (1, 8))
        torch.testing.assert_close(
            output["pose_rel"][:, :, C.ROOT_JOINT], torch.zeros(1, 8, 3)
        )

    def test_temporal_average_does_not_use_padding(self) -> None:
        values = torch.tensor([[[1.0], [3.0], [0.0], [0.0]]])
        mask = torch.tensor([[True, True, False, False]])
        averaged = masked_temporal_average(values, mask, width=3)
        torch.testing.assert_close(
            averaged, torch.tensor([[[2.0], [2.0], [0.0], [0.0]]])
        )

    def test_keyframes_keep_partial_valid_window(self) -> None:
        values = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        mask = torch.tensor([[True, True, True, True, True]])
        keyframes, key_mask = temporal_keyframes(values, mask, stride=4)
        torch.testing.assert_close(keyframes.flatten(), torch.tensor([1.5, 4.0]))
        self.assertTrue(key_mask.all())

    def test_zero_residual_preserves_baseline(self) -> None:
        from notifi_pose.nets import GraphPoseNet

        baseline = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", dropout=0.0,
        )
        motion = MotionFirstEncoder(
            hidden=64, temporal_layers=1, heads=4, dropout=0.0
        )
        model = MotionResidualPoseNet(baseline, motion, hidden=64, dropout=0.0)
        csi = torch.randn(1, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 8, C.N_LINKS, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            expected = baseline(csi, mask)
            actual = model(csi, mask)
        torch.testing.assert_close(actual["pose_rel"], expected["pose_rel"])
        torch.testing.assert_close(actual["root"], expected["root"])

        action_model = ActionMotionResidualPoseNet(
            baseline, motion, hidden=64, dropout=0.0
        )
        action_model.eval()
        with torch.no_grad():
            action_output = action_model(csi, mask)
        torch.testing.assert_close(action_output["pose_rel"], expected["pose_rel"])
        torch.testing.assert_close(action_output["root"], expected["root"])

    def test_action_residual_scale_zero_preserves_baseline(self) -> None:
        from notifi_pose.nets import GraphPoseNet

        baseline = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", dropout=0.0,
        )
        motion = MotionFirstEncoder(
            hidden=64, temporal_layers=1, heads=4, dropout=0.0
        )
        model = ActionMotionResidualPoseNet(
            baseline, motion, hidden=64, dropout=0.0
        )
        with torch.no_grad():
            model.pose_refiner.head[-1].bias.fill_(1.0)
        model.set_residual_scale(0.0)
        model.eval()
        csi = torch.randn(1, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 8, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            expected = baseline(csi, mask)
            actual = model(csi, mask)
        torch.testing.assert_close(actual["pose_rel"], expected["pose_rel"])

        with self.assertRaises(ValueError):
            model.set_residual_scale(1.1)

    def test_zero_root_residual_preserves_pose_model(self) -> None:
        from notifi_pose.nets import GraphPoseNet

        baseline = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", dropout=0.0,
        )
        motion = MotionFirstEncoder(
            hidden=64, temporal_layers=1, heads=4, dropout=0.0
        )
        pose_model = ActionMotionResidualPoseNet(
            baseline, motion, hidden=64, dropout=0.0
        )
        model = KeyframeRootResidualNet(
            pose_model, hidden=64, dropout=0.0
        )
        model.eval()
        csi = torch.randn(1, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 8, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            expected = pose_model(csi, mask)
            actual = model(csi, mask)
        torch.testing.assert_close(actual["pose_rel"], expected["pose_rel"])
        torch.testing.assert_close(actual["root"], expected["root"])


if __name__ == "__main__":
    unittest.main()
