import unittest

import torch
from torch.optim.swa_utils import AveragedModel

from notifi_pose import contract as C
from notifi_pose import losses as L
from notifi_pose.nets import GraphPoseNet
from notifi_pose.tools.evaluate_sealed import smooth_valid


class GraphFormerTests(unittest.TestCase):
    def test_shapes_and_kinematic_root(self) -> None:
        model = GraphPoseNet(hidden=64, n_blocks=1, heads=4, graph_blocks=1, dropout=0.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        mask[0, -2:] = False
        output = model(csi, mask)

        self.assertEqual(output["pose_rel"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertEqual(output["root"].shape, (2, 12, 3))
        self.assertEqual(output["motion"].shape, (2, 12))
        self.assertEqual(output["class_logits"].shape, (2, C.N_CLASSES))
        self.assertTrue(torch.isfinite(output["pose_rel"]).all())
        torch.testing.assert_close(
            output["pose_rel"][:, :, C.ROOT_JOINT],
            torch.zeros(2, 12, 3),
        )

    def test_missing_links_do_not_create_nan(self) -> None:
        model = GraphPoseNet(hidden=64, n_blocks=1, heads=4, graph_blocks=1, dropout=0.0)
        csi = torch.randn(2, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 8, C.N_LINKS, dtype=torch.bool)
        mask[0] = False
        mask[1, :, :2] = False
        output = model(csi, mask)
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_hybrid_decoder_keeps_pelvis_at_origin(self) -> None:
        model = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", dropout=0.0,
        )
        csi = torch.randn(1, 6, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 6, C.N_LINKS, dtype=torch.bool)
        pose = model(csi, mask)["pose_rel"]
        torch.testing.assert_close(
            pose[:, :, C.ROOT_JOINT], torch.zeros(1, 6, 3)
        )

    def test_robust_heads_have_domain_and_fall_outputs(self) -> None:
        model = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", robust_heads=True, dropout=0.0,
        )
        csi = torch.randn(2, 6, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 6, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["phase_logits"].shape, (2, 6, 4))
        self.assertEqual(output["contact_logits"].shape, (2, 6, 4))
        self.assertEqual(output["domain_logits"].shape, (2, 9))
        self.assertEqual(output["embedding"].shape, (2, 64))

    def test_temporal_refiner_is_identity_at_initialization(self) -> None:
        model = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            decoder="hybrid", robust_heads=True, temporal_refiner=True,
            dropout=0.0,
        )
        csi = torch.randn(2, 9, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 9, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["pose_coarse"].shape, output["pose_rel"].shape)
        torch.testing.assert_close(output["pose_rel"], output["pose_coarse"])
        torch.testing.assert_close(
            output["pose_rel"][:, :, C.ROOT_JOINT], torch.zeros(2, 9, 3)
        )

    def test_impact_window_is_danger_only(self) -> None:
        pose = torch.zeros(2, 12, C.N_JOINTS, 3)
        root = torch.zeros(2, 12, 3)
        valid = torch.ones(2, 12, dtype=torch.bool)
        pose[:, 6:, C.JOINT_INDEX["head"], 1] = -0.8
        risk = torch.tensor([2, 0])
        selected = L.impact_window(pose, root, valid, risk, radius=2)
        self.assertTrue(selected[0].any())
        self.assertTrue(bool(selected[0, 6]))
        self.assertFalse(selected[1].any())

    def test_parameter_average_copies_boolean_norm_buffers(self) -> None:
        model = GraphPoseNet(
            hidden=64, n_blocks=1, heads=4, graph_blocks=1,
            robust_heads=True, dropout=0.0,
        )
        model.norm.fitted.fill_(True)
        averaged = AveragedModel(model, use_buffers=False)
        averaged.update_parameters(model)
        averaged.update_parameters(model)
        self.assertTrue(bool(averaged.module.norm.fitted.item()))

    def test_smoothing_preserves_shape_and_padding(self) -> None:
        values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        valid = torch.tensor([[True, True, True, True, True, False, False, False]])
        filtered = smooth_valid(values, valid, 3)
        self.assertEqual(filtered.shape, values.shape)
        torch.testing.assert_close(filtered[:, 5:], values[:, 5:])
        self.assertLess(float(filtered[0, 4]), float(values[0, 4]))


if __name__ == "__main__":
    unittest.main()
