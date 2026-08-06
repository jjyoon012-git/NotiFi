import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from notifi_pose import contract as C
from notifi_pose.hybrid_v10 import (
    AnatomicalResidualCalibration,
    ClassificationExpertBlend,
    ConditionalLinkFailureLogitBlend,
    ConditionalLinkFailurePoseBlend,
    ConditionalLinkFailureRootBlend,
    HierarchicalRiskCalibration,
    InputMomentCalibration,
    P2V9HybridNet,
    P2V11GraphHybridNet,
    P2V11CartesianHybridNet,
    P2V11DirectRootHybridNet,
    P2V11SpectralHybridNet,
    P2V11SubcarrierHybridNet,
    PoseModelEnsemble,
    SharedBackboneCache,
    SharedBackboneExecution,
    RootExpertBlend,
    RootComponentBlend,
    RiskAdaptivePoseBlend,
    p2_motion_features,
    sequence_bone_projection,
)
from notifi_pose.seen_v2 import _local_bones
from notifi_pose.tools.train_p2_v9_hybrid import (
    global_shift_pose_loss,
    global_shift_root_loss,
    pose_selection_score,
    root_only_reconstruction_loss,
    root_selection_score,
)
from notifi_pose.tools.calibrate_v11_residual_temporal import (
    ResidualTemporalCalibration,
)
from notifi_pose.tools.evaluate_v11_final import _similarity_aligned_mpjpe
from notifi_pose.tools.make_v12_model_soup import average_state_dicts
from notifi_pose.tools.audit_v11_input_robustness import PerturbedDataset
from notifi_pose.tools.calibrate_v12_root_ensemble import simplex_weights


class DummyP2(nn.Module):
    hidden = 8

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        pose = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
        pose[:, :, C.ROOT_JOINT, 0] = 0.02
        for joint in range(1, C.N_JOINTS):
            pose[:, :, joint, 1] = joint * 0.01
        return {
            "pose_rel": pose,
            "root": csi.new_zeros(batch, frames, 3),
            "class_logits": csi.new_zeros(batch, C.N_CLASSES),
            "risk_logits": csi.new_zeros(batch, C.N_RISK),
            "temporal_features": csi.new_zeros(batch, frames, self.hidden),
        }


class SingleCSIDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        csi = torch.zeros(4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        csi[..., 0] = 2.0
        return {
            "csi": csi,
            "link_mask": torch.ones(4, C.N_LINKS, dtype=torch.bool),
        }


class HybridV10Test(unittest.TestCase):
    def test_simplex_root_weights_are_complete_and_normalized(self):
        weights = simplex_weights(3, 0.5)
        self.assertEqual(len(weights), 6)
        self.assertIn([1.0, 0.0, 0.0], weights)
        self.assertIn([0.0, 0.5, 0.5], weights)
        self.assertTrue(all(abs(sum(value) - 1.0) < 1e-8 for value in weights))

    def test_shared_backbone_runs_once_per_top_level_forward(self):
        class CountedBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, csi, link_mask):
                self.calls += 1
                return {"value": csi.sum()}

        class TwoExperts(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, csi, link_mask):
                first = self.backbone(csi, link_mask)["value"]
                second = self.backbone(csi, link_mask)["value"]
                return {"value": first + second}

        counted = CountedBackbone()
        shared = SharedBackboneCache(counted)
        model = SharedBackboneExecution(TwoExperts(shared), shared)
        csi = torch.ones(1, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 4, C.N_LINKS, dtype=torch.bool)
        self.assertEqual(float(model(csi, mask)["value"]), 2.0 * csi.numel())
        self.assertEqual(counted.calls, 1)
        model(csi, mask)
        self.assertEqual(counted.calls, 2)

    def test_gain_phase_audit_respects_amp_phase_representation(self):
        sample = PerturbedDataset(SingleCSIDataset(), "gain_phase")[0]
        amplitude = sample["csi"][..., 0]
        phase = sample["csi"][..., 1]
        self.assertTrue((amplitude > 0).all())
        self.assertTrue(torch.allclose(
            phase.mean(-1), torch.zeros_like(phase.mean(-1)), atol=1e-6
        ))

    def test_burst_link_audit_moves_the_same_half_length_outage(self):
        expected = {
            "drop_link_burst_early": (0, 2),
            "drop_link_burst": (1, 3),
            "drop_link_burst_late": (2, 4),
            "drop_link_burst_shifted": (0, 2),
        }
        for mode, (start, stop) in expected.items():
            with self.subTest(mode=mode):
                sample = PerturbedDataset(SingleCSIDataset(), mode)[0]
                missing = ~sample["link_mask"][:, 0]
                target = torch.zeros(4, dtype=torch.bool)
                target[start:stop] = True
                self.assertTrue(torch.equal(missing, target))
                self.assertTrue((sample["csi"][start:stop, 0] == 0).all())

    def test_risk_adaptive_pose_blend_uses_soft_danger_probability(self):
        model = RiskAdaptivePoseBlend(
            DummyP2(), DummyP2(), non_danger_strength=1.0,
            danger_strength=0.4, danger_logit_bias=0.0,
        )
        csi = torch.zeros(2, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 4, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.allclose(
            output["pose_expert_strength"], torch.full((2,), 0.8)
        ))
        model.set_calibration(1.0, 0.4, 2.0, "hard")
        hard = model(csi, mask)
        self.assertTrue(torch.allclose(
            hard["pose_expert_strength"], torch.full((2,), 0.4)
        ))

    def test_link_failure_expert_changes_only_failed_samples(self):
        class ConstantPose(DummyP2):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["pose_rel"].fill_(self.value)
                return output

        model = ConditionalLinkFailurePoseBlend(
            ConstantPose(1.0), ConstantPose(3.0), strength=0.5
        )
        csi = torch.zeros(2, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 4, C.N_LINKS, dtype=torch.bool)
        mask[1, :, 0] = False
        output = model(csi, mask)
        self.assertTrue(torch.all(output["pose_rel"][0] == 1.0))
        self.assertTrue(torch.all(output["pose_rel"][1] == 2.0))
        self.assertTrue(torch.equal(
            output["link_failure_gate"], torch.tensor((False, True))
        ))

    def test_link_failure_logit_blend_preserves_healthy_samples(self):
        class LogitModel(DummyP2):
            def __init__(self, selected):
                super().__init__()
                self.selected = selected

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["class_logits"][:, self.selected] = 4.0
                output["risk_logits"][:, self.selected % C.N_RISK] = 4.0
                return output

        model = ConditionalLinkFailureLogitBlend(
            LogitModel(0), LogitModel(2), 1.0, 1.0
        )
        csi = torch.zeros(2, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 4, C.N_LINKS, dtype=torch.bool)
        mask[1, :, 1] = False
        output = model(csi, mask)
        self.assertEqual(int(output["class_logits"][0].argmax()), 0)
        self.assertEqual(int(output["class_logits"][1].argmax()), 2)
        self.assertEqual(int(output["risk_logits"][0].argmax()), 0)
        self.assertEqual(int(output["risk_logits"][1].argmax()), 2)

    def test_link_failure_logit_blend_supports_link_specific_strengths(self):
        class LogitModel(DummyP2):
            def __init__(self, selected):
                super().__init__()
                self.selected = selected

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["class_logits"][:, self.selected] = 4.0
                output["risk_logits"][:, self.selected % C.N_RISK] = 4.0
                return output

        model = ConditionalLinkFailureLogitBlend(
            LogitModel(0), LogitModel(2),
            class_strength=[0.0, 1.0, 0.0],
            risk_strength=[0.0, 1.0, 0.0],
        )
        csi = torch.zeros(2, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 4, C.N_LINKS, dtype=torch.bool)
        mask[0, :, 0] = False
        mask[1, :, 1] = False
        output = model(csi, mask)
        self.assertEqual(int(output["class_logits"][0].argmax()), 0)
        self.assertEqual(int(output["class_logits"][1].argmax()), 2)

    def test_link_failure_root_blend_changes_only_failed_samples(self):
        class ConstantRoot(DummyP2):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["root"].fill_(self.value)
                return output

        model = ConditionalLinkFailureRootBlend(
            ConstantRoot(1.0), ConstantRoot(3.0), strength=0.25
        )
        csi = torch.zeros(2, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 4, C.N_LINKS, dtype=torch.bool)
        mask[1, :, 2] = False
        output = model(csi, mask)
        self.assertTrue(torch.all(output["root"][0] == 1.0))
        self.assertTrue(torch.all(output["root"][1] == 1.5))

    def test_link_failure_root_blend_routes_selected_links(self):
        class ConstantRoot(DummyP2):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["root"].fill_(self.value)
                return output

        model = ConditionalLinkFailureRootBlend(
            ConstantRoot(1.0),
            ConstantRoot(3.0),
            strength=1.0,
            secondary_expert=ConstantRoot(5.0),
            secondary_strength=1.0,
            secondary_links=(2,),
        )
        csi = torch.zeros(3, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(3, 4, C.N_LINKS, dtype=torch.bool)
        mask[0, :, 0] = False
        mask[1, :, 1] = False
        mask[2, :, 2] = False
        output = model(csi, mask)
        self.assertTrue(torch.all(output["root"][0] == 3.0))
        self.assertTrue(torch.all(output["root"][1] == 3.0))
        self.assertTrue(torch.all(output["root"][2] == 5.0))
        self.assertEqual(output["link_failure_missing_link"].tolist(), [0, 1, 2])

    def test_link_failure_gate_can_use_temporal_coverage(self):
        class ConstantPose(DummyP2):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["pose_rel"].fill_(self.value)
                return output

        model = ConditionalLinkFailurePoseBlend(
            ConstantPose(1.0), ConstantPose(3.0), strength=1.0,
            minimum_link_coverage=0.75,
        )
        csi = torch.zeros(1, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 8, C.N_LINKS, dtype=torch.bool)
        mask[:, 2:6, 0] = False
        output = model(csi, mask)
        self.assertTrue(output["link_failure_gate"].item())
        self.assertTrue(torch.all(output["pose_rel"] == 3.0))

    def test_partial_link_failure_can_use_reduced_expert_strength(self):
        class ConstantPose(DummyP2):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, csi, link_mask):
                output = super().forward(csi, link_mask)
                output["pose_rel"].fill_(self.value)
                output["root"].fill_(self.value)
                return output

        pose_model = ConditionalLinkFailurePoseBlend(
            ConstantPose(1.0), ConstantPose(3.0), strength=1.0,
            minimum_link_coverage=0.75, partial_strength_scale=0.25,
        )
        root_model = ConditionalLinkFailureRootBlend(
            ConstantPose(1.0), ConstantPose(3.0), strength=1.0,
            minimum_link_coverage=0.75, partial_strength_scale=0.25,
        )
        csi = torch.zeros(1, 8, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 8, C.N_LINKS, dtype=torch.bool)
        mask[:, 2:6, 0] = False
        pose_output = pose_model(csi, mask)
        root_output = root_model(csi, mask)
        self.assertTrue(torch.all(pose_output["pose_rel"] == 1.5))
        self.assertTrue(torch.all(root_output["root"] == 1.5))

    def test_model_soup_averages_float_and_preserves_integer_state(self):
        states = [
            {"weight": torch.tensor((1.0, 3.0)), "count": torch.tensor(2)},
            {"weight": torch.tensor((3.0, 7.0)), "count": torch.tensor(2)},
        ]
        soup = average_state_dicts(states, [0.25, 0.75])
        self.assertTrue(torch.equal(soup["weight"], torch.tensor((2.5, 6.0))))
        self.assertEqual(int(soup["count"]), 2)

    def test_hierarchical_risk_aggregates_class_probabilities(self):
        model = HierarchicalRiskCalibration(DummyP2(), class_weight=1.0)
        csi = torch.zeros(2, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 4, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        probability = torch.softmax(output["risk_logits"], dim=-1)
        expected = torch.tensor((9.0, 3.0, 5.0)) / 17.0
        self.assertTrue(torch.allclose(probability, expected[None].expand(2, -1)))

    def test_global_shift_pose_loss_recovers_constant_offset(self):
        target = torch.zeros(1, 8, C.N_JOINTS, 3)
        predicted = torch.zeros_like(target)
        target[:, 3, 0, 0] = 1.0
        predicted[:, 2, 0, 0] = 1.0
        valid = torch.ones(1, 8, dtype=torch.bool)
        loss = global_shift_pose_loss(
            predicted, target, valid, max_shift=2, shift_penalty=0.0
        )
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss)))

    def test_global_shift_root_loss_recovers_constant_offset(self):
        target = torch.zeros(1, 8, 3)
        predicted = torch.zeros_like(target)
        target[:, 3, 0] = 1.0
        predicted[:, 2, 0] = 1.0
        valid = torch.ones(1, 8, dtype=torch.bool)
        loss = global_shift_root_loss(
            predicted, target, valid, max_shift=2, shift_penalty=0.0
        )
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss)))

    def test_multiscale_root_loss_is_zero_for_exact_prediction(self):
        root = torch.randn(2, 40, 3)
        valid = torch.ones(2, 40, dtype=torch.bool)
        loss, _ = root_only_reconstruction_loss(
            {"root": root},
            {
                "root": root,
                "valid": valid,
                "risk_id": torch.tensor((0, 2)),
            },
            velocity_weight=1.0,
            displacement_weight=1.0,
            endpoint_weight=1.0,
            velocity_lags=(5, 15, 30),
        )
        self.assertEqual(float(loss), 0.0)

    def test_pose_ensemble_weights_can_be_recalibrated(self):
        ensemble = PoseModelEnsemble([DummyP2(), DummyP2()])
        ensemble.set_weights([1.0, 3.0])
        self.assertTrue(torch.allclose(
            ensemble.weights, torch.tensor((0.25, 0.75))
        ))

    @staticmethod
    def _moment_calibrator(strength):
        reference_mu = torch.zeros(
            C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2,
        )
        reference_sigma = torch.ones_like(reference_mu)
        return InputMomentCalibration(
            nn.Identity(), reference_mu, reference_sigma, strength=strength,
        )

    @staticmethod
    def _unit_covariance_csi():
        points = torch.tensor((
            (2.0 ** 0.5, 0.0),
            (-2.0 ** 0.5, 0.0),
            (0.0, 2.0 ** 0.5),
            (0.0, -2.0 ** 0.5),
        ))
        count = 2 * C.N_LIVE_SUBCARRIERS
        values = points.repeat((count + 3) // 4, 1)[:count]
        return values.reshape(1, 2, 1, C.N_LIVE_SUBCARRIERS, 2).repeat(
            1, 1, C.N_LINKS, 1, 1,
        )

    def test_input_moment_zero_strength_is_exact_identity(self):
        csi = torch.randn(2, 5, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 5, C.N_LINKS, dtype=torch.bool)
        calibrated = self._moment_calibrator(0.0).calibrate(csi, mask)
        self.assertTrue(torch.equal(calibrated, csi))

    def test_input_moment_matching_distribution_is_near_identity(self):
        csi = self._unit_covariance_csi()
        mask = torch.ones(1, 2, C.N_LINKS, dtype=torch.bool)
        calibrated = self._moment_calibrator(1.0).calibrate(csi, mask)
        self.assertTrue(torch.allclose(calibrated, csi, atol=1e-4))

    def test_input_moment_removes_iq_affine_shift(self):
        csi = self._unit_covariance_csi()
        transform = torch.tensor(((2.0, 0.7), (-0.3, 0.5)))
        shifted = csi @ transform + torch.tensor((3.0, -2.0))
        mask = torch.ones(1, 2, C.N_LINKS, dtype=torch.bool)
        calibrated = self._moment_calibrator(1.0).calibrate(shifted, mask)
        flat = calibrated.permute(0, 2, 1, 3, 4).reshape(
            1, C.N_LINKS, -1, 2,
        )
        mean = flat.mean(2)
        centered = flat - mean[:, :, None]
        covariance = torch.einsum("blni,blnj->blij", centered, centered)
        covariance = covariance / flat.shape[2]
        identity = torch.eye(2)[None, None].expand_as(covariance)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(covariance, identity, atol=5e-4))

    def test_pa_mpjpe_removes_similarity_transform(self):
        target = torch.randn(4, C.N_JOINTS, 3)
        angle = torch.tensor(0.7)
        rotation = torch.tensor((
            (torch.cos(angle), -torch.sin(angle), 0.0),
            (torch.sin(angle), torch.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ))
        predicted = 1.7 * (target @ rotation) + torch.tensor((2.0, -1.0, 0.5))
        error = _similarity_aligned_mpjpe(predicted, target)
        self.assertLess(float(error.max()), 1e-5)

    def test_zero_calibration_recovers_p2(self):
        model = P2V9HybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        model.set_calibration(0.0, 0.0, 0.0, 0.0)
        csi = torch.ones(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.equal(output["pose_rel"], output["pose_p2"]))
        self.assertTrue(torch.equal(output["root"], output["root_p2"]))

    def test_direct_root_starts_as_exact_p2_residual(self):
        model = P2V11DirectRootHybridNet(
            DummyP2(), hidden=16, dropout=0.0, raw_size=8,
        ).eval()
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.equal(output["root"], output["root_p2"]))
        self.assertEqual(output["root_position_delta_v11"].shape, (2, 12, 3))
        self.assertTrue(torch.equal(output["class_logits"], output["class_logits_p2"]))
        self.assertTrue(torch.equal(output["risk_logits"], output["risk_logits_p2"]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.base.parameters()))

    def test_output_shapes(self):
        model = P2V9HybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        csi = torch.ones(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["pose_rel"].shape, (2, 12, C.N_JOINTS, 3))
        self.assertEqual(output["root"].shape, (2, 12, 3))
        self.assertEqual(output["class_logits"].shape, (2, C.N_CLASSES))
        self.assertEqual(output["risk_logits"].shape, (2, C.N_RISK))

    def test_graph_decoder_zero_calibration_recovers_p2(self):
        model = P2V11GraphHybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        model.set_calibration(0.0, 0.0, 0.0, 0.0)
        csi = torch.ones(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertEqual(output["rotation_6d_delta_v10"].shape, (2, 12, C.N_JOINTS, 6))
        self.assertTrue(torch.equal(output["pose_rel"], output["pose_p2"]))

    def test_spectral_adapter_zero_calibration_recovers_p2(self):
        model = P2V11SpectralHybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        model.set_calibration(0.0, 0.0, 0.0, 0.0)
        csi = torch.randn(2, 15, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 15, C.N_LINKS, dtype=torch.bool)
        mask[0, -2:] = False
        output = model(csi, mask)
        self.assertTrue(torch.isfinite(output["temporal_features_v10"]).all())
        self.assertTrue(torch.equal(output["pose_rel"], output["pose_p2"]))

    def test_cartesian_adapter_is_bounded_and_root_relative(self):
        model = P2V11CartesianHybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        model.set_calibration(1.0, 0.0, 0.0, 0.0)
        with torch.no_grad():
            model.cartesian_head[-1].bias.fill_(10.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        delta = output["cartesian_delta_v11"]
        self.assertLessEqual(float(delta.abs().max().detach()), 0.100001)
        self.assertTrue(torch.allclose(
            output["pose_rel"][:, :, C.ROOT_JOINT],
            torch.zeros_like(output["pose_rel"][:, :, C.ROOT_JOINT]),
        ))

    def test_motion_features_are_finite_and_masked(self):
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        mask[0, 4] = False
        feature = p2_motion_features(csi, mask)
        self.assertEqual(feature.shape, (2, 12, 4))
        self.assertTrue(torch.isfinite(feature).all())
        self.assertTrue(torch.equal(feature[0, 4], torch.zeros(4)))

    def test_subcarrier_adapter_zero_calibration_recovers_p2(self):
        model = P2V11SubcarrierHybridNet(DummyP2(), hidden=16, dropout=0.0).eval()
        model.set_calibration(0.0, 0.0, 0.0, 0.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        mask[0, 4, 1] = False
        output = model(csi, mask)
        self.assertTrue(torch.isfinite(output["temporal_features_v10"]).all())
        self.assertTrue(torch.equal(output["pose_rel"], output["pose_p2"]))

    def test_pose_and_root_selection_are_decoupled(self):
        metrics = {
            "mpjpe_m": 0.15,
            "dynamic_mpjpe_m": 0.18,
            "danger_mpjpe_m": 0.45,
            "danger_endpoint_mpjpe_m": 0.60,
            "pose_speed_ratio": 1.0,
            "root_error_m": 0.35,
            "danger_root_error_m": 0.40,
            "danger_root_drop_mae_m": 0.10,
        }
        pose_score = pose_selection_score(metrics)
        metrics["root_error_m"] = 9.0
        self.assertEqual(pose_selection_score(metrics), pose_score)
        self.assertGreater(root_selection_score(metrics), pose_score)

    def test_root_expert_blend_changes_only_root(self):
        class ConstantModel(nn.Module):
            def __init__(self, root_value):
                super().__init__()
                self.root_value = root_value

            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                return {
                    "pose_rel": csi.new_ones(batch, frames, C.N_JOINTS, 3),
                    "root": csi.new_full((batch, frames, 3), self.root_value),
                    "class_logits": csi.new_zeros(batch, C.N_CLASSES),
                    "risk_logits": csi.new_zeros(batch, C.N_RISK),
                }

        model = RootExpertBlend(ConstantModel(1.0), ConstantModel(3.0))
        model.set_root_strength(0.25)
        csi = torch.ones(1, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 4, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.equal(output["pose_rel"], torch.ones_like(output["pose_rel"])))
        self.assertTrue(torch.allclose(output["root"], torch.full_like(output["root"], 1.5)))

    def test_classification_expert_changes_only_logits(self):
        class ConstantModel(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                return {
                    "pose_rel": csi.new_full(
                        (batch, frames, C.N_JOINTS, 3), self.value
                    ),
                    "root": csi.new_full((batch, frames, 3), self.value),
                    "class_logits": csi.new_full(
                        (batch, C.N_CLASSES), self.value
                    ),
                    "risk_logits": csi.new_full((batch, C.N_RISK), self.value),
                }

        model = ClassificationExpertBlend(ConstantModel(1.0), ConstantModel(3.0))
        model.set_calibration(0.5, 0.25)
        csi = torch.ones(1, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 4, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.all(output["pose_rel"] == 1.0))
        self.assertTrue(torch.all(output["root"] == 1.0))
        self.assertTrue(torch.all(output["class_logits"] == 2.0))
        self.assertTrue(torch.all(output["risk_logits"] == 1.5))

    def test_hybrid_logit_fast_path_matches_full_forward(self):
        model = P2V11SubcarrierHybridNet(
            DummyP2(), hidden=16, dropout=0.0
        ).eval()
        model.set_calibration(0.0, 0.0, 1.0, 1.0)
        csi = torch.randn(2, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(2, 12, C.N_LINKS, dtype=torch.bool)
        full = model(csi, mask)
        fast = model.forward_logits(csi, mask)
        self.assertTrue(torch.equal(full["class_logits"], fast["class_logits"]))
        self.assertTrue(torch.equal(full["risk_logits"], fast["risk_logits"]))

    def test_root_components_have_identity_fallback(self):
        class Primary(nn.Module):
            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                root = torch.arange(frames, device=csi.device)[None, :, None]
                return {"root": root.expand(batch, -1, 3).to(csi.dtype)}

        class Expert(nn.Module):
            def forward(self, csi, link_mask):
                output = Primary()(csi, link_mask)
                batch, frames = csi.shape[:2]
                output.update({
                    "root_anchor_delta_v10": csi.new_ones(batch, 3),
                    "root_step_delta_v10": csi.new_ones(batch, frames, 3),
                })
                return output

        csi = torch.ones(1, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 4, C.N_LINKS, dtype=torch.bool)
        model = RootComponentBlend(Primary(), Expert())
        model.set_calibration(0.0, 0.0)
        self.assertTrue(torch.equal(
            model(csi, mask)["root"], Primary()(csi, mask)["root"]
        ))

    def test_residual_smoothing_preserves_p2_and_reduces_impulse(self):
        class PosePair(nn.Module):
            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                coarse = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
                refined = coarse.clone()
                refined[:, frames // 2, :, 0] = 1.0
                return {"pose_p2": coarse, "pose_rel": refined}

        csi = torch.ones(1, 7, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 7, C.N_LINKS, dtype=torch.bool)
        model = ResidualTemporalCalibration(PosePair())
        model.set_calibration(window=3, blend=1.0)
        output = model(csi, mask)
        self.assertTrue(torch.equal(
            output["pose_p2"], torch.zeros_like(output["pose_p2"])
        ))
        self.assertAlmostEqual(
            float(output["pose_rel"][0, 3, 0, 0]), 1.0 / 3.0, places=6
        )
        self.assertAlmostEqual(
            float(output["pose_rel"][0, 2, 0, 0]), 1.0 / 3.0, places=6
        )

    def test_hard_danger_gate_preserves_unsmoothed_residual(self):
        class DangerPosePair(nn.Module):
            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                coarse = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
                refined = coarse.clone()
                refined[:, frames // 2, :, 0] = 1.0
                risk_logits = csi.new_zeros(batch, C.N_RISK)
                risk_logits[:, 2] = 2.0
                return {
                    "pose_p2": coarse,
                    "pose_rel": refined,
                    "risk_logits": risk_logits,
                }

        csi = torch.ones(1, 7, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 7, C.N_LINKS, dtype=torch.bool)
        model = ResidualTemporalCalibration(DangerPosePair())
        model.set_calibration(window=7, blend=1.0, risk_adaptive="hard")
        output = model(csi, mask)
        self.assertEqual(float(output["pose_rel"][0, 3, 0, 0]), 1.0)

    def test_root_residual_smoothing_preserves_primary_root(self):
        class RootPair(nn.Module):
            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                coarse_pose = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
                primary_root = csi.new_zeros(batch, frames, 3)
                refined_root = primary_root.clone()
                refined_root[:, frames // 2, 0] = 1.0
                return {
                    "pose_p2": coarse_pose,
                    "pose_rel": coarse_pose,
                    "root_primary": primary_root,
                    "root": refined_root,
                    "risk_logits": csi.new_zeros(batch, C.N_RISK),
                }

        csi = torch.ones(1, 7, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 7, C.N_LINKS, dtype=torch.bool)
        model = ResidualTemporalCalibration(RootPair())
        model.set_root_calibration(window=3, blend=1.0)
        output = model(csi, mask)
        self.assertEqual(float(output["root_primary"].abs().sum()), 0.0)
        self.assertAlmostEqual(float(output["root"][0, 3, 0]), 1.0 / 3.0, 6)

    def test_sequence_bone_projection_uses_constant_trial_lengths(self):
        pose = torch.zeros(2, 7, C.N_JOINTS, 3)
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                direction = torch.tensor((0.2, 1.0, -0.1))
                scale = 0.05 * child * torch.linspace(0.8, 1.2, 7)
                pose[:, :, child] = pose[:, :, parent] + scale[None, :, None] * direction
        valid = torch.ones(2, 7, dtype=torch.bool)
        projected = sequence_bone_projection(pose, valid)
        lengths = torch.linalg.vector_norm(_local_bones(projected), dim=-1)
        variation = lengths[:, :, 1:].amax(1) - lengths[:, :, 1:].amin(1)
        self.assertTrue(torch.allclose(variation, torch.zeros_like(variation), atol=1e-6))
        self.assertTrue(torch.equal(
            projected[:, :, C.ROOT_JOINT],
            torch.zeros_like(projected[:, :, C.ROOT_JOINT]),
        ))

    def test_anatomical_zero_strength_recovers_coarse_pose(self):
        class PosePair(nn.Module):
            def forward(self, csi, link_mask):
                batch, frames = csi.shape[:2]
                coarse = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
                refined = coarse.clone()
                for child, parent in enumerate(C.JOINT_PARENTS):
                    if parent >= 0:
                        coarse[:, :, child] = coarse[:, :, parent]
                        coarse[:, :, child, 1] += 0.1
                        refined[:, :, child] = refined[:, :, parent]
                        refined[:, :, child, 0] += 0.1
                return {"pose_p2": coarse, "pose_rel": refined}

        model = AnatomicalResidualCalibration(PosePair(), 0.0, 0.0, 0.0)
        csi = torch.ones(1, 4, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 4, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.allclose(output["pose_rel"], output["pose_p2"]))


if __name__ == "__main__":
    unittest.main()
