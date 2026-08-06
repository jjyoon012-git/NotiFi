from __future__ import annotations

import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.continuous_pose import CSILatentPoseRegressor
from notifi_pose.doppler_pose import DopplerTimeFrequencyEncoder
from notifi_pose.geometry_phase import temporal_phase_contrastive
from notifi_pose.geometry_phase_pose import (
    GeometryPhaseCoarseResidual,
    GeometryPhasePoseRegressor,
)
from notifi_pose.hierarchical_pose import HierarchicalCSIPoseRegressor
from notifi_pose.motion_tokens import KinematicMotionTokenizer
from notifi_pose.nets import PerLinkNorm, PoseNet
from notifi_pose.tools.train_geometry_phase_pose import (
    _trainable_groups,
    set_training_stage,
    warmup_cosine_factor,
)
from notifi_pose.tools.train_coarse_motion_pose import (
    _parameter_groups as _coarse_parameter_groups,
    set_training_stage as set_coarse_training_stage,
)


class GeometryPhaseTests(unittest.TestCase):
    @staticmethod
    def make_model() -> GeometryPhasePoseRegressor:
        base = PoseNet(hidden=32, dilations=(1,), n_blocks=1, dropout=0.0)
        tokenizer = KinematicMotionTokenizer(
            hidden=32, code_dim=16, downsample=2, continuous=True
        )
        lengths = torch.full((C.N_JOINTS,), 0.2)
        lengths[C.ROOT_JOINT] = 0.0
        backbone = CSILatentPoseRegressor(
            base, tokenizer.decoder, torch.zeros(16), torch.ones(16), lengths,
            hidden=32, code_dim=16, temporal_layers=1, heads=4, dropout=0.0,
        )
        return GeometryPhasePoseRegressor(backbone, dropout=0.0)

    def test_installation_geometry_contract(self):
        self.assertEqual(C.BOARD_BEARINGS["RX"], (0.0, 1.0))
        self.assertEqual(C.BOARD_BEARINGS["TX1"], (0.0, -1.0))
        self.assertEqual(C.BOARD_BEARINGS["TX2"], (-1.0, 0.0))
        self.assertEqual(C.BOARD_BEARINGS["TX3"], (1.0, 0.0))
        self.assertEqual(len(C.LINK_GEOMETRY), C.N_LINKS)

    def test_geometry_residual_starts_at_exact_zero(self):
        encoder = DopplerTimeFrequencyEncoder(
            PerLinkNorm(), hidden=16, heads=4, dropout=0.0
        )
        final = encoder.geometry_projection[-1]
        self.assertEqual(torch.count_nonzero(final.weight), 0)
        self.assertEqual(torch.count_nonzero(final.bias), 0)

        links = torch.randn(2, 7, C.N_LINKS, 16)
        mask = torch.ones(2, 7, C.N_LINKS, dtype=torch.bool)
        residual = encoder.geometry_projection(
            encoder.directional_moments(links, mask)
        )
        self.assertEqual(torch.count_nonzero(residual), 0)

    def test_directional_moments_distinguish_east_and_west(self):
        encoder = DopplerTimeFrequencyEncoder(
            PerLinkNorm(), hidden=8, heads=2, dropout=0.0
        )
        links = torch.zeros(1, 1, C.N_LINKS, 8)
        links[:, :, C.LINK_INDEX["TX2"]] = 1.0
        links[:, :, C.LINK_INDEX["TX3"]] = 3.0
        mask = torch.ones(1, 1, C.N_LINKS, dtype=torch.bool)
        moments = encoder.directional_moments(links, mask).reshape(1, 1, 4, 8)
        # TX bearing x: west contributes -1 and east contributes +3.
        self.assertTrue(torch.all(moments[:, :, 0] > 0.0))
        # TX bearing y has no contribution from east/west links.
        self.assertEqual(torch.count_nonzero(moments[:, :, 1]), 0)

    def test_phase_contrastive_prefers_correct_temporal_order(self):
        torch.manual_seed(4)
        target = torch.randn(2, 24, 8).cumsum(1)
        valid = torch.ones(2, 24, dtype=torch.bool)
        predicted = target.clone().requires_grad_(True)
        aligned, aligned_stats = temporal_phase_contrastive(
            predicted, target, valid, temperature=0.08
        )
        reversed_loss, reversed_stats = temporal_phase_contrastive(
            target.flip(1), target, valid, temperature=0.08
        )
        self.assertLess(float(aligned.detach()), float(reversed_loss))
        self.assertGreater(
            float(aligned_stats["phase_top1"]),
            float(reversed_stats["phase_top1"]),
        )
        aligned.backward()
        self.assertGreater(torch.count_nonzero(predicted.grad), 0)

    def test_phase_neighborhood_tolerates_one_token_shift(self):
        torch.manual_seed(8)
        target = torch.randn(1, 20, 6).cumsum(1)
        shifted = torch.zeros_like(target)
        shifted[:, 1:] = target[:, :-1]
        valid = torch.ones(1, 20, dtype=torch.bool)
        tolerant, _ = temporal_phase_contrastive(
            shifted, target, valid, positive_radius=1
        )
        strict, _ = temporal_phase_contrastive(
            shifted, target, valid, positive_radius=0
        )
        self.assertLess(float(tolerant), float(strict))

    def test_staged_training_only_opens_intended_parameters(self):
        model = self.make_model()
        groups = _trainable_groups(model)
        set_training_stage(groups, "geometry_warmup")
        self.assertTrue(all(
            parameter.requires_grad
            for name in ("geometry", "phase")
            for parameter in groups[name]["named"].values()
        ))
        self.assertFalse(any(
            parameter.requires_grad
            for name in ("hierarchy", "backbone")
            for parameter in groups[name]["named"].values()
        ))
        set_training_stage(groups, "joint_finetune")
        self.assertTrue(all(
            parameter.requires_grad
            for group in groups.values()
            for parameter in group["named"].values()
        ))

    def test_phase_head_starts_equal_but_is_parameter_independent(self):
        model = self.make_model().eval()
        model.initialize_phase_head_from_pose()
        csi = torch.randn(1, 36, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 36, C.N_LINKS, dtype=torch.bool)
        output = model(csi, mask)
        torch.testing.assert_close(
            output["phase_motion_latent"],
            output["normalized_motion_latent"],
        )
        self.assertIsNot(
            model.phase_head.weight, model.backbone.latent_head.weight
        )

    def test_coarse_residual_starts_as_exact_v13s_identity(self):
        source = self.make_model().eval()
        model = GeometryPhaseCoarseResidual(
            source, dropout=0.0, max_delta=0.25
        ).eval()
        csi = torch.randn(1, 36, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 36, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            coarse = torch.randn_like(source(csi, mask)["pose_rel"])
        coarse = coarse - coarse[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        output = model(csi, mask, coarse)
        torch.testing.assert_close(output["pose_rel"], coarse)
        self.assertEqual(torch.count_nonzero(output["pose_delta"]), 0)

    def test_proposal_conditioned_residual_starts_at_locked_proposal(self):
        source = self.make_model().eval()
        model = GeometryPhaseCoarseResidual(
            source, dropout=0.0, proposal_strength=0.30
        ).eval()
        csi = torch.randn(1, 36, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        mask = torch.ones(1, 36, C.N_LINKS, dtype=torch.bool)
        with torch.no_grad():
            candidate = source(csi, mask)["pose_rel"]
        coarse = torch.randn_like(candidate)
        coarse = coarse - coarse[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        output = model(csi, mask, coarse)
        expected = coarse + 0.30 * (candidate - coarse)
        expected = expected - expected[
            :, :, C.ROOT_JOINT:C.ROOT_JOINT + 1
        ]
        torch.testing.assert_close(output["pose_rel"], expected)
        self.assertEqual(torch.count_nonzero(output["pose_delta"]), 0)

    def test_coarse_stage_keeps_redundant_seen_geometry_frozen(self):
        model = GeometryPhaseCoarseResidual(self.make_model())
        groups = _coarse_parameter_groups(model)
        set_coarse_training_stage(model, groups, "residual_warmup")
        self.assertTrue(all(
            parameter.requires_grad
            for name in ("residual", "phase")
            for parameter in groups[name]["named"].values()
        ))
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in groups["backbone"]["named"].values()
        ))
        geometry = [
            parameter for name, parameter in model.named_parameters()
            if "geometry_projection." in name
        ]
        self.assertTrue(geometry)
        self.assertFalse(any(parameter.requires_grad for parameter in geometry))

    def test_warmup_cosine_reaches_minimum_ratio(self):
        factors = [
            warmup_cosine_factor(epoch, 60, 3, 0.05)
            for epoch in range(60)
        ]
        self.assertAlmostEqual(factors[0], 1.0 / 3.0)
        self.assertAlmostEqual(factors[2], 1.0)
        self.assertAlmostEqual(factors[-1], 0.05)
        self.assertGreater(factors[10], factors[40])


if __name__ == "__main__":
    unittest.main()
