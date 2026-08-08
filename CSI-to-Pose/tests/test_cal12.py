"""CAL20 physics calibration, domain generalization, and CAL17 transport tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from notifi_pose import contract as C
from notifi_pose.cal12 import (
    PhysicsSupportCanonicalizer,
    circular_delta,
    cross_site_supervised_contrastive,
)
from notifi_pose.cal13 import (
    MOTION_DESCRIPTOR_DIM,
    pose_motion_descriptor,
    shift_robust_motion_loss,
    temporal_motion_signature,
)
from notifi_pose.cal14 import CosineClassifier
from notifi_pose.cal17 import (
    anchor_geometry_error,
    transport_class_prototypes,
)
from notifi_pose.cal20 import CAL20RelativeMotionDG, MotionProgressEncoder
from notifi_pose.deployment import CAL20Deployment
from notifi_pose.model_factory import build_calibration_model
from scripts.train_cal20_source_folds import (
    cal12_site_selection_score,
    experiment_name,
    load_clean_source_state,
    nested_site_split,
    validate_training_options,
)
from scripts.evaluate_cal20_rf_stress import TargetShiftStore
from scripts.export_cal20_deployment import require_source_clean
from scripts.source_calibration_data import (
    reflect_east_west,
    select_absence,
    select_source_rows,
    transfer_site_style,
    temporal_warp_trials,
)


class CalibrationModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)
        self.frames = 24
        live = 16
        self.query = torch.randn(3, self.frames, C.N_LINKS, live, 2)
        self.query[..., 0].abs_()
        self.query_mask = torch.ones(
            3, self.frames, C.N_LINKS, dtype=torch.bool
        )
        classes = (0, 1, 2, 3, 4, 5, 7, 8)
        self.support_labels = torch.tensor(classes).repeat_interleave(2)
        self.support = torch.randn(
            len(self.support_labels), self.frames, C.N_LINKS, live, 2
        )
        self.support[..., 0].abs_()
        self.support_mask = torch.ones(
            len(self.support), self.frames, C.N_LINKS, dtype=torch.bool
        )
        self.absence = torch.randn(2, self.frames, C.N_LINKS, live, 2)
        self.absence[..., 0].abs_()
        self.absence_mask = torch.ones(
            2, self.frames, C.N_LINKS, dtype=torch.bool
        )

    def test_circular_delta_does_not_jump_at_wrap(self) -> None:
        phase = torch.tensor((3.13, -3.13))[None, :, None, None]
        delta = circular_delta(phase, 1)
        self.assertLess(float(delta[:, 1].abs().max()), 0.05)

    def test_zero_strength_uses_source_motion_scale(self) -> None:
        canonicalizer = PhysicsSupportCanonicalizer(phase_strength=0.25)
        canonicalizer.source_amp_scale.copy_(torch.tensor((0.5, 0.7, 0.9)))
        canonicalizer.source_phase_scale.copy_(torch.tensor((0.4, 0.6, 0.8)))
        canonicalizer.source_initialized.fill_(True)
        output = canonicalizer.prepare(
            self.query, self.query_mask,
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask, calibration_strength=0.0,
        )
        torch.testing.assert_close(
            output["safe_amp_scale"], canonicalizer.source_amp_scale
        )
        torch.testing.assert_close(
            output["safe_phase_scale"], canonicalizer.source_phase_scale
        )

    def test_cross_site_contrastive_needs_cross_site_positive(self) -> None:
        embedding = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor((0, 1, 2, 3))
        domains = torch.tensor((0, 0, 1, 1))
        loss = cross_site_supervised_contrastive(
            embedding, labels, domains
        )
        self.assertEqual(float(loss.detach()), 0.0)

    def test_rf_stress_changes_only_declared_target_rows(self) -> None:
        class Store:
            def get(self, rows, device):
                count = len(rows)
                values = torch.ones(count, 4, C.N_LINKS, 3, 2)
                mask = torch.ones(count, 4, C.N_LINKS, dtype=torch.bool)
                return values, mask

        shifted = TargetShiftStore(
            Store(), affected_rows={11}, gain_phase=False, dropped_link=0
        )
        values, mask = shifted.get([10, 11], "cpu")
        self.assertTrue(bool(mask[0].all()))
        self.assertFalse(bool(mask[1, :, 0].any()))
        self.assertTrue(bool((values[0] == 1).all()))
        self.assertTrue(bool((values[1, :, 0] == 0).all()))

    def test_absence_selector_honors_deployment_window_count(self) -> None:
        index = pd.DataFrame({
            "subject": ["ajh"] * 12,
            "environment": ["E01"] * 12,
            "task": [C.TASK_CLS] * 12,
            "class_id": [6] * 12,
            "cache_ok": [True] * 12,
            "trial_id": [f"absence_{number:02d}" for number in range(12)],
        })
        selected = select_absence(
            "ajh_E01", index, seed=17, trials=12
        )
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(selected.tolist())), 12)
        with self.assertRaisesRegex(RuntimeError, "fewer than 13"):
            select_absence("ajh_E01", index, seed=17, trials=13)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            select_absence("ajh_E01", index, seed=17, trials=0)

    def test_site_style_transfer_preserves_motion_residual(self) -> None:
        source = torch.zeros(1, 2, C.N_LINKS, 3, 2)
        source[..., 0] = 2.0
        source[:, 1:, ..., 0] = 4.0
        source[..., 1] = 0.2
        absence = torch.zeros_like(source)
        absence[..., 0] = 2.0
        absence[..., 1] = 0.2
        donor = torch.zeros_like(source)
        donor[..., 0] = 4.0
        donor[..., 1] = 1.0
        mask = torch.ones(1, 2, C.N_LINKS, dtype=torch.bool)
        (_, _), (transferred, transferred_mask) = transfer_site_style(
            [(absence, mask), (source, mask)], (donor, mask), strength=1.0
        )
        torch.testing.assert_close(
            transferred[:, 1, ..., 0] / transferred[:, 0, ..., 0],
            source[:, 1, ..., 0] / source[:, 0, ..., 0],
        )
        torch.testing.assert_close(
            transferred[:, 0, ..., 0], torch.full_like(source[:, 0, ..., 0], 4.0)
        )
        torch.testing.assert_close(
            transferred[..., 1], torch.full_like(source[..., 1], 1.0)
        )
        self.assertTrue(torch.equal(transferred_mask, mask))

    def test_source_rows_are_derived_without_legacy_checkpoint(self) -> None:
        index = pd.DataFrame({
            "subject": ["ajh", "lmh", "lmh", "yja", "mhw"],
            "environment": ["E01", "E01", "E02", "E02", "E03"],
            "task": [C.TASK_POSE] * 5,
            "class_id": [0, 1, 2, 3, 6],
            "cache_ok": [True] * 5,
            "role": ["train", "train", "train", "train", "train"],
        })
        self.assertEqual(select_source_rows(index).tolist(), [0, 1])

    def test_export_rejects_target_contaminated_results(self) -> None:
        clean = {
            "target_subject_used": False,
            "sealed_yja_used": False,
            "query_labels_or_pose_gt_at_inference": False,
        }
        require_source_clean(clean, "test")
        for key in clean:
            contaminated = dict(clean)
            contaminated[key] = True
            with self.assertRaises(RuntimeError):
                require_source_clean(contaminated, "test")

    def test_deployment_runs_without_query_labels_or_gt(self) -> None:
        model = CAL20RelativeMotionDG(
            hidden=16, width=32, domains=3, dropout=0.0,
            use_doppler=False, phase_strength=0.25,
        )
        bundle = {
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "support_contract": {
                "prompt_classes": list(model.prompt_classes),
                "shots_per_prompt": 2,
                "absence_trials": 2,
            },
            "source_library": [{
                "classes": torch.randn(C.N_CLASSES, model.hidden),
                "anchors": torch.randn(9, model.hidden),
            }],
            "action_config": {
                "strength": 1.0, "anchor_temperature": 0.1,
                "prototype_temperature": 0.1, "site_temperature": 0.1,
                "mixture": 0.5,
            },
            "risk_config": {
                "safe_weight": 1.0, "fusion": 0.0, "danger_bias": 0.5,
            },
            "sealed_yja_used": False,
            "target_subject_used": False,
            "query_labels_or_pose_gt_used": False,
        }
        runtime = CAL20Deployment(bundle, device="cpu")
        calibration = runtime.calibrate(
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask,
        )
        output = runtime.predict(
            self.query[:1], self.query_mask[:1], calibration,
            simulate_pose=False,
        )
        self.assertEqual(tuple(output["action_logits"].shape), (1, C.N_CLASSES))
        self.assertEqual(tuple(output["risk_logits"].shape), (1, C.N_RISK))
        self.assertNotIn("pose_rel", output)
        self.assertFalse(bool(output["abstain"].item()))
        one_link = torch.zeros_like(self.query_mask[:1])
        one_link[..., 0] = True
        low_quality = runtime.predict(
            self.query[:1], one_link, calibration, simulate_pose=False
        )
        self.assertTrue(bool(low_quality["abstain"].item()))
        strict_bundle = dict(bundle)
        strict_bundle["calibration_geometry_threshold"] = -1.0
        strict = CAL20Deployment(strict_bundle, device="cpu")
        strict_calibration = strict.calibrate(
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask,
        )
        domain_rejected = strict.predict(
            self.query[:1], self.query_mask[:1], strict_calibration,
            simulate_pose=False,
        )
        self.assertFalse(bool(
            domain_rejected["calibration_domain_pass"].item()
        ))
        self.assertTrue(bool(
            domain_rejected["calibration_domain_warning"].item()
        ))
        self.assertFalse(bool(domain_rejected["abstain"].item()))
        contaminated = dict(bundle)
        contaminated["target_subject_used"] = True
        with self.assertRaisesRegex(ValueError, "exclude target"):
            CAL20Deployment(contaminated, device="cpu")
        contaminated = dict(bundle)
        contaminated["query_labels_or_pose_gt_used"] = True
        with self.assertRaisesRegex(ValueError, "exclude query"):
            CAL20Deployment(contaminated, device="cpu")

    def test_deployment_rejects_incomplete_support(self) -> None:
        model = CAL20RelativeMotionDG(
            hidden=16, width=32, domains=3, dropout=0.0,
            use_doppler=False,
        )
        bundle = {
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "support_contract": {
                "prompt_classes": list(model.prompt_classes),
                "shots_per_prompt": 2,
                "absence_trials": 2,
            },
            "source_library": [{
                "classes": torch.randn(C.N_CLASSES, model.hidden),
                "anchors": torch.randn(9, model.hidden),
            }],
            "action_config": {
                "strength": 1.0, "anchor_temperature": 0.1,
                "prototype_temperature": 0.1, "site_temperature": 0.1,
                "mixture": 0.5,
            },
            "risk_config": {
                "safe_weight": 1.0, "fusion": 0.0, "danger_bias": 0.5,
            },
            "sealed_yja_used": False,
            "target_subject_used": False,
            "query_labels_or_pose_gt_used": False,
        }
        runtime = CAL20Deployment(bundle, device="cpu")
        with self.assertRaisesRegex(ValueError, "class 8"):
            runtime.calibrate(
                self.support[:-2], self.support_mask[:-2],
                self.support_labels[:-2], self.absence, self.absence_mask,
            )

    def test_deployment_simulates_pose_from_source_library(self) -> None:
        model = CAL20RelativeMotionDG(
            hidden=16, width=32, domains=3, dropout=0.0,
            use_doppler=False,
        )
        candidates_per_class = 5
        candidate_count = C.N_CLASSES * candidates_per_class
        pose = torch.randn(
            candidate_count, self.frames, C.N_JOINTS, 3
        )
        valid = torch.ones(candidate_count, self.frames, dtype=torch.bool)
        descriptors = pose_motion_descriptor(pose, valid)
        signatures = temporal_motion_signature(descriptors, valid)
        center = signatures.mean(0)
        scale = signatures.std(0).clamp_min(0.05)
        bundle = {
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "support_contract": {
                "prompt_classes": list(model.prompt_classes),
                "shots_per_prompt": 2,
                "absence_trials": 2,
            },
            "source_library": [{
                "classes": torch.randn(C.N_CLASSES, model.hidden),
                "anchors": torch.randn(9, model.hidden),
            }],
            "action_config": {
                "strength": 1.0, "anchor_temperature": 0.1,
                "prototype_temperature": 0.1, "site_temperature": 0.1,
                "mixture": 0.5,
            },
            "risk_config": {
                "safe_weight": 0.0, "fusion": 0.0, "danger_bias": 1.0,
            },
            "pose_config": {
                "neighbors": 5, "temperature": 0.5, "bone_blend": 0.5,
            },
            "pose_library": {
                "pose": pose,
                "valid": valid,
                "descriptors": descriptors,
                "normalized_signatures": (signatures - center) / scale,
                "signature_center": center,
                "signature_scale": scale,
                "labels": torch.arange(C.N_CLASSES).repeat_interleave(
                    candidates_per_class
                ),
                "trial_ids": [
                    f"source_{number}" for number in range(candidate_count)
                ],
            },
            "sealed_yja_used": False,
            "target_subject_used": False,
            "query_labels_or_pose_gt_used": False,
        }
        runtime = CAL20Deployment(bundle, device="cpu")
        calibration = runtime.calibrate(
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask,
        )
        output = runtime.predict(
            self.query[:1], self.query_mask[:1], calibration,
            simulate_pose=True,
        )
        self.assertEqual(
            tuple(output["pose_rel"].shape),
            (1, self.frames, C.N_JOINTS, 3),
        )
        self.assertEqual(tuple(output["pose_valid"].shape), (1, self.frames))
        self.assertEqual(len(output["retrieval_trial_ids"][0]), 5)

    def test_nested_split_never_selects_outer_subject(self) -> None:
        sites = [
            "ajh_E01", "ajh_E02", "ajh_E03",
            "mhw_E01", "mhw_E02", "mhw_E03", "lmh_E01",
        ]
        train, inner, outer = nested_site_split(sites, "ajh")
        self.assertTrue(all(not site.startswith("ajh_") for site in train + inner))
        self.assertTrue(all(site.startswith("ajh_") for site in outer))

    def test_motion_descriptor_removes_uniform_body_scale(self) -> None:
        pose = torch.randn(2, self.frames, C.N_JOINTS, 3)
        valid = torch.ones(2, self.frames, dtype=torch.bool)
        descriptor = pose_motion_descriptor(pose, valid)
        scaled = pose_motion_descriptor(pose * 1.7, valid)
        torch.testing.assert_close(descriptor, scaled, atol=2e-5, rtol=2e-5)

    def test_shift_robust_loss_recovers_small_offset(self) -> None:
        target = torch.randn(1, self.frames, MOTION_DESCRIPTOR_DIM)
        predicted = torch.zeros_like(target)
        predicted[:, 2:] = target[:, :-2]
        valid = torch.ones(1, self.frames, dtype=torch.bool)
        loss, shift = shift_robust_motion_loss(
            predicted, target, valid, max_shift=3
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(abs(int(shift)), 2)

    def test_temporal_signature_preserves_order(self) -> None:
        first = torch.zeros(1, 16, MOTION_DESCRIPTOR_DIM)
        first[:, 8:] = 1.0
        valid = torch.ones(1, 16, dtype=torch.bool)
        self.assertFalse(torch.allclose(
            temporal_motion_signature(first, valid),
            temporal_motion_signature(torch.flip(first, dims=(1,)), valid),
        ))

    def test_cosine_classifier_ignores_feature_magnitude(self) -> None:
        classifier = CosineClassifier(8, 4, initial_scale=7.0)
        feature = torch.randn(3, 8)
        torch.testing.assert_close(
            classifier(feature), classifier(feature * 5.0),
            atol=1e-5, rtol=1e-5,
        )

    def test_style_transport_identity_keeps_prototypes(self) -> None:
        classes = torch.randn(C.N_CLASSES, 8)
        anchors = torch.nn.functional.normalize(classes[:4], dim=-1)
        transported = transport_class_prototypes(
            classes, anchors, anchors, (0, 1, 2, 3), 1.0, 0.1
        )
        torch.testing.assert_close(
            transported, torch.nn.functional.normalize(classes, dim=-1)
        )
        self.assertLess(float(anchor_geometry_error(anchors, anchors)), 1e-7)

    def test_style_transport_sets_observed_anchors_exactly(self) -> None:
        classes = torch.randn(C.N_CLASSES, 8)
        source = torch.randn(3, 8)
        target = torch.randn(3, 8)
        transported = transport_class_prototypes(
            classes, source, target, (0, 4, 8), 0.5, 0.2
        )
        torch.testing.assert_close(
            transported[[0, 4, 8]], torch.nn.functional.normalize(target, dim=-1)
        )

    def test_cal20_requires_support_relative_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            CAL20RelativeMotionDG(
                hidden=16, width=32, domains=3, relative_support=False
            )

    def test_east_west_reflection_swaps_tx2_tx3(self) -> None:
        csi = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(
            1, 2, 3, 4, 2,
        )
        mask = torch.tensor([[[True, False, True], [False, True, True]]])
        (reflected, reflected_mask), = reflect_east_west([(csi, mask)])
        torch.testing.assert_close(reflected[:, :, 0], csi[:, :, 0])
        torch.testing.assert_close(reflected[:, :, 1], csi[:, :, 2])
        torch.testing.assert_close(reflected[:, :, 2], csi[:, :, 1])
        torch.testing.assert_close(reflected_mask[:, :, 1], mask[:, :, 2])

        (restored, restored_mask), = reflect_east_west([
            (reflected, reflected_mask)
        ])
        torch.testing.assert_close(restored, csi)
        self.assertTrue(torch.equal(restored_mask, mask))

    def test_temporal_warp_preserves_shape_and_endpoints(self) -> None:
        csi = torch.randn(3, 20, 3, 4, 2)
        mask = torch.ones(3, 20, 3, dtype=torch.bool)
        warped, warped_mask = temporal_warp_trials(
            csi, mask, seed=57, strength=0.25,
        )
        self.assertEqual(warped.shape, csi.shape)
        self.assertEqual(warped_mask.shape, mask.shape)
        torch.testing.assert_close(warped[:, 0], csi[:, 0])
        torch.testing.assert_close(warped[:, -1], csi[:, -1])

        ramp = torch.arange(20, dtype=torch.float32).reshape(1, 20, 1, 1, 1)
        ramp = ramp.expand(1, 20, 3, 4, 2)
        warped_ramp, _ = temporal_warp_trials(
            ramp, mask[:1], seed=57, strength=0.25,
        )
        self.assertTrue(bool((warped_ramp[:, 1:] >= warped_ramp[:, :-1]).all()))

    def test_zero_strength_temporal_warp_is_identity(self) -> None:
        csi = torch.randn(2, 17, 3, 4, 2)
        mask = torch.rand(2, 17, 3) > 0.25

        warped, warped_mask = temporal_warp_trials(
            csi, mask, seed=91, strength=0.0,
        )

        torch.testing.assert_close(warped, csi)
        self.assertTrue(torch.equal(warped_mask, mask))

    def test_combined_physical_invariance_has_cal60_name(self) -> None:
        options = argparse.Namespace(
            reflection_probability=0.25,
            temporal_warp_probability=0.25,
            motion_phase_bins=4,
            cross_site_style_probability=0.75,
        )
        self.assertEqual(
            experiment_name(options), "CAL60-PHYSICAL-INVARIANCE-DG"
        )

    def test_invalid_physical_invariance_options_fail_before_training(self) -> None:
        options = argparse.Namespace(
            cross_site_style_probability=0.75,
            reflection_probability=1.1,
            temporal_warp_probability=0.25,
            temporal_warp_strength=0.25,
            motion_phase_bins=8,
        )
        with self.assertRaisesRegex(ValueError, "reflection_probability"):
            validate_training_options(options)

        options.reflection_probability = 0.25
        options.motion_phase_bins = 1
        with self.assertRaisesRegex(ValueError, "motion_phase_bins"):
            validate_training_options(options)

    def test_source_initialization_rejects_target_contamination(self) -> None:
        clean = {
            "model": {"weight": torch.ones(1)},
            "outer_holdout_used_for_selection": False,
            "target_subject_used": False,
            "sealed_yja_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(clean, path)
            loaded = load_clean_source_state(path)
            torch.testing.assert_close(loaded["weight"], torch.ones(1))

            contaminated = dict(clean)
            contaminated["target_subject_used"] = True
            torch.save(contaminated, path)
            with self.assertRaisesRegex(RuntimeError, "not clean"):
                load_clean_source_state(path)

    def test_cal20_outputs_compatible_heads(self) -> None:
        model = CAL20RelativeMotionDG(
            hidden=16, width=32, domains=4, dropout=0.0,
            use_doppler=False,
        )
        output = model(
            self.query, self.query_mask,
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask,
        )
        self.assertEqual(output["action_logits"].shape, (3, C.N_CLASSES))
        self.assertEqual(output["risk_logits"].shape, (3, C.N_RISK))
        self.assertEqual(
            output["pose_motion"].shape,
            (3, self.frames, MOTION_DESCRIPTOR_DIM),
        )

    def test_cal34_motion_phase_preserves_output_contract(self) -> None:
        model = CAL20RelativeMotionDG(
            hidden=16, width=32, domains=4, dropout=0.0,
            use_doppler=False, motion_phase_bins=4,
        )
        output = model(
            self.query, self.query_mask,
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask,
        )
        self.assertEqual(output["motion_phase_embedding"].shape, (3, 16))
        self.assertEqual(output["action_logits"].shape, (3, C.N_CLASSES))
        self.assertTrue(torch.isfinite(output["motion_phase_embedding"]).all())

    def test_motion_progress_ignores_masked_tail_padding(self) -> None:
        encoder = MotionProgressEncoder(hidden=8, bins=4, dropout=0.0).eval()
        valid = torch.randn(2, 7, 8)
        short_mask = torch.ones(2, 7, dtype=torch.bool)
        padded = torch.cat((valid, torch.randn(2, 5, 8)), dim=1)
        padded_mask = torch.cat((
            short_mask, torch.zeros(2, 5, dtype=torch.bool)
        ), dim=1)

        with torch.no_grad():
            short_output = encoder(valid, short_mask)
            padded_output = encoder(padded, padded_mask)

        torch.testing.assert_close(short_output, padded_output)

    def test_model_factory_restores_cal20(self) -> None:
        source = CAL20RelativeMotionDG(
            hidden=16, width=32, domains=3,
            use_doppler=False, motion_phase_bins=4,
        )
        restored = build_calibration_model(source.model_config())
        restored.load_state_dict(source.state_dict(), strict=True)

        self.assertIsInstance(restored, CAL20RelativeMotionDG)
        self.assertEqual(restored.motion_phase_bins, 4)
        self.assertIsNotNone(restored.motion_progress)

    def test_selection_rejects_action_collapse(self) -> None:
        collapsed = {
            "action_macro_f1": 0.015, "action_accuracy": 0.06,
            "risk_macro_f1": 0.30, "danger_recall": 0.90,
            "danger_action_accuracy": 0.0, "safe_to_danger": 8,
            "safe_total": 76,
        }
        useful = {
            "action_macro_f1": 0.09, "action_accuracy": 0.12,
            "risk_macro_f1": 0.28, "danger_recall": 0.25,
            "danger_action_accuracy": 0.08, "safe_to_danger": 8,
            "safe_total": 76,
        }
        self.assertGreater(
            cal12_site_selection_score(useful)["score"],
            cal12_site_selection_score(collapsed)["score"],
        )


if __name__ == "__main__":
    unittest.main()
