"""CAL20 physics calibration, domain generalization, and CAL17 transport tests."""

from __future__ import annotations

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
from notifi_pose.cal20 import CAL20RelativeMotionDG
from notifi_pose.deployment import CAL20Deployment
from notifi_pose.model_factory import build_calibration_model
from scripts.train_cal20_source_folds import (
    cal12_site_selection_score,
    nested_site_split,
)
from scripts.evaluate_cal20_rf_stress import TargetShiftStore
from scripts.export_cal20_deployment import require_source_clean
from scripts.source_calibration_data import select_absence


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

    def test_model_factory_restores_cal20(self) -> None:
        model = build_calibration_model({
            "architecture": "cal20_relative_motion_dg",
            "hidden": 16, "width": 32, "domains": 3,
            "use_doppler": False,
        })
        self.assertIsInstance(model, CAL20RelativeMotionDG)

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
