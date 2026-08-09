"""낙상 calibration prototype과 제한 결합 경로를 검증한다."""

from __future__ import annotations

import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.cal20 import CAL20RelativeMotionDG
from notifi_pose.danger_support import (
    DANGER_SUPPORT_CLASSES,
    apply_danger_support,
    class_prototypes,
    support_evidence,
)
from notifi_pose.deployment import CAL20Deployment


class DangerSupportTests(unittest.TestCase):
    def test_subtype_blend_preserves_danger_group_mass(self) -> None:
        action = torch.randn(3, 17)
        risk = torch.randn(3, 3)
        subtype = torch.randn(3, 5).log_softmax(-1)
        margin = torch.tensor((0.2, -0.1, 0.0))
        before = torch.logsumexp(action[:, 12:17], dim=-1)
        adjusted, adjusted_risk, diagnostics = apply_danger_support(
            action, risk, [(subtype, margin)], {
                "subtype_weight": 0.5,
                "action_margin_gain": 0.0,
                "risk_margin_gain": 0.0,
            },
        )
        after = torch.logsumexp(adjusted[:, 12:17], dim=-1)
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(risk, adjusted_risk)
        self.assertEqual(
            tuple(diagnostics["danger_support_probability"].shape), (3, 5)
        )

    def test_support_evidence_and_prototypes_have_fixed_shapes(self) -> None:
        labels = torch.tensor(DANGER_SUPPORT_CLASSES)
        embedding = torch.eye(5, 8)
        prototypes = class_prototypes(embedding, labels)
        safe = torch.randn(8, 8)
        subtype, margin = support_evidence(
            embedding[:2], safe, prototypes, temperature=0.1
        )
        self.assertEqual(tuple(prototypes.shape), (5, 8))
        self.assertEqual(tuple(subtype.shape), (2, 5))
        self.assertEqual(tuple(margin.shape), (2,))

    def test_missing_danger_class_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing class 16"):
            class_prototypes(
                torch.randn(4, 8), torch.tensor(DANGER_SUPPORT_CLASSES[:-1])
            )

    def test_single_bundle_requires_and_uses_danger_support(self) -> None:
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
            "danger_support_contract": {
                "classes": list(DANGER_SUPPORT_CLASSES),
                "shots_per_class": 1,
            },
            "danger_support_config": {
                "temperature": 0.1, "subtype_weight": 0.5,
                "action_margin_gain": 0.0, "risk_margin_gain": 0.0,
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
                "safe_weight": 0.0, "fusion": 0.0, "danger_bias": 0.0,
            },
            "sealed_yja_used": False,
            "target_subject_used": False,
            "query_labels_or_pose_gt_used": False,
        }
        runtime = CAL20Deployment(bundle, device="cpu")
        frames, subcarriers = 24, 16
        support_labels = torch.tensor(model.prompt_classes).repeat_interleave(2)
        support = torch.randn(
            len(support_labels), frames, C.N_LINKS, subcarriers, 2
        )
        support[..., 0].abs_()
        support_mask = torch.ones(
            len(support), frames, C.N_LINKS, dtype=torch.bool
        )
        absence = torch.randn(2, frames, C.N_LINKS, subcarriers, 2)
        absence[..., 0].abs_()
        absence_mask = torch.ones(2, frames, C.N_LINKS, dtype=torch.bool)
        danger = torch.randn(5, frames, C.N_LINKS, subcarriers, 2)
        danger[..., 0].abs_()
        danger_mask = torch.ones(5, frames, C.N_LINKS, dtype=torch.bool)
        danger_labels = torch.tensor(DANGER_SUPPORT_CLASSES)
        with self.assertRaisesRegex(ValueError, "requires danger"):
            runtime.calibrate(
                support, support_mask, support_labels,
                absence, absence_mask,
            )
        calibration = runtime.calibrate(
            support, support_mask, support_labels,
            absence, absence_mask,
            danger, danger_mask, danger_labels,
        )
        output = runtime.predict(
            support[:1], support_mask[:1], calibration, simulate_pose=False
        )
        self.assertEqual(
            tuple(output["danger_support_probability"].shape), (1, 5)
        )
        torch.testing.assert_close(
            output["danger_support_probability"].sum(-1), torch.ones(1)
        )


if __name__ == "__main__":
    unittest.main()
