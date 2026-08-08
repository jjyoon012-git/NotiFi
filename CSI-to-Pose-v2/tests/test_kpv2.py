"""KP v2 행동 중심 encoder와 checkpoint 계약을 검증한다."""

from __future__ import annotations

import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.cal13 import MOTION_DESCRIPTOR_DIM
from notifi_pose.kpv2 import KPV2ActionPose, ScaleFreeMotionEncoder
from notifi_pose.model_factory import build_calibration_model
from notifi_pose.pose_semantic_teacher import (
    PoseSemanticTeacher,
    cross_modal_supervised_contrastive,
)


class KPV2Tests(unittest.TestCase):
    """행동 content와 환경 style의 분리 및 출력 shape를 확인한다."""

    def setUp(self) -> None:
        """각 prompt class를 빠짐없이 포함하는 작은 calibration episode를 만든다."""
        torch.manual_seed(22)
        self.frames = 24
        self.live = 16
        self.query = torch.randn(3, self.frames, C.N_LINKS, self.live, 2)
        self.query[..., 0].abs_()
        self.query_mask = torch.ones(
            3, self.frames, C.N_LINKS, dtype=torch.bool
        )
        classes = (0, 1, 2, 3, 4, 5, 7, 8)
        self.support_labels = torch.tensor(classes).repeat_interleave(2)
        self.support = torch.randn(
            len(self.support_labels), self.frames, C.N_LINKS, self.live, 2
        )
        self.support[..., 0].abs_()
        self.support_mask = torch.ones(
            len(self.support), self.frames, C.N_LINKS, dtype=torch.bool
        )
        self.absence = torch.randn(2, self.frames, C.N_LINKS, self.live, 2)
        self.absence[..., 0].abs_()
        self.absence_mask = torch.ones(
            2, self.frames, C.N_LINKS, dtype=torch.bool
        )

    def _forward(self, model: KPV2ActionPose) -> dict[str, torch.Tensor]:
        """공통 synthetic episode를 모델에 통과시킨다."""
        return model(
            self.query, self.query_mask,
            self.support, self.support_mask, self.support_labels,
            self.absence, self.absence_mask,
        )

    def test_output_contract(self) -> None:
        """분류·위험·pose motion·domain 보조 head shape를 보장한다."""
        model = KPV2ActionPose(
            hidden=16, width=32, domains=4, dropout=0.0,
            progress_bins=8, frequency_bins=4,
        )
        output = self._forward(model)
        self.assertEqual(output["action_logits"].shape, (3, C.N_CLASSES))
        self.assertEqual(output["risk_logits"].shape, (3, C.N_RISK))
        self.assertEqual(output["coarse_logits"].shape, (3, 5))
        self.assertEqual(output["start_logits"].shape, (3, 4))
        self.assertEqual(output["domain_logits"].shape, (3, 4))
        self.assertEqual(output["style_domain_logits"].shape, (3, 4))
        self.assertEqual(
            output["pose_motion"].shape,
            (3, self.frames, MOTION_DESCRIPTOR_DIM),
        )

    def test_motion_grounded_classifier_contract(self) -> None:
        """복원된 motion descriptor가 독립 행동·위험 head로 연결되는지 확인한다."""
        model = KPV2ActionPose(
            hidden=16, width=32, domains=4, dropout=0.0,
            progress_bins=8, frequency_bins=4,
            use_distance_features=False, use_motion_classifier=True,
        )
        output = self._forward(model)
        self.assertEqual(output["motion_action_logits"].shape, (3, C.N_CLASSES))
        self.assertEqual(output["motion_risk_logits"].shape, (3, C.N_RISK))
        self.assertEqual(output["motion_embedding"].shape, (3, 16))
        self.assertTrue(bool((output["motion_fusion"] > 0.0).all()))

    def test_pose_semantic_teacher_and_alignment_contract(self) -> None:
        teacher = PoseSemanticTeacher(hidden=16, bins=4, dropout=0.0)
        descriptor = torch.randn(6, self.frames, MOTION_DESCRIPTOR_DIM)
        valid = torch.ones(6, self.frames, dtype=torch.bool)
        output = teacher(descriptor, valid)
        self.assertEqual(output["embedding"].shape, (6, 16))
        self.assertEqual(output["action_logits"].shape, (6, C.N_CLASSES))
        labels = torch.tensor((0, 0, 1, 1, 2, 2))
        student = torch.randn(6, 16, requires_grad=True)
        loss = cross_modal_supervised_contrastive(
            student, output["embedding"].detach(), labels
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_style_projection_cannot_change_action_logits(self) -> None:
        """환경 style branch가 행동 분류 경로와 직접 연결되지 않았는지 검사한다."""
        model = KPV2ActionPose(
            hidden=16, width=32, domains=4, dropout=0.0,
            progress_bins=8, frequency_bins=4,
        ).eval()
        with torch.no_grad():
            before = self._forward(model)["action_logits"]
            for parameter in model.motion_encoder.style_projection.parameters():
                parameter.add_(torch.randn_like(parameter) * 10.0)
            after = self._forward(model)["action_logits"]
        torch.testing.assert_close(before, after)

    def test_padding_does_not_change_content(self) -> None:
        """유효 동작 뒤의 masked padding 길이가 content를 바꾸지 않게 한다."""
        encoder = ScaleFreeMotionEncoder(
            hidden=16, progress_bins=8, frequency_bins=4, dropout=0.0
        ).eval()
        motion = torch.randn(2, 14, C.N_LINKS, self.live, 6)
        mask = torch.ones(2, 14, C.N_LINKS, dtype=torch.bool)
        padded_motion = torch.cat((
            motion, torch.randn(2, 9, C.N_LINKS, self.live, 6)
        ), dim=1)
        padded_mask = torch.cat((
            mask, torch.zeros(2, 9, C.N_LINKS, dtype=torch.bool)
        ), dim=1)
        with torch.no_grad():
            short = encoder(motion, mask)["content"]
            padded = encoder(padded_motion, padded_mask)["content"]
        torch.testing.assert_close(short, padded, atol=2e-5, rtol=2e-5)

    def test_factory_round_trip(self) -> None:
        """best.pt의 config만으로 동일한 KP v2 구조를 복원한다."""
        source = KPV2ActionPose(
            hidden=16, width=32, domains=3, dropout=0.0,
            progress_bins=8, frequency_bins=4,
        )
        restored = build_calibration_model(source.model_config())
        restored.load_state_dict(source.state_dict(), strict=True)
        self.assertIsInstance(restored, KPV2ActionPose)

    def test_support_relative_energy_round_trip(self) -> None:
        """보정 뒤 남은 움직임 세기 경로가 checkpoint와 출력을 보존한다."""
        source = KPV2ActionPose(
            hidden=16, width=32, domains=3, dropout=0.0,
            progress_bins=8, frequency_bins=4,
            use_support_relative_energy=True,
            use_learned_support_matcher=True,
            use_explicit_support_energy=True,
        ).eval()
        restored = build_calibration_model(source.model_config()).eval()
        restored.load_state_dict(source.state_dict(), strict=True)
        with torch.no_grad():
            expected = self._forward(source)["action_logits"]
            actual = self._forward(restored)["action_logits"]
        torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
