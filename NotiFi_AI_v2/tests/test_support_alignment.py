"""Support ridge alignment의 수치 안정성과 방향 처리를 검증한다."""

import unittest

import torch

from notifi_ai_v2.support_alignment import (
    action_to_risk_log_probability,
    aligned_logits,
    apply_affine_map,
    identity_ridge_map,
)


class SupportAlignmentTest(unittest.TestCase):
    """작은 support에서 affine calibration이 정상 동작하는지 확인한다."""

    def test_identity_pairs_keep_embeddings_close(self) -> None:
        """source와 target이 같으면 identity에 가까운 mapping을 만든다."""
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(14, 8, generator=generator)
        mapping = identity_ridge_map(values, values, regularization=1.0)
        mapped = apply_affine_map(values, mapping)
        expected = torch.nn.functional.normalize(values, dim=-1)
        self.assertLess(float((mapped - expected).abs().max()), 1e-4)

    def test_alignment_returns_all_class_logits(self) -> None:
        """두 정렬 방향 모두 query별 17개 logit을 반환한다."""
        generator = torch.Generator().manual_seed(11)
        anchors = torch.randn(9, 8, generator=generator)
        danger = torch.randn(5, 8, generator=generator)
        library = [{
            "anchors": anchors.clone(),
            "classes": torch.randn(17, 8, generator=generator),
        }]
        query = torch.randn(4, 8, generator=generator)
        for direction in ("source_to_target", "target_to_source"):
            logits = aligned_logits(
                query,
                anchors,
                danger,
                library,
                regularization=1.0,
                prototype_temperature=0.1,
                site_temperature=0.1,
                direction=direction,
            )
            self.assertEqual(tuple(logits.shape), (4, 17))
            self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_unnormalized_mapping_preserves_scale(self) -> None:
        """motion signature용 mapping은 벡터 크기를 정규화하지 않고 보존한다."""
        source = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]])
        target = source * 2.0 + 0.5
        mapping = identity_ridge_map(
            source, target, regularization=1e-4, normalize_inputs=False
        )
        mapped = apply_affine_map(source, mapping, normalize_output=False)
        self.assertLess(float((mapped - target).abs().max()), 2e-3)

    def test_warning_support_extends_alignment_pairs(self) -> None:
        """warning support를 추가해도 17개 행동 logit 계약을 유지한다."""
        generator = torch.Generator().manual_seed(7)
        query = torch.randn(4, 8, generator=generator)
        anchors = torch.randn(9, 8, generator=generator)
        warning = torch.randn(3, 8, generator=generator)
        danger = torch.randn(5, 8, generator=generator)
        source = [{
            "classes": torch.randn(17, 8, generator=generator),
            "anchors": torch.randn(9, 8, generator=generator),
        }]
        logits = aligned_logits(
            query, anchors, danger, source, 1.0, 0.1, 0.1,
            target_warning=warning,
        )
        self.assertEqual(tuple(logits.shape), (4, 17))

    def test_action_risk_probabilities_sum_to_one(self) -> None:
        """행동 확률을 위험 그룹으로 합쳐도 전체 확률 질량을 보존한다."""
        action = torch.randn(5, 17)
        risk = action_to_risk_log_probability(action)
        self.assertTrue(torch.allclose(
            risk.exp().sum(-1), torch.ones(5), atol=1e-6
        ))



if __name__ == "__main__":
    unittest.main()
