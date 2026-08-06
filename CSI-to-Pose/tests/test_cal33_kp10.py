import torch

from notifi_pose.cal33_kp10 import (
    META_RISK_FEATURES,
    MetaRiskHead,
    apply_risk_group_gate,
    build_safe_context,
    meta_risk_features,
)
from notifi_pose.calibration_quality import SAFE_CALIBRATION_CLASSES


def test_meta_risk_features_are_support_relative_and_finite():
    labels = torch.tensor(list(SAFE_CALIBRATION_CLASSES) * 2)
    support = torch.randn(len(labels), 32)
    risk = torch.randn(len(labels), 3)
    context = build_safe_context(support, risk, labels)
    features = meta_risk_features(torch.randn(5, 32), torch.randn(5, 3), context)
    assert features.shape == (5, META_RISK_FEATURES)
    assert torch.isfinite(features).all()


def test_meta_risk_head_shape():
    output = MetaRiskHead()(torch.randn(4, META_RISK_FEATURES))
    assert output.shape == (4, 3)


def test_risk_group_gate_respects_predicted_group():
    logits = torch.randn(3, 17)
    gated = apply_risk_group_gate(logits, torch.tensor((0, 1, 2)))
    prediction = gated.argmax(-1)
    assert prediction[0] < 9
    assert 9 <= prediction[1] < 12
    assert prediction[2] >= 12
