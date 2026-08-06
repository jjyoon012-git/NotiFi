import torch

from notifi_pose import contract as C
from notifi_pose.cal23_kp10 import DynamicMotionClassifier, dynamic_motion_channels


def test_dynamic_channels_are_finite_and_identity_preserving():
    csi = torch.randn(2, 40, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(2, 40, C.N_LINKS, dtype=torch.bool)
    temporal, summary = dynamic_motion_channels(csi, mask)
    assert temporal.shape == (2, 40, C.N_LINKS * 12)
    assert summary.shape == (2, C.N_LINKS * 6)
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(summary).all()


def test_dynamic_channels_remove_constant_static_profile():
    csi = torch.randn(1, 32, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(1, 32, C.N_LINKS, dtype=torch.bool)
    static = torch.randn(1, 1, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2) * 50.0
    first, _ = dynamic_motion_channels(csi, mask)
    second, _ = dynamic_motion_channels(csi + static, mask)
    assert torch.allclose(first, second, atol=2e-4, rtol=2e-4)


def test_dynamic_model_output_contract():
    model = DynamicMotionClassifier(width=32)
    csi = torch.randn(2, 40, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(2, 40, C.N_LINKS, dtype=torch.bool)
    output = model(csi, mask)
    assert output["action_logits"].shape == (2, C.N_CLASSES)
    assert output["risk_logits"].shape == (2, C.N_RISK)
    assert output["embedding"].shape == (2, 32)


def test_physical_phase_channels_are_finite():
    csi = torch.randn(2, 40, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(2, 40, C.N_LINKS, dtype=torch.bool)
    temporal, summary = dynamic_motion_channels(
        csi, mask, feature_mode="physical_phase"
    )
    assert temporal.shape == (2, 40, C.N_LINKS * 12)
    assert summary.shape == (2, C.N_LINKS * 6)
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(summary).all()


def test_physical_phase_ignores_static_link_phase_rotation():
    csi = torch.randn(2, 40, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(2, 40, C.N_LINKS, dtype=torch.bool)
    angle = torch.linspace(-1.2, 1.4, C.N_LINKS)
    complex_csi = torch.view_as_complex(csi.contiguous())
    rotated = torch.view_as_real(
        complex_csi * torch.exp(1j * angle)[None, None, :, None]
    )
    first = dynamic_motion_channels(csi, mask, feature_mode="physical_phase")
    second = dynamic_motion_channels(rotated, mask, feature_mode="physical_phase")
    assert torch.allclose(first[0], second[0], atol=5e-5, rtol=5e-5)
    assert torch.allclose(first[1], second[1], atol=5e-5, rtol=5e-5)


def test_physical_phase_masks_missing_link_and_nan():
    csi = torch.randn(1, 40, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(1, 40, C.N_LINKS, dtype=torch.bool)
    mask[:, :, 0] = False
    csi[:, :, 0] = float("nan")
    temporal, summary = dynamic_motion_channels(
        csi, mask, feature_mode="physical_phase"
    )
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(summary).all()
