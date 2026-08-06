import torch

from notifi_pose.cal16_kp10 import (
    apply_dynamic_spectrum_calibration,
    assess_spectrum_calibration,
    fit_identity_spectrum_calibration,
    fit_site_balanced_reference,
    trial_dynamic_spectrum,
)


def test_dynamic_spectrum_ignores_constant_offsets():
    csi = torch.randn(2, 40, 3, 12, 2)
    mask = torch.ones(2, 40, 3, dtype=torch.bool)
    offset = torch.randn(2, 1, 3, 12, 2)
    assert torch.allclose(
        trial_dynamic_spectrum(csi, mask),
        trial_dynamic_spectrum(csi + offset, mask), atol=1e-5,
    )


def test_calibration_preserves_link_order_and_low_frequency_level():
    csi = torch.zeros(1, 64, 3, 8, 2)
    time = torch.arange(64).float()
    csi[0, :, 0, :, 0] = torch.sin(time)[:, None]
    csi[0, :, 1, :, 0] = 10.0 + 0.1 * torch.sin(time)[:, None]
    csi[0, :, 2, :, 0] = -7.0
    mask = torch.ones(1, 64, 3, dtype=torch.bool)
    gain = torch.ones(3, 8, 2)
    gain[0, :, 0] = 1.5
    output = apply_dynamic_spectrum_calibration(
        csi, mask, gain, 1.0, lowpass_window=15
    )
    assert output[0, :, 0, :, 0].std() > csi[0, :, 0, :, 0].std()
    assert torch.allclose(output[0, :, 2], csi[0, :, 2], atol=1e-5)
    assert abs(float(output[0, :, 1, :, 0].mean()) - 10.0) < 0.02


def test_site_balanced_fit_recovers_bounded_gain():
    labels = torch.tensor([0, 0, 1, 1])
    sites = torch.tensor([0, 1, 0, 1])
    source = torch.zeros(4, 3, 3, 12, 2)
    reference = fit_site_balanced_reference(
        source, labels, sites, classes=(0, 1)
    )
    target = source - torch.log(torch.tensor(1.5))
    result = fit_identity_spectrum_calibration(
        reference, target, labels, max_gain=1.8, smoothing_width=3
    )
    assert torch.allclose(result["gain"], torch.full_like(result["gain"], 1.5), atol=1e-4)
    assert result["boundary_fraction"] == 0.0


def test_quality_gate_rejects_missing_link():
    audit = {"boundary_fraction": 0.0, "relative_improvement": 0.5}
    decision = assess_spectrum_calibration(audit, [1.0, 0.4, 1.0])
    assert decision.status == "REJECT"
