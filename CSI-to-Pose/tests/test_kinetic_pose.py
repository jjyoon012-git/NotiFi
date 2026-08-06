from __future__ import annotations

import torch
from torch import nn

from notifi_pose import contract as C
from notifi_pose.kinetic_pose import KineticDynamicEncoder, KineticPoseResidual
from notifi_pose.nets import PerLinkNorm


class DummyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        batch, frames = csi.shape[:2]
        pose = csi.new_zeros(batch, frames, C.N_JOINTS, 3)
        pose[:, :, C.JOINT_INDEX["head"], 1] = 0.65 + self.anchor
        return {
            "pose_rel": pose,
            "root": csi.new_zeros(batch, frames, 3),
            "class_logits": csi.new_zeros(batch, C.N_CLASSES),
            "risk_logits": csi.new_zeros(batch, C.N_RISK),
        }


def fitted_normalizer() -> PerLinkNorm:
    normalizer = PerLinkNorm()
    normalizer.mu.copy_(torch.randn_like(normalizer.mu))
    normalizer.sigma.copy_(0.5 + torch.rand_like(normalizer.sigma))
    normalizer.fitted.fill_(True)
    return normalizer


def test_dynamic_inputs_ignore_static_per_link_offsets():
    torch.manual_seed(3)
    encoder = KineticDynamicEncoder(
        fitted_normalizer(), hidden=32, temporal_layers=1, heads=4, dropout=0.0
    )
    csi = torch.randn(2, 24, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(2, 24, C.N_LINKS, dtype=torch.bool)
    offset = torch.randn(2, 1, C.N_LINKS, 1, 2)
    original, original_masks = encoder.dynamic_inputs(csi, mask)
    shifted, shifted_masks = encoder.dynamic_inputs(csi + offset, mask)
    for left, right in zip(original, shifted):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    for left, right in zip(original_masks, shifted_masks):
        assert torch.equal(left, right)


def test_strength_zero_is_exact_frozen_baseline():
    torch.manual_seed(5)
    baseline = DummyBaseline()
    model = KineticPoseResidual(
        baseline, fitted_normalizer(), hidden=32, temporal_layers=1,
        heads=4, dropout=0.0,
    )
    csi = torch.randn(2, 20, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(2, 20, C.N_LINKS, dtype=torch.bool)
    expected = baseline(csi, mask)["pose_rel"]
    model.set_residual_strength(0.0)
    actual = model(csi, mask)["pose_rel"]
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert not any(parameter.requires_grad for parameter in model.baseline.parameters())


def test_all_missing_frames_remain_finite_and_zero_activity():
    model = KineticPoseResidual(
        DummyBaseline(), fitted_normalizer(), hidden=32, temporal_layers=1,
        heads=4, dropout=0.0,
    )
    csi = torch.zeros(1, 12, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.zeros(1, 12, C.N_LINKS, dtype=torch.bool)
    output = model(csi, mask)
    assert torch.isfinite(output["pose_rel"]).all()
    assert torch.count_nonzero(output["kinetic_activity"]) == 0


def test_compact_checkpoint_excludes_frozen_baseline():
    model = KineticPoseResidual(
        DummyBaseline(), fitted_normalizer(), hidden=32, temporal_layers=1,
        heads=4, dropout=0.0,
    )
    state = model.trainable_state_dict()
    assert state
    assert all(key.startswith(("dynamic.", "refiner.", "velocity_head.")) for key in state)
    assert not any(key.startswith("baseline.") for key in state)


def test_cached_coarse_pose_path_does_not_require_baseline():
    model = KineticPoseResidual(
        None, fitted_normalizer(), hidden=32, temporal_layers=1,
        heads=4, dropout=0.0,
    )
    csi = torch.randn(1, 10, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    mask = torch.ones(1, 10, C.N_LINKS, dtype=torch.bool)
    coarse = torch.randn(1, 10, C.N_JOINTS, 3)
    coarse = coarse - coarse[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
    model.set_residual_strength(0.0)
    output = model(csi, mask, coarse_pose=coarse)
    torch.testing.assert_close(output["pose_rel"], coarse, atol=0.0, rtol=0.0)


def test_signal_gated_head_is_exact_fallback_for_constant_csi():
    model = KineticPoseResidual(
        None, fitted_normalizer(), hidden=32, temporal_layers=1,
        heads=4, dropout=0.0, condition_on_coarse=False,
        activity_floor=0.0,
    )
    csi = torch.randn(1, 1, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    csi = csi.expand(1, 18, -1, -1, -1).clone()
    mask = torch.ones(1, 18, C.N_LINKS, dtype=torch.bool)
    coarse = torch.randn(1, 18, C.N_JOINTS, 3)
    coarse = coarse - coarse[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
    output = model(csi, mask, coarse_pose=coarse)
    assert torch.count_nonzero(output["kinetic_activity"]) == 0
    torch.testing.assert_close(output["pose_rel"], coarse, atol=0.0, rtol=0.0)
