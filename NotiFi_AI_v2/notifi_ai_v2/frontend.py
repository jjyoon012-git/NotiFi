"""Differentiable, motion-centered CSI feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .constants import N_IQ, N_LINKS, N_SUBCARRIERS


@dataclass
class FrontendOutput:
    features: torch.Tensor
    valid: torch.Tensor
    phase_quality: torch.Tensor
    activity: torch.Tensor


def _validate(csi: torch.Tensor, link_mask: torch.Tensor) -> None:
    expected = (N_LINKS, N_SUBCARRIERS, N_IQ)
    if csi.ndim != 5 or tuple(csi.shape[2:]) != expected:
        raise ValueError(f"csi must have shape [B,T,{N_LINKS},{N_SUBCARRIERS},2]")
    if link_mask.shape != csi.shape[:3]:
        raise ValueError("link_mask must have shape [B,T,3]")
    if not torch.isfinite(csi).all():
        raise ValueError("csi contains NaN or infinity")


def _masked_time_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    expanded = valid[..., None].expand_as(values)
    masked = values.masked_fill(~expanded, torch.nan)
    center = torch.nanmedian(masked, dim=1, keepdim=True).values
    return torch.nan_to_num(center)


def _temporal_delta(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    pair = valid[:, 1:] & valid[:, :-1]
    result[:, 1:] = (
        values[:, 1:] - values[:, :-1]
    ) * pair[..., None].to(values.dtype)
    return result


class PhysicsMotionFrontend(nn.Module):
    """Suppress static RF fingerprints while preserving temporal motion cues."""

    feature_names = (
        "relative_log_amplitude_spectrum",
        "dynamic_log_amplitude_residual",
        "amplitude_delta",
        "amplitude_acceleration",
        "local_motion_energy",
        "differential_phase_residual",
        "differential_phase_delta",
    )

    def __init__(self, energy_window: int = 9, eps: float = 1e-6):
        super().__init__()
        if energy_window < 1 or energy_window % 2 == 0:
            raise ValueError("energy_window must be a positive odd integer")
        self.energy_window = energy_window
        self.eps = eps

    def forward(
        self,
        csi: torch.Tensor,
        link_mask: torch.Tensor,
        representation: str = "iq",
    ) -> FrontendOutput:
        _validate(csi, link_mask)
        valid = link_mask.to(torch.bool)
        if representation == "iq":
            real, imag = csi[..., 0], csi[..., 1]
            amplitude = torch.sqrt(
                real.square() + imag.square()
            ).clamp_min(self.eps)
            phase = torch.atan2(imag, real)
        elif representation == "amp_phase":
            amplitude = csi[..., 0].clamp_min(self.eps)
            phase = csi[..., 1]
        else:
            raise ValueError("representation must be 'iq' or 'amp_phase'")

        log_amplitude = torch.log(amplitude)
        amp_center = _masked_time_median(log_amplitude, valid)
        dynamic_amplitude = log_amplitude - amp_center
        expanded = valid[..., None].expand_as(log_amplitude)
        packed = log_amplitude.masked_fill(~expanded, torch.nan).permute(
            0, 2, 1, 3
        ).flatten(2)
        global_center = torch.nanmedian(packed, dim=-1).values[:, None, :, None]
        global_center = torch.nan_to_num(global_center)
        relative_spectrum = log_amplitude - global_center
        amp_delta = _temporal_delta(dynamic_amplitude, valid)
        amp_acceleration = _temporal_delta(amp_delta, valid)

        b, t, links, subcarriers = amp_delta.shape
        energy = amp_delta.square().permute(0, 2, 3, 1).reshape(-1, 1, t)
        energy = F.avg_pool1d(
            energy,
            kernel_size=self.energy_window,
            stride=1,
            padding=self.energy_window // 2,
        )
        energy = energy.reshape(b, links, subcarriers, t).permute(0, 3, 1, 2)

        adjacent = phase[..., 1:] - phase[..., :-1]
        adjacent = torch.atan2(torch.sin(adjacent), torch.cos(adjacent))
        adjacent = F.pad(adjacent, (0, 1))
        adjacent = adjacent - adjacent.mean(dim=-1, keepdim=True)
        phase_center = _masked_time_median(adjacent, valid)
        phase_residual = adjacent - phase_center
        phase_delta = _temporal_delta(phase_residual, valid)

        weight = valid[..., None].to(csi.dtype)
        denominator = (
            weight.sum(dim=1).squeeze(-1) * phase_delta.shape[-1]
        ).clamp_min(1.0)
        cosine = (torch.cos(phase_delta) * weight).sum(dim=(1, 3)) / denominator
        sine = (torch.sin(phase_delta) * weight).sum(dim=(1, 3)) / denominator
        phase_quality = torch.sqrt(cosine.square() + sine.square()).clamp(0.0, 1.0)
        phase_weight = phase_quality[:, None, :, None]

        features = torch.stack(
            (
                relative_spectrum,
                dynamic_amplitude,
                amp_delta,
                amp_acceleration,
                torch.sqrt(energy + self.eps),
                phase_residual * phase_weight,
                phase_delta * phase_weight,
            ),
            dim=-1,
        )
        features = features * valid[..., None, None].to(features.dtype)
        activity = (
            amp_delta.abs().mean(dim=-1) * valid.to(amp_delta.dtype)
        ).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1).to(amp_delta.dtype)
        activity = activity * valid.any(dim=-1).to(activity.dtype)
        return FrontendOutput(features, valid, phase_quality, activity)
