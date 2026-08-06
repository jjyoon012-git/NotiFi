"""Environment-resistant temporal motion evidence for CAL23-KP10."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C


MOTION_LAGS = (1, 3, 7)


def _shifted_delta(values: torch.Tensor, mask: torch.Tensor,
                   lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.zeros_like(values)
    valid = torch.zeros_like(mask)
    delta[:, lag:] = values[:, lag:] - values[:, :-lag]
    valid[:, lag:] = mask[:, lag:] & mask[:, :-lag]
    return delta, valid


def dynamic_motion_channels(
    csi: torch.Tensor,
    link_mask: torch.Tensor,
    feature_mode: str = "energy",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build gain-resistant motion channels while retaining TX identity.

    The descriptor discards the static subcarrier mean.  It keeps temporal
    differences at three scales, represented by robust mean-absolute and RMS
    energy across subcarriers.  Every channel is standardized within a trial;
    a separate log-energy summary preserves relative motion magnitude.
    """
    if csi.ndim != 5:
        raise ValueError("csi must have shape [B,T,L,F,2]")
    if link_mask.shape != csi.shape[:3]:
        raise ValueError("link_mask shape does not match csi")
    frame_mask = link_mask.any(-1)
    values = torch.nan_to_num(csi.float())
    if feature_mode not in {"energy", "physical_phase"}:
        raise ValueError(f"unknown dynamic feature mode {feature_mode}")
    channels = []
    valid_channels = []
    energy_summary = []
    for lag in MOTION_LAGS:
        delta, valid = _shifted_delta(values, link_mask.bool(), lag)
        weight = valid[..., None, None].to(delta.dtype)
        delta = delta * weight
        mean_abs = delta.abs().mean(-2)
        rms = torch.sqrt(delta.square().mean(-2) + 1e-8)
        if feature_mode == "energy":
            current = torch.cat((mean_abs, rms), dim=-1).flatten(2)
            energy = rms
        else:
            complex_values = torch.view_as_complex(values.contiguous())
            product = torch.zeros_like(complex_values)
            product[:, lag:] = (
                complex_values[:, lag:]
                * complex_values[:, :-lag].conj()
            )
            product = product / (
                complex_values.abs()
                * torch.cat((
                    complex_values[:, :lag].abs(),
                    complex_values[:, :-lag].abs(),
                ), dim=1)
                + 1e-6
            )
            phase = torch.view_as_real(product).mean(-2)
            amplitude = torch.log1p(complex_values.abs())
            amplitude_delta = torch.zeros_like(amplitude)
            amplitude_delta[:, lag:] = (
                amplitude[:, lag:] - amplitude[:, :-lag]
            )
            amplitude_mean = amplitude_delta.mean(-1)
            amplitude_rms = torch.sqrt(
                amplitude_delta.square().mean(-1) + 1e-8
            )
            current = torch.cat((phase, torch.stack((
                amplitude_mean, amplitude_rms,
            ), dim=-1)), dim=-1).flatten(2)
            phase_dispersion = 1.0 - torch.sqrt(
                phase.square().sum(-1).clamp_max(1.0)
            )
            energy = torch.stack((
                amplitude_rms, phase_dispersion,
            ), dim=-1)
        current_valid = valid[..., None].expand(
            -1, -1, -1, current.shape[-1] // C.N_LINKS
        ).flatten(2)
        channels.append(current)
        valid_channels.append(current_valid)
        denominator = valid.sum(1).clamp_min(1).to(delta.dtype)
        energy = (energy * valid[..., None]).sum(1) / denominator[..., None]
        energy_summary.append(torch.log1p(10.0 * energy).flatten(1))
    temporal = torch.cat(channels, dim=-1)
    temporal_valid = torch.cat(valid_channels, dim=-1)
    weight = temporal_valid.to(temporal.dtype)
    count = weight.sum(1).clamp_min(1.0)
    mean = (temporal * weight).sum(1) / count
    variance = ((temporal - mean[:, None]).square() * weight).sum(1) / count
    normalized = (temporal - mean[:, None]) / torch.sqrt(variance[:, None] + 1e-5)
    normalized = normalized * temporal_valid.to(normalized.dtype)
    summary = torch.cat(energy_summary, dim=-1)
    return normalized, summary


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.depthwise = nn.Conv1d(
            width, width, 5, padding=2 * dilation,
            dilation=dilation, groups=width,
        )
        self.pointwise = nn.Conv1d(width, width * 2, 1)
        self.output = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(F.gelu(self.norm(values)))
        values, gate = self.pointwise(values).chunk(2, dim=1)
        values = values * torch.sigmoid(gate)
        return residual + self.dropout(self.output(values))


class DynamicMotionClassifier(nn.Module):
    """Classify action and risk from trial-normalized temporal dynamics."""

    def __init__(self, width: int = 128, dropout: float = 0.12,
                 shape_only: bool = False, feature_mode: str = "energy"):
        super().__init__()
        self.shape_only = bool(shape_only)
        self.feature_mode = feature_mode
        temporal_dim = C.N_LINKS * 2 * 2 * len(MOTION_LAGS)
        summary_dim = C.N_LINKS * 2 * len(MOTION_LAGS)
        self.input = nn.Conv1d(temporal_dim, width, 7, padding=3)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(summary_dim), nn.Linear(summary_dim, width), nn.GELU(),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(width * 3), nn.Linear(width * 3, width), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(width),
        )
        self.action_head = nn.Linear(width, C.N_CLASSES)
        self.risk_head = nn.Linear(width, C.N_RISK)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        temporal, summary = dynamic_motion_channels(
            csi, link_mask, self.feature_mode
        )
        frame_mask = link_mask.any(-1)
        values = self.input(temporal.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        values = values.transpose(1, 2)
        values = values * frame_mask[..., None].to(values.dtype)
        score = self.attention(values).squeeze(-1).masked_fill(~frame_mask, -1e4)
        attended = (values * torch.softmax(score, dim=1)[..., None]).sum(1)
        weight = frame_mask[..., None].to(values.dtype)
        average = (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        summary_feature = self.summary(summary)
        if self.shape_only:
            summary_feature = torch.zeros_like(summary_feature)
        pooled = self.fusion(torch.cat((attended, average, summary_feature), -1))
        return {
            "embedding": pooled,
            "action_logits": self.action_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "frame_features": values,
        }
