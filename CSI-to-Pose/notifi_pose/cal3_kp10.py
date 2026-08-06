"""Target-safe feature adaptation for CAL3-KP10."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SafeSupportFeatureAdapter(nn.Module):
    """Small site-specific transform fitted only on instructed safe actions."""

    def __init__(self, feature_dim: int = 128, rank: int = 12,
                 max_scale: float = 0.20, max_bias: float = 0.20,
                 max_dynamic: float = 0.15, max_low_rank: float = 0.12,
                 smooth_window: int = 15):
        super().__init__()
        if smooth_window % 2 != 1:
            raise ValueError("smooth_window must be odd")
        self.feature_dim = int(feature_dim)
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.max_dynamic = float(max_dynamic)
        self.max_low_rank = float(max_low_rank)
        self.smooth_window = int(smooth_window)
        self.scale = nn.Parameter(torch.zeros(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.dynamic = nn.Parameter(torch.zeros(feature_dim))
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def _highpass(self, features: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool1d(
            features.transpose(1, 2), self.smooth_window,
            stride=1, padding=self.smooth_window // 2,
        ).transpose(1, 2)
        return features - smooth

    def parameter_penalty(self) -> torch.Tensor:
        return (
            self.scale.square().mean() + self.bias.square().mean()
            + self.dynamic.square().mean()
            + self.down.weight.square().mean()
            + self.up.weight.square().mean()
        )

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor,
                strength: float = 1.0) -> torch.Tensor:
        if float(strength) == 0.0:
            return features
        scale = self.max_scale * torch.tanh(self.scale)
        bias = self.max_bias * torch.tanh(self.bias)
        dynamic = self.max_dynamic * torch.tanh(self.dynamic)
        highpass = self._highpass(features)
        low_rank = self.up(torch.tanh(self.down(features)))
        residual = (
            scale[None, None] * features + bias[None, None]
            + dynamic[None, None] * highpass
            + self.max_low_rank * torch.tanh(low_rank)
        )
        residual = residual * frame_mask[..., None].to(residual.dtype)
        return features + float(strength) * residual


__all__ = ["SafeSupportFeatureAdapter"]
