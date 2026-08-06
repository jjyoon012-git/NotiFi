"""Support-conditioned feature calibration for CAL1-KP10."""

from __future__ import annotations

import torch
from torch import nn


SAFE_SUPPORT_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)


def _masked_moments(features: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Summarize level and motion without depending on trial duration."""
    weight = mask[..., None].to(features.dtype)
    count = weight.sum(-2).clamp_min(1.0)
    mean = (features * weight).sum(-2) / count
    centered = (features - mean[..., None, :]) * weight
    std = torch.sqrt(centered.square().sum(-2) / count + 1e-6)

    delta = torch.zeros_like(features)
    delta[..., 1:, :] = features[..., 1:, :] - features[..., :-1, :]
    delta_mask = mask.clone()
    delta_mask[..., 0] = False
    delta_mask[..., 1:] &= mask[..., :-1]
    delta_weight = delta_mask[..., None].to(features.dtype)
    delta_count = delta_weight.sum(-2).clamp_min(1.0)
    delta_mean = (delta * delta_weight).sum(-2) / delta_count
    delta_abs = (delta.abs() * delta_weight).sum(-2) / delta_count
    return torch.cat((mean, std, delta_mean, delta_abs), dim=-1)


class CalibrationSetEncoder(nn.Module):
    """Encode labeled safe support trials into one environment token."""

    def __init__(self, feature_dim: int = 128, token_dim: int = 96,
                 prompt_dim: int = 24, heads: int = 4,
                 layers: int = 2, dropout: float = 0.08):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.prompt = nn.Embedding(17, prompt_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(feature_dim * 4 + prompt_dim),
            nn.Linear(feature_dim * 4 + prompt_dim, token_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads,
            dim_feedforward=token_dim * 3, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.set_attention = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(token_dim)
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim // 2),
            nn.Tanh(), nn.Linear(token_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim),
            nn.GELU(), nn.LayerNorm(token_dim),
        )

    def forward(self, support_features: torch.Tensor,
                support_mask: torch.Tensor,
                support_class: torch.Tensor) -> torch.Tensor:
        unbatched = support_features.ndim == 3
        if unbatched:
            support_features = support_features[None]
            support_mask = support_mask[None]
            support_class = support_class[None]
        if support_features.ndim != 4:
            raise ValueError("support_features must have shape [B,S,T,D]")
        if support_features.shape[-1] != self.feature_dim:
            raise ValueError("support feature dimension does not match encoder")
        descriptor = _masked_moments(support_features, support_mask)
        values = self.input(torch.cat((
            descriptor, self.prompt(support_class.long()),
        ), dim=-1))
        values = self.set_attention(values)
        score = self.pool(values).squeeze(-1)
        token = (values * torch.softmax(score, dim=-1)[..., None]).sum(-2)
        token = self.output(token)
        return token[0] if unbatched else token


class Cal1KP10Adapter(nn.Module):
    """Apply a bounded support-conditioned residual to frozen KP10 features."""

    def __init__(self, feature_dim: int = 128, token_dim: int = 96,
                 rank: int = 64, dropout: float = 0.08):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.support_encoder = CalibrationSetEncoder(
            feature_dim=feature_dim, token_dim=token_dim,
            dropout=dropout,
        )
        self.query_norm = nn.LayerNorm(feature_dim)
        self.dynamic = nn.Sequential(
            nn.Linear(feature_dim * 3, rank), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(rank, feature_dim),
        )
        self.film = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, feature_dim * 2),
        )
        self.dynamic_gate = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, feature_dim),
        )
        self._reset_identity()

    def _reset_identity(self) -> None:
        nn.init.zeros_(self.dynamic[-1].weight)
        nn.init.zeros_(self.dynamic[-1].bias)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.dynamic_gate[-1].weight)
        nn.init.zeros_(self.dynamic_gate[-1].bias)

    def encode_support(self, support_features: torch.Tensor,
                       support_mask: torch.Tensor,
                       support_class: torch.Tensor) -> torch.Tensor:
        return self.support_encoder(
            support_features, support_mask, support_class
        )

    def adapt(self, features: torch.Tensor, frame_mask: torch.Tensor,
              token: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if float(strength) == 0.0:
            return features
        if token.ndim == 1:
            token = token[None].expand(len(features), -1)
        normalized = self.query_norm(features)
        delta1 = torch.zeros_like(normalized)
        delta3 = torch.zeros_like(normalized)
        delta1[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        delta3[:, 3:] = normalized[:, 3:] - normalized[:, :-3]
        dynamic = self.dynamic(torch.cat((normalized, delta1, delta3), dim=-1))
        scale, bias = self.film(token).chunk(2, dim=-1)
        gate = torch.sigmoid(self.dynamic_gate(token))
        residual = (
            torch.tanh(scale)[:, None] * normalized
            + 0.25 * torch.tanh(bias)[:, None]
            + gate[:, None] * dynamic
        )
        residual = residual * frame_mask[..., None].to(residual.dtype)
        return features + float(strength) * residual

    def forward(self, query_features: torch.Tensor,
                query_mask: torch.Tensor,
                support_features: torch.Tensor,
                support_mask: torch.Tensor,
                support_class: torch.Tensor,
                strength: float = 1.0) -> dict[str, torch.Tensor]:
        token = self.encode_support(
            support_features, support_mask, support_class
        )
        adapted = self.adapt(query_features, query_mask, token, strength)
        return {"features": adapted, "calibration_token": token}
