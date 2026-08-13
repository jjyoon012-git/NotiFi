"""Motion-centered shared-link encoder for unseen-domain development."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .constants import N_ACTIONS, N_RISKS
from .frontend import PhysicsMotionFrontend
from .geometry import InstallationGeometry


@dataclass(frozen=True)
class MotionEncoderConfig:
    hidden: int = 96
    temporal_layers: int = 4
    dropout: float = 0.10
    motion_targets: int = 8


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden,
        )
        self.mix = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values)
        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        values = self.mix(values)
        return (residual + values) * valid[..., None].to(values.dtype)


class SharedLinkEncoder(nn.Module):
    """Apply identical subcarrier and temporal weights to every CSI link."""

    def __init__(self, feature_dim: int, config: MotionEncoderConfig):
        super().__init__()
        hidden = config.hidden
        self.frequency = nn.Sequential(
            nn.Conv1d(feature_dim, hidden // 2, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal = nn.ModuleList(
            ResidualTemporalBlock(hidden, 2 ** index, config.dropout)
            for index in range(config.temporal_layers)
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, links, subcarriers, channels = features.shape
        packed = features.permute(0, 1, 2, 4, 3).reshape(
            b * t * links, channels, subcarriers
        )
        encoded = self.frequency(packed).squeeze(-1).reshape(b, t, links, -1)
        encoded = encoded * valid[..., None].to(encoded.dtype)
        per_link = encoded.permute(0, 2, 1, 3).reshape(b * links, t, -1)
        link_valid = valid.permute(0, 2, 1).reshape(b * links, t)
        for block in self.temporal:
            per_link = block(per_link, link_valid)
        return per_link.reshape(b, links, t, -1).permute(0, 2, 1, 3)


class GeometryAwareLinkFusion(nn.Module):
    """Fuse available links with geometry-conditioned masked attention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.geometry = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = values + self.geometry(geometry)[None, None]
        logits = self.score(conditioned).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e4)
        maximum = logits.max(dim=-1, keepdim=True).values
        weight = torch.exp(logits - maximum) * valid.to(logits.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = (conditioned * weight[..., None]).sum(dim=2)
        frame_valid = valid.any(dim=-1)
        fused = self.output(fused) * frame_valid[..., None].to(fused.dtype)
        return fused, weight


class MotionCalibratedEncoder(nn.Module):
    """Produce action, risk, and dense motion predictions from CSI only."""

    def __init__(self, config: MotionEncoderConfig = MotionEncoderConfig()):
        super().__init__()
        self.config = config
        self.frontend = PhysicsMotionFrontend()
        self.link_encoder = SharedLinkEncoder(6, config)
        self.fusion = GeometryAwareLinkFusion(config.hidden)
        self.trunk = nn.ModuleList(
            ResidualTemporalBlock(config.hidden, 2 ** index, config.dropout)
            for index in range(3)
        )
        self.action_head = nn.Linear(config.hidden, N_ACTIONS)
        self.risk_head = nn.Linear(config.hidden, N_RISKS)
        self.motion_head = nn.Linear(config.hidden, config.motion_targets)

    def forward(
        self,
        csi: torch.Tensor,
        link_mask: torch.Tensor,
        geometry: InstallationGeometry | None = None,
    ) -> dict[str, torch.Tensor]:
        frontend = self.frontend(csi, link_mask)
        per_link = self.link_encoder(frontend.features, frontend.valid)
        geometry = geometry or InstallationGeometry.cardinal_default()
        vectors = geometry.normalized_tensor(device=csi.device, dtype=csi.dtype)
        sequence, link_weight = self.fusion(per_link, frontend.valid, vectors)
        frame_valid = frontend.valid.any(dim=-1)
        for block in self.trunk:
            sequence = block(sequence, frame_valid)
        weight = frame_valid[..., None].to(sequence.dtype)
        pooled = (sequence * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return {
            "action_logits": self.action_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "motion": self.motion_head(sequence) * weight,
            "embedding": pooled,
            "sequence": sequence,
            "link_weight": link_weight,
            "frame_valid": frame_valid,
            "activity": frontend.activity,
            "phase_quality": frontend.phase_quality,
        }
