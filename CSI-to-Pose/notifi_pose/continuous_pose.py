"""Continuous-latent CSI-to-pose model for the KP2 family."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .doppler_pose import DopplerMotionEncoder
from .nets import LocalTemporalBlock


class CSILatentPoseRegressor(nn.Module):
    """Fuse frozen P2 posture context with Doppler motion evidence."""

    def __init__(self, base_model: nn.Module, motion_decoder: nn.Module,
                 latent_mean: torch.Tensor, latent_std: torch.Tensor,
                 bone_lengths: torch.Tensor, hidden: int = 128,
                 code_dim: int = 128, temporal_layers: int = 2,
                 heads: int = 4, dropout: float = 0.08):
        super().__init__()
        self.base = base_model
        self.decoder = copy.deepcopy(motion_decoder)
        for module in (self.base, self.decoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        base_hidden = int(base_model.hidden)
        self.dynamic = DopplerMotionEncoder(
            base_model.norm, hidden=hidden, temporal_layers=temporal_layers,
            heads=heads, dropout=dropout,
        )
        self.static_projection = nn.Sequential(
            nn.LayerNorm(base_hidden), nn.Linear(base_hidden, hidden), nn.GELU()
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid()
        )
        self.fusion_residual = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU()
        )
        self.refiner = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.latent_head = nn.Conv1d(
            hidden, code_dim, kernel_size=4, stride=2, padding=1
        )
        self.register_buffer("latent_mean", latent_mean.float().reshape(1, 1, -1))
        self.register_buffer("latent_std", latent_std.float().reshape(1, 1, -1))
        self.register_buffer("bone_lengths", bone_lengths.float().reshape(1, -1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.decoder.eval()
        return self

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = (
            "dynamic.", "static_projection.", "fusion_gate.",
            "fusion_residual.", "refiner.", "latent_head.",
        )
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"unknown continuous-pose weights: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            static = self.base(csi, link_mask)["temporal_features"]
        dynamic, activity = self.dynamic(csi, link_mask)
        static = self.static_projection(static)
        joined = torch.cat((static, dynamic), dim=-1)
        gate = self.fusion_gate(joined)
        features = gate * static + (1.0 - gate) * dynamic
        features = features + self.fusion_residual(joined)
        frame_mask = link_mask.any(-1)
        features = features * frame_mask[..., None].to(features.dtype)
        for block in self.refiner:
            features = block(features) * frame_mask[..., None].to(features.dtype)
        normalized_latent = self.latent_head(
            features.transpose(1, 2)
        ).transpose(1, 2)
        latent = normalized_latent * self.latent_std + self.latent_mean
        lengths = self.bone_lengths.expand(len(csi), -1)
        decoded = self.decoder(latent, lengths, csi.shape[1], frame_mask)
        return {
            **decoded,
            "csi_pose_features": features,
            "normalized_motion_latent": normalized_latent,
            "motion_latent": latent,
            "kinetic_activity": activity,
            "fusion_gate": gate,
        }
