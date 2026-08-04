"""Temporal denoising motion prior used by fall reconstruction V9C."""

from __future__ import annotations

import torch
from torch import nn

from . import contract as C
from .nets import LocalTemporalBlock


def corrupt_motion(pose: torch.Tensor, valid: torch.Tensor,
                   noise_std: float = 0.025,
                   frame_drop: float = 0.08,
                   joint_drop: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy, locally missing observations without changing the target."""
    corrupted = pose + noise_std * torch.randn_like(pose)
    observed = valid[..., None].expand(-1, -1, C.N_JOINTS).clone()

    frame_missing = (
        torch.rand(*valid.shape, device=pose.device) < frame_drop
    ) & valid
    previous = torch.cat((pose[:, :1], pose[:, :-1]), dim=1)
    corrupted = torch.where(
        frame_missing[..., None, None], previous, corrupted
    )
    observed &= ~frame_missing[..., None]

    joint_missing = (
        torch.rand(*observed.shape, device=pose.device) < joint_drop
    ) & valid[..., None]
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            corrupted[:, :, child] = torch.where(
                joint_missing[:, :, child, None],
                corrupted[:, :, parent], corrupted[:, :, child],
            )
    observed &= ~joint_missing
    corrupted = corrupted * valid[..., None, None]
    return corrupted, observed


class TemporalMotionDenoiser(nn.Module):
    """Denoise a complete pose sequence using long temporal context."""

    def __init__(self, hidden: int = 128, dropout: float = 0.05):
        super().__init__()
        input_size = C.N_JOINTS * 4
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, 1)
        self.correction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS * 3),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, pose: torch.Tensor, valid: torch.Tensor,
                observed: torch.Tensor | None = None) -> dict:
        valid = valid.bool()
        if observed is None:
            observed = valid[..., None].expand(-1, -1, C.N_JOINTS)
        values = torch.cat((
            pose, observed.to(pose.dtype)[..., None]
        ), dim=-1).flatten(-2)
        feature = self.input_projection(values) * valid[..., None]
        for block in self.temporal:
            feature = block(feature) * valid[..., None]
        feature = self.context(
            feature, src_key_padding_mask=~valid
        ) * valid[..., None]
        correction = 0.35 * torch.tanh(
            self.correction_head(feature).reshape_as(pose)
        )
        reconstructed = pose + correction
        reconstructed = reconstructed - reconstructed[
            :, :, C.ROOT_JOINT:C.ROOT_JOINT + 1
        ]
        reconstructed = reconstructed * valid[..., None, None]
        return {
            "pose_rel": reconstructed,
            "pose_correction": correction,
            "motion_prior_features": feature,
        }


class MotionPriorTrajectoryWrapper(nn.Module):
    """Apply a frozen denoising prior to a frozen CSI trajectory model."""

    def __init__(self, base: nn.Module, prior: TemporalMotionDenoiser):
        super().__init__()
        self.base = base
        self.prior = prior
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("prior_strength", torch.tensor(1.0), persistent=False)

    def set_prior_strength(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("prior strength must be between 0 and 1")
        self.prior_strength.fill_(value)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            output = self.base(csi, link_mask)
            valid = link_mask.any(-1)
            prior = self.prior(output["pose_rel"], valid)
        result = dict(output)
        result["pose_before_prior"] = output["pose_rel"]
        result["pose_rel"] = output["pose_rel"] + self.prior_strength * (
            prior["pose_rel"] - output["pose_rel"]
        )
        result["motion_prior_correction"] = prior["pose_correction"]
        return result
