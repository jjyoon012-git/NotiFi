"""Deployment-style support-conditioned adaptation for a frozen V13S model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .hybrid_v10 import p2_motion_features
from .nets import LocalTemporalBlock
from .seen_v2 import _forward_kinematics, _local_bones
from .seen_v4 import _masked_temporal_mean
from .v3 import rotation_6d_to_matrix


class CalibrationSupportEncoder(nn.Module):
    """Encode absence and instructed-pose CSI statistics into one site token."""

    def __init__(self, profile_features: int = 14, hidden: int = 128):
        super().__init__()
        self.profile_features = int(profile_features)
        self.hidden = int(hidden)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(profile_features),
            nn.Linear(profile_features, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.link_embedding = nn.Parameter(torch.zeros(C.N_LINKS, hidden))
        nn.init.normal_(self.link_embedding, std=0.02)
        self.link_attention = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        if profile.ndim == 3:
            profile = profile.unsqueeze(0)
        if profile.shape[1:3] != (C.N_LINKS, C.N_LIVE_SUBCARRIERS):
            raise ValueError(f"unexpected calibration profile shape {profile.shape}")
        feature = self.point_encoder(profile)
        feature = feature.mean(dim=2) + self.link_embedding[None]
        attention = torch.softmax(self.link_attention(feature).squeeze(-1), dim=-1)
        pooled = (feature * attention[..., None]).sum(dim=1)
        return self.output(pooled)


class SupportConditionedP2(nn.Module):
    """Inject a support-derived FiLM transform before the frozen P2 temporal encoder."""

    def __init__(self, base: nn.Module, profile_features: int = 14,
                 support_hidden: int = 128):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        hidden = int(base.hidden)
        self.support_encoder = CalibrationSupportEncoder(
            profile_features=profile_features, hidden=support_hidden
        )
        self.film = nn.Sequential(
            nn.Linear(support_hidden, support_hidden),
            nn.GELU(),
            nn.Linear(support_hidden, hidden * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.register_buffer(
            "support_mean", torch.zeros(1, 1, profile_features)
        )
        self.register_buffer(
            "support_std", torch.ones(1, 1, profile_features)
        )
        self.register_buffer("strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("active_profile", torch.empty(0), persistent=False)

    @property
    def norm(self):
        return self.base.norm

    @torch.no_grad()
    def set_support_normalization(self, mean: torch.Tensor,
                                  std: torch.Tensor) -> None:
        self.support_mean.copy_(mean.reshape_as(self.support_mean).float())
        self.support_std.copy_(
            std.reshape_as(self.support_std).float().clamp_min(1e-4)
        )

    def set_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("support conditioning strength must be in [0, 1]")
        self.strength.fill_(strength)

    def set_profile(self, profile: torch.Tensor) -> None:
        self.active_profile = profile.to(next(self.parameters()).device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                support_profile: torch.Tensor | None = None) -> dict:
        if support_profile is None:
            if not self.active_profile.numel():
                raise RuntimeError("support profile has not been set")
            support_profile = self.active_profile
        if support_profile.ndim == 3:
            support_profile = support_profile.unsqueeze(0)
        if len(support_profile) == 1 and len(csi) > 1:
            support_profile = support_profile.expand(len(csi), -1, -1, -1)

        normalized_profile = (
            support_profile - self.support_mean
        ) / self.support_std
        token = self.support_encoder(normalized_profile)
        gamma, beta = self.film(token).chunk(2, dim=-1)

        batch, frames = csi.shape[:2]
        x = self.base.norm(csi, link_mask)
        feature = self.base.encoder(x)
        feature = self.base.fusion(feature, link_mask)
        feature = feature * (
            1.0 + self.strength * 0.25 * torch.tanh(gamma[:, None])
        )
        feature = feature + self.strength * 0.25 * torch.tanh(beta[:, None])
        temporal = self.base.temporal(feature)
        pose = self.base.pose_head(temporal).reshape(
            batch, frames, C.N_JOINTS, 3
        )
        root = self.base.root_head(temporal)
        frame_mask = link_mask.any(dim=-1, keepdim=True).to(temporal.dtype)
        pooled = (
            (temporal * frame_mask).sum(1)
            / frame_mask.sum(1).clamp_min(1.0)
        )
        return {
            "pose_rel": pose,
            "root": root,
            "class_logits": self.base.class_head(pooled),
            "risk_logits": self.base.risk_head(pooled),
            "temporal_features": temporal,
            "calibration_support_token": token,
        }


class MomentAlignedSupportConditionedP2(SupportConditionedP2):
    """Align support-conditioned CSI moments before the frozen P2 encoder.

    The alignment uses only standing, sitting, and lying support CSI.  It maps
    each target site's first two moments to a source reference distribution,
    then lets the learned support FiLM handle residual differences.
    """

    def __init__(self, base: nn.Module, profile_features: int = 14,
                 support_hidden: int = 128):
        super().__init__(
            base, profile_features=profile_features,
            support_hidden=support_hidden,
        )
        shape = (C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        self.register_buffer("reference_mean", torch.zeros(shape))
        self.register_buffer("reference_std", torch.ones(shape))
        self.register_buffer(
            "alignment_strength", torch.tensor(1.0), persistent=False
        )

    @staticmethod
    def profile_moments(profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover pooled basic-pose moments from a 14-channel profile."""
        means = torch.stack((
            profile[..., 2:4],
            profile[..., 6:8],
            profile[..., 10:12],
        ), dim=0)
        stds = torch.stack((
            profile[..., 4:6],
            profile[..., 8:10],
            profile[..., 12:14],
        ), dim=0)
        mean = means.mean(dim=0)
        second = (stds.square() + means.square()).mean(dim=0)
        std = (second - mean.square()).clamp_min(1e-8).sqrt()
        return mean, std

    @torch.no_grad()
    def set_reference_statistics(self, mean: torch.Tensor,
                                 std: torch.Tensor) -> None:
        self.reference_mean.copy_(mean.reshape_as(self.reference_mean).float())
        self.reference_std.copy_(
            std.reshape_as(self.reference_std).float().clamp_min(1e-4)
        )

    def set_alignment_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("moment alignment strength must be in [0, 1]")
        self.alignment_strength.fill_(strength)

    def _align(self, csi: torch.Tensor,
               support_profile: torch.Tensor) -> torch.Tensor:
        if not float(self.alignment_strength):
            return csi
        if support_profile.ndim == 4:
            site_mean, site_std = zip(*(
                self.profile_moments(item) for item in support_profile
            ))
            site_mean = torch.stack(site_mean)
            site_std = torch.stack(site_std)
        else:
            site_mean, site_std = self.profile_moments(support_profile)
            site_mean = site_mean.unsqueeze(0)
            site_std = site_std.unsqueeze(0)
        scale = (self.reference_std[None] / site_std.clamp_min(1e-4)).clamp(
            0.5, 2.0
        )
        aligned = (
            (csi - site_mean[:, None]) * scale[:, None]
            + self.reference_mean[None, None]
        )
        return torch.lerp(csi, aligned, self.alignment_strength)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                support_profile: torch.Tensor | None = None) -> dict:
        if support_profile is None:
            if not self.active_profile.numel():
                raise RuntimeError("support profile has not been set")
            support_profile = self.active_profile
        profile = support_profile
        if profile.ndim == 3:
            profile = profile.unsqueeze(0)
        if len(profile) == 1 and len(csi) > 1:
            profile = profile.expand(len(csi), -1, -1, -1)
        aligned = self._align(csi, profile)
        output = super().forward(aligned, link_mask, profile)
        output["calibration_aligned_csi"] = aligned
        return output


class CalibrationAwareV14(nn.Module):
    """Condition bounded V13S residual heads on a deployment support profile.

    The frozen V13S output is an exact fallback at zero strengths. Only this
    adapter is trained; the support set never contains warning or danger trials.
    """

    def __init__(self, base: nn.Module, hidden: int = 128,
                 profile_features: int = 14, dropout: float = 0.08):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.support_encoder = CalibrationSupportEncoder(
            profile_features=profile_features, hidden=hidden
        )
        self.register_buffer(
            "support_mean", torch.zeros(1, 1, profile_features)
        )
        self.register_buffer(
            "support_std", torch.ones(1, 1, profile_features)
        )
        query_size = 96 + 128 + 4
        self.query_projection = nn.Sequential(
            nn.Linear(query_size, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.film = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden * 2)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.support_residual = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.support_residual.weight)
        nn.init.zeros_(self.support_residual.bias)
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS * 6),
        )
        self.root_anchor_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.root_step_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_CLASSES)
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_RISK)
        )
        for head in (
            self.rotation_head,
            self.root_anchor_head,
            self.root_step_head,
            self.class_head,
            self.risk_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer("pose_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("root_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("class_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("risk_strength", torch.tensor(1.0), persistent=False)

    @torch.no_grad()
    def set_support_normalization(self, mean: torch.Tensor,
                                  std: torch.Tensor) -> None:
        self.support_mean.copy_(mean.reshape_as(self.support_mean).float())
        self.support_std.copy_(
            std.reshape_as(self.support_std).float().clamp_min(1e-4)
        )

    def set_strengths(self, pose: float = 1.0, root: float = 1.0,
                      classification: float = 1.0, risk: float = 1.0) -> None:
        values = (pose, root, classification, risk)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("calibration-aware strengths must be in [0, 1]")
        self.pose_strength.fill_(pose)
        self.root_strength.fill_(root)
        self.class_strength.fill_(classification)
        self.risk_strength.fill_(risk)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def _condition(self, base: dict, csi: torch.Tensor,
                   link_mask: torch.Tensor,
                   support_profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if support_profile.ndim == 3:
            support_profile = support_profile.unsqueeze(0)
        if len(support_profile) == 1 and len(csi) > 1:
            support_profile = support_profile.expand(len(csi), -1, -1, -1)
        normalized = (support_profile - self.support_mean) / self.support_std
        token = self.support_encoder(normalized)
        motion = p2_motion_features(csi, link_mask)
        query = torch.cat((
            base["temporal_features"],
            base["temporal_features_v10"],
            motion,
        ), dim=-1)
        feature = self.query_projection(query)
        gamma, beta = self.film(token).chunk(2, dim=-1)
        feature = feature * (1.0 + 0.25 * torch.tanh(gamma[:, None]))
        feature = feature + 0.25 * torch.tanh(beta[:, None])
        feature = feature + self.support_residual(token)[:, None]
        valid = link_mask.any(-1)
        feature = feature * valid[..., None]
        for block in self.temporal:
            feature = block(feature) * valid[..., None]
        return feature, token

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                support_profile: torch.Tensor) -> dict:
        with torch.no_grad():
            base = self.base(csi, link_mask)
        feature, token = self._condition(base, csi, link_mask, support_profile)
        valid = link_mask.any(-1)

        rotation_delta = self.rotation_head(feature).reshape(
            *feature.shape[:2], C.N_JOINTS, 6
        )
        identity = rotation_delta.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        rotation = rotation_6d_to_matrix(rotation_delta + identity)
        bones = _local_bones(base["pose_rel"])
        rotated = torch.matmul(rotation, bones.unsqueeze(-1)).squeeze(-1)
        length = torch.linalg.vector_norm(bones, dim=-1, keepdim=True)
        rotated = F.normalize(rotated, dim=-1) * length
        rotated[:, :, C.ROOT_JOINT] = 0.0
        pose_candidate = _forward_kinematics(rotated)

        pooled = _masked_temporal_mean(feature, valid)
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        step_delta = 0.006 * torch.tanh(self.root_step_head(feature))
        step_delta[:, 0] = 0.0
        step_delta = step_delta * valid[..., None]
        root_candidate = base["root"] + anchor_delta[:, None]
        root_candidate = root_candidate + torch.cumsum(step_delta, dim=1)

        output = dict(base)
        output.update({
            "pose_rel": base["pose_rel"] + self.pose_strength * (
                pose_candidate - base["pose_rel"]
            ),
            "root": base["root"] + self.root_strength * (
                root_candidate - base["root"]
            ),
            "class_logits": base["class_logits"] + self.class_strength * self.class_head(pooled),
            "risk_logits": base["risk_logits"] + self.risk_strength * self.risk_head(pooled),
            "calibration_support_token": token,
            "calibration_rotation_delta": rotation_delta,
            "calibration_root_anchor_delta": anchor_delta,
            "calibration_root_step_delta": step_delta,
        })
        return output
