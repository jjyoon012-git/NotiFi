"""Dynamic-only CSI refinement for the KineticPose model family."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import (
    LinkAttentionFusion,
    LocalTemporalBlock,
    PerLinkNorm,
    PoseTemporalRefiner,
    TemporalTransformer,
)


def lagged_difference(values: torch.Tensor, link_mask: torch.Tensor,
                      lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a masked temporal difference without crossing missing packets."""
    if lag < 1:
        raise ValueError("lag must be positive")
    difference = torch.zeros_like(values)
    pair_mask = torch.zeros_like(link_mask)
    if values.shape[1] <= lag:
        return difference, pair_mask
    pair_mask[:, lag:] = link_mask[:, lag:] & link_mask[:, :-lag]
    difference[:, lag:] = (values[:, lag:] - values[:, :-lag]) / lag ** 0.5
    difference = difference * pair_mask[..., None, None].to(difference.dtype)
    return difference, pair_mask


def masked_link_average(values: torch.Tensor, link_mask: torch.Tensor,
                        width: int) -> torch.Tensor:
    """Moving average for [B,T,L,S,C], respecting each link independently."""
    if width % 2 == 0:
        raise ValueError("width must be odd")
    batch, frames, links, subcarriers, channels = values.shape
    expanded_mask = (
        link_mask[..., None, None].expand_as(values).to(values.dtype)
    )
    flat_values = values.reshape(batch, frames, -1).transpose(1, 2)
    flat_mask = expanded_mask.reshape(batch, frames, -1).transpose(1, 2)
    numerator = F.avg_pool1d(
        flat_values * flat_mask, width, stride=1, padding=width // 2,
        count_include_pad=False,
    )
    denominator = F.avg_pool1d(
        flat_mask, width, stride=1, padding=width // 2,
        count_include_pad=False,
    )
    averaged = numerator / denominator.clamp_min(1e-6)
    return averaged.transpose(1, 2).reshape(
        batch, frames, links, subcarriers, channels
    )


class MultiScaleSubcarrierEncoder(nn.Module):
    """Encode four two-channel temporal scales in one frequency pass."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        width = max(32, hidden // 2)
        groups = 8 if width % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv1d(8, width, 7, padding=3),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=2),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(width * 2, hidden), nn.LayerNorm(hidden)
        )
        self.link_embedding = nn.Parameter(torch.zeros(C.N_LINKS, hidden))
        nn.init.normal_(self.link_embedding, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, frames, links, subcarriers = values.shape[:4]
        encoded = values.reshape(
            batch * frames * links, subcarriers, 8
        ).transpose(1, 2)
        encoded = self.stem(encoded)
        pooled = torch.cat((encoded.mean(-1), encoded.amax(-1)), dim=-1)
        pooled = self.projection(pooled).reshape(batch, frames, links, -1)
        return pooled + self.link_embedding


class KineticDynamicEncoder(nn.Module):
    """Encode temporal CSI changes while excluding the static CSI level.

    Additive per-link CSI offsets cancel from all four inputs: differences at
    1, 3, and 7 frames plus a 15-frame high-pass residual. The frozen source
    normalizer supplies units only; its mean cannot leak into the differences.
    """

    def __init__(self, normalizer: PerLinkNorm, hidden: int = 64,
                 temporal_layers: int = 1, heads: int = 4,
                 dropout: float = 0.08):
        super().__init__()
        self.hidden = hidden
        self.norm = copy.deepcopy(normalizer)
        for parameter in self.norm.parameters():
            parameter.requires_grad_(False)
        self.frequency_encoder = MultiScaleSubcarrierEncoder(hidden, dropout)
        self.link_fusion = LinkAttentionFusion(
            hidden, heads=heads, dropout=dropout
        )
        self.temporal = TemporalTransformer(
            hidden, temporal_layers, heads, dropout
        )

    def dynamic_inputs(self, csi: torch.Tensor,
                       link_mask: torch.Tensor) -> tuple[list[torch.Tensor],
                                                         list[torch.Tensor]]:
        normalized = self.norm(csi, link_mask)
        features: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for lag in (1, 3, 7):
            difference, pair_mask = lagged_difference(
                normalized, link_mask, lag
            )
            features.append(difference)
            masks.append(pair_mask)
        local_mean = masked_link_average(normalized, link_mask, width=15)
        high_pass = (normalized - local_mean) * link_mask[..., None, None]
        features.append(high_pass)
        masks.append(link_mask)
        return features, masks

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        features, masks = self.dynamic_inputs(csi, link_mask)
        dynamic_mask = torch.stack(masks).any(0)
        mixed = self.frequency_encoder(torch.cat(features, dim=-1))
        mixed = mixed * dynamic_mask[..., None].to(mixed.dtype)
        fused = self.link_fusion(mixed, dynamic_mask)
        frame_mask = link_mask.any(-1)
        temporal = self.temporal(fused, frame_mask)

        first_difference = features[0].abs().mean((-1, -2))
        available = masks[0].to(first_difference.dtype)
        activity = (first_difference * available).sum(-1)
        activity = activity / available.sum(-1).clamp_min(1.0)
        activity = F.max_pool1d(
            activity[:, None], kernel_size=9, stride=1, padding=4
        ).squeeze(1)
        activity = torch.tanh(activity / 1.5)
        activity = activity * frame_mask.to(activity.dtype)
        return temporal, activity


class DynamicResidualHead(nn.Module):
    """Decode a pose residual without seeing the frozen coarse pose."""

    def __init__(self, hidden: int, dropout: float, max_delta: float,
                 joint_scale: list[float]):
        super().__init__()
        self.max_delta = max_delta
        self.register_buffer(
            "joint_scale", torch.tensor(joint_scale), persistent=False
        )
        self.blocks = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, temporal: torch.Tensor,
                frame_mask: torch.Tensor) -> torch.Tensor:
        features = temporal
        for block in self.blocks:
            features = block(features)
        delta = self.max_delta * torch.tanh(self.head(features))
        delta = delta.reshape(*temporal.shape[:2], C.N_JOINTS, 3)
        delta = delta * self.joint_scale[None, None, :, None]
        delta = delta * frame_mask[:, :, None, None].to(delta.dtype)
        return delta - delta[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]


class KineticPoseResidual(nn.Module):
    """Refine a frozen coarse pose using only dynamic CSI evidence."""

    def __init__(self, baseline: nn.Module | None, normalizer: PerLinkNorm,
                 hidden: int = 64, temporal_layers: int = 1,
                 heads: int = 4, dropout: float = 0.08,
                 max_delta: float = 0.25,
                 condition_on_coarse: bool = True,
                 activity_floor: float = 0.15):
        super().__init__()
        self.baseline = baseline
        self.dynamic = KineticDynamicEncoder(
            normalizer, hidden=hidden, temporal_layers=temporal_layers,
            heads=heads, dropout=dropout,
        )
        joint_scale = torch.ones(C.N_JOINTS)
        distal = set(
            C.JOINT_GROUPS["head"]
            + C.JOINT_GROUPS["left_arm"][-1:]
            + C.JOINT_GROUPS["right_arm"][-1:]
            + C.JOINT_GROUPS["left_leg"][-2:]
            + C.JOINT_GROUPS["right_leg"][-2:]
        )
        for joint in distal:
            joint_scale[joint] = 1.20
        self.condition_on_coarse = bool(condition_on_coarse)
        self.activity_floor = float(activity_floor)
        if self.condition_on_coarse:
            self.refiner = PoseTemporalRefiner(
                hidden, dropout=dropout, max_delta=max_delta,
                joint_scale=joint_scale.tolist(),
            )
        else:
            self.refiner = DynamicResidualHead(
                hidden, dropout, max_delta, joint_scale.tolist()
            )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        self.register_buffer("residual_strength", torch.tensor(1.0))
        self.register_buffer("activity_threshold", torch.tensor(0.0))
        if self.baseline is not None:
            for parameter in self.baseline.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.baseline is not None:
            self.baseline.eval()
        self.dynamic.norm.eval()
        return self

    def set_residual_strength(self, strength: float) -> None:
        self.residual_strength.fill_(float(strength))

    def set_activity_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold < 1.0:
            raise ValueError("activity threshold must be in [0, 1)")
        self.activity_threshold.fill_(float(threshold))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(("dynamic.", "refiner.", "velocity_head."))
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"unknown KineticPose weights: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor | None = None) -> dict:
        if coarse_pose is None:
            if self.baseline is None:
                raise ValueError("coarse_pose is required without a frozen baseline")
            with torch.no_grad():
                baseline = self.baseline(csi, link_mask)
            coarse = baseline["pose_rel"]
        else:
            coarse = coarse_pose
            baseline = {
                "pose_rel": coarse,
                "root": coarse.new_zeros(*coarse.shape[:2], 3),
            }
        frame_mask = link_mask.any(-1)
        temporal, activity = self.dynamic(csi, link_mask)
        if self.condition_on_coarse:
            refined = self.refiner(coarse, temporal, frame_mask)
            raw_delta = refined - coarse
        else:
            raw_delta = self.refiner(temporal, frame_mask)
        calibrated_activity = (
            (activity - self.activity_threshold).clamp_min(0.0)
            / (1.0 - self.activity_threshold).clamp_min(1e-6)
        )
        gate = (
            self.activity_floor
            + (1.0 - self.activity_floor) * calibrated_activity
        )
        effective_delta = raw_delta * gate[..., None, None]
        pose = coarse + self.residual_strength * effective_delta
        pose = pose - pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        output = dict(baseline)
        output.update({
            "pose_coarse": coarse,
            "pose_delta": effective_delta,
            "pose_rel": pose,
            "kinetic_activity": activity,
            "kinetic_features": temporal,
            "kinetic_velocity": (
                self.velocity_head(temporal).reshape(
                    *temporal.shape[:2], C.N_JOINTS, 3
                ) * activity[..., None, None]
            ),
        })
        return output
