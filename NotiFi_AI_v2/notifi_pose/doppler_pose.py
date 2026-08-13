"""Doppler-aware CSI motion encoding for the KP2 model family."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .kinetic_pose import KineticDynamicEncoder, KineticPoseResidual, lagged_difference
from .nets import LinkAttentionFusion, PerLinkNorm, TemporalTransformer


def _hann_filter_bank(width: int, cycles: tuple[float, ...]) -> torch.Tensor:
    """Create zero-mean, unit-energy local sine/cosine Doppler filters."""
    position = torch.arange(width, dtype=torch.float32)
    window = torch.hann_window(width, periodic=False)
    centered = position - (width - 1) / 2
    filters = []
    for cycle in cycles:
        angle = 2.0 * math.pi * cycle * centered / width
        for wave in (torch.sin(angle), torch.cos(angle)):
            kernel = wave * window
            kernel = kernel - kernel.mean()
            filters.append(kernel / kernel.square().sum().sqrt().clamp_min(1e-6))
    return torch.stack(filters)


class DopplerFilterBank(nn.Module):
    """Apply fixed 17/33/65-frame local Fourier filters to learned RF channels."""

    WINDOWS = (17, 33, 65)
    CYCLES = (1.0, 2.0)

    def __init__(self, channels: int, output: int, dropout: float):
        super().__init__()
        self.channels = channels
        for width in self.WINDOWS:
            self.register_buffer(
                f"filters_{width}", _hann_filter_bank(width, self.CYCLES),
                persistent=False,
            )
        components = len(self.WINDOWS) * len(self.CYCLES) * 3
        self.projection = nn.Sequential(
            nn.LayerNorm(channels * components),
            nn.Linear(channels * components, output),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B,T,L,C]. Each sine/cosine pair also contributes magnitude.
        batch, frames, links, channels = values.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        flat = values.permute(0, 2, 3, 1).reshape(batch * links, channels, frames)
        responses = []
        for width in self.WINDOWS:
            kernels = getattr(self, f"filters_{width}")
            weight = kernels[:, None].repeat(channels, 1, 1)
            filtered = F.conv1d(
                flat, weight, padding=width // 2, groups=channels
            ).reshape(batch * links, channels, len(kernels), frames)
            for band in range(len(self.CYCLES)):
                sine = filtered[:, :, 2 * band]
                cosine = filtered[:, :, 2 * band + 1]
                magnitude = torch.sqrt(sine.square() + cosine.square() + 1e-8)
                responses.extend((sine, cosine, magnitude))
        stacked = torch.stack(responses, dim=3)
        stacked = stacked.permute(0, 3, 1, 2).reshape(
            batch, links, frames, -1
        ).permute(0, 2, 1, 3)
        return self.projection(stacked)


class CrossLinkDirectionalFusion(nn.Module):
    """Fuse TX tokens while retaining ordered pairwise link differences."""

    PAIRS = ((0, 1), (0, 2), (1, 2))

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.attention = LinkAttentionFusion(hidden, heads=heads, dropout=dropout)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(hidden * len(self.PAIRS)),
            nn.Linear(hidden * len(self.PAIRS), hidden),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid()
        )

    def forward(self, links: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended = self.attention(links, mask)
        differences = []
        for left, right in self.PAIRS:
            valid = (mask[:, :, left] & mask[:, :, right]).to(links.dtype)
            differences.append(
                (links[:, :, left] - links[:, :, right]) * valid[..., None]
            )
        paired = self.pair_projection(torch.cat(differences, dim=-1))
        gate = self.mix(torch.cat((attended, paired), dim=-1))
        available = mask.any(-1, keepdim=True).to(links.dtype)
        return (gate * attended + (1.0 - gate) * paired) * available


class DopplerTimeFrequencyEncoder(nn.Module):
    """Encode dynamic amplitude/phase with multi-resolution Doppler filters."""

    def __init__(self, normalizer: PerLinkNorm, hidden: int,
                 heads: int, dropout: float):
        super().__init__()
        import copy

        self.norm = copy.deepcopy(normalizer)
        for parameter in self.norm.parameters():
            parameter.requires_grad_(False)
        spectral = max(24, hidden // 2)
        groups = 8 if spectral % 8 == 0 else 1
        self.subcarrier = nn.Sequential(
            nn.Conv1d(4, spectral, 7, padding=3),
            nn.GroupNorm(groups, spectral),
            nn.GELU(),
            nn.Conv1d(spectral, spectral, 5, padding=2),
            nn.GroupNorm(groups, spectral),
            nn.GELU(),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(spectral * 2, spectral), nn.LayerNorm(spectral)
        )
        self.tx_embedding = nn.Parameter(torch.zeros(C.N_LINKS, spectral))
        nn.init.normal_(self.tx_embedding, std=0.02)
        self.filter_bank = DopplerFilterBank(spectral, hidden, dropout)
        self.link_fusion = CrossLinkDirectionalFusion(hidden, heads, dropout)
        self.register_buffer(
            "link_geometry",
            torch.tensor(C.LINK_GEOMETRY, dtype=torch.float32),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(hidden * self.link_geometry.shape[1]),
            nn.Linear(hidden * self.link_geometry.shape[1], hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # Existing KP2 checkpoints remain exact at initialization. Geometry is
        # learned only when the new KP3 training objective supplies gradients.
        nn.init.zeros_(self.geometry_projection[-1].weight)
        nn.init.zeros_(self.geometry_projection[-1].bias)

    def directional_moments(self, links: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """Project link tokens onto fixed physical installation directions."""
        geometry = self.link_geometry.to(dtype=links.dtype)
        weight = mask.to(links.dtype)
        moments = torch.einsum("btlh,lc->btch", links * weight[..., None], geometry)
        denominator = torch.einsum(
            "btl,lc->btc", weight, geometry.abs()
        ).clamp_min(1.0)
        moments = moments / denominator[..., None]
        return moments.flatten(-2)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(csi, link_mask)
        delta1, mask1 = lagged_difference(normalized, link_mask, 1)
        delta3, mask3 = lagged_difference(normalized, link_mask, 3)
        dynamic_mask = mask1 | mask3
        values = torch.cat((delta1, delta3), dim=-1)
        batch, frames, links, subcarriers, _ = values.shape
        spatial = self.subcarrier(
            values.reshape(batch * frames * links, subcarriers, 4).transpose(1, 2)
        )
        pooled = torch.cat((spatial.mean(-1), spatial.amax(-1)), dim=-1)
        pooled = self.pool_projection(pooled).reshape(batch, frames, links, -1)
        pooled = pooled + self.tx_embedding[None, None]
        pooled = pooled * dynamic_mask[..., None].to(pooled.dtype)
        filtered = self.filter_bank(pooled)
        filtered = filtered * dynamic_mask[..., None].to(filtered.dtype)
        fused = self.link_fusion(filtered, dynamic_mask)
        geometry = self.geometry_projection(
            self.directional_moments(filtered, dynamic_mask)
        )
        available = dynamic_mask.any(-1, keepdim=True).to(fused.dtype)
        return (fused + geometry) * available, dynamic_mask


class DopplerMotionEncoder(nn.Module):
    """Combine KP1 finite differences with Doppler and ordered-link evidence."""

    def __init__(self, normalizer: PerLinkNorm, hidden: int = 64,
                 temporal_layers: int = 1, heads: int = 4,
                 dropout: float = 0.08):
        super().__init__()
        self.legacy = KineticDynamicEncoder(
            normalizer, hidden=hidden, temporal_layers=temporal_layers,
            heads=heads, dropout=dropout,
        )
        self.doppler = DopplerTimeFrequencyEncoder(
            normalizer, hidden=hidden, heads=heads, dropout=dropout
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.temporal = TemporalTransformer(
            hidden, temporal_layers, heads, dropout
        )

    @property
    def norm(self) -> PerLinkNorm:
        return self.legacy.norm

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        legacy, activity = self.legacy(csi, link_mask)
        doppler, _ = self.doppler(csi, link_mask)
        frame_mask = link_mask.any(-1)
        fused = legacy + self.fusion(torch.cat((legacy, doppler), dim=-1))
        return self.temporal(fused, frame_mask), activity


class MaskedTrialEmbedding(nn.Module):
    """Pool frame features into a normalized trial-level embedding."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask[..., None].to(values.dtype)
        count = weight.sum(1).clamp_min(1.0)
        mean = (values * weight).sum(1) / count
        variance = ((values - mean[:, None]).square() * weight).sum(1) / count
        maximum = values.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        embedding = self.projection(torch.cat((mean, variance.sqrt(), maximum), dim=-1))
        return F.normalize(embedding, dim=-1)


class PoseMotionEmbedding(nn.Module):
    """Encode GT joint velocity for cross-modal trial correspondence."""

    def __init__(self, hidden: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(C.N_JOINTS * 3, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.pool = MaskedTrialEmbedding(hidden, embedding_dim, dropout)

    def forward(self, pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(pose)
        pair = valid[:, 1:] & valid[:, :-1]
        velocity[:, 1:] = (
            (pose[:, 1:] - pose[:, :-1]) * C.TARGET_FPS
            * pair[:, :, None, None].to(pose.dtype)
        )
        features = self.temporal(
            velocity.flatten(2).transpose(1, 2)
        ).transpose(1, 2)
        return self.pool(features, valid)


class DopplerPoseResidual(KineticPoseResidual):
    """KP2-A diagnostic model with Doppler and trial correspondence heads."""

    def __init__(self, baseline: nn.Module | None, normalizer: PerLinkNorm,
                 hidden: int = 64, temporal_layers: int = 1,
                 heads: int = 4, dropout: float = 0.08,
                 max_delta: float = 0.25,
                 condition_on_coarse: bool = True,
                 activity_floor: float = 0.0,
                 embedding_dim: int = 64):
        super().__init__(
            baseline, normalizer, hidden, temporal_layers, heads, dropout,
            max_delta, condition_on_coarse, activity_floor,
        )
        self.dynamic = DopplerMotionEncoder(
            normalizer, hidden, temporal_layers, heads, dropout
        )
        self.csi_motion_embedding = MaskedTrialEmbedding(
            hidden, embedding_dim, dropout
        )
        self.pose_motion_embedding = PoseMotionEmbedding(
            hidden, embedding_dim, dropout
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = (
            "dynamic.", "refiner.", "velocity_head.",
            "csi_motion_embedding.", "pose_motion_embedding.",
        )
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor | None = None) -> dict:
        output = super().forward(csi, link_mask, coarse_pose)
        frame_mask = link_mask.any(-1)
        output["csi_motion_embedding"] = self.csi_motion_embedding(
            output["kinetic_features"], frame_mask
        )
        return output

    def encode_target_motion(self, pose: torch.Tensor,
                             valid: torch.Tensor) -> torch.Tensor:
        return self.pose_motion_embedding(pose, valid)


def correspondence_loss(csi_embedding: torch.Tensor,
                        pose_embedding: torch.Tensor,
                        rows: torch.Tensor, class_ids: torch.Tensor,
                        domain_ids: torch.Tensor, temperature: float = 0.10,
                        same_class_bias: float = 0.35,
                        same_domain_bias: float = 0.15,
                        ) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric multi-positive InfoNCE with same-class/site hard negatives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = csi_embedding @ pose_embedding.transpose(0, 1) / temperature
    positive = rows[:, None] == rows[None, :]
    negative = ~positive
    same_class = class_ids[:, None] == class_ids[None, :]
    same_domain = domain_ids[:, None] == domain_ids[None, :]
    hard_bias = negative.to(logits.dtype) * (
        same_class.to(logits.dtype) * same_class_bias
        + (same_class & same_domain).to(logits.dtype) * same_domain_bias
    )
    adjusted = logits + hard_bias

    def direction(values: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        numerator = torch.logsumexp(values.masked_fill(~positives, -torch.inf), dim=1)
        denominator = torch.logsumexp(values, dim=1)
        return (denominator - numerator).mean()

    loss = 0.5 * (
        direction(adjusted, positive) + direction(adjusted.T, positive.T)
    )
    with torch.no_grad():
        prediction = logits.argmax(1)
        retrieval = positive.gather(1, prediction[:, None]).float().mean()
        hard = negative & same_class & same_domain
        hard_count = int(hard.sum().item())
        hard_similarity = (
            float(logits[hard].mean()) if hard_count else math.nan
        )
        positive_similarity = float(logits[positive].mean())
    return loss, {
        "correspondence": float(loss.detach()),
        "retrieval_at_1": float(retrieval),
        "positive_similarity": positive_similarity,
        "hard_negative_similarity": hard_similarity,
        "hard_negative_pairs": float(hard_count),
    }
