"""Shared amplitude-Doppler pretraining across heterogeneous CSI datasets.

Only hardware-neutral encoder weights are transferred to KP2. Dataset-specific
classification heads, absolute gain, static phase, and link identity never cross
the pretraining boundary.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .doppler_pose import DopplerFilterBank, DopplerMotionEncoder, MaskedTrialEmbedding
from .nets import LinkAttentionFusion, TemporalTransformer


def _amplitude_difference(
    amplitude: torch.Tensor, mask: torch.Tensor, lag: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if lag <= 0:
        raise ValueError("lag must be positive")
    output = torch.zeros_like(amplitude)
    valid = torch.zeros_like(mask)
    if amplitude.shape[1] > lag:
        valid[:, lag:] = mask[:, lag:] & mask[:, :-lag]
        output[:, lag:] = (
            amplitude[:, lag:] - amplitude[:, :-lag]
        ) * valid[:, lag:, :, None].to(amplitude.dtype)
    return output, valid


class UniversalDopplerEncoder(nn.Module):
    """Variable-link encoder whose transferable layers match KP2 Doppler.

    Input is the canonical external CSI view [B,T,L,S,2]. Only channel zero,
    robust-standardized log amplitude, is used. The four convolution channels
    match KP2's [lag1 amp, lag1 phase, lag3 amp, lag3 phase] order; phase slots
    are zero during external pretraining.
    """

    def __init__(self, hidden: int = 128, temporal_layers: int = 2,
                 heads: int = 4, dropout: float = 0.08):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        spectral = max(24, hidden // 2)
        groups = 8 if spectral % 8 == 0 else 1
        self.hidden = int(hidden)
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
        self.filter_bank = DopplerFilterBank(spectral, hidden, dropout)
        self.link_fusion = LinkAttentionFusion(hidden, heads=heads, dropout=dropout)
        self.temporal = TemporalTransformer(
            hidden, layers=temporal_layers, heads=heads, dropout=dropout
        )

    def forward(self, values: torch.Tensor, link_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 5 or values.shape[-1] != 2:
            raise ValueError("values must have shape [B,T,L,S,2]")
        if link_mask.shape != values.shape[:3]:
            raise ValueError("link_mask must match values [B,T,L]")
        amplitude = values[..., 0]
        delta1, mask1 = _amplitude_difference(amplitude, link_mask, 1)
        delta3, mask3 = _amplitude_difference(amplitude, link_mask, 3)
        zeros1, zeros3 = torch.zeros_like(delta1), torch.zeros_like(delta3)
        frequency_input = torch.stack(
            (delta1, zeros1, delta3, zeros3), dim=-1
        )
        dynamic_mask = mask1 | mask3
        batch, frames, links, subcarriers, _ = frequency_input.shape
        spatial = self.subcarrier(
            frequency_input.reshape(
                batch * frames * links, subcarriers, 4
            ).transpose(1, 2)
        )
        pooled = torch.cat((spatial.mean(-1), spatial.amax(-1)), dim=-1)
        pooled = self.pool_projection(pooled).reshape(batch, frames, links, -1)
        pooled = pooled * dynamic_mask[..., None].to(pooled.dtype)
        filtered = self.filter_bank(pooled)
        filtered = filtered * dynamic_mask[..., None].to(filtered.dtype)
        fused = self.link_fusion(filtered, dynamic_mask)
        frame_mask = dynamic_mask.any(-1)
        encoded = self.temporal(fused, frame_mask)
        return encoded * frame_mask[..., None].to(encoded.dtype), frame_mask


class MultiSourceCSIPretrainer(nn.Module):
    """Task heads for external action, impact, and pose-motion supervision."""

    def __init__(self, dataset_classes: Mapping[str, int], hidden: int = 128,
                 temporal_layers: int = 2, heads: int = 4,
                 motion_dim: int = 96, dropout: float = 0.08):
        super().__init__()
        if not dataset_classes:
            raise ValueError("at least one dataset classification head is required")
        self.encoder = UniversalDopplerEncoder(
            hidden, temporal_layers, heads, dropout
        )
        self.trial_pool = MaskedTrialEmbedding(hidden, hidden, dropout)
        self.class_heads = nn.ModuleDict({
            dataset_id: nn.Linear(hidden, int(classes))
            for dataset_id, classes in dataset_classes.items()
        })
        self.impact_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, motion_dim)
        )

    def forward(self, values: torch.Tensor, link_mask: torch.Tensor,
                dataset_id: str) -> dict[str, torch.Tensor]:
        if dataset_id not in self.class_heads:
            raise KeyError(f"no classification head for {dataset_id}")
        temporal, frame_mask = self.encoder(values, link_mask)
        trial = self.trial_pool(temporal, frame_mask)
        return {
            "temporal_features": temporal,
            "frame_mask": frame_mask,
            "trial_embedding": trial,
            "class_logits": self.class_heads[dataset_id](trial),
            "impact_logits": self.impact_head(temporal).squeeze(-1),
            "motion_embedding": self.motion_head(temporal),
        }

    def shared_checkpoint(self) -> dict:
        return {
            "format": "notifi_external_amplitude_doppler_v1",
            "hidden": self.encoder.hidden,
            "encoder": {
                key: value.detach().cpu()
                for key, value in self.encoder.state_dict().items()
            },
        }


def transplant_external_encoder(
    target: DopplerMotionEncoder, checkpoint: Mapping
) -> dict[str, int]:
    """Load only compatible shared weights into a KP2 motion encoder.

    External pretraining has no phase evidence. For the first convolution only
    amplitude channels 0 and 2 are copied; target phase channels 1 and 3 remain
    untouched. Link pooling and dataset heads are never transferred.
    """

    if checkpoint.get("format") != "notifi_external_amplitude_doppler_v1":
        raise ValueError("unsupported external pretraining checkpoint")
    source = checkpoint.get("encoder")
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint is missing encoder weights")

    copied = 0
    amplitude_channels = 0
    target_state = target.state_dict()
    mapping: dict[str, str] = {}
    for key in source:
        if key.startswith(("subcarrier.", "pool_projection.", "filter_bank.")):
            mapping[key] = f"doppler.{key}"
        elif key.startswith("temporal."):
            mapping[key] = key
        elif key.startswith("link_fusion."):
            mapping[key] = f"doppler.link_fusion.attention.{key[len('link_fusion.'):]}"

    for source_key, target_key in mapping.items():
        if target_key not in target_state:
            continue
        incoming = source[source_key]
        current = target_state[target_key]
        if source_key == "subcarrier.0.weight":
            if incoming.shape != current.shape:
                raise ValueError(
                    f"first-conv shape mismatch: {incoming.shape} != {current.shape}"
                )
            incoming = incoming.to(device=current.device, dtype=current.dtype)
            current = current.clone()
            current[:, (0, 2)] = incoming[:, (0, 2)]
            target_state[target_key] = current
            amplitude_channels += 2
            copied += 1
        elif incoming.shape == current.shape:
            target_state[target_key] = incoming.to(dtype=current.dtype)
            copied += 1
    target.load_state_dict(target_state, strict=True)
    return {
        "tensors_copied": copied,
        "first_conv_amplitude_channels": amplitude_channels,
    }
