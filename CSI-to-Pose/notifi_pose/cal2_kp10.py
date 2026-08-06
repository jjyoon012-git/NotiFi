"""Raw link-level support calibration for CAL2-KP10."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .cal1_kp10 import SAFE_SUPPORT_CLASSES


def support_statistics(support_csi: torch.Tensor,
                       support_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return link/subcarrier moments from [B,S,T,L,F,2] support CSI."""
    unbatched = support_csi.ndim == 5
    if unbatched:
        support_csi = support_csi[None]
        support_mask = support_mask[None]
    if support_csi.ndim != 6:
        raise ValueError("support_csi must have shape [B,S,T,L,F,2]")
    weight = support_mask[..., None, None].to(support_csi.dtype)
    count = weight.sum((1, 2)).clamp_min(1.0)
    mean = (support_csi * weight).sum((1, 2)) / count
    variance = (
        (support_csi - mean[:, None, None]).square() * weight
    ).sum((1, 2)) / count

    delta = support_csi[:, :, 1:] - support_csi[:, :, :-1]
    delta_mask = support_mask[:, :, 1:] & support_mask[:, :, :-1]
    delta_weight = delta_mask[..., None, None].to(support_csi.dtype)
    delta_count = delta_weight.sum((1, 2)).clamp_min(1.0)
    dynamic = torch.sqrt(
        (delta.square() * delta_weight).sum((1, 2)) / delta_count + 1e-6
    )
    result = {
        "mean": mean,
        "std": torch.sqrt(variance + 1e-6),
        "dynamic": dynamic,
    }
    if unbatched:
        result = {key: value[0] for key, value in result.items()}
    return result


def support_trial_descriptors(support_csi: torch.Tensor,
                              support_mask: torch.Tensor) -> torch.Tensor:
    weight = support_mask[..., None, None].to(support_csi.dtype)
    count = weight.sum(2).clamp_min(1.0)
    mean = (support_csi * weight).sum(2) / count
    variance = (
        (support_csi - mean[:, :, None]).square() * weight
    ).sum(2) / count
    delta = support_csi[:, :, 1:] - support_csi[:, :, :-1]
    delta_mask = support_mask[:, :, 1:] & support_mask[:, :, :-1]
    delta_weight = delta_mask[..., None, None].to(support_csi.dtype)
    delta_count = delta_weight.sum(2).clamp_min(1.0)
    delta_abs = (delta.abs() * delta_weight).sum(2) / delta_count
    # Average subcarriers only after preserving link and I/Q identity.
    return torch.cat((
        mean.mean(-2), torch.sqrt(variance + 1e-6).mean(-2),
        delta_abs.mean(-2),
    ), dim=-1).flatten(2)


class RawSupportSetEncoder(nn.Module):
    """Encode prompted calibration trials without positional leakage."""

    def __init__(self, token_dim: int = 96, prompt_dim: int = 24,
                 layers: int = 2, heads: int = 4, dropout: float = 0.08):
        super().__init__()
        descriptor_dim = C.N_LINKS * 6
        self.prompt = nn.Embedding(C.N_CLASSES, prompt_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(descriptor_dim + prompt_dim),
            nn.Linear(descriptor_dim + prompt_dim, token_dim), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads,
            dim_feedforward=token_dim * 3, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(token_dim)
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim // 2),
            nn.Tanh(), nn.Linear(token_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim),
            nn.GELU(), nn.LayerNorm(token_dim),
        )

    def encode_descriptors(self, descriptor: torch.Tensor,
                           support_class: torch.Tensor) -> torch.Tensor:
        unbatched = descriptor.ndim == 2
        if unbatched:
            descriptor = descriptor[None]
            support_class = support_class[None]
        values = self.input(torch.cat((
            descriptor, self.prompt(support_class.long()),
        ), dim=-1))
        values = self.set_encoder(values)
        score = torch.softmax(self.attention(values).squeeze(-1), dim=-1)
        token = self.output((values * score[..., None]).sum(1))
        return token[0] if unbatched else token

    def forward(self, support_csi: torch.Tensor,
                support_mask: torch.Tensor,
                support_class: torch.Tensor) -> torch.Tensor:
        unbatched = support_csi.ndim == 5
        if unbatched:
            support_csi = support_csi[None]
            support_mask = support_mask[None]
            support_class = support_class[None]
        descriptor = support_trial_descriptors(support_csi, support_mask)
        token = self.encode_descriptors(descriptor, support_class)
        return token[0] if unbatched else token


class RawLinkCanonicalizer(nn.Module):
    """Align static channel and dynamic CSI scales before any link encoder."""

    def __init__(self, token_dim: int = 96, basis_rank: int = 8,
                 lowpass_window: int = 31, dropout: float = 0.08):
        super().__init__()
        if lowpass_window % 2 != 1:
            raise ValueError("lowpass_window must be odd")
        self.token_dim = int(token_dim)
        self.basis_rank = int(basis_rank)
        self.lowpass_window = int(lowpass_window)
        self.support_encoder = RawSupportSetEncoder(
            token_dim=token_dim, dropout=dropout
        )
        self.alignment = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, C.N_LINKS * 2),
        )
        self.curve = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, C.N_LINKS * 2 * basis_rank * 2),
        )
        nn.init.zeros_(self.alignment[-1].weight)
        nn.init.constant_(self.alignment[-1].bias, -1.0)
        nn.init.zeros_(self.curve[-1].weight)
        nn.init.zeros_(self.curve[-1].bias)

        frequency = torch.linspace(0.0, 1.0, C.N_LIVE_SUBCARRIERS)
        basis = [torch.ones_like(frequency)]
        for order in range(1, basis_rank):
            basis.append(torch.cos(math.pi * order * frequency))
        self.register_buffer("frequency_basis", torch.stack(basis, dim=-1))
        shape = (C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        self.register_buffer("reference_mean", torch.zeros(shape))
        self.register_buffer("reference_std", torch.ones(shape))
        self.register_buffer("reference_dynamic", torch.ones(shape))
        self.register_buffer("reference_fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def set_reference(self, mean: torch.Tensor, std: torch.Tensor,
                      dynamic: torch.Tensor) -> None:
        self.reference_mean.copy_(mean.reshape_as(self.reference_mean).float())
        self.reference_std.copy_(
            std.reshape_as(self.reference_std).float().clamp_min(1e-4)
        )
        self.reference_dynamic.copy_(
            dynamic.reshape_as(self.reference_dynamic).float().clamp_min(1e-4)
        )
        self.reference_fitted.fill_(True)

    def encode_support(self, support_csi: torch.Tensor,
                       support_mask: torch.Tensor,
                       support_class: torch.Tensor) -> dict[str, torch.Tensor]:
        stats = support_statistics(support_csi, support_mask)
        stats["token"] = self.support_encoder(
            support_csi, support_mask, support_class
        )
        return stats

    def encode_summary(self, statistics: dict[str, torch.Tensor],
                       descriptors: torch.Tensor,
                       support_class: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            **statistics,
            "token": self.support_encoder.encode_descriptors(
                descriptors, support_class
            ),
        }

    def _lowpass(self, csi: torch.Tensor) -> torch.Tensor:
        batch, frames, links, subcarriers, components = csi.shape
        values = csi.permute(0, 2, 3, 4, 1).reshape(
            batch * links * subcarriers * components, 1, frames
        )
        values = F.avg_pool1d(
            values, self.lowpass_window, stride=1,
            padding=self.lowpass_window // 2,
        )
        return values.reshape(
            batch, links, subcarriers, components, frames
        ).permute(0, 4, 1, 2, 3)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                context: dict[str, torch.Tensor],
                strength: float = 1.0) -> dict[str, torch.Tensor]:
        if float(strength) == 0.0:
            return {
                "csi": csi,
                "static_blend": csi.new_zeros(len(csi), C.N_LINKS),
                "dynamic_blend": csi.new_zeros(len(csi), C.N_LINKS),
            }
        if not bool(self.reference_fitted):
            raise RuntimeError("source reference statistics are not fitted")
        token = context["token"]
        if token.ndim == 1:
            token = token[None].expand(len(csi), -1)
        mean = context["mean"]
        std = context["std"]
        dynamic = context["dynamic"]
        if mean.ndim == 3:
            mean = mean[None].expand(len(csi), -1, -1, -1)
            std = std[None].expand_as(mean)
            dynamic = dynamic[None].expand_as(mean)

        static_logit, dynamic_logit = self.alignment(token).reshape(
            len(csi), C.N_LINKS, 2
        ).unbind(-1)
        static_blend = torch.sigmoid(static_logit)
        dynamic_blend = torch.sigmoid(dynamic_logit)
        low = self._lowpass(csi)
        high = csi - low
        static_scale = (self.reference_std[None] / std.clamp_min(1e-4)).clamp(
            0.60, 1.67
        )
        dynamic_scale = (
            self.reference_dynamic[None] / dynamic.clamp_min(1e-4)
        ).clamp(0.60, 1.67)
        aligned_low = (
            (low - mean[:, None]) * static_scale[:, None]
            + self.reference_mean[None, None]
        )
        aligned_high = high * dynamic_scale[:, None]
        static_weight = static_blend.to(low.dtype)[:, None, :, None, None]
        dynamic_weight = dynamic_blend.to(high.dtype)[:, None, :, None, None]
        output = torch.lerp(
            low, aligned_low.to(low.dtype), static_weight
        ) + torch.lerp(
            high, aligned_high.to(high.dtype), dynamic_weight
        )

        coefficients = self.curve(token).reshape(
            len(csi), C.N_LINKS, 2, self.basis_rank, 2
        )
        curves = torch.einsum(
            "fr,blirc->blfic", self.frequency_basis, coefficients
        )
        gain = 1.0 + 0.15 * torch.tanh(curves[..., 0])
        bias = 0.10 * torch.tanh(curves[..., 1]) * self.reference_std[None]
        output = output * gain[:, None] + bias[:, None]
        output = torch.lerp(csi, output, float(strength))
        output = output * link_mask[..., None, None].to(output.dtype)
        return {
            "csi": output,
            "static_blend": float(strength) * static_blend,
            "dynamic_blend": float(strength) * dynamic_blend,
        }


class RawCalibratedKP4(nn.Module):
    """Run frozen KP4 after support-conditioned raw CSI canonicalization."""

    def __init__(self, base: nn.Module, canonicalizer: RawLinkCanonicalizer):
        super().__init__()
        self.base = base
        self.canonicalizer = canonicalizer
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor,
                support_context: dict[str, torch.Tensor],
                strength: float = 1.0) -> dict[str, torch.Tensor]:
        calibrated = self.canonicalizer(
            csi, link_mask, support_context, strength
        )
        output = self.base(calibrated["csi"], link_mask, coarse_pose)
        return {
            **output,
            "calibrated_csi": calibrated["csi"],
            "calibration_static_blend": calibrated["static_blend"],
            "calibration_dynamic_blend": calibrated["dynamic_blend"],
        }


__all__ = [
    "SAFE_SUPPORT_CLASSES", "RawLinkCanonicalizer", "RawCalibratedKP4",
    "RawSupportSetEncoder", "support_statistics", "support_trial_descriptors",
]
