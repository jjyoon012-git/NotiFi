"""Kinematic VQ motion tokens for the KP2 model family."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import LocalTemporalBlock
from .v3 import _canonical_directions


def pose_to_bones(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pelvis-relative joints to parent-relative directions and lengths."""
    bones = torch.zeros_like(pose)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            bones[:, :, child] = pose[:, :, child] - pose[:, :, parent]
    lengths = torch.linalg.vector_norm(bones, dim=-1)
    directions = F.normalize(bones, dim=-1)
    directions[:, :, C.ROOT_JOINT] = 0.0
    return directions, lengths


def trial_bone_lengths(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Estimate one stable bone-length vector per trial."""
    _, lengths = pose_to_bones(pose)
    weight = valid[..., None].to(lengths.dtype)
    result = (lengths * weight).sum(1) / weight.sum(1).clamp_min(1.0)
    result[:, C.ROOT_JOINT] = 0.0
    return result


def forward_kinematics(directions: torch.Tensor,
                       lengths: torch.Tensor) -> torch.Tensor:
    """Compose parent-relative unit directions into pelvis-relative joints."""
    if lengths.ndim == 2:
        bones = directions * lengths[:, None, :, None]
    elif lengths.ndim == 3:
        bones = directions * lengths[..., None]
    else:
        raise ValueError("bone lengths must have shape [B,J] or [B,T,J]")
    pose = torch.zeros_like(bones)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            pose[:, :, child] = pose[:, :, parent] + bones[:, :, child]
    return pose


def downsample_valid(valid: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """A token is valid when any source frame in its block is valid."""
    return F.max_pool1d(
        valid[:, None].float(), kernel_size=factor, stride=factor,
    ).squeeze(1).bool()


class VectorQuantizer(nn.Module):
    def __init__(self, codes: int, dimension: int, commitment: float = 0.25):
        super().__init__()
        self.codes = int(codes)
        self.dimension = int(dimension)
        self.commitment = float(commitment)
        self.codebook = nn.Embedding(codes, dimension)
        nn.init.uniform_(self.codebook.weight, -1.0 / codes, 1.0 / codes)
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _initialize_from_latent(self, latent: torch.Tensor,
                                mask: torch.Tensor) -> None:
        selected = latent[mask]
        if not len(selected):
            return
        if len(selected) < self.codes:
            repeat = math.ceil(self.codes / len(selected))
            selected = selected.repeat(repeat, 1)
        order = torch.randperm(len(selected), device=selected.device)[:self.codes]
        initial = selected[order]
        initial = initial + 1e-3 * torch.randn_like(initial)
        self.codebook.weight.copy_(initial)
        self.initialized.fill_(True)

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> dict:
        if self.training and not bool(self.initialized):
            self._initialize_from_latent(latent.detach(), mask)
        flat = latent.reshape(-1, self.dimension)
        distance = (
            flat.square().sum(-1, keepdim=True)
            + self.codebook.weight.square().sum(-1)[None]
            - 2.0 * flat @ self.codebook.weight.T
        )
        indices = distance.argmin(-1).reshape(latent.shape[:-1])
        quantized = self.codebook(indices)
        weight = mask[..., None].to(latent.dtype)
        count = weight.sum().clamp_min(1.0)
        codebook_loss = (
            (quantized - latent.detach()).square() * weight
        ).sum() / (count * self.dimension)
        commitment_loss = (
            (latent - quantized.detach()).square() * weight
        ).sum() / (count * self.dimension)
        straight_through = latent + (quantized - latent).detach()
        straight_through = straight_through * weight
        soft_assignment = torch.softmax(-distance.float() / 0.10, dim=-1)
        soft_assignment = soft_assignment.reshape(*mask.shape, self.codes)
        average_assignment = (
            soft_assignment * mask[..., None]
        ).sum((0, 1)) / mask.sum().clamp_min(1)
        diversity_loss = (
            average_assignment
            * (average_assignment.clamp_min(1e-12) * self.codes).log()
        ).sum()
        with torch.no_grad():
            selected = indices[mask]
            histogram = torch.bincount(selected, minlength=self.codes).float()
            probability = histogram / histogram.sum().clamp_min(1.0)
            perplexity = torch.exp(
                -(probability * probability.clamp_min(1e-12).log()).sum()
            )
            active = (histogram > 0).sum()
        return {
            "quantized": straight_through,
            "embedding": quantized * weight,
            "token_ids": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": self.commitment * commitment_loss,
            "diversity_loss": diversity_loss,
            "codebook_perplexity": perplexity,
            "active_codes": active,
        }


class IdentityQuantizer(nn.Module):
    """Continuous control path used to isolate discrete-codebook error."""

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> dict:
        weight = mask[..., None].to(latent.dtype)
        zero = latent.sum() * 0.0
        return {
            "quantized": latent * weight,
            "embedding": latent * weight,
            "token_ids": torch.zeros_like(mask, dtype=torch.long),
            "codebook_loss": zero,
            "commitment_loss": zero,
            "diversity_loss": zero,
            "codebook_perplexity": latent.new_tensor(1.0),
            "active_codes": torch.ones((), device=latent.device, dtype=torch.long),
        }


class ResidualVectorQuantizer(nn.Module):
    """Hierarchical base-plus-residual motion quantization."""

    def __init__(self, levels: int, codes: int, dimension: int,
                 commitment: float = 0.25):
        super().__init__()
        if levels < 2:
            raise ValueError("residual quantization needs at least two levels")
        self.levels = nn.ModuleList(
            VectorQuantizer(codes, dimension, commitment)
            for _ in range(levels)
        )

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> dict:
        residual = latent
        quantized = torch.zeros_like(latent)
        outputs = []
        for level in self.levels:
            current = level(residual, mask)
            outputs.append(current)
            quantized = quantized + current["embedding"]
            residual = residual - current["embedding"].detach()
        straight_through = latent + (quantized - latent).detach()
        straight_through = straight_through * mask[..., None].to(latent.dtype)
        return {
            "quantized": straight_through,
            "token_ids": torch.stack(
                [item["token_ids"] for item in outputs], dim=-1
            ),
            "codebook_loss": sum(item["codebook_loss"] for item in outputs),
            "commitment_loss": sum(
                item["commitment_loss"] for item in outputs
            ),
            "diversity_loss": sum(item["diversity_loss"] for item in outputs),
            "codebook_perplexity": torch.stack([
                item["codebook_perplexity"] for item in outputs
            ]).mean(),
            "active_codes": torch.stack([
                item["active_codes"] for item in outputs
            ]).min(),
        }


class MotionTokenEncoder(nn.Module):
    def __init__(self, hidden: int, code_dim: int, dropout: float,
                 downsample: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(C.N_JOINTS * 3, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        stages = int(math.log2(downsample))
        if downsample < 1 or 2 ** stages != downsample:
            raise ValueError("downsample must be a positive power of two")
        layers = []
        current = hidden
        for stage in range(stages):
            output = code_dim if stage == stages - 1 else hidden
            layers.append(nn.Conv1d(current, output, 4, stride=2, padding=1))
            if stage != stages - 1:
                layers.append(nn.GELU())
            current = output
        if not layers:
            layers.append(nn.Conv1d(hidden, code_dim, 1))
        self.downsample = nn.Sequential(*layers)

    def forward(self, directions: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        features = self.input(directions.flatten(2))
        features = features * valid[..., None].to(features.dtype)
        for block in self.temporal:
            features = block(features) * valid[..., None].to(features.dtype)
        return self.downsample(features.transpose(1, 2)).transpose(1, 2)


class MotionTokenDecoder(nn.Module):
    def __init__(self, hidden: int, code_dim: int, dropout: float,
                 downsample: int):
        super().__init__()
        stages = int(math.log2(downsample))
        layers = []
        current = code_dim
        for _ in range(stages):
            layers.extend((
                nn.ConvTranspose1d(current, hidden, 4, stride=2, padding=1),
                nn.GELU(),
            ))
            current = hidden
        if not layers:
            layers.extend((nn.Conv1d(code_dim, hidden, 1), nn.GELU()))
        self.upsample = nn.Sequential(*layers)
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        nn.init.zeros_(self.direction_head[-1].weight)
        nn.init.zeros_(self.direction_head[-1].bias)
        self.register_buffer("canonical", _canonical_directions())

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor,
                frames: int, valid: torch.Tensor | None = None) -> dict:
        features = self.upsample(tokens.transpose(1, 2)).transpose(1, 2)
        if features.shape[1] != frames:
            features = F.interpolate(
                features.transpose(1, 2), size=frames, mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        for block in self.temporal:
            features = block(features)
        delta = self.direction_head(features).reshape(
            *features.shape[:2], C.N_JOINTS, 3
        )
        directions = F.normalize(
            self.canonical[None, None] + delta, dim=-1
        )
        directions[:, :, C.ROOT_JOINT] = 0.0
        pose = forward_kinematics(directions, lengths)
        if valid is not None:
            pose = pose * valid[..., None, None].to(pose.dtype)
        return {
            "pose_rel": pose,
            "bone_direction": directions,
            "motion_token_features": features,
        }


class KinematicMotionTokenizer(nn.Module):
    """Encode 4-frame motion blocks into discrete kinematic tokens."""

    def __init__(self, hidden: int = 128, code_dim: int = 64,
                 codes: int = 256, dropout: float = 0.05,
                 commitment: float = 0.25,
                 downsample: int = 4, quantizer_levels: int = 1,
                 continuous: bool = False):
        super().__init__()
        self.hidden = hidden
        self.code_dim = code_dim
        self.codes = 1 if continuous else codes
        self.downsample = downsample
        self.quantizer_levels = 1 if continuous else quantizer_levels
        self.continuous = bool(continuous)
        self.encoder = MotionTokenEncoder(hidden, code_dim, dropout, downsample)
        self.quantizer = (
            IdentityQuantizer()
            if continuous else VectorQuantizer(codes, code_dim, commitment)
            if quantizer_levels == 1 else
            ResidualVectorQuantizer(
                quantizer_levels, codes, code_dim, commitment
            )
        )
        self.decoder = MotionTokenDecoder(hidden, code_dim, dropout, downsample)

    def encode(self, pose: torch.Tensor, valid: torch.Tensor) -> dict:
        directions, _ = pose_to_bones(pose)
        token_mask = downsample_valid(valid, self.downsample)
        latent = self.encoder(directions, valid)
        quantized = self.quantizer(latent, token_mask)
        return {"token_mask": token_mask, "latent": latent, **quantized}

    def decode(self, tokens: torch.Tensor, lengths: torch.Tensor,
               frames: int, valid: torch.Tensor | None = None) -> dict:
        return self.decoder(tokens, lengths, frames, valid)

    def forward(self, pose: torch.Tensor, valid: torch.Tensor) -> dict:
        encoded = self.encode(pose, valid)
        lengths = trial_bone_lengths(pose, valid)
        decoded = self.decode(
            encoded["quantized"], lengths, pose.shape[1], valid
        )
        return {**decoded, **encoded, "bone_lengths": lengths}


FACTOR_JOINTS = {
    "torso": (1, 2, 3, 6, 9, 12, 13, 14, 15),
    "left_arm": (16, 18, 20),
    "right_arm": (17, 19, 21),
    "left_leg": (4, 7, 10),
    "right_leg": (5, 8, 11),
}


class FactorizedMotionEncoder(nn.Module):
    """Use whole-body context before producing body-part token latents."""

    def __init__(self, hidden: int, code_dim: int, dropout: float,
                 downsample: int, parts: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(C.N_JOINTS * 3, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        stages = int(math.log2(downsample))
        if downsample < 1 or 2 ** stages != downsample:
            raise ValueError("downsample must be a positive power of two")
        layers = []
        for _ in range(stages):
            layers.extend((
                nn.Conv1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU()
            ))
        self.downsample = nn.Sequential(*layers) if layers else nn.Identity()
        self.part_projection = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, code_dim))
            for _ in range(parts)
        )

    def forward(self, directions: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        features = self.input(directions.flatten(2))
        features = features * valid[..., None].to(features.dtype)
        for block in self.temporal:
            features = block(features) * valid[..., None].to(features.dtype)
        features = self.downsample(features.transpose(1, 2)).transpose(1, 2)
        return torch.stack(
            [projection(features) for projection in self.part_projection], dim=2
        )


class BodyPartDecoder(nn.Module):
    def __init__(self, joints: tuple[int, ...], hidden: int,
                 code_dim: int, dropout: float, downsample: int):
        super().__init__()
        self.joints = joints
        stages = int(math.log2(downsample))
        layers = []
        current = code_dim
        for _ in range(stages):
            layers.extend((
                nn.ConvTranspose1d(current, hidden, 4, stride=2, padding=1),
                nn.GELU(),
            ))
            current = hidden
        self.upsample = (
            nn.Sequential(*layers) if layers
            else nn.Sequential(nn.Conv1d(code_dim, hidden, 1), nn.GELU())
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, len(joints) * 3)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, token: torch.Tensor, frames: int) -> torch.Tensor:
        features = self.upsample(token.transpose(1, 2)).transpose(1, 2)
        if features.shape[1] != frames:
            features = F.interpolate(
                features.transpose(1, 2), size=frames, mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        for block in self.temporal:
            features = block(features)
        return self.head(features).reshape(
            *features.shape[:2], len(self.joints), 3
        )


class FactorizedMotionTokenizer(nn.Module):
    """Quantize torso and four limbs into independent token streams."""

    def __init__(self, hidden: int = 192, code_dim: int = 64,
                 codes: int = 256, dropout: float = 0.05,
                 commitment: float = 0.25, downsample: int = 2,
                 part_hidden: int = 96):
        super().__init__()
        self.hidden = hidden
        self.code_dim = code_dim
        self.codes = codes
        self.downsample = downsample
        self.part_names = tuple(FACTOR_JOINTS)
        self.quantizer_levels = len(self.part_names)
        self.encoder = FactorizedMotionEncoder(
            hidden, code_dim, dropout, downsample, self.quantizer_levels
        )
        self.quantizers = nn.ModuleList(
            VectorQuantizer(codes, code_dim, commitment)
            for _ in self.part_names
        )
        self.decoders = nn.ModuleList(
            BodyPartDecoder(
                FACTOR_JOINTS[name], part_hidden, code_dim, dropout, downsample
            )
            for name in self.part_names
        )
        self.register_buffer("canonical", _canonical_directions())

    def encode(self, pose: torch.Tensor, valid: torch.Tensor) -> dict:
        directions, _ = pose_to_bones(pose)
        token_mask = downsample_valid(valid, self.downsample)
        latent = self.encoder(directions, valid)
        levels = [
            quantizer(latent[:, :, index], token_mask)
            for index, quantizer in enumerate(self.quantizers)
        ]
        return {
            "token_mask": token_mask,
            "latent": latent,
            "quantized": torch.stack(
                [item["quantized"] for item in levels], dim=2
            ),
            "token_ids": torch.stack(
                [item["token_ids"] for item in levels], dim=-1
            ),
            "codebook_loss": sum(item["codebook_loss"] for item in levels),
            "commitment_loss": sum(
                item["commitment_loss"] for item in levels
            ),
            "diversity_loss": sum(item["diversity_loss"] for item in levels),
            "codebook_perplexity": torch.stack([
                item["codebook_perplexity"] for item in levels
            ]).mean(),
            "active_codes": torch.stack([
                item["active_codes"] for item in levels
            ]).min(),
        }

    def decode(self, tokens: torch.Tensor, lengths: torch.Tensor,
               frames: int, valid: torch.Tensor | None = None) -> dict:
        directions = self.canonical[None, None].expand(
            tokens.shape[0], frames, -1, -1
        ).clone()
        for index, (name, decoder) in enumerate(zip(self.part_names, self.decoders)):
            joints = FACTOR_JOINTS[name]
            delta = decoder(tokens[:, :, index], frames)
            directions[:, :, joints] = F.normalize(
                self.canonical[list(joints)][None, None] + delta, dim=-1
            )
        directions[:, :, C.ROOT_JOINT] = 0.0
        pose = forward_kinematics(directions, lengths)
        if valid is not None:
            pose = pose * valid[..., None, None].to(pose.dtype)
        return {"pose_rel": pose, "bone_direction": directions}

    def forward(self, pose: torch.Tensor, valid: torch.Tensor) -> dict:
        encoded = self.encode(pose, valid)
        lengths = trial_bone_lengths(pose, valid)
        decoded = self.decode(
            encoded["quantized"], lengths, pose.shape[1], valid
        )
        return {**decoded, **encoded, "bone_lengths": lengths}
