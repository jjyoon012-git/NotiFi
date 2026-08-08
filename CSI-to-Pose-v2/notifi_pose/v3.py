"""Domain-robust CSI-to-pose V3 model.

The model keeps frequency and link structure until after per-link temporal
encoding.  It predicts a pelvis-relative kinematic skeleton and a separately
factorized root trajectory.  Optional board geometry is consumed only when a
measured calibration file exists; no synthetic coordinates are assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import HybridGraphDecoder, LocalTemporalBlock, PerLinkNorm


def rotation_6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. 6D rotations to orthonormal matrices."""
    first = F.normalize(values[..., 0:3], dim=-1)
    second_raw = values[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class DualViewFrequencyTokenizer(nn.Module):
    """Tokenize raw I/Q and temporal CSI changes without global frequency pooling."""

    def __init__(self, hidden: int, n_frequency_tokens: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        width = max(32, hidden // 2)
        groups = 8 if width % 8 == 0 else 1
        self.n_frequency_tokens = n_frequency_tokens
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 7, padding=3),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2, groups=width),
            nn.Conv1d(width, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2, groups=hidden),
            nn.Conv1d(hidden, hidden, 1),
            nn.Dropout(dropout),
        )
        self.frequency_embedding = nn.Parameter(
            torch.zeros(n_frequency_tokens, hidden)
        )
        self.link_embedding = nn.Parameter(torch.zeros(C.N_LINKS, hidden))
        self.norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.frequency_embedding, std=0.02)
        nn.init.normal_(self.link_embedding, std=0.02)

    def forward(self, normalized: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        b, time, links, subcarriers = raw.shape[:4]
        amplitude = torch.linalg.vector_norm(raw, dim=-1).clamp_min(1e-5)
        log_amplitude = torch.log1p(amplitude)
        phase = torch.atan2(raw[..., 1], raw[..., 0])

        delta_amplitude = torch.zeros_like(log_amplitude)
        delta_phase = torch.zeros_like(phase)
        delta_amplitude[:, 1:] = log_amplitude[:, 1:] - log_amplitude[:, :-1]
        phase_step = phase[:, 1:] - phase[:, :-1]
        delta_phase[:, 1:] = torch.atan2(torch.sin(phase_step), torch.cos(phase_step))

        channels = torch.cat(
            (normalized, delta_amplitude[..., None], delta_phase[..., None]),
            dim=-1,
        )
        channels = channels.reshape(b * time * links, subcarriers, 4).transpose(1, 2)
        tokens = self.stem(channels)
        tokens = F.adaptive_avg_pool1d(tokens, self.n_frequency_tokens)
        tokens = tokens.transpose(1, 2).reshape(
            b, time, links, self.n_frequency_tokens, -1
        )
        tokens = tokens + self.frequency_embedding[None, None, None]
        tokens = tokens + self.link_embedding[None, None, :, None]
        return self.norm(tokens)


class FrequencyQueryPool(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, time, links, frequencies, hidden = tokens.shape
        flat = tokens.reshape(b * time * links, frequencies, hidden)
        query = self.query.expand(len(flat), -1, -1)
        pooled, _ = self.attention(query, flat, flat, need_weights=False)
        return self.norm(pooled[:, 0]).reshape(b, time, links, hidden)


class SequenceEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.local = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        block = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            block, max(1, layers), norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.local:
            values = block(values)
        safe = mask.clone()
        empty = ~safe.any(1)
        if empty.any():
            safe[empty, 0] = True
        values = self.global_encoder(values, src_key_padding_mask=~safe)
        return values * mask[..., None].to(values.dtype)


class PerLinkTemporalEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.encoder = SequenceEncoder(hidden, layers, heads, dropout)

    def forward(self, links: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        b, time, count, hidden = links.shape
        values = links.permute(0, 2, 1, 3).reshape(b * count, time, hidden)
        mask = link_mask.permute(0, 2, 1).reshape(b * count, time)
        values = self.encoder(values, mask)
        return values.reshape(b, count, time, hidden).permute(0, 2, 1, 3)


class LateLinkFusion(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, links: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, time, count, hidden = links.shape
        values = links.reshape(b * time, count, hidden)
        query = self.query.expand(b * time, -1, -1)
        alive = mask.reshape(b * time, count)
        safe = alive.clone()
        empty = ~safe.any(1)
        if empty.any():
            safe[empty, 0] = True
        fused, _ = self.attention(
            query, values, values, key_padding_mask=~safe, need_weights=False
        )
        fused = self.norm1(fused[:, 0])
        fused = self.norm2(fused + self.ffn(fused)).reshape(b, time, hidden)
        return fused * alive.any(1).reshape(b, time, 1).to(fused.dtype)


def _canonical_directions(device=None) -> torch.Tensor:
    directions = torch.zeros(C.N_JOINTS, 3, device=device)
    names = {name: index for index, name in enumerate(C.JOINT_NAMES)}
    values = {
        "left_hip": (-1, 0, 0), "right_hip": (1, 0, 0),
        "spine1": (0, 1, 0), "left_knee": (0, -1, 0),
        "right_knee": (0, -1, 0), "spine2": (0, 1, 0),
        "left_ankle": (0, -1, 0), "right_ankle": (0, -1, 0),
        "spine3": (0, 1, 0), "left_foot": (0, 0, 1),
        "right_foot": (0, 0, 1), "neck": (0, 1, 0),
        "left_collar": (-1, 0.4, 0), "right_collar": (1, 0.4, 0),
        "head": (0, 1, 0), "left_shoulder": (-1, 0, 0),
        "right_shoulder": (1, 0, 0), "left_elbow": (-1, -0.2, 0),
        "right_elbow": (1, -0.2, 0), "left_wrist": (-1, 0, 0),
        "right_wrist": (1, 0, 0),
    }
    for name, value in values.items():
        if name in names:
            directions[names[name]] = torch.tensor(value, device=device)
    return F.normalize(directions, dim=-1)


class KinematicBoneDecoder(nn.Module):
    """Predict bone directions with trial-constant, bounded body proportions."""

    def __init__(self, hidden: int, blocks: int, dropout: float):
        super().__init__()
        self.code_size = C.N_JOINTS * 4
        if hidden < self.code_size:
            raise ValueError(
                f"kinematic latent requires hidden >= {self.code_size}, got {hidden}"
            )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_JOINTS * 3),
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS)
        )
        default_lengths = torch.full((C.N_JOINTS,), 0.20)
        default_lengths[C.ROOT_JOINT] = 0.0
        self.register_buffer("bone_lengths", default_lengths)
        self.register_buffer("template_fitted", torch.zeros(1, dtype=torch.bool))
        nn.init.zeros_(self.direction_head[-1].weight)
        nn.init.zeros_(self.direction_head[-1].bias)
        nn.init.zeros_(self.length_head[-1].weight)
        nn.init.zeros_(self.length_head[-1].bias)

    @torch.no_grad()
    def set_bone_lengths(self, lengths: torch.Tensor) -> None:
        lengths = lengths.to(self.bone_lengths).clamp(0.02, 0.65)
        lengths[C.ROOT_JOINT] = 0.0
        self.bone_lengths.copy_(lengths)
        self.template_fitted.fill_(True)

    def forward(self, temporal: torch.Tensor,
                valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        directions = _canonical_directions(temporal.device)
        direction_code = temporal[..., :C.N_JOINTS * 3].reshape(
            *temporal.shape[:2], C.N_JOINTS, 3
        )
        direction_delta = direction_code + 0.1 * torch.tanh(
            self.direction_head(temporal).reshape(
                *temporal.shape[:2], C.N_JOINTS, 3
            )
        )
        directions = F.normalize(
            directions[None, None] + direction_delta, dim=-1
        )
        if valid is None:
            pooled = temporal.mean(1)
        else:
            weight = valid.to(temporal.dtype)
            pooled = (temporal * weight[..., None]).sum(1)
            pooled = pooled / weight.sum(1, keepdim=True).clamp_min(1)
        # Bone lengths may vary by subject, but never by frame. The bounded scale
        # prevents CSI noise from stretching limbs during a fall.
        start = C.N_JOINTS * 3
        length_code = pooled[:, start:start + C.N_JOINTS]
        length_code = length_code + 0.25 * torch.tanh(self.length_head(pooled))
        length_scale = 1.0 + 0.15 * torch.tanh(length_code)
        lengths = self.bone_lengths[None] * length_scale
        bones = directions * lengths[:, None, :, None]

        pose = torch.zeros_like(bones)
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                pose[:, :, child] = pose[:, :, parent] + bones[:, :, child]
        return pose, directions


class MotionPriorEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.hidden = hidden
        self.code_size = C.N_JOINTS * 4
        if hidden < self.code_size:
            raise ValueError(
                f"kinematic latent requires hidden >= {self.code_size}, got {hidden}"
            )
        default_lengths = torch.full((C.N_JOINTS,), 0.20)
        default_lengths[C.ROOT_JOINT] = 0.0
        self.register_buffer("bone_lengths", default_lengths)

    @torch.no_grad()
    def set_bone_lengths(self, lengths: torch.Tensor) -> None:
        lengths = lengths.to(self.bone_lengths).clamp(0.02, 0.65)
        lengths[C.ROOT_JOINT] = 0.0
        self.bone_lengths.copy_(lengths)

    def forward(self, pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        local = torch.zeros_like(pose)
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                local[:, :, child] = pose[:, :, child] - pose[:, :, parent]
        length = torch.linalg.vector_norm(local, dim=-1)
        direction = F.normalize(local, dim=-1)
        canonical = _canonical_directions(pose.device)
        direction_code = (direction - canonical[None, None]).flatten(-2)

        base = self.bone_lengths.clamp_min(1e-6)
        ratio = length / base[None, None]
        ratio[:, :, C.ROOT_JOINT] = 1.0
        normalized_scale = ((ratio - 1.0) / 0.15).clamp(-0.999, 0.999)
        length_code = torch.atanh(normalized_scale)

        latent = pose.new_zeros(*pose.shape[:2], self.hidden)
        split = C.N_JOINTS * 3
        latent[..., :split] = direction_code
        latent[..., split:split + C.N_JOINTS] = length_code
        return latent * valid[..., None]


class MotionPriorAutoencoder(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 3, heads: int = 4,
                 graph_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = MotionPriorEncoder(hidden, layers, heads, dropout)
        self.decoder = KinematicBoneDecoder(hidden, graph_blocks, dropout)

    @torch.no_grad()
    def set_bone_lengths(self, lengths: torch.Tensor) -> None:
        self.encoder.set_bone_lengths(lengths)
        self.decoder.set_bone_lengths(lengths)

    def forward(self, pose: torch.Tensor, valid: torch.Tensor) -> dict:
        latent = self.encoder(pose, valid)
        reconstructed, directions = self.decoder(latent, valid)
        return {"pose_rel": reconstructed, "bone_direction": directions,
                "motion_latent": latent}


def load_geometry(path: str | Path | None) -> tuple[torch.Tensor, bool]:
    geometry = torch.zeros(C.N_LINKS, 6)
    if not path:
        return geometry, False
    path = Path(path)
    if not path.exists():
        return geometry, False
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in payload.get("links", [])}
    for index, name in enumerate(C.LINKS):
        item = by_name.get(name)
        if item is None:
            return torch.zeros_like(geometry), False
        geometry[index, :3] = torch.tensor(item["tx"], dtype=torch.float32)
        geometry[index, 3:] = torch.tensor(item["rx"], dtype=torch.float32)
    return geometry, True


class V3PoseNet(nn.Module):
    def __init__(self, hidden: int = 128, n_blocks: int = 2, dropout: float = 0.1,
                 heads: int = 4, graph_blocks: int = 2,
                 frequency_tokens: int = 12, geometry_path: str | None = None,
                 domain_grl: float = 0.2, **_: object):
        super().__init__()
        self.hidden = hidden
        self.domain_grl = domain_grl
        self.norm = PerLinkNorm()
        self.tokenizer = DualViewFrequencyTokenizer(
            hidden, frequency_tokens, dropout
        )
        self.frequency_pool = FrequencyQueryPool(hidden, heads, dropout)
        self.per_link_temporal = PerLinkTemporalEncoder(
            hidden, max(1, n_blocks - 1), heads, dropout
        )
        self.link_fusion = LateLinkFusion(hidden, heads, dropout)
        self.motion_encoder = SequenceEncoder(hidden, n_blocks, heads, dropout)
        self.latent_projection = nn.Linear(hidden, hidden)
        nn.init.eye_(self.latent_projection.weight)
        nn.init.zeros_(self.latent_projection.bias)
        self.pose_decoder = HybridGraphDecoder(hidden, graph_blocks, dropout)
        self.kinematic_decoder = KinematicBoneDecoder(hidden, graph_blocks, dropout)
        self.kinematic_mix_logit = nn.Parameter(torch.tensor(-2.0))
        self.target_motion_encoder = MotionPriorEncoder(
            hidden, n_blocks, heads, dropout
        )
        self.register_buffer("motion_prior_loaded", torch.zeros(1, dtype=torch.bool))
        for parameter in self.target_motion_encoder.parameters():
            parameter.requires_grad = False

        geometry, available = load_geometry(geometry_path)
        self.register_buffer("board_geometry", geometry)
        self.register_buffer("geometry_available", torch.tensor([available]))
        self.geometry_projection = nn.Sequential(
            nn.Linear(C.N_LINKS * 6, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.root_initial = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.root_velocity = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.phase_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.contact_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.class_head = nn.Linear(hidden, C.N_CLASSES)
        self.risk_head = nn.Linear(hidden, C.N_RISK)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 9)
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )

    @torch.no_grad()
    def set_bone_lengths(self, lengths: torch.Tensor) -> None:
        self.target_motion_encoder.set_bone_lengths(lengths)
        self.kinematic_decoder.set_bone_lengths(lengths)

    @torch.no_grad()
    def load_motion_prior(self, checkpoint: dict) -> None:
        self.target_motion_encoder.load_state_dict(checkpoint["encoder"])
        self.kinematic_decoder.load_state_dict(checkpoint["decoder"])
        for parameter in self.target_motion_encoder.parameters():
            parameter.requires_grad = False
        self.motion_prior_loaded.fill_(True)

    def encode_pose_target(self, pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        self.target_motion_encoder.eval()
        with torch.no_grad():
            return self.target_motion_encoder(pose, valid)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        normalized = self.norm(csi, link_mask)
        frequency = self.tokenizer(normalized, csi)
        frequency = frequency * link_mask[..., None, None].to(frequency.dtype)
        links = self.frequency_pool(frequency)
        links = self.per_link_temporal(links, link_mask)
        fused = self.link_fusion(links, link_mask)
        frame_mask = link_mask.any(-1)
        normalized_motion = self.motion_encoder(fused, frame_mask)
        motion_latent = self.latent_projection(normalized_motion)
        motion_latent = motion_latent * frame_mask[..., None].to(motion_latent.dtype)
        direct_pose = self.pose_decoder(motion_latent)
        kinematic_pose, bone_direction = self.kinematic_decoder(
            motion_latent, frame_mask
        )
        kinematic_mix = torch.sigmoid(self.kinematic_mix_logit)
        pose = direct_pose * (1.0 - kinematic_mix) + kinematic_pose * kinematic_mix

        weights = frame_mask.to(motion_latent.dtype)
        pooled = (motion_latent * weights[..., None]).sum(1)
        pooled = pooled / weights.sum(1, keepdim=True).clamp_min(1)
        geometry = self.geometry_projection(self.board_geometry.flatten()[None])
        geometry = geometry.expand(len(csi), -1)
        root0 = self.root_initial(torch.cat((pooled, geometry), dim=-1))
        root_step = 0.08 * torch.tanh(self.root_velocity(motion_latent))
        root_step = root_step * frame_mask[..., None].to(root_step.dtype)
        root = root0[:, None] + torch.cumsum(root_step, dim=1)

        embedding = F.normalize(self.embedding_head(pooled), dim=-1)
        reversed_embedding = GradientReverse.apply(embedding, self.domain_grl)
        return {
            "pose_rel": pose,
            "kinematic_pose": kinematic_pose,
            "kinematic_mix": kinematic_mix,
            "root": root,
            "bone_direction": bone_direction,
            "motion_latent": motion_latent,
            "motion": self.motion_head(motion_latent).squeeze(-1),
            "phase_logits": self.phase_head(motion_latent),
            "contact_logits": self.contact_head(motion_latent),
            "class_logits": self.class_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "domain_logits": self.domain_head(reversed_embedding),
            "embedding": embedding,
        }

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def describe(self) -> str:
        geometry = "measured" if bool(self.geometry_available.item()) else "unavailable"
        return (
            f"V3PoseNet(hidden={self.hidden}, frequency_tokens="
            f"{self.tokenizer.n_frequency_tokens}, late_link_fusion, "
            f"hybrid_graph+kinematic_bone_code, geometry={geometry}) "
            f"params={self.n_params():,}"
        )
