"""Temporal CSI-to-motion embedding selector for train-bank retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import LocalTemporalBlock


def masked_temporal_bins(features: torch.Tensor, mask: torch.Tensor,
                         bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average variable-length frame features into stable temporal bins."""
    weight = mask[:, None].to(features.dtype)
    numerator = F.adaptive_avg_pool1d(
        (features.transpose(1, 2) * weight), bins
    )
    denominator = F.adaptive_avg_pool1d(weight, bins)
    pooled = numerator / denominator.clamp_min(1e-5)
    pooled_mask = denominator.squeeze(1) > 0.0
    return pooled.transpose(1, 2), pooled_mask


class TemporalMotionSelector(nn.Module):
    """Predict a train-motion embedding from frozen semantic CSI features."""

    def __init__(self, input_dim: int, embedding_dim: int, width: int = 192,
                 bins: int = 38, layers: int = 2, heads: int = 6,
                 dropout: float = 0.10):
        super().__init__()
        self.bins = int(bins)
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.position = nn.Parameter(torch.zeros(1, bins, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 3,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            block, num_layers=layers, norm=nn.LayerNorm(width)
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, embedding_dim),
        )
        self.action_head = nn.Linear(width * 2, C.N_CLASSES)
        self.risk_head = nn.Linear(width * 2, C.N_RISK)

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> dict:
        values, valid = masked_temporal_bins(features, frame_mask, self.bins)
        values = self.input(values) + self.position
        values = self.temporal(values, src_key_padding_mask=~valid)
        score = self.attention(values).squeeze(-1).masked_fill(~valid, -1e4)
        attention = torch.softmax(score, dim=-1)
        attended = (values * attention[..., None]).sum(1)
        weight = valid[..., None].to(values.dtype)
        mean = (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled = torch.cat((attended, mean), dim=-1)
        return {
            "motion_embedding": self.embedding_head(pooled),
            "action_logits": self.action_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "temporal_attention": attention,
            "pooled_features": pooled,
        }


class CandidateMotionReranker(nn.Module):
    """Score several plausible train motions against one CSI observation."""

    def __init__(self, query_dim: int, embedding_dim: int,
                 class_dim: int = 32, hidden: int = 256,
                 dropout: float = 0.20):
        super().__init__()
        self.class_embedding = nn.Embedding(C.N_CLASSES, class_dim)
        input_dim = (
            query_dim + embedding_dim * 4 + class_dim
            + C.N_RISK + 2
        )
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, query_features: torch.Tensor,
                query_embedding: torch.Tensor,
                candidate_embedding: torch.Tensor,
                candidate_class: torch.Tensor,
                risk_probability: torch.Tensor,
                retrieval_score: torch.Tensor,
                action_log_probability: torch.Tensor) -> torch.Tensor:
        candidates = candidate_embedding.shape[1]
        query = query_features[:, None].expand(-1, candidates, -1)
        embedding = query_embedding[:, None].expand(-1, candidates, -1)
        risk = risk_probability[:, None].expand(-1, candidates, -1)
        values = torch.cat((
            query,
            embedding,
            candidate_embedding,
            (embedding - candidate_embedding).abs(),
            embedding * candidate_embedding,
            self.class_embedding(candidate_class),
            risk,
            retrieval_score[..., None],
            action_log_probability[..., None],
        ), dim=-1)
        return self.score(values).squeeze(-1)


class ProfileCandidateRanker(nn.Module):
    """Rank train motions from pose and anatomical motion-profile evidence."""

    def __init__(self, feature_dim: int = 8, class_dim: int = 16,
                 hidden: int = 64, dropout: float = 0.10,
                 context_dim: int = 0):
        super().__init__()
        self.context_dim = int(context_dim)
        self.class_embedding = nn.Embedding(C.N_CLASSES, class_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim + class_dim + self.context_dim),
            nn.Linear(feature_dim + class_dim + self.context_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor,
                class_id: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.class_embedding(class_id)
        values = [features, embedded]
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"Expected context dimension {self.context_dim}"
                )
            values.append(context)
        return self.score(torch.cat(values, dim=-1)).squeeze(-1)


class MotionMixturePredictor(CandidateMotionReranker):
    """Predict candidate weights and a bounded CSI-conditioned blend strength."""

    def __init__(self, query_dim: int, embedding_dim: int,
                 class_dim: int = 32, hidden: int = 256,
                 dropout: float = 0.20, min_strength: float = 0.25,
                 max_strength: float = 0.80):
        super().__init__(
            query_dim=query_dim, embedding_dim=embedding_dim,
            class_dim=class_dim, hidden=hidden, dropout=dropout,
        )
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.strength = nn.Sequential(
            nn.LayerNorm(query_dim + embedding_dim + C.N_RISK),
            nn.Linear(query_dim + embedding_dim + C.N_RISK, hidden // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, query_features: torch.Tensor,
                query_embedding: torch.Tensor,
                candidate_embedding: torch.Tensor,
                candidate_class: torch.Tensor,
                risk_probability: torch.Tensor,
                retrieval_score: torch.Tensor,
                action_log_probability: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = super().forward(
            query_features, query_embedding, candidate_embedding,
            candidate_class, risk_probability, retrieval_score,
            action_log_probability,
        )
        strength_input = torch.cat((
            query_features, query_embedding, risk_probability,
        ), dim=-1)
        unit = torch.sigmoid(self.strength(strength_input)).squeeze(-1)
        strength = self.min_strength + (
            self.max_strength - self.min_strength
        ) * unit
        return {"candidate_logits": logits, "blend_strength": strength}


class MotionPriorResidualRefiner(nn.Module):
    """Learn a small CSI-conditioned correction around a locked motion prior."""

    def __init__(self, feature_dim: int, hidden: int = 128,
                 layers: int = 4, dropout: float = 0.12,
                 max_residual: float = 0.18):
        super().__init__()
        pose_dim = C.N_JOINTS * 3
        self.feature = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU(),
        )
        self.pose = nn.Sequential(
            nn.LayerNorm(pose_dim * 2), nn.Linear(pose_dim * 2, hidden), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        dilations = (1, 2, 4, 8)[:layers]
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in dilations
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, pose_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.max_residual = float(max_residual)

    def forward(self, features: torch.Tensor, base_pose: torch.Tensor,
                frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        velocity = torch.zeros_like(base_pose)
        velocity[:, 1:] = base_pose[:, 1:] - base_pose[:, :-1]
        pose = torch.cat((base_pose, velocity), dim=-1).flatten(2)
        hidden = self.fusion(torch.cat((
            self.feature(features), self.pose(pose),
        ), dim=-1))
        valid = frame_mask[..., None].to(hidden.dtype)
        hidden = hidden * valid
        for block in self.temporal:
            hidden = block(hidden) * valid
        residual = self.max_residual * torch.tanh(
            self.residual(hidden).reshape(
                len(features), features.shape[1], C.N_JOINTS, 3
            )
        )
        refined = base_pose + residual
        refined = refined - refined[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        return {"pose": refined, "residual": residual, "features": hidden}


def geometric_pair_features(baseline: torch.Tensor,
                            candidates: torch.Tensor) -> torch.Tensor:
    """Explicit pose, part, dynamics, height, and contact comparison features."""
    distal_joints = tuple(sorted(set(
        C.JOINT_GROUPS["head"]
        + C.JOINT_GROUPS["left_arm"][-1:]
        + C.JOINT_GROUPS["right_arm"][-1:]
        + C.JOINT_GROUPS["left_leg"][-2:]
        + C.JOINT_GROUPS["right_leg"][-2:]
    )))
    baseline = baseline[:, None]
    distance = torch.linalg.vector_norm(candidates - baseline, dim=-1)
    values = [distance.mean((2, 3))]
    for group in C.JOINT_GROUPS.values():
        values.append(distance[..., list(group)].mean((2, 3)))
    values.extend((distance[:, :, 0].mean(-1), distance[:, :, -1].mean(-1)))

    baseline_velocity = baseline[:, :, 1:] - baseline[:, :, :-1]
    candidate_velocity = candidates[:, :, 1:] - candidates[:, :, :-1]
    velocity_distance = torch.linalg.vector_norm(
        candidate_velocity - baseline_velocity, dim=-1
    )
    values.extend((
        velocity_distance.mean((2, 3)),
        velocity_distance[..., list(distal_joints)].mean((2, 3)),
    ))
    baseline_acceleration = baseline_velocity[:, :, 1:] - baseline_velocity[:, :, :-1]
    candidate_acceleration = candidate_velocity[:, :, 1:] - candidate_velocity[:, :, :-1]
    values.append(torch.linalg.vector_norm(
        candidate_acceleration - baseline_acceleration, dim=-1
    ).mean((2, 3)))

    baseline_speed = torch.linalg.vector_norm(baseline_velocity, dim=-1).mean(-1)
    candidate_speed = torch.linalg.vector_norm(candidate_velocity, dim=-1).mean(-1)
    values.append((candidate_speed - baseline_speed).abs().mean(-1))
    threshold = torch.quantile(baseline_speed, 0.75, dim=-1, keepdim=True)
    moving = baseline_speed >= threshold
    dynamic_distance = distance[:, :, 1:].mean(-1)
    values.append(
        (dynamic_distance * moving).sum(-1) / moving.sum(-1).clamp_min(1)
    )

    baseline_height = (
        baseline[..., C.UP_AXIS].amax(-1) - baseline[..., C.UP_AXIS].amin(-1)
    )
    candidate_height = (
        candidates[..., C.UP_AXIS].amax(-1) - candidates[..., C.UP_AXIS].amin(-1)
    )
    values.append((candidate_height - baseline_height).abs().mean(-1))
    contact_joints = (0, 1, 2, 4, 5, 15, 20, 21)
    baseline_floor = baseline[..., C.UP_AXIS].amin(-1, keepdim=True)
    candidate_floor = candidates[..., C.UP_AXIS].amin(-1, keepdim=True)
    baseline_contact = torch.sigmoid((
        0.12 - (
            baseline[..., list(contact_joints), C.UP_AXIS] - baseline_floor
        )
    ) / 0.04)
    candidate_contact = torch.sigmoid((
        0.12 - (
            candidates[..., list(contact_joints), C.UP_AXIS] - candidate_floor
        )
    ) / 0.04)
    values.append((candidate_contact - baseline_contact).square().mean((2, 3)))
    return torch.stack(values, dim=-1)


class GeometryResidualReranker(nn.Module):
    """Correct a frozen semantic reranker with explicit trajectory evidence."""

    def __init__(self, query_dim: int, embedding_dim: int,
                 pair_dim: int, width: int = 96, dropout: float = 0.15,
                 correction_scale: float = 0.75):
        super().__init__()
        self.base = CandidateMotionReranker(query_dim, embedding_dim)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.query = nn.Sequential(
            nn.LayerNorm(query_dim + embedding_dim + C.N_RISK),
            nn.Linear(query_dim + embedding_dim + C.N_RISK, width), nn.GELU(),
        )
        self.pair = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, width), nn.GELU(),
        )
        self.correction = nn.Sequential(
            nn.LayerNorm(width * 3 + 1), nn.Linear(width * 3 + 1, width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.correction_scale = float(correction_scale)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, query_features: torch.Tensor,
                query_embedding: torch.Tensor,
                candidate_embedding: torch.Tensor,
                candidate_class: torch.Tensor,
                risk_probability: torch.Tensor,
                retrieval_score: torch.Tensor,
                action_log_probability: torch.Tensor,
                pair_features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.base(
                query_features, query_embedding, candidate_embedding,
                candidate_class, risk_probability, retrieval_score,
                action_log_probability,
            )
        query = self.query(torch.cat((
            query_features, query_embedding, risk_probability,
        ), dim=-1))[:, None].expand(-1, pair_features.shape[1], -1)
        pair = self.pair(pair_features)
        correction = self.correction(torch.cat((
            query, pair, query * pair, base[..., None],
        ), dim=-1)).squeeze(-1)
        return base + self.correction_scale * torch.tanh(correction)


class MotionProfileHead(nn.Module):
    """Predict framewise body speed and motion state from frozen CSI features."""

    def __init__(self, input_dim: int, width: int = 128,
                 dropout: float = 0.10):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.speed = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width // 2, 1), nn.Softplus(),
        )
        self.motion = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 1)
        )

    def forward(self, features: torch.Tensor,
                frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        return {
            "speed": self.speed(values).squeeze(-1) * frame_mask,
            "motion_logits": self.motion(values).squeeze(-1),
            "profile_features": values,
        }


class MotionProgressHead(nn.Module):
    """Predict a monotonic CSI-conditioned action progress curve."""

    def __init__(self, input_dim: int, width: int = 128,
                 dropout: float = 0.10):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.increment = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width // 2, 1), nn.Softplus(),
        )

    def forward(self, features: torch.Tensor,
                frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        increment = self.increment(values).squeeze(-1) * frame_mask
        cumulative = torch.cumsum(increment, dim=1)
        total = cumulative.gather(
            1, (frame_mask.long().sum(1) - 1).clamp_min(0)[:, None]
        ).clamp_min(1e-6)
        progress = cumulative / total
        return {
            "progress": progress * frame_mask,
            "increment": increment,
            "features": values,
        }


class ContactProfileHead(nn.Module):
    """Predict framewise body-region floor proximity from frozen CSI features."""

    def __init__(self, input_dim: int, contacts: int = 8,
                 width: int = 128, dropout: float = 0.10):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.contact = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width // 2, contacts),
        )

    def forward(self, features: torch.Tensor,
                frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        logits = self.contact(values)
        return {"contact_logits": logits * weight, "features": values}


class PartMotionProfileHead(nn.Module):
    """Predict framewise motion profiles for six anatomical regions."""

    def __init__(self, input_dim: int, parts: int = 6, width: int = 160,
                 dropout: float = 0.12):
        super().__init__()
        self.parts = int(parts)
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.part_speed = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, self.parts), nn.Softplus(),
        )
        self.part_motion = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, self.parts)
        )

    def forward(self, features: torch.Tensor,
                frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        return {
            "part_speed": self.part_speed(values) * weight,
            "part_motion_logits": self.part_motion(values),
            "profile_features": values,
        }


class PartTrajectoryHead(nn.Module):
    """Predict coarse 3D trajectories for six anatomical regions from CSI."""

    def __init__(self, input_dim: int, parts: int = 6, width: int = 192,
                 dropout: float = 0.15):
        super().__init__()
        self.parts = int(parts)
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.trajectory = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, self.parts * 3),
        )

    def forward(self, features: torch.Tensor,
                frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        trajectory = self.trajectory(values).reshape(
            values.shape[0], values.shape[1], self.parts, 3
        )
        return {
            "part_trajectory": trajectory * weight[..., None],
            "trajectory_features": values,
        }


class PartCandidateMotionReranker(CandidateMotionReranker):
    """Score each candidate independently for six kinematic body regions."""

    def __init__(self, query_dim: int, embedding_dim: int,
                 parts: int = 6, class_dim: int = 32,
                 hidden: int = 256, dropout: float = 0.25):
        super().__init__(
            query_dim=query_dim, embedding_dim=embedding_dim,
            class_dim=class_dim, hidden=hidden, dropout=dropout,
        )
        self.parts = int(parts)
        self.score[-1] = nn.Linear(hidden // 2, self.parts)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return super().forward(*args, **kwargs)


class TemporalCandidateMotionReranker(nn.Module):
    """Cross-match temporal CSI evidence with each candidate trajectory."""

    def __init__(self, csi_dim: int, motion_dim: int = C.N_JOINTS * 6,
                 width: int = 128, bins: int = 38,
                 class_dim: int = 24, dropout: float = 0.18):
        super().__init__()
        self.bins = int(bins)
        self.csi_projection = nn.Sequential(
            nn.LayerNorm(csi_dim), nn.Linear(csi_dim, width), nn.GELU()
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(motion_dim), nn.Linear(motion_dim, width), nn.GELU()
        )
        self.difference_projection = nn.Sequential(
            nn.LayerNorm(motion_dim), nn.Linear(motion_dim, width), nn.GELU()
        )
        self.position = nn.Parameter(torch.zeros(1, 1, bins, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(width * 4), nn.Linear(width * 4, width), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.class_embedding = nn.Embedding(C.N_CLASSES, class_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(width * 2 + class_dim + C.N_RISK + 2),
            nn.Linear(width * 2 + class_dim + C.N_RISK + 2, width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def forward(self, csi_features: torch.Tensor, frame_mask: torch.Tensor,
                baseline_motion: torch.Tensor,
                candidate_motion: torch.Tensor,
                candidate_class: torch.Tensor,
                risk_probability: torch.Tensor,
                retrieval_score: torch.Tensor,
                action_log_probability: torch.Tensor) -> torch.Tensor:
        csi, valid = masked_temporal_bins(
            csi_features, frame_mask, self.bins
        )
        csi = self.csi_projection(csi)
        candidate = self.motion_projection(candidate_motion)
        difference = self.difference_projection(
            candidate_motion - baseline_motion[:, None]
        )
        csi = csi[:, None].expand(-1, candidate.shape[1], -1, -1)
        pair = self.pair_projection(torch.cat((
            csi, candidate, difference, csi * candidate,
        ), dim=-1)) + self.position
        batch, candidates, frames, width = pair.shape
        pair = pair.reshape(batch * candidates, frames, width)
        repeated_valid = valid[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, frames
        )
        for block in self.temporal:
            pair = block(pair)
            pair = pair * repeated_valid[..., None].to(pair.dtype)
        attention = self.temporal_attention(pair).squeeze(-1)
        attention = torch.softmax(
            attention.masked_fill(~repeated_valid, -1e4), dim=-1
        )
        attended = (pair * attention[..., None]).sum(1)
        weight = repeated_valid[..., None].to(pair.dtype)
        mean = (pair * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled = torch.cat((attended, mean), dim=-1).reshape(
            batch, candidates, width * 2
        )
        risk = risk_probability[:, None].expand(-1, candidates, -1)
        values = torch.cat((
            pooled, self.class_embedding(candidate_class), risk,
            retrieval_score[..., None], action_log_probability[..., None],
        ), dim=-1)
        return self.score(values).squeeze(-1)
