"""Direct continuous 3D motion generation from calibrated framewise CSI features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from notifi_pose import contract as C
from .motion_residual import DilatedTemporalBlock, local_bones


def forward_kinematics(bones: torch.Tensor) -> torch.Tensor:
    """Reconstruct root-relative joints from parent-relative bone vectors."""
    joints = []
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent < 0:
            joints.append(torch.zeros_like(bones[:, :, child]))
        else:
            joints.append(joints[parent] + bones[:, :, child])
    return torch.stack(joints, dim=2)


class ContinuousMotionGenerator(nn.Module):
    """Generate SMPL body-22 trajectories without copying a bank motion."""

    def __init__(
        self,
        feature_dim: int,
        bone_lengths: torch.Tensor | list[float],
        canonical_directions: torch.Tensor | list[list[float]],
        hidden: int = 192,
        temporal_layers: int = 6,
        attention_layers: int = 2,
        dropout: float = 0.10,
        max_cartesian_residual: float = 0.06,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.temporal_layers = int(temporal_layers)
        self.attention_layers = int(attention_layers)
        self.dropout = float(dropout)
        self.max_cartesian_residual = float(max_cartesian_residual)
        lengths = torch.as_tensor(bone_lengths, dtype=torch.float32)
        directions = torch.as_tensor(canonical_directions, dtype=torch.float32)
        if lengths.shape != (C.N_JOINTS,):
            raise ValueError("bone_lengths must contain one value per joint")
        if directions.shape != (C.N_JOINTS, 3):
            raise ValueError("canonical_directions must be [22,3]")
        directions = directions.clone()
        directions[C.ROOT_JOINT] = directions.new_tensor((1.0, 0.0, 0.0))
        self.register_buffer("bone_lengths", lengths)
        self.register_buffer(
            "canonical_directions", F.normalize(directions, dim=-1)
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden),
            nn.GELU(),
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(C.N_CLASSES + C.N_RISK),
            nn.Linear(C.N_CLASSES + C.N_RISK, hidden),
            nn.GELU(),
        )
        self.temporal = nn.ModuleList(
            DilatedTemporalBlock(hidden, 2 ** (index % 5), dropout)
            for index in range(temporal_layers)
        )
        layer = nn.TransformerEncoderLayer(
            hidden,
            nhead=6,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, attention_layers)
        self.joint_embedding = nn.Parameter(
            torch.randn(C.N_JOINTS, hidden) * 0.02
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.cartesian_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.confidence_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.zeros_(self.direction_head[-1].weight)
        nn.init.zeros_(self.direction_head[-1].bias)
        nn.init.zeros_(self.length_head[-1].weight)
        nn.init.zeros_(self.length_head[-1].bias)
        nn.init.zeros_(self.cartesian_head[-1].weight)
        nn.init.zeros_(self.cartesian_head[-1].bias)

    def config(self) -> dict:
        """Return a serialization-safe constructor configuration."""
        return {
            "feature_dim": self.feature_dim,
            "bone_lengths": self.bone_lengths.detach().cpu().tolist(),
            "canonical_directions": self.canonical_directions.detach().cpu().tolist(),
            "hidden": self.hidden,
            "temporal_layers": self.temporal_layers,
            "attention_layers": self.attention_layers,
            "dropout": self.dropout,
            "max_cartesian_residual": self.max_cartesian_residual,
        }

    def forward(
        self,
        csi_features: torch.Tensor,
        frame_mask: torch.Tensor,
        action_probability: torch.Tensor,
        risk_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict kinematic pose and a bounded non-rigid joint correction."""
        if csi_features.ndim != 3:
            raise ValueError("csi_features must be [B,T,F]")
        if frame_mask.shape != csi_features.shape[:2]:
            raise ValueError("frame_mask must match CSI batch and time")
        velocity = torch.zeros_like(csi_features)
        velocity[:, 1:] = csi_features[:, 1:] - csi_features[:, :-1]
        condition = self.condition_projection(torch.cat((
            action_probability, risk_probability,
        ), dim=-1))
        hidden = self.feature_projection(torch.cat((csi_features, velocity), dim=-1))
        hidden = hidden + condition[:, None]
        hidden = hidden * frame_mask[..., None].to(hidden.dtype)
        for block in self.temporal:
            hidden = block(hidden, frame_mask)
        hidden = self.context(hidden, src_key_padding_mask=~frame_mask)
        hidden = hidden * frame_mask[..., None].to(hidden.dtype)
        joint_features = hidden[:, :, None] + self.joint_embedding[None, None]
        direction_delta = 2.0 * torch.tanh(self.direction_head(joint_features))
        direction = F.normalize(
            self.canonical_directions[None, None] + direction_delta, dim=-1
        )
        weights = frame_mask.to(hidden.dtype)
        pooled = (hidden * weights[..., None]).sum(1)
        pooled = pooled / weights.sum(1, keepdim=True).clamp_min(1.0)
        length_features = pooled[:, None] + self.joint_embedding[None]
        length_ratio = 1.0 + 0.18 * torch.tanh(self.length_head(length_features))
        lengths = self.bone_lengths[None, :, None] * length_ratio
        length_mask = torch.ones_like(lengths)
        length_mask[:, C.ROOT_JOINT] = 0.0
        lengths = lengths * length_mask
        bones = direction * lengths[:, None]
        kinematic_pose = forward_kinematics(bones)
        cartesian = self.max_cartesian_residual * torch.tanh(
            self.cartesian_head(joint_features)
        )
        joint_mask = cartesian.new_ones(C.N_JOINTS)
        joint_mask[C.ROOT_JOINT] = 0.0
        cartesian = cartesian * joint_mask[None, None, :, None]
        pose = kinematic_pose + cartesian
        pose = pose - pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        pose = pose * frame_mask[..., None, None].to(pose.dtype)
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
        return {
            "pose_rel": pose,
            "kinematic_pose": kinematic_pose,
            "bone_direction": direction,
            "bone_length": lengths.squeeze(-1),
            "cartesian_residual": cartesian,
            "motion_features": hidden,
            "frame_confidence": confidence,
        }


def continuous_motion_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    risk: torch.Tensor,
    distal_joints: tuple[int, ...],
    danger_weight: float = 2.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize position, normalized shape, bone direction, and dynamics."""
    predicted = output["pose_rel"]
    sample_weight = torch.where(
        risk == 2,
        predicted.new_tensor(float(danger_weight)),
        predicted.new_tensor(1.0),
    )
    target_velocity = torch.zeros_like(target)
    target_velocity[:, 1:] = target[:, 1:] - target[:, :-1]
    salience = torch.linalg.vector_norm(target_velocity, dim=-1).mean(-1)
    salience = 1.0 + 1.5 * (salience / 0.025).clamp_max(2.0)
    joint_weight = predicted.new_ones(C.N_JOINTS)
    joint_weight[list(distal_joints)] = 1.8
    coordinate = F.smooth_l1_loss(
        predicted, target, reduction="none", beta=0.04
    ).mean(-1)
    weight = (
        valid[..., None].to(predicted.dtype)
        * salience[..., None]
        * joint_weight[None, None]
        * sample_weight[:, None, None]
    )
    pose_loss = (coordinate * weight).sum() / weight.sum().clamp_min(1.0)

    predicted_scale = predicted.square().mean(
        (-1, -2), keepdim=True
    ).clamp_min(0.05 ** 2).sqrt()
    target_scale = target.square().mean(
        (-1, -2), keepdim=True
    ).clamp_min(0.05 ** 2).sqrt()
    shape_error = F.smooth_l1_loss(
        predicted / predicted_scale,
        target / target_scale,
        reduction="none",
        beta=0.05,
    ).mean((-1, -2))
    shape_loss = (
        shape_error * valid.to(shape_error.dtype) * sample_weight[:, None]
    ).sum() / (
        valid.to(shape_error.dtype) * sample_weight[:, None]
    ).sum().clamp_min(1.0)

    target_bones = local_bones(target)
    target_direction = F.normalize(target_bones, dim=-1)
    direction_error = 1.0 - (
        output["bone_direction"] * target_direction
    ).sum(-1).clamp(-1.0, 1.0)
    direction_weight = weight.clone()
    direction_weight[:, :, C.ROOT_JOINT] = 0.0
    direction_loss = (
        direction_error * direction_weight
    ).sum() / direction_weight.sum().clamp_min(1.0)

    dynamics = []
    for lag in (1, 3):
        pair = valid[:, lag:] & valid[:, :-lag]
        predicted_delta = predicted[:, lag:] - predicted[:, :-lag]
        target_delta = target[:, lag:] - target[:, :-lag]
        error = F.smooth_l1_loss(
            predicted_delta, target_delta, reduction="none", beta=0.015
        ).mean((-1, -2))
        dynamics.append((
            error * pair.to(error.dtype) * sample_weight[:, None]
        ).sum() / (
            pair.to(error.dtype) * sample_weight[:, None]
        ).sum().clamp_min(1.0))
    velocity_loss = torch.stack(dynamics).mean()
    residual_loss = output["cartesian_residual"].square().mean()
    total = (
        pose_loss
        + 0.20 * shape_loss
        + 0.12 * direction_loss
        + 0.12 * velocity_loss
        + 0.02 * residual_loss
    )
    return total, {
        "total": float(total.detach()),
        "pose": float(pose_loss.detach()),
        "shape": float(shape_loss.detach()),
        "direction": float(direction_loss.detach()),
        "velocity": float(velocity_loss.detach()),
        "residual": float(residual_loss.detach()),
    }


__all__ = (
    "ContinuousMotionGenerator",
    "continuous_motion_loss",
    "forward_kinematics",
)
