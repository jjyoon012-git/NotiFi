"""CSI-conditioned continuous residual refinement for retrieved SMPL-22 motion."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from notifi_pose import contract as C


def local_bones(pose: torch.Tensor) -> torch.Tensor:
    """Convert pelvis-relative joints to parent-relative bone vectors."""
    bones = torch.zeros_like(pose)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            bones[:, :, child] = pose[:, :, child] - pose[:, :, parent]
    return bones


def forward_kinematics(bones: torch.Tensor) -> torch.Tensor:
    """Reconstruct pelvis-relative joints along the SMPL body-22 tree."""
    joints = []
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent < 0:
            joints.append(torch.zeros_like(bones[:, :, child]))
        else:
            joints.append(joints[parent] + bones[:, :, child])
    return torch.stack(joints, dim=2)


class DilatedTemporalBlock(nn.Module):
    """Mix local and long-range motion while preserving the frame grid."""

    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=hidden,
        )
        self.channel = nn.Sequential(
            nn.Conv1d(hidden, hidden * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply one residual temporal block to valid frames only."""
        residual = features
        hidden = self.norm(features).transpose(1, 2)
        hidden = self.channel(self.depthwise(hidden)).transpose(1, 2)
        return (residual + hidden) * mask[..., None].to(hidden.dtype)


class MotionResidualDecoder(nn.Module):
    """Refine a retrieved motion using CSI features without query labels or GT."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.10,
        max_bone_delta: float = 0.16,
        bone_length_blend: float = 0.0,
        risk_gate_floor: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.max_bone_delta = float(max_bone_delta)
        self.bone_length_blend = float(bone_length_blend)
        self.risk_gate_floor = float(risk_gate_floor)
        pose_dim = C.N_JOINTS * 3
        self.csi_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.pose_projection = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, hidden),
            nn.GELU(),
        )
        self.velocity_projection = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, hidden),
            nn.GELU(),
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(C.N_CLASSES + C.N_RISK),
            nn.Linear(C.N_CLASSES + C.N_RISK, hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.ModuleList(
            DilatedTemporalBlock(hidden, 2 ** index, dropout)
            for index in range(layers)
        )
        self.joint_embedding = nn.Parameter(
            torch.randn(C.N_JOINTS, hidden) * 0.02
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def config(self) -> dict:
        """Return the constructor contract needed for artifact loading."""
        return {
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "max_bone_delta": self.max_bone_delta,
            "bone_length_blend": self.bone_length_blend,
            "risk_gate_floor": self.risk_gate_floor,
        }

    def forward(
        self,
        csi_features: torch.Tensor,
        coarse_pose: torch.Tensor,
        frame_mask: torch.Tensor,
        action_probability: torch.Tensor,
        risk_probability: torch.Tensor,
        strength: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Predict bounded parent-relative bone corrections and refined pose."""
        if csi_features.shape[:2] != coarse_pose.shape[:2]:
            raise ValueError("CSI features and coarse pose time axes must match")
        if coarse_pose.shape[-2:] != (C.N_JOINTS, 3):
            raise ValueError("coarse pose must be [B,T,22,3]")
        if frame_mask.shape != coarse_pose.shape[:2]:
            raise ValueError("frame mask must match pose time axes")
        batch, frames = coarse_pose.shape[:2]
        coarse_bones = local_bones(coarse_pose)
        velocity = torch.zeros_like(coarse_pose)
        velocity[:, 1:] = coarse_pose[:, 1:] - coarse_pose[:, :-1]
        condition = torch.cat((action_probability, risk_probability), dim=-1)
        condition = self.condition_projection(condition)[:, None].expand(
            batch, frames, -1
        )
        hidden = self.fusion(torch.cat((
            self.csi_projection(csi_features),
            self.pose_projection(coarse_bones.reshape(batch, frames, -1)),
            self.velocity_projection(velocity.reshape(batch, frames, -1)),
            condition,
        ), dim=-1))
        hidden = hidden * frame_mask[..., None].to(hidden.dtype)
        for block in self.temporal:
            hidden = block(hidden, frame_mask)
        joint_features = hidden[:, :, None] + self.joint_embedding[None, None]
        delta = self.max_bone_delta * torch.tanh(self.delta_head(joint_features))
        gate = torch.sigmoid(self.gate_head(joint_features))
        joint_mask = delta.new_ones(C.N_JOINTS)
        joint_mask[C.ROOT_JOINT] = 0.0
        delta = delta * joint_mask[None, None, :, None]
        gate = gate * joint_mask[None, None, :, None]
        amount = torch.as_tensor(
            strength, dtype=delta.dtype, device=delta.device
        )
        motion_probability = (
            risk_probability[:, 2] + 0.25 * risk_probability[:, 1]
        ).clamp(0.0, 1.0).sqrt()
        semantic_gate = self.risk_gate_floor + (
            1.0 - self.risk_gate_floor
        ) * motion_probability
        refined_bones = coarse_bones + (
            amount * semantic_gate[:, None, None, None] * gate * delta
        )
        if self.bone_length_blend > 0.0:
            coarse_length = torch.linalg.vector_norm(
                coarse_bones, dim=-1, keepdim=True
            )
            refined_length = torch.linalg.vector_norm(
                refined_bones, dim=-1, keepdim=True
            )
            stable_length = refined_length + self.bone_length_blend * (
                coarse_length - refined_length
            )
            refined_bones = F.normalize(refined_bones, dim=-1) * stable_length
        refined_pose = forward_kinematics(refined_bones)
        refined_pose = torch.where(
            frame_mask[..., None, None], refined_pose, coarse_pose
        )
        return {
            "pose_rel": refined_pose,
            "coarse_pose": coarse_pose,
            "bone_delta": delta,
            "joint_gate": gate,
        }


def motion_residual_loss(
    output: dict[str, torch.Tensor],
    target_pose: torch.Tensor,
    valid: torch.Tensor,
    risk: torch.Tensor,
    distal_joints: tuple[int, ...],
    danger_weight: float = 2.5,
    distal_weight: float = 2.0,
    velocity_weight: float = 0.04,
    identity_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize pose, distal joints, dynamics, and conservative residual size."""
    predicted = output["pose_rel"]
    coordinate = F.smooth_l1_loss(
        predicted, target_pose, reduction="none", beta=0.05
    ).mean(-1)
    joint_weight = coordinate.new_ones(C.N_JOINTS)
    joint_weight[list(distal_joints)] = float(distal_weight)
    sample_weight = torch.where(
        risk == 2,
        coordinate.new_tensor(float(danger_weight)),
        coordinate.new_tensor(1.0),
    )
    weight = (
        valid[..., None].to(coordinate.dtype)
        * joint_weight[None, None]
        * sample_weight[:, None, None]
    )
    pose_loss = (coordinate * weight).sum() / weight.sum().clamp_min(1.0)
    pair = valid[:, 1:] & valid[:, :-1]
    predicted_velocity = predicted[:, 1:] - predicted[:, :-1]
    target_velocity = target_pose[:, 1:] - target_pose[:, :-1]
    velocity = F.smooth_l1_loss(
        predicted_velocity, target_velocity, reduction="none", beta=0.02
    ).mean((-1, -2))
    velocity_loss = (
        velocity * pair.to(velocity.dtype) * sample_weight[:, None]
    ).sum() / (
        pair.to(velocity.dtype) * sample_weight[:, None]
    ).sum().clamp_min(1.0)
    identity_loss = (
        output["bone_delta"].abs() * output["joint_gate"]
    ).mean()
    total = (
        pose_loss
        + float(velocity_weight) * velocity_loss
        + float(identity_weight) * identity_loss
    )
    return total, {
        "total": float(total.detach()),
        "pose": float(pose_loss.detach()),
        "velocity": float(velocity_loss.detach()),
        "identity": float(identity_loss.detach()),
        "gate": float(output["joint_gate"].mean().detach()),
    }


__all__ = (
    "DilatedTemporalBlock",
    "MotionResidualDecoder",
    "motion_residual_loss",
)
