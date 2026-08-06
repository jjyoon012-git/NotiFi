"""Directional, label-conditioned CSI pose refinement with contact outputs."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import LocalTemporalBlock


CONTACT_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "head", "left_wrist", "right_wrist",
)
CONTACT_JOINTS = tuple(C.JOINT_INDEX[name] for name in CONTACT_JOINT_NAMES)
N_CONTACT_JOINTS = len(CONTACT_JOINTS)


def _zero_last(module: nn.Sequential) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[..., None].to(values.dtype)
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def _local_bones(pose: torch.Tensor) -> torch.Tensor:
    bones = torch.zeros_like(pose)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            bones[:, :, child] = pose[:, :, child] - pose[:, :, parent]
    return bones


def _forward_kinematics(bones: torch.Tensor) -> torch.Tensor:
    joints = []
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent < 0:
            joints.append(torch.zeros_like(bones[:, :, child]))
        else:
            joints.append(joints[parent] + bones[:, :, child])
    return torch.stack(joints, dim=2)


class DirectionalConditionedContactPose(nn.Module):
    """Refine the locked seen pose using trainable directional CSI evidence.

    The validated 30 percent KP2-DH proposal stays frozen. A copied motion
    encoder can therefore learn physical link directions without moving the
    deployment baseline. All output heads are initialized so that the first
    forward pass exactly reproduces that locked proposal.
    """

    def __init__(self, pose_model: nn.Module, hidden: int | None = None,
                 dropout: float = 0.08, initial_gate: float = 0.30,
                 max_gate_adjustment: float = 0.15,
                 direction_scale: float = 0.15,
                 cartesian_scale: float = 0.04):
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial gate must be strictly between zero and one")
        self.pose_model = pose_model
        self.pose_model.eval()
        for parameter in self.pose_model.parameters():
            parameter.requires_grad_(False)

        source_hidden = int(pose_model.backbone.latent_head.in_channels)
        hidden = source_hidden if hidden is None else int(hidden)
        self.hidden = hidden
        self.initial_gate = float(initial_gate)
        if max_gate_adjustment <= 0.0:
            raise ValueError("maximum gate adjustment must be positive")
        if not 0.0 <= initial_gate - max_gate_adjustment:
            raise ValueError("gate adjustment permits a negative confidence")
        if initial_gate + max_gate_adjustment > 1.0:
            raise ValueError("gate adjustment permits confidence above one")
        self.max_gate_adjustment = float(max_gate_adjustment)
        self.direction_scale = float(direction_scale)
        self.cartesian_scale = float(cartesian_scale)

        # The copied encoder is independent of the frozen proposal model. It
        # retains the TX ordering and LINK_GEOMETRY buffers from KP2/KP3.
        self.motion_encoder = copy.deepcopy(pose_model.backbone.dynamic)
        self.source_projection = nn.Linear(source_hidden, hidden)
        self.motion_projection = nn.Linear(source_hidden, hidden)
        self.feature_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trial_attention = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        )
        self.action_head = nn.Linear(hidden, C.N_CLASSES)
        self.risk_head = nn.Linear(hidden, C.N_RISK)
        self.phase_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.contact_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_CONTACT_JOINTS)
        )

        condition_size = C.N_CLASSES + C.N_RISK + 4 + N_CONTACT_JOINTS + 1
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_size, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        _zero_last(self.condition_projection)
        self.condition_blocks = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )

        self.joint_gate_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_JOINTS),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_JOINTS * 3),
        )
        self.cartesian_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_JOINTS * 3),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        for head in (
            self.joint_gate_head, self.direction_head,
            self.cartesian_head, self.velocity_head,
        ):
            _zero_last(head)
        self.register_buffer(
            "deployment_strength", torch.tensor(1.0), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_model.eval()
        return self

    def set_residual_strength(self, strength: float) -> None:
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("residual strength must be between zero and one")
        self.deployment_strength.fill_(float(strength))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if value.dtype.is_floating_point and not key.startswith("pose_model.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"unknown KP4-DCC weights: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)

    def _pool(self, features: torch.Tensor,
              frame_mask: torch.Tensor) -> torch.Tensor:
        score = self.trial_attention(features).squeeze(-1)
        score = score.masked_fill(~frame_mask, -1e4)
        attention = torch.softmax(score, dim=1)
        attended = (features * attention[..., None]).sum(1)
        return 0.5 * (attended + _masked_mean(features, frame_mask))

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor) -> dict:
        with torch.no_grad():
            source = self.pose_model(csi, link_mask)
        frame_mask = link_mask.any(-1)
        motion, activity = self.motion_encoder(csi, link_mask)
        base = self.source_projection(source["csi_pose_features"])
        moving = self.motion_projection(motion)
        gate = self.feature_gate(torch.cat((base, moving), dim=-1))
        features = self.fusion(torch.cat((base, gate * moving), dim=-1))
        features = features * frame_mask[..., None].to(features.dtype)

        pooled = self._pool(features, frame_mask)
        action_logits = self.action_head(pooled)
        risk_logits = self.risk_head(pooled)
        phase_logits = self.phase_head(features)
        contact_logits = self.contact_head(features)
        condition = torch.cat((
            torch.softmax(action_logits, dim=-1)[:, None].expand(
                -1, features.shape[1], -1
            ),
            torch.softmax(risk_logits, dim=-1)[:, None].expand(
                -1, features.shape[1], -1
            ),
            torch.softmax(phase_logits, dim=-1),
            torch.sigmoid(contact_logits),
            torch.tanh(activity)[..., None],
        ), dim=-1)
        conditioned = features + self.condition_projection(condition)
        for block in self.condition_blocks:
            conditioned = block(conditioned)
            conditioned = conditioned * frame_mask[..., None].to(conditioned.dtype)

        candidate = source["pose_rel"]
        adjustment = torch.tanh(self.joint_gate_head(conditioned))
        joint_gate = self.initial_gate + self.max_gate_adjustment * adjustment
        joint_gate = joint_gate * frame_mask[..., None].to(joint_gate.dtype)
        adaptive = coarse_pose + joint_gate[..., None] * (candidate - coarse_pose)

        bones = _local_bones(adaptive)
        lengths = torch.linalg.vector_norm(bones, dim=-1, keepdim=True)
        direction_delta = torch.tanh(
            self.direction_head(conditioned).reshape(
                *conditioned.shape[:2], C.N_JOINTS, 3
            )
        )
        directions = F.normalize(
            bones + self.direction_scale * lengths * direction_delta,
            dim=-1,
        )
        refined_bones = directions * lengths
        refined_bones[:, :, C.ROOT_JOINT] = 0.0
        kinematic = _forward_kinematics(refined_bones)
        cartesian = self.cartesian_scale * torch.tanh(
            self.cartesian_head(conditioned).reshape(
                *conditioned.shape[:2], C.N_JOINTS, 3
            )
        )
        cartesian[:, :, C.ROOT_JOINT] = 0.0
        raw_pose = kinematic + cartesian
        raw_pose = raw_pose - raw_pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        locked = coarse_pose + self.initial_gate * (candidate - coarse_pose)
        locked = locked - locked[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        pose = locked + self.deployment_strength * (raw_pose - locked)
        pose = pose * frame_mask[..., None, None].to(pose.dtype)
        velocity = self.velocity_head(conditioned).reshape(
            *conditioned.shape[:2], C.N_JOINTS, 3
        )
        velocity = velocity * frame_mask[..., None, None].to(velocity.dtype)
        return {
            **source,
            "pose_rel": pose,
            "pose_v13s": coarse_pose,
            "pose_candidate": candidate,
            "pose_coarse": locked,
            "pose_delta": pose - locked,
            "pose_full_refinement": raw_pose,
            "kinematic_pose": kinematic,
            "kinetic_velocity": velocity,
            "joint_confidence_gate": joint_gate,
            "action_logits": action_logits,
            "class_logits": action_logits,
            "risk_logits": risk_logits,
            "phase_logits": phase_logits,
            "contact_logits": contact_logits,
            "conditioned_features": conditioned,
            "motion_activity": activity,
            "direction_delta": direction_delta,
            "cartesian_residual": cartesian,
        }
