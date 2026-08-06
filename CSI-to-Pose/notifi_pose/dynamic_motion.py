"""Direct CSI-to-kinematic-motion model for KP11-DYNAMIC-MOTION."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .conditioned_contact_pose import N_CONTACT_JOINTS
from .kinetic_pose import KineticDynamicEncoder
from .motion_tokens import forward_kinematics, pose_to_bones, trial_bone_lengths
from .nets import LocalTemporalBlock, TemporalTransformer


DISTAL_JOINTS = tuple(sorted(set(
    C.JOINT_GROUPS["head"]
    + C.JOINT_GROUPS["left_arm"][-1:]
    + C.JOINT_GROUPS["right_arm"][-1:]
    + C.JOINT_GROUPS["left_leg"][-2:]
    + C.JOINT_GROUPS["right_leg"][-2:]
)))


def _zero_last(module: nn.Sequential) -> None:
    layer = module[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("the final layer must be linear")
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def _masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[..., None].to(features.dtype)
    mean = (features * weight).sum(1) / weight.sum(1).clamp_min(1.0)
    maximum = features.masked_fill(~mask[..., None], -1e4).amax(1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return 0.5 * (mean + maximum)


class MultiScaleMotionFusion(nn.Module):
    """Fuse posture context with short, medium, and global CSI motion."""

    def __init__(self, static_hidden: int, hidden: int, heads: int,
                 dropout: float):
        super().__init__()
        self.static_projection = nn.Sequential(
            nn.LayerNorm(static_hidden), nn.Linear(static_hidden, hidden), nn.GELU()
        )
        self.short = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout) for dilation in (1, 2)
        )
        self.medium = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout) for dilation in (4, 8)
        )
        self.global_context = TemporalTransformer(
            hidden, layers=1, heads=heads, dropout=dropout
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid()
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        static = self.static_projection(static)
        short = dynamic
        for block in self.short:
            short = block(short)
            short = short * mask[..., None].to(short.dtype)
        medium = short
        for block in self.medium:
            medium = block(medium)
            medium = medium * mask[..., None].to(medium.dtype)
        global_context = self.global_context(medium, mask)
        motion = self.motion_projection(torch.cat(
            (dynamic, short, medium, global_context), dim=-1
        ))
        joined = torch.cat((static, motion), dim=-1)
        gate = self.gate(joined)
        fused = gate * static + (1.0 - gate) * motion + self.residual(joined)
        fused = self.output_norm(fused) * mask[..., None].to(fused.dtype)
        return fused, motion, static, gate


class DynamicMotionPoseNet(nn.Module):
    """Predict the full pelvis-relative trajectory without a coarse pose blend.

    The frozen P2 model supplies a CSI-derived direction anchor, posture context,
    and stable classification logits. Dynamic CSI rotates parent-relative bone
    directions directly, so a fall is not limited by a bounded Cartesian residual
    around a standing pose.
    """

    def __init__(self, base_model: nn.Module, direction_prior: torch.Tensor,
                 bone_lengths: torch.Tensor, hidden: int = 128,
                 dynamic_layers: int = 1, heads: int = 4,
                 dropout: float = 0.08, endpoint_scale: float = 0.05,
                 shape_scale: float = 0.15):
        super().__init__()
        if direction_prior.shape != (C.N_JOINTS, 3):
            raise ValueError("direction_prior must have shape [22, 3]")
        if bone_lengths.shape != (C.N_JOINTS,):
            raise ValueError("bone_lengths must have shape [22]")
        self.base = base_model
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.hidden = int(hidden)
        self.endpoint_scale = float(endpoint_scale)
        self.shape_scale = float(shape_scale)
        self.dynamic = KineticDynamicEncoder(
            base_model.norm, hidden=hidden, temporal_layers=dynamic_layers,
            heads=heads, dropout=dropout,
        )
        self.fusion = MultiScaleMotionFusion(
            int(base_model.hidden), hidden, heads, dropout
        )
        self.action_embedding = nn.Parameter(torch.empty(C.N_CLASSES, hidden))
        self.risk_embedding = nn.Parameter(torch.empty(C.N_RISK, hidden))
        nn.init.normal_(self.action_embedding, std=0.02)
        nn.init.normal_(self.risk_embedding, std=0.02)
        self.semantic_modulation = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden * 2),
        )
        self.semantic_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid(),
        )
        self.anchor_direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_JOINTS * 3),
        )
        self.motion_direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_JOINTS * 3),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        self.endpoint_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, len(DISTAL_JOINTS) * 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS),
        )
        self.phase_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.contact_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_CONTACT_JOINTS)
        )
        self.motion_profile_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 7)
        )

        self.class_temporal = nn.ModuleList(
            LocalTemporalBlock(hidden * 2, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, C.N_RISK),
        )
        for head in (
            self.anchor_direction_head, self.motion_direction_head,
            self.velocity_head, self.endpoint_head, self.shape_head,
            self.class_head, self.risk_head,
        ):
            _zero_last(head)
        nn.init.zeros_(self.semantic_modulation[-1].weight)
        nn.init.zeros_(self.semantic_modulation[-1].bias)

        prior = F.normalize(direction_prior.float(), dim=-1)
        prior[C.ROOT_JOINT] = 0.0
        lengths = bone_lengths.float().clone()
        lengths[C.ROOT_JOINT] = 0.0
        self.register_buffer("direction_prior", prior)
        self.register_buffer("bone_length_prior", lengths)
        self.register_buffer("danger_logit_bias", torch.zeros(()))

    def set_danger_logit_bias(self, value: float) -> None:
        self.danger_logit_bias.fill_(float(value))

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.dynamic.norm.eval()
        return self

    def pose_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and not name.startswith((
                "class_temporal.", "class_head.", "risk_head."
            )):
                yield parameter

    def classification_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(("class_temporal.", "class_head.", "risk_head.")):
                yield parameter

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("base.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"unknown KP11 weights: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                pose_anchor: torch.Tensor | None = None) -> dict:
        frame_mask = link_mask.any(-1)
        with torch.no_grad():
            base = self.base(csi, link_mask)
        dynamic, activity = self.dynamic(csi, link_mask)
        fused, motion, static, fusion_gate = self.fusion(
            base["temporal_features"], dynamic, frame_mask
        )
        action_probability = torch.softmax(base["class_logits"].detach(), dim=-1)
        risk_probability = torch.softmax(base["risk_logits"].detach(), dim=-1)
        semantic = (
            action_probability @ self.action_embedding
            + risk_probability @ self.risk_embedding
        )
        semantic = semantic[:, None].expand(-1, fused.shape[1], -1)
        semantic_input = torch.cat((fused, semantic), dim=-1)
        scale, shift = self.semantic_modulation(semantic_input).chunk(2, dim=-1)
        semantic_gate = self.semantic_gate(semantic_input)
        conditioned = (
            fused * (1.0 + semantic_gate * torch.tanh(scale))
            + semantic_gate * shift
        )
        conditioned *= frame_mask[..., None].to(conditioned.dtype)
        batch, frames = csi.shape[:2]
        anchor_delta = self.anchor_direction_head(conditioned).reshape(
            batch, frames, C.N_JOINTS, 3
        )
        motion_delta = self.motion_direction_head(motion).reshape(
            batch, frames, C.N_JOINTS, 3
        )
        base_pose = (
            base["pose_rel"] if pose_anchor is None else pose_anchor
        ).detach()
        if base_pose.shape != (batch, frames, C.N_JOINTS, 3):
            raise ValueError("pose anchor must have shape [B,T,22,3]")
        base_directions, base_frame_lengths = pose_to_bones(base_pose)
        directions = F.normalize(
            base_directions
            + torch.tanh(anchor_delta)
            + torch.tanh(motion_delta),
            dim=-1,
        )
        directions[:, :, C.ROOT_JOINT] = 0.0

        trial_static = _masked_pool(conditioned, frame_mask)
        shape_delta = self.shape_scale * torch.tanh(self.shape_head(trial_static))
        base_lengths = trial_bone_lengths(base_pose, frame_mask)
        minimum = 0.7 * self.bone_length_prior[None]
        maximum = 1.3 * self.bone_length_prior[None]
        base_lengths = torch.maximum(torch.minimum(base_lengths, maximum), minimum)
        lengths = base_lengths * (1.0 + shape_delta)
        lengths[:, C.ROOT_JOINT] = 0.0
        length_ratio = lengths / base_lengths.clamp_min(1e-6)
        frame_lengths = base_frame_lengths * length_ratio[:, None]
        frame_lengths[:, :, C.ROOT_JOINT] = 0.0
        kinematic_pose = forward_kinematics(directions, frame_lengths)
        endpoint_delta = self.endpoint_scale * torch.tanh(
            self.endpoint_head(conditioned).reshape(
                batch, frames, len(DISTAL_JOINTS), 3
            )
        )
        pose = kinematic_pose.clone()
        pose[:, :, list(DISTAL_JOINTS)] += endpoint_delta
        pose = pose * frame_mask[..., None, None].to(pose.dtype)

        detached_class_features = torch.cat(
            (static.detach(), motion.detach()), dim=-1
        )
        for block in self.class_temporal:
            detached_class_features = block(detached_class_features)
            detached_class_features *= frame_mask[..., None].to(
                detached_class_features.dtype
            )
        classification = _masked_pool(detached_class_features, frame_mask)
        class_logits = base["class_logits"].detach() + self.class_head(classification)
        risk_logits = base["risk_logits"].detach() + self.risk_head(classification)
        risk_logits = risk_logits.clone()
        risk_logits[:, 2] += self.danger_logit_bias

        valid_weight = frame_mask[..., None, None].to(pose.dtype)
        velocity = self.velocity_head(motion).reshape(
            batch, frames, C.N_JOINTS, 3
        ) * valid_weight
        return {
            "pose_rel": pose,
            "pose_anchor": forward_kinematics(base_directions, base_frame_lengths)
            * valid_weight,
            "kinematic_pose": kinematic_pose * valid_weight,
            "bone_direction": directions * valid_weight,
            "bone_lengths": lengths,
            "endpoint_delta": endpoint_delta,
            "kinetic_velocity": velocity,
            "phase_logits": self.phase_head(conditioned),
            "contact_logits": self.contact_head(conditioned),
            "motion_profile": F.softplus(self.motion_profile_head(motion)),
            "class_logits": class_logits,
            "risk_logits": risk_logits,
            "root": base["root"].detach(),
            "kinetic_activity": activity,
            "dynamic_features": dynamic,
            "motion_features": motion,
            "fused_features": fused,
            "fusion_gate": fusion_gate,
            "semantic_gate": semantic_gate,
            "action_probability": action_probability,
            "risk_probability": risk_probability,
        }

    def n_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)
