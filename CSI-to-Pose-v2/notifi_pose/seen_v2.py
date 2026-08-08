"""Seen-first CSI reconstruction with phase, kinematics, root flow, and injury heads."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from . import losses as L
from .motion_first import MotionFirstEncoder, interpolate_keyframes, temporal_keyframes
from .nets import LocalTemporalBlock
from .v3 import rotation_6d_to_matrix


INJURY_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "head", "left_wrist", "right_wrist",
)
INJURY_JOINTS = tuple(C.JOINT_INDEX[name] for name in INJURY_JOINT_NAMES)
N_INJURY_JOINTS = len(INJURY_JOINTS)


def _zero_last(module: nn.Sequential) -> None:
    layer = module[-1]
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    total = (values * weight[..., None]).sum(1)
    return total / weight.sum(1, keepdim=True).clamp_min(1)


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


def injury_targets(pose: torch.Tensor, root: torch.Tensor,
                   valid: torch.Tensor, risk: torch.Tensor) -> dict:
    """Derive CSI-independent floor/contact/injury supervision from GT pose."""
    absolute = pose + root[:, :, None]
    selected = absolute[:, :, INJURY_JOINTS]
    _, floor = L.contact_targets(pose, root, valid)
    height = selected[..., 1] - floor[:, None, None]
    speed = torch.zeros_like(height)
    speed[:, 1:] = torch.linalg.vector_norm(
        selected[:, 1:] - selected[:, :-1], dim=-1
    ) * C.TARGET_FPS
    contact = (height < 0.12) & valid[..., None]
    impact = L.impact_window(pose, root, valid, risk)
    first_contact = torch.full(
        (len(pose),), -1, dtype=torch.long, device=pose.device
    )
    for item in range(len(pose)):
        if int(risk[item]) != 2:
            continue
        frames = torch.nonzero(impact[item], as_tuple=False).flatten()
        if len(frames) == 0:
            continue
        local_contact = contact[item, frames]
        occupied = torch.nonzero(local_contact.any(-1), as_tuple=False).flatten()
        frame = frames[int(occupied[0])] if len(occupied) else frames[len(frames) // 2]
        first_contact[item] = torch.argmin(height[item, frame])
    return {
        "injury_contact": contact,
        "joint_speed": speed,
        "first_contact": first_contact,
        "first_contact_valid": first_contact >= 0,
        "floor_height": floor,
        "impact_mask": impact,
    }


class SeenReconstructionV2Net(nn.Module):
    """Identity-initialized refinement of validated seen baseline and motion models."""

    def __init__(self, baseline: nn.Module, motion: MotionFirstEncoder,
                 hidden: int = 128, dropout: float = 0.05):
        super().__init__()
        self.baseline = baseline
        self.motion = motion
        self.hidden = hidden
        baseline_hidden = int(getattr(baseline, "hidden", hidden))
        condition_size = 1 + 1 + 4 + 1 + C.N_CLASSES + C.N_RISK
        self.base_projection = nn.Linear(baseline_hidden, hidden)
        self.motion_projection = nn.Linear(motion.hidden, hidden)
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_size, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.motion_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid()
        )
        self.action_embedding = nn.Parameter(torch.zeros(C.N_CLASSES, hidden))
        nn.init.normal_(self.action_embedding, std=0.02)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.fusion_blocks = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )

        self.rotation_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS * 6),
        )
        self.high_pose_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS * 3),
        )
        self.pose_scale_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        _zero_last(self.rotation_head)
        _zero_last(self.high_pose_head)
        nn.init.zeros_(self.pose_scale_head[-1].weight)
        nn.init.zeros_(self.pose_scale_head[-1].bias)

        self.root_anchor_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.root_step_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.root_scale_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        _zero_last(self.root_anchor_head)
        _zero_last(self.root_step_head)
        nn.init.zeros_(self.root_scale_head[-1].weight)
        nn.init.zeros_(self.root_scale_head[-1].bias)

        self.phase_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.impact_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.foot_contact_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.injury_contact_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_INJURY_JOINTS)
        )
        self.joint_speed_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_INJURY_JOINTS)
        )
        self.first_contact_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_INJURY_JOINTS)
        )
        self.floor_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)
        nn.init.zeros_(self.impact_head[-1].weight)
        nn.init.zeros_(self.impact_head[-1].bias)
        self.register_buffer(
            "rotation_strength", torch.tensor(1.0), persistent=False
        )
        self.register_buffer(
            "high_pose_strength", torch.tensor(1.0), persistent=False
        )
        self.register_buffer(
            "root_refinement_strength", torch.tensor(1.0), persistent=False
        )

        self.backbones_trainable = False
        self.set_partial_finetune(False)

    def set_calibration(self, rotation: float = 1.0, high_pose: float = 1.0,
                        root: float = 1.0) -> None:
        values = (rotation, high_pose, root)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("calibration strengths must be between 0 and 1")
        self.rotation_strength.fill_(rotation)
        self.high_pose_strength.fill_(high_pose)
        self.root_refinement_strength.fill_(root)

    def set_partial_finetune(self, enabled: bool) -> None:
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)
        if enabled:
            baseline_core = self.baseline
            if hasattr(baseline_core, "pose_model"):
                baseline_core = baseline_core.pose_model
            if hasattr(baseline_core, "baseline"):
                baseline_core = baseline_core.baseline
            baseline_last = baseline_core.temporal.transformer.layers[-1]
            motion_last = self.motion.temporal.transformer.layers[-1]
            for parameter in baseline_last.parameters():
                parameter.requires_grad_(True)
            for parameter in motion_last.parameters():
                parameter.requires_grad_(True)
        self.backbones_trainable = bool(enabled)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        self.motion.eval()
        return self

    def _backbone_outputs(self, csi: torch.Tensor,
                          link_mask: torch.Tensor) -> tuple[dict, dict]:
        context = nullcontext() if self.backbones_trainable else torch.no_grad()
        with context:
            return self.baseline(csi, link_mask), self.motion(csi, link_mask)

    def _fuse(self, baseline: dict, motion: dict,
              frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_projection(baseline["temporal_features"])
        moving = torch.sigmoid(motion["moving_logits"])[..., None]
        phase = torch.softmax(motion["phase_logits"], dim=-1)
        impact = torch.sigmoid(motion["impact_logits"])[..., None]
        motion_class = torch.softmax(motion["class_logits"], dim=-1)
        baseline_class = torch.softmax(baseline["class_logits"], dim=-1)
        action_probability = 0.5 * (motion_class + baseline_class)
        risk = torch.softmax(motion["risk_logits"], dim=-1)
        condition = torch.cat((
            motion["speed_log"][..., None], moving, phase, impact,
            motion_class[:, None].expand(-1, base.shape[1], -1),
            risk[:, None].expand(-1, base.shape[1], -1),
        ), dim=-1)
        motion_feature = self.motion_projection(motion["temporal_features"])
        gate = self.motion_gate(torch.cat((base, motion_feature), dim=-1))
        action = action_probability @ self.action_embedding
        fused = self.fusion_norm(
            base + gate * motion_feature + self.condition_projection(condition)
            + action[:, None]
        )
        for block in self.fusion_blocks:
            fused = block(fused)
        return fused * frame_mask[..., None], action_probability

    def _decode_pose(self, coarse: torch.Tensor, fused: torch.Tensor,
                     frame_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        key_features, _ = temporal_keyframes(fused, frame_mask, stride=4)
        rotation_delta = interpolate_keyframes(
            self.rotation_head(key_features).reshape(
                *key_features.shape[:2], C.N_JOINTS, 6
            ), fused.shape[1]
        )
        identity = rotation_delta.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        rotation = rotation_6d_to_matrix(rotation_delta + identity)
        coarse_bones = _local_bones(coarse)
        rotated = torch.matmul(rotation, coarse_bones.unsqueeze(-1)).squeeze(-1)
        pose_scale = 0.10 + 0.80 * torch.sigmoid(self.pose_scale_head(fused))
        rotation_scale = pose_scale * self.rotation_strength
        mixed = coarse_bones + rotation_scale[..., None] * (rotated - coarse_bones)
        length = torch.linalg.vector_norm(coarse_bones, dim=-1, keepdim=True)
        mixed = F.normalize(mixed, dim=-1) * length
        mixed[:, :, C.ROOT_JOINT] = 0.0
        low_pose = _forward_kinematics(mixed)

        high = 0.02 * torch.tanh(self.high_pose_head(fused))
        high = high.reshape(*fused.shape[:2], C.N_JOINTS, 3)
        high = high * frame_mask[:, :, None, None]
        high_scale = pose_scale * self.high_pose_strength
        pose = low_pose + high_scale[..., None] * high
        pose = pose - pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        return pose, {
            "pose_low": low_pose,
            "pose_high_residual": high,
            "pose_scale": rotation_scale.squeeze(-1),
            "high_pose_scale": high_scale.squeeze(-1),
            "rotation_6d_delta": rotation_delta,
        }

    def _decode_root(self, coarse: torch.Tensor, fused: torch.Tensor,
                     frame_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pooled = _masked_mean(fused, frame_mask)
        anchor_delta = 0.40 * torch.tanh(self.root_anchor_head(pooled))
        key_features, _ = temporal_keyframes(fused, frame_mask, stride=4)
        residual_step = interpolate_keyframes(
            0.005 * torch.tanh(self.root_step_head(key_features)), fused.shape[1]
        )
        residual_step[:, 0] = 0.0
        residual_step = residual_step * frame_mask[..., None]
        root_scale = 0.10 + 0.80 * torch.sigmoid(self.root_scale_head(fused))
        base_step = torch.zeros_like(coarse)
        base_step[:, 1:] = coarse[:, 1:] - coarse[:, :-1]
        effective_root_scale = root_scale * self.root_refinement_strength
        step = base_step + effective_root_scale * residual_step
        root = (
            coarse[:, :1]
            + self.root_refinement_strength * anchor_delta[:, None]
            + torch.cumsum(step, dim=1)
        )
        return root, {
            "root_anchor_delta": anchor_delta,
            "root_step_residual": residual_step,
            "root_velocity_mps": step * C.TARGET_FPS,
            "root_scale": effective_root_scale.squeeze(-1),
        }

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        baseline, motion = self._backbone_outputs(csi, link_mask)
        frame_mask = link_mask.any(-1)
        fused, action_probability = self._fuse(baseline, motion, frame_mask)
        pose, pose_extra = self._decode_pose(
            baseline["pose_rel"], fused, frame_mask
        )
        root, root_extra = self._decode_root(
            baseline["root"], fused, frame_mask
        )
        pooled = _masked_mean(fused, frame_mask)
        impact_score = motion["impact_logits"].masked_fill(~frame_mask, -1e4)
        impact_attention = torch.softmax(impact_score, dim=1)
        impact_feature = (fused * impact_attention[..., None]).sum(1)
        output = dict(baseline)
        output.update({
            "pose_coarse": baseline["pose_rel"],
            "root_coarse": baseline["root"],
            "pose_rel": pose,
            "root": root,
            "motion": motion["speed_log"],
            "phase_logits": motion["phase_logits"] + self.phase_head(fused),
            "impact_logits": motion["impact_logits"] + self.impact_head(fused).squeeze(-1),
            "contact_logits": self.foot_contact_head(fused),
            "injury_contact_logits": self.injury_contact_head(fused),
            "joint_impact_speed": F.softplus(self.joint_speed_head(fused)),
            "first_contact_logits": self.first_contact_head(impact_feature),
            "floor_height": self.floor_head(pooled).squeeze(-1),
            "action_probability": action_probability,
            "temporal_features": fused,
            "baseline_features": baseline["temporal_features"],
            "motion_features": motion["temporal_features"],
            **pose_extra,
            **root_extra,
        })
        return output


def weighted_seen_v2_loss(output: dict, batch: dict,
                          teacher_output: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Metric, trajectory, phase, contact, and injury objective with trial quality."""
    valid = batch["valid"].bool()
    quality = batch.get(
        "quality_weight", torch.ones(len(valid), device=valid.device)
    ).to(output["pose_rel"].dtype)
    speed, pair_valid = L.target_motion(batch["pose_rel"], batch["root"], valid)
    impact = L.impact_window(
        batch["pose_rel"], batch["root"], valid, batch["risk_id"]
    )
    frame_weight = 1.0 + 2.0 * impact.to(output["pose_rel"].dtype)
    pose = L.smooth_l1_per_sample(
        output["pose_rel"], batch["pose_rel"], valid, frame_weight=frame_weight
    )
    root = L.smooth_l1_per_sample(
        output["root"], batch["root"], valid, frame_weight=frame_weight
    )
    bone = L.BoneLoss().to(output["pose_rel"].device).per_sample(
        output["pose_rel"], batch["pose_rel"], valid
    )
    displacement = L.displacement_per_sample(
        output["pose_rel"], output["root"], batch["pose_rel"], batch["root"], valid
    )
    lag = 5
    interval = valid[:, lag:] & valid[:, :-lag]
    predicted_amplitude = torch.linalg.vector_norm(
        output["pose_rel"][:, lag:] - output["pose_rel"][:, :-lag], dim=-1
    ).mean(-1) * (C.TARGET_FPS / lag)
    target_amplitude = torch.linalg.vector_norm(
        batch["pose_rel"][:, lag:] - batch["pose_rel"][:, :-lag], dim=-1
    ).mean(-1) * (C.TARGET_FPS / lag)
    motion_amplitude = L.smooth_l1_per_sample(
        predicted_amplitude, target_amplitude, interval, beta=0.10
    )

    root_target_velocity = torch.zeros_like(batch["root"])
    root_target_velocity[:, 1:] = (
        batch["root"][:, 1:] - batch["root"][:, :-1]
    ) * C.TARGET_FPS
    root_velocity = L.smooth_l1_per_sample(
        output["root_velocity_mps"], root_target_velocity, valid, beta=0.20
    )

    phase_target, phase_mask = L.phase_targets(speed, valid, batch["risk_id"])
    phase_element = F.cross_entropy(
        output["phase_logits"].transpose(1, 2), phase_target, reduction="none"
    )
    phase = L.masked_per_sample(phase_element, phase_mask)
    impact_target = impact.to(output["impact_logits"].dtype)
    impact_positive = impact_target[valid].sum()
    impact_negative = valid.sum() - impact_positive
    impact_weight = (impact_negative / impact_positive.clamp_min(1.0)).clamp(1.0, 20.0)
    impact_element = F.binary_cross_entropy_with_logits(
        output["impact_logits"], impact_target, reduction="none",
        pos_weight=impact_weight,
    )
    impact_class = L.masked_per_sample(impact_element, valid)

    feet_contact, _ = L.contact_targets(
        batch["pose_rel"], batch["root"], valid
    )
    foot_element = F.binary_cross_entropy_with_logits(
        output["contact_logits"], feet_contact.to(output["contact_logits"].dtype),
        reduction="none",
    )
    foot_contact = L.masked_per_sample(foot_element, valid)
    injury = injury_targets(
        batch["pose_rel"], batch["root"], valid, batch["risk_id"]
    )
    injury_target = injury["injury_contact"].to(
        output["injury_contact_logits"].dtype
    )
    injury_mask = valid[..., None].expand_as(injury_target)
    injury_positive = injury_target[injury_mask].sum()
    injury_negative = injury_mask.sum() - injury_positive
    injury_weight = (injury_negative / injury_positive.clamp_min(1.0)).clamp(1.0, 20.0)
    injury_element = F.binary_cross_entropy_with_logits(
        output["injury_contact_logits"],
        injury_target, reduction="none", pos_weight=injury_weight,
    )
    injury_contact = L.masked_per_sample(injury_element, valid)
    impact_speed = L.smooth_l1_per_sample(
        output["joint_impact_speed"], injury["joint_speed"],
        injury["impact_mask"], beta=0.20,
    )
    first_contact = torch.zeros_like(pose)
    first_valid = injury["first_contact_valid"]
    if first_valid.any():
        first_contact[first_valid] = F.cross_entropy(
            output["first_contact_logits"][first_valid],
            injury["first_contact"][first_valid], reduction="none",
        )
    floor = (output["floor_height"] - injury["floor_height"]).abs()

    class_loss = F.cross_entropy(
        output["class_logits"], batch["class_id"], reduction="none"
    )
    risk_loss = F.cross_entropy(
        output["risk_logits"], batch["risk_id"], reduction="none"
    )
    high_regularization = (
        output["pose_high_residual"] / 0.02
    ).square().mean((1, 2, 3))
    root_step_regularization = (
        output["root_step_residual"] / 0.005
    ).square().mean((1, 2))
    scale_regularization = (
        (output["pose_scale"] - 0.5).square().mean(1)
        + (output["root_scale"] - 0.5).square().mean(1)
    )
    distillation = torch.zeros_like(pose)
    if teacher_output is not None:
        base_error = (
            output["baseline_features"] - teacher_output["baseline_features"]
        ).square().mean(-1)
        motion_error = (
            output["motion_features"] - teacher_output["motion_features"]
        ).square().mean(-1)
        distillation = L.masked_per_sample(base_error + motion_error, valid)

    per_sample = (
        pose + 1.00 * root + 0.05 * bone + 0.15 * displacement
        + 0.10 * motion_amplitude
        + 0.10 * root_velocity + 0.05 * phase + 0.05 * impact_class
        + 0.03 * foot_contact + 0.05 * injury_contact
        + 0.05 * first_contact + 0.03 * impact_speed + 0.05 * floor
        + 0.02 * class_loss + 0.02 * risk_loss
        + 0.005 * high_regularization + 0.002 * root_step_regularization
        + 0.01 * scale_regularization
        + 0.02 * distillation
    )
    total = (per_sample * quality).sum() / quality.sum().clamp_min(1e-6)
    parts = {
        "total": float(total.detach()),
        "pose": float(pose.mean().detach()),
        "root": float(root.mean().detach()),
        "bone": float(bone.mean().detach()),
        "displacement": float(displacement.mean().detach()),
        "motion_amplitude": float(motion_amplitude.mean().detach()),
        "root_velocity": float(root_velocity.mean().detach()),
        "phase": float(phase.mean().detach()),
        "impact_class": float(impact_class.mean().detach()),
        "foot_contact": float(foot_contact.mean().detach()),
        "injury_contact": float(injury_contact.mean().detach()),
        "first_contact": float(first_contact[first_valid].mean().detach()) if first_valid.any() else 0.0,
        "impact_speed": float(impact_speed.mean().detach()),
        "floor": float(floor.mean().detach()),
        "distillation": float(distillation.mean().detach()),
        "quality_mean": float(quality.mean().detach()),
    }
    return total, parts
