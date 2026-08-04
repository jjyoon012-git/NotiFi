"""Alignment-robust full-trajectory refinement for fall reconstruction V9."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from . import losses as L
from .impact_event import raw_csi_event_features as raw_csi_motion_features
from .nets import LocalTemporalBlock
from .seen_v2 import _forward_kinematics, _local_bones
from .v3 import rotation_6d_to_matrix


MOTION_GROUPS = tuple(C.JOINT_GROUPS.values())


def _masked_temporal_mean(values: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)[..., None]
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def _masked_per_sample(values: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.dim() < values.dim():
        weight = weight.unsqueeze(-1)
    axes = tuple(range(1, values.dim()))
    return (values * weight).sum(axes) / weight.expand_as(values).sum(axes).clamp_min(1)


def _quality_mean(values: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    quality = quality.to(values.dtype)
    return (values * quality).sum() / quality.sum().clamp_min(1e-6)


def _body_group_speed(pose: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    absolute = pose + root[:, :, None]
    velocity = torch.zeros_like(absolute)
    velocity[:, 1:] = (absolute[:, 1:] - absolute[:, :-1]) * C.TARGET_FPS
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    return torch.stack(
        [speed[:, :, group].mean(-1) for group in MOTION_GROUPS], dim=-1
    )


def _masked_smooth(values: torch.Tensor, valid: torch.Tensor,
                   width: int = 5) -> torch.Tensor:
    """Smooth valid frames without leaking zero padding into sequence ends."""
    if width <= 1:
        return values
    batch, frames = values.shape[:2]
    flat = values.reshape(batch, frames, -1).transpose(1, 2)
    weight = valid.to(values.dtype)[:, None]
    numerator = F.avg_pool1d(
        flat * weight, width, stride=1, padding=width // 2
    )
    denominator = F.avg_pool1d(
        weight, width, stride=1, padding=width // 2
    )
    smoothed = numerator / denominator.clamp_min(1e-6)
    return smoothed.transpose(1, 2).reshape_as(values)


def trajectory_descriptor(pose: torch.Tensor, root: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
    """Compact motion descriptor used only by the bounded alignment loss."""
    valid = valid.bool()
    absolute = _masked_smooth(pose + root[:, :, None], valid, 5)
    root_smooth = absolute[:, :, C.ROOT_JOINT]
    first = valid.to(torch.long).argmax(1)
    first_root = root_smooth[
        torch.arange(len(root_smooth), device=root.device), first
    ]
    root_delta = (root_smooth - first_root[:, None]).clamp(-2.0, 2.0)

    head = C.JOINT_INDEX["head"]
    left_shoulder = C.JOINT_INDEX["left_shoulder"]
    right_shoulder = C.JOINT_INDEX["right_shoulder"]
    torso = F.normalize(
        absolute[:, :, head] - absolute[:, :, C.ROOT_JOINT], dim=-1
    )
    shoulder = F.normalize(
        absolute[:, :, right_shoulder] - absolute[:, :, left_shoulder], dim=-1
    )

    lag = 3
    velocity = torch.zeros_like(absolute)
    velocity[:, lag:] = (
        absolute[:, lag:] - absolute[:, :-lag]
    ) * (C.TARGET_FPS / lag)
    root_velocity = velocity[:, :, C.ROOT_JOINT].clamp(-3.0, 3.0) / 2.0
    joint_speed = torch.linalg.vector_norm(velocity, dim=-1)
    group_speed = torch.stack(
        [joint_speed[:, :, group].mean(-1) for group in MOTION_GROUPS], dim=-1
    ).clamp(0.0, 4.0) / 2.0
    descriptor = torch.cat((
        root_delta, torso, shoulder, root_velocity, group_speed,
    ), dim=-1)
    return descriptor * valid[..., None]


def _shifted(values: torch.Tensor, valid: torch.Tensor,
             shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    shifted = torch.zeros_like(values)
    shifted_valid = torch.zeros_like(valid)
    if shift >= 0:
        if shift < values.shape[1]:
            shifted[:, :values.shape[1] - shift] = values[:, shift:]
            shifted_valid[:, :values.shape[1] - shift] = valid[:, shift:]
    else:
        amount = -shift
        if amount < values.shape[1]:
            shifted[:, amount:] = values[:, :values.shape[1] - amount]
            shifted_valid[:, amount:] = valid[:, :values.shape[1] - amount]
    return shifted, shifted_valid


def bounded_piecewise_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    max_shift: int = 15,
    segments: int = 8,
    shift_penalty: float = 0.03,
    transition_penalty: float = 0.06,
) -> torch.Tensor:
    """Near-identity piecewise alignment with a smooth offset path.

    Each long temporal segment may select a small offset. Dynamic programming
    penalizes absolute offset and changes between adjacent segment offsets, so
    the loss cannot freely reorder or arbitrarily warp a fall sequence.
    """
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if segments < 1:
        raise ValueError("segments must be positive")
    shifts = torch.arange(
        -max_shift, max_shift + 1, device=predicted.device,
        dtype=predicted.dtype,
    )
    frame_costs = []
    frame_masks = []
    for shift in range(-max_shift, max_shift + 1):
        shifted_target, shifted_valid = _shifted(target, valid, shift)
        mask = valid & shifted_valid
        cost = F.smooth_l1_loss(
            predicted, shifted_target, beta=0.10, reduction="none"
        ).mean(-1)
        frame_costs.append(cost)
        frame_masks.append(mask)
    frame_cost = torch.stack(frame_costs, dim=-1)
    frame_mask = torch.stack(frame_masks, dim=-1)

    boundaries = torch.linspace(
        0, predicted.shape[1], segments + 1, device=predicted.device
    ).round().to(torch.long)
    segment_costs = []
    for index in range(segments):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        mask = frame_mask[:, start:stop].to(frame_cost.dtype)
        cost = (
            frame_cost[:, start:stop] * mask
        ).sum(1) / mask.sum(1).clamp_min(1.0)
        segment_costs.append(cost)
    costs = torch.stack(segment_costs, dim=1)

    normalizer = float(max(max_shift, 1))
    absolute = shifts.abs() / normalizer
    transition = (shifts[:, None] - shifts[None]).abs() / normalizer
    state = costs[:, 0] + 2.0 * shift_penalty * absolute
    for segment in range(1, segments):
        candidates = state[:, :, None] + transition_penalty * transition[None]
        state = costs[:, segment] + candidates.min(1).values
    state = state + shift_penalty * absolute
    return state.min(-1).values / segments


class AlignmentRobustTrajectoryNet(nn.Module):
    """Identity-initialized full-trajectory pose/root refiner for V9."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.hidden = hidden
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        input_size = hidden * 3 + 4 + 3 + len(MOTION_GROUPS) + C.N_RISK
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(encoder_layer, 1)
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS * 6),
        )
        self.root_anchor_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.root_step_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_CLASSES)
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_RISK)
        )
        for head in (self.rotation_head, self.root_anchor_head, self.root_step_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer("pose_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("root_strength", torch.tensor(1.0), persistent=False)

    def set_calibration(self, pose: float = 1.0, root: float = 1.0) -> None:
        if not 0.0 <= pose <= 1.0 or not 0.0 <= root <= 1.0:
            raise ValueError("calibration strengths must be between 0 and 1")
        self.pose_strength.fill_(pose)
        self.root_strength.fill_(root)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            base = self.base(csi, link_mask)
        valid = link_mask.any(-1)
        pose = base["pose_rel"]
        root = base["root"]
        root_velocity = torch.zeros_like(root)
        root_velocity[:, 1:] = (root[:, 1:] - root[:, :-1]) * C.TARGET_FPS
        group_speed = _body_group_speed(pose, root)
        risk = torch.softmax(base["risk_logits"], dim=-1)
        risk = risk[:, None].expand(-1, pose.shape[1], -1)
        raw_motion = raw_csi_motion_features(csi, link_mask)
        feature = self.input_projection(torch.cat((
            base["temporal_features"], base["motion_features"],
            base["temporal_features_v3"], raw_motion, root_velocity,
            group_speed, risk,
        ), dim=-1))
        feature = feature * valid[..., None]
        for block in self.temporal:
            feature = block(feature) * valid[..., None]
        feature = self.context(
            feature, src_key_padding_mask=~valid
        ) * valid[..., None]

        rotation_delta = self.rotation_head(feature).reshape(
            *feature.shape[:2], C.N_JOINTS, 6
        )
        identity = rotation_delta.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        rotation = rotation_6d_to_matrix(rotation_delta + identity)
        bones = _local_bones(pose)
        rotated = torch.matmul(rotation, bones.unsqueeze(-1)).squeeze(-1)
        mixed = bones + self.pose_strength * (rotated - bones)
        length = torch.linalg.vector_norm(bones, dim=-1, keepdim=True)
        mixed = F.normalize(mixed, dim=-1) * length
        mixed[:, :, C.ROOT_JOINT] = 0.0
        refined_pose = _forward_kinematics(mixed)

        anchor_delta = 0.30 * torch.tanh(
            self.root_anchor_head(_masked_temporal_mean(feature, valid))
        )
        step_delta = 0.006 * torch.tanh(self.root_step_head(feature))
        step_delta[:, 0] = 0.0
        step_delta = step_delta * valid[..., None]
        base_step = torch.zeros_like(root)
        base_step[:, 1:] = root[:, 1:] - root[:, :-1]
        refined_root = (
            root[:, :1] + self.root_strength * anchor_delta[:, None]
            + torch.cumsum(base_step + self.root_strength * step_delta, dim=1)
        )
        pooled = _masked_temporal_mean(feature, valid)

        output = dict(base)
        output.update({
            "pose_rel": refined_pose,
            "root": refined_root,
            "pose_v7": pose,
            "root_v7": root,
            "rotation_6d_delta_v9": rotation_delta,
            "root_anchor_delta_v9": anchor_delta,
            "root_step_delta_v9": step_delta,
            "temporal_features_v9": feature,
            "base_class_logits": base["class_logits"],
            "base_risk_logits": base["risk_logits"],
            "class_logits": self.class_head(pooled),
            "risk_logits": self.risk_head(pooled),
        })
        return output


def _endpoint_error(predicted: torch.Tensor, target: torch.Tensor,
                    valid: torch.Tensor, width: int = 15) -> torch.Tensor:
    rows = []
    for item in range(len(predicted)):
        frames = torch.nonzero(valid[item], as_tuple=False).flatten()
        if len(frames) == 0:
            rows.append(predicted.new_zeros(()))
            continue
        selected = frames[-width:]
        rows.append(torch.linalg.vector_norm(
            predicted[item, selected] - target[item, selected], dim=-1
        ).mean())
    return torch.stack(rows)


def trajectory_reconstruction_loss(
    output: dict,
    batch: dict,
    alignment_weight: float = 0.15,
    max_shift: int = 15,
    class_weight: torch.Tensor | None = None,
    risk_weight: torch.Tensor | None = None,
    lambda_class: float = 0.0,
    lambda_risk: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Full-sequence objective with no impact-frame or first-contact target."""
    valid = batch["valid"].bool()
    quality = batch.get(
        "quality_weight", torch.ones(len(valid), device=valid.device)
    ).to(output["pose_rel"].dtype)
    danger = batch["risk_id"].eq(2)
    sample_weight = quality * torch.where(
        danger, quality.new_tensor(2.0), quality.new_tensor(1.0)
    )

    pose = L.smooth_l1_per_sample(
        output["pose_rel"], batch["pose_rel"], valid
    )
    root = L.smooth_l1_per_sample(output["root"], batch["root"], valid)
    bone = L.BoneLoss().to(output["pose_rel"].device).per_sample(
        output["pose_rel"], batch["pose_rel"], valid
    )

    lag = 5
    interval = valid[:, lag:] & valid[:, :-lag]
    predicted_absolute = output["pose_rel"] + output["root"][:, :, None]
    target_absolute = batch["pose_rel"] + batch["root"][:, :, None]
    predicted_displacement = (
        predicted_absolute[:, lag:] - predicted_absolute[:, :-lag]
    ) * (C.TARGET_FPS / lag)
    target_displacement = (
        target_absolute[:, lag:] - target_absolute[:, :-lag]
    ) * (C.TARGET_FPS / lag)
    velocity = _masked_per_sample(
        F.smooth_l1_loss(
            predicted_displacement, target_displacement,
            beta=0.20, reduction="none",
        ), interval,
    )

    predicted_drop = output["root"][..., C.UP_AXIS] - output["root"][:, :1, C.UP_AXIS]
    target_drop = batch["root"][..., C.UP_AXIS] - batch["root"][:, :1, C.UP_AXIS]
    root_drop = _masked_per_sample(
        F.smooth_l1_loss(
            predicted_drop, target_drop, beta=0.10, reduction="none"
        ), valid,
    )

    head = C.JOINT_INDEX["head"]
    left_shoulder = C.JOINT_INDEX["left_shoulder"]
    right_shoulder = C.JOINT_INDEX["right_shoulder"]
    predicted_torso = F.normalize(
        output["pose_rel"][:, :, head], dim=-1
    )
    target_torso = F.normalize(batch["pose_rel"][:, :, head], dim=-1)
    torso = _masked_per_sample(
        1.0 - (predicted_torso * target_torso).sum(-1), valid
    )
    predicted_shoulder = F.normalize(
        output["pose_rel"][:, :, right_shoulder]
        - output["pose_rel"][:, :, left_shoulder], dim=-1,
    )
    target_shoulder = F.normalize(
        batch["pose_rel"][:, :, right_shoulder]
        - batch["pose_rel"][:, :, left_shoulder], dim=-1,
    )
    shoulder = _masked_per_sample(
        1.0 - (predicted_shoulder * target_shoulder).sum(-1), valid
    )
    endpoint = _endpoint_error(
        predicted_absolute, target_absolute, valid
    )

    alignment = pose.new_zeros(pose.shape)
    if alignment_weight > 0.0 and danger.any():
        predicted_descriptor = trajectory_descriptor(
            output["pose_rel"], output["root"], valid
        )
        target_descriptor = trajectory_descriptor(
            batch["pose_rel"], batch["root"], valid
        )
        alignment = bounded_piecewise_alignment_loss(
            predicted_descriptor, target_descriptor, valid,
            max_shift=max_shift,
        )
        alignment = alignment * danger.to(alignment.dtype)

    class_loss = F.cross_entropy(
        output["class_logits"], batch["class_id"],
        weight=class_weight, reduction="none",
    )
    risk_loss = F.cross_entropy(
        output["risk_logits"], batch["risk_id"],
        weight=risk_weight, reduction="none",
    )

    per_sample = (
        pose + root + 0.05 * bone + 0.15 * velocity
        + 0.25 * root_drop + 0.10 * torso + 0.05 * shoulder
        + 0.25 * endpoint + alignment_weight * alignment
        + lambda_class * class_loss + lambda_risk * risk_loss
    )
    total = _quality_mean(per_sample, sample_weight)
    parts = {
        "total": float(total.detach()),
        "pose": float(pose.mean().detach()),
        "root": float(root.mean().detach()),
        "bone": float(bone.mean().detach()),
        "velocity": float(velocity.mean().detach()),
        "root_drop": float(root_drop.mean().detach()),
        "torso": float(torso.mean().detach()),
        "shoulder": float(shoulder.mean().detach()),
        "endpoint": float(endpoint.mean().detach()),
        "alignment": float(
            alignment[danger].mean().detach() if danger.any()
            else alignment.new_zeros(())
        ),
        "danger_fraction": float(danger.float().mean().detach()),
        "class": float(class_loss.mean().detach()),
        "risk": float(risk_loss.mean().detach()),
    }
    return total, parts
