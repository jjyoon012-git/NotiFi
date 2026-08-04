"""P2 coarse reconstruction with a validation-gated V9 trajectory refiner."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import LocalTemporalBlock
from .seen_v2 import _forward_kinematics, _local_bones
from .seen_v4 import _body_group_speed, _masked_temporal_mean
from .v3 import rotation_6d_to_matrix


def p2_motion_features(csi: torch.Tensor,
                       link_mask: torch.Tensor) -> torch.Tensor:
    """Multi-scale change energy for amplitude + sanitized-phase input."""
    amplitude_delta = csi[:, 1:, ..., 0] - csi[:, :-1, ..., 0]
    phase_delta_raw = csi[:, 1:, ..., 1] - csi[:, :-1, ..., 1]
    phase_delta = torch.atan2(torch.sin(phase_delta_raw), torch.cos(phase_delta_raw))
    subcarrier_energy = torch.sqrt(
        amplitude_delta.square() + phase_delta.square() + 1e-8
    ).median(-1).values
    pair_link = link_mask[:, 1:] & link_mask[:, :-1]
    link_weight = pair_link.to(subcarrier_energy.dtype)
    energy = torch.zeros(csi.shape[:2], dtype=csi.dtype, device=csi.device)
    energy[:, 1:] = (
        (subcarrier_energy * link_weight).sum(-1)
        / link_weight.sum(-1).clamp_min(1.0)
    )
    valid = link_mask.any(-1)
    masked = energy.masked_fill(~valid, 0.0)
    mean = masked.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
    variance = (
        ((masked - mean).square() * valid).sum(1, keepdim=True)
        / valid.sum(1, keepdim=True).clamp_min(1)
    )
    normalized = ((masked - mean) / variance.sqrt().clamp_min(1e-3)).clamp(-5, 5)
    channels = [normalized]
    source = normalized[:, None]
    for width in (3, 7, 15):
        channels.append(F.avg_pool1d(
            source, width, stride=1, padding=width // 2
        ).squeeze(1))
    return torch.stack(channels, dim=-1) * valid[..., None]


class P2V9HybridNet(nn.Module):
    """Freeze a strong P2 model and learn bounded V9 residuals around it.

    Calibration strengths are selected on validation data. Setting every
    strength to zero exactly recovers the frozen P2 output.
    """

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        base_hidden = int(getattr(base, "hidden", 96))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        group_count = len(C.JOINT_GROUPS)
        input_size = base_hidden + 4 + 3 + group_count + C.N_CLASSES + C.N_RISK
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, 1)

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
        self.class_delta_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_CLASSES)
        )
        self.risk_delta_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_RISK)
        )
        for head in (
            self.rotation_head, self.root_anchor_head, self.root_step_head,
            self.class_delta_head, self.risk_delta_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.register_buffer("pose_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("root_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("class_strength", torch.tensor(1.0), persistent=False)
        self.register_buffer("risk_strength", torch.tensor(1.0), persistent=False)

    def set_calibration(self, pose: float = 1.0, root: float = 1.0,
                        classification: float = 1.0,
                        risk: float = 1.0) -> None:
        values = (pose, root, classification, risk)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("calibration strengths must be between 0 and 1")
        self.pose_strength.fill_(pose)
        self.root_strength.fill_(root)
        self.class_strength.fill_(classification)
        self.risk_strength.fill_(risk)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            base = self.base(csi, link_mask)
        if "temporal_features" not in base:
            raise KeyError("P2 base must expose temporal_features")

        valid = link_mask.any(-1)
        pose = base["pose_rel"]
        root = base["root"]
        root_velocity = torch.zeros_like(root)
        root_velocity[:, 1:] = (root[:, 1:] - root[:, :-1]) * C.TARGET_FPS
        group_speed = _body_group_speed(pose, root)
        class_probability = torch.softmax(base["class_logits"], dim=-1)
        risk_probability = torch.softmax(base["risk_logits"], dim=-1)
        class_probability = class_probability[:, None].expand(-1, pose.shape[1], -1)
        risk_probability = risk_probability[:, None].expand(-1, pose.shape[1], -1)

        feature = self.input_projection(torch.cat((
            base["temporal_features"],
            p2_motion_features(csi, link_mask),
            root_velocity,
            group_speed,
            class_probability,
            risk_probability,
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
        length = torch.linalg.vector_norm(bones, dim=-1, keepdim=True)
        rotated = F.normalize(rotated, dim=-1) * length
        rotated[:, :, C.ROOT_JOINT] = 0.0
        kinematic_pose = _forward_kinematics(rotated)
        refined_pose = pose + self.pose_strength * (kinematic_pose - pose)

        pooled = _masked_temporal_mean(feature, valid)
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        step_delta = 0.006 * torch.tanh(self.root_step_head(feature))
        step_delta[:, 0] = 0.0
        step_delta = step_delta * valid[..., None]
        base_step = torch.zeros_like(root)
        base_step[:, 1:] = root[:, 1:] - root[:, :-1]
        adjusted_root = (
            root[:, :1] + anchor_delta[:, None]
            + torch.cumsum(base_step + step_delta, dim=1)
        )
        refined_root = root + self.root_strength * (adjusted_root - root)

        class_delta = self.class_delta_head(pooled)
        risk_delta = self.risk_delta_head(pooled)
        output = dict(base)
        output.update({
            "pose_rel": refined_pose,
            "root": refined_root,
            "pose_p2": pose,
            "root_p2": root,
            "class_logits_p2": base["class_logits"],
            "risk_logits_p2": base["risk_logits"],
            "class_logits": base["class_logits"] + self.class_strength * class_delta,
            "risk_logits": base["risk_logits"] + self.risk_strength * risk_delta,
            "rotation_6d_delta_v10": rotation_delta,
            "root_anchor_delta_v10": anchor_delta,
            "root_step_delta_v10": step_delta,
            "class_delta_v10": class_delta,
            "risk_delta_v10": risk_delta,
            "temporal_features_v10": feature,
        })
        return output


class RootExpertBlend(nn.Module):
    """Use one CSI model for pose/logits and another only for root trajectory."""

    def __init__(self, primary: nn.Module, root_expert: nn.Module):
        super().__init__()
        self.primary = primary
        self.root_expert = root_expert
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("root_strength", torch.tensor(0.0), persistent=False)

    def set_root_strength(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("root strength must be between 0 and 1")
        self.root_strength.fill_(value)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
            expert = self.root_expert(csi, link_mask)
        output = dict(primary)
        output["root_primary"] = primary["root"]
        output["root_expert"] = expert["root"]
        output["root"] = primary["root"] + self.root_strength * (
            expert["root"] - primary["root"]
        )
        return output
