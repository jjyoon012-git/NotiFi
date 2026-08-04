"""Contact-guided local/global refinement for Seen Reconstruction V3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from . import losses as L
from .nets import LocalTemporalBlock


FOOT_NAMES = ("left_ankle", "right_ankle", "left_foot", "right_foot")
FOOT_JOINTS = tuple(C.JOINT_INDEX[name] for name in FOOT_NAMES)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.dim() < values.dim():
        weight = weight.unsqueeze(-1)
    total = (values * weight).sum(tuple(range(1, values.dim())))
    count = weight.expand_as(values).sum(tuple(range(1, values.dim())))
    return total / count.clamp_min(1.0)


def _masked_temporal_mean(values: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)[..., None]
    return (
        (values * weight).sum(1)
        / weight.sum(1).clamp_min(1.0)
    )


def _weighted_mean(values: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    weight = quality.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


class ContactGuidedRootNet(nn.Module):
    """Refine a frozen local pose with contact-aware global root dynamics.

    The wrapped V2 model supplies a calibrated root-relative pose and CSI motion
    features. This module predicts a fresh root anchor and body trajectory, then
    uses predicted foot support to reconcile root velocity with foot kinematics.
    """

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.hidden = hidden
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        input_size = hidden + 3 + 4 + 4 + 1 + 4
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.contact_residual = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.anchor_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.contact_gate = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.contact_residual[-1].weight)
        nn.init.zeros_(self.contact_residual[-1].bias)
        nn.init.zeros_(self.anchor_head[-1].weight)
        nn.init.zeros_(self.anchor_head[-1].bias)
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.zeros_(self.contact_gate[-1].weight)
        nn.init.constant_(self.contact_gate[-1].bias, -4.0)
        self.register_buffer(
            "root_strength", torch.tensor(1.0), persistent=False
        )

    def set_root_strength(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError("root strength must be between 0 and 1")
        self.root_strength.fill_(value)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            base = self.base(csi, link_mask)
        frame_mask = link_mask.any(-1)
        pose = base["pose_rel"]
        root = base["root"]

        base_velocity = torch.zeros_like(root)
        base_velocity[:, 1:] = (root[:, 1:] - root[:, :-1]) * C.TARGET_FPS
        feet = pose[:, :, FOOT_JOINTS]
        foot_velocity = torch.zeros_like(feet)
        foot_velocity[:, 1:] = (
            feet[:, 1:] - feet[:, :-1]
        ) * C.TARGET_FPS
        foot_speed = torch.linalg.vector_norm(foot_velocity, dim=-1)
        base_contact = torch.sigmoid(base["contact_logits"])
        phase = torch.softmax(base["phase_logits"], dim=-1)
        impact = torch.sigmoid(base["impact_logits"])[..., None]

        feature = self.input_projection(torch.cat((
            base["temporal_features"], base_velocity, base_contact,
            phase, impact, foot_speed,
        ), dim=-1))
        feature = feature * frame_mask[..., None]
        for block in self.temporal:
            feature = block(feature) * frame_mask[..., None]

        contact_logits = base["contact_logits"] + self.contact_residual(feature)
        contact = torch.sigmoid(contact_logits)
        support_weight = contact[..., None]
        support_velocity = -(
            support_weight * foot_velocity
        ).sum(2) / support_weight.sum(2).clamp_min(1e-4)
        support_confidence = 1.0 - torch.exp(-contact.sum(-1, keepdim=True))
        support_gate = (
            torch.sigmoid(self.contact_gate(feature))
            * support_confidence * (1.0 - impact)
        )

        learned_velocity = base_velocity + 0.75 * torch.tanh(
            self.velocity_head(feature)
        )
        candidate_velocity = learned_velocity + support_gate * (
            support_velocity - learned_velocity
        )
        candidate_velocity = candidate_velocity * frame_mask[..., None]
        candidate_velocity[:, 0] = 0.0

        pooled = _masked_temporal_mean(feature, frame_mask)
        anchor_delta = 0.75 * torch.tanh(self.anchor_head(pooled))
        anchor = root[:, 0] + anchor_delta
        step = candidate_velocity / C.TARGET_FPS
        step[:, 0] = 0.0
        candidate_root = anchor[:, None] + torch.cumsum(step, dim=1)
        refined_root = root + self.root_strength * (candidate_root - root)

        output = dict(base)
        output.update({
            "root": refined_root,
            "root_v2": root,
            "root_candidate": candidate_root,
            "root_anchor_delta_v3": anchor_delta,
            "root_velocity_v3": candidate_velocity,
            "contact_logits": contact_logits,
            "contact_support_gate": support_gate.squeeze(-1),
            "contact_support_velocity": support_velocity,
            "temporal_features_v3": feature,
        })
        return output


def contact_guided_root_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    """Quality-weighted root, contact, support, and floor objective."""
    valid = batch["valid"].bool()
    quality = batch.get(
        "quality_weight", torch.ones(len(valid), device=valid.device)
    ).to(output["root"].dtype)
    impact = L.impact_window(
        batch["pose_rel"], batch["root"], valid, batch["risk_id"]
    )
    frame_weight = valid.to(output["root"].dtype) * (
        1.0 + 2.0 * impact.to(output["root"].dtype)
    )

    root_distance = torch.linalg.vector_norm(
        output["root"] - batch["root"], dim=-1
    )
    position = _masked_mean(root_distance * frame_weight, valid)

    pair = valid[:, 1:] & valid[:, :-1]
    predicted_velocity = (
        output["root"][:, 1:] - output["root"][:, :-1]
    ) * C.TARGET_FPS
    target_velocity = (
        batch["root"][:, 1:] - batch["root"][:, :-1]
    ) * C.TARGET_FPS
    velocity = _masked_mean(
        torch.linalg.vector_norm(predicted_velocity - target_velocity, dim=-1),
        pair,
    )

    lag = 5
    interval = valid[:, lag:] & valid[:, :-lag]
    displacement = _masked_mean(torch.linalg.vector_norm(
        (output["root"][:, lag:] - output["root"][:, :-lag])
        - (batch["root"][:, lag:] - batch["root"][:, :-lag]), dim=-1,
    ), interval)
    anchor = torch.linalg.vector_norm(
        output["root"][:, 0] - batch["root"][:, 0], dim=-1
    )

    foot_target, floor = L.contact_targets(
        batch["pose_rel"], batch["root"], valid
    )
    contact_element = F.binary_cross_entropy_with_logits(
        output["contact_logits"], foot_target.to(output["root"].dtype),
        reduction="none",
    )
    contact = _masked_mean(contact_element, valid)

    absolute_feet = output["pose_rel"][:, :, FOOT_JOINTS] + output["root"][:, :, None]
    absolute_foot_velocity = torch.linalg.vector_norm(
        absolute_feet[:, 1:] - absolute_feet[:, :-1], dim=-1
    ) * C.TARGET_FPS
    support_mask = foot_target[:, 1:] & pair[..., None]
    foot_slip = _masked_mean(absolute_foot_velocity, support_mask)
    contact_height = (
        absolute_feet[..., 1] - floor[:, None, None]
    ).abs()
    floor_contact = _masked_mean(contact_height, foot_target & valid[..., None])
    penetration = _masked_mean(
        F.relu(floor[:, None, None] - absolute_feet[..., 1]), valid[..., None]
    )

    parts = {
        "root_position": _weighted_mean(position, quality),
        "root_velocity": _weighted_mean(velocity, quality),
        "root_displacement": _weighted_mean(displacement, quality),
        "root_anchor": _weighted_mean(anchor, quality),
        "foot_contact": _weighted_mean(contact, quality),
        "foot_slip": _weighted_mean(foot_slip, quality),
        "floor_contact": _weighted_mean(floor_contact, quality),
        "floor_penetration": _weighted_mean(penetration, quality),
    }
    total = (
        parts["root_position"]
        + 0.20 * parts["root_velocity"]
        + 0.20 * parts["root_displacement"]
        + 0.10 * parts["root_anchor"]
        + 0.05 * parts["foot_contact"]
        + 0.10 * parts["foot_slip"]
        + 0.05 * parts["floor_contact"]
        + 0.05 * parts["floor_penetration"]
    )
    return total, {key: float(value.detach()) for key, value in parts.items()}
