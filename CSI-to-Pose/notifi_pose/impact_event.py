"""Event-centric impact timing and body-part localization for CSI-only pose."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from . import losses as L
from .nets import LocalTemporalBlock
from .seen_v2 import INJURY_JOINTS, N_INJURY_JOINTS, injury_targets


IMPACT_REGION_NAMES = ("pelvis_hip", "knee", "head", "wrist")
JOINT_TO_REGION = (0, 0, 0, 1, 1, 2, 3, 3)
N_IMPACT_REGIONS = len(IMPACT_REGION_NAMES)


def _masked_temporal_mean(values: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)[..., None]
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def raw_csi_event_features(csi: torch.Tensor,
                           link_mask: torch.Tensor) -> torch.Tensor:
    """Return normalized multi-scale CSI change energy [B,T,4]."""
    real, imag = csi[..., 0], csi[..., 1]
    amplitude = torch.sqrt(real.square() + imag.square() + 1e-8)
    log_amplitude = torch.log(amplitude + 1e-4)
    amplitude_delta = log_amplitude[:, 1:] - log_amplitude[:, :-1]
    cross_real = real[:, 1:] * real[:, :-1] + imag[:, 1:] * imag[:, :-1]
    cross_imag = imag[:, 1:] * real[:, :-1] - real[:, 1:] * imag[:, :-1]
    phase_delta = torch.atan2(cross_imag, cross_real)
    subcarrier_energy = torch.sqrt(
        amplitude_delta.square() + phase_delta.square() + 1e-8
    ).median(-1).values
    pair_link = link_mask[:, 1:] & link_mask[:, :-1]
    link_weight = pair_link.to(subcarrier_energy.dtype)
    energy = torch.zeros(
        csi.shape[:2], dtype=csi.dtype, device=csi.device
    )
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
    energy = ((masked - mean) / variance.sqrt().clamp_min(1e-3)).clamp(-5, 5)
    channels = [energy]
    source = energy[:, None]
    for width in (3, 7, 15):
        channels.append(F.avg_pool1d(
            source, width, stride=1, padding=width // 2
        ).squeeze(1))
    return torch.stack(channels, dim=-1) * valid[..., None]


def _normalize_target(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, 0.0)
    # Normalize each anatomical joint over time so high-variance wrists do not
    # suppress lower-amplitude but plausible hip/pelvis impacts.
    scale = masked.amax(1, keepdim=True).clamp_min(1e-6)
    return masked / scale


def physical_impact_targets(pose: torch.Tensor, root: torch.Tensor,
                            valid: torch.Tensor,
                            risk: torch.Tensor,
                            sigma: float = 2.0) -> dict:
    """Derive a joint-time collision proxy from GT kinematics.

    Unlike the legacy height-only target, the score requires a combination of
    surface proximity, downward approach, acceleration, and post-impact slowing.
    It remains a biomechanical proxy, not a medical injury annotation.
    """
    valid = valid.bool()
    absolute = pose + root[:, :, None]
    joints = absolute[:, :, INJURY_JOINTS]
    _, floor = L.contact_targets(pose, root, valid)

    velocity = torch.zeros_like(joints)
    velocity[:, 1:] = (joints[:, 1:] - joints[:, :-1]) * C.TARGET_FPS
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.zeros_like(speed)
    acceleration[:, 2:] = torch.linalg.vector_norm(
        velocity[:, 2:] - velocity[:, 1:-1], dim=-1
    ) * C.TARGET_FPS
    post_speed = torch.cat((speed[:, 1:], speed[:, -1:]), dim=1)
    deceleration = F.relu(speed - post_speed)
    downward = F.relu(-velocity[..., 1])
    height = (joints[..., 1] - floor[:, None, None]).clamp_min(0.0)
    proximity = torch.exp(-torch.square(height / 0.18))

    joint_mask = valid[..., None].expand_as(speed)
    acceleration = _normalize_target(acceleration, joint_mask)
    deceleration = _normalize_target(deceleration, joint_mask)
    downward = _normalize_target(downward, joint_mask)

    root_height = root[..., 1]
    high = root_height.masked_fill(~valid, float("-inf")).amax(1, keepdim=True)
    low = root_height.masked_fill(~valid, float("inf")).amin(1, keepdim=True)
    high = torch.where(torch.isfinite(high), high, torch.zeros_like(high))
    low = torch.where(torch.isfinite(low), low, torch.zeros_like(low))
    fall_progress = ((high - root_height) / (high - low).clamp_min(0.10)).clamp(0, 1)

    score = (
        0.35 * deceleration
        + 0.25 * acceleration
        + 0.20 * downward
        + 0.20 * proximity
    ) * (0.35 + 0.65 * fall_progress[..., None])
    danger = risk.eq(2)
    score_mask = joint_mask & danger[:, None, None]
    score = score.masked_fill(~score_mask, -1.0)
    flat_index = score.flatten(1).argmax(1)
    frames = flat_index // N_INJURY_JOINTS
    joints_index = flat_index % N_INJURY_JOINTS
    has_event = danger & valid.any(1)
    frames = torch.where(has_event, frames, torch.full_like(frames, -1))
    joints_index = torch.where(
        has_event, joints_index, torch.full_like(joints_index, -1)
    )

    time = torch.arange(pose.shape[1], device=pose.device)[None]
    center = frames.clamp_min(0)[:, None]
    event_soft = torch.exp(-0.5 * torch.square((time - center) / sigma))
    event_soft = event_soft * valid.to(event_soft.dtype) * has_event[:, None]
    event_soft = event_soft / event_soft.sum(1, keepdim=True).clamp_min(1e-6)
    event_binary = event_soft >= torch.exp(
        torch.tensor(-0.5 * (3.0 / sigma) ** 2, device=pose.device)
    )

    joint_time_soft = event_soft[..., None] * F.one_hot(
        joints_index.clamp_min(0), N_INJURY_JOINTS
    ).to(event_soft.dtype)[:, None]
    region_map = torch.tensor(JOINT_TO_REGION, device=pose.device)
    regions = region_map[joints_index.clamp_min(0)]
    regions = torch.where(has_event, regions, torch.full_like(regions, -1))
    region_time_soft = event_soft[..., None] * F.one_hot(
        regions.clamp_min(0), N_IMPACT_REGIONS
    ).to(event_soft.dtype)[:, None]
    legacy = injury_targets(pose, root, valid, risk)
    return {
        "event_frame": frames,
        "event_joint": joints_index,
        "event_region": regions,
        "event_valid": has_event,
        "event_soft": event_soft,
        "event_binary": event_binary,
        "joint_time_soft": joint_time_soft,
        "region_time_soft": region_time_soft,
        "joint_speed": speed,
        "legacy_contact": legacy["injury_contact"],
        "floor_height": floor,
        "physical_score": score.clamp_min(0.0),
    }


class ImpactEventLocalizer(nn.Module):
    """Localize an impact jointly in time and anatomical joint space."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.hidden = hidden
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        input_size = hidden * 3 + 4 + 4 + 1 + 1 + N_INJURY_JOINTS * 2
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
        self.event_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        )
        self.joint_time_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_INJURY_JOINTS)
        )
        self.region_time_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_IMPACT_REGIONS)
        )
        self.contact_residual = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_INJURY_JOINTS)
        )
        self.speed_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, N_INJURY_JOINTS)
        )
        for head in (
            self.event_head, self.joint_time_head,
            self.region_time_head, self.contact_residual, self.speed_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer(
            "event_strength", torch.tensor(1.0), persistent=False
        )
        self.register_buffer(
            "joint_strength", torch.tensor(1.0), persistent=False
        )
        self.register_buffer(
            "contact_strength", torch.tensor(1.0), persistent=False
        )
        self.register_buffer(
            "speed_strength", torch.tensor(1.0), persistent=False
        )

    def set_calibration(self, event: float | None = None,
                        joint: float | None = None,
                        contact: float | None = None,
                        speed: float | None = None) -> None:
        for name, value in (
            ("event_strength", event), ("joint_strength", joint),
            ("contact_strength", contact), ("speed_strength", speed),
        ):
            if value is None:
                continue
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            getattr(self, name).fill_(value)

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
        absolute = pose[:, :, INJURY_JOINTS] + root[:, :, None]
        velocity = torch.zeros_like(absolute)
        velocity[:, 1:] = (
            absolute[:, 1:] - absolute[:, :-1]
        ) * C.TARGET_FPS
        joint_speed = torch.linalg.vector_norm(velocity, dim=-1)
        root_velocity = torch.zeros_like(root)
        root_velocity[:, 1:] = (
            root[:, 1:] - root[:, :-1]
        ) * C.TARGET_FPS
        root_speed = torch.linalg.vector_norm(root_velocity, dim=-1, keepdim=True)
        phase = torch.softmax(base["phase_logits"], dim=-1)
        impact = torch.sigmoid(base["impact_logits"])[..., None]
        legacy_contact = torch.sigmoid(base["injury_contact_logits"])
        raw_event = raw_csi_event_features(csi, link_mask)
        temporal = base.get("temporal_features_v3", base["temporal_features"])
        motion_feature = base.get("motion_features", base["temporal_features"])
        baseline_feature = base.get(
            "baseline_features", base["temporal_features"]
        )

        feature = self.input_projection(torch.cat((
            temporal, motion_feature, baseline_feature,
            raw_event, phase, impact, root_speed, joint_speed, legacy_contact,
        ), dim=-1))
        feature = feature * frame_mask[..., None]
        for block in self.temporal:
            feature = block(feature) * frame_mask[..., None]
        feature = self.context(feature, src_key_padding_mask=~frame_mask)
        feature = feature * frame_mask[..., None]

        event_logits = (
            base["impact_logits"]
            + self.event_strength * self.event_head(feature).squeeze(-1)
        )
        event_logits = event_logits.masked_fill(~frame_mask, -1e4)
        joint_time_logits = (
            base["first_contact_logits"][:, None]
            + self.joint_strength * self.joint_time_head(feature)
        )
        joint_time_logits = joint_time_logits.masked_fill(
            ~frame_mask[..., None], -1e4
        )
        combined = joint_time_logits + event_logits[..., None]
        first_contact_logits = torch.logsumexp(combined, dim=1)
        base_joint_logits = base["first_contact_logits"]
        base_region_logits = torch.stack((
            torch.logsumexp(base_joint_logits[:, 0:3], dim=-1),
            torch.logsumexp(base_joint_logits[:, 3:5], dim=-1),
            base_joint_logits[:, 5],
            torch.logsumexp(base_joint_logits[:, 6:8], dim=-1),
        ), dim=-1)
        region_time_logits = (
            base_region_logits[:, None]
            + self.joint_strength * self.region_time_head(feature)
        )
        region_combined = region_time_logits + event_logits[..., None]
        first_region_logits = torch.logsumexp(region_combined, dim=1)
        contact_logits = (
            base["injury_contact_logits"]
            + self.contact_strength * self.contact_residual(feature)
        )
        impact_speed = F.relu(
            base["joint_impact_speed"]
            + self.speed_strength * 0.50 * torch.tanh(self.speed_head(feature))
        )

        output = dict(base)
        output.update({
            "event_logits_v8": event_logits,
            "joint_time_logits_v8": joint_time_logits,
            "joint_time_combined_v8": combined,
            "region_time_logits_v8": region_time_logits,
            "region_time_combined_v8": region_combined,
            "first_contact_logits": first_contact_logits,
            "first_region_logits_v8": first_region_logits,
            "injury_contact_logits": contact_logits,
            "joint_impact_speed": impact_speed,
            "event_features_v8": feature,
            "raw_csi_event_features_v8": raw_event,
        })
        return output


def _focal_bce(logits: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha = 0.75 * target + 0.25 * (1.0 - target)
    loss = alpha * torch.square(1.0 - pt) * bce
    weight = mask.to(loss.dtype)
    return (loss * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def impact_event_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    valid = batch["valid"].bool()
    quality = batch.get(
        "quality_weight", torch.ones(len(valid), device=valid.device)
    ).to(output["event_logits_v8"].dtype)
    target = physical_impact_targets(
        batch["pose_rel"], batch["root"], valid, batch["risk_id"]
    )
    event_valid = target["event_valid"]

    event_binary = target["event_binary"].to(output["event_logits_v8"].dtype)
    event_focal = _focal_bce(output["event_logits_v8"], event_binary, valid)
    event_log_probability = torch.log_softmax(
        output["event_logits_v8"], dim=1
    )
    event_time = -(target["event_soft"] * event_log_probability).sum(1)

    flat_logits = output["joint_time_combined_v8"].flatten(1)
    flat_target = target["joint_time_soft"].flatten(1)
    joint_time = -(flat_target * torch.log_softmax(flat_logits, dim=1)).sum(1)
    flat_region_logits = output["region_time_combined_v8"].flatten(1)
    flat_region_target = target["region_time_soft"].flatten(1)
    region_time = -(
        flat_region_target * torch.log_softmax(flat_region_logits, dim=1)
    ).sum(1)
    first_contact = torch.zeros_like(event_time)
    if event_valid.any():
        first_contact[event_valid] = F.cross_entropy(
            output["first_contact_logits"][event_valid],
            target["event_joint"][event_valid], reduction="none",
        )
    first_region = torch.zeros_like(event_time)
    if event_valid.any():
        first_region[event_valid] = F.cross_entropy(
            output["first_region_logits_v8"][event_valid],
            target["event_region"][event_valid], reduction="none",
        )

    legacy_target = target["legacy_contact"].to(
        output["injury_contact_logits"].dtype
    )
    contact_element = F.binary_cross_entropy_with_logits(
        output["injury_contact_logits"], legacy_target, reduction="none"
    ).mean(-1)
    contact = (
        (contact_element * valid).sum(1)
        / valid.sum(1).clamp_min(1.0)
    )
    event_mask = target["event_soft"] > 0.01
    speed_element = F.smooth_l1_loss(
        output["joint_impact_speed"], target["joint_speed"],
        beta=0.20, reduction="none",
    ).mean(-1)
    speed = (
        (speed_element * event_mask).sum(1)
        / event_mask.sum(1).clamp_min(1.0)
    )

    per_sample = (
        0.50 * event_focal
        + event_time
        + 0.50 * joint_time
        + region_time
        + 0.25 * first_contact
        + 0.75 * first_region
        + 0.10 * contact
        + 0.05 * speed
    )
    total = (per_sample * quality).sum() / quality.sum().clamp_min(1e-6)
    parts = {
        "event_focal": event_focal.mean(),
        "event_time": event_time[event_valid].mean() if event_valid.any() else total * 0,
        "joint_time": joint_time[event_valid].mean() if event_valid.any() else total * 0,
        "region_time": region_time[event_valid].mean() if event_valid.any() else total * 0,
        "first_contact": first_contact[event_valid].mean() if event_valid.any() else total * 0,
        "first_region": first_region[event_valid].mean() if event_valid.any() else total * 0,
        "contact": contact.mean(),
        "impact_speed": speed[event_valid].mean() if event_valid.any() else total * 0,
    }
    return total, {key: float(value.detach()) for key, value in parts.items()}
