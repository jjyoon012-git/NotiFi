"""Motion-first CSI encoder used before any pose decoder is trained."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import (
    HybridGraphDecoder,
    LinkAttentionFusion,
    LocalTemporalBlock,
    PerLinkNorm,
    PoseTemporalRefiner,
    SubcarrierConvEncoder,
    TemporalTransformer,
)


def temporal_difference(values: torch.Tensor,
                        link_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    difference = torch.zeros_like(values)
    pair_mask = torch.zeros_like(link_mask)
    pair_mask[:, 1:] = link_mask[:, 1:] & link_mask[:, :-1]
    difference[:, 1:] = values[:, 1:] - values[:, :-1]
    difference = difference * pair_mask[..., None, None].to(difference.dtype)
    return difference, pair_mask


def masked_temporal_average(values: torch.Tensor, frame_mask: torch.Tensor,
                            width: int = 5) -> torch.Tensor:
    """Average a [B,T,...] trajectory without leaking padded frames."""
    batch, frames = values.shape[:2]
    channels = int(values[0, 0].numel())
    flat = values.reshape(batch, frames, channels).transpose(1, 2)
    mask = frame_mask[:, None].to(values.dtype)
    numerator = F.avg_pool1d(
        flat * mask, kernel_size=width, stride=1, padding=width // 2,
        count_include_pad=False,
    )
    denominator = F.avg_pool1d(
        mask, kernel_size=width, stride=1, padding=width // 2,
        count_include_pad=False,
    )
    averaged = numerator / denominator.clamp_min(1e-6)
    averaged = averaged.transpose(1, 2).reshape_as(values)
    return averaged * frame_mask.reshape(batch, frames, *([1] * (values.ndim - 2)))


def temporal_keyframes(values: torch.Tensor, frame_mask: torch.Tensor,
                       stride: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool temporal features into valid keyframes for coherent decoding."""
    transposed = values.transpose(1, 2)
    mask = frame_mask[:, None].to(values.dtype)
    numerator = F.avg_pool1d(
        transposed * mask, kernel_size=stride, stride=stride, ceil_mode=True,
        count_include_pad=False,
    )
    denominator = F.avg_pool1d(
        mask, kernel_size=stride, stride=stride, ceil_mode=True,
        count_include_pad=False,
    )
    keyframes = (numerator / denominator.clamp_min(1e-6)).transpose(1, 2)
    return keyframes, denominator.squeeze(1) > 0


def interpolate_keyframes(values: torch.Tensor, frames: int) -> torch.Tensor:
    batch, keyframes = values.shape[:2]
    channels = int(values[0, 0].numel())
    flat = values.reshape(batch, keyframes, channels).transpose(1, 2)
    interpolated = F.interpolate(
        flat, size=frames, mode="linear", align_corners=False
    )
    return interpolated.transpose(1, 2).reshape(batch, frames, *values.shape[2:])


class MotionFirstEncoder(nn.Module):
    """Encode static CSI and temporal CSI differences with separate branches."""

    def __init__(self, hidden: int = 128, temporal_layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.norm = PerLinkNorm()
        self.raw_encoder = SubcarrierConvEncoder(hidden, dropout=dropout)
        self.delta_encoder = SubcarrierConvEncoder(hidden, dropout=dropout)
        self.raw_fusion = LinkAttentionFusion(hidden, heads=heads, dropout=dropout)
        self.delta_fusion = LinkAttentionFusion(hidden, heads=heads, dropout=dropout)
        self.branch_mixer = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = TemporalTransformer(
            hidden, temporal_layers, heads, dropout
        )
        self.motion_attention = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        )
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.moving_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.phase_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
        self.impact_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        self.class_head = nn.Linear(hidden, C.N_CLASSES)
        self.risk_head = nn.Linear(hidden, C.N_RISK)

    def encode(self, csi: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(csi, link_mask)
        difference, pair_mask = temporal_difference(normalized, link_mask)
        raw = self.raw_fusion(self.raw_encoder(normalized), link_mask)
        delta = self.delta_fusion(self.delta_encoder(difference), pair_mask)
        fused = self.branch_mixer(torch.cat((raw, delta), dim=-1))
        frame_mask = link_mask.any(-1)
        return self.temporal(fused, frame_mask)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        temporal = self.encode(csi, link_mask)
        frame_mask = link_mask.any(-1)
        attention_score = self.motion_attention(temporal).squeeze(-1)
        attention_score = attention_score.masked_fill(~frame_mask, -1e4)
        attention = torch.softmax(attention_score, dim=1)
        attended = (temporal * attention[..., None]).sum(1)
        mean = (temporal * frame_mask[..., None]).sum(1)
        mean = mean / frame_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = 0.5 * (attended + mean)
        return {
            "speed_log": self.speed_head(temporal).squeeze(-1),
            "moving_logits": self.moving_head(temporal).squeeze(-1),
            "phase_logits": self.phase_head(temporal),
            "impact_logits": self.impact_head(temporal).squeeze(-1),
            "class_logits": self.class_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "embedding": F.normalize(self.embedding_head(pooled), dim=-1),
            "temporal_features": temporal,
        }

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MotionFirstPoseNet(nn.Module):
    """Attach a pose decoder while retaining the pretrained motion objectives."""

    def __init__(self, hidden: int = 128, temporal_layers: int = 3,
                 heads: int = 4, graph_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = MotionFirstEncoder(
            hidden=hidden, temporal_layers=temporal_layers,
            heads=heads, dropout=dropout,
        )
        self.pose_decoder = HybridGraphDecoder(hidden, graph_blocks, dropout)
        self.pose_velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        self.velocity_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.root_decoder = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    @property
    def norm(self) -> PerLinkNorm:
        return self.backbone.norm

    def load_motion_backbone(self, state: dict) -> None:
        self.backbone.load_state_dict(state, strict=True)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = self.backbone(csi, link_mask)
        temporal = output["temporal_features"]
        frame_mask = link_mask.any(-1)
        key_features, _ = temporal_keyframes(temporal, frame_mask, stride=4)
        coarse = interpolate_keyframes(
            self.pose_decoder(key_features), temporal.shape[1]
        )
        velocity = 0.5 * torch.tanh(self.pose_velocity_head(temporal))
        velocity = velocity.reshape(*temporal.shape[:2], C.N_JOINTS, 3)
        velocity = velocity * frame_mask[:, :, None, None].to(velocity.dtype)
        velocity = masked_temporal_average(velocity, frame_mask, width=5)
        pose = coarse
        root_keyframes = self.root_decoder(key_features)
        root = interpolate_keyframes(root_keyframes, temporal.shape[1])
        output.update({
            "pose_rel": pose,
            "pose_coarse": coarse,
            "pose_velocity": velocity,
            "root": root,
            "motion": output["speed_log"],
        })
        return output

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MotionResidualPoseNet(nn.Module):
    """Refine a frozen seen-pose baseline with frozen motion-first features."""

    def __init__(self, baseline: nn.Module, motion: MotionFirstEncoder,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.baseline = baseline
        self.motion = motion
        baseline_hidden = int(getattr(baseline, "hidden", hidden))
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(baseline_hidden + motion.hidden),
            nn.Linear(baseline_hidden + motion.hidden, hidden),
            nn.GELU(),
        )
        self.pose_refiner = PoseTemporalRefiner(
            hidden, dropout=dropout, max_delta=0.15
        )
        self.root_blocks = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.root_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        nn.init.zeros_(self.root_head[-1].weight)
        nn.init.zeros_(self.root_head[-1].bias)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        self.motion.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            baseline = self.baseline(csi, link_mask)
            motion = self.motion(csi, link_mask)
        frame_mask = link_mask.any(-1)
        fused = self.feature_fusion(torch.cat((
            baseline["temporal_features"], motion["temporal_features"]
        ), dim=-1))
        pose = self.pose_refiner(baseline["pose_rel"], fused, frame_mask)
        root_features = fused
        for block in self.root_blocks:
            root_features = block(root_features)
        root_delta = 0.50 * torch.tanh(self.root_head(root_features))
        root_delta = root_delta * frame_mask[..., None].to(root_delta.dtype)
        output = dict(baseline)
        output.update({
            "pose_coarse": baseline["pose_rel"],
            "pose_rel": pose,
            "root": baseline["root"] + root_delta,
            "motion": motion["speed_log"],
            "phase_logits": motion["phase_logits"],
            "embedding": motion["embedding"],
            "temporal_features": fused,
        })
        return output


class ActionMotionResidualPoseNet(nn.Module):
    """Use predicted CSI action probabilities to refine relative pose only."""

    def __init__(self, baseline: nn.Module, motion: MotionFirstEncoder,
                 hidden: int = 128, dropout: float = 0.05):
        super().__init__()
        self.baseline = baseline
        self.motion = motion
        baseline_hidden = int(getattr(baseline, "hidden", hidden))
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(baseline_hidden + motion.hidden),
            nn.Linear(baseline_hidden + motion.hidden, hidden),
            nn.GELU(),
        )
        self.action_embedding = nn.Parameter(torch.zeros(C.N_CLASSES, hidden))
        nn.init.normal_(self.action_embedding, std=0.02)
        self.pose_refiner = PoseTemporalRefiner(
            hidden, dropout=dropout, max_delta=0.08
        )
        self.register_buffer(
            "residual_scale", torch.tensor(1.0), persistent=False
        )
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        self.motion.eval()
        return self

    def set_residual_scale(self, scale: float) -> None:
        if scale < 0.0 or scale > 1.0:
            raise ValueError("residual scale must be between 0 and 1")
        self.residual_scale.fill_(scale)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            baseline = self.baseline(csi, link_mask)
            motion = self.motion(csi, link_mask)
        action_probability = torch.softmax(baseline["class_logits"], dim=-1)
        action_context = action_probability @ self.action_embedding
        fused = self.feature_fusion(torch.cat((
            baseline["temporal_features"], motion["temporal_features"]
        ), dim=-1))
        fused = fused + action_context[:, None]
        frame_mask = link_mask.any(-1)
        refined_pose = self.pose_refiner(
            baseline["pose_rel"], fused, frame_mask
        )
        pose = baseline["pose_rel"] + self.residual_scale * (
            refined_pose - baseline["pose_rel"]
        )
        output = dict(baseline)
        output.update({
            "pose_coarse": baseline["pose_rel"],
            "pose_rel": pose,
            "motion": motion["speed_log"],
            "phase_logits": motion["phase_logits"],
            "embedding": motion["embedding"],
            "temporal_features": fused,
        })
        return output


class KeyframeRootResidualNet(nn.Module):
    """Refine only the root path of a frozen seen pose model."""

    def __init__(self, pose_model: ActionMotionResidualPoseNet,
                 hidden: int = 128, dropout: float = 0.05,
                 max_delta: float = 0.60):
        super().__init__()
        self.pose_model = pose_model
        self.root_decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        self.max_delta = max_delta
        nn.init.zeros_(self.root_decoder[-1].weight)
        nn.init.zeros_(self.root_decoder[-1].bias)
        for parameter in self.pose_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_model.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            output = self.pose_model(csi, link_mask)
        frame_mask = link_mask.any(-1)
        key_features, _ = temporal_keyframes(
            output["temporal_features"], frame_mask, stride=4
        )
        root_delta = self.max_delta * torch.tanh(
            self.root_decoder(key_features)
        )
        root_delta = interpolate_keyframes(root_delta, csi.shape[1])
        result = dict(output)
        result.update({
            "root_coarse": output["root"],
            "root": output["root"] + root_delta,
        })
        return result
