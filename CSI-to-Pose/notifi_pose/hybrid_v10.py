"""P2 coarse reconstruction with a validation-gated V9 trajectory refiner."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .nets import JointGraphBlock, LocalTemporalBlock
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
        self.residual_input_size = input_size
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

    def predict_rotation_delta(self, feature: torch.Tensor,
                               pose: torch.Tensor) -> torch.Tensor:
        return self.rotation_head(feature).reshape(
            *feature.shape[:2], C.N_JOINTS, 6
        )

    def pose_candidate(self, feature: torch.Tensor,
                       pose: torch.Tensor) -> tuple[torch.Tensor, dict]:
        rotation_delta = self.predict_rotation_delta(feature, pose)
        identity = rotation_delta.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        rotation = rotation_6d_to_matrix(rotation_delta + identity)
        bones = _local_bones(pose)
        rotated = torch.matmul(rotation, bones.unsqueeze(-1)).squeeze(-1)
        length = torch.linalg.vector_norm(bones, dim=-1, keepdim=True)
        rotated = F.normalize(rotated, dim=-1) * length
        rotated[:, :, C.ROOT_JOINT] = 0.0
        return _forward_kinematics(rotated), {
            "rotation_6d_delta_v10": rotation_delta,
        }

    def root_candidate(
        self, feature: torch.Tensor, root: torch.Tensor,
        valid: torch.Tensor, pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        step_delta = 0.006 * torch.tanh(self.root_step_head(feature))
        step_delta[:, 0] = 0.0
        step_delta = step_delta * valid[..., None]
        base_step = torch.zeros_like(root)
        base_step[:, 1:] = root[:, 1:] - root[:, :-1]
        adjusted = (
            root[:, :1] + anchor_delta[:, None]
            + torch.cumsum(base_step + step_delta, dim=1)
        )
        return adjusted, {
            "root_anchor_delta_v10": anchor_delta,
            "root_step_delta_v10": step_delta,
        }

    def encode_residual_context(self, feature: torch.Tensor,
                                valid: torch.Tensor) -> torch.Tensor:
        for block in self.temporal:
            feature = block(feature) * valid[..., None]
        return self.context(
            feature, src_key_padding_mask=~valid
        ) * valid[..., None]

    def residual_features(
        self, csi: torch.Tensor, link_mask: torch.Tensor, base: dict,
        pose: torch.Tensor, root: torch.Tensor,
    ) -> torch.Tensor:
        root_velocity = torch.zeros_like(root)
        root_velocity[:, 1:] = (root[:, 1:] - root[:, :-1]) * C.TARGET_FPS
        group_speed = _body_group_speed(pose, root)
        class_probability = torch.softmax(base["class_logits"], dim=-1)
        risk_probability = torch.softmax(base["risk_logits"], dim=-1)
        class_probability = class_probability[:, None].expand(-1, pose.shape[1], -1)
        risk_probability = risk_probability[:, None].expand(-1, pose.shape[1], -1)
        motion_feature = base.get("_v12_motion_features")
        if motion_feature is None:
            motion_feature = p2_motion_features(csi, link_mask)
        return torch.cat((
            base["temporal_features"],
            motion_feature,
            root_velocity,
            group_speed,
            class_probability,
            risk_probability,
        ), dim=-1)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            base = self.base(csi, link_mask)
        if "temporal_features" not in base:
            raise KeyError("P2 base must expose temporal_features")

        valid = link_mask.any(-1)
        pose = base["pose_rel"]
        root = base["root"]
        feature = self.input_projection(self.residual_features(
            csi, link_mask, base, pose, root
        ))
        feature = feature * valid[..., None]
        feature = self.encode_residual_context(feature, valid)

        pose_candidate, pose_auxiliary = self.pose_candidate(feature, pose)
        refined_pose = pose + self.pose_strength * (pose_candidate - pose)

        pooled = _masked_temporal_mean(feature, valid)
        adjusted_root, root_auxiliary = self.root_candidate(
            feature, root, valid, pooled,
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
            "class_delta_v10": class_delta,
            "risk_delta_v10": risk_delta,
            "temporal_features_v10": feature,
            **pose_auxiliary,
            **root_auxiliary,
        })
        return output

    def forward_logits(self, csi: torch.Tensor,
                       link_mask: torch.Tensor) -> dict:
        """Compute classification outputs without unused pose/root heads."""
        with torch.no_grad():
            base = self.base(csi, link_mask)
        valid = link_mask.any(-1)
        pose = base["pose_rel"]
        root = base["root"]
        feature = self.input_projection(self.residual_features(
            csi, link_mask, base, pose, root
        ))
        feature = feature * valid[..., None]
        feature = self.encode_residual_context(feature, valid)
        pooled = _masked_temporal_mean(feature, valid)
        return {
            "class_logits": base["class_logits"] + self.class_strength * (
                self.class_delta_head(pooled)
            ),
            "risk_logits": base["risk_logits"] + self.risk_strength * (
                self.risk_delta_head(pooled)
            ),
        }


class GraphRotationResidualHead(nn.Module):
    """Predict joint rotations from global motion and local skeleton tokens."""

    def __init__(self, temporal_hidden: int, joint_hidden: int = 64,
                 blocks: int = 2, dropout: float = 0.05):
        super().__init__()
        self.global_projection = nn.Linear(temporal_hidden, joint_hidden)
        self.bone_projection = nn.Sequential(
            nn.Linear(4, joint_hidden), nn.GELU(), nn.LayerNorm(joint_hidden)
        )
        self.joint_queries = nn.Parameter(
            torch.zeros(C.N_JOINTS, joint_hidden)
        )
        self.blocks = nn.ModuleList(
            JointGraphBlock(joint_hidden, dropout) for _ in range(blocks)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(joint_hidden), nn.Linear(joint_hidden, 6)
        )
        nn.init.normal_(self.joint_queries, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feature: torch.Tensor,
                pose: torch.Tensor) -> torch.Tensor:
        bones = _local_bones(pose)
        length = torch.linalg.vector_norm(bones, dim=-1, keepdim=True)
        descriptor = torch.cat((F.normalize(bones, dim=-1), length), dim=-1)
        joints = (
            self.global_projection(feature)[:, :, None]
            + self.bone_projection(descriptor)
            + self.joint_queries[None, None]
        )
        for block in self.blocks:
            joints = block(joints)
        return self.head(joints)


class P2V11GraphHybridNet(P2V9HybridNet):
    """V10 temporal adapter with a skeleton-aware graph rotation decoder."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05):
        super().__init__(base, hidden=hidden, dropout=dropout)
        self.rotation_head = GraphRotationResidualHead(
            hidden, joint_hidden=64, blocks=2, dropout=dropout
        )

    def predict_rotation_delta(self, feature: torch.Tensor,
                               pose: torch.Tensor) -> torch.Tensor:
        return self.rotation_head(feature, pose)


class SpectralContextBlock(nn.Module):
    """Fuse long-term trend and fast temporal residual without discarding either."""

    def __init__(self, hidden: int, modes: int = 24):
        super().__init__()
        self.modes = modes
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, feature: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(feature) * valid[..., None]
        frequency = torch.fft.rfft(
            normalized.float(), dim=1, norm="ortho"
        )
        kept = min(self.modes, frequency.shape[1])
        low_frequency = torch.zeros_like(frequency)
        low_frequency[:, :kept] = frequency[:, :kept]
        low = torch.fft.irfft(
            low_frequency, n=feature.shape[1], dim=1, norm="ortho"
        ).to(feature.dtype)
        high = normalized - low
        residual = self.projection(torch.cat((normalized, low, high), dim=-1))
        return feature + residual * valid[..., None]


class P2V11SpectralHybridNet(P2V9HybridNet):
    """V10 adapter augmented with a PoseFormerV2-style frequency view."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05):
        super().__init__(base, hidden=hidden, dropout=dropout)
        self.spectral_context = SpectralContextBlock(hidden)

    def encode_residual_context(self, feature: torch.Tensor,
                                valid: torch.Tensor) -> torch.Tensor:
        feature = super().encode_residual_context(feature, valid)
        return self.spectral_context(feature, valid)


class P2V11CartesianHybridNet(P2V9HybridNet):
    """Bounded Cartesian correction when rotation-only residual is too rigid."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, max_delta: float = 0.10):
        super().__init__(base, hidden=hidden, dropout=dropout)
        self.max_delta = float(max_delta)
        self.cartesian_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, C.N_JOINTS * 3),
        )
        nn.init.zeros_(self.cartesian_head[-1].weight)
        nn.init.zeros_(self.cartesian_head[-1].bias)

    def pose_candidate(self, feature: torch.Tensor,
                       pose: torch.Tensor) -> tuple[torch.Tensor, dict]:
        delta = self.max_delta * torch.tanh(
            self.cartesian_head(feature).reshape_as(pose)
        )
        delta = delta - delta[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        candidate = pose + delta
        candidate = candidate - candidate[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        return candidate, {"cartesian_delta_v11": delta}


class SubcarrierMotionBranch(nn.Module):
    """Retain local frequency structure for the residual path only."""

    def __init__(self, output_size: int = 64, width: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        self.frequency = nn.Sequential(
            nn.Conv1d(4, width, 7, padding=3),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, groups=width),
            nn.Conv1d(width, width, 1),
            nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(dropout),
        )
        self.link_fusion = nn.Sequential(
            nn.Linear(width * 2 * C.N_LINKS + C.N_LINKS, output_size),
            nn.GELU(), nn.LayerNorm(output_size),
        )

    def forward(self, normalized: torch.Tensor,
                link_mask: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(normalized)
        delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        value = torch.cat((normalized, delta), dim=-1)
        batch, frames, links, subcarriers = value.shape[:4]
        value = value.reshape(
            batch * frames * links, subcarriers, 4
        ).transpose(1, 2)
        encoded = self.frequency(value)
        encoded = torch.cat((encoded.mean(-1), encoded.amax(-1)), dim=-1)
        encoded = encoded.reshape(batch, frames, links, -1)
        mask = link_mask.to(encoded.dtype)
        flattened = (encoded * mask[..., None]).flatten(2)
        return self.link_fusion(torch.cat((flattened, mask), dim=-1))


class P2V11SubcarrierHybridNet(P2V9HybridNet):
    """P2 adapter with a lightweight raw subcarrier residual branch."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, raw_size: int = 64):
        super().__init__(base, hidden=hidden, dropout=dropout)
        self.subcarrier_motion = SubcarrierMotionBranch(raw_size, dropout=dropout)
        self.input_projection = nn.Sequential(
            nn.Linear(self.residual_input_size + raw_size, hidden),
            nn.GELU(), nn.LayerNorm(hidden),
        )

    def residual_features(
        self, csi: torch.Tensor, link_mask: torch.Tensor, base: dict,
        pose: torch.Tensor, root: torch.Tensor,
    ) -> torch.Tensor:
        coarse = super().residual_features(csi, link_mask, base, pose, root)
        with torch.no_grad():
            normalizer = getattr(self.base, "norm", None)
            normalized = base.get("_v12_normalized_csi")
            if normalized is None:
                normalized = (
                    normalizer(csi, link_mask) if normalizer is not None else csi
                )
        raw = self.subcarrier_motion(normalized, link_mask)
        return torch.cat((coarse, raw), dim=-1)


class P2V11DirectRootHybridNet(P2V11SubcarrierHybridNet):
    """Predict bounded root positions directly instead of accumulating drift."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, raw_size: int = 64,
                 max_delta: float = 0.75):
        super().__init__(
            base, hidden=hidden, dropout=dropout, raw_size=raw_size,
        )
        self.max_root_delta = float(max_delta)
        self.root_trajectory_context = nn.Sequential(
            nn.Conv1d(hidden, hidden, 9, padding=4, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
        )
        self.root_position_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.root_position_head[-1].weight)
        nn.init.zeros_(self.root_position_head[-1].bias)

    def root_candidate(
        self, feature: torch.Tensor, root: torch.Tensor,
        valid: torch.Tensor, pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        trajectory_feature = self.root_trajectory_context(
            feature.transpose(1, 2)
        ).transpose(1, 2)
        position_delta = self.max_root_delta * torch.tanh(
            self.root_position_head(trajectory_feature)
        )
        first = valid.to(torch.long).argmax(1)
        origin = position_delta[
            torch.arange(len(position_delta), device=position_delta.device), first,
        ]
        position_delta = (position_delta - origin[:, None]) * valid[..., None]
        adjusted = root + anchor_delta[:, None] + position_delta
        return adjusted, {
            "root_anchor_delta_v10": anchor_delta,
            "root_position_delta_v11": position_delta,
        }


class P2V13StateRootHybridNet(P2V11DirectRootHybridNet):
    """Fuse direct root observations with an integrated velocity trajectory."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, raw_size: int = 64,
                 max_delta: float = 0.75, max_step: float = 0.025):
        super().__init__(
            base, hidden=hidden, dropout=dropout, raw_size=raw_size,
            max_delta=max_delta,
        )
        self.max_root_step = float(max_step)
        self.root_velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.root_state_gate = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.root_velocity_head[-1].weight)
        nn.init.zeros_(self.root_velocity_head[-1].bias)
        nn.init.zeros_(self.root_state_gate[-1].weight)
        nn.init.constant_(self.root_state_gate[-1].bias, 8.0)

    def root_candidate(
        self, feature: torch.Tensor, root: torch.Tensor,
        valid: torch.Tensor, pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        trajectory_feature = self.root_trajectory_context(
            feature.transpose(1, 2)
        ).transpose(1, 2)

        position_delta = self.max_root_delta * torch.tanh(
            self.root_position_head(trajectory_feature)
        )
        first = valid.to(torch.long).argmax(1)
        origin = position_delta[
            torch.arange(len(position_delta), device=position_delta.device), first,
        ]
        position_delta = (position_delta - origin[:, None]) * valid[..., None]
        direct = root + anchor_delta[:, None] + position_delta

        step_delta = self.max_root_step * torch.tanh(
            self.root_velocity_head(trajectory_feature)
        )
        step_delta[:, 0] = 0.0
        step_delta = step_delta * valid[..., None]
        base_step = torch.zeros_like(root)
        base_step[:, 1:] = root[:, 1:] - root[:, :-1]
        integrated = (
            root[:, :1] + anchor_delta[:, None]
            + torch.cumsum(base_step + step_delta, dim=1)
        )

        gate = torch.sigmoid(self.root_state_gate(trajectory_feature))
        gate = torch.where(valid[..., None], gate, torch.ones_like(gate))
        adjusted = gate * direct + (1.0 - gate) * integrated
        return adjusted, {
            "root_anchor_delta_v10": anchor_delta,
            "root_position_delta_v11": position_delta,
            "root_velocity_delta_v13": step_delta,
            "root_state_gate_v13": gate,
            "root_integrated_candidate_v13": integrated,
        }


class P2V13MotionRootHybridNet(P2V11DirectRootHybridNet):
    """Direct-root decoder with an auxiliary motion-observability head."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, raw_size: int = 64,
                 max_delta: float = 0.75):
        super().__init__(
            base, hidden=hidden, dropout=dropout, raw_size=raw_size,
            max_delta=max_delta,
        )
        self.motion_observation_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.motion_observation_head[-1].weight)
        nn.init.zeros_(self.motion_observation_head[-1].bias)

    def root_candidate(
        self, feature: torch.Tensor, root: torch.Tensor,
        valid: torch.Tensor, pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        adjusted, auxiliary = super().root_candidate(
            feature, root, valid, pooled
        )
        trajectory_feature = self.root_trajectory_context(
            feature.transpose(1, 2)
        ).transpose(1, 2)
        motion = self.motion_observation_head(trajectory_feature)
        auxiliary.update({
            "root_velocity_observation_v13": 1.5 * torch.tanh(motion[..., :3]),
            "pose_speed_observation_v13": motion[..., 3],
        })
        return adjusted, auxiliary


class P2V13ConditionedRootHybridNet(P2V13MotionRootHybridNet):
    """Feed the supervised motion estimate back into direct-root decoding."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, raw_size: int = 64,
                 max_delta: float = 0.75):
        super().__init__(
            base, hidden=hidden, dropout=dropout, raw_size=raw_size,
            max_delta=max_delta,
        )
        self.motion_condition = nn.Linear(4, hidden, bias=False)
        nn.init.zeros_(self.motion_condition.weight)

    def root_candidate(
        self, feature: torch.Tensor, root: torch.Tensor,
        valid: torch.Tensor, pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        trajectory_feature = self.root_trajectory_context(
            feature.transpose(1, 2)
        ).transpose(1, 2)
        motion_raw = self.motion_observation_head(trajectory_feature)
        motion = torch.cat((
            1.5 * torch.tanh(motion_raw[..., :3]),
            motion_raw[..., 3:4],
        ), dim=-1)
        conditioned = trajectory_feature + self.motion_condition(motion)
        position_delta = self.max_root_delta * torch.tanh(
            self.root_position_head(conditioned)
        )
        first = valid.to(torch.long).argmax(1)
        origin = position_delta[
            torch.arange(len(position_delta), device=position_delta.device), first,
        ]
        position_delta = (position_delta - origin[:, None]) * valid[..., None]
        adjusted = root + anchor_delta[:, None] + position_delta
        return adjusted, {
            "root_anchor_delta_v10": anchor_delta,
            "root_position_delta_v11": position_delta,
            "root_velocity_observation_v13": motion[..., :3],
            "pose_speed_observation_v13": motion[..., 3],
            "motion_condition_v13": self.motion_condition(motion),
        }


class P2V13DecoupledMotionRootHybridNet(P2V13ConditionedRootHybridNet):
    """Keep root context stable while a separate branch learns CSI motion."""

    def __init__(self, base: nn.Module, hidden: int = 128,
                 dropout: float = 0.05, raw_size: int = 64,
                 max_delta: float = 0.75):
        super().__init__(
            base, hidden=hidden, dropout=dropout, raw_size=raw_size,
            max_delta=max_delta,
        )
        self.motion_trajectory_context = copy.deepcopy(
            self.root_trajectory_context
        )

    def initialize_motion_context_from_root(self) -> None:
        self.motion_trajectory_context.load_state_dict(
            self.root_trajectory_context.state_dict()
        )

    def root_candidate(
        self, feature: torch.Tensor, root: torch.Tensor,
        valid: torch.Tensor, pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        anchor_delta = 0.30 * torch.tanh(self.root_anchor_head(pooled))
        root_feature = self.root_trajectory_context(
            feature.transpose(1, 2)
        ).transpose(1, 2)
        motion_feature = self.motion_trajectory_context(
            feature.transpose(1, 2)
        ).transpose(1, 2)
        motion_raw = self.motion_observation_head(motion_feature)
        motion = torch.cat((
            1.5 * torch.tanh(motion_raw[..., :3]),
            motion_raw[..., 3:4],
        ), dim=-1)
        conditioned = root_feature + self.motion_condition(motion)
        position_delta = self.max_root_delta * torch.tanh(
            self.root_position_head(conditioned)
        )
        first = valid.to(torch.long).argmax(1)
        origin = position_delta[
            torch.arange(len(position_delta), device=position_delta.device), first,
        ]
        position_delta = (position_delta - origin[:, None]) * valid[..., None]
        adjusted = root + anchor_delta[:, None] + position_delta
        return adjusted, {
            "root_anchor_delta_v10": anchor_delta,
            "root_position_delta_v11": position_delta,
            "root_velocity_observation_v13": motion[..., :3],
            "pose_speed_observation_v13": motion[..., 3],
            "motion_condition_v13": self.motion_condition(motion),
            "motion_temporal_features_v13": motion_feature,
        }


def build_residual_hybrid(base: nn.Module, decoder: str = "dense") -> P2V9HybridNet:
    if decoder == "dense":
        return P2V9HybridNet(base)
    if decoder == "graph":
        return P2V11GraphHybridNet(base)
    if decoder == "spectral":
        return P2V11SpectralHybridNet(base)
    if decoder == "cartesian":
        return P2V11CartesianHybridNet(base)
    if decoder == "subcarrier":
        return P2V11SubcarrierHybridNet(base)
    if decoder == "direct_root":
        return P2V11DirectRootHybridNet(base)
    if decoder == "state_root":
        return P2V13StateRootHybridNet(base)
    if decoder == "motion_root":
        return P2V13MotionRootHybridNet(base)
    if decoder == "conditioned_root":
        return P2V13ConditionedRootHybridNet(base)
    if decoder == "decoupled_motion_root":
        return P2V13DecoupledMotionRootHybridNet(base)
    raise ValueError(f"unknown residual decoder: {decoder}")


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


class ClassificationExpertBlend(nn.Module):
    """Keep primary pose/root and use an independently trained logit expert."""

    def __init__(self, primary: nn.Module, expert: nn.Module):
        super().__init__()
        self.primary = primary
        self.expert = expert
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.class_strength = 1.0
        self.risk_strength = 1.0

    def set_calibration(self, classification: float, risk: float) -> None:
        if any(not 0.0 <= value <= 1.0 for value in (classification, risk)):
            raise ValueError("logit strengths must be between 0 and 1")
        self.class_strength = float(classification)
        self.risk_strength = float(risk)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
            expert = (
                self.expert.forward_logits(csi, link_mask)
                if hasattr(self.expert, "forward_logits")
                else self.expert(csi, link_mask)
            )
        output = dict(primary)
        output["class_logits"] = primary["class_logits"] + self.class_strength * (
            expert["class_logits"] - primary["class_logits"]
        )
        output["risk_logits"] = primary["risk_logits"] + self.risk_strength * (
            expert["risk_logits"] - primary["risk_logits"]
        )
        return output


class HierarchicalRiskCalibration(nn.Module):
    """Reconcile the 3-risk head with probabilities from the 17-class head."""

    CLASS_RANGES = ((0, 9), (9, 12), (12, 17))

    def __init__(self, base: nn.Module, class_weight: float = 0.0,
                 danger_logit_bias: float = 0.0):
        super().__init__()
        self.base = base
        self.set_calibration(class_weight, danger_logit_bias)

    def set_calibration(self, class_weight: float,
                        danger_logit_bias: float) -> None:
        if not 0.0 <= class_weight <= 1.0:
            raise ValueError("hierarchical class weight must be in [0, 1]")
        self.class_weight = float(class_weight)
        self.danger_logit_bias = float(danger_logit_bias)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = dict(self.base(csi, link_mask))
        class_probability = torch.softmax(output["class_logits"], dim=-1)
        class_risk = torch.stack([
            class_probability[:, start:stop].sum(-1)
            for start, stop in self.CLASS_RANGES
        ], dim=-1)
        risk_probability = torch.softmax(output["risk_logits"], dim=-1)
        mixed = (
            (1.0 - self.class_weight) * risk_probability
            + self.class_weight * class_risk
        ).clamp_min(1e-8)
        output["risk_logits_raw"] = output["risk_logits"]
        output["risk_logits"] = mixed.log()
        output["risk_logits"][:, 2] += self.danger_logit_bias
        return output


class InputMomentCalibration(nn.Module):
    """Align per-trial complex I/Q moments to the training reference."""

    def __init__(self, base: nn.Module, reference_mu: torch.Tensor,
                 reference_sigma: torch.Tensor, strength: float = 0.0,
                 epsilon: float = 1e-4):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        reference_mean = reference_mu.mean(1)
        centered = reference_mu - reference_mean[:, None]
        covariance = torch.einsum("lsi,lsj->lij", centered, centered)
        covariance = covariance / reference_mu.shape[1]
        covariance = covariance + torch.diag_embed(
            reference_sigma.square().mean(1)
        )
        self.register_buffer("reference_mean", reference_mean)
        self.register_buffer("reference_covariance", covariance)
        self.epsilon = float(epsilon)
        self.strength = 0.0
        self.set_calibration(strength)

    def set_calibration(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("moment calibration strength must be in [0, 1]")
        self.strength = float(strength)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _matrix_power(self, covariance: torch.Tensor,
                      exponent: float) -> torch.Tensor:
        identity = torch.eye(
            2, dtype=covariance.dtype, device=covariance.device
        )
        eigenvalue, eigenvector = torch.linalg.eigh(
            covariance + self.epsilon * identity
        )
        powered = eigenvalue.clamp_min(self.epsilon).pow(exponent)
        return eigenvector @ torch.diag_embed(powered) @ eigenvector.transpose(-2, -1)

    def calibrate(self, csi: torch.Tensor,
                  link_mask: torch.Tensor) -> torch.Tensor:
        if not self.strength:
            return csi
        weight = link_mask[..., None, None].to(csi.dtype)
        count = link_mask.sum(1).to(csi.dtype) * csi.shape[3]
        mean = (csi * weight).sum((1, 3)) / count[..., None].clamp_min(1.0)
        centered = csi - mean[:, None, :, None]
        weighted = centered * weight
        covariance = torch.einsum(
            "btlsi,btlsj->blij", weighted, centered
        ) / count[..., None, None].clamp_min(1.0)
        observed_inverse = self._matrix_power(covariance, -0.5)
        reference_sqrt = self._matrix_power(
            self.reference_covariance.to(csi.dtype), 0.5
        )
        transform = observed_inverse @ reference_sqrt[None]
        aligned = torch.einsum(
            "btlsi,blij->btlsj", centered, transform
        ) + self.reference_mean.to(csi.dtype)[None, None, :, None]
        calibrated = csi + self.strength * (aligned - csi)
        return calibrated * weight

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        return self.base(self.calibrate(csi, link_mask), link_mask)


class RootComponentBlend(nn.Module):
    """Gate a clean P2 root expert's anchor and integrated steps separately."""

    def __init__(self, primary: nn.Module, root_expert: nn.Module):
        super().__init__()
        self.primary = primary
        self.root_expert = root_expert
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.anchor_strength = 0.0
        self.step_strength = 0.0

    def set_calibration(self, anchor: float, step: float) -> None:
        if any(not 0.0 <= value <= 1.0 for value in (anchor, step)):
            raise ValueError("root component strengths must be between 0 and 1")
        self.anchor_strength = float(anchor)
        self.step_strength = float(step)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
            expert = self.root_expert(csi, link_mask)
        required = ("root_anchor_delta_v10", "root_step_delta_v10")
        if any(key not in expert for key in required):
            raise KeyError("root component calibration requires a P2 hybrid expert")
        root = primary["root"]
        base_step = torch.zeros_like(root)
        base_step[:, 1:] = root[:, 1:] - root[:, :-1]
        adjusted = (
            root[:, :1]
            + self.anchor_strength * expert["root_anchor_delta_v10"][:, None]
            + torch.cumsum(
                base_step
                + self.step_strength * expert["root_step_delta_v10"], dim=1,
            )
        )
        output = dict(primary)
        output["root_primary"] = root
        output["root"] = adjusted
        return output


def sequence_bone_projection(pose: torch.Tensor, valid: torch.Tensor,
                             symmetric: bool = False) -> torch.Tensor:
    """Keep predicted directions but make bone lengths constant per trial."""
    bones = _local_bones(pose)
    lengths = torch.linalg.vector_norm(bones, dim=-1)
    masked = lengths.masked_fill(~valid[..., None], float("nan"))
    canonical = torch.nanmedian(masked, dim=1).values
    fallback = lengths.mean(1)
    canonical = torch.where(torch.isfinite(canonical), canonical, fallback)
    if symmetric:
        names = {name: index for index, name in enumerate(C.JOINT_NAMES)}
        for left_name, left_index in names.items():
            if not left_name.startswith("left_"):
                continue
            right_index = names.get("right_" + left_name.removeprefix("left_"))
            if right_index is None:
                continue
            average = 0.5 * (
                canonical[:, left_index] + canonical[:, right_index]
            )
            canonical[:, left_index] = average
            canonical[:, right_index] = average
    direction = F.normalize(bones, dim=-1)
    projected_bones = direction * canonical[:, None, :, None]
    projected_bones[:, :, C.ROOT_JOINT] = 0.0
    return _forward_kinematics(projected_bones)


class SequenceBoneCalibration(nn.Module):
    """Apply CSI-only trial-level bone-length consistency to a pose model."""

    def __init__(self, base: nn.Module, blend: float = 0.0,
                 symmetric: bool = False):
        super().__init__()
        self.base = base
        self.blend = float(blend)
        self.symmetric = bool(symmetric)

    def set_calibration(self, blend: float, symmetric: bool) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError("bone calibration blend must be between 0 and 1")
        self.blend = float(blend)
        self.symmetric = bool(symmetric)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = dict(self.base(csi, link_mask))
        projected = sequence_bone_projection(
            output["pose_rel"], link_mask.any(-1), self.symmetric
        )
        output["pose_rel_raw"] = output["pose_rel"]
        output["pose_rel"] = output["pose_rel"] + self.blend * (
            projected - output["pose_rel"]
        )
        return output


class AnatomicalResidualCalibration(nn.Module):
    """Gate pose residuals with three symmetric anatomical groups."""

    def __init__(self, base: nn.Module, torso: float = 0.35,
                 arms: float = 0.35, legs: float = 0.35):
        super().__init__()
        self.base = base
        group_index = torch.zeros(C.N_JOINTS, dtype=torch.long)
        for name in ("left_arm", "right_arm"):
            group_index[list(C.JOINT_GROUPS[name])] = 1
        for name in ("left_leg", "right_leg"):
            group_index[list(C.JOINT_GROUPS[name])] = 2
        self.register_buffer("group_index", group_index, persistent=False)
        self.set_calibration(torso, arms, legs)

    def set_calibration(self, torso: float, arms: float, legs: float) -> None:
        values = (torso, arms, legs)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("anatomical strengths must be between 0 and 1")
        self.strengths = tuple(float(value) for value in values)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = dict(self.base(csi, link_mask))
        if "pose_p2" not in output:
            raise KeyError("anatomical calibration requires pose_p2")
        base_bones = _local_bones(output["pose_p2"])
        refined_bones = _local_bones(output["pose_rel"])
        strengths = base_bones.new_tensor(self.strengths)[self.group_index]
        mixed = base_bones + strengths[None, None, :, None] * (
            refined_bones - base_bones
        )
        length = torch.linalg.vector_norm(base_bones, dim=-1, keepdim=True)
        mixed = F.normalize(mixed, dim=-1) * length
        mixed[:, :, C.ROOT_JOINT] = 0.0
        output["pose_rel_uniform"] = output["pose_rel"]
        output["pose_rel"] = _forward_kinematics(mixed)
        return output


class PoseModelEnsemble(nn.Module):
    """Equal-protocol ensemble used only after seed-level validation checks."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        if not models:
            raise ValueError("pose ensemble needs at least one model")
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models) or sum(weights) <= 0:
            raise ValueError("ensemble weights must match models and have positive sum")
        normalized = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", normalized / normalized.sum())

    def set_weights(self, weights: list[float]) -> None:
        if len(weights) != len(self.models) or sum(weights) <= 0:
            raise ValueError("ensemble weights must match models and have positive sum")
        normalized = self.weights.new_tensor(weights)
        self.weights.copy_(normalized / normalized.sum())

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            predictions = [model(csi, link_mask) for model in self.models]
        output = dict(predictions[0])
        for key in ("pose_rel", "root", "class_logits", "risk_logits"):
            output[key] = sum(
                self.weights[index] * prediction[key]
                for index, prediction in enumerate(predictions)
            )
        return output

    def forward_logits(self, csi: torch.Tensor,
                       link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            predictions = [
                model.forward_logits(csi, link_mask)
                if hasattr(model, "forward_logits")
                else model(csi, link_mask)
                for model in self.models
            ]
        return {
            key: sum(
                self.weights[index] * prediction[key]
                for index, prediction in enumerate(predictions)
            )
            for key in ("class_logits", "risk_logits")
        }


class SharedBackboneCache(nn.Module):
    """Memoize one verified-identical frozen backbone per top-level forward."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self._active = False
        self._key = None
        self._output = None

    def begin(self) -> None:
        self._active = True
        self._key = None
        self._output = None

    def end(self) -> None:
        self._active = False
        self._key = None
        self._output = None

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @property
    def norm(self):
        """Preserve the P2 normalization interface used by raw-CSI experts."""
        return getattr(self.base, "norm", None)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        if not self._active:
            return self.base(csi, link_mask)
        key = (
            csi.data_ptr(), link_mask.data_ptr(), tuple(csi.shape),
            tuple(link_mask.shape), csi._version, link_mask._version,
        )
        if key != self._key:
            self._output = dict(self.base(csi, link_mask))
            self._output["_v12_motion_features"] = p2_motion_features(
                csi, link_mask
            )
            normalizer = self.norm
            self._output["_v12_normalized_csi"] = (
                normalizer(csi, link_mask) if normalizer is not None else csi
            )
            self._key = key
        return self._output


class SharedBackboneExecution(nn.Module):
    """Scope a shared-backbone cache to exactly one top-level inference."""

    def __init__(self, model: nn.Module, backbone: SharedBackboneCache):
        super().__init__()
        self.model = model
        self.backbone = backbone
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        self.backbone.begin()
        try:
            output = dict(self.model(csi, link_mask))
            output.pop("_v12_motion_features", None)
            output.pop("_v12_normalized_csi", None)
            return output
        finally:
            self.backbone.end()


class RiskAdaptivePoseBlend(nn.Module):
    """Blend two pose experts continuously using CSI-predicted danger risk."""

    def __init__(self, primary: nn.Module, expert: nn.Module,
                 non_danger_strength: float = 1.0,
                 danger_strength: float = 0.5,
                 danger_logit_bias: float = 1.1,
                 gate_mode: str = "probability"):
        super().__init__()
        self.primary = primary
        self.expert = expert
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.set_calibration(
            non_danger_strength, danger_strength, danger_logit_bias, gate_mode
        )

    def set_calibration(self, non_danger: float, danger: float,
                        danger_logit_bias: float,
                        gate_mode: str = "probability") -> None:
        if any(not 0.0 <= value <= 1.0 for value in (non_danger, danger)):
            raise ValueError("pose blend strengths must be in [0, 1]")
        self.non_danger_strength = float(non_danger)
        self.danger_strength = float(danger)
        self.danger_logit_bias = float(danger_logit_bias)
        if gate_mode not in {"probability", "hard"}:
            raise ValueError(f"unknown pose gate mode: {gate_mode}")
        self.gate_mode = gate_mode

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
            expert = self.expert(csi, link_mask)
        logits = primary["risk_logits"].clone()
        logits[:, 2] += self.danger_logit_bias
        if self.gate_mode == "probability":
            danger_gate = torch.softmax(logits, dim=-1)[:, 2]
        else:
            danger_gate = logits.argmax(-1).eq(2).to(logits.dtype)
        strength = self.non_danger_strength + danger_gate * (
            self.danger_strength - self.non_danger_strength
        )
        output = dict(primary)
        output["pose_rel"] = primary["pose_rel"] + strength[:, None, None, None] * (
            expert["pose_rel"] - primary["pose_rel"]
        )
        output["pose_expert_strength"] = strength
        return output


class ConditionalLinkFailurePoseBlend(nn.Module):
    """Use a robustness expert only when fewer than the expected links survive."""

    def __init__(self, primary: nn.Module, expert: nn.Module,
                 strength: float = 0.0, expected_links: int = C.N_LINKS,
                 minimum_link_coverage: float = 0.0,
                 partial_strength_scale: float = 1.0):
        super().__init__()
        self.primary = primary
        self.expert = expert
        self.expected_links = int(expected_links)
        self.minimum_link_coverage = float(minimum_link_coverage)
        if not 0.0 <= self.minimum_link_coverage <= 1.0:
            raise ValueError("minimum link coverage must be in [0, 1]")
        self.partial_strength_scale = float(partial_strength_scale)
        if not 0.0 <= self.partial_strength_scale <= 1.0:
            raise ValueError("partial pose strength scale must be in [0, 1]")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.set_strength(strength)

    def set_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("link-failure pose strength must be in [0, 1]")
        self.strength = float(strength)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
        output = dict(primary)
        coverage = link_mask.float().mean(dim=1)
        available = coverage.gt(self.minimum_link_coverage).sum(dim=-1)
        failed = available < self.expected_links
        if self.strength == 0.0 or not failed.any():
            output["link_failure_gate"] = failed
            return output
        with torch.no_grad():
            expert = self.expert(csi[failed], link_mask[failed])
        partial = coverage[failed].amin(dim=-1).gt(0.0)
        strength = torch.full_like(
            partial, self.strength, dtype=primary["pose_rel"].dtype
        )
        strength[partial] *= self.partial_strength_scale
        pose = primary["pose_rel"].clone()
        pose[failed] = pose[failed] + strength[:, None, None, None] * (
            expert["pose_rel"] - pose[failed]
        )
        output["pose_rel"] = pose
        output["link_failure_gate"] = failed
        output["link_failure_pose_strength"] = strength
        return output


class ConditionalLinkFailureLogitBlend(nn.Module):
    """Probability-blend a classifier expert only for missing-link samples."""

    def __init__(self, primary: nn.Module, expert: nn.Module,
                 class_strength: float | list[float] = 0.0,
                 risk_strength: float | list[float] = 0.0,
                 danger_logit_bias: float | list[float] = 0.0,
                 expected_links: int = C.N_LINKS,
                 minimum_link_coverage: float = 0.0):
        super().__init__()
        self.primary = primary
        self.expert = expert
        self.expected_links = int(expected_links)
        self.minimum_link_coverage = float(minimum_link_coverage)
        if not 0.0 <= self.minimum_link_coverage <= 1.0:
            raise ValueError("minimum link coverage must be in [0, 1]")
        self.register_buffer(
            "class_strengths", torch.zeros(self.expected_links)
        )
        self.register_buffer(
            "risk_strengths", torch.zeros(self.expected_links)
        )
        self.register_buffer(
            "danger_logit_biases", torch.zeros(self.expected_links)
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.set_strengths(class_strength, risk_strength, danger_logit_bias)

    def _values(self, value: float | list[float]) -> torch.Tensor:
        values = [float(value)] * self.expected_links if isinstance(
            value, (float, int)
        ) else list(value)
        if len(values) != self.expected_links:
            raise ValueError("link-specific calibration must match link count")
        return self.class_strengths.new_tensor(values)

    def set_strengths(self, class_strength: float | list[float],
                      risk_strength: float | list[float],
                      danger_logit_bias: float | list[float] = 0.0) -> None:
        class_values = self._values(class_strength)
        risk_values = self._values(risk_strength)
        if ((class_values < 0) | (class_values > 1)).any() or (
            (risk_values < 0) | (risk_values > 1)
        ).any():
            raise ValueError("link-failure logit strengths must be in [0, 1]")
        self.class_strengths.copy_(class_values)
        self.risk_strengths.copy_(risk_values)
        self.danger_logit_biases.copy_(self._values(danger_logit_bias))

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @staticmethod
    def _mix(primary: torch.Tensor, expert: torch.Tensor,
             strength: torch.Tensor) -> torch.Tensor:
        amount = strength[..., None]
        probability = (1.0 - amount) * torch.softmax(primary, dim=-1)
        probability += amount * torch.softmax(expert, dim=-1)
        return probability.clamp_min(1e-8).log()

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
        output = dict(primary)
        coverage = link_mask.float().mean(dim=1)
        alive = coverage.gt(self.minimum_link_coverage)
        available = alive.sum(dim=-1)
        failed = available < self.expected_links
        if (
            not failed.any()
            or (
                not self.class_strengths.any()
                and not self.risk_strengths.any()
                and not self.danger_logit_biases.any()
            )
        ):
            output["link_failure_gate"] = failed
            return output
        with torch.no_grad():
            expert = (
                self.expert.forward_logits(csi[failed], link_mask[failed])
                if hasattr(self.expert, "forward_logits")
                else self.expert(csi[failed], link_mask[failed])
            )
        missing = coverage[failed].argmin(dim=-1)
        class_strength = self.class_strengths[missing]
        risk_strength = self.risk_strengths[missing]
        danger_bias = self.danger_logit_biases[missing]
        if class_strength.any():
            logits = primary["class_logits"].clone()
            logits[failed] = self._mix(
                logits[failed], expert["class_logits"], class_strength
            )
            output["class_logits"] = logits
        if risk_strength.any():
            logits = primary["risk_logits"].clone()
            logits[failed] = self._mix(
                logits[failed], expert["risk_logits"], risk_strength
            )
            output["risk_logits"] = logits
        if danger_bias.any():
            logits = output["risk_logits"].clone()
            logits[failed, C.N_RISK - 1] += danger_bias
            output["risk_logits"] = logits
        output["link_failure_gate"] = failed
        return output


class ConditionalLinkFailureRootBlend(nn.Module):
    """Blend a direct-root specialist only for missing-link samples."""

    def __init__(self, primary: nn.Module, expert: nn.Module,
                 strength: float = 0.0, expected_links: int = C.N_LINKS,
                 minimum_link_coverage: float = 0.0,
                 secondary_expert: nn.Module | None = None,
                 secondary_strength: float = 0.0,
                 secondary_links: tuple[int, ...] = (),
                 partial_strength_scale: float = 1.0):
        super().__init__()
        self.primary = primary
        self.expert = expert
        self.secondary_expert = secondary_expert
        self.expected_links = int(expected_links)
        self.minimum_link_coverage = float(minimum_link_coverage)
        if not 0.0 <= self.minimum_link_coverage <= 1.0:
            raise ValueError("minimum link coverage must be in [0, 1]")
        self.partial_strength_scale = float(partial_strength_scale)
        if not 0.0 <= self.partial_strength_scale <= 1.0:
            raise ValueError("partial root strength scale must be in [0, 1]")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.set_strength(strength)
        self.secondary_strength = float(secondary_strength)
        if not 0.0 <= self.secondary_strength <= 1.0:
            raise ValueError("secondary root strength must be in [0, 1]")
        self.secondary_links = tuple(int(link) for link in secondary_links)
        if any(
            link < 0 or link >= self.expected_links
            for link in self.secondary_links
        ):
            raise ValueError("secondary root link is out of range")
        if self.secondary_links and self.secondary_expert is None:
            raise ValueError("secondary root links require a secondary expert")

    def set_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("link-failure root strength must be in [0, 1]")
        self.strength = float(strength)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            primary = self.primary(csi, link_mask)
        output = dict(primary)
        coverage = link_mask.float().mean(dim=1)
        available = coverage.gt(self.minimum_link_coverage).sum(dim=-1)
        failed = available < self.expected_links
        if not failed.any() or (
            self.strength == 0.0 and self.secondary_strength == 0.0
        ):
            output["link_failure_gate"] = failed
            return output
        root = primary["root"].clone()
        failed_index = torch.nonzero(failed, as_tuple=False).flatten()
        missing = coverage[failed].argmin(dim=-1)
        partial = coverage[failed].amin(dim=-1).gt(0.0)
        scale = torch.ones_like(missing, dtype=root.dtype)
        scale[partial] = self.partial_strength_scale
        secondary = torch.zeros_like(missing, dtype=torch.bool)
        for link in self.secondary_links:
            secondary |= missing.eq(link)
        main_index = failed_index[~secondary]
        if self.strength and len(main_index):
            with torch.no_grad():
                expert = self.expert(csi[main_index], link_mask[main_index])
            amount = self.strength * scale[~secondary]
            root[main_index] = root[main_index] + amount[:, None, None] * (
                expert["root"] - root[main_index]
            )
        secondary_index = failed_index[secondary]
        if self.secondary_strength and len(secondary_index):
            with torch.no_grad():
                expert = self.secondary_expert(
                    csi[secondary_index], link_mask[secondary_index]
                )
            amount = self.secondary_strength * scale[secondary]
            root[secondary_index] = root[secondary_index] + (
                amount[:, None, None]
                * (expert["root"] - root[secondary_index])
            )
        output["root"] = root
        output["link_failure_gate"] = failed
        missing_link = torch.full(
            (len(coverage),), -1, dtype=torch.long, device=coverage.device
        )
        missing_link[failed] = missing
        output["link_failure_missing_link"] = missing_link
        output["link_failure_root_strength_scale"] = scale
        return output
