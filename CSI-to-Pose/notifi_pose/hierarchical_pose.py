"""Hierarchical CSI-to-pose refinement for the KP2-DH model family."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .continuous_pose import CSILatentPoseRegressor
from .motion_tokens import forward_kinematics


TORSO_BONES = (1, 2, 3, 6, 9, 12, 13, 14, 15)
LIMB_BONES = tuple(
    joint for joint in range(1, C.N_JOINTS) if joint not in TORSO_BONES
)
DISTAL_JOINTS = tuple(sorted(set(
    C.JOINT_GROUPS["head"]
    + C.JOINT_GROUPS["left_arm"][-1:]
    + C.JOINT_GROUPS["right_arm"][-1:]
    + C.JOINT_GROUPS["left_leg"][-2:]
    + C.JOINT_GROUPS["right_leg"][-2:]
)))


def _zero_residual_head(hidden: int, output: int, dropout: float) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class HierarchicalCSIPoseRegressor(nn.Module):
    """Refine a continuous-latent pose through explicit body targets.

    The continuous latent remains a whole-body prior. Separate heads predict
    torso and limb directions, distal Cartesian residuals, and joint velocity.
    Zero-initialized residual heads make the untrained model exactly reproduce
    its KP2-C backbone.
    """

    def __init__(self, backbone: CSILatentPoseRegressor,
                 direction_scale: float = 1.0,
                 endpoint_scale: float = 0.40,
                 dropout: float = 0.08):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.latent_head.in_channels)
        self.direction_scale = float(direction_scale)
        self.endpoint_scale = float(endpoint_scale)
        self.torso_direction_head = _zero_residual_head(
            hidden, len(TORSO_BONES) * 3, dropout
        )
        self.limb_direction_head = _zero_residual_head(
            hidden, len(LIMB_BONES) * 3, dropout
        )
        self.endpoint_head = _zero_residual_head(
            hidden, len(DISTAL_JOINTS) * 3, dropout
        )
        self.velocity_head = _zero_residual_head(
            hidden, C.N_JOINTS * 3, dropout
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if value.dtype.is_floating_point and self._is_trainable_key(key)
        }

    def _is_trainable_key(self, key: str) -> bool:
        trainable_prefixes = (
            "backbone.dynamic.", "backbone.static_projection.",
            "backbone.fusion_gate.", "backbone.fusion_residual.",
            "backbone.refiner.", "backbone.latent_head.",
            "torso_direction_head.", "limb_direction_head.",
            "endpoint_head.", "velocity_head.",
        )
        return key.startswith(trainable_prefixes)

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"unknown hierarchical-pose weights: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)

    def load_kp2c_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Initialize the backbone from a KP2-C trainable checkpoint."""
        self.backbone.load_trainable_state_dict(state)

    @staticmethod
    def _place(values: torch.Tensor, joints: tuple[int, ...],
               output: torch.Tensor) -> torch.Tensor:
        output[:, :, list(joints)] = values
        return output

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        base = self.backbone(csi, link_mask)
        features = base["csi_pose_features"]
        batch, frames, _ = features.shape
        frame_mask = link_mask.any(-1)

        torso_delta = self.torso_direction_head(features).reshape(
            batch, frames, len(TORSO_BONES), 3
        )
        limb_delta = self.limb_direction_head(features).reshape(
            batch, frames, len(LIMB_BONES), 3
        )
        direction_delta = torch.zeros(
            batch, frames, C.N_JOINTS, 3,
            device=features.device, dtype=torso_delta.dtype,
        )
        direction_delta = self._place(
            torso_delta, TORSO_BONES, direction_delta
        )
        direction_delta = self._place(
            limb_delta, LIMB_BONES, direction_delta
        )
        directions = F.normalize(
            base["bone_direction"]
            + self.direction_scale * torch.tanh(direction_delta),
            dim=-1,
        )
        directions[:, :, C.ROOT_JOINT] = 0.0
        lengths = self.backbone.bone_lengths.expand(batch, -1)
        kinematic_pose = forward_kinematics(directions, lengths)

        endpoint_delta = self.endpoint_scale * torch.tanh(
            self.endpoint_head(features).reshape(
                batch, frames, len(DISTAL_JOINTS), 3
            )
        )
        pose = kinematic_pose.clone()
        pose[:, :, list(DISTAL_JOINTS)] = (
            pose[:, :, list(DISTAL_JOINTS)] + endpoint_delta
        )
        velocity = self.velocity_head(features).reshape(
            batch, frames, C.N_JOINTS, 3
        )

        valid_weight = frame_mask[..., None, None].to(pose.dtype)
        return {
            **base,
            "pose_rel": pose * valid_weight,
            "kinematic_pose": kinematic_pose * valid_weight,
            "bone_direction": directions * valid_weight,
            "torso_direction_delta": torso_delta,
            "limb_direction_delta": limb_delta,
            "endpoint_delta": endpoint_delta,
            "kinetic_velocity": velocity * valid_weight,
        }


class JointConfidencePoseGate(nn.Module):
    """Select coarse or hierarchical pose per frame and joint from CSI only."""

    def __init__(self, pose_model: HierarchicalCSIPoseRegressor,
                 initial_strength: float = 0.30,
                 hidden: int = 96, dropout: float = 0.10,
                 max_adjustment: float = 0.20):
        super().__init__()
        if not 0.0 < initial_strength < 1.0:
            raise ValueError("initial_strength must be strictly between 0 and 1")
        self.pose_model = pose_model
        if max_adjustment <= 0.0:
            raise ValueError("max_adjustment must be positive")
        if initial_strength - max_adjustment < 0.0:
            raise ValueError("gate adjustment would allow negative confidence")
        if initial_strength + max_adjustment > 1.0:
            raise ValueError("gate adjustment would exceed unit confidence")
        self.initial_strength = float(initial_strength)
        self.max_adjustment = float(max_adjustment)
        self.pose_model.eval()
        for parameter in self.pose_model.parameters():
            parameter.requires_grad_(False)
        input_dim = int(pose_model.backbone.latent_head.in_channels)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, C.N_JOINTS),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_model.eval()
        return self

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.gate_head.state_dict().items()
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.gate_head.load_state_dict(state, strict=True)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor) -> dict:
        with torch.no_grad():
            output = self.pose_model(csi, link_mask)
        gate = (
            self.initial_strength
            + self.max_adjustment * torch.tanh(
                self.gate_head(output["csi_pose_features"])
            )
        )
        valid = link_mask.any(-1)
        gate = gate * valid[..., None].to(gate.dtype)
        candidate = output["pose_rel"]
        delta = candidate - coarse_pose
        pose = coarse_pose + gate[..., None] * delta
        pose = pose * valid[..., None, None].to(pose.dtype)
        return {
            **output,
            "pose_rel": pose,
            "pose_coarse": coarse_pose,
            "pose_candidate": candidate,
            "pose_delta": delta,
            "joint_confidence_gate": gate,
        }
