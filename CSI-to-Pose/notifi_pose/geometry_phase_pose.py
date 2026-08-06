"""KP3 pose model with an auxiliary temporal-phase projection head."""

from __future__ import annotations

import copy

import torch
from torch import nn

from . import contract as C
from .hierarchical_pose import HierarchicalCSIPoseRegressor
from .nets import PoseTemporalRefiner


class GeometryPhasePoseRegressor(HierarchicalCSIPoseRegressor):
    """Separate phase discrimination from the pose-regression latent space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_head = copy.deepcopy(self.backbone.latent_head)

    def _is_trainable_key(self, key: str) -> bool:
        return key.startswith("phase_head.") or super()._is_trainable_key(key)

    @torch.no_grad()
    def initialize_phase_head_from_pose(self) -> None:
        self.phase_head.load_state_dict(self.backbone.latent_head.state_dict())

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = super().forward(csi, link_mask)
        phase_latent = self.phase_head(
            output["csi_pose_features"].transpose(1, 2)
        ).transpose(1, 2)
        return {**output, "phase_motion_latent": phase_latent}


class GeometryPhaseCoarseResidual(nn.Module):
    """Refine a stable coarse pose with KP3 temporal CSI features.

    The zero-initialized residual decoder makes the initial output exactly the
    coarse V13S pose. The source pose remains available as a diagnostic, but it
    is not blended into the final output.
    """

    def __init__(self, pose_model: GeometryPhasePoseRegressor,
                 dropout: float = 0.08, max_delta: float = 0.25,
                 proposal_strength: float = 0.0):
        super().__init__()
        if not 0.0 <= proposal_strength <= 1.0:
            raise ValueError("proposal strength must be in [0, 1]")
        self.pose_model = pose_model
        self.proposal_strength = float(proposal_strength)
        hidden = int(pose_model.backbone.latent_head.in_channels)
        joint_scale = torch.ones(C.N_JOINTS)
        distal = set(
            C.JOINT_GROUPS["head"]
            + C.JOINT_GROUPS["left_arm"][-1:]
            + C.JOINT_GROUPS["right_arm"][-1:]
            + C.JOINT_GROUPS["left_leg"][-2:]
            + C.JOINT_GROUPS["right_leg"][-2:]
        )
        for joint in distal:
            joint_scale[joint] = 1.20
        self.refiner = PoseTemporalRefiner(
            hidden, dropout=dropout, max_delta=max_delta,
            joint_scale=joint_scale.tolist(),
        )

    @property
    def backbone(self):
        return self.pose_model.backbone

    def set_residual_strength(self, strength: float) -> None:
        if float(strength) != 1.0:
            raise ValueError("KP3-CMR uses a fixed residual strength of 1.0")

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if value.dtype.is_floating_point and (
                key.startswith("refiner.")
                or (
                    key.startswith("pose_model.")
                    and self.pose_model._is_trainable_key(
                        key.removeprefix("pose_model.")
                    )
                )
            )
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state) - set(current))
        if unknown:
            raise RuntimeError(f"unknown coarse-residual weights: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor) -> dict:
        source = self.pose_model(csi, link_mask)
        frame_mask = link_mask.any(-1)
        proposal = coarse_pose + self.proposal_strength * (
            source["pose_rel"] - coarse_pose
        )
        proposal = proposal - proposal[
            :, :, C.ROOT_JOINT:C.ROOT_JOINT + 1
        ]
        pose = self.refiner(
            proposal, source["csi_pose_features"], frame_mask
        )
        return {
            **source,
            "pose_candidate": source["pose_rel"],
            "pose_v13s": coarse_pose,
            "pose_coarse": proposal,
            "pose_delta": pose - proposal,
            "pose_rel": pose,
        }
