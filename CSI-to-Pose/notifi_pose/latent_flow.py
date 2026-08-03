"""CSI-conditioned rectified flow over a frozen kinematic motion prior."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from . import losses as L
from .nets import GraphPoseNet, LocalTemporalBlock
from .v3 import KinematicBoneDecoder, MotionPriorEncoder


class ConditionalLatentFlow(nn.Module):
    """Predict a rectified-flow velocity conditioned on CSI temporal features."""

    def __init__(self, hidden: int, blocks: int, dropout: float):
        super().__init__()
        self.latent_projection = nn.Linear(hidden, hidden)
        self.condition_projection = nn.Linear(hidden, hidden)
        self.time_projection = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        dilations = (1, 2, 4, 8)
        self.blocks = nn.ModuleList(
            LocalTemporalBlock(hidden, dilations[index % len(dilations)], dropout)
            for index in range(max(1, blocks + 1))
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)

    @staticmethod
    def time_code(time: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            time,
            torch.sin(torch.pi * time),
            torch.cos(torch.pi * time),
            torch.sin(2.0 * torch.pi * time),
        ), dim=-1)

    def forward(self, latent: torch.Tensor, time: torch.Tensor,
                condition: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_projection(self.time_code(time))[:, None]
        hidden = (
            self.latent_projection(latent)
            + self.condition_projection(condition)
            + time_embedding
        )
        for block in self.blocks:
            hidden = block(hidden)
        velocity = self.velocity(hidden)
        return velocity * valid[..., None].to(velocity.dtype)


class LatentFlowPoseNet(GraphPoseNet):
    """Robust GraphFormer plus deterministic CSI-conditioned latent flow.

    GT poses are encoded by a frozen kinematic prior. During training, conditional
    flow matching learns a vector field from a CSI-derived latent (with noise) to
    that GT latent. Inference integrates the field with a fixed midpoint solver,
    so CSI-only output is deterministic and reproducible.
    """

    adaptation_parameter_prefixes = (
        "latent_condition.", "conditional_flow.", "flow_mix",
    )

    def __init__(self, hidden: int = 128, n_blocks: int = 3,
                 dropout: float = 0.1, heads: int = 4,
                 graph_blocks: int = 2, decoder: str = "hybrid",
                 domain_grl: float = 0.2, flow_steps: int = 4,
                 flow_noise: float = 0.25, **kwargs: object):
        super().__init__(
            hidden=hidden, n_blocks=n_blocks, dropout=dropout, heads=heads,
            graph_blocks=graph_blocks, decoder=decoder, robust_heads=True,
            domain_grl=domain_grl, **kwargs,
        )
        self.flow_steps = flow_steps
        self.flow_noise = flow_noise
        self.latent_condition = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden)
        )
        self.conditional_flow = ConditionalLatentFlow(hidden, n_blocks, dropout)
        self.target_motion_encoder = MotionPriorEncoder(
            hidden, n_blocks, heads, dropout
        )
        self.flow_decoder = KinematicBoneDecoder(hidden, graph_blocks, dropout)
        self.flow_mix = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "motion_prior_loaded", torch.zeros(1, dtype=torch.bool)
        )
        for module in (self.target_motion_encoder, self.flow_decoder):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def allows_warm_start_missing(self, key: str) -> bool:
        return key == "flow_mix" or key == "motion_prior_loaded" or key.startswith((
            "latent_condition.", "conditional_flow.",
            "target_motion_encoder.", "flow_decoder.",
        ))

    @torch.no_grad()
    def set_bone_lengths(self, lengths: torch.Tensor) -> None:
        self.target_motion_encoder.set_bone_lengths(lengths)
        self.flow_decoder.set_bone_lengths(lengths)

    @torch.no_grad()
    def load_motion_prior(self, checkpoint: dict) -> None:
        self.target_motion_encoder.load_state_dict(checkpoint["encoder"])
        self.flow_decoder.load_state_dict(checkpoint["decoder"])
        for module in (self.target_motion_encoder, self.flow_decoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.motion_prior_loaded.fill_(True)

    def encode_pose_target(self, pose: torch.Tensor,
                           valid: torch.Tensor) -> torch.Tensor:
        self.target_motion_encoder.eval()
        with torch.no_grad():
            return self.target_motion_encoder(pose, valid)

    def integrate_flow(self, initial: torch.Tensor, condition: torch.Tensor,
                       valid: torch.Tensor) -> torch.Tensor:
        latent = initial
        step_size = 1.0 / max(self.flow_steps, 1)
        for step in range(max(self.flow_steps, 1)):
            time = latent.new_full((len(latent),), step * step_size)
            first = self.conditional_flow(latent, time, condition, valid)
            midpoint = latent + 0.5 * step_size * first
            middle_time = time + 0.5 * step_size
            second = self.conditional_flow(midpoint, middle_time, condition, valid)
            latent = latent + step_size * second
        return latent * valid[..., None].to(latent.dtype)

    def flow_matching_per_sample(self, output: dict, batch: dict) -> torch.Tensor:
        valid = batch["valid"]
        target = self.encode_pose_target(batch["pose_rel"], valid)
        initial = output["flow_initial_latent"].detach()
        noisy_initial = initial + self.flow_noise * torch.randn_like(initial)
        time = torch.rand(len(initial), device=initial.device)
        weight = time[:, None, None]
        interpolated = (1.0 - weight) * noisy_initial + weight * target
        target_velocity = target - noisy_initial
        predicted_velocity = self.conditional_flow(
            interpolated, time, output["temporal_features"], valid
        )
        return L.smooth_l1_per_sample(
            predicted_velocity, target_velocity, valid, beta=0.20
        )

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        output = super().forward(csi, link_mask)
        condition = output["temporal_features"]
        valid = link_mask.any(-1)
        initial = self.latent_condition(condition)
        latent = self.integrate_flow(initial, condition, valid)
        prior_pose, bone_direction = self.flow_decoder(latent, valid)
        coarse = output["pose_rel"]
        mix = 0.25 * torch.tanh(self.flow_mix)
        pose = coarse + mix * (prior_pose - coarse)
        pose = pose - pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
        output.update({
            "pose_rel": pose,
            "pose_coarse": coarse,
            "motion_latent": latent,
            "flow_initial_latent": initial,
            "flow_prior_pose": prior_pose,
            "flow_bone_direction": bone_direction,
            "flow_mix": mix,
        })
        return output

    def describe(self) -> str:
        return (
            f"LatentFlowPoseNet(hidden={self.hidden}, flow_steps={self.flow_steps}, "
            f"decoder={self.decoder_kind}, frozen_kinematic_prior) "
            f"params={self.n_params():,}"
        )
