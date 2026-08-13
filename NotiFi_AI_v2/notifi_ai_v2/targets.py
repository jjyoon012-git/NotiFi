"""Dense, root-offset-independent motion targets derived from GVHMR GT."""

from __future__ import annotations

import torch
import torch.nn.functional as F


PARTS = {
    "left_arm": (16, 18, 20),
    "right_arm": (17, 19, 21),
    "left_leg": (4, 7, 10),
    "right_leg": (5, 8, 11),
}


def motion_targets(
    pose_rel: torch.Tensor,
    root: torch.Tensor,
    valid: torch.Tensor,
    fps: float = 30.0,
    maximum_speed: float = 4.0,
) -> torch.Tensor:
    """Return torso direction and five normalized body-part speed channels."""

    if pose_rel.ndim != 4 or pose_rel.shape[-2:] != (22, 3):
        raise ValueError("pose_rel must have shape [B,T,22,3]")
    if root.shape != pose_rel.shape[:2] + (3,) or valid.shape != pose_rel.shape[:2]:
        raise ValueError("root or valid shape does not match pose")

    torso = pose_rel[:, :, 12] - pose_rel[:, :, 0]
    torso = F.normalize(torso, dim=-1)
    world = pose_rel + root[:, :, None]
    velocity = torch.zeros_like(world)
    pair = valid[:, 1:] & valid[:, :-1]
    velocity[:, 1:] = (
        world[:, 1:] - world[:, :-1]
    ) * pair[..., None, None].to(world.dtype) * fps
    speed = velocity.norm(dim=-1)
    channels = [speed.mean(dim=-1)]
    for joints in PARTS.values():
        channels.append(speed[:, :, joints].mean(dim=-1))
    normalized_speed = torch.stack(channels, dim=-1).clamp(0.0, maximum_speed)
    normalized_speed = normalized_speed / maximum_speed
    output = torch.cat((torso, normalized_speed), dim=-1)
    return output * valid[..., None].to(output.dtype)
