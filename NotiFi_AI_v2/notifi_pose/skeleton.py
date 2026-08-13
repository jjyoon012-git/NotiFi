"""배포 pose에 필요한 최소 SMPL body-22 기구학 보정을 제공한다."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import contract as C


def local_bones(pose: torch.Tensor) -> torch.Tensor:
    """pelvis-relative 관절 좌표를 부모 관절 기준 뼈 벡터로 바꾼다."""
    bones = torch.zeros_like(pose)
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent >= 0:
            bones[:, :, child] = pose[:, :, child] - pose[:, :, parent]
    return bones


def forward_kinematics(bones: torch.Tensor) -> torch.Tensor:
    """SMPL body-22 부모 트리를 따라 뼈 벡터를 관절 좌표로 복원한다."""
    joints = []
    for child, parent in enumerate(C.JOINT_PARENTS):
        if parent < 0:
            joints.append(torch.zeros_like(bones[:, :, child]))
        else:
            joints.append(joints[parent] + bones[:, :, child])
    return torch.stack(joints, dim=2)


def sequence_bone_projection(
    pose: torch.Tensor,
    valid: torch.Tensor,
    symmetric: bool = False,
) -> torch.Tensor:
    """관절 방향은 유지하면서 trial 전체의 뼈 길이를 일정하게 만든다."""
    bones = local_bones(pose)
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
            right_name = "right_" + left_name.removeprefix("left_")
            right_index = names.get(right_name)
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
    return forward_kinematics(projected_bones)


__all__ = ("forward_kinematics", "local_bones", "sequence_bone_projection")
