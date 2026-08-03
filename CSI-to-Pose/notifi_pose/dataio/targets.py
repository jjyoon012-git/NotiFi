"""GVHMR 22관절 타깃 로딩.

`gt_pose.npz` 는 subject 마다 스키마가 다르다(yja 789개만 SMPL 파라미터 포함).
공통으로 보장되는 키는 `joints_world` / `transl` / `frame_index` 뿐이며,
학습 경로는 이 셋만 쓴다. SMPL 파라미터에 의존하면 yja 가 test 로 빠지는 fold 에서
코드가 깨진다.

좌표계(실측 확인):
    Y-up, 바닥 y ≈ 0.  정지 서기에서 pelvis 0.877 / head 1.537 / 발 0.005.
    joints_world 는 이미 절대좌표다(root-relative 아님).

root 정의(주의):
    root = joints_world[:, 0] (pelvis).  `transl` 이 아니다. transl 은 SMPL 의
    pelvis translation 이라 joints_world[:, 0] 과 y축으로 약 -0.35m 어긋난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import contract as C


@dataclass
class PoseTarget:
    pose_rel: np.ndarray   # [T, 22, 3] root-relative
    root: np.ndarray       # [T, 3] 절대 root(pelvis) 위치
    valid: np.ndarray      # [T] bool
    frame_index: np.ndarray  # [T] int64
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.pose_rel.shape[0])

    def absolute(self) -> np.ndarray:
        """절대좌표 skeleton [T, 22, 3] 복원. GT root 를 빌려오는 게 아니라
        pose_rel + root 를 그대로 합치는 것이므로 예측에도 동일하게 쓴다."""
        return self.pose_rel + self.root[:, None, :]


class TargetError(RuntimeError):
    pass


def load_pose_target(gt_path: Path) -> PoseTarget:
    """gt_pose.npz → root-relative pose + 절대 root."""
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise TargetError(f"gt_pose.npz not found: {gt_path}")

    with np.load(gt_path, allow_pickle=False) as z:
        missing = [k for k in C.GT_REQUIRED_KEYS if k not in z.files]
        if missing:
            raise TargetError(f"{gt_path.name}: 필수 키 누락 {missing}")
        joints = np.asarray(z["joints_world"], dtype=np.float32)
        transl = np.asarray(z["transl"], dtype=np.float32)
        frame_index = np.asarray(z["frame_index"], dtype=np.int64)
        has_smpl = any(k.startswith("smpl_incam__") for k in z.files)

    if joints.ndim != 3 or joints.shape[1] != C.N_JOINTS or joints.shape[2] != 3:
        raise TargetError(
            f"{gt_path.name}: joints_world shape {joints.shape}, "
            f"expected [T, {C.N_JOINTS}, 3]"
        )
    if len(frame_index) != len(joints):
        raise TargetError(
            f"{gt_path.name}: frame_index({len(frame_index)}) != joints({len(joints)})"
        )

    root = joints[:, C.ROOT_JOINT].copy()
    pose_rel = joints - root[:, None, :]

    valid = np.isfinite(joints).all(axis=(1, 2))

    return PoseTarget(
        pose_rel=pose_rel,
        root=root,
        valid=valid,
        frame_index=frame_index,
        meta={
            "has_smpl": bool(has_smpl),
            "n_invalid_frames": int((~valid).sum()),
            # transl 과 root 의 차이 — 계약 위반(둘을 혼동)을 조기에 잡기 위한 기록
            "root_minus_transl_mean": (root - transl).mean(axis=0).tolist(),
        },
    )


def bone_lengths(joints_abs: np.ndarray) -> np.ndarray:
    """[T, 22, 3] → [T, 21] 뼈 길이. SKELETON_EDGES 순서."""
    a = np.asarray([p for p, _ in C.SKELETON_EDGES])
    b = np.asarray([c for _, c in C.SKELETON_EDGES])
    return np.linalg.norm(joints_abs[:, b] - joints_abs[:, a], axis=-1)
