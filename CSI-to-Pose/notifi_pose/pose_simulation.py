"""CSI-only source trajectory retrieval에 필요한 시간 정렬과 pose 지표."""

from __future__ import annotations

import torch

from .losses import distal_mpjpe, mpjpe, pa_mpjpe


def best_motion_shift(
    predicted: torch.Tensor,
    candidate: torch.Tensor,
    valid: torch.Tensor,
    max_shift: int = 6,
) -> int:
    """CSI 예측 motion과 source GT 후보가 가장 가까운 전역 시간 이동을 찾는다."""
    best = (float("inf"), 0)
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            left, right = predicted[:shift], candidate[-shift:]
            mask = valid[:shift] & valid[-shift:]
        elif shift > 0:
            left, right = predicted[shift:], candidate[:-shift]
            mask = valid[shift:] & valid[:-shift]
        else:
            left, right, mask = predicted, candidate, valid
        if not bool(mask.any()):
            continue
        error = float((left[mask] - right[mask]).square().mean())
        if error < best[0]:
            best = (error, shift)
    return best[1]


def shift_pose(pose: torch.Tensor, shift: int) -> torch.Tensor:
    """CSI motion으로 선택한 시간 이동을 source pose sequence에 적용한다."""
    if shift == 0:
        return pose
    output = torch.empty_like(pose)
    if shift > 0:
        output[shift:] = pose[:-shift]
        output[:shift] = pose[0]
    else:
        amount = -shift
        output[:-amount] = pose[amount:]
        output[-amount:] = pose[-1]
    return output


def retrieval_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    risk: torch.Tensor,
) -> dict[str, float]:
    """root-relative 전체·distal·PA 오차와 danger subset 오차를 계산한다."""
    danger = risk == 2
    return {
        "pose_cm": 100.0 * mpjpe(predicted, target, valid),
        "distal_cm": 100.0 * distal_mpjpe(predicted, target, valid),
        "pa_pose_cm": 100.0 * pa_mpjpe(predicted, target, valid),
        "danger_pose_cm": (
            100.0 * mpjpe(predicted[danger], target[danger], valid[danger])
            if bool(danger.any()) else float("nan")
        ),
        "danger_distal_cm": (
            100.0 * distal_mpjpe(
                predicted[danger], target[danger], valid[danger]
            ) if bool(danger.any()) else float("nan")
        ),
    }
