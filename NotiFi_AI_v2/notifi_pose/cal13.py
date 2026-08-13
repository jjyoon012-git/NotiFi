"""CAL13: 사람 크기를 제거한 동작 교사와 계층 분류를 결합한다."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from . import contract as C
from .cal12 import CAL12PhysicsDG


SPEED_GROUPS = (
    tuple(range(C.N_JOINTS)),
    (0, 3, 6, 9, 12, 15),
    (16, 18, 20),
    (17, 19, 21),
    (1, 4, 7, 10),
    (2, 5, 8, 11),
)
MOTION_DESCRIPTOR_DIM = len(SPEED_GROUPS) + 4


def pose_motion_descriptor(
    pose_rel: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """체형 크기를 제거한 몸통·사지 속도와 상대 높이 10채널을 만든다."""
    if pose_rel.ndim != 4 or pose_rel.shape[-2:] != (C.N_JOINTS, 3):
        raise ValueError("pose_rel must be [B,T,22,3]")
    if valid.shape != pose_rel.shape[:2]:
        raise ValueError("valid must match pose time axes")
    weight = valid.to(pose_rel.dtype)
    # pelvis-neck 길이는 성별·키보다 동작 모양을 남기기 위한 체형 기준이다.
    body_scale = torch.linalg.vector_norm(pose_rel[:, :, 12], dim=-1)
    body_scale = (
        (body_scale * weight).sum(1)
        / weight.sum(1).clamp_min(1.0)
    ).clamp_min(0.10)
    normalized = pose_rel / body_scale[:, None, None, None]
    velocity = torch.zeros_like(normalized)
    pair = valid[:, 1:] & valid[:, :-1]
    velocity[:, 1:] = (
        normalized[:, 1:] - normalized[:, :-1]
    ) * C.TARGET_FPS * pair[:, :, None, None].to(normalized.dtype)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    speed_channels = [
        torch.log1p(speed[:, :, list(group)].mean(-1))
        for group in SPEED_GROUPS
    ]
    vertical = normalized[..., 1]
    neck = normalized[:, :, 12]
    neck_norm = torch.linalg.vector_norm(neck, dim=-1).clamp_min(1e-4)
    state_channels = (
        vertical[:, :, 15],
        vertical[:, :, [20, 21]].mean(-1),
        vertical[:, :, [7, 8, 10, 11]].mean(-1),
        torch.linalg.vector_norm(neck[..., (0, 2)], dim=-1) / neck_norm,
    )
    descriptor = torch.stack((*speed_channels, *state_channels), dim=-1)
    descriptor = F.avg_pool1d(
        descriptor.transpose(1, 2), kernel_size=5, stride=1, padding=2
    ).transpose(1, 2)
    return descriptor * valid[..., None].to(descriptor.dtype)


def temporal_motion_signature(
    descriptor: torch.Tensor,
    valid: torch.Tensor,
    bins: int = 8,
) -> torch.Tensor:
    """평균·표준편차·최대값과 시간 순서 bin을 결합한 검색 벡터를 만든다."""
    weight = valid[..., None].to(descriptor.dtype)
    count = weight.sum(1).clamp_min(1.0)
    mean = (descriptor * weight).sum(1) / count
    variance = ((descriptor - mean[:, None]).square() * weight).sum(1) / count
    maximum = descriptor.masked_fill(~valid[..., None], -torch.inf).amax(1)
    maximum = torch.where(
        torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
    )
    segments = []
    boundaries = torch.linspace(
        0, descriptor.shape[1], bins + 1, device=descriptor.device
    ).round().long()
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        local_weight = weight[:, left:right]
        segments.append(
            (descriptor[:, left:right] * local_weight).sum(1)
            / local_weight.sum(1).clamp_min(1.0)
        )
    return torch.cat((mean, variance.sqrt(), maximum, *segments), dim=-1)


def shift_robust_motion_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    max_shift: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """trial별 ±shift 중 최적 정렬로 timestamp 미세 오차에 강한 회귀를 계산한다."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target descriptors must have equal shape")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("valid must match descriptor time axes")
    if max_shift < 0:
        raise ValueError("max_shift cannot be negative")
    candidates = []
    shifts = list(range(-max_shift, max_shift + 1))
    for shift in shifts:
        if shift < 0:
            pred = predicted[:, :shift]
            truth = target[:, -shift:]
            current_valid = valid[:, :shift] & valid[:, -shift:]
        elif shift > 0:
            pred = predicted[:, shift:]
            truth = target[:, :-shift]
            current_valid = valid[:, shift:] & valid[:, :-shift]
        else:
            pred = predicted
            truth = target
            current_valid = valid
        per_frame = F.smooth_l1_loss(
            pred, truth, reduction="none", beta=0.20
        ).mean(-1)
        per_trial = (
            (per_frame * current_valid.to(per_frame.dtype)).sum(1)
            / current_valid.sum(1).clamp_min(1)
        )
        candidates.append(per_trial)
    stacked = torch.stack(candidates, dim=1)
    best_loss, best_index = stacked.min(1)
    best_shift = predicted.new_tensor(shifts)[best_index]
    return best_loss.mean(), best_shift.float().mean().detach()


class CAL13MotionGrounded(CAL12PhysicsDG):
    """CAL12의 CSI 특징을 실제 신체 움직임에 고정하고 위험을 먼저 분리한다."""

    def __init__(
        self, *args, hierarchy_initial: float = 0.50,
        use_hierarchy: bool = True, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        initial = min(max(float(hierarchy_initial), 1e-4), 1.0 - 1e-4)
        self.hierarchy_initial = initial
        self.use_hierarchy = bool(use_hierarchy)
        self.hierarchy_logit = nn.Parameter(torch.tensor(math.log(initial / (1 - initial))))
        self.hierarchy_logit.requires_grad_(self.use_hierarchy)
        self.pose_motion_head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, MOTION_DESCRIPTOR_DIM),
        )

    def model_config(self) -> dict:
        """CAL13 재현에 필요한 구조 설정을 checkpoint에 기록한다."""
        config = super().model_config()
        config["hierarchy_initial"] = self.hierarchy_initial
        config["use_hierarchy"] = self.use_hierarchy
        return config

    @staticmethod
    def expand_risk_prior(risk_log_probability: torch.Tensor) -> torch.Tensor:
        """safe/warning/danger log 확률을 해당 17행동 구간으로 펼친다."""
        counts = (9, 3, 5)
        return torch.cat([
            risk_log_probability[:, group:group + 1].expand(-1, count)
            for group, count in enumerate(counts)
        ], dim=-1)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """위험 prior로 세부행동을 제한하고 학습용 관절 동작 예측을 함께 낸다."""
        output = super().forward(*args, **kwargs)
        unconditioned = output["action_logits"]
        direct_risk = output["direct_risk_logits"]
        hierarchy = (
            torch.sigmoid(self.hierarchy_logit)
            if self.use_hierarchy else self.hierarchy_logit.new_zeros(())
        )
        action_logits = unconditioned + hierarchy * self.expand_risk_prior(
            direct_risk.log_softmax(-1)
        )
        action_risk = self.action_to_risk(action_logits)
        fusion = torch.sigmoid(self.risk_fusion)
        risk_logits = (1.0 - fusion) * direct_risk + fusion * action_risk
        output.update({
            "unconditioned_action_logits": unconditioned,
            "action_logits": action_logits,
            "action_risk_logits": action_risk,
            "risk_logits": risk_logits,
            "hierarchy_strength": hierarchy,
            "pose_motion": self.pose_motion_head(output["query_features"]),
        })
        return output
