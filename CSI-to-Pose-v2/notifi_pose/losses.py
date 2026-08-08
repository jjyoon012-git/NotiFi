"""손실 함수.

주 타깃은 pose_rel(골반 기준 22관절)과 root(골반 절대 위치)이고,
class/risk 는 보조다. 보조 과제를 두는 이유는 분류 성능 자체가 아니라 **pose head 가
'전부 서있음'으로 붕괴하는 것을 막기 위해서**다(legacy 에서 실제로 겪은 문제).

이때 감독 라벨은 risk(3종)가 아니라 class(17종)여야 한다. risk 로 감독하면
서있기·앉기·눕기가 전부 safe 로 묶여 자세를 구분하라는 신호가 되지 못한다.

타깃은 정규화하지 않고 미터 단위 그대로 쓴다. 손실 값이 곧 오차(m)라 해석이 쉽고,
pose_rel 범위가 대략 ±1m 라 스케일 문제도 없다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C

#: SmoothL1 의 전환점(m). 이보다 작은 오차는 제곱, 크면 절댓값으로 벌점을 준다.
#: 관절 오차 5cm 를 기준으로 잡았다. 기본값 1.0 을 쓰면 우리 범위(±1m)에서는
#: 사실상 전부 제곱이 되어 이상값 하나가 학습을 흔든다.
SMOOTH_L1_BETA = 0.05

DISTAL_JOINT_NAMES = (
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_foot", "right_foot", "head", "left_wrist", "right_wrist",
)
DISTAL_JOINTS = tuple(C.JOINT_INDEX[name] for name in DISTAL_JOINT_NAMES)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, beta: float = SMOOTH_L1_BETA,
                     frame_weight: torch.Tensor | None = None) -> torch.Tensor:
    """mask [B, T] 가 True 인 프레임만 채점한다.

    패딩 구간(짧은 trial 의 뒷부분)과 GT 가 없는 absence trial 이 여기서 걸러진다.
    """
    l = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    while mask.dim() < l.dim():
        mask = mask.unsqueeze(-1)
    m = mask.to(l.dtype)
    if frame_weight is not None:
        weight = frame_weight
        while weight.dim() < l.dim():
            weight = weight.unsqueeze(-1)
        m = m * weight.to(l.dtype)
    return (l * m).sum() / m.expand_as(l).sum().clamp(min=1.0)


def masked_per_sample(loss: torch.Tensor, mask: torch.Tensor,
                      frame_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce an element-wise loss to one value per batch item."""
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    weight = mask.to(loss.dtype)
    if frame_weight is not None:
        frame_weight = frame_weight
        while frame_weight.dim() < loss.dim():
            frame_weight = frame_weight.unsqueeze(-1)
        weight = weight * frame_weight.to(loss.dtype)
    numerator = (loss * weight).flatten(1).sum(1)
    denominator = weight.expand_as(loss).flatten(1).sum(1).clamp_min(1.0)
    return numerator / denominator


def smooth_l1_per_sample(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor, beta: float = SMOOTH_L1_BETA,
                         frame_weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return masked_per_sample(loss, mask, frame_weight)


def target_motion(pose_rel: torch.Tensor, root: torch.Tensor,
                  valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical body speed [m/s] and valid adjacent-frame mask."""
    absolute = pose_rel + root[:, :, None, :]
    pair_valid = valid[:, 1:] & valid[:, :-1]
    speed = torch.zeros_like(root[..., 0])
    delta = torch.linalg.vector_norm(absolute[:, 1:] - absolute[:, :-1], dim=-1)
    speed[:, 1:] = delta.mean(-1) * C.TARGET_FPS
    speed = speed * valid.to(speed.dtype)
    return speed, pair_valid


def impact_window(pose_rel: torch.Tensor, root: torch.Tensor,
                  valid: torch.Tensor, risk: torch.Tensor,
                  radius: int = 5) -> torch.Tensor:
    """Locate a GT acceleration peak and mark its local danger-trial window.

    This is a training/evaluation target derived only from GT. It is never used
    as a model input, so CSI-only inference remains unchanged.
    """
    selected = torch.zeros_like(valid, dtype=torch.bool)
    if pose_rel.shape[1] < 3:
        return selected
    absolute = pose_rel + root[:, :, None, :]
    acceleration = (
        absolute[:, 2:] - 2.0 * absolute[:, 1:-1] + absolute[:, :-2]
    ) * (C.TARGET_FPS ** 2)
    energy = torch.linalg.vector_norm(acceleration, dim=-1).mean(-1)
    triplet_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    energy = energy.masked_fill(~triplet_valid, float("-inf"))
    for item in range(len(pose_rel)):
        if int(risk[item]) != 2 or not triplet_valid[item].any():
            continue
        peak = int(torch.argmax(energy[item]).item()) + 2
        start = max(0, peak - radius)
        stop = min(valid.shape[1], peak + radius + 1)
        selected[item, start:stop] = valid[item, start:stop]
    return selected


def weighted_joint_error_per_sample(pred_pose: torch.Tensor,
                                    pred_root: torch.Tensor,
                                    target_pose: torch.Tensor,
                                    target_root: torch.Tensor,
                                    mask: torch.Tensor) -> torch.Tensor:
    """Absolute-position error emphasizing injury-relevant distal joints."""
    pred = pred_pose + pred_root[:, :, None, :]
    target = target_pose + target_root[:, :, None, :]
    distance = torch.linalg.vector_norm(pred - target, dim=-1)
    weights = distance.new_ones(C.N_JOINTS)
    weights[list(DISTAL_JOINTS)] = 3.0
    weights = weights / weights.mean()
    return masked_per_sample(distance * weights[None, None], mask)


def velocity_loss(pred: torch.Tensor, target: torch.Tensor,
                  pair_valid: torch.Tensor) -> torch.Tensor:
    pred_velocity = (pred[:, 1:] - pred[:, :-1]) * C.TARGET_FPS
    target_velocity = (target[:, 1:] - target[:, :-1]) * C.TARGET_FPS
    return masked_smooth_l1(
        pred_velocity, target_velocity, pair_valid, beta=0.20
    )


def derivative_per_sample(pred: torch.Tensor, target: torch.Tensor,
                          pair_valid: torch.Tensor, order: int) -> torch.Tensor:
    pred_delta, target_delta = pred, target
    mask = pair_valid
    for derivative in range(order):
        pred_delta = (pred_delta[:, 1:] - pred_delta[:, :-1]) * C.TARGET_FPS
        target_delta = (target_delta[:, 1:] - target_delta[:, :-1]) * C.TARGET_FPS
        if derivative:
            mask = mask[:, 1:] & mask[:, :-1]
    return smooth_l1_per_sample(pred_delta, target_delta, mask, beta=0.20)


def displacement_per_sample(pred_pose: torch.Tensor, pred_root: torch.Tensor,
                            target_pose: torch.Tensor, target_root: torch.Tensor,
                            valid: torch.Tensor, lag: int = 5) -> torch.Tensor:
    """Match coherent average velocity over a multi-frame interval.

    Unlike adjacent-frame velocity, this cannot be minimized by frame-to-frame
    jitter that disappears under the five-frame evaluation smoother.
    """
    if pred_pose.shape[1] <= lag:
        return pred_pose.new_zeros(pred_pose.shape[0])
    predicted = pred_pose + pred_root[:, :, None, :]
    target = target_pose + target_root[:, :, None, :]
    scale = C.TARGET_FPS / lag
    predicted_delta = (predicted[:, lag:] - predicted[:, :-lag]) * scale
    target_delta = (target[:, lag:] - target[:, :-lag]) * scale
    interval_valid = valid[:, lag:] & valid[:, :-lag]
    return smooth_l1_per_sample(
        predicted_delta, target_delta, interval_valid, beta=0.20
    )


def cross_domain_supervised_contrastive(
    embedding: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Pull together the same action only when it comes from another RF domain."""
    if len(embedding) < 2:
        return embedding.new_zeros(())
    similarity = embedding @ embedding.transpose(0, 1) / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    valid_domain = domains >= 0
    positives = (
        (labels[:, None] == labels[None, :])
        & (domains[:, None] != domains[None, :])
        & valid_domain[:, None] & valid_domain[None, :] & ~identity
    )
    candidates = ~identity
    logits = similarity - similarity.max(1, keepdim=True).values.detach()
    log_probability = logits - torch.log(
        (torch.exp(logits) * candidates).sum(1, keepdim=True).clamp_min(1e-8)
    )
    count = positives.sum(1)
    anchors = count > 0
    if not anchors.any():
        return embedding.new_zeros(())
    return -(log_probability * positives).sum(1)[anchors].div(count[anchors]).mean()


def phase_targets(speed: torch.Tensor, valid: torch.Tensor,
                  risk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create pre-motion, transition, impact, and post-impact fall phases."""
    target = torch.zeros_like(speed, dtype=torch.long)
    mask = valid & (risk[:, None] == 2)
    for index in range(len(speed)):
        positions = torch.nonzero(mask[index], as_tuple=False).flatten()
        if len(positions) == 0:
            continue
        active = positions[speed[index, positions] > 0.25]
        if len(active) == 0:
            continue
        peak = int(torch.argmax(speed[index]).item())
        first, last = int(active[0]), int(active[-1])
        target[index, first:peak] = 1
        target[index, max(first, peak - 2):min(last + 1, peak + 3)] = 2
        target[index, min(last + 1, peak + 3):] = 3
    return target, mask


def contact_targets(pose: torch.Tensor, root: torch.Tensor,
                    valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    names = {name: index for index, name in enumerate(C.JOINT_NAMES)}
    indices = [names[name] for name in (
        "left_ankle", "right_ankle", "left_foot", "right_foot"
    )]
    feet = pose[:, :, indices] + root[:, :, None]
    floor = feet[..., 1].masked_fill(~valid[..., None], float("inf")).amin((1, 2))
    floor = torch.where(torch.isfinite(floor), floor, torch.zeros_like(floor))
    velocity = torch.zeros_like(feet[..., 0])
    velocity[:, 1:] = torch.linalg.vector_norm(
        feet[:, 1:] - feet[:, :-1], dim=-1
    ) * C.TARGET_FPS
    near_floor = feet[..., 1] - floor[:, None, None] < 0.08
    contact = near_floor & (velocity < 0.25) & valid[..., None]
    return contact, floor


class BoneLoss(nn.Module):
    """뼈 길이 일관성 (legacy L_bone 을 22관절로 이식).

    관절별 오차만 채점하면 팔·다리가 프레임마다 늘었다 줄었다 하는 해부학적 붕괴를
    잡지 못한다. 예측과 GT 각각에서 뼈 21개의 길이를 재고 그 차이에 벌점을 준다.

    GVHMR GT 의 뼈 길이는 프레임 간 상대표준편차가 0.000000 이라(전수 확인)
    제약 조건으로 삼기에 이상적이다.

    절대좌표가 아니라 root-relative 좌표로 계산해도 무방하다 — 뼈는 관절 간 차이라
    root 평행이동에 불변이다.
    """

    def __init__(self):
        super().__init__()
        edges = np.asarray(C.SKELETON_EDGES, dtype=np.int64)   # [21, 2] (parent, child)
        self.register_buffer("idx_a", torch.from_numpy(edges[:, 0]))
        self.register_buffer("idx_b", torch.from_numpy(edges[:, 1]))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """pred/target [B, T, 22, 3], mask [B, T]"""
        lp = torch.linalg.norm(pred[:, :, self.idx_a] - pred[:, :, self.idx_b], dim=-1)
        lt = torch.linalg.norm(target[:, :, self.idx_a] - target[:, :, self.idx_b], dim=-1)
        m = mask.to(lp.dtype).unsqueeze(-1)
        return (((lp - lt) ** 2) * m).sum() / m.expand_as(lp).sum().clamp(min=1.0)

    def per_sample(self, pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        lp = torch.linalg.norm(
            pred[:, :, self.idx_a] - pred[:, :, self.idx_b], dim=-1
        )
        lt = torch.linalg.norm(
            target[:, :, self.idx_a] - target[:, :, self.idx_b], dim=-1
        )
        return masked_per_sample((lp - lt) ** 2, mask)


def inverse_freq_weights(counts: np.ndarray, device=None) -> torch.Tensor:
    """클래스 역빈도 가중. 다수 클래스가 CE 를 지배하지 않게 한다.

    우리 개발셋은 17클래스 90~270개(3.0배), 3클래스 450~1349개(3.0배)로 불균형하다.
    """
    c = np.asarray(counts, dtype=np.float64)
    w = c.sum() / (len(c) * np.maximum(c, 1.0))
    return torch.tensor(w, dtype=torch.float32, device=device)


class PoseLoss(nn.Module):
    """전체 손실 = pose + λ_root·root + λ_bone·bone + λ_cls·CE(17) + λ_risk·CE(3)"""

    def __init__(self, class_counts: np.ndarray | None = None,
                 risk_counts: np.ndarray | None = None,
                 lambda_root: float = 1.0, lambda_bone: float = 0.1,
                 lambda_cls: float = 0.2, lambda_risk: float = 0.1,
                 lambda_velocity: float = 0.0, lambda_motion: float = 0.0,
                 lambda_acceleration: float = 0.0,
                 lambda_jerk: float = 0.0, lambda_impact: float = 0.0,
                 lambda_coarse: float = 0.0,
                 lambda_displacement: float = 0.0,
                 lambda_flow: float = 0.0,
                 lambda_contact: float = 0.0, lambda_phase: float = 0.0,
                 lambda_foot_slide: float = 0.0, lambda_floor: float = 0.0,
                 lambda_domain: float = 0.0, lambda_supcon: float = 0.0,
                 lambda_latent: float = 0.0, motion_weight: float = 0.0,
                 device=None):
        super().__init__()
        self.bone = BoneLoss()
        self.l_root = lambda_root
        self.l_bone = lambda_bone
        self.l_cls = lambda_cls
        self.l_risk = lambda_risk
        self.l_velocity = lambda_velocity
        self.l_motion = lambda_motion
        self.l_acceleration = lambda_acceleration
        self.l_jerk = lambda_jerk
        self.l_impact = lambda_impact
        self.l_coarse = lambda_coarse
        self.l_displacement = lambda_displacement
        self.l_flow = lambda_flow
        self.l_contact = lambda_contact
        self.l_phase = lambda_phase
        self.l_foot_slide = lambda_foot_slide
        self.l_floor = lambda_floor
        self.l_domain = lambda_domain
        self.l_supcon = lambda_supcon
        self.l_latent = lambda_latent
        self.motion_weight = motion_weight
        cw = inverse_freq_weights(class_counts, device) if class_counts is not None else None
        rw = inverse_freq_weights(risk_counts, device) if risk_counts is not None else None
        self.ce_class = nn.CrossEntropyLoss(weight=cw)
        self.ce_risk = nn.CrossEntropyLoss(weight=rw)

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        valid = batch["valid"]                                  # [B, T]
        speed, pair_valid = target_motion(batch["pose_rel"], batch["root"], valid)
        frame_weight = 1.0 + self.motion_weight * (speed / 1.0).clamp(0.0, 1.0)
        pose_ps = smooth_l1_per_sample(
            out["pose_rel"], batch["pose_rel"], valid, frame_weight=frame_weight
        )
        root_ps = smooth_l1_per_sample(
            out["root"], batch["root"], valid, frame_weight=frame_weight
        )
        bone_ps = self.bone.per_sample(out["pose_rel"], batch["pose_rel"], valid)
        cls_ps = F.cross_entropy(
            out["class_logits"], batch["class_id"],
            weight=self.ce_class.weight, reduction="none",
        )
        risk_ps = F.cross_entropy(
            out["risk_logits"], batch["risk_id"],
            weight=self.ce_risk.weight, reduction="none",
        )

        velocity_ps = torch.zeros_like(pose_ps)
        if self.l_velocity:
            velocity_ps = derivative_per_sample(
                out["pose_rel"], batch["pose_rel"], pair_valid, 1
            ) + 0.25 * derivative_per_sample(
                out["root"], batch["root"], pair_valid, 1
            )

        acceleration_ps = torch.zeros_like(pose_ps)
        if self.l_acceleration:
            acceleration_ps = derivative_per_sample(
                out["pose_rel"], batch["pose_rel"], pair_valid, 2
            ) + 0.25 * derivative_per_sample(
                out["root"], batch["root"], pair_valid, 2
            )

        jerk_ps = torch.zeros_like(pose_ps)
        if self.l_jerk:
            jerk_ps = derivative_per_sample(
                out["pose_rel"], batch["pose_rel"], pair_valid, 3
            ) + 0.25 * derivative_per_sample(
                out["root"], batch["root"], pair_valid, 3
            )

        impact_ps = torch.zeros_like(pose_ps)
        if self.l_impact:
            impact_mask = impact_window(
                batch["pose_rel"], batch["root"], valid, batch["risk_id"]
            )
            impact_ps = weighted_joint_error_per_sample(
                out["pose_rel"], out["root"], batch["pose_rel"], batch["root"],
                impact_mask,
            )

        coarse_ps = torch.zeros_like(pose_ps)
        if self.l_coarse and "pose_coarse" in out:
            coarse_ps = smooth_l1_per_sample(
                out["pose_coarse"], batch["pose_rel"], valid,
                frame_weight=frame_weight,
            )

        displacement_ps = torch.zeros_like(pose_ps)
        if self.l_displacement:
            displacement_ps = displacement_per_sample(
                out["pose_rel"], out["root"], batch["pose_rel"], batch["root"],
                valid,
            )

        flow_ps = torch.zeros_like(pose_ps)
        if self.l_flow and "flow_per_sample" in out:
            flow_ps = out["flow_per_sample"]

        motion_ps = torch.zeros_like(pose_ps)
        if self.l_motion and "motion" in out:
            target = torch.log1p(speed * 10.0)
            motion_ps = smooth_l1_per_sample(
                out["motion"], target, valid, beta=0.10
            )

        phase_ps = torch.zeros_like(pose_ps)
        if self.l_phase and "phase_logits" in out:
            target, phase_mask = phase_targets(speed, valid, batch["risk_id"])
            element = F.cross_entropy(
                out["phase_logits"].transpose(1, 2), target, reduction="none"
            )
            phase_ps = masked_per_sample(element, phase_mask)

        contact_ps = torch.zeros_like(pose_ps)
        foot_slide_ps = torch.zeros_like(pose_ps)
        floor_ps = torch.zeros_like(pose_ps)
        if any((self.l_contact, self.l_foot_slide, self.l_floor)):
            contact, floor = contact_targets(
                batch["pose_rel"], batch["root"], valid
            )
            if self.l_contact and "contact_logits" in out:
                element = F.binary_cross_entropy_with_logits(
                    out["contact_logits"], contact.to(out["contact_logits"].dtype),
                    reduction="none",
                )
                contact_ps = masked_per_sample(element, valid)

            names = {name: index for index, name in enumerate(C.JOINT_NAMES)}
            feet_index = [names[name] for name in (
                "left_ankle", "right_ankle", "left_foot", "right_foot"
            )]
            predicted_feet = (
                out["pose_rel"][:, :, feet_index] + out["root"][:, :, None]
            )
            if self.l_foot_slide:
                foot_speed = torch.linalg.vector_norm(
                    predicted_feet[:, 1:] - predicted_feet[:, :-1], dim=-1
                ) * C.TARGET_FPS
                stable = contact[:, 1:] & contact[:, :-1] & pair_valid[..., None]
                foot_slide_ps = masked_per_sample(foot_speed, stable)
            if self.l_floor:
                penetration = F.relu(
                    floor[:, None, None] - predicted_feet[..., 1]
                )
                floor_ps = masked_per_sample(penetration, valid)

        domain_ps = torch.zeros_like(pose_ps)
        if self.l_domain and "domain_logits" in out and "domain_id" in batch:
            domain_valid = batch["domain_id"] >= 0
            if domain_valid.any():
                domain_ps[domain_valid] = F.cross_entropy(
                    out["domain_logits"][domain_valid], batch["domain_id"][domain_valid],
                    reduction="none",
                )

        latent_ps = torch.zeros_like(pose_ps)
        if self.l_latent and "target_motion_latent" in out:
            latent_ps = smooth_l1_per_sample(
                out["motion_latent"], out["target_motion_latent"], valid, beta=0.10
            )

        per_sample = (
            pose_ps + self.l_root * root_ps + self.l_bone * bone_ps
            + self.l_cls * cls_ps + self.l_risk * risk_ps
            + self.l_velocity * velocity_ps
            + self.l_acceleration * acceleration_ps
            + self.l_jerk * jerk_ps + self.l_impact * impact_ps
            + self.l_coarse * coarse_ps
            + self.l_displacement * displacement_ps
            + self.l_flow * flow_ps
            + self.l_motion * motion_ps + self.l_phase * phase_ps
            + self.l_contact * contact_ps + self.l_foot_slide * foot_slide_ps
            + self.l_floor * floor_ps + self.l_domain * domain_ps
            + self.l_latent * latent_ps
        )
        supcon = out["pose_rel"].new_zeros(())
        if self.l_supcon and "embedding" in out and "domain_id" in batch:
            supcon = cross_domain_supervised_contrastive(
                out["embedding"], batch["class_id"], batch["domain_id"]
            )
        total = per_sample.mean() + self.l_supcon * supcon

        tensors = {
            "total": total, "pose": pose_ps.mean(), "root": root_ps.mean(),
            "bone": bone_ps.mean(), "cls": cls_ps.mean(), "risk": risk_ps.mean(),
            "velocity": velocity_ps.mean(), "acceleration": acceleration_ps.mean(),
            "jerk": jerk_ps.mean(), "impact": impact_ps.mean(),
            "coarse": coarse_ps.mean(), "displacement": displacement_ps.mean(),
            "flow": flow_ps.mean(),
            "motion": motion_ps.mean(), "phase": phase_ps.mean(),
            "contact": contact_ps.mean(), "foot_slide": foot_slide_ps.mean(),
            "floor": floor_ps.mean(), "domain": domain_ps.mean(),
            "supcon": supcon, "latent": latent_ps.mean(),
        }
        parts = {key: float(value.detach()) for key, value in tensors.items()}
        parts["_per_sample_total"] = per_sample
        parts["_supcon_total"] = self.l_supcon * supcon
        return total, parts


# ------------------------------------------------------------------ 평가 지표


@torch.no_grad()
def mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean Per Joint Position Error (m). pose 복원의 표준 지표.

    관절마다 예측과 GT 사이 거리를 재서 평균낸다. root-relative 좌표에 적용하면
    '자세가 얼마나 맞았나', 절대좌표에 적용하면 '위치까지 포함해 얼마나 맞았나'다.
    """
    d = torch.linalg.norm(pred - target, dim=-1)                # [B, T, J]
    m = mask.to(d.dtype).unsqueeze(-1)
    return float((d * m).sum() / m.expand_as(d).sum().clamp(min=1.0))


@torch.no_grad()
def root_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """골반 절대 위치 오차 (m)."""
    d = torch.linalg.norm(pred - target, dim=-1)                # [B, T]
    m = mask.to(d.dtype)
    return float((d * m).sum() / m.sum().clamp(min=1.0))


@torch.no_grad()
def distal_mpjpe(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> float:
    """MPJPE over knees, ankles, feet, head, and wrists."""
    return mpjpe(pred[:, :, DISTAL_JOINTS], target[:, :, DISTAL_JOINTS], mask)


@torch.no_grad()
def impact_mpjpe(pred_pose: torch.Tensor, pred_root: torch.Tensor,
                 target_pose: torch.Tensor, target_root: torch.Tensor,
                 valid: torch.Tensor, risk: torch.Tensor) -> float:
    """Absolute-position MPJPE around GT impact peaks; NaN if none exist."""
    mask = impact_window(target_pose, target_root, valid, risk)
    if not mask.any():
        return float("nan")
    pred = pred_pose + pred_root[:, :, None, :]
    target = target_pose + target_root[:, :, None, :]
    return mpjpe(pred, target, mask)


@torch.no_grad()
def pa_mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Procrustes 정렬 후 MPJPE. 전역 회전·평행이동·스케일을 맞춘 뒤의 순수 자세 오차.

    프레임 단위로 정렬한다. 유효 프레임이 없으면 nan.
    """
    B, T, J, _ = pred.shape
    p = pred.reshape(B * T, J, 3).double()
    t = target.reshape(B * T, J, 3).double()
    m = mask.reshape(B * T).bool()
    if m.sum() == 0:
        return float("nan")
    p, t = p[m], t[m]

    pc = p - p.mean(1, keepdim=True)
    tc = t - t.mean(1, keepdim=True)
    H = pc.transpose(1, 2) @ tc
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, dtype=p.dtype, device=p.device).expand(len(p), 3, 3).clone()
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    trace = S[:, 0] + S[:, 1] + d * S[:, 2]
    scale = trace / pc.square().sum((1, 2)).clamp(min=1e-12)
    aligned = scale[:, None, None] * (pc @ R.transpose(1, 2))
    return float(torch.linalg.norm(aligned - tc, dim=-1).mean())
