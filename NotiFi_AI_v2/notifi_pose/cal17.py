"""CAL17: 기본자세의 domain 이동을 미관측 행동 prototype으로 전달한다."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .meta_calibration import MOTION_PROMPT_CLASSES


ANCHOR_CLASSES = tuple(int(value) for value in MOTION_PROMPT_CLASSES) + (6,)


def transport_class_prototypes(
    source_classes: torch.Tensor,
    source_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    anchor_class_ids: tuple[int, ...],
    strength: float,
    temperature: float,
) -> torch.Tensor:
    """기본자세별 source-target 이동을 유사한 미관측 행동 중심에 가중 전달한다."""
    if source_anchors.shape != target_anchors.shape:
        raise ValueError("source and target anchors must have equal shapes")
    if source_classes.shape[-1] != source_anchors.shape[-1]:
        raise ValueError("class and anchor dimensions must match")
    if len(anchor_class_ids) != len(source_anchors):
        raise ValueError("anchor ids must match anchor rows")
    source_classes = F.normalize(source_classes, dim=-1)
    source_anchors = F.normalize(source_anchors, dim=-1)
    target_anchors = F.normalize(target_anchors, dim=-1)
    similarity = source_classes @ source_anchors.transpose(0, 1)
    weights = torch.softmax(
        similarity / max(float(temperature), 1e-4), dim=-1
    )
    shift = weights @ (target_anchors - source_anchors)
    transported = F.normalize(
        source_classes + float(strength) * shift, dim=-1
    )
    transported = transported.clone()
    transported[list(anchor_class_ids)] = target_anchors
    return transported


def procrustes_transport_class_prototypes(
    source_classes: torch.Tensor,
    source_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    anchor_class_ids: tuple[int, ...],
    strength: float,
    regularization: float,
) -> torch.Tensor:
    """기본 동작 대응으로 identity-prior 직교 좌표 변환을 추정한다."""
    if source_anchors.shape != target_anchors.shape:
        raise ValueError("source and target anchors must have equal shapes")
    if source_classes.shape[-1] != source_anchors.shape[-1]:
        raise ValueError("class and anchor dimensions must match")
    source_classes = F.normalize(source_classes, dim=-1)
    source_anchors = F.normalize(source_anchors, dim=-1)
    target_anchors = F.normalize(target_anchors, dim=-1)
    source_center = source_anchors.mean(0, keepdim=True)
    target_center = target_anchors.mean(0, keepdim=True)
    source_centered = source_anchors - source_center
    target_centered = target_anchors - target_center
    identity = torch.eye(
        source_classes.shape[-1], dtype=source_classes.dtype,
        device=source_classes.device,
    )
    covariance = (
        source_centered.transpose(0, 1) @ target_centered
        + float(regularization) * identity
    )
    left, _, right = torch.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right
    mapped = (source_classes - source_center) @ rotation + target_center
    transported = F.normalize(
        source_classes + float(strength) * (mapped - source_classes), dim=-1
    )
    transported = transported.clone()
    transported[list(anchor_class_ids)] = target_anchors
    return transported


def anchor_geometry_error(
    source_anchors: torch.Tensor, target_anchors: torch.Tensor,
) -> torch.Tensor:
    """회전과 크기에 무관한 기본자세 간 cosine geometry 차이를 계산한다."""
    source = F.normalize(source_anchors, dim=-1)
    target = F.normalize(target_anchors, dim=-1)
    source_geometry = source @ source.transpose(0, 1)
    target_geometry = target @ target.transpose(0, 1)
    return (source_geometry - target_geometry).square().mean()


def ensemble_transport_logits(
    query_embedding: torch.Tensor,
    transported: list[torch.Tensor],
    geometry_errors: torch.Tensor,
    prototype_temperature: float,
    site_temperature: float,
) -> torch.Tensor:
    """기본자세 geometry가 가까운 source site의 이동 prototype을 더 신뢰한다."""
    if not transported:
        raise ValueError("at least one transported prototype set is required")
    query = F.normalize(query_embedding, dim=-1)
    site_weight = torch.softmax(
        -geometry_errors / max(float(site_temperature), 1e-4), dim=0
    )
    logits = torch.stack([
        query @ F.normalize(prototypes, dim=-1).transpose(0, 1)
        / max(float(prototype_temperature), 1e-4)
        for prototypes in transported
    ])
    return (logits * site_weight[:, None, None]).sum(0)


def transported_logits(
    target: dict[str, torch.Tensor],
    source_library: list[dict[str, torch.Tensor]],
    strength: float,
    anchor_temperature: float,
    prototype_temperature: float,
    site_temperature: float,
) -> torch.Tensor:
    """여러 source site에서 target anchor로 옮긴 행동 logit을 결합한다."""
    transported = []
    errors = []
    for source in source_library:
        transported.append(transport_class_prototypes(
            source["classes"], source["anchors"], target["anchors"],
            ANCHOR_CLASSES, strength, anchor_temperature,
        ))
        errors.append(anchor_geometry_error(
            source["anchors"], target["anchors"]
        ))
    return ensemble_transport_logits(
        target["embedding"], transported, torch.stack(errors),
        prototype_temperature, site_temperature,
    )


def procrustes_transported_logits(
    target: dict[str, torch.Tensor],
    source_library: list[dict[str, torch.Tensor]],
    strength: float,
    regularization: float,
    prototype_temperature: float,
    site_temperature: float,
) -> torch.Tensor:
    """현장 anchor로 직교 정렬한 여러 source 행동 prototype을 결합한다."""
    transported = []
    errors = []
    for source in source_library:
        transported.append(procrustes_transport_class_prototypes(
            source["classes"], source["anchors"], target["anchors"],
            ANCHOR_CLASSES, strength, regularization,
        ))
        errors.append(anchor_geometry_error(
            source["anchors"], target["anchors"]
        ))
    return ensemble_transport_logits(
        target["embedding"], transported, torch.stack(errors),
        prototype_temperature, site_temperature,
    )


def cal17_action(
    target: dict[str, torch.Tensor],
    source_library: list[dict[str, torch.Tensor]],
    config: dict,
) -> torch.Tensor:
    """고정 설정으로 base와 transported action log-probability를 결합한다."""
    transport = transported_logits(
        target, source_library,
        config["strength"], config["anchor_temperature"],
        config["prototype_temperature"], config["site_temperature"],
    )
    return (
        (1.0 - config["mixture"]) * target["action"].log_softmax(-1)
        + config["mixture"] * transport.log_softmax(-1)
    )


def cal17_risk(
    model,
    target: dict[str, torch.Tensor],
    action: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """고정 설정으로 safe evidence와 action-derived risk를 결합한다."""
    direct = target["direct_risk"].clone()
    similarity = F.normalize(target["embedding"], dim=-1) @ target[
        "anchors"
    ].transpose(0, 1)
    top2 = similarity.topk(2, dim=-1).values
    direct[:, 0] += config["safe_weight"] * (
        top2[:, 0] - top2[:, 1]
    ).clamp_min(0.0)
    direct[:, 2] += config["danger_bias"]
    return (
        (1.0 - config["fusion"]) * direct
        + config["fusion"] * model.action_to_risk(action)
    )
