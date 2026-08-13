"""라벨이 있는 calibration support로 CSI 행동 공간을 source 공간에 정렬한다."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def geometry_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """두 support 집합의 클래스 간 cosine geometry 차이를 계산한다."""
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return (
        left @ left.transpose(0, 1) - right @ right.transpose(0, 1)
    ).square().mean()


def action_to_risk_log_probability(action: torch.Tensor) -> torch.Tensor:
    """17개 행동 logit을 safe, warning, danger의 3개 log 확률로 합친다."""
    if action.ndim != 2 or action.shape[-1] != 17:
        raise ValueError("action logits must have shape [B,17]")
    action = action.log_softmax(-1)
    return torch.stack((
        torch.logsumexp(action[:, :9], dim=-1),
        torch.logsumexp(action[:, 9:12], dim=-1),
        torch.logsumexp(action[:, 12:17], dim=-1),
    ), dim=-1)


def identity_ridge_map(
    source: torch.Tensor,
    target: torch.Tensor,
    regularization: float,
    normalize_inputs: bool = True,
) -> torch.Tensor:
    """적은 support에서도 폭주하지 않도록 identity prior가 있는 affine map을 맞춘다."""
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError("source and target must have the same [N,D] shape")
    if len(source) < 2:
        raise ValueError("at least two support pairs are required")
    regularization = float(regularization)
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if normalize_inputs:
        source = F.normalize(source, dim=-1)
        target = F.normalize(target, dim=-1)
    dimension = source.shape[-1]
    ones = source.new_ones((len(source), 1))
    design = torch.cat((source, ones), dim=-1)
    prior = source.new_zeros((dimension + 1, dimension))
    prior[:dimension] = torch.eye(
        dimension, dtype=source.dtype, device=source.device
    )
    system = design.transpose(0, 1) @ design
    system = system + regularization * torch.eye(
        dimension + 1, dtype=source.dtype, device=source.device
    )
    target_term = design.transpose(0, 1) @ target + regularization * prior
    return torch.linalg.solve(system, target_term)


def apply_affine_map(
    values: torch.Tensor,
    mapping: torch.Tensor,
    normalize_output: bool = True,
) -> torch.Tensor:
    """마지막 열을 bias로 쓰는 affine map을 적용하고 cosine 공간으로 정규화한다."""
    if values.ndim != 2 or mapping.ndim != 2:
        raise ValueError("values and mapping must be matrices")
    if mapping.shape != (values.shape[-1] + 1, values.shape[-1]):
        raise ValueError("mapping has an incompatible affine shape")
    augmented = torch.cat((values, values.new_ones((len(values), 1))), dim=-1)
    output = augmented @ mapping
    return F.normalize(output, dim=-1) if normalize_output else output


def aligned_logits(
    query: torch.Tensor,
    target_anchors: torch.Tensor,
    target_danger: torch.Tensor,
    source_library: list[dict[str, torch.Tensor]],
    regularization: float,
    prototype_temperature: float,
    site_temperature: float,
    direction: str = "source_to_target",
    target_warning: torch.Tensor | None = None,
) -> torch.Tensor:
    """기본·낙상 support로 전체 17개 source prototype을 target 좌표에 전달한다."""
    if not source_library:
        raise ValueError("source library cannot be empty")
    if target_anchors.ndim != 2 or target_danger.ndim != 2:
        raise ValueError("target supports must be [classes, dimensions]")
    errors = torch.stack(
        [geometry_error(item["anchors"], target_anchors) for item in source_library]
    )
    weights = torch.softmax(
        -errors / max(float(site_temperature), 1e-4), dim=0
    )
    source_classes = torch.stack(
        [F.normalize(item["classes"], dim=-1) for item in source_library]
    )
    source_anchors = torch.stack(
        [F.normalize(item["anchors"], dim=-1) for item in source_library]
    )
    reference_classes = (source_classes * weights[:, None, None]).sum(0)
    reference_anchors = (source_anchors * weights[:, None, None]).sum(0)
    reference_danger = reference_classes[12:17]
    source_parts = [reference_anchors]
    target_parts = [target_anchors]
    if target_warning is not None:
        if target_warning.shape != (3, query.shape[-1]):
            raise ValueError("warning support must have shape [3,D]")
        source_parts.append(reference_classes[9:12])
        target_parts.append(target_warning)
    source_parts.append(reference_danger)
    target_parts.append(target_danger)
    source_pairs = torch.cat(source_parts, dim=0)
    target_pairs = torch.cat(target_parts, dim=0)

    if direction == "source_to_target":
        mapping = identity_ridge_map(source_pairs, target_pairs, regularization)
        prototypes = apply_affine_map(reference_classes, mapping)
        encoded_query = F.normalize(query, dim=-1)
    elif direction == "target_to_source":
        mapping = identity_ridge_map(target_pairs, source_pairs, regularization)
        prototypes = F.normalize(reference_classes, dim=-1)
        encoded_query = apply_affine_map(query, mapping)
    else:
        raise ValueError(f"unsupported alignment direction: {direction}")
    return (
        encoded_query @ prototypes.transpose(0, 1)
        / max(float(prototype_temperature), 1e-4)
    )


__all__ = (
    "action_to_risk_log_probability",
    "aligned_logits",
    "apply_affine_map",
    "geometry_error",
    "identity_ridge_map",
)
