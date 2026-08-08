"""CAL27: 기본동작 대응만으로 query를 source latent 공간에 비선형 정렬한다."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cal17 import ANCHOR_CLASSES, anchor_geometry_error


def kernel_transport_query(
    query: torch.Tensor,
    target_anchors: torch.Tensor,
    source_anchors: torch.Tensor,
    strength: float,
    kernel_temperature: float,
    regularization: float,
) -> torch.Tensor:
    """RBF kernel ridge로 target anchor의 이동장을 query까지 부드럽게 전달한다."""
    if target_anchors.shape != source_anchors.shape:
        raise ValueError("target and source anchors must have identical shapes")
    if query.shape[-1] != target_anchors.shape[-1]:
        raise ValueError("query and anchor dimensions must match")
    target = F.normalize(target_anchors, dim=-1)
    source = F.normalize(source_anchors, dim=-1)
    query = F.normalize(query, dim=-1)
    temperature = max(float(kernel_temperature), 1e-4)
    kernel = torch.exp((target @ target.transpose(0, 1) - 1.0) / temperature)
    identity = torch.eye(
        len(target), dtype=target.dtype, device=target.device,
    )
    coefficient = torch.linalg.solve(
        kernel + float(regularization) * identity,
        source - target,
    )
    query_kernel = torch.exp((query @ target.transpose(0, 1) - 1.0) / temperature)
    mapped = query + float(strength) * (query_kernel @ coefficient)
    return F.normalize(mapped, dim=-1)


def kernel_transported_logits(
    target: dict[str, torch.Tensor],
    source_library: list[dict[str, torch.Tensor]],
    strength: float,
    kernel_temperature: float,
    regularization: float,
    prototype_temperature: float,
    site_temperature: float,
) -> torch.Tensor:
    """각 source 공간으로 옮긴 query logit을 anchor geometry 신뢰도로 결합한다."""
    if not source_library:
        raise ValueError("at least one source site is required")
    logits = []
    errors = []
    for source in source_library:
        mapped = kernel_transport_query(
            target["embedding"], target["anchors"], source["anchors"],
            strength, kernel_temperature, regularization,
        )
        classes = F.normalize(source["classes"], dim=-1)
        logits.append(
            mapped @ classes.transpose(0, 1)
            / max(float(prototype_temperature), 1e-4)
        )
        errors.append(anchor_geometry_error(source["anchors"], target["anchors"]))
    weight = torch.softmax(
        -torch.stack(errors) / max(float(site_temperature), 1e-4), dim=0,
    )
    return (torch.stack(logits) * weight[:, None, None]).sum(0)


def anchor_reconstruction_error(
    target_anchors: torch.Tensor,
    source_anchors: torch.Tensor,
    strength: float,
    kernel_temperature: float,
    regularization: float,
) -> torch.Tensor:
    """query 정답 없이 알려진 9개 anchor 대응의 transport 오차를 계산한다."""
    mapped = kernel_transport_query(
        target_anchors, target_anchors, source_anchors,
        strength, kernel_temperature, regularization,
    )
    return (mapped - F.normalize(source_anchors, dim=-1)).square().mean()


def cal27_action(
    target: dict[str, torch.Tensor],
    source_library: list[dict[str, torch.Tensor]],
    config: dict,
) -> torch.Tensor:
    """고정 설정으로 base와 query-to-source kernel transport 확률을 결합한다."""
    transport = kernel_transported_logits(
        target, source_library,
        config["strength"], config["kernel_temperature"],
        config["regularization"], config["prototype_temperature"],
        config["site_temperature"],
    )
    mixture = float(config["mixture"])
    return (
        (1.0 - mixture) * target["action"].log_softmax(-1)
        + mixture * transport.log_softmax(-1)
    )


__all__ = (
    "ANCHOR_CLASSES", "anchor_reconstruction_error", "cal27_action",
    "kernel_transport_query", "kernel_transported_logits",
)
