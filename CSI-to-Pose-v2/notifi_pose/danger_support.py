"""통제된 낙상 support로 danger 세부 행동을 현장에 맞게 보정한다."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DANGER_SUPPORT_CLASSES = (12, 13, 14, 15, 16)


def class_prototypes(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    classes: tuple[int, ...] = DANGER_SUPPORT_CLASSES,
) -> torch.Tensor:
    """낙상 support embedding을 class 순서가 고정된 단위 prototype으로 만든다."""
    if embedding.ndim != 2 or labels.ndim != 1:
        raise ValueError("danger embeddings and labels must have shapes [B,D] and [B]")
    if len(embedding) != len(labels):
        raise ValueError("danger embeddings and labels must have equal batches")
    prototypes = []
    for class_id in classes:
        keep = labels == class_id
        if not bool(keep.any()):
            raise ValueError(f"danger support is missing class {class_id}")
        prototypes.append(F.normalize(embedding[keep].mean(0), dim=0))
    return torch.stack(prototypes)


def support_evidence(
    query_embedding: torch.Tensor,
    safe_anchors: torch.Tensor,
    danger_prototypes: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """query의 낙상 세부 분포와 낙상-기본동작 유사도 차이를 계산한다."""
    if query_embedding.ndim != 2:
        raise ValueError("query embedding must have shape [B,D]")
    if safe_anchors.ndim != 2 or danger_prototypes.ndim != 2:
        raise ValueError("support prototypes must have shape [C,D]")
    if not (
        query_embedding.shape[-1]
        == safe_anchors.shape[-1]
        == danger_prototypes.shape[-1]
    ):
        raise ValueError("query and support prototype dimensions must match")
    temperature = max(float(temperature), 1e-4)
    query = F.normalize(query_embedding, dim=-1)
    danger_similarity = query @ F.normalize(
        danger_prototypes, dim=-1
    ).transpose(0, 1)
    safe_similarity = query @ F.normalize(
        safe_anchors, dim=-1
    ).transpose(0, 1)
    danger_log_probability = (
        danger_similarity / temperature
    ).log_softmax(-1)
    danger_affinity = temperature * (
        torch.logsumexp(danger_similarity / temperature, dim=-1)
        - math.log(danger_similarity.shape[-1])
    )
    safe_affinity = temperature * (
        torch.logsumexp(safe_similarity / temperature, dim=-1)
        - math.log(safe_similarity.shape[-1])
    )
    return danger_log_probability, danger_affinity - safe_affinity


def apply_danger_support(
    action_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    evidence: list[tuple[torch.Tensor, torch.Tensor]],
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """낙상 support를 danger 조건부 분포와 선택적 risk logit에만 결합한다."""
    if not evidence:
        raise ValueError("at least one danger-support evidence member is required")
    if action_logits.ndim != 2 or action_logits.shape[-1] < 17:
        raise ValueError("action logits must include the five danger classes")
    if risk_logits.ndim != 2 or risk_logits.shape[-1] != 3:
        raise ValueError("risk logits must have shape [B,3]")
    subtype = torch.logsumexp(
        torch.stack([item[0] for item in evidence]), dim=0
    ) - math.log(len(evidence))
    margin = torch.stack([item[1] for item in evidence]).mean(0)
    weight = float(config.get("subtype_weight", 0.0))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("danger subtype weight must be in [0,1]")

    action = action_logits.clone()
    base_probability = action[:, 12:17].log_softmax(-1)
    if weight <= 0.0:
        conditional = base_probability
    elif weight >= 1.0:
        conditional = subtype
    else:
        conditional = torch.logaddexp(
            base_probability + math.log1p(-weight),
            subtype + math.log(weight),
        )
    danger_mass = torch.logsumexp(action[:, 12:17], dim=-1, keepdim=True)
    group_shift = (
        float(config.get("action_margin_gain", 0.0)) * margin
        + float(config.get("action_bias", 0.0))
    )
    action[:, 12:17] = (
        danger_mass + conditional + group_shift[:, None]
    )

    risk = risk_logits.clone()
    risk_shift = (
        float(config.get("risk_margin_gain", 0.0)) * margin
        + float(config.get("risk_bias", 0.0))
    )
    risk[:, 2] += risk_shift
    return action, risk, {
        "danger_support_probability": subtype.exp(),
        "danger_support_margin": margin,
        "danger_support_action_shift": group_shift,
        "danger_support_risk_shift": risk_shift,
    }


__all__ = (
    "DANGER_SUPPORT_CLASSES", "apply_danger_support", "class_prototypes",
    "support_evidence",
)
