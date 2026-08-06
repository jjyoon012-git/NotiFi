"""Geometry-aware temporal phase objectives for the KP3 model family."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_phase_features(latent: torch.Tensor,
                            valid: torch.Tensor) -> torch.Tensor:
    """Represent pose state together with one- and two-token motion."""
    if latent.ndim != 3 or valid.shape != latent.shape[:2]:
        raise ValueError("latent must be [B,T,D] and valid must be [B,T]")
    weight = valid[..., None].to(latent.dtype)
    center = (latent * weight).sum(1, keepdim=True) / weight.sum(
        1, keepdim=True
    ).clamp_min(1.0)
    centered = (latent - center) * weight
    delta1 = torch.zeros_like(latent)
    delta2 = torch.zeros_like(latent)
    pair1 = valid[:, 1:] & valid[:, :-1]
    pair2 = valid[:, 2:] & valid[:, :-2]
    delta1[:, 1:] = (
        latent[:, 1:] - latent[:, :-1]
    ) * pair1[..., None].to(latent.dtype)
    delta2[:, 2:] = 0.5 * (
        latent[:, 2:] - latent[:, :-2]
    ) * pair2[..., None].to(latent.dtype)
    return torch.cat((centered, delta1, delta2), dim=-1) * weight


def _motion_query_mask(target: torch.Tensor, valid: torch.Tensor,
                       quantile: float, minimum: int) -> torch.Tensor:
    delta = torch.zeros_like(target)
    pair = valid[:, 1:] & valid[:, :-1]
    delta[:, 1:] = (
        target[:, 1:] - target[:, :-1]
    ) * pair[..., None].to(target.dtype)
    energy = torch.linalg.vector_norm(delta.float(), dim=-1)
    selected = torch.zeros_like(valid)
    for batch in range(len(target)):
        indices = torch.nonzero(valid[batch], as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        values = energy[batch, indices]
        threshold = torch.quantile(values, quantile)
        current = indices[values >= threshold]
        if len(current) < min(minimum, len(indices)):
            count = min(minimum, len(indices))
            current = indices[values.topk(count).indices]
        selected[batch, current] = True
    return selected


def temporal_phase_contrastive(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 0.08,
    positive_radius: int = 1,
    motion_quantile: float = 0.60,
    minimum_queries: int = 12,
    max_queries: int = 96,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match each CSI token to the correct GT motion phase within its trial.

    Positives are a small timestamp neighborhood. Other valid tokens from the
    same trial are negatives, so the encoder cannot solve the task by learning
    only subject, room, or action identity.
    """
    if predicted.shape != target.shape:
        raise ValueError("predicted and target latent shapes must match")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("valid shape must match latent time axes")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if positive_radius < 0:
        raise ValueError("positive_radius cannot be negative")
    if not 0.0 <= motion_quantile < 1.0:
        raise ValueError("motion_quantile must be in [0, 1)")

    target = target.detach()
    query_phase = temporal_phase_features(predicted, valid).float()
    target_phase = temporal_phase_features(target, valid).float()
    query_features = F.normalize(query_phase, dim=-1)
    key_features = F.normalize(target_phase, dim=-1)
    query_mask = _motion_query_mask(
        target, valid, motion_quantile, minimum_queries
    )
    losses = []
    correct = predicted.new_zeros((), dtype=torch.float32)
    query_count = 0
    for batch in range(len(predicted)):
        query_indices = torch.nonzero(
            query_mask[batch], as_tuple=False
        ).flatten()
        key_indices = torch.nonzero(valid[batch], as_tuple=False).flatten()
        if len(query_indices) == 0 or len(key_indices) < 2:
            continue
        if len(query_indices) > max_queries:
            energy = torch.linalg.vector_norm(
                target_phase[batch, query_indices, -target.shape[-1]:], dim=-1
            )
            query_indices = query_indices[energy.topk(max_queries).indices]
        logits = (
            query_features[batch, query_indices]
            @ key_features[batch, key_indices].T
        ) / temperature
        distance = (
            query_indices[:, None] - key_indices[None]
        ).abs()
        positive = distance <= positive_radius
        numerator = torch.logsumexp(
            logits.masked_fill(~positive, -torch.inf), dim=-1
        )
        denominator = torch.logsumexp(logits, dim=-1)
        losses.append(-(numerator - denominator))
        nearest = key_indices[logits.argmax(-1)]
        correct = correct + (
            (nearest - query_indices).abs() <= positive_radius
        ).float().sum()
        query_count += len(query_indices)

    if not losses:
        zero = predicted.sum() * 0.0
        return zero, {
            "phase_top1": zero.detach(),
            "phase_queries": zero.detach(),
        }
    loss = torch.cat(losses).mean()
    return loss, {
        "phase_top1": (correct / max(query_count, 1)).detach(),
        "phase_queries": predicted.new_tensor(float(query_count)).detach(),
    }
