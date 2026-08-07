"""Support-conditioned dynamic and hierarchical calibration for CAL44-KP10.

CAL44 keeps the frozen CAL42 encoders.  It uses labelled, non-fall calibration
actions to replace local safe/warning prototypes and to select the strength of
hierarchical action-to-risk evidence by support-repeat cross-validation only.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import contract as C
from .calibration_quality import CalibrationRejectedError, SAFE_CALIBRATION_CLASSES


WARNING_CALIBRATION_CLASSES = (9, 10, 11)
DYNAMIC_CALIBRATION_CLASSES = SAFE_CALIBRATION_CLASSES + WARNING_CALIBRATION_CLASSES


def class_prototypes(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        embedding[labels == class_id].mean(0)
        for class_id in range(C.N_CLASSES)
    ])


def dynamic_prototype_logits(
    embedding: torch.Tensor,
    support_embedding: torch.Tensor,
    support_labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    classes: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    """Transport source prototypes and replace observed target classes."""
    deltas = []
    prototypes = source_prototypes.clone()
    for class_id in classes:
        local = support_embedding[support_labels == class_id]
        if not len(local):
            raise CalibrationRejectedError(
                f"calibration support is missing dynamic class {class_id}"
            )
        local_mean = local.mean(0)
        deltas.append(local_mean - source_prototypes[class_id])
        prototypes[class_id] = local_mean
    shift = torch.stack(deltas).mean(0)
    observed = set(int(value) for value in classes)
    for class_id in range(C.N_CLASSES):
        if class_id not in observed:
            prototypes[class_id] = prototypes[class_id] + shift
    return (
        F.normalize(embedding, dim=-1)
        @ F.normalize(prototypes, dim=-1).T
    ) / float(temperature)


def fit_dynamic_prototypes(
    support_embedding: torch.Tensor,
    direct_logits: torch.Tensor,
    labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    classes: Sequence[int] = DYNAMIC_CALIBRATION_CLASSES,
) -> dict:
    """Select local evidence using repeat CV without target query labels."""
    groups = []
    for class_id in classes:
        group = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if len(group) < 2:
            raise CalibrationRejectedError(
                f"at least two repeats are required for dynamic class {class_id}"
            )
        groups.append(group)
    candidates = []
    for temperature in (0.05, 0.10, 0.20, 0.50):
        for weight in (0.25, 0.5, 1.0, 2.0):
            scores = []
            losses = []
            for parity in (0, 1):
                fit, check = [], []
                for group in groups:
                    for index, position in enumerate(group.tolist()):
                        (check if index % 2 == parity else fit).append(position)
                fit_index = torch.as_tensor(fit, device=labels.device)
                check_index = torch.as_tensor(check, device=labels.device)
                evidence = dynamic_prototype_logits(
                    support_embedding.index_select(0, check_index),
                    support_embedding.index_select(0, fit_index),
                    labels.index_select(0, fit_index),
                    source_prototypes, classes, temperature,
                )
                logits = direct_logits.index_select(0, check_index) + weight * evidence
                target = labels.index_select(0, check_index)
                scores.append(float((logits.argmax(-1) == target).float().mean()))
                losses.append(float(F.cross_entropy(logits, target)))
            candidates.append({
                "temperature": temperature,
                "weight": weight,
                "accuracy": float(np.mean(scores)),
                "cross_entropy": float(np.mean(losses)),
            })
    selected = max(candidates, key=lambda value: (
        value["accuracy"], -value["cross_entropy"], -value["weight"]
    ))
    return {"classes": tuple(int(value) for value in classes),
            "selected": selected, "candidates": candidates}


def apply_dynamic_prototypes(
    embedding: torch.Tensor,
    direct_logits: torch.Tensor,
    support_embedding: torch.Tensor,
    support_labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    calibration: Mapping,
) -> torch.Tensor:
    selected = calibration["selected"]
    evidence = dynamic_prototype_logits(
        embedding, support_embedding, support_labels, source_prototypes,
        calibration["classes"], selected["temperature"],
    )
    return direct_logits + selected["weight"] * evidence


def risk_logits_from_action(action_logits: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.logsumexp(action_logits[:, :9], dim=-1),
        torch.logsumexp(action_logits[:, 9:12], dim=-1),
        torch.logsumexp(action_logits[:, 12:], dim=-1),
    ], dim=-1)


def fit_hierarchical_risk(
    action_logits: torch.Tensor,
    direct_risk_logits: torch.Tensor,
    risk_labels: torch.Tensor,
) -> dict:
    """Choose soft action-to-risk routing from calibration support only."""
    grouped = risk_logits_from_action(action_logits)
    candidates = []
    for weight in (0.0, 0.10, 0.25, 0.50, 1.0):
        logits = (
            F.log_softmax(direct_risk_logits, dim=-1)
            + weight * F.log_softmax(grouped, dim=-1)
        )
        prediction = logits.argmax(-1)
        recalls = []
        for risk_id in torch.unique(risk_labels).tolist():
            mask = risk_labels == int(risk_id)
            recalls.append(float((prediction[mask] == int(risk_id)).float().mean()))
        candidates.append({
            "weight": weight,
            "balanced_accuracy": float(np.mean(recalls)),
            "cross_entropy": float(F.cross_entropy(logits, risk_labels)),
        })
    selected = max(candidates, key=lambda value: (
        value["balanced_accuracy"], -value["cross_entropy"], -value["weight"]
    ))
    return {"selected": selected, "candidates": candidates}


def apply_hierarchical_risk(
    action_logits: torch.Tensor,
    direct_risk_logits: torch.Tensor,
    calibration: Mapping,
) -> torch.Tensor:
    weight = float(calibration["selected"]["weight"])
    return (
        F.log_softmax(direct_risk_logits, dim=-1)
        + weight * F.log_softmax(risk_logits_from_action(action_logits), dim=-1)
    )


def preserve_control_danger(
    control_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    danger_start: int,
) -> torch.Tensor:
    """Keep every danger decision already made by the locked control branch."""
    if control_logits.shape != candidate_logits.shape:
        raise ValueError("control and candidate logits must have identical shape")
    preserve = control_logits.argmax(-1) >= int(danger_start)
    return torch.where(preserve.unsqueeze(-1), control_logits, candidate_logits)
