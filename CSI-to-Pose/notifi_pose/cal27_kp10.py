"""Deployable target-local action calibration for CAL27-KP10.

CAL27 adapts only the action evidence used by the KP10 pose retriever.  Its
calibration prompt contains safe actions, so the risk output is deliberately
exposed as experimental and cannot be promoted by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .cal23_kp10 import DynamicMotionClassifier
from .calibration_quality import (
    SAFE_CALIBRATION_CLASSES,
    CalibrationRejectedError,
)


def validate_runtime_csi(
    csi: torch.Tensor,
    link_mask: torch.Tensor,
    minimum_link_coverage: float = 0.50,
) -> None:
    """Reject malformed trials before they can produce a plausible pose."""
    if csi.ndim != 5 or link_mask.shape != csi.shape[:3]:
        raise CalibrationRejectedError(
            "runtime CSI must have [B,T,L,F,2] data and [B,T,L] mask"
        )
    if not csi.shape[0] or not csi.shape[1] or csi.shape[2] != C.N_LINKS:
        raise CalibrationRejectedError(
            f"runtime CSI requires {C.N_LINKS} physical links and non-empty trials"
        )
    if csi.shape[-1] != 2:
        raise CalibrationRejectedError("runtime CSI must store real/imaginary pairs")
    mask = link_mask.bool()
    coverage = mask.float().mean(1)
    if bool((coverage < float(minimum_link_coverage)).any()):
        raise CalibrationRejectedError(
            "one or more physical links have less than "
            f"{minimum_link_coverage:.0%} valid frame coverage"
        )
    finite_packet = torch.isfinite(csi).all(dim=(-1, -2))
    if not bool((finite_packet | ~mask).all()):
        raise CalibrationRejectedError("valid runtime CSI contains NaN or infinity")


def class_prototypes(embedding: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
    """Compute one source embedding prototype for every action class."""
    return torch.stack([
        embedding[labels == class_id].mean(0)
        for class_id in range(C.N_CLASSES)
    ])


def local_safe_prototype_logits(
    embedding: torch.Tensor,
    support_embedding: torch.Tensor,
    support_labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_safe_mean: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Score target embeddings against local safe and shifted source prototypes."""
    target_mean = support_embedding.mean(0)
    prototypes = source_prototypes + (target_mean - source_safe_mean)
    prototypes = prototypes.clone()
    for class_id in SAFE_CALIBRATION_CLASSES:
        class_support = support_embedding[support_labels == class_id]
        if not len(class_support):
            raise CalibrationRejectedError(
                f"calibration support is missing safe class {class_id}"
            )
        prototypes[class_id] = class_support.mean(0)
    return (
        F.normalize(embedding, dim=-1)
        @ F.normalize(prototypes, dim=-1).T
    ) / float(temperature)


def fit_local_prototype(
    support_embedding: torch.Tensor,
    direct_logits: torch.Tensor,
    labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_safe_mean: torch.Tensor,
) -> dict:
    """Select prototype strength using support-repeat cross-validation only."""
    counts = {
        class_id: int((labels == class_id).sum())
        for class_id in SAFE_CALIBRATION_CLASSES
    }
    missing = [class_id for class_id, count in counts.items() if count < 2]
    if missing:
        raise CalibrationRejectedError(
            f"at least two repeats are required for safe classes {missing}"
        )
    candidates = []
    for temperature in (0.05, 0.10, 0.20, 0.50):
        for weight in (0.5, 1.0, 2.0, 4.0):
            folds = []
            groups = [
                torch.nonzero(labels == class_id, as_tuple=False).flatten()
                for class_id in SAFE_CALIBRATION_CLASSES
            ]
            for parity in (0, 1):
                fit, check = [], []
                for group in groups:
                    for index, position in enumerate(group.tolist()):
                        (check if index % 2 == parity else fit).append(position)
                fit_index = torch.as_tensor(fit, device=labels.device)
                check_index = torch.as_tensor(check, device=labels.device)
                evidence = local_safe_prototype_logits(
                    support_embedding.index_select(0, check_index),
                    support_embedding.index_select(0, fit_index),
                    labels.index_select(0, fit_index),
                    source_prototypes,
                    source_safe_mean,
                    temperature,
                )
                logits = direct_logits.index_select(0, check_index) + weight * evidence
                target = labels.index_select(0, check_index)
                folds.append({
                    "accuracy": float((logits.argmax(-1) == target).float().mean()),
                    "cross_entropy": float(F.cross_entropy(logits, target)),
                })
            candidates.append({
                "temperature": temperature,
                "weight": weight,
                "accuracy": float(np.mean([value["accuracy"] for value in folds])),
                "cross_entropy": float(np.mean([
                    value["cross_entropy"] for value in folds
                ])),
            })
    selected = max(candidates, key=lambda value: (
        value["accuracy"], -value["cross_entropy"], -value["weight"]
    ))
    return {"selected": selected, "candidates": candidates}


def apply_local_prototype(
    embedding: torch.Tensor,
    direct_logits: torch.Tensor,
    support_embedding: torch.Tensor,
    support_labels: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_safe_mean: torch.Tensor,
    calibration: Mapping,
) -> torch.Tensor:
    selected = calibration["selected"]
    evidence = local_safe_prototype_logits(
        embedding,
        support_embedding,
        support_labels,
        source_prototypes,
        source_safe_mean,
        selected["temperature"],
    )
    return direct_logits + selected["weight"] * evidence


def action_risk_consistency(
    action_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    class_risk = action_logits.new_tensor(
        [0] * 9 + [1] * 3 + [2] * 5, dtype=torch.long
    )
    evidence = F.log_softmax(risk_logits, dim=-1).index_select(1, class_risk)
    return action_logits + float(weight) * evidence


class Cal27ActionCalibrator(nn.Module):
    """Inference wrapper for a support-validated CAL27 action calibration."""

    def __init__(self, artifact: Mapping, *, allow_experimental: bool = False):
        super().__init__()
        quality = artifact.get("calibration_quality", {})
        ready = artifact.get(
            "action_pose_deployable", quality.get("action_pose_ready", False)
        )
        experimental = artifact.get(
            "experimental_action_pose", quality.get(
                "experimental_action_pose_candidate", False
            )
        )
        if not ready and not (allow_experimental and experimental):
            raise CalibrationRejectedError(
                "CAL27 artifact did not pass the action/pose support gate"
            )
        self.action_pose_ready = bool(ready)
        self.experimental_action_pose_candidate = bool(experimental)
        self.dynamic_model = DynamicMotionClassifier(
            **artifact.get("dynamic_model_config", {})
        )
        self.dynamic_model.load_state_dict(artifact["dynamic_model_state_dict"])
        self.model_feature_mode = self.dynamic_model.feature_mode
        self.support_rows = tuple(artifact.get("support_rows", ()))
        self.hierarchy_weight = float(artifact["hierarchy_weight"])
        self.prototype_calibration = artifact["prototype_calibration"]
        for name in (
            "source_prototypes", "source_safe_mean", "support_embedding",
            "support_labels",
        ):
            self.register_buffer(name, torch.as_tensor(artifact[name]))

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu", *,
             allow_experimental: bool = False) -> "Cal27ActionCalibrator":
        artifact = torch.load(path, map_location=map_location, weights_only=False)
        return cls(artifact, allow_experimental=allow_experimental)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        validate_runtime_csi(csi, link_mask)
        output = self.dynamic_model(csi, link_mask)
        direct = action_risk_consistency(
            output["action_logits"], output["risk_logits"], self.hierarchy_weight
        )
        action_logits = apply_local_prototype(
            output["embedding"], direct,
            self.support_embedding, self.support_labels,
            self.source_prototypes, self.source_safe_mean,
            self.prototype_calibration,
        )
        return {
            **output,
            "direct_action_logits": direct,
            "action_logits": action_logits,
            "risk_logits_experimental": output["risk_logits"],
            "risk_certified": False,
            "accepted_for_action_pose_inference": self.action_pose_ready,
            "experimental_action_pose_candidate": (
                self.experimental_action_pose_candidate
            ),
            "accepted_for_normal_inference": False,
        }
