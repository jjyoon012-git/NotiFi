"""Support-conditioned episodic risk calibration for CAL33-KP10."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .calibration_quality import SAFE_CALIBRATION_CLASSES, CalibrationRejectedError


META_RISK_FEATURES = 22


def build_safe_context(embedding: torch.Tensor, risk_logits: torch.Tensor,
                       labels: torch.Tensor,
                       anchor_classes=SAFE_CALIBRATION_CLASSES) -> dict:
    prototypes = []
    for class_id in anchor_classes:
        values = embedding[labels == class_id]
        if not len(values):
            raise CalibrationRejectedError(
                f"calibration support is missing safe class {class_id}"
            )
        prototypes.append(values.mean(0))
    mean = embedding.mean(0)
    scale = embedding.std(0, unbiased=False).clamp_min(0.10)
    risk_mean = risk_logits.mean(0)
    risk_scale = risk_logits.std(0, unbiased=False).clamp_min(0.10)
    return {
        "prototypes": torch.stack(prototypes),
        "embedding_mean": mean,
        "embedding_scale": scale,
        "risk_mean": risk_mean,
        "risk_scale": risk_scale,
        "feature_dim": 2 * len(anchor_classes) + 6,
    }


def meta_risk_features(embedding: torch.Tensor, risk_logits: torch.Tensor,
                       context: Mapping) -> torch.Tensor:
    prototypes = context["prototypes"]
    cosine = (
        F.normalize(embedding.float(), dim=-1)
        @ F.normalize(prototypes.float(), dim=-1).T
    )
    difference = (
        embedding[:, None].float() - prototypes[None].float()
    ) / context["embedding_scale"][None, None]
    distance = torch.log1p(difference.square().mean(-1))
    normalized = (
        embedding.float() - context["embedding_mean"]
    ) / context["embedding_scale"]
    risk = (
        risk_logits.float() - context["risk_mean"]
    ) / context["risk_scale"]
    shape = torch.stack((
        normalized.abs().mean(-1),
        normalized.square().mean(-1),
        normalized.abs().amax(-1),
    ), dim=-1)
    output = torch.cat((cosine, distance, risk, shape), dim=-1)
    if output.shape[-1] != int(context["feature_dim"]):
        raise RuntimeError("unexpected CAL33 feature dimension")
    return output


class MetaRiskHead(nn.Module):
    def __init__(self, input_features: int = META_RISK_FEATURES,
                 width: int = 96, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def apply_risk_group_gate(action_logits: torch.Tensor,
                          risk_prediction: torch.Tensor) -> torch.Tensor:
    """Restrict action selection to the risk group predicted by CAL33."""
    output = action_logits.clone()
    ranges = ((0, 9), (9, 12), (12, 17))
    for risk_id, (start, stop) in enumerate(ranges):
        keep = risk_prediction == risk_id
        output[keep, :start] = -1e4
        output[keep, stop:] = -1e4
    return output
