"""Source-only pose teacher used to ground CSI features in action semantics."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import contract as C
from .cal13 import MOTION_DESCRIPTOR_DIM, temporal_motion_signature
from .cal14 import CosineClassifier


class PoseSemanticTeacher(nn.Module):
    """Encode scale-normalized GVHMR motion into action and risk semantics."""

    def __init__(
        self,
        hidden: int = 64,
        bins: int = 8,
        dropout: float = 0.15,
        cosine_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.bins = int(bins)
        dimension = MOTION_DESCRIPTOR_DIM * (bins + 3)
        self.projection = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.action = CosineClassifier(hidden, C.N_CLASSES, cosine_scale)
        self.risk = CosineClassifier(hidden, C.N_RISK, cosine_scale)

    def forward(
        self,
        descriptor: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return a trial embedding and source-only classification logits."""
        signature = temporal_motion_signature(
            descriptor, valid, bins=self.bins
        )
        embedding = self.projection(signature)
        return {
            "embedding": embedding,
            "action_logits": self.action(embedding),
            "risk_logits": self.risk(embedding),
        }


def cross_modal_supervised_contrastive(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Align CSI and pose trials by action without requiring paired timestamps."""
    if student.shape != teacher.shape:
        raise ValueError("student and teacher embeddings must have equal shape")
    if labels.shape != student.shape[:1]:
        raise ValueError("labels must match the embedding batch")
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    logits = student @ teacher.transpose(0, 1) / float(temperature)
    positive = labels[:, None].eq(labels[None]).to(logits.dtype)
    positive /= positive.sum(-1, keepdim=True).clamp_min(1.0)
    forward = -(positive * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    backward = -(
        positive.transpose(0, 1)
        * F.log_softmax(logits.transpose(0, 1), dim=-1)
    ).sum(-1).mean()
    return 0.5 * (forward + backward)


def paired_pose_distillation_loss(
    student_embedding: torch.Tensor,
    student_action: torch.Tensor,
    student_risk: torch.Tensor,
    teacher_output: dict[str, torch.Tensor],
) -> torch.Tensor:
    """같은 source trial의 GT motion 의미만 CSI embedding에 직접 증류한다."""
    teacher_embedding = teacher_output["embedding"].detach()
    if student_embedding.shape != teacher_embedding.shape:
        raise ValueError("paired student and teacher embeddings must match")
    embedding = 1.0 - F.cosine_similarity(
        student_embedding, teacher_embedding, dim=-1
    ).mean()
    action = F.kl_div(
        student_action.log_softmax(-1),
        teacher_output["action_logits"].detach().softmax(-1),
        reduction="batchmean",
    )
    risk = F.kl_div(
        student_risk.log_softmax(-1),
        teacher_output["risk_logits"].detach().softmax(-1),
        reduction="batchmean",
    )
    return embedding + 0.50 * action + 0.30 * risk
