"""모델·데이터셋 의존성 없이 action과 risk 분류 지표를 계산한다."""

from __future__ import annotations

import torch

from . import contract as C


def macro_f1(
    predicted: torch.Tensor, target: torch.Tensor, classes: int,
) -> float:
    """지정한 모든 class를 동일 가중치로 평균한 macro F1을 계산한다."""
    values = []
    for class_id in range(classes):
        true_positive = ((predicted == class_id) & (target == class_id)).sum().float()
        false_positive = ((predicted == class_id) & (target != class_id)).sum().float()
        false_negative = ((predicted != class_id) & (target == class_id)).sum().float()
        values.append(
            2.0 * true_positive
            / (2.0 * true_positive + false_positive + false_negative).clamp_min(1.0)
        )
    return float(torch.stack(values).mean())


def classification_metrics(
    action_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    action: torch.Tensor,
    risk: torch.Tensor,
) -> dict:
    """17-action, 3-risk, danger recall·세부형, safe 오경보를 함께 보고한다."""
    action_prediction = action_logits.argmax(-1)
    risk_prediction = risk_logits.argmax(-1)
    danger = risk == 2
    safe = risk == 0
    return {
        "action_accuracy": float((action_prediction == action).float().mean()),
        "action_macro_f1": macro_f1(action_prediction, action, C.N_CLASSES),
        "risk_accuracy": float((risk_prediction == risk).float().mean()),
        "risk_macro_f1": macro_f1(risk_prediction, risk, C.N_RISK),
        "danger_recall": float((risk_prediction[danger] == 2).float().mean()),
        "danger_correct": int((risk_prediction[danger] == 2).sum()),
        "danger_total": int(danger.sum()),
        "danger_action_accuracy": float(
            (action_prediction[danger] == action[danger]).float().mean()
        ),
        "safe_to_danger": int((safe & (risk_prediction == 2)).sum()),
        "safe_total": int(safe.sum()),
        "trials": len(action),
    }


__all__ = ("classification_metrics", "macro_f1")
