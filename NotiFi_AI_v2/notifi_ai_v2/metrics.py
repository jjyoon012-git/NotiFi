"""Classification and dense motion metrics without external dependencies."""

from __future__ import annotations

import torch


def macro_f1(predicted: torch.Tensor, target: torch.Tensor, classes: int) -> float:
    values = []
    for class_id in range(classes):
        tp = ((predicted == class_id) & (target == class_id)).sum().float()
        fp = ((predicted == class_id) & (target != class_id)).sum().float()
        fn = ((predicted != class_id) & (target == class_id)).sum().float()
        values.append(2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0))
    return float(torch.stack(values).mean())


def classification_metrics(
    action_logits: torch.Tensor,
    risk_logits: torch.Tensor,
    action: torch.Tensor,
    risk: torch.Tensor,
) -> dict[str, float | int]:
    action_prediction = action_logits.argmax(-1)
    risk_prediction = risk_logits.argmax(-1)
    danger = risk == 2
    safe = risk == 0
    return {
        "action_accuracy": float((action_prediction == action).float().mean()),
        "action_macro_f1": macro_f1(action_prediction, action, 17),
        "risk_accuracy": float((risk_prediction == risk).float().mean()),
        "risk_macro_f1": macro_f1(risk_prediction, risk, 3),
        "danger_recall": float((risk_prediction[danger] == 2).float().mean()),
        "danger_correct": int((risk_prediction[danger] == 2).sum()),
        "danger_total": int(danger.sum()),
        "danger_action_accuracy": float(
            (action_prediction[danger] == action[danger]).float().mean()
        ),
        "safe_to_danger": int((safe & (risk_prediction == 2)).sum()),
        "safe_total": int(safe.sum()),
        "safe_to_danger_rate": float(
            (safe & (risk_prediction == 2)).sum() / safe.sum().clamp_min(1)
        ),
        "trials": int(len(action)),
    }


def selection_score(metrics: dict[str, float | int]) -> float:
    """Choose epochs using inner source sites only and penalize false alarms."""

    return float(
        0.65 * metrics["action_macro_f1"]
        + 0.20 * metrics["risk_macro_f1"]
        + 0.15 * metrics["danger_recall"]
        - 0.35 * metrics["safe_to_danger_rate"]
    )
