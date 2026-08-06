"""Fail-closed input boundary for the deployed KP10 prediction path."""

from __future__ import annotations

from collections.abc import Mapping


REQUIRED_FIELDS = (
    "checkpoint",
    "baseline",
    "baseline_bank",
    "train_bank",
    "fused_action",
    "risk_probability",
    "pool",
    "logits",
    "scalar_distance",
    "part_distance",
    "predicted_scalar_profile",
    "predicted_part_profile",
    "inference_valid",
)
OPTIONAL_FIELDS = (
    "predicted_proximity",
    "include_selector_distance",
    "selector_embedding",
)
POOL_FIELDS = (
    "indices",
    "retrieval_score",
    "action_log_probability",
)


def inference_view(data: Mapping) -> dict:
    """Return only CSI-derived state and train-only retrieval artifacts.

    Evaluation targets and the GT-derived candidate ``target_cost`` never
    cross this boundary. Missing CSI masks fail immediately instead of
    falling back to a target-validity mask.
    """
    missing = [name for name in REQUIRED_FIELDS if name not in data]
    if missing:
        raise KeyError(f"missing KP10 inference fields: {missing}")
    pool = data["pool"]
    missing_pool = [name for name in POOL_FIELDS if name not in pool]
    if missing_pool:
        raise KeyError(f"missing KP10 inference pool fields: {missing_pool}")
    result = {
        name: data[name] for name in REQUIRED_FIELDS if name != "pool"
    }
    result["pool"] = {name: pool[name] for name in POOL_FIELDS}
    for name in OPTIONAL_FIELDS:
        if name in data:
            result[name] = data[name]
    return result
