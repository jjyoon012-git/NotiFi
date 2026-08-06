"""Deployment readiness checks for site calibration.

The pose model must not silently extrapolate when the calibration protocol is
incomplete.  This module turns the support-set evidence into an explicit
READY/DEGRADED/REJECT decision without looking at any query labels or poses.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Iterable, Mapping, Sequence


SAFE_CALIBRATION_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)


class CalibrationRejectedError(RuntimeError):
    """Raised when an untrusted site calibration is used for inference."""


def require_ready_calibration(state: Mapping) -> None:
    quality = state.get("calibration_quality", state)
    if quality.get("status") != "READY" or not state.get(
        "deployable", quality.get("accepted_for_normal_inference", False)
    ):
        reasons = quality.get("reasons", ())
        detail = "; ".join(str(value) for value in reasons) or "quality gate failed"
        raise CalibrationRejectedError(f"site calibration is not READY: {detail}")


def _finite(value) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return isfinite(float(value))
    return True


def assess_calibration_quality(
    support_labels: Iterable[int],
    link_coverage: Sequence[float],
    transport_audit: Mapping,
    post_adapter_audit: Mapping,
    mapping_candidates: Sequence[Mapping] = (),
    *,
    mapping_mode: str = "auto",
    allow_automatic_mapping: bool = False,
    absence_trials: int | None = None,
    minimum_absence_trials: int = 3,
    minimum_per_class: int = 2,
    minimum_link_coverage: float = 0.50,
    risk_ready: bool = False,
) -> dict:
    """Assess calibration using support-side information only."""
    counts = Counter(int(value) for value in support_labels)
    missing = [
        class_id for class_id in SAFE_CALIBRATION_CLASSES
        if counts[class_id] < minimum_per_class
    ]
    links_ok = (
        len(link_coverage) == 3
        and all(float(value) >= minimum_link_coverage for value in link_coverage)
    )
    absence_ok = (
        absence_trials is None or absence_trials >= minimum_absence_trials
    )

    identity = transport_audit.get("identity", {})
    selected = transport_audit.get("selected", identity)
    identity_objective = float(identity.get("objective", float("inf")))
    selected_objective = float(selected.get("objective", identity_objective))
    objective_gain = (
        (identity_objective - selected_objective) / max(identity_objective, 1e-8)
        if isfinite(identity_objective) and isfinite(selected_objective)
        else float("nan")
    )
    trajectory_before = float(selected.get(
        "final_trajectory_cosine_before", 0.0
    ))
    trajectory_after = float(selected.get(
        "final_trajectory_cosine_after", trajectory_before
    ))
    transport_ok = bool(transport_audit.get("skipped", False)) or (
        objective_gain >= 0.02 and trajectory_after >= trajectory_before
    )

    final_adapter = post_adapter_audit.get(
        "cross_validation", post_adapter_audit.get("final", {})
    )
    support_accuracy = float(final_adapter.get(
        "classifier", selected.get("accuracy", identity.get("accuracy", 0.0))
    ))
    selector_accuracy = float(final_adapter.get("selector", support_accuracy))

    mapping_margin = None
    mapping_ok = mapping_mode != "auto" or allow_automatic_mapping
    if mapping_mode == "auto" and len(mapping_candidates) >= 2:
        first, second = mapping_candidates[:2]
        mapping_margin = float(first["accuracy"] - second["accuracy"])
        mapping_ok = mapping_ok and mapping_margin >= 0.05

    finite = _finite({
        "link_coverage": list(link_coverage),
        "identity": identity,
        "selected": selected,
        "post_adapter": post_adapter_audit,
    })
    hard_ok = not missing and links_ok and absence_ok and finite
    ready = (
        hard_ok and transport_ok and mapping_ok
        and support_accuracy >= 0.60 and selector_accuracy >= 0.50
    )
    degraded = (
        hard_ok and support_accuracy >= 0.40 and selector_accuracy >= 0.30
    )
    status = "READY" if ready else "DEGRADED" if degraded else "REJECT"

    reasons = []
    if missing:
        reasons.append(f"insufficient safe classes: {missing}")
    if not links_ok:
        reasons.append("one or more CSI links have insufficient valid coverage")
    if not absence_ok:
        reasons.append("empty-room baseline has fewer than three trials")
    if not finite:
        reasons.append("calibration produced a non-finite value")
    if not transport_ok:
        reasons.append("support cross-validation did not validate transport")
    if not mapping_ok:
        if mapping_mode == "auto" and not allow_automatic_mapping:
            reasons.append(
                "automatic link permutation is exploratory; use registered "
                "MAC-to-direction mapping"
            )
        else:
            reasons.append("automatic link mapping is ambiguous")
    if support_accuracy < 0.60:
        reasons.append("post-calibration support action accuracy is below 60%")
    if selector_accuracy < 0.50:
        reasons.append("post-calibration support selector accuracy is below 50%")

    return {
        "status": status,
        "accepted_for_action_pose_inference": status == "READY",
        "risk_ready": bool(risk_ready),
        "risk_reason": (
            None if risk_ready
            else "safe_only_calibration_cannot_validate_danger_direction"
        ),
        "accepted_for_normal_inference": status == "READY" and bool(risk_ready),
        "support_counts": {
            str(class_id): counts[class_id]
            for class_id in SAFE_CALIBRATION_CLASSES
        },
        "link_coverage": [float(value) for value in link_coverage],
        "minimum_link_coverage": minimum_link_coverage,
        "absence_trials": absence_trials,
        "minimum_absence_trials": minimum_absence_trials,
        "transport_objective_relative_gain": objective_gain,
        "trajectory_cosine_gain": trajectory_after - trajectory_before,
        "support_classifier_accuracy": support_accuracy,
        "support_selector_accuracy": selector_accuracy,
        "mapping_mode": mapping_mode,
        "automatic_mapping_allowed": allow_automatic_mapping,
        "mapping_accuracy_margin": mapping_margin,
        "reasons": reasons,
    }
