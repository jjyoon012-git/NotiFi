from notifi_pose.calibration_quality import (
    CalibrationRejectedError,
    assess_calibration_quality,
    require_ready_calibration,
)


LABELS = [value for value in (0, 1, 2, 3, 4, 5, 7, 8) for _ in range(2)]


def audit(final_accuracy=0.75):
    return (
        {
            "skipped": False,
            "identity": {"objective": 10.0, "accuracy": 0.25},
            "selected": {
                "objective": 7.0,
                "accuracy": 0.25,
                "final_trajectory_cosine_before": 0.7,
                "final_trajectory_cosine_after": 0.9,
            },
        },
        {"final": {"classifier": final_accuracy, "selector": 0.625}},
    )


def test_ready_safe_calibration_is_action_only():
    transport, adapter = audit()
    result = assess_calibration_quality(
        LABELS, [0.99, 0.98, 0.97], transport, adapter,
        [
            {"accuracy": 0.25},
            {"accuracy": 0.18},
        ],
        allow_automatic_mapping=True,
    )
    assert result["status"] == "READY"
    assert result["accepted_for_action_pose_inference"]
    assert result["risk_ready"] is False
    assert not result["accepted_for_normal_inference"]


def test_independent_risk_evidence_allows_normal_inference():
    transport, adapter = audit()
    result = assess_calibration_quality(
        LABELS, [0.99, 0.98, 0.97], transport, adapter,
        [{"accuracy": 0.25}, {"accuracy": 0.18}],
        allow_automatic_mapping=True,
        risk_ready=True,
    )
    assert result["status"] == "READY"
    assert result["accepted_for_normal_inference"]


def test_runtime_refuses_action_ready_safe_only_quality():
    transport, adapter = audit()
    quality = assess_calibration_quality(
        LABELS, [0.99, 0.98, 0.97], transport, adapter,
        [{"accuracy": 0.25}, {"accuracy": 0.18}],
        allow_automatic_mapping=True,
    )
    try:
        require_ready_calibration({"calibration_quality": quality})
    except CalibrationRejectedError:
        return
    raise AssertionError("safe-only quality reached normal inference")


def test_missing_prompt_class_is_rejected():
    transport, adapter = audit()
    result = assess_calibration_quality(
        LABELS[:-2], [1.0, 1.0, 1.0], transport, adapter,
        mapping_mode="identity",
    )
    assert result["status"] == "REJECT"
    assert not result["accepted_for_normal_inference"]


def test_weak_but_complete_calibration_is_degraded():
    transport, adapter = audit(final_accuracy=0.45)
    result = assess_calibration_quality(
        LABELS, [1.0, 1.0, 1.0], transport, adapter,
        mapping_mode="identity",
    )
    assert result["status"] == "DEGRADED"
    assert not result["accepted_for_normal_inference"]


def test_missing_empty_room_baseline_is_rejected():
    transport, adapter = audit()
    result = assess_calibration_quality(
        LABELS, [1.0, 1.0, 1.0], transport, adapter,
        mapping_mode="identity", absence_trials=1,
    )
    assert result["status"] == "REJECT"


def test_runtime_refuses_non_ready_artifact():
    state = {
        "deployable": False,
        "calibration_quality": {
            "status": "DEGRADED", "reasons": ["weak support"],
        },
    }
    try:
        require_ready_calibration(state)
    except CalibrationRejectedError as error:
        assert "weak support" in str(error)
        return
    raise AssertionError("non-ready calibration must not reach inference")
