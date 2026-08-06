import torch

from notifi_pose import contract as C
from notifi_pose.cal23_kp10 import DynamicMotionClassifier
from notifi_pose.cal27_kp10 import (
    Cal27ActionCalibrator,
    fit_local_prototype,
)
from notifi_pose.calibration_quality import CalibrationRejectedError


def _artifact(action_pose_deployable=True):
    model = DynamicMotionClassifier(width=32)
    labels = torch.tensor(list((0, 1, 2, 3, 4, 5, 7, 8)) * 2)
    return {
        "action_pose_deployable": action_pose_deployable,
        "experimental_action_pose": not action_pose_deployable,
        "dynamic_model_config": {"width": 32},
        "dynamic_model_state_dict": model.state_dict(),
        "hierarchy_weight": 0.0,
        "prototype_calibration": {
            "selected": {"temperature": 0.1, "weight": 1.0}
        },
        "source_prototypes": torch.randn(C.N_CLASSES, 32),
        "source_safe_mean": torch.randn(32),
        "support_embedding": torch.randn(len(labels), 32),
        "support_labels": labels,
    }


def test_runtime_exposes_action_but_never_certifies_safe_only_risk():
    model = Cal27ActionCalibrator(_artifact()).eval()
    csi = torch.randn(2, 24, C.N_LINKS, C.N_SUBCARRIERS, 2)
    mask = torch.ones(2, 24, C.N_LINKS, dtype=torch.bool)
    output = model(csi, mask)
    assert output["action_logits"].shape == (2, C.N_CLASSES)
    assert output["risk_logits_experimental"].shape == (2, C.N_RISK)
    assert output["risk_certified"] is False
    assert output["accepted_for_action_pose_inference"] is True
    assert output["accepted_for_normal_inference"] is False


def test_runtime_rejects_artifact_that_failed_action_pose_gate():
    try:
        Cal27ActionCalibrator(_artifact(action_pose_deployable=False))
    except CalibrationRejectedError:
        return
    raise AssertionError("failed calibration artifact was accepted")


def test_experimental_artifact_requires_explicit_opt_in():
    artifact = _artifact(action_pose_deployable=False)
    model = Cal27ActionCalibrator(artifact, allow_experimental=True)
    assert isinstance(model, Cal27ActionCalibrator)
    csi = torch.randn(1, 24, C.N_LINKS, C.N_SUBCARRIERS, 2)
    mask = torch.ones(1, 24, C.N_LINKS, dtype=torch.bool)
    output = model(csi, mask)
    assert output["accepted_for_action_pose_inference"] is False
    assert output["experimental_action_pose_candidate"] is True


def test_local_fit_rejects_missing_safe_class():
    embedding = torch.randn(14, 32)
    labels = torch.tensor(list((0, 1, 2, 3, 4, 5, 7)) * 2)
    try:
        fit_local_prototype(
            embedding, torch.randn(14, C.N_CLASSES), labels,
            torch.randn(C.N_CLASSES, 32), torch.randn(32),
        )
    except CalibrationRejectedError:
        return
    raise AssertionError("incomplete safe calibration was accepted")


def test_runtime_rejects_missing_physical_link():
    model = Cal27ActionCalibrator(_artifact()).eval()
    csi = torch.randn(1, 24, C.N_LINKS, C.N_SUBCARRIERS, 2)
    mask = torch.ones(1, 24, C.N_LINKS, dtype=torch.bool)
    mask[:, :, 1] = False
    try:
        model(csi, mask)
    except CalibrationRejectedError:
        return
    raise AssertionError("trial with a missing physical link was accepted")


def test_runtime_rejects_non_finite_valid_csi():
    model = Cal27ActionCalibrator(_artifact()).eval()
    csi = torch.randn(1, 24, C.N_LINKS, C.N_SUBCARRIERS, 2)
    mask = torch.ones(1, 24, C.N_LINKS, dtype=torch.bool)
    csi[0, 4, 0, 0, 0] = float("nan")
    try:
        model(csi, mask)
    except CalibrationRejectedError:
        return
    raise AssertionError("trial with a valid NaN sample was accepted")
