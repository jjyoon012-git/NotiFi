import torch
from torch import nn

from notifi_pose.cal42_kp10 import (
    Cal42GuardedCalibrator,
    guarded_phase_blend,
    risk_logits_from_action,
)
from notifi_pose.calibration_quality import CalibrationRejectedError


def test_guard_preserves_energy_danger_prediction_exactly():
    energy = torch.tensor([[0.0] * 12 + [5.0, 1.0, 0.0, 0.0, 0.0]])
    phase = torch.tensor([[8.0] + [0.0] * 16])
    output = guarded_phase_blend(energy, phase)
    assert output.argmax(-1).item() == 12
    assert torch.equal(output, torch.log_softmax(energy, dim=-1))


def test_guard_is_differentiable_for_mixed_batches():
    energy = torch.randn(3, 17, requires_grad=True)
    energy.data[0, 12] = 8.0
    phase = torch.randn(3, 17, requires_grad=True)
    output = guarded_phase_blend(energy, phase)
    output.sum().backward()
    assert torch.isfinite(energy.grad).all()
    assert torch.isfinite(phase.grad).all()


def test_guard_rejects_invalid_weight():
    try:
        guarded_phase_blend(torch.zeros(1, 17), torch.zeros(1, 17), 1.1)
    except ValueError:
        return
    raise AssertionError("invalid phase weight was accepted")


def test_runtime_requires_explicit_experimental_opt_in():
    try:
        Cal42GuardedCalibrator(nn.Identity(), nn.Identity())
    except CalibrationRejectedError:
        return
    raise AssertionError("CAL42 loaded without experimental opt-in")


def test_runtime_rejects_invalid_phase_weight_at_construction():
    try:
        Cal42GuardedCalibrator(
            nn.Identity(), nn.Identity(), phase_weight=-0.1,
            allow_experimental=True,
        )
    except ValueError:
        return
    raise AssertionError("invalid phase weight reached inference")


def test_runtime_rejects_mismatched_branch_support():
    class Branch(nn.Identity):
        model_feature_mode = "energy"

        def __init__(self, rows):
            super().__init__()
            self.support_rows = rows

    energy = Branch(("trial-a", "trial-b"))
    phase = Branch(("trial-a", "trial-c"))
    phase.model_feature_mode = "physical_phase"
    try:
        Cal42GuardedCalibrator(energy, phase, allow_experimental=True)
    except CalibrationRejectedError:
        return
    raise AssertionError("branches with different support were combined")


def test_runtime_rejects_wrong_branch_feature_mode():
    class Branch(nn.Identity):
        support_rows = ("trial-a",)
        model_feature_mode = "energy"

    try:
        Cal42GuardedCalibrator(Branch(), Branch(), allow_experimental=True)
    except CalibrationRejectedError:
        return
    raise AssertionError("two energy branches were combined as CAL42")


def test_action_risk_groups_have_fixed_shape():
    logits = torch.randn(4, 17)
    grouped = risk_logits_from_action(logits)
    assert grouped.shape == (4, 3)
    assert torch.isfinite(grouped).all()
    equal = risk_logits_from_action(torch.zeros(1, 17))
    assert torch.allclose(equal, torch.zeros_like(equal), atol=1e-6)


def test_experimental_runtime_preserves_energy_contract():
    class Branch(nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, csi, link_mask):
            return {
                "action_logits": self.logits.expand(len(csi), -1),
                "risk_logits_experimental": torch.zeros(len(csi), 3),
            }

    energy_logits = torch.tensor([[0.0] * 12 + [5.0, 0.0, 0.0, 0.0, 0.0]])
    phase_logits = torch.tensor([[8.0] + [0.0] * 16])
    model = Cal42GuardedCalibrator(
        Branch(energy_logits), Branch(phase_logits), allow_experimental=True
    )
    output = model(torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool))
    assert output["action_logits"].argmax(-1).tolist() == [12, 12]
    assert output["calibration_status"] == "EXPERIMENTAL"
    assert output["risk_certified"] is False
    assert output["accepted_for_action_pose_inference"] is False
    assert output["accepted_for_normal_inference"] is False
