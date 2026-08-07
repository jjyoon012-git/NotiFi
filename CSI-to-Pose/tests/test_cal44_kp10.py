import torch

from notifi_pose.cal44_kp10 import (
    apply_hierarchical_risk,
    fit_hierarchical_risk,
    preserve_control_danger,
    risk_logits_from_action,
)


def test_risk_logits_group_all_actions():
    logits = torch.zeros(2, 17)
    grouped = risk_logits_from_action(logits)
    assert grouped.shape == (2, 3)
    assert torch.allclose(grouped[0].exp(), torch.tensor([9.0, 3.0, 5.0]))


def test_hierarchy_selection_uses_support_only():
    action = torch.full((6, 17), -4.0)
    action[:3, 0] = 4.0
    action[3:, 9] = 4.0
    direct = torch.zeros(6, 3)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    calibration = fit_hierarchical_risk(action, direct, labels)
    output = apply_hierarchical_risk(action, direct, calibration)
    assert calibration["selected"]["weight"] > 0
    assert torch.equal(output.argmax(-1), labels)


def test_control_danger_is_immutable():
    control = torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]])
    candidate = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    output = preserve_control_danger(control, candidate, danger_start=2)
    assert output[0].argmax().item() == 2
    assert output[1].argmax().item() == 1
