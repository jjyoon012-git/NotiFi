import torch
from torch import nn

from notifi_pose.cal43_kp10 import Cal43GuardedCalibrator


class Branch(nn.Module):
    support_rows = ("trial-a",)

    def __init__(self, mode, logits):
        super().__init__()
        self.model_feature_mode = mode
        self.register_buffer("logits", logits)

    def forward(self, csi, link_mask):
        return {
            "action_logits": self.logits.expand(len(csi), -1),
            "risk_logits_experimental": torch.zeros(len(csi), 3),
        }


def test_cal43_locks_phase_weight_to_quarter():
    energy = Branch("energy", torch.zeros(1, 17))
    phase = Branch("physical_phase", torch.ones(1, 17))
    model = Cal43GuardedCalibrator(
        energy, phase, allow_experimental=True
    )
    assert model.phase_weight == 0.25
    output = model(torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool))
    assert output["calibration_status"] == "EXPERIMENTAL"
    assert output["accepted_for_normal_inference"] is False
