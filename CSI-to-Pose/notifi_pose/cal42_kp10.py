"""Guarded energy/physical-phase evidence fusion for CAL42-KP10."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .calibration_quality import CalibrationRejectedError


def guarded_phase_blend(energy_logits: torch.Tensor,
                        phase_logits: torch.Tensor,
                        phase_weight: float = 0.15) -> torch.Tensor:
    """Add phase evidence without changing an energy-branch danger decision."""
    if energy_logits.shape != phase_logits.shape:
        raise ValueError("energy and phase logits must have the same shape")
    if not 0.0 <= float(phase_weight) <= 1.0:
        raise ValueError("phase_weight must be in [0, 1]")
    energy = F.log_softmax(energy_logits.float(), dim=-1)
    phase = F.log_softmax(phase_logits.float(), dim=-1)
    output = (1.0 - float(phase_weight)) * energy + float(phase_weight) * phase
    danger_anchor = energy_logits.argmax(-1) >= 12
    return torch.where(danger_anchor[..., None], energy, output)


def risk_logits_from_action(action_logits: torch.Tensor) -> torch.Tensor:
    """Aggregate 17 calibrated actions into the fixed three risk groups."""
    return torch.stack((
        torch.logsumexp(action_logits[:, :9], dim=-1) - math.log(9),
        torch.logsumexp(action_logits[:, 9:12], dim=-1) - math.log(3),
        torch.logsumexp(action_logits[:, 12:17], dim=-1) - math.log(5),
    ), dim=-1)


class Cal42GuardedCalibrator(nn.Module):
    """Experimental runtime wrapper around two fitted CAL27 calibrators."""

    def __init__(self, energy_calibrator: nn.Module, phase_calibrator: nn.Module,
                 phase_weight: float = 0.15, *, allow_experimental: bool = False):
        super().__init__()
        if not allow_experimental:
            raise CalibrationRejectedError(
                "CAL42 has not passed the unseen danger capability gate"
            )
        if not 0.0 <= float(phase_weight) <= 1.0:
            raise ValueError("phase_weight must be in [0, 1]")
        energy_mode = getattr(energy_calibrator, "model_feature_mode", None)
        phase_mode = getattr(phase_calibrator, "model_feature_mode", None)
        if energy_mode is not None and energy_mode != "energy":
            raise CalibrationRejectedError("CAL42 energy branch has the wrong feature mode")
        if phase_mode is not None and phase_mode != "physical_phase":
            raise CalibrationRejectedError("CAL42 phase branch has the wrong feature mode")
        energy_rows = getattr(energy_calibrator, "support_rows", None)
        phase_rows = getattr(phase_calibrator, "support_rows", None)
        if (
            energy_rows is not None
            and phase_rows is not None
            and tuple(energy_rows) != tuple(phase_rows)
        ):
            raise CalibrationRejectedError(
                "CAL42 branches were fitted with different calibration support"
            )
        self.energy_calibrator = energy_calibrator
        self.phase_calibrator = phase_calibrator
        self.phase_weight = float(phase_weight)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        energy = self.energy_calibrator(csi, link_mask)
        phase = self.phase_calibrator(csi, link_mask)
        action = guarded_phase_blend(
            energy["action_logits"], phase["action_logits"], self.phase_weight
        )
        grouped_risk = risk_logits_from_action(action)
        return {
            **energy,
            "action_logits": action,
            "phase_action_logits": phase["action_logits"],
            "risk_logits_grouped_experimental": grouped_risk,
            "risk_certified": False,
            "calibration_status": "EXPERIMENTAL",
            "accepted_for_action_pose_inference": False,
            "experimental_action_pose_candidate": True,
            "accepted_for_normal_inference": False,
        }
