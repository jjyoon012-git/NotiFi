import unittest

import torch
from torch import nn

from notifi_pose.tools.calibrate_v13s_yja import (
    MomentInputCalibration,
    OutputCalibration,
    fit_affine,
    with_prototype_targets,
)
from notifi_pose import contract as C


class EchoModel(nn.Module):
    def forward(self, csi, link_mask):
        return {"echo": csi, "mask": link_mask}


class CalibrationStub(nn.Module):
    def __init__(self, logit_value):
        super().__init__()
        self.logit_value = float(logit_value)

    def forward(self, csi, link_mask):
        batch, frames = csi.shape[:2]
        device = csi.device
        return {
            "pose_rel": torch.zeros(batch, frames, C.N_JOINTS, 3, device=device),
            "root": torch.full(
                (batch, frames, 3), self.logit_value, device=device
            ),
            "class_logits": torch.full(
                (batch, C.N_CLASSES), self.logit_value, device=device
            ),
            "risk_logits": torch.full(
                (batch, C.N_RISK), self.logit_value, device=device
            ),
        }


class V13YjaCalibrationTests(unittest.TestCase):
    def test_zero_moment_strength_preserves_input_exactly(self):
        source_mu = torch.zeros(3, 4, 2)
        source_sigma = torch.ones(3, 4, 2)
        target_mu = torch.full((3, 4, 2), 2.0)
        target_sigma = torch.full((3, 4, 2), 3.0)
        model = MomentInputCalibration(
            EchoModel(), source_mu, source_sigma, target_mu, target_sigma
        )
        model.set_strength(0.0)
        csi = torch.randn(2, 5, 3, 4, 2)
        mask = torch.ones(2, 5, 3, dtype=torch.bool)
        output = model(csi, mask)
        self.assertTrue(torch.equal(output["echo"], csi))

    def test_affine_fit_recovers_known_transform(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(512, 3, generator=generator)
        matrix = torch.tensor([
            [1.2, 0.1, -0.2],
            [0.0, 0.8, 0.3],
            [0.2, -0.1, 1.1],
        ])
        bias = torch.tensor([0.4, -0.3, 0.2])
        y = x @ matrix + bias
        fitted_matrix, fitted_bias = fit_affine(x, y, ridge=0.0)
        self.assertTrue(torch.allclose(fitted_matrix, matrix, atol=1e-5))
        self.assertTrue(torch.allclose(fitted_bias, bias, atol=1e-5))

    def test_basic_pose_calibration_preserves_raw_classification_logits(self):
        calibrated = OutputCalibration(
            CalibrationStub(9.0),
            torch.eye(3), torch.zeros(3), torch.zeros(C.N_JOINTS, 3),
            torch.eye(3), torch.zeros(3),
            torch.ones(C.N_CLASSES), torch.zeros(C.N_CLASSES),
            torch.ones(C.N_RISK), torch.zeros(C.N_RISK),
            raw_logit_base=CalibrationStub(1.0),
            preserve_raw_root=True,
        )
        output = calibrated(
            torch.zeros(2, 5, 3, 4, 2),
            torch.ones(2, 5, 3, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(output["class_logits"], torch.ones(2, C.N_CLASSES)))
        self.assertTrue(torch.equal(output["risk_logits"], torch.ones(2, C.N_RISK)))
        self.assertTrue(torch.equal(output["root"], torch.ones(2, 5, 3)))

    def test_basic_pose_targets_use_source_prototype_not_target_gt(self):
        data = {
            "pose_target": torch.full((2, 3, C.N_JOINTS, 3), 99.0),
            "class_target": torch.tensor([1, 2]),
        }
        prototypes = {
            1: torch.full((C.N_JOINTS, 3), 1.0),
            2: torch.full((C.N_JOINTS, 3), 2.0),
        }
        replaced = with_prototype_targets(data, prototypes)
        self.assertTrue(torch.equal(
            replaced["pose_target"][0], torch.ones(3, C.N_JOINTS, 3)
        ))
        self.assertTrue(torch.equal(
            replaced["pose_target"][1], torch.full((3, C.N_JOINTS, 3), 2.0)
        ))


if __name__ == "__main__":
    unittest.main()
