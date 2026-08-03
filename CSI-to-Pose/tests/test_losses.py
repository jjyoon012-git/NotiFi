import unittest

import numpy as np
import torch

from notifi_pose import contract as C
from notifi_pose.losses import PoseLoss, target_motion


class LossTests(unittest.TestCase):
    def test_motion_supervision_and_dynamic_loss_are_finite(self) -> None:
        batch_size, frames = 2, 8
        pose = torch.zeros(batch_size, frames, C.N_JOINTS, 3)
        root = torch.zeros(batch_size, frames, 3)
        pose[:, 3:, 15, 1] = 0.4
        valid = torch.ones(batch_size, frames, dtype=torch.bool)
        speed, pair_valid = target_motion(pose, root, valid)
        self.assertTrue(speed[:, 3].gt(0).all())
        self.assertTrue(pair_valid.all())

        output = {
            "pose_rel": torch.zeros_like(pose, requires_grad=True),
            "root": torch.zeros_like(root, requires_grad=True),
            "motion": torch.zeros(batch_size, frames, requires_grad=True),
            "class_logits": torch.zeros(batch_size, C.N_CLASSES, requires_grad=True),
            "risk_logits": torch.zeros(batch_size, C.N_RISK, requires_grad=True),
        }
        batch = {
            "pose_rel": pose,
            "root": root,
            "valid": valid,
            "class_id": torch.zeros(batch_size, dtype=torch.long),
            "risk_id": torch.zeros(batch_size, dtype=torch.long),
        }
        criterion = PoseLoss(
            class_counts=np.ones(C.N_CLASSES),
            risk_counts=np.ones(C.N_RISK),
            lambda_velocity=0.2,
            lambda_motion=0.1,
            lambda_displacement=0.2,
            motion_weight=3.0,
        )
        loss, parts = criterion(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["velocity"], 0)
        self.assertGreater(parts["motion"], 0)
        self.assertGreater(parts["displacement"], 0)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
