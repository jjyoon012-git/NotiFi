import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.losses import PoseLoss
from notifi_pose.v3 import V3PoseNet, rotation_6d_to_matrix


class V3ModelTest(unittest.TestCase):
    def test_rotation_6d_is_orthonormal(self):
        matrix = rotation_6d_to_matrix(torch.randn(3, 5, 6))
        identity = matrix.transpose(-1, -2) @ matrix
        self.assertTrue(torch.allclose(identity, torch.eye(3), atol=1e-5))

    def test_forward_loss_and_backward(self):
        torch.manual_seed(7)
        batch, time, subcarriers = 2, 16, C.N_LIVE_SUBCARRIERS
        model = V3PoseNet(
            hidden=96, n_blocks=1, heads=4, graph_blocks=1,
            frequency_tokens=4, dropout=0.0,
        )
        csi = torch.randn(batch, time, C.N_LINKS, subcarriers, 2)
        link_mask = torch.ones(batch, time, C.N_LINKS, dtype=torch.bool)
        output = model(csi, link_mask)

        self.assertEqual(output["pose_rel"].shape, (batch, time, C.N_JOINTS, 3))
        self.assertEqual(output["root"].shape, (batch, time, 3))
        self.assertEqual(output["phase_logits"].shape, (batch, time, 4))
        self.assertEqual(output["contact_logits"].shape, (batch, time, 4))
        self.assertEqual(output["bone_direction"].shape, (batch, time, C.N_JOINTS, 3))
        child, parent = C.SKELETON_EDGES[0]
        length = torch.linalg.vector_norm(
            output["kinematic_pose"][:, :, child]
            - output["kinematic_pose"][:, :, parent],
            dim=-1,
        )
        self.assertTrue(torch.allclose(length.std(1), torch.zeros(batch), atol=1e-5))

        target = {
            "pose_rel": torch.randn(batch, time, C.N_JOINTS, 3) * 0.2,
            "root": torch.randn(batch, time, 3) * 0.2,
            "valid": torch.ones(batch, time, dtype=torch.bool),
            "class_id": torch.tensor([0, 1]),
            "risk_id": torch.tensor([0, 2]),
            "domain_id": torch.tensor([0, 4]),
        }
        loss_fn = PoseLoss(
            class_counts=torch.ones(C.N_CLASSES),
            risk_counts=torch.ones(C.N_RISK),
            lambda_acceleration=0.1, lambda_contact=0.1,
            lambda_phase=0.1, lambda_domain=0.1,
            device="cpu",
        )
        loss, parts = loss_fn(output, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["_per_sample_total"].shape, (batch,))
        loss.backward()
        self.assertIsNotNone(model.tokenizer.stem[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
