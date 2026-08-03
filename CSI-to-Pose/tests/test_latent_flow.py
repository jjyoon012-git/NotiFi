import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.latent_flow import LatentFlowPoseNet


class LatentFlowTests(unittest.TestCase):
    def test_identity_start_and_flow_objective(self) -> None:
        model = LatentFlowPoseNet(
            hidden=96, n_blocks=1, heads=4, graph_blocks=1,
            flow_steps=2, dropout=0.0,
        )
        csi = torch.randn(2, 10, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
        link_mask = torch.ones(2, 10, C.N_LINKS, dtype=torch.bool)
        output = model(csi, link_mask)
        torch.testing.assert_close(output["pose_rel"], output["pose_coarse"])
        self.assertEqual(output["motion_latent"].shape, (2, 10, 96))
        self.assertEqual(output["flow_prior_pose"].shape, (2, 10, C.N_JOINTS, 3))

        batch = {
            "pose_rel": torch.randn(2, 10, C.N_JOINTS, 3) * 0.1,
            "valid": torch.ones(2, 10, dtype=torch.bool),
        }
        loss = model.flow_matching_per_sample(output, batch)
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertIsNotNone(model.conditional_flow.velocity[-1].weight.grad)

    def test_robust_checkpoint_missing_keys_are_explicitly_allowed(self) -> None:
        model = LatentFlowPoseNet(
            hidden=96, n_blocks=1, heads=4, graph_blocks=1, dropout=0.0
        )
        self.assertTrue(model.allows_warm_start_missing("flow_mix"))
        self.assertTrue(
            model.allows_warm_start_missing("conditional_flow.velocity.1.weight")
        )
        self.assertFalse(model.allows_warm_start_missing("pose_decoder.tree_logit"))


if __name__ == "__main__":
    unittest.main()
