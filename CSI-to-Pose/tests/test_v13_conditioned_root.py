import unittest
from types import SimpleNamespace

import torch
from torch import nn

from notifi_pose.hybrid_v10 import (
    P2V13ConditionedRootHybridNet,
    P2V13DecoupledMotionRootHybridNet,
)
from notifi_pose.tools.train_p2_v9_hybrid import configure_trainable_parameters


class _Base(nn.Module):
    hidden = 96


class V13ConditionedRootTests(unittest.TestCase):
    def test_zero_condition_preserves_initial_root(self):
        model = P2V13ConditionedRootHybridNet(_Base())
        feature = torch.randn(2, 12, 128)
        root = torch.randn(2, 12, 3)
        valid = torch.ones(2, 12, dtype=torch.bool)
        adjusted, auxiliary = model.root_candidate(
            feature, root, valid, feature.mean(1)
        )
        self.assertTrue(torch.allclose(adjusted, root, atol=1e-7))
        self.assertTrue(torch.allclose(
            auxiliary["motion_condition_v13"],
            torch.zeros_like(auxiliary["motion_condition_v13"]),
        ))

    def test_decoupled_motion_pretraining_only_updates_motion_branch(self):
        model = P2V13DecoupledMotionRootHybridNet(_Base())
        selected = configure_trainable_parameters(model, SimpleNamespace(
            objective="motion_only",
            residual_decoder="decoupled_motion_root",
            freeze_motion_branch=False,
        ))
        selected_ids = {id(parameter) for parameter in selected}
        for name, parameter in model.named_parameters():
            expected = name.startswith((
                "motion_trajectory_context.", "motion_observation_head.",
            ))
            self.assertEqual(id(parameter) in selected_ids, expected, name)

    def test_decoupled_context_starts_as_root_context_copy(self):
        model = P2V13DecoupledMotionRootHybridNet(_Base()).eval()
        feature = torch.randn(2, 128, 12)
        root_context = model.root_trajectory_context(feature)
        motion_context = model.motion_trajectory_context(feature)
        self.assertTrue(torch.allclose(root_context, motion_context, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
