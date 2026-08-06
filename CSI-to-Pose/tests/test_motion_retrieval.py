import copy
import unittest

import torch

from notifi_pose import contract as C
from notifi_pose.motion_retrieval import (
    ContactProfileHead,
    GeometryResidualReranker,
    MotionProgressHead,
    MotionProfileHead,
    PartMotionProfileHead,
    PartTrajectoryHead,
    ProfileCandidateRanker,
    TemporalMotionSelector,
    geometric_pair_features,
    masked_temporal_bins,
)
from notifi_pose.tools.calibrate_profile_action_retrieval import retrieval_features
from notifi_pose.tools.calibrate_motion_profile_warping import (
    monotonic_energy_warp,
)
from notifi_pose.tools.calibrate_risk_adaptive_blend import adaptive_strength
from notifi_pose.tools.train_profile_candidate_ranker import (
    render_ranked_action,
)


class MotionRetrievalTest(unittest.TestCase):
    def test_motion_progress_is_monotonic_and_masked(self):
        model = MotionProgressHead(input_dim=20, width=32)
        mask = torch.zeros(2, 24, dtype=torch.bool)
        mask[:, :17] = True
        progress = model(torch.randn(2, 24, 20), mask)["progress"]
        self.assertEqual(tuple(progress.shape), (2, 24))
        self.assertTrue((progress[:, 1:17] >= progress[:, :16]).all())
        self.assertTrue(torch.allclose(progress[:, 16], torch.ones(2)))
        self.assertTrue(torch.equal(progress[:, 17:], torch.zeros(2, 7)))

    def test_contact_profile_respects_frame_mask(self):
        model = ContactProfileHead(input_dim=20, width=32)
        mask = torch.zeros(2, 24, dtype=torch.bool)
        mask[:, :17] = True
        logits = model(torch.randn(2, 24, 20), mask)["contact_logits"]
        self.assertEqual(tuple(logits.shape), (2, 24, 8))
        self.assertTrue(torch.equal(logits[:, 17:], torch.zeros(2, 7, 8)))

    def test_profile_ranker_scores_each_candidate(self):
        model = ProfileCandidateRanker(feature_dim=8, hidden=32)
        score = model(
            torch.randn(3, 5, 8), torch.randint(0, C.N_CLASSES, (3, 5))
        )
        self.assertEqual(tuple(score.shape), (3, 5))
        self.assertTrue(torch.isfinite(score).all())

    def test_contextual_profile_ranker_requires_query_context(self):
        model = ProfileCandidateRanker(
            feature_dim=8, hidden=32, context_dim=20
        )
        score = model(
            torch.randn(3, 5, 8),
            torch.randint(0, C.N_CLASSES, (3, 5)),
            torch.randn(3, 5, 20),
        )
        self.assertEqual(tuple(score.shape), (3, 5))
        with self.assertRaises(ValueError):
            model(
                torch.randn(3, 5, 8),
                torch.randint(0, C.N_CLASSES, (3, 5)),
            )

    def test_profile_retrieval_excludes_training_query(self):
        trials = 4
        bank = torch.randn(trials, C.CACHE_FRAMES, C.N_JOINTS, 3)
        valid = torch.ones(trials, C.CACHE_FRAMES, dtype=torch.bool)
        data = {
            "checkpoint": {"train_class": torch.zeros(trials, dtype=torch.long)},
            "train_bank": bank,
            "baseline_bank": bank + 0.01,
            "fused_action": torch.zeros(trials, C.N_CLASSES),
            "risk_probability": torch.full(
                (trials, C.N_RISK), 1.0 / C.N_RISK
            ),
            "target_valid": valid,
            "inference_valid": valid.clone(),
            "predicted_scalar_profile": torch.rand(trials, C.CACHE_FRAMES),
            "predicted_part_profile": torch.rand(
                trials, C.CACHE_FRAMES, len(C.JOINT_GROUPS)
            ),
        }
        groups = retrieval_features(
            data, action_top_k=1, self_indices=torch.arange(trials)
        )
        for item, choices in enumerate(groups):
            self.assertNotIn(item, choices[0]["indices"].tolist())

    def test_profile_retrieval_uses_csi_mask_not_target_mask(self):
        trials = 4
        frames = C.CACHE_FRAMES
        bank = torch.randn(trials, frames, C.N_JOINTS, 3)
        inference_valid = torch.zeros(trials, frames, dtype=torch.bool)
        inference_valid[:, : frames // 2] = True
        data = {
            "checkpoint": {"train_class": torch.zeros(trials, dtype=torch.long)},
            "train_bank": bank,
            "baseline_bank": bank + 0.01,
            "fused_action": torch.zeros(trials, C.N_CLASSES),
            "risk_probability": torch.full(
                (trials, C.N_RISK), 1.0 / C.N_RISK
            ),
            "target_valid": torch.ones(trials, frames, dtype=torch.bool),
            "inference_valid": inference_valid,
            "predicted_scalar_profile": torch.rand(trials, frames),
            "predicted_part_profile": torch.rand(
                trials, frames, len(C.JOINT_GROUPS)
            ),
        }
        changed_target = copy.deepcopy(data)
        changed_target["target_valid"][:, frames // 3:] = False
        left = retrieval_features(data, action_top_k=1)
        right = retrieval_features(changed_target, action_top_k=1)
        for left_groups, right_groups in zip(left, right):
            self.assertTrue(torch.equal(
                left_groups[0]["indices"], right_groups[0]["indices"]
            ))
            self.assertTrue(torch.allclose(
                left_groups[0]["part_values"],
                right_groups[0]["part_values"],
            ))

    def test_profile_pose_render_ignores_target_mask(self):
        trials = 4
        frames = C.CACHE_FRAMES
        bank = torch.randn(trials, frames, C.N_JOINTS, 3)
        inference_valid = torch.zeros(trials, frames, dtype=torch.bool)
        inference_valid[:, : frames // 2] = True
        fused_action = torch.full((trials, C.N_CLASSES), -10.0)
        fused_action[:, 0] = 10.0
        data = {
            "checkpoint": {"train_class": torch.zeros(trials, dtype=torch.long)},
            "train_bank": bank,
            "baseline_bank": bank + 0.01,
            "fused_action": fused_action,
            "risk_probability": torch.full(
                (trials, C.N_RISK), 1.0 / C.N_RISK
            ),
            "target_valid": torch.ones(trials, frames, dtype=torch.bool),
            "inference_valid": inference_valid,
            "predicted_scalar_profile": torch.rand(trials, frames),
            "predicted_part_profile": torch.rand(
                trials, frames, len(C.JOINT_GROUPS)
            ),
        }
        changed_target = copy.deepcopy(data)
        changed_target["target_valid"][:, frames // 3:] = False
        model = ProfileCandidateRanker(feature_dim=8, hidden=32)
        left = render_ranked_action(
            model, data, retrieval_features(data, action_top_k=1), "cpu"
        )
        right = render_ranked_action(
            model, changed_target,
            retrieval_features(changed_target, action_top_k=1), "cpu",
        )
        self.assertTrue(torch.equal(left, right))

    def test_masked_temporal_bins_ignores_padding(self):
        features = torch.ones(2, 12, 5)
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[0, :8] = True
        mask[1, :12] = True
        features[0, 8:] = 99.0
        pooled, valid = masked_temporal_bins(features, mask, bins=4)
        self.assertEqual(tuple(pooled.shape), (2, 4, 5))
        self.assertEqual(tuple(valid.shape), (2, 4))
        self.assertTrue(torch.allclose(pooled[0][valid[0]], torch.ones_like(pooled[0][valid[0]])))

    def test_motion_selector_outputs_pooled_query(self):
        model = TemporalMotionSelector(
            input_dim=12, embedding_dim=8, width=24,
            bins=6, layers=1, heads=4,
        )
        output = model(
            torch.randn(3, 18, 12), torch.ones(3, 18, dtype=torch.bool)
        )
        self.assertEqual(tuple(output["motion_embedding"].shape), (3, 8))
        self.assertEqual(tuple(output["pooled_features"].shape), (3, 48))
        self.assertEqual(tuple(output["action_logits"].shape), (3, C.N_CLASSES))

    def test_geometric_pair_features_are_finite(self):
        baseline = torch.randn(2, 20, C.N_JOINTS, 3)
        candidates = torch.randn(2, 4, 20, C.N_JOINTS, 3)
        values = geometric_pair_features(baseline, candidates)
        self.assertEqual(tuple(values.shape), (2, 4, 16))
        self.assertTrue(torch.isfinite(values).all())

    def test_geometry_reranker_keeps_base_frozen(self):
        model = GeometryResidualReranker(
            query_dim=32, embedding_dim=16, pair_dim=16, width=24
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.base.parameters()))
        output = model(
            torch.randn(2, 32), torch.randn(2, 16),
            torch.randn(2, 5, 16),
            torch.randint(0, C.N_CLASSES, (2, 5)),
            torch.softmax(torch.randn(2, C.N_RISK), dim=-1),
            torch.randn(2, 5), torch.randn(2, 5), torch.randn(2, 5, 16),
        )
        self.assertEqual(tuple(output.shape), (2, 5))

    def test_motion_profile_respects_frame_mask(self):
        model = MotionProfileHead(input_dim=20, width=32)
        mask = torch.zeros(2, 24, dtype=torch.bool)
        mask[:, :17] = True
        output = model(torch.randn(2, 24, 20), mask)
        self.assertEqual(tuple(output["speed"].shape), (2, 24))
        self.assertTrue(torch.equal(output["speed"][:, 17:], torch.zeros(2, 7)))
        self.assertTrue((output["speed"][:, :17] >= 0).all())

    def test_part_motion_profile_respects_frame_mask(self):
        model = PartMotionProfileHead(input_dim=20, width=32)
        mask = torch.zeros(2, 24, dtype=torch.bool)
        mask[:, :17] = True
        output = model(torch.randn(2, 24, 20), mask)
        self.assertEqual(tuple(output["part_speed"].shape), (2, 24, 6))
        self.assertTrue(torch.equal(
            output["part_speed"][:, 17:], torch.zeros(2, 7, 6)
        ))
        self.assertTrue((output["part_speed"][:, :17] >= 0).all())

    def test_zero_strength_motion_warp_is_identity_on_valid_frames(self):
        pose = torch.randn(2, 24, C.N_JOINTS, 3)
        activity = torch.rand(2, 24)
        mask = torch.zeros(2, 24, dtype=torch.bool)
        mask[:, :17] = True
        output = monotonic_energy_warp(pose, activity, mask, 0.0, 0.15)
        self.assertTrue(torch.allclose(output[:, :17], pose[:, :17]))
        self.assertTrue(torch.equal(
            output[:, 17:], torch.zeros_like(output[:, 17:])
        ))

    def test_part_trajectory_respects_frame_mask(self):
        model = PartTrajectoryHead(input_dim=20, width=32)
        mask = torch.zeros(2, 24, dtype=torch.bool)
        mask[:, :17] = True
        output = model(torch.randn(2, 24, 20), mask)["part_trajectory"]
        self.assertEqual(tuple(output.shape), (2, 24, 6, 3))
        self.assertTrue(torch.equal(
            output[:, 17:], torch.zeros(2, 7, 6, 3)
        ))

    def test_adaptive_strength_is_bounded_and_danger_sensitive(self):
        risk = torch.tensor(((0.95, 0.04, 0.01), (0.01, 0.04, 0.95)))
        strength = adaptive_strength(risk, 0.65, 0.15, -0.10)
        self.assertTrue(((strength >= 0.40) & (strength <= 0.95)).all())
        self.assertGreater(float(strength[1]), float(strength[0]))


if __name__ == "__main__":
    unittest.main()
