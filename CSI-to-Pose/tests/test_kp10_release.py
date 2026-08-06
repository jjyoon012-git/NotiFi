import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from notifi_pose import contract as C
from notifi_pose.tools import verify_kp10_release as release
from notifi_pose.tools.diagnose_kp10_hypothesis_coverage import (
    export_hypotheses, paired_bootstrap,
)
from notifi_pose.tools.audit_kp10_paired_bootstrap import class_delta_audit
from notifi_pose.tools.audit_kp10_target_invariance import (
    poison_evaluation_targets,
)
from notifi_pose.tools.kp10_inference import inference_view


class KP10ReleaseManifestTest(unittest.TestCase):
    def test_target_poisoning_does_not_mutate_source(self):
        source = {
            "target_pose": torch.zeros(1, 2, 3, 3),
            "target_valid": torch.ones(1, 2, dtype=torch.bool),
            "target_class": torch.zeros(1, dtype=torch.long),
            "target_risk": torch.zeros(1, dtype=torch.long),
            "pool": {"target_cost": torch.zeros(1, 2)},
        }
        poisoned = poison_evaluation_targets(source)
        self.assertTrue(torch.equal(source["target_pose"], torch.zeros_like(
            source["target_pose"]
        )))
        self.assertEqual(float(poisoned["target_pose"].mean()), 123.0)
        self.assertEqual(float(poisoned["pool"]["target_cost"].mean()), -999.0)

    def test_class_delta_audit_keeps_sign_of_kp10_minus_kp6(self):
        kp6 = {
            "pose": torch.tensor([0.20, 0.30]),
            "distal": torch.tensor([0.40, 0.50]),
        }
        kp10 = {
            "pose": torch.tensor([0.10, 0.40]),
            "distal": torch.tensor([0.30, 0.70]),
        }
        audit = class_delta_audit(kp6, kp10, torch.tensor([0, 0]))
        row = audit[next(iter(audit))]
        self.assertAlmostEqual(row["pose_delta_cm"], 0.0, places=5)
        self.assertAlmostEqual(row["distal_delta_cm"], 5.0, places=5)
        self.assertAlmostEqual(row["pose_trial_win_rate"], 0.5)

    def test_inference_view_strips_evaluation_targets_and_costs(self):
        marker = object()
        required = {
            name: marker for name in (
                "checkpoint", "baseline", "baseline_bank", "train_bank",
                "fused_action", "risk_probability", "logits",
                "scalar_distance", "part_distance",
                "predicted_scalar_profile", "predicted_part_profile",
                "inference_valid",
            )
        }
        required["pool"] = {
            "indices": marker,
            "retrieval_score": marker,
            "action_log_probability": marker,
            "target_cost": "GT-derived training label",
        }
        required.update({
            "target_pose": marker,
            "target_valid": marker,
            "target_class": marker,
            "target_risk": marker,
        })
        view = inference_view(required)
        self.assertFalse(any(name.startswith("target_") for name in view))
        self.assertNotIn("target_cost", view["pool"])

    def test_inference_view_requires_csi_validity_mask(self):
        with self.assertRaisesRegex(KeyError, "inference_valid"):
            inference_view({})

    def test_paired_bootstrap_reports_trial_level_gain(self):
        result = paired_bootstrap(
            torch.tensor([0.01, 0.02, 0.03]), samples=1_000, seed=7
        )
        self.assertAlmostEqual(result["mean_gain_cm"], 2.0, places=5)
        self.assertEqual(result["trials"], 3)
        self.assertGreater(result["ci95_cm"][0], 0.0)

    def test_multi5_export_contains_no_gt(self):
        predicted = torch.zeros(2, 5, 8, C.N_JOINTS, 3)
        probability = torch.rand(2, 5)
        classes = torch.arange(5)[None].expand(2, -1)
        valid = torch.ones(2, 8, dtype=torch.bool)
        rows = torch.tensor([11, 22])
        with tempfile.TemporaryDirectory() as root:
            metadata = export_hypotheses(
                Path(root), [1], predicted, probability,
                classes, valid, rows, "val",
            )
            self.assertEqual(metadata["contains_gt"], False)
            with np.load(Path(root) / metadata["files"][0]) as archive:
                self.assertEqual(set(archive.files), {
                    "pose_hypotheses", "action_probability",
                    "action_class_id", "frame_mask", "dataset_row",
                })
                self.assertEqual(archive["pose_hypotheses"].shape, (
                    5, 8, C.N_JOINTS, 3,
                ))
                self.assertAlmostEqual(
                    float(archive["action_probability"].sum()), 1.0,
                    places=6,
                )

    def test_verify_accepts_exact_artifact_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact = root / "model.pt"
            artifact.write_bytes(b"locked-model")
            manifest = {"model": "KP10", "artifacts": [{
                "path": "model.pt",
                "bytes": artifact.stat().st_size,
                "sha256": release.sha256(artifact),
            }]}
            with patch.object(release, "REPO_ROOT", root):
                self.assertEqual(release.verify(manifest)["status"], "verified")
                artifact.write_bytes(b"changed-model")
                result = release.verify(manifest)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failures"][0]["error"], "artifact_changed")

    def test_verify_reports_missing_artifact(self):
        manifest = {"model": "KP10", "artifacts": [{
            "path": "missing.pt", "bytes": 1, "sha256": "0" * 64,
        }]}
        with tempfile.TemporaryDirectory() as root:
            with patch.object(release, "REPO_ROOT", Path(root)):
                result = release.verify(manifest)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failures"][0]["error"], "missing")

    def test_release_policy_rejects_an_incomplete_artifact_set(self):
        failures = release.release_policy_failures({
            "schema_version": 1,
            "model": "KP10-ACTION-FUSED-45",
            "protocol": "single_split_lmh_e01",
            "split_counts": {
                "train": 1210, "validation": 315, "test": 315,
            },
            "selection": {
                "selection_split": "validation",
                "test_used_for_selection": False,
            },
            "simulation_extension": {
                "hypotheses": 5,
                "selection_split": "validation",
                "test_used_for_selection": False,
                "inference_hypotheses_use_gt": False,
                "coverage_metric_uses_gt": True,
            },
            "artifacts": [],
        })
        self.assertTrue(any(
            failure.get("field") == "artifact_paths"
            for failure in failures
        ))


if __name__ == "__main__":
    unittest.main()
