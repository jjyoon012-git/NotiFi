"""CAL40 배포 파일, 결과, README의 상호 무결성 테스트."""

import hashlib
import json
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """파일 바이트의 소문자 SHA-256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CAL40ReleaseIntegrityTests(unittest.TestCase):
    def test_sha256_manifest_matches_every_release_binary(self) -> None:
        """SHA256SUMS의 모든 파일이 존재하고 실제 hash와 일치하는지 확인한다."""
        entries = [line.split("  ", 1) for line in (
            ROOT / "SHA256SUMS"
        ).read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(entries), 13)
        for expected, relative in entries:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256(path), expected, relative)

    def test_bundle_provenance_resolves_inside_release(self) -> None:
        """bundle 생성 입력과 두 encoder checkpoint hash가 저장소 파일과 맞는지 검사한다."""
        bundle = torch.load(
            ROOT / "artifacts/cal40_full_deployment.pt",
            map_location="cpu", weights_only=False,
        )
        self.assertEqual(
            bundle["bundle_version"],
            "cal40_fixed_deep_action_safety_risk_v1",
        )
        self.assertIs(bundle["target_subject_used"], False)
        self.assertIs(bundle["sealed_yja_used"], False)
        self.assertIs(bundle["query_labels_or_pose_gt_used"], False)
        provenance = bundle["provenance"]
        expected = {
            "deployment_model_sha256": "checkpoints/cal60/deployment_model.pt",
            "calibration_result_sha256": "results/cal60_cal17_seed17017.json",
            "pose_result_sha256": "results/cal60_cal23_seed17017.json",
            "fixed_dual_result_sha256": "results/a30_cal32_safety_risk_5seed_summary.json",
            "deep_action_model_sha256": "checkpoints/cal66_grl0/deployment_model.pt",
            "fixed_deep_action_result_sha256": "results/a44_cal40_fixed_deep_action_safety_5seed.json",
        }
        for key, relative in expected.items():
            self.assertEqual(provenance[key], sha256(ROOT / relative), key)

    def test_every_checkpoint_and_pose_id_excludes_sealed_subject(self) -> None:
        """모든 배포 binary 내부 메타와 pose ID에서 봉인 대상이 빠졌는지 검사한다."""
        checkpoints = sorted((ROOT / "checkpoints").rglob("*.pt"))
        self.assertEqual(len(checkpoints), 12)
        for path in checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIs(checkpoint["sealed_yja_used"], False, str(path))
            self.assertIs(checkpoint["target_subject_used"], False, str(path))
            self.assertIs(
                checkpoint["outer_holdout_used_for_selection"], False, str(path)
            )

        bundle = torch.load(
            ROOT / "artifacts/cal40_full_deployment.pt",
            map_location="cpu", weights_only=False,
        )
        source_sites = {
            "ajh_E01", "ajh_E02", "ajh_E03", "lmh_E01",
            "mhw_E01", "mhw_E02", "mhw_E03",
        }
        self.assertEqual(set(bundle["source_sites"]), source_sites)
        for library in (
            bundle["source_library"],
            bundle["deep_action"]["source_library"],
        ):
            self.assertEqual({entry["site"] for entry in library}, source_sites)
        trial_ids = bundle["pose_library"]["trial_ids"]
        self.assertEqual(len(trial_ids), 1210)
        self.assertEqual(
            {str(trial_id).split("_")[0] for trial_id in trial_ids},
            {"ajh", "lmh", "mhw"},
        )

    def test_readme_current_metrics_match_machine_results(self) -> None:
        """README 최상단 CAL40 수치와 runtime이 결과 JSON에서 벗어나지 않게 고정한다."""
        result = json.loads((
            ROOT / "results/a44_cal40_fixed_deep_action_safety_5seed.json"
        ).read_text(encoding="utf-8"))
        runtime = json.loads((
            ROOT / "results/cal40_full_runtime_benchmark.json"
        ).read_text(encoding="utf-8"))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        checks = {
            f"{100 * result['aggregate']['action_accuracy']['mean']:.2f}": "action",
            f"{100 * result['aggregate']['action_macro_f1']['mean']:.2f}": "macro F1",
            f"{100 * result['aggregate']['danger_recall']['mean']:.2f}": "danger recall",
            f"{runtime['calibration']['median_ms']:.2f}": "calibration runtime",
            f"{runtime['classification_only']['median_ms']:.2f}": "classification runtime",
            f"{runtime['classification_and_pose']['median_ms']:.2f}": "pose runtime",
        }
        for value, name in checks.items():
            self.assertIn(value, readme, name)

    def test_health_audits_are_source_only_and_match_readme(self) -> None:
        """A52-A54 safety audit가 봉인 대상을 쓰지 않고 문서 수치와 맞는지 검사한다."""
        paths = (
            "a52_cal40_dual_geometry_health_gate.json",
            "a53_source_only_link_threshold_audit.json",
            "a54_cal40_health_negative_controls.json",
            "a55_cal40_site_reliability.json",
            "a56_cal42_safe_anchor_risk.json",
            "a57_cal43_safe_anchor_shrinkage.json",
        )
        reports = {
            name: json.loads((ROOT / "results" / name).read_text(
                encoding="utf-8"
            ))
            for name in paths
        }
        for report in reports.values():
            self.assertIs(report["target_subject_used"], False)
            self.assertIs(report["sealed_yja_used"], False)
            query_key = (
                "query_labels_or_pose_gt_used"
                if "query_labels_or_pose_gt_used" in report
                else "query_labels_or_pose_gt_at_inference"
            )
            self.assertIs(report[query_key], False)

        link_audit = reports[paths[1]]
        self.assertEqual(link_audit["basic_quality_links"], 5772)
        self.assertAlmostEqual(link_audit["quantiles"]["q03"], 0.6475499261)
        self.assertEqual(link_audit["below_threshold"], 184)

        negative = reports[paths[2]]["summary"]
        self.assertEqual(negative["prompt_labels_rolled"]["accepted"], 0)
        self.assertEqual(negative["one_link_only"]["accepted"], 0)
        self.assertEqual(negative["tx12_swapped"]["accepted"], 6)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("5,772", readme)
        self.assertIn("TX 교환 6/35", readme)
        cal42 = reports[paths[4]]["aggregate"]
        self.assertAlmostEqual(
            cal42["cal42"]["danger_recall"]["mean"], 0.6722282981
        )
        self.assertAlmostEqual(
            cal42["cal42"]["safe_to_danger_rate"]["mean"], 0.3269303202
        )
        self.assertIn("52.45→67.22%", readme)
        self.assertIn("18.68→32.69%", readme)
        cal43 = reports[paths[5]]["aggregate"]["cal43"]
        self.assertAlmostEqual(cal43["risk_macro_f1"]["mean"], 0.4430269289)
        self.assertAlmostEqual(cal43["danger_recall"]["mean"], 0.4340072922)
        self.assertAlmostEqual(
            cal43["safe_to_danger_rate"]["mean"], 0.1190207156
        )
        self.assertIn("52.45→43.40%", readme)

    def test_cal40_confusion_diagnosis_matches_current_interpretation(self) -> None:
        """A58이 봉인 대상을 제외하고 README의 hard-class 결론과 맞는지 검사한다."""
        report = json.loads((
            ROOT / "results/a58_cal40_confusion_diagnosis.json"
        ).read_text(encoding="utf-8"))
        for key in (
            "target_subject_used", "sealed_yja_used",
            "query_labels_or_pose_gt_at_inference",
            "outer_holdout_used_for_selection",
        ):
            self.assertIs(report[key], False)
        recall = report["action_summary"]["class_recall"]
        self.assertAlmostEqual(recall["lie_to_stand"], 0.6485714316)
        self.assertAlmostEqual(recall["fall_while_walking"], 0.0047619049)
        self.assertAlmostEqual(recall["bed_exit_failed"], 0.0067226892)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("fall_while_walking` recall은 0.48%", readme)
        self.assertIn("bed_exit_failed`는 0.67%", readme)
        balance = json.loads((
            ROOT / "results/a59_training_class_balance_audit.json"
        ).read_text(encoding="utf-8"))
        self.assertIs(balance["sealed_yja_used"], False)
        self.assertIs(balance["target_subject_used"], False)
        self.assertEqual(balance["absence_in_query_sampler"], 0)
        self.assertAlmostEqual(balance["raw_max_to_min_ratio"], 10.0 / 3.0)
        self.assertLess(balance["pooled_max_to_min_ratio"], 1.5)
        self.assertIn("3.33배→sampler 1.42배", readme)

        hard_pair = json.loads((
            ROOT / "results/a60_cal68_hard_pair_source_loso.json"
        ).read_text(encoding="utf-8"))
        self.assertIs(hard_pair["sealed_yja_used"], False)
        self.assertIs(hard_pair["target_subject_used"], False)
        self.assertAlmostEqual(
            hard_pair["site_macro"]["action_accuracy"], 0.3635297801
        )
        self.assertAlmostEqual(
            hard_pair["danger_recall_pooled"], 0.4190476190
        )
        self.assertIn("A60 / CAL68", readme)
        self.assertIn("옵션 코드도 제거", readme)

        calibrated = json.loads((
            ROOT / "results/a61_cal68_hard_pair_cal17_seed17017.json"
        ).read_text(encoding="utf-8"))
        self.assertIs(calibrated["sealed_yja_used"], False)
        self.assertIs(calibrated["target_subject_used"], False)
        self.assertIs(
            calibrated["query_labels_or_pose_gt_at_inference"], False
        )
        outer = [
            row
            for fold in calibrated["folds"].values()
            for row in fold["outer_metrics"]
        ]
        self.assertAlmostEqual(
            sum(row["action_accuracy"] for row in outer) / len(outer),
            0.3844345590,
        )
        self.assertIn("A61 / CAL68+CAL17", readme)
        self.assertIn("40.81→38.44%", readme)


if __name__ == "__main__":
    unittest.main()
