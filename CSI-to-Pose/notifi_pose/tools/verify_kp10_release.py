"""Create or verify the immutable artifact manifest for KP10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .. import contract as C


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    C.WORK_ROOT / "runs" / "kp10_action_strength"
    / "release_manifest.json"
)
LOCKED_ARTIFACTS = (
    "work_v2/runs/kp4_dcc_staged_seed17/deployment_model.pt",
    "work_v2/runs/kp1_v13s_coarse_single_split_lmh_e01.pt",
    "work_v2/runs/kp5_mpr_selector_seed17/best_model.pt",
    "work_v2/runs/kp5_mpr_reranker_seed17/best_model.pt",
    "work_v2/runs/kp5_motion_profile_seed79/best_model.pt",
    "work_v2/runs/kp5_part_motion_profile_seed101/best_model.pt",
    "work_v2/runs/kp8_profile_candidate_ranker_seed127/best_model.pt",
    "work_v2/runs/kp10_action_classifier_seed181/best_model.pt",
    "work_v2/runs/kp5_part_motion_profile_seed83/reranking_calibration.json",
    "work_v2/runs/kp5_motion_warp/warping_calibration.json",
    "work_v2/runs/kp6_semantic_prior/calibration.json",
    "work_v2/runs/kp6_risk_adaptive_blend/calibration.json",
    "work_v2/runs/kp10_action_classifier_pose/calibration.json",
    "work_v2/runs/kp10_action_strength/calibration.json",
    "work_v2/runs/kp10_action_strength/test_fixed.json",
    "work_v2/runs/kp10_action_strength/paired_bootstrap_audit.json",
    "work_v2/runs/kp10_action_strength/target_invariance_validation.json",
    "work_v2/runs/kp10_action_strength/split_integrity.json",
    "work_v2/runs/kp28_hypothesis_coverage_top5/validation.json",
    "work_v2/runs/kp28_hypothesis_coverage_top5/test_fixed.json",
)
LOCKED_TEST_METRICS = {
    "pose_cm": 12.885332910255306,
    "distal_cm": 18.665696063211985,
    "danger_pose_cm": 19.829331072313444,
    "danger_distal_cm": 29.249977852616993,
}
LOCKED_MULTI5_COVERAGE = {
    "pose_cm": 11.889958301825184,
    "danger_pose_cm": 18.10176589127098,
    "danger_distal_cm": 26.646802233798162,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_row(relative: str) -> dict:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_manifest() -> dict:
    test_result = json.loads(
        (C.WORK_ROOT / "runs" / "kp10_action_strength"
         / "test_fixed.json").read_text(encoding="utf-8")
    )
    fixed_name = test_result["fixed_configuration"]["name"]
    selected = test_result["metrics"][f"kp10_{fixed_name}"]
    multi_result = json.loads(
        (C.WORK_ROOT / "runs" / "kp28_hypothesis_coverage_top5"
         / "test_fixed.json").read_text(encoding="utf-8")
    )
    multi = multi_result["metrics"]["best_of_5_oracle_coverage"]
    return {
        "schema_version": 1,
        "model": "KP10-ACTION-FUSED-45",
        "performance_code_commit": "50ff4dc",
        "protocol": "single_split_lmh_e01",
        "split_counts": {"train": 1210, "validation": 315, "test": 315},
        "inference_inputs": ["4-board CSI", "link mask"],
        "forbidden_inference_inputs": [
            "GT pose", "action/risk label", "video", "subject ID",
            "environment ID",
        ],
        "selection": {
            "action_fusion": "1.50 * base + 0.75 * independent",
            "motion_strength": 0.45,
            "selection_split": "validation",
            "test_used_for_selection": False,
        },
        "fixed_test_metrics": {
            "pose_cm": selected["mpjpe_m"] * 100,
            "distal_cm": selected["distal_mpjpe_m"] * 100,
            "danger_pose_cm": selected["danger_pose_mpjpe_m"] * 100,
            "danger_distal_cm": selected["danger_distal_mpjpe_m"] * 100,
        },
        "simulation_extension": {
            "model": "KP28-MULTI5",
            "hypotheses": 5,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "inference_hypotheses_use_gt": False,
            "coverage_metric_uses_gt": True,
            "fixed_test_best_of_5_coverage": {
                "pose_cm": multi["mpjpe_m"] * 100,
                "danger_pose_cm": multi["danger_pose_mpjpe_m"] * 100,
                "danger_distal_cm": multi["danger_distal_mpjpe_m"] * 100,
            },
            "warning": "coverage is not an automatically selected point estimate",
        },
        "artifacts": [artifact_row(path) for path in LOCKED_ARTIFACTS],
    }


def release_policy_failures(manifest: dict) -> list[dict]:
    failures = []

    def require(name, actual, expected):
        if actual != expected:
            failures.append({
                "error": "release_policy_violation",
                "field": name,
                "expected": expected,
                "actual": actual,
            })

    require("schema_version", manifest.get("schema_version"), 1)
    require("model", manifest.get("model"), "KP10-ACTION-FUSED-45")
    require("protocol", manifest.get("protocol"), "single_split_lmh_e01")
    require(
        "split_counts", manifest.get("split_counts"),
        {"train": 1210, "validation": 315, "test": 315},
    )
    require(
        "artifact_paths",
        [item.get("path") for item in manifest.get("artifacts", [])],
        list(LOCKED_ARTIFACTS),
    )
    selection = manifest.get("selection", {})
    require(
        "selection.action_fusion", selection.get("action_fusion"),
        "1.50 * base + 0.75 * independent",
    )
    require("selection.motion_strength", selection.get("motion_strength"), 0.45)
    require("selection.selection_split", selection.get("selection_split"), "validation")
    require(
        "selection.test_used_for_selection",
        selection.get("test_used_for_selection"), False,
    )
    simulation = manifest.get("simulation_extension", {})
    require("simulation.hypotheses", simulation.get("hypotheses"), 5)
    require(
        "simulation.selection_split",
        simulation.get("selection_split"), "validation",
    )
    require(
        "simulation.test_used_for_selection",
        simulation.get("test_used_for_selection"), False,
    )
    require(
        "simulation.inference_hypotheses_use_gt",
        simulation.get("inference_hypotheses_use_gt"), False,
    )
    require(
        "simulation.coverage_metric_uses_gt",
        simulation.get("coverage_metric_uses_gt"), True,
    )
    require(
        "fixed_test_metrics", manifest.get("fixed_test_metrics"),
        LOCKED_TEST_METRICS,
    )
    require(
        "simulation.fixed_test_best_of_5_coverage",
        simulation.get("fixed_test_best_of_5_coverage"),
        LOCKED_MULTI5_COVERAGE,
    )

    json_policies = (
        (
            "work_v2/runs/kp10_action_strength/calibration.json",
            {"test_used_for_selection": False},
        ),
        (
            "work_v2/runs/kp10_action_strength/test_fixed.json",
            {"test_used_for_selection": False},
        ),
        (
            "work_v2/runs/kp10_action_strength/target_invariance_validation.json",
            {
                "test_split_touched": False,
                "test_used_for_selection": False,
                "exact_prediction_equality": True,
                "maximum_absolute_pose_difference": 0.0,
            },
        ),
        (
            "work_v2/runs/kp10_action_strength/split_integrity.json",
            {
                "status": "passed",
                "test_used_for_selection": False,
                "duplicate_trial_ids": 0,
                "missing_gt_is_classification_only_absence": True,
            },
        ),
        (
            "work_v2/runs/kp28_hypothesis_coverage_top5/validation.json",
            {
                "test_used_for_selection": False,
                "inference_hypotheses_use_gt": False,
                "coverage_metric_uses_gt": True,
            },
        ),
        (
            "work_v2/runs/kp28_hypothesis_coverage_top5/test_fixed.json",
            {
                "test_used_for_selection": False,
                "inference_hypotheses_use_gt": False,
                "coverage_metric_uses_gt": True,
                "gt_selects_deployment_hypothesis": False,
            },
        ),
    )
    for relative, expected_fields in json_policies:
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append({
                "error": "invalid_policy_artifact",
                "path": relative,
                "detail": str(error),
            })
            continue
        for field, expected in expected_fields.items():
            require(f"{relative}:{field}", payload.get(field), expected)
    return failures


def verify(manifest: dict) -> dict:
    failures = (
        release_policy_failures(manifest)
        if "schema_version" in manifest else []
    )
    for expected in manifest.get("artifacts", []):
        path = REPO_ROOT / expected["path"]
        if not path.is_file():
            failures.append({"path": expected["path"], "error": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != expected["bytes"] or actual_hash != expected["sha256"]:
            failures.append({
                "path": expected["path"],
                "error": "artifact_changed",
                "expected_bytes": expected["bytes"],
                "actual_bytes": actual_size,
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual_hash,
            })
    return {
        "status": "verified" if not failures else "failed",
        "model": manifest.get("model"),
        "checked_artifacts": len(manifest.get("artifacts", [])),
        "release_policy_checked": "schema_version" in manifest,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.write:
        if args.manifest.exists() and not args.force:
            raise FileExistsError(
                f"refusing to overwrite locked manifest: {args.manifest}"
            )
        manifest = build_manifest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "written", "manifest": str(args.manifest),
            "artifacts": len(manifest["artifacts"]),
        }, indent=2))
        return
    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"release manifest does not exist: {args.manifest}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = verify(manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
