"""Audit the immutable seen-domain split used by KP10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import contract as C


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=Path,
        default=C.WORK_ROOT / "splits" / "dev_index.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "split_integrity.json",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.index)
    selected = frame[
        ((frame.subject.isin(("ajh", "mhw")))
         & frame.environment.isin(("E01", "E02", "E03")))
        | ((frame.subject == "lmh") & (frame.environment == "E01"))
    ].copy()
    roles = ("train", "val", "test")
    trial_sets = {
        role: set(selected.loc[selected.role == role, "trial_id"])
        for role in roles
    }
    raw_counts = {role: len(trial_sets[role]) for role in roles}
    pose = selected[selected.gt_pose.notna()]
    pose_counts = {
        role: int((pose.role == role).sum()) for role in roles
    }
    intersections = {
        "train_val": len(trial_sets["train"] & trial_sets["val"]),
        "train_test": len(trial_sets["train"] & trial_sets["test"]),
        "val_test": len(trial_sets["val"] & trial_sets["test"]),
    }
    duplicate_paths = {
        column: int(selected[column].dropna().duplicated().sum())
        for column in ("csi", "gt_pose", "original_video")
    }
    missing_gt = selected[selected.gt_pose.isna()]
    missing_gt_is_expected_absence = bool(
        (missing_gt.task == "classification_only").all()
        and (missing_gt.detail_label == "absence").all()
    )
    passed = bool(
        pose_counts == {"train": 1210, "val": 315, "test": 315}
        and not any(intersections.values())
        and int(selected.trial_id.duplicated().sum()) == 0
        and not any(duplicate_paths.values())
        and len(missing_gt) == 84
        and missing_gt_is_expected_absence
    )
    result = {
        "status": "passed" if passed else "failed",
        "protocol": "single_split_lmh_e01",
        "test_used_for_selection": False,
        "raw_counts": raw_counts,
        "pose_counts": pose_counts,
        "role_intersections": intersections,
        "duplicate_trial_ids": int(selected.trial_id.duplicated().sum()),
        "duplicate_paths": duplicate_paths,
        "missing_gt_trials": int(len(missing_gt)),
        "missing_gt_is_classification_only_absence": missing_gt_is_expected_absence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
