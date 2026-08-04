"""Audit V12 split disjointness, exclusions, locks, and file fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .build_splits import verify


SUBJECT_ENVIRONMENTS = {
    "ajh": {"E01", "E02", "E03"},
    "mhw": {"E01", "E02", "E03"},
    "lmh": {"E01"},
}
EXPECTED_COUNTS = {"train": 1266, "val": 329, "test": 329}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_rows(index: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(False, index=index.index)
    for subject, environments in SUBJECT_ENVIRONMENTS.items():
        keep |= (
            index["subject"].eq(subject)
            & index["environment"].isin(environments)
        )
    return index[keep & index["role"].isin(EXPECTED_COUNTS)].copy()


def _path_values(rows: pd.DataFrame, column: str) -> set[str]:
    values = rows[column].dropna().astype(str)
    return {value for value in values if value and value.lower() != "nan"}


def _calibration_check(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    protocol = value.get("protocol")
    selection_split = value.get("selection_split")
    flags = {
        key: value[key]
        for key in ("test_used", "test_used_for_selection", "test_opened")
        if key in value
    }
    safe = protocol == "single_split_lmh_e01"
    safe &= flags.get("test_used", False) is False
    safe &= flags.get("test_used_for_selection", False) is False
    safe &= flags.get("test_opened", False) is False
    return {
        "path": path.as_posix(),
        "protocol": protocol,
        "selection_split": selection_split,
        "test_flags": flags,
        "safe": bool(safe),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-index", type=Path,
        default=Path("work_v2/splits/dev_index.csv"),
    )
    parser.add_argument(
        "--sealed-index", type=Path,
        default=Path("work_v2/splits/sealed_index.csv"),
    )
    parser.add_argument(
        "--experiments", type=Path,
        default=Path("work_v2/splits/experiments.json"),
    )
    parser.add_argument(
        "--calibrations", type=Path, nargs="+",
        default=(
            Path("work_v2/runs/p2_v12aa_pose_seed23_w30/validation.json"),
            Path("docs/results/v12_shift_robust_root_calibration.json"),
            Path("work_v2/runs/p2_v12w_robust_classification_ensemble/validation.json"),
            Path("docs/results/v12_link_failure_pose_calibration.json"),
            Path("docs/results/v12_link_failure_root_calibration.json"),
            Path("docs/results/v12_link_specific_classification_calibration.json"),
        ),
    )
    parser.add_argument(
        "--final-evaluation", type=Path,
        default=Path("docs/results/v12_final_evaluation.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/v12_protocol_integrity.json"),
    )
    args = parser.parse_args()

    dev = pd.read_csv(args.dev_index)
    sealed = pd.read_csv(args.sealed_index)
    protocol = _protocol_rows(dev)
    split_rows = {
        split: protocol[protocol["role"].eq(split)]
        for split in EXPECTED_COUNTS
    }
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    counts = {split: len(rows) for split, rows in split_rows.items()}
    check("protocol_counts", counts == EXPECTED_COUNTS, counts)
    check(
        "trial_ids_unique", not protocol["trial_id"].duplicated().any(),
        {"trials": len(protocol)},
    )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(split_rows[left]["trial_id"]) & set(split_rows[right]["trial_id"])
        check(f"trial_disjoint_{left}_{right}", not overlap, sorted(overlap)[:10])
        for column in ("csi", "gt_pose", "original_video"):
            paths = _path_values(split_rows[left], column) & _path_values(
                split_rows[right], column
            )
            check(
                f"{column}_disjoint_{left}_{right}", not paths,
                sorted(paths)[:10],
            )
    environments = {
        subject: sorted(set(rows["environment"]))
        for subject, rows in protocol.groupby("subject")
    }
    check(
        "subject_environment_contract",
        environments == {
            subject: sorted(values)
            for subject, values in SUBJECT_ENVIRONMENTS.items()
        },
        environments,
    )
    check("yja_excluded", "yja" not in set(protocol["subject"]), environments)
    check(
        "lmh_e02_e03_excluded",
        not set(protocol.loc[protocol["subject"].eq("lmh"), "environment"])
        & {"E02", "E03"},
        environments.get("lmh", []),
    )

    calibration_checks = [_calibration_check(path) for path in args.calibrations]
    check(
        "calibrations_validation_only",
        all(item["safe"] for item in calibration_checks),
        calibration_checks,
    )
    final = json.loads(args.final_evaluation.read_text(encoding="utf-8"))
    check(
        "final_test_opened_after_lock",
        final.get("protocol") == "single_split_lmh_e01"
        and final.get("test_opened") is True
        and final.get("selection_split") == "validation"
        and all(item["safe"] for item in calibration_checks),
        {
            "protocol": final.get("protocol"),
            "test_opened": final.get("test_opened"),
            "selection_split": final.get("selection_split"),
            "all_calibration_locks_safe": all(
                item["safe"] for item in calibration_checks
            ),
        },
    )

    base_verification = verify(dev, sealed)
    check(
        "base_split_verifier",
        all(item["ok"] for item in base_verification["checks"]),
        base_verification["checks"],
    )
    report = {
        "protocol": "single_split_lmh_e01",
        "passed": all(item["passed"] for item in checks),
        "counts": counts,
        "time_methods": {
            str(key): int(value)
            for key, value in protocol["time_method"].value_counts().items()
        },
        "fingerprints": {
            path.as_posix(): _sha256(path)
            for path in (args.dev_index, args.sealed_index, args.experiments)
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": report["passed"],
        "checks": len(checks),
        "counts": counts,
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
