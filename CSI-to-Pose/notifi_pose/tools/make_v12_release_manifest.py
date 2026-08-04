"""Hash the V12 data contract, checkpoints, calibrations, and core source."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TEXT_SUFFIXES = frozenset({
    ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml",
})


def _hash_mode(path: Path) -> str:
    return "text-lf" if path.suffix.lower() in TEXT_SUFFIXES else "raw"


def _sha256(path: Path, mode: str) -> str:
    digest = hashlib.sha256()
    if mode == "text-lf":
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path, role: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    mode = _hash_mode(path)
    return {
        "role": role,
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "hash_mode": mode,
        "sha256": _sha256(path, mode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-evaluation", type=Path,
        default=Path("docs/results/v12_final_evaluation.json"),
    )
    parser.add_argument(
        "--guard-calibrations", type=Path, nargs="+",
        default=(
            Path("docs/results/v12_link_failure_pose_calibration.json"),
            Path("docs/results/v12_link_failure_root_calibration.json"),
            Path("docs/results/v12_link_specific_classification_calibration.json"),
        ),
    )
    parser.add_argument(
        "--candidate-root-calibration", type=Path,
        default=Path("docs/results/v12_shift_robust_root_calibration.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/v12_release_manifest.json"),
    )
    args = parser.parse_args()

    final = json.loads(args.final_evaluation.read_text(encoding="utf-8"))
    configuration = final["configuration"]
    paths: dict[Path, str] = {
        args.final_evaluation: "locked final evaluation",
        Path("docs/results/v12_pose_root_calibration.json"): "root validation lock",
        Path("docs/results/v12_robust_classification_lock.json"):
            "classification validation lock",
        Path("docs/results/v12_final_summary.json"): "final evaluation summary",
        Path(configuration["p2_checkpoint"]): "coarse P2 checkpoint",
        Path("work_v2/splits/experiments.json"): "split protocol",
        Path("work_v2/splits/dev_index.csv"): "development split index",
        Path("work_v2/splits/sealed_index.csv"): "sealed split index",
    }
    for path in configuration["pose_checkpoints"]:
        paths[Path(path)] = "pose expert checkpoint"
    for path in configuration["root_checkpoints"]:
        paths[Path(path)] = "root expert checkpoint"
    for path in configuration["classification_expert_checkpoints"]:
        paths[Path(path)] = "classification expert checkpoint"
    for path in Path("work_v2/cache").iterdir():
        if path.is_file():
            paths[path] = "training cache contract"
    for calibration_path in args.guard_calibrations:
        lock = json.loads(calibration_path.read_text(encoding="utf-8"))
        paths[calibration_path] = "link-failure validation lock"
        paths[Path(lock["expert_checkpoint"])] = "link-failure expert checkpoint"
    candidate_root = json.loads(
        args.candidate_root_calibration.read_text(encoding="utf-8")
    )
    paths[args.candidate_root_calibration] = "V12RG root validation lock"
    for checkpoint in candidate_root["source"]["root_checkpoints"]:
        paths[Path(checkpoint)] = "V12RG root expert checkpoint"
    for path in (
        Path("notifi_pose/hybrid_v10.py"),
        Path("notifi_pose/dataio/dataset.py"),
        Path("notifi_pose/tools/train_p2_v9_hybrid.py"),
        Path("notifi_pose/tools/evaluate_v12_final.py"),
        Path("notifi_pose/tools/audit_v12_link_failure_guard.py"),
        Path("notifi_pose/tools/audit_v12_protocol_integrity.py"),
        Path("notifi_pose/tools/audit_v12_alignment_strata.py"),
        Path("notifi_pose/tools/verify_v12_release_manifest.py"),
        Path("docs/results/v12_protocol_integrity.json"),
    ):
        paths[path] = "core source"

    artifacts = [
        _entry(path, role)
        for path, role in sorted(paths.items(), key=lambda item: item[0].as_posix())
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": final["protocol"],
        "test_opened_once": final.get("test_opened") is True,
        "artifact_count": len(artifacts),
        "total_bytes_hashed": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "artifact_count": report["artifact_count"],
        "total_bytes_hashed": report["total_bytes_hashed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
