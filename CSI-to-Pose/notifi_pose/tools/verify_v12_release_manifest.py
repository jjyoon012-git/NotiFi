"""Verify V12 release artifact hashes and distinguish external model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXTERNAL_MODEL_ROLES = frozenset({
    "training cache contract",
    "coarse P2 checkpoint",
    "classification expert checkpoint",
    "link-failure expert checkpoint",
    "V12RG root expert checkpoint",
    "root expert checkpoint",
    "pose expert checkpoint",
})


def _sha256(path: Path, mode: str = "raw") -> str:
    digest = hashlib.sha256()
    if mode == "text-lf":
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(
    manifest_path: Path,
    root: Path,
    allow_missing_model_artifacts: bool = False,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = []
    missing_required = []
    missing_external = []
    mismatched = []
    for artifact in manifest["artifacts"]:
        relative_path = artifact["path"]
        path = root / Path(relative_path)
        if not path.is_file():
            item = {"path": relative_path, "role": artifact["role"]}
            if (
                allow_missing_model_artifacts
                and artifact["role"] in EXTERNAL_MODEL_ROLES
            ):
                missing_external.append(item)
            else:
                missing_required.append(item)
            continue
        actual = _sha256(path, artifact.get("hash_mode", "raw"))
        if actual == artifact["sha256"]:
            verified.append(relative_path)
        else:
            mismatched.append({
                "path": relative_path,
                "expected": artifact["sha256"],
                "actual": actual,
            })
    return {
        "manifest": manifest_path.as_posix(),
        "passed": not missing_required and not mismatched,
        "artifact_count": len(manifest["artifacts"]),
        "verified_count": len(verified),
        "missing_external_count": len(missing_external),
        "missing_required": missing_required,
        "missing_external": missing_external,
        "mismatched": mismatched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("docs/results/v12_release_manifest.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--allow-missing-model-artifacts", action="store_true")
    args = parser.parse_args()
    report = verify_manifest(
        args.manifest,
        args.root,
        allow_missing_model_artifacts=args.allow_missing_model_artifacts,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
