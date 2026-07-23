from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_trial_checksums(trial_dir: Path) -> Path:
    checksum_path = trial_dir / "checksums.sha256"
    files = sorted(
        path
        for path in trial_dir.iterdir()
        if path.is_file() and path.name != checksum_path.name
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    return checksum_path


def verify_trial_checksums(trial_dir: Path) -> list[str]:
    checksum_path = trial_dir / "checksums.sha256"
    if not checksum_path.exists():
        return ["missing_checksums"]
    problems: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, filename = line.split("  ", 1)
        except ValueError:
            problems.append("invalid_checksum_line")
            continue
        path = trial_dir / filename
        if not path.exists():
            problems.append(f"checksum_file_missing:{filename}")
        elif sha256_file(path) != expected:
            problems.append(f"checksum_mismatch:{filename}")
    return problems
