"""Create a deterministic source-and-artifact release ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "runtime",
    "outputs",
}

EXCLUDED_FILES = {
    "build_artifacts.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root.parent / "NotiFi_AI_v1_release.zip"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.name not in EXCLUDED_FILES
        and path.suffix != ".zip"
    )
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("NotiFi_AI_v1") / path.relative_to(root))
    print(f"{output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
