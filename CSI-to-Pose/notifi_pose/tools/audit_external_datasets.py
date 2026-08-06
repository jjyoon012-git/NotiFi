"""Audit external dataset gates and locally available data.

Example:
    python -m notifi_pose.tools.audit_external_datasets \
      --root up_fall_3d=C:/path/to/UP-Fall-3D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..external_data import load_registry, summarize_upfall


def _parse_roots(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"root must be DATASET_ID=PATH, got {value!r}")
        dataset_id, path = value.split("=", 1)
        output[dataset_id] = Path(path).expanduser()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--json", default=None, help="optional audit output path")
    args = parser.parse_args()

    registry = load_registry(args.registry) if args.registry else load_registry()
    roots = _parse_roots(args.root)
    report = {"datasets": []}
    for dataset_id, spec in registry.items():
        root = roots.get(dataset_id)
        row = {
            "id": dataset_id,
            "enabled": spec.enabled,
            "license": spec.license,
            "license_verified": spec.license_verified,
            "roles": list(spec.roles),
            "root": str(root) if root else None,
            "local": bool(root and root.exists()),
        }
        if dataset_id == "up_fall_3d" and row["local"]:
            row["summary"] = summarize_upfall(root)
        report["datasets"].append(row)
        state = "READY" if row["local"] and spec.enabled else "GATED" if not spec.enabled else "MISSING"
        license_state = "verified" if spec.license_verified else "review-required"
        print(f"{dataset_id:22s} {state:7s} {license_state:15s} {','.join(spec.roles)}")
        if "summary" in row:
            print(f"  {row['summary']}")

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
