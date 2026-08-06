"""Build a compact external-data cache without extracting full archives."""

from __future__ import annotations

import argparse
import json

from ..external_cache import build_mmfi_zip_cache
from ..external_data import load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mmfi",))
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--frames", type=int, default=304)
    parser.add_argument("--subcarriers", type=int, default=114)
    args = parser.parse_args()

    spec = load_registry()[args.dataset]
    spec.assert_usable("csi_pose")
    manifest = build_mmfi_zip_cache(
        args.archive,
        args.output,
        target_frames=args.frames,
        target_subcarriers=args.subcarriers,
        max_sequences=args.max_sequences,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
