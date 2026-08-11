"""Verify artifact hashes and load every public runtime component."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notifi_ai import NotiFiAIv1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = ROOT
    manifest_path = root / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = manifest_path.parent / item["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"artifact size mismatch: {path}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {path}")
    model = NotiFiAIv1(root / "artifacts", device=args.device)
    if model.describe()["model_name"] != "NotiFi_AI_v1":
        raise RuntimeError("public model name mismatch")
    if args.smoke:
        csi = np.zeros((304, 3, 114, 2), dtype=np.float32)
        mask = np.ones((304, 3), dtype=bool)
        prediction = model.predict(csi, mask)
        if prediction.pose_rel.shape != (304, 22, 3):
            raise RuntimeError("unexpected pose output shape")
        if not np.isfinite(prediction.pose_rel).all():
            raise RuntimeError("non-finite pose output")
    print(f"verified {len(manifest['artifacts'])} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
