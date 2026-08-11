from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class ReleaseTest(unittest.TestCase):
    def test_artifact_manifest(self):
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "artifacts" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_name"], "NotiFi_AI_v1")
        for item in manifest["artifacts"]:
            path = manifest_path.parent / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"])


if __name__ == "__main__":
    unittest.main()

