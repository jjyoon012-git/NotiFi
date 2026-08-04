import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from notifi_pose.tools.verify_v12_release_manifest import verify_manifest


class ReleaseManifestTests(unittest.TestCase):
    def _write_manifest(self, root: Path, artifact: dict) -> Path:
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"artifacts": [artifact]}), encoding="utf-8"
        )
        return manifest

    def test_available_artifact_hash_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "model.py"
            artifact_path.write_bytes(b"stable")
            manifest = self._write_manifest(root, {
                "path": "model.py",
                "role": "core source",
                "sha256": hashlib.sha256(b"stable").hexdigest(),
            })
            report = verify_manifest(manifest, root)
            self.assertTrue(report["passed"])
            self.assertEqual(report["verified_count"], 1)

    def test_only_known_model_artifacts_may_be_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(root, {
                "path": "best_model.pt",
                "role": "pose expert checkpoint",
                "sha256": "0" * 64,
            })
            self.assertFalse(verify_manifest(manifest, root)["passed"])
            report = verify_manifest(
                manifest, root, allow_missing_model_artifacts=True
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["missing_external_count"], 1)

    def test_hash_mismatch_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pt").write_bytes(b"changed")
            manifest = self._write_manifest(root, {
                "path": "model.pt",
                "role": "pose expert checkpoint",
                "sha256": hashlib.sha256(b"original").hexdigest(),
            })
            report = verify_manifest(
                manifest, root, allow_missing_model_artifacts=True
            )
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["mismatched"]), 1)

    def test_text_lf_hash_accepts_crlf_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(b"{\r\n}\r\n")
            manifest = self._write_manifest(root, {
                "path": "config.json",
                "role": "core source",
                "hash_mode": "text-lf",
                "sha256": hashlib.sha256(b"{\n}\n").hexdigest(),
            })
            self.assertTrue(verify_manifest(manifest, root)["passed"])


if __name__ == "__main__":
    unittest.main()
