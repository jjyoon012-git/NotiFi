import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from notifi_pose.external_data import (
    _read_upfall_csv,
    canonicalize_csi,
    load_registry,
    summarize_upfall,
)


class ExternalRegistryTests(unittest.TestCase):
    def test_registry_enforces_task_compatibility(self) -> None:
        registry = load_registry()
        registry["mmfi"].assert_usable("csi_pose")
        with self.assertRaises(RuntimeError):
            registry["mmfi"].assert_usable("contact_prior")
        with self.assertRaises(RuntimeError):
            registry["bodycontact4d"].assert_usable("contact_prior")

    def test_unverified_data_requires_explicit_override(self) -> None:
        upfall = load_registry()["up_fall_3d"]
        with self.assertRaises(RuntimeError):
            upfall.assert_usable("impact_prior")
        upfall.assert_usable("impact_prior", allow_unverified=True)


class CanonicalCSITests(unittest.TestCase):
    def test_mmfi_preserves_three_links_and_removes_static_gain(self) -> None:
        rng = np.random.default_rng(4)
        source = 5.0 + rng.random((3, 3, 114, 10), dtype=np.float32)
        view = canonicalize_csi(
            source, "mmfi", target_frames=24, target_subcarriers=32
        )
        self.assertEqual(view.values.shape, (24, 3, 32, 2))
        self.assertEqual(view.link_mask.shape, (24, 3))
        self.assertTrue(view.link_mask.all())
        self.assertLess(abs(float(np.median(view.values[..., 0]))), 1e-5)

    def test_person_in_wifi_keeps_nine_links(self) -> None:
        rng = np.random.default_rng(5)
        source = (
            rng.normal(size=(3, 3, 30, 20))
            + 1j * rng.normal(size=(3, 3, 30, 20))
        )
        view = canonicalize_csi(
            source, "person_in_wifi_3d", target_frames=16
        )
        self.assertEqual(view.values.shape, (16, 9, 114, 2))

    def test_csi_bench_requires_declared_link_count(self) -> None:
        source = np.arange(10 * 232, dtype=np.float32).reshape(10, 232, 1)
        view = canonicalize_csi(
            source, "csi_bench", flattened_links=4, target_frames=12
        )
        self.assertEqual(view.values.shape, (12, 4, 114, 2))
        with self.assertRaises(ValueError):
            canonicalize_csi(source, "csi_bench")


class UPFallLoaderTests(unittest.TestCase):
    @staticmethod
    def _csv_bytes(frames: int = 4) -> bytes:
        columns = {
            f"Joint{joint}_{axis}": np.linspace(0.1, 0.4, frames)
            for joint in range(1, 34)
            for axis in ("X", "Y", "Z")
        }
        columns["LABEL"] = [0, 1, 1, 0]
        return pd.DataFrame(columns).to_csv(index=False).encode("utf-8")

    def test_read_and_phase_targets(self) -> None:
        sample = _read_upfall_csv(self._csv_bytes(), "C2S1_A4_T3.csv")
        self.assertEqual(sample.pose.shape, (4, 33, 3))
        self.assertTrue(sample.is_fall)
        self.assertEqual(sample.impact_index, 1)
        np.testing.assert_array_equal(sample.phase_targets(), [0, 1, 1, 2])
        self.assertEqual(sample.root_relative_pose().shape, sample.pose.shape)

    def test_archive_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "SUBJECT1.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("C1S1A1T1.csv", self._csv_bytes())
                bundle.writestr("C1S1A7T1.csv", self._csv_bytes())
            summary = summarize_upfall(directory)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["fall_samples"], 1)
        self.assertEqual(summary["impact_labeled_samples"], 2)
        self.assertEqual(summary["skipped_schema"], 0)


if __name__ == "__main__":
    unittest.main()
