from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import savemat

from notifi_pose.external_cache import (
    ExternalCSICacheDataset,
    build_mmfi_zip_cache,
)


class MMFiZipCacheTests(unittest.TestCase):
    @staticmethod
    def _mat_bytes(offset: float) -> bytes:
        stream = BytesIO()
        values = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
        savemat(stream, {"CSIamp": values + offset})
        return stream.getvalue()

    def test_builds_cache_without_extracting_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "E01.zip"
            with ZipFile(archive, "w") as bundle:
                for action in ("A01", "A02"):
                    base = f"E01/S01/{action}"
                    bundle.writestr(
                        f"{base}/wifi-csi/frame001.mat", self._mat_bytes(0.0)
                    )
                    bundle.writestr(
                        f"{base}/wifi-csi/frame002.mat", self._mat_bytes(1.0)
                    )
                    pose = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
                    stream = BytesIO()
                    np.save(stream, pose, allow_pickle=False)
                    bundle.writestr(f"{base}/ground_truth.npy", stream.getvalue())
            output = root / "cache"
            manifest = build_mmfi_zip_cache(
                archive, output, target_frames=8, target_subcarriers=6
            )
            dataset = ExternalCSICacheDataset(output)

            self.assertEqual(manifest["sequences"], 2)
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["csi"].shape), (8, 3, 6, 2))
            self.assertEqual(tuple(sample["pose_native"].shape), (8, 17, 3))
            self.assertEqual(int(sample["action_id"]), 0)
            dataset.close()

    def test_refuses_to_overwrite_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "empty.zip"
            with ZipFile(archive, "w"):
                pass
            output = root / "cache"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                build_mmfi_zip_cache(archive, output)


if __name__ == "__main__":
    unittest.main()
