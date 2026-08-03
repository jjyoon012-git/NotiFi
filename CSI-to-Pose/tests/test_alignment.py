from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from notifi_pose import contract as C
from notifi_pose.dataio.align import frame_times


def _trial(tmp_path: Path, elapsed: list[float]) -> Path:
    trial = tmp_path / "trial"
    trial.mkdir()
    pd.DataFrame({"pc_elapsed_s": elapsed}).to_csv(
        trial / "video_timestamps.csv", index=False
    )
    return trial / "csi.csv"


class AlignmentTests(unittest.TestCase):
    def test_complete_timestamps_are_used_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csi = _trial(Path(directory), [0.04, 0.073, 0.121])
            actual = frame_times(csi, 3, C.TIME_METHOD_TIMESTAMPS)
            np.testing.assert_allclose(actual, [0.04, 0.073, 0.121], atol=1e-6)

    def test_partial_timestamps_preserve_observed_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csi = _trial(Path(directory), [0.05, 0.11, 0.30])
            actual = frame_times(csi, 7, C.TIME_METHOD_UNIFORM)
            self.assertEqual(actual[0], np.float32(0.05))
            self.assertEqual(actual[-1], np.float32(0.30))
            self.assertTrue(np.all(np.diff(actual) > 0))
            self.assertFalse(np.isclose(actual[-1], 0.05 + 6.0 / 30.0))


if __name__ == "__main__":
    unittest.main()
