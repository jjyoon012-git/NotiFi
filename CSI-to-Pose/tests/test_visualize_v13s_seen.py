import unittest

import numpy as np
import pandas as pd

from notifi_pose.tools.visualize_kp10_pose import deterministic_representatives
from notifi_pose.tools.visualize_v13s_seen import (
    IsometricProjector,
    _align_vectors,
    _rotation_between,
)


class V13SSeenVisualizationTests(unittest.TestCase):
    def test_kp10_representatives_use_trial_order_not_pose_error(self):
        index = pd.DataFrame({
            "trial_id": ("w_c", "w_a", "w_b", "d_b", "d_c", "d_a"),
            "class_id": (0, 0, 0, 13, 13, 13),
        })
        self.assertEqual(deterministic_representatives(index), [2, 3])

    def test_rotation_between_maps_source_direction(self):
        source = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        target = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        rotation = _rotation_between(source, target)
        self.assertTrue(np.allclose(rotation @ source, target, atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=5)

    def test_multi_vector_alignment_is_proper_rotation(self):
        source = np.eye(3, dtype=np.float32)
        expected = _rotation_between(
            np.array((1.0, 0.0, 0.0), dtype=np.float32),
            np.array((0.0, 0.0, 1.0), dtype=np.float32),
        )
        target = (expected @ source.T).T
        actual = _align_vectors(source, target)
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.det(actual)), 1.0, places=5)

    def test_projector_returns_finite_screen_coordinates(self):
        trajectory = np.array((
            ((0.0, 0.0, 0.0), (0.0, 1.7, 0.0)),
            ((1.0, 0.0, 0.5), (1.0, 1.7, 0.5)),
        ), dtype=np.float32)
        projected = IsometricProjector(trajectory)(trajectory)
        self.assertEqual(projected.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(projected).all())


if __name__ == "__main__":
    unittest.main()
