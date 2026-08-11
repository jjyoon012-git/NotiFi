from __future__ import annotations

import unittest

import numpy as np

from notifi_ai.preprocessing import pad_or_trim


class PreprocessingTest(unittest.TestCase):
    def test_pad_and_mask(self):
        csi = np.ones((20, 3, 114, 2), np.float32)
        mask = np.ones((20, 3), bool)
        output, output_mask = pad_or_trim(csi, mask)
        self.assertEqual(output.shape, (304, 3, 114, 2))
        self.assertEqual(output_mask.shape, (304, 3))
        self.assertTrue(output_mask[:20].all())
        self.assertFalse(output_mask[20:].any())
        self.assertEqual(float(output[20:].sum()), 0.0)

    def test_rejects_wrong_subcarrier_count(self):
        with self.assertRaises(ValueError):
            pad_or_trim(
                np.zeros((10, 3, 100, 2), np.float32),
                np.ones((10, 3), bool),
            )


if __name__ == "__main__":
    unittest.main()

