"""Public action, risk, and calibration label contract tests."""

from __future__ import annotations

import unittest

from notifi_pose import contract as C


class LabelContractTests(unittest.TestCase):
    """Keep output IDs and field-calibration prompts mutually consistent."""

    def test_action_and_risk_maps_are_bijective(self) -> None:
        self.assertEqual(len(C.ACTION_NAMES), C.N_CLASSES)
        self.assertEqual(len(C.ACTION_TO_ID), C.N_CLASSES)
        self.assertEqual(len(C.RISK_NAMES), C.N_RISK)
        self.assertEqual(len(C.RISK_TO_ID), C.N_RISK)
        for index, name in enumerate(C.ACTION_NAMES):
            self.assertEqual(C.ACTION_TO_ID[name], index)
        for index, name in enumerate(C.RISK_NAMES):
            self.assertEqual(C.RISK_TO_ID[name], index)

    def test_prompt_classes_are_safe_nonabsence_actions(self) -> None:
        self.assertEqual(C.CALIBRATION_PROMPT_CLASSES, (0, 1, 2, 3, 4, 5, 7, 8))
        self.assertNotIn(C.ACTION_TO_ID["absence"], C.CALIBRATION_PROMPT_CLASSES)
        self.assertTrue(all(class_id < 9 for class_id in C.CALIBRATION_PROMPT_CLASSES))
        self.assertEqual(C.DANGER_CALIBRATION_CLASSES, (12, 13, 14, 15, 16))


if __name__ == "__main__":
    unittest.main()
