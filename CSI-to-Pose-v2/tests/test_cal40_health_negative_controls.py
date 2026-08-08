"""CAL40 calibration negative-control transforms."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "evaluate_cal40_health_negative_controls.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_cal40_health_negative_controls", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CAL40NegativeControlTests(unittest.TestCase):
    """Verify that each corruption changes only its declared input axis."""

    def setUp(self) -> None:
        self.support = torch.arange(2 * 4 * 3 * 2).reshape(2, 4, 3, 1, 2)
        self.support_mask = torch.ones(2, 4, 3, dtype=torch.bool)
        self.labels = torch.tensor([0, 1])
        self.absence = self.support[:1].clone()
        self.absence_mask = self.support_mask[:1].clone()

    def apply(self, name: str):
        return MODULE.corrupt(
            name, self.support, self.support_mask, self.labels,
            self.absence, self.absence_mask,
        )

    def test_tx_swap_exchanges_only_first_two_links(self) -> None:
        support, mask, labels, _, _ = self.apply("tx12_swapped")

        torch.testing.assert_close(support[:, :, 0], self.support[:, :, 1])
        torch.testing.assert_close(support[:, :, 1], self.support[:, :, 0])
        torch.testing.assert_close(support[:, :, 2], self.support[:, :, 2])
        self.assertTrue(torch.equal(mask, self.support_mask))
        self.assertTrue(torch.equal(labels, self.labels))

    def test_time_reverse_flips_frame_axis(self) -> None:
        support, mask, _, absence, absence_mask = self.apply("time_reversed")

        torch.testing.assert_close(support, self.support.flip(1))
        self.assertTrue(torch.equal(mask, self.support_mask.flip(1)))
        torch.testing.assert_close(absence, self.absence.flip(1))
        self.assertTrue(torch.equal(absence_mask, self.absence_mask.flip(1)))

    def test_one_link_control_masks_every_other_link(self) -> None:
        _, support_mask, _, _, absence_mask = self.apply("one_link_only")

        self.assertTrue(support_mask[..., 0].all())
        self.assertFalse(support_mask[..., 1:].any())
        self.assertTrue(absence_mask[..., 0].all())
        self.assertFalse(absence_mask[..., 1:].any())


if __name__ == "__main__":
    unittest.main()
