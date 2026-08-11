from __future__ import annotations

import unittest

from notifi_ai.schemas import DeviceConfig


class DeviceConfigTest(unittest.TestCase):
    def test_fixed_board_directions(self):
        config = DeviceConfig(
            device_id="home-001",
            rx_id="rx",
            tx1_id="tx1",
            tx2_id="tx2",
            tx3_id="tx3",
        )
        self.assertEqual(config.tx2_direction, "West")
        with self.assertRaises(ValueError):
            DeviceConfig(
                device_id="home-002",
                rx_id="rx",
                tx1_id="tx1",
                tx2_id="tx2",
                tx3_id="tx3",
                tx2_direction="East",
            )

    def test_duplicate_board_id_is_rejected(self):
        with self.assertRaises(ValueError):
            DeviceConfig(
                device_id="home-003",
                rx_id="same",
                tx1_id="same",
                tx2_id="tx2",
                tx3_id="tx3",
            )


if __name__ == "__main__":
    unittest.main()
