#!/usr/bin/env python3
"""Set the compile-time TX MAC suffix for esp-csi csi_send.

Examples:
    python scripts/set_tx_mac.py --esp-csi C:\\Users\\mhw\\NotiFI\\esp-csi --tx tx2
    python scripts/set_tx_mac.py --esp-csi ~/Desktop/NotiFi/esp-csi --suffix 02
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TX_SUFFIX = {
    "tx1": "00",
    "tx2": "01",
    "tx3": "02",
}

MAC_RE = re.compile(
    r"static\s+const\s+uint8_t\s+CONFIG_CSI_SEND_MAC\[\]\s*=\s*"
    r"\{0x1a,\s*0x00,\s*0x00,\s*0x00,\s*0x00,\s*0x[0-9a-fA-F]{2}\};"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Change csi_send TX MAC suffix.")
    parser.add_argument(
        "--esp-csi",
        required=True,
        help="Path to the esp-csi repository.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tx", choices=sorted(TX_SUFFIX), help="TX board id: tx1, tx2, tx3")
    group.add_argument("--suffix", choices=["00", "01", "02"], help="Last byte of TX MAC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = TX_SUFFIX[args.tx] if args.tx else args.suffix
    app_main = (
        Path(args.esp_csi).expanduser()
        / "examples"
        / "get-started"
        / "csi_send"
        / "main"
        / "app_main.c"
    )

    if not app_main.exists():
        raise SystemExit(f"[ERROR] app_main.c not found: {app_main}")

    text = app_main.read_text(encoding="utf-8")
    replacement = (
        "static const uint8_t CONFIG_CSI_SEND_MAC[] = "
        f"{{0x1a, 0x00, 0x00, 0x00, 0x00, 0x{suffix}}};"
    )
    updated, count = MAC_RE.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit("[ERROR] CONFIG_CSI_SEND_MAC line was not found or was ambiguous.")

    app_main.write_text(updated, encoding="utf-8")
    print(f"[OK] {app_main}")
    print(f"[OK] TX MAC set to 1a:00:00:00:00:{suffix}")


if __name__ == "__main__":
    main()
