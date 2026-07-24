#!/usr/bin/env python3
"""Check 3TX + 1RX CSI link balance.

The RX serial output must contain CSI_DATA lines. This script counts frames
from TX1/TX2/TX3 MAC addresses for a fixed duration.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter


EXPECTED = {
    "1a:00:00:00:00:00": "TX1",
    "1a:00:00:00:00:01": "TX2",
    "1a:00:00:00:00:02": "TX3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count CSI_DATA frames per TX MAC.")
    parser.add_argument("port", help="RX serial port, for example COM4 or /dev/cu.usbmodem101")
    parser.add_argument("--sec", type=float, default=10.0, help="Measurement duration in seconds")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--rate", type=int, default=30, help="Expected TX packet rate for pass threshold")
    parser.add_argument("--min-ratio", type=float, default=0.80, help="Pass threshold ratio per TX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial is missing. Install with: python -m pip install -r requirements.txt")
        sys.exit(1)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        raise SystemExit(
            f"[ERROR] Could not open {args.port}: {exc}\n"
            "Check port name and close idf.py monitor first."
        )

    counts: Counter[str] = Counter()
    others: Counter[str] = Counter()
    total = 0
    end_time = time.time() + args.sec

    print(f"[CHECK] Reading {args.port} at {args.baud} baud for {args.sec:.0f}s")
    print("[CHECK] Power on TX1/TX2/TX3 and connect only RX to this computer.")

    while time.time() < end_time:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line.startswith("CSI_DATA,"):
            continue
        total += 1
        parts = line.split(",")
        if len(parts) < 3:
            continue
        mac = parts[2].lower()
        if mac in EXPECTED:
            counts[mac] += 1
        else:
            others[mac] += 1

    ser.close()

    target = int(args.rate * args.sec * args.min_ratio)
    print("\n================ Result ================")
    print(f"Total CSI_DATA frames: {total}")
    print(
        f"Pass threshold: >= {target} frames per TX "
        f"({args.rate} pkt/s x {args.sec:.0f}s x {args.min_ratio:.0%})"
    )

    all_ok = True
    for mac, name in EXPECTED.items():
        n = counts.get(mac, 0)
        ok = n >= target
        all_ok = all_ok and ok
        mark = "OK" if ok else "LOW"
        print(f"[{mark:3}] {name} {mac}: {n}")

    if others:
        print("\nOther MACs:")
        for mac, n in others.most_common():
            print(f"  {mac}: {n}")

    if all_ok:
        print("\n=> 3 links look usable for collection.")
    else:
        print("\n=> Recheck power, MAC suffix, antenna connection, distance, and channel.")


if __name__ == "__main__":
    main()
