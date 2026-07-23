from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.csi import EXPECTED_TX, parse_csi_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all three NotiFi TX links.")
    parser.add_argument("port", help="RX serial port, e.g. COM4")
    parser.add_argument("--sec", type=float, default=10.0)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--pass-ratio", type=float, default=0.90)
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"[ERROR] Cannot open {args.port}: {exc}")
        print("Close idf.py monitor and verify the RX port.")
        raise SystemExit(2) from exc

    counts: Counter[str] = Counter()
    mac_counts: Counter[str] = Counter()
    started = time.monotonic()
    print(
        f"[CHECK] port={args.port} baud={args.baud} duration={args.sec:.0f}s "
        f"expected_rate={args.rate}/TX"
    )
    try:
        while time.monotonic() - started < args.sec:
            raw = ser.readline()
            if not raw:
                continue
            frame = parse_csi_line(raw.decode("utf-8", errors="ignore"))
            if frame is None:
                continue
            counts[frame.sender_id] += 1
            mac_counts[frame.sender_mac] += 1
    finally:
        ser.close()

    minimum = int(args.rate * args.sec * args.pass_ratio)
    print("\n================ 3TX LINK RESULT ================")
    print(
        f"Minimum per TX: {minimum} frames "
        f"({args.rate} pkt/s x {args.sec:.0f}s x {args.pass_ratio:.0%})"
    )
    all_ok = True
    for mac, tx_id in EXPECTED_TX.items():
        count = counts.get(tx_id, 0)
        ok = count >= minimum
        all_ok = all_ok and ok
        print(f"[{'OK' if ok else 'FAIL'}] {tx_id} {mac}: {count} frames")
    unknown = counts.get("UNKNOWN", 0)
    if unknown:
        print(f"[WARN] unknown MAC frames: {unknown}")
        for mac, count in mac_counts.items():
            if mac not in EXPECTED_TX:
                print(f"       {mac}: {count}")
    print("==================================================")
    if not all_ok:
        print("One or more links failed. Do not start collection.")
        raise SystemExit(1)
    print("All three links passed. Collection can begin.")


if __name__ == "__main__":
    main()
