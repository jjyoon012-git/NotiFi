import argparse
import datetime as dt
import os
import re
import sys
import time

try:
    import serial
except ImportError:
    print("pyserial is not installed. Run: py -m pip install pyserial", file=sys.stderr)
    raise


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label or "csi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read ESP CSI serial logs and save only CSI_DATA lines to CSV."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Capture duration in seconds. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument("--label", required=True, help="Label used in output filename")
    parser.add_argument("--outdir", default="data", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_label(args.label)}_{timestamp}.csv"
    out_path = os.path.abspath(os.path.join(args.outdir, filename))

    deadline = None if args.duration == 0 else time.monotonic() + args.duration
    count = 0

    print(f"Opening {args.port} at {args.baud} baud")
    print(f"Writing CSI_DATA rows to {out_path}")

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser, open(
            out_path, "w", encoding="utf-8", newline=""
        ) as out_file:
            while deadline is None or time.monotonic() < deadline:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("CSI_DATA"):
                    out_file.write(line + "\n")
                    count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 2

    print(f"Saved {count} CSI_DATA rows")
    print(out_path)
    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
