import argparse
import ast
import csv
import math
import os
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly inspect ESP CSI CSV files.")
    parser.add_argument("files", nargs="*", help="CSI CSV files to summarize")
    parser.add_argument("--idle", help="Idle CSV file for comparison")
    parser.add_argument("--move", help="Move CSV file for comparison")
    parser.add_argument(
        "--assume-duration",
        type=float,
        default=None,
        help="Fallback duration in seconds if device timestamps are unavailable",
    )
    return parser.parse_args()


def to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_csi_array(value: str) -> list[int]:
    text = value.strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (SyntaxError, ValueError, TypeError):
        pass

    text = text.strip("[]")
    if not text:
        return []
    result = []
    for item in text.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return result


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None, "stdev": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def get_timestamp_us(row: list[str]) -> int | None:
    # ESP32-C5/C6 current esp-csi format:
    # type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,...
    if len(row) == 15:
        return to_int(row[9])

    # Legacy ESP32/S2/S3/C3 format:
    # ...,channel,secondary_channel,local_timestamp,ant,sig_len,...
    if len(row) >= 24:
        return to_int(row[18])

    return None


def summarize(path: str, assume_duration: float | None = None) -> dict[str, Any]:
    rssi_values: list[float] = []
    csi_lengths: list[float] = []
    mean_abs_values: list[float] = []
    timestamps: list[int] = []
    rows = 0
    malformed = 0

    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            if not row or row[0] != "CSI_DATA":
                continue

            rows += 1
            if len(row) < 5:
                malformed += 1
                continue

            rssi = to_int(row[3])
            if rssi is not None:
                rssi_values.append(float(rssi))

            reported_len = to_int(row[-3]) if len(row) >= 3 else None

            try:
                csi = parse_csi_array(row[-1])
            except ValueError:
                malformed += 1
                csi = []

            csi_lengths.append(float(reported_len if reported_len is not None else len(csi)))

            if csi:
                mean_abs_values.append(statistics.fmean(abs(x) for x in csi))

            timestamp = get_timestamp_us(row)
            if timestamp is not None:
                timestamps.append(timestamp)

    duration_s = None
    if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
        duration_s = (timestamps[-1] - timestamps[0]) / 1_000_000.0
    elif assume_duration and assume_duration > 0:
        duration_s = assume_duration

    rows_per_second = rows / duration_s if duration_s and duration_s > 0 else None

    return {
        "path": os.path.abspath(path),
        "rows": rows,
        "malformed": malformed,
        "duration_s": duration_s,
        "rows_per_second": rows_per_second,
        "rssi": stats(rssi_values),
        "csi_len": stats(csi_lengths),
        "mean_abs": stats(mean_abs_values),
    }


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def print_stats(name: str, values: dict[str, float | None]) -> None:
    print(
        f"{name}: min={fmt(values['min'])}, "
        f"mean={fmt(values['mean'])}, max={fmt(values['max'])}, "
        f"stdev={fmt(values['stdev'])}"
    )


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\nFile: {summary['path']}")
    print(f"CSI_DATA rows: {summary['rows']}")
    print(f"Malformed rows: {summary['malformed']}")
    print(f"Estimated duration: {fmt(summary['duration_s'])} s")
    print(f"Average rows/sec: {fmt(summary['rows_per_second'])}")
    print_stats("RSSI", summary["rssi"])
    print_stats("CSI length", summary["csi_len"])
    print_stats("CSI mean abs", summary["mean_abs"])


def compare(idle: dict[str, Any], move: dict[str, Any]) -> None:
    def delta(metric: str, key: str = "mean") -> str:
        left = idle[metric][key]
        right = move[metric][key]
        if left is None or right is None:
            return "n/a"
        return fmt(right - left)

    print("\nComparison: move - idle")
    print(f"Rows/sec delta: {fmt((move['rows_per_second'] or 0) - (idle['rows_per_second'] or 0))}")
    print(f"RSSI mean delta: {delta('rssi')}")
    print(f"CSI length mean delta: {delta('csi_len')}")
    print(f"CSI mean abs delta: {delta('mean_abs')}")


def main() -> int:
    args = parse_args()
    paths = list(args.files)

    if args.idle:
        paths.append(args.idle)
    if args.move:
        paths.append(args.move)

    if not paths:
        print("No files given.")
        return 2

    summaries: dict[str, dict[str, Any]] = {}
    for path in paths:
        summary = summarize(path, args.assume_duration)
        summaries[path] = summary
        print_summary(summary)

    if args.idle and args.move:
        compare(summaries[args.idle], summaries[args.move])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
