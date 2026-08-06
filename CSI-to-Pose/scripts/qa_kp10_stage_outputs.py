"""Validate rendered pose videos and build 4x4 preview sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def contact_sheet(paths: list[Path], destination: Path) -> None:
    tiles = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read preview: {path}")
        tiles.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    if not tiles:
        raise RuntimeError(f"no previews for {destination}")
    blank = np.full_like(tiles[0], 245)
    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start:start + 4]
        row.extend([blank] * (4 - len(row)))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(destination), np.concatenate(rows, axis=0))


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frames // 2, 0))
    readable, frame = capture.read()
    capture.release()
    return {
        "file": str(path),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "readable": bool(readable),
        "mean": float(frame.mean()) if readable else 0.0,
        "std": float(frame.std()) if readable else 0.0,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    videos = sorted(args.root.glob("*/*.mp4"))
    rows = [inspect_video(path) for path in videos]
    bad = [
        row for row in rows
        if not row["readable"]
        or row["frames"] <= 0
        or row["width"] != 1280
        or row["height"] != 720
        or abs(row["fps"] - 30.0) > 0.1
        or row["std"] < 5.0
    ]
    for mode in ("stickman", "gvhmr"):
        contact_sheet(
            sorted((args.root / mode).glob("*.png")),
            args.root / f"preview_{mode}_contact_sheet.png",
        )
    report = {
        "videos": len(videos),
        "bad_count": len(bad),
        "min_frames": min(row["frames"] for row in rows),
        "max_frames": max(row["frames"] for row in rows),
        "total_mb": round(sum(row["bytes"] for row in rows) / 1024**2, 1),
        "bad": bad,
    }
    (args.root / "qa_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if bad or len(videos) != 32:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
