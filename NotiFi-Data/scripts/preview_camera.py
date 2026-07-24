from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.camera_guard import ensure_no_mobile_camera, open_laptop_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview and verify the collection camera.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    camera_safety_report = ensure_no_mobile_camera()
    print(f"[CAMERA SAFETY] {camera_safety_report.raw_summary}")

    cap = open_laptop_camera(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("[CAMERA] q 또는 ESC를 누르면 종료합니다.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 200, 255), 1)
            cv2.line(frame, (0, height // 2), (width, height // 2), (0, 200, 255), 1)
            cv2.putText(
                frame,
                "Full body + mat + chair + bed must remain visible",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("NotiFi camera setup", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
