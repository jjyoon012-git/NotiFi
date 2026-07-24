from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.camera_guard import ensure_no_mobile_camera


def main() -> None:
    report = ensure_no_mobile_camera()
    print("[OK] No iPhone/iPad/Continuity Camera source was detected.")
    print(f"[CAMERA SUMMARY] {report.raw_summary}")


if __name__ == "__main__":
    main()
