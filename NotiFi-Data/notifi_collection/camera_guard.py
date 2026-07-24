from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Any

import cv2


BLOCKED_CAMERA_TOKENS = (
    "iphone",
    "ipad",
    "continuity camera",
)


@dataclass(frozen=True)
class CameraSafetyReport:
    platform: str
    checked: bool
    blocked_devices: tuple[str, ...]
    raw_summary: str

    @property
    def ok(self) -> bool:
        return not self.blocked_devices

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "checked": self.checked,
            "blocked_devices": list(self.blocked_devices),
            "raw_summary": self.raw_summary,
        }


def macos_camera_report() -> CameraSafetyReport:
    system = platform.system()
    if system != "Darwin":
        return CameraSafetyReport(
            platform=system,
            checked=False,
            blocked_devices=(),
            raw_summary="macOS camera source check skipped on this platform",
        )

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return CameraSafetyReport(
            platform=system,
            checked=False,
            blocked_devices=(),
            raw_summary=f"system_profiler camera check failed: {exc}",
        )

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    blocked_lines: list[str] = []
    for line in output.splitlines():
        normalized = line.lower()
        if any(token in normalized for token in BLOCKED_CAMERA_TOKENS):
            blocked_lines.append(line.strip())

    summary_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return CameraSafetyReport(
        platform=system,
        checked=True,
        blocked_devices=tuple(blocked_lines),
        raw_summary=" | ".join(summary_lines[:12]),
    )


def ensure_no_mobile_camera() -> CameraSafetyReport:
    report = macos_camera_report()
    if report.blocked_devices:
        devices = "; ".join(report.blocked_devices)
        raise RuntimeError(
            "iPhone/iPad/Continuity Camera appears in the macOS camera list. "
            "Disable Continuity Camera or disconnect the phone before collecting. "
            f"Detected: {devices}"
        )
    return report


def open_laptop_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)
