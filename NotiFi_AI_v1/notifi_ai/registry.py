"""On-device registry for ESP installation metadata and calibration profiles."""

from __future__ import annotations

import json
from pathlib import Path

from .calibration import CalibrationProfile
from .schemas import DeviceConfig


class DeviceRegistry:
    def __init__(self, root: str | Path = "runtime/devices"):
        self.root = Path(root)

    def register(self, config: DeviceConfig) -> Path:
        folder = self.root / config.device_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "device.json"
        path.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def load_device(self, device_id: str) -> DeviceConfig:
        path = self.root / device_id / "device.json"
        return DeviceConfig(**json.loads(path.read_text(encoding="utf-8")))

    def save_calibration(self, profile: CalibrationProfile) -> Path:
        return profile.save(self.root / profile.device_id / "calibration.pt")

    def load_calibration(self, device_id: str) -> CalibrationProfile:
        return CalibrationProfile.load(self.root / device_id / "calibration.pt")

    def list_devices(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            folder.name
            for folder in self.root.iterdir()
            if folder.is_dir() and (folder / "device.json").exists()
        )

