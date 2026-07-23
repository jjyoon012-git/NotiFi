from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .labels import ENVIRONMENTS, SUBJECTS


DEFAULT_CONFIG_PATH = Path(".notifi_session.json")


@dataclass(frozen=True)
class SessionConfig:
    subject: str
    environment: str
    session_id: str
    port: str
    camera_index: int
    firmware_commit: str
    device_config_id: str
    channel: int = 11
    packet_rate: int = 30
    baud: int = 921600
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: float = 30.0
    tx1_position: str = "opposite_rx_1.50m_height_0.80m"
    tx2_position: str = "side_height_1.45m"
    tx3_position: str = "diagonal_floor_side_height_0.35m"
    rx_position: str = "activity_center_front_height_0.80m"

    @classmethod
    def from_dict(cls, data: dict) -> "SessionConfig":
        config = cls(
            subject=str(data["subject"]).lower(),
            environment=str(data["environment"]).upper(),
            session_id=str(data["session_id"]),
            port=str(data["port"]),
            camera_index=int(data["camera_index"]),
            firmware_commit=str(data["firmware_commit"]),
            device_config_id=str(data["device_config_id"]),
            channel=int(data.get("channel", 11)),
            packet_rate=int(data.get("packet_rate", 30)),
            baud=int(data.get("baud", 921600)),
            camera_width=int(data.get("camera_width", 1280)),
            camera_height=int(data.get("camera_height", 720)),
            camera_fps=float(data.get("camera_fps", 30.0)),
            tx1_position=str(data.get("tx1_position", cls.tx1_position)),
            tx2_position=str(data.get("tx2_position", cls.tx2_position)),
            tx3_position=str(data.get("tx3_position", cls.tx3_position)),
            rx_position=str(data.get("rx_position", cls.rx_position)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.subject not in SUBJECTS:
            raise ValueError(f"subject must be one of {SUBJECTS}: {self.subject}")
        if self.environment not in ENVIRONMENTS:
            raise ValueError(f"environment must be one of {ENVIRONMENTS}: {self.environment}")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not self.port.strip():
            raise ValueError("port cannot be empty")
        if not self.firmware_commit.strip():
            raise ValueError("firmware_commit cannot be empty")
        if not self.device_config_id.strip():
            raise ValueError("device_config_id cannot be empty")
        if self.packet_rate <= 0:
            raise ValueError("packet_rate must be positive")
        if self.channel <= 0:
            raise ValueError("channel must be positive")

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "environment": self.environment,
            "session_id": self.session_id,
            "port": self.port,
            "camera_index": self.camera_index,
            "firmware_commit": self.firmware_commit,
            "device_config_id": self.device_config_id,
            "channel": self.channel,
            "packet_rate": self.packet_rate,
            "baud": self.baud,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "camera_fps": self.camera_fps,
            "tx1_position": self.tx1_position,
            "tx2_position": self.tx2_position,
            "tx3_position": self.tx3_position,
            "rx_position": self.rx_position,
        }


def load_session(path: Path = DEFAULT_CONFIG_PATH) -> SessionConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Session config not found: {path}. Run scripts/create_session.py first."
        )
    with path.open(encoding="utf-8") as handle:
        return SessionConfig.from_dict(json.load(handle))


def save_session(config: SessionConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    config.validate()
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
