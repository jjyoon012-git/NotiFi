"""Serializable input and output contracts for the deployment runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np

from .constants import LINK_DIRECTIONS, MODEL_NAME


@dataclass(frozen=True)
class DeviceConfig:
    """Physical installation metadata for one registered receiver."""

    device_id: str
    rx_id: str
    tx1_id: str
    tx2_id: str
    tx3_id: str
    firmware_version: str = "unknown"
    rx_direction: str = LINK_DIRECTIONS["RX"]
    tx1_direction: str = LINK_DIRECTIONS["TX1"]
    tx2_direction: str = LINK_DIRECTIONS["TX2"]
    tx3_direction: str = LINK_DIRECTIONS["TX3"]
    notes: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.device_id):
            raise ValueError("device_id must be 1-64 safe filename characters")
        board_ids = (self.rx_id, self.tx1_id, self.tx2_id, self.tx3_id)
        if any(not value.strip() for value in board_ids):
            raise ValueError("RX/TX board IDs must not be empty")
        if len(set(board_ids)) != len(board_ids):
            raise ValueError("RX/TX board IDs must be unique")
        actual = {
            "RX": self.rx_direction,
            "TX1": self.tx1_direction,
            "TX2": self.tx2_direction,
            "TX3": self.tx3_direction,
        }
        if actual != LINK_DIRECTIONS:
            raise ValueError(
                "board direction must be RX=North, TX1=South, "
                "TX2=West, TX3=East"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SupportTrial:
    """A labeled calibration trial that is excluded from later queries."""

    csi: np.ndarray
    link_mask: np.ndarray
    action_id: int
    risk_id: int
    trial_id: str = ""

    def __post_init__(self) -> None:
        if not 0 <= int(self.action_id) < 17:
            raise ValueError("support action_id must be between 0 and 16")
        if not 0 <= int(self.risk_id) < 3:
            raise ValueError("support risk_id must be between 0 and 2")


@dataclass
class Prediction:
    """One CSI-window prediction returned by the public API."""

    action_id: int
    action_label: str
    action_probability: list[float]
    risk_id: int
    risk_label: str
    risk_probability: list[float]
    pose_rel: np.ndarray
    root: np.ndarray
    frame_valid: np.ndarray
    quality: dict[str, Any]
    model_name: str = MODEL_NAME
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_pose: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "action_id": self.action_id,
            "action_label": self.action_label,
            "action_probability": self.action_probability,
            "risk_id": self.risk_id,
            "risk_label": self.risk_label,
            "risk_probability": self.risk_probability,
            "frame_valid": self.frame_valid.astype(bool).tolist(),
            "quality": self.quality,
            "metadata": self.metadata,
        }
        if include_pose:
            payload["pose_rel"] = self.pose_rel.astype(float).tolist()
            payload["root"] = self.root.astype(float).tolist()
        return payload

    def save_npz(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            pose_rel=self.pose_rel.astype(np.float32),
            root=self.root.astype(np.float32),
            frame_valid=self.frame_valid.astype(bool),
            action_probability=np.asarray(self.action_probability, dtype=np.float32),
            risk_probability=np.asarray(self.risk_probability, dtype=np.float32),
            action_id=np.asarray(self.action_id, dtype=np.int64),
            risk_id=np.asarray(self.risk_id, dtype=np.int64),
        )
        return output
