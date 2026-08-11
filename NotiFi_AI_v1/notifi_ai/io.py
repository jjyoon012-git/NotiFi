"""NPZ interchange helpers used by the CLI and HTTP API."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .preprocessing import load_csv
from .schemas import SupportTrial


def _open_npz(source: str | Path | bytes | BinaryIO):
    if isinstance(source, bytes):
        return np.load(BytesIO(source), allow_pickle=False)
    if hasattr(source, "read"):
        return np.load(source, allow_pickle=False)
    return np.load(Path(source), allow_pickle=False)


def load_query_npz(
    source: str | Path | bytes | BinaryIO,
) -> tuple[np.ndarray, np.ndarray]:
    with _open_npz(source) as archive:
        return archive["csi"].astype(np.float32), archive["link_mask"].astype(bool)


def load_calibration_npz(
    source: str | Path | bytes | BinaryIO,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[SupportTrial]]:
    with _open_npz(source) as archive:
        absence_csi = archive["absence_csi"].astype(np.float32)
        absence_mask = archive["absence_mask"].astype(bool)
        if len(absence_csi) != len(absence_mask):
            raise ValueError("absence_csi and absence_mask counts differ")
        absence = list(zip(absence_csi, absence_mask))
        support: list[SupportTrial] = []
        if "support_csi" in archive.files:
            required = {
                "support_mask",
                "support_action",
                "support_risk",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"missing support arrays: {missing}")
            csi = archive["support_csi"].astype(np.float32)
            mask = archive["support_mask"].astype(bool)
            action = archive["support_action"].astype(np.int64)
            risk = archive["support_risk"].astype(np.int64)
            if not (len(csi) == len(mask) == len(action) == len(risk)):
                raise ValueError("support array counts differ")
            support = [
                SupportTrial(x, m, int(a), int(r), f"support_{index:03d}")
                for index, (x, m, a, r) in enumerate(zip(csi, mask, action, risk))
            ]
        return absence, support


def load_calibration_manifest(
    path: str | Path,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[SupportTrial]]:
    """Load raw ESP CSI CSV files listed in a calibration manifest."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    absence = []
    for value in payload.get("absence", []):
        csi, mask, _ = load_csv(resolve(value))
        absence.append((csi, mask))
    support = []
    for index, item in enumerate(payload.get("support", [])):
        csi, mask, _ = load_csv(resolve(item["path"]))
        support.append(
            SupportTrial(
                csi=csi,
                link_mask=mask,
                action_id=int(item["action_id"]),
                risk_id=int(item["risk_id"]),
                trial_id=str(item.get("trial_id", f"support_{index:03d}")),
            )
        )
    if not absence:
        raise ValueError("calibration manifest requires at least one absence CSV")
    return absence, support
