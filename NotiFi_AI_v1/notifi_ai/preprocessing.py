"""Shared CSI preprocessing for CSV, serial packets, and tensor APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .csi_parser import (
    CsiParseError,
    load_csi_trial,
    packets_to_grid,
    parse_csi_line,
)

from .constants import MAX_FRAMES, N_CHANNELS, N_LINKS, N_SUBCARRIERS, TARGET_FPS


def validate_trial(csi: np.ndarray, link_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert one trial without silently reshaping channels."""

    values = np.asarray(csi, dtype=np.float32)
    mask = np.asarray(link_mask, dtype=bool)
    expected_tail = (N_LINKS, N_SUBCARRIERS, N_CHANNELS)
    if values.ndim != 4 or tuple(values.shape[1:]) != expected_tail:
        raise ValueError(f"csi must have shape [T,{N_LINKS},{N_SUBCARRIERS},2]")
    if mask.shape != values.shape[:2]:
        raise ValueError("link_mask must have shape [T,3]")
    if not np.isfinite(values).all():
        raise ValueError("csi contains NaN or infinity")
    if not mask.any():
        raise ValueError("trial contains no valid CSI link")
    return values, mask


def pad_or_trim(csi: np.ndarray, link_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a variable-length trial to the fixed 304-frame model contract."""

    values, mask = validate_trial(csi, link_mask)
    output = np.zeros((MAX_FRAMES, N_LINKS, N_SUBCARRIERS, N_CHANNELS), np.float32)
    output_mask = np.zeros((MAX_FRAMES, N_LINKS), bool)
    length = min(len(values), MAX_FRAMES)
    output[:length] = values[:length]
    output_mask[:length] = mask[:length]
    output *= output_mask[:, :, None, None]
    return output, output_mask


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one dataset-compatible CSI CSV into the deployment tensor contract."""

    trial = load_csi_trial(Path(path))
    csi, mask = pad_or_trim(trial.iq, trial.link_mask)
    return csi, mask, trial.meta


def packets_to_trial(
    per_link_times: Sequence[np.ndarray],
    per_link_iq: Sequence[np.ndarray],
    duration_s: float = MAX_FRAMES / TARGET_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert three ESP packet buffers to the same grid used during training."""

    grid = np.arange(MAX_FRAMES, dtype=np.float64) / TARGET_FPS
    if duration_s > 0 and duration_s < grid[-1]:
        grid = grid[grid <= duration_s]
    csi, mask = packets_to_grid(list(per_link_times), list(per_link_iq), grid)
    return pad_or_trim(csi, mask)


__all__ = [
    "CsiParseError",
    "load_csv",
    "packets_to_grid",
    "packets_to_trial",
    "pad_or_trim",
    "parse_csi_line",
    "validate_trial",
]
