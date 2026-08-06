"""Contracts and loaders for external CSI, motion, and contact datasets.

External datasets never enter the native NotiFi cache directly. CSI sources are
converted to an amplitude-dynamics view for encoder pretraining, while motion
sources retain their native skeleton and supervise only compatible prior tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "external_datasets.json"
VALID_ROLES = {
    "csi_action",
    "csi_pose",
    "csi_fall",
    "domain_pretraining",
    "motion_prior",
    "impact_prior",
    "contact_prior",
}


@dataclass(frozen=True)
class ExternalDatasetSpec:
    dataset_id: str
    title: str
    roles: tuple[str, ...]
    modalities: tuple[str, ...]
    license: str
    license_verified: bool
    commercial_use: bool
    redistribute_raw: bool
    access: str
    expected_layout: str
    enabled: bool
    notes: str

    @classmethod
    def from_dict(cls, value: dict) -> "ExternalDatasetSpec":
        return cls(
            dataset_id=str(value["id"]),
            title=str(value["title"]),
            roles=tuple(value["roles"]),
            modalities=tuple(value["modalities"]),
            license=str(value["license"]),
            license_verified=bool(value["license_verified"]),
            commercial_use=bool(value["commercial_use"]),
            redistribute_raw=bool(value["redistribute_raw"]),
            access=str(value["access"]),
            expected_layout=str(value["expected_layout"]),
            enabled=bool(value["enabled"]),
            notes=str(value["notes"]),
        )

    def assert_usable(self, role: str, *, allow_unverified: bool = False) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"unknown external-data role: {role}")
        if not self.enabled:
            raise RuntimeError(f"{self.dataset_id} is gated: {self.notes}")
        if role not in self.roles:
            raise RuntimeError(
                f"{self.dataset_id} cannot supervise {role}; allowed={self.roles}"
            )
        if not self.license_verified and not allow_unverified:
            raise RuntimeError(
                f"{self.dataset_id} has an unverified dataset license"
            )


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, ExternalDatasetSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported external dataset registry schema")
    specs = [ExternalDatasetSpec.from_dict(item) for item in payload["datasets"]]
    by_id = {spec.dataset_id: spec for spec in specs}
    if len(by_id) != len(specs):
        raise ValueError("duplicate external dataset id")
    for spec in specs:
        unknown = set(spec.roles) - VALID_ROLES
        if unknown:
            raise ValueError(f"{spec.dataset_id} has unknown roles: {sorted(unknown)}")
        if spec.redistribute_raw and not spec.license_verified:
            raise ValueError(
                f"{spec.dataset_id} cannot permit raw redistribution before license review"
            )
    return by_id


@dataclass(frozen=True)
class CanonicalCSIView:
    """Hardware-neutral CSI dynamics used only for external pretraining.

    values has shape [T, L, S, 2]. Channel 0 is robust-standardized log
    amplitude and channel 1 is its first temporal difference. Static phase and
    absolute gain are intentionally absent.
    """

    values: np.ndarray
    link_mask: np.ndarray
    source_layout: str


def _interp_axis(values: np.ndarray, size: int, axis: int) -> np.ndarray:
    if values.shape[axis] == size:
        return values
    moved = np.moveaxis(values, axis, -1)
    old = np.linspace(0.0, 1.0, moved.shape[-1], dtype=np.float64)
    new = np.linspace(0.0, 1.0, size, dtype=np.float64)
    flat = moved.reshape(-1, moved.shape[-1])
    output = np.empty((len(flat), size), dtype=np.float32)
    for index, row in enumerate(flat):
        output[index] = np.interp(new, old, row)
    return np.moveaxis(output.reshape(*moved.shape[:-1], size), -1, axis)


def _layout_to_amplitude(
    values: np.ndarray,
    layout: str,
    *,
    flattened_links: int | None = None,
    notifi_representation: str = "amp_phase",
) -> np.ndarray:
    data = np.asarray(values)
    if layout == "mmfi":
        if data.ndim != 4:
            raise ValueError("MM-Fi CSI must have shape [frames, links, subcarriers, packets]")
        frames, links, _, packets = data.shape
        data = data.transpose(0, 3, 1, 2).reshape(frames * packets, links, -1)
    elif layout == "person_in_wifi_3d":
        if data.ndim != 4:
            raise ValueError("Person-in-WiFi CSI must have shape [rx, antennas, subcarriers, packets]")
        receivers, antennas, subcarriers, packets = data.shape
        data = data.transpose(3, 0, 1, 2).reshape(
            packets, receivers * antennas, subcarriers
        )
    elif layout == "csi_bench":
        data = np.squeeze(data)
        if data.ndim != 2 or not flattened_links:
            raise ValueError("CSI-Bench requires [time, flattened_features] and flattened_links")
        if data.shape[1] % flattened_links:
            raise ValueError("CSI-Bench feature count is not divisible by flattened_links")
        data = data.reshape(data.shape[0], flattened_links, -1)
    elif layout == "notifi":
        if data.ndim != 4 or data.shape[-1] != 2:
            raise ValueError("NotiFi CSI must have shape [time, links, subcarriers, 2]")
        if notifi_representation == "iq":
            data = np.hypot(data[..., 0], data[..., 1])
        elif notifi_representation == "amp_phase":
            data = np.abs(data[..., 0])
        else:
            raise ValueError("notifi_representation must be 'iq' or 'amp_phase'")
    elif layout == "amplitude":
        if data.ndim != 3:
            raise ValueError("amplitude CSI must have shape [time, links, subcarriers]")
    else:
        raise ValueError(f"unsupported CSI layout: {layout}")
    if np.iscomplexobj(data):
        data = np.abs(data)
    return np.asarray(data, dtype=np.float32)


def canonicalize_csi(
    values: np.ndarray,
    layout: str,
    *,
    target_frames: int = 304,
    target_subcarriers: int = 114,
    flattened_links: int | None = None,
    notifi_representation: str = "amp_phase",
    eps: float = 1e-4,
) -> CanonicalCSIView:
    """Convert heterogeneous CSI to gain-invariant amplitude dynamics.

    Link count is deliberately not forced to three. The shared pretraining
    encoder applies the same frequency weights to every link and pools with a
    mask, so a source with 3, 4, or 9 links can use the same backbone.
    """

    amplitude = _layout_to_amplitude(
        values,
        layout,
        flattened_links=flattened_links,
        notifi_representation=notifi_representation,
    )
    if min(amplitude.shape) == 0:
        raise ValueError("CSI contains an empty dimension")
    finite = np.isfinite(amplitude)
    link_valid = finite.any(axis=(0, 2))
    amplitude = np.where(finite, amplitude, np.nan)
    median = np.nanmedian(amplitude, axis=0, keepdims=True)
    amplitude = np.where(np.isfinite(amplitude), amplitude, median)
    amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
    amplitude = np.log1p(np.maximum(amplitude, 0.0))
    amplitude = _interp_axis(amplitude, target_subcarriers, axis=2)
    amplitude = _interp_axis(amplitude, target_frames, axis=0)

    center = np.median(amplitude, axis=0, keepdims=True)
    mad = np.median(np.abs(amplitude - center), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, eps)
    normalized = np.clip((amplitude - center) / scale, -12.0, 12.0)
    delta = np.diff(normalized, axis=0, prepend=normalized[:1])
    dynamic = np.stack((normalized, delta), axis=-1).astype(np.float32)
    frame_mask = np.broadcast_to(link_valid[None], (target_frames, len(link_valid))).copy()
    dynamic *= frame_mask[:, :, None, None]
    return CanonicalCSIView(dynamic, frame_mask, layout)


_UPFALL_NAME = re.compile(
    r"^C(?P<camera>\d+)S(?P<subject>\d+)_?A(?P<action>\d+)_?T(?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UPFallMotionSample:
    sample_id: str
    camera: int
    subject: int
    action: int
    trial: int
    pose: np.ndarray
    impact: np.ndarray

    @property
    def is_fall(self) -> bool:
        return 1 <= self.action <= 5

    @property
    def impact_index(self) -> int | None:
        indices = np.flatnonzero(self.impact)
        return int(indices[0]) if len(indices) else None

    def phase_targets(self) -> np.ndarray:
        """Return frame labels: 0 pre-impact, 1 impact, 2 post-impact.

        This intentionally does not invent a descent-onset label. UP-Fall only
        provides impact/non-impact supervision in the released CSV files.
        """

        output = np.zeros(len(self.impact), dtype=np.int64)
        hit = np.flatnonzero(self.impact)
        if not len(hit):
            return output
        output[hit] = 1
        output[hit[-1] + 1 :] = 2
        return output

    def root_relative_pose(self) -> np.ndarray:
        pose = self.pose.astype(np.float32, copy=True)
        hip = 0.5 * (pose[:, 23] + pose[:, 24])
        shoulder = 0.5 * (pose[:, 11] + pose[:, 12])
        scale = np.linalg.norm(shoulder - hip, axis=-1)
        valid = np.isfinite(scale) & (scale > 1e-5)
        body_scale = float(np.median(scale[valid])) if valid.any() else 1.0
        return (pose - hip[:, None]) / max(body_scale, 1e-5)


def _read_upfall_csv(payload: bytes, filename: str) -> UPFallMotionSample:
    match = _UPFALL_NAME.match(Path(filename).name)
    if not match:
        raise ValueError(f"invalid UP-Fall filename: {filename}")
    table = pd.read_csv(BytesIO(payload))
    coordinate_columns = [
        f"Joint{joint}_{axis}"
        for joint in range(1, 34)
        for axis in ("X", "Y", "Z")
    ]
    missing = [column for column in coordinate_columns if column not in table]
    label_columns = [
        str(column) for column in table.columns
        if str(column).upper().endswith("LABEL")
    ]
    # Two released files use LLABEL, and one uses the first label value ("0")
    # as the final column name. Accept the last column only when all 99 pose
    # columns are present and there is exactly one additional column.
    if not label_columns and not missing and len(table.columns) == 100:
        label_columns = [str(table.columns[-1])]
    if missing or len(label_columns) != 1:
        raise ValueError(
            f"UP-Fall CSV schema mismatch: missing={missing[:3]}, "
            f"label_candidates={label_columns}"
        )
    pose = table[coordinate_columns].to_numpy(dtype=np.float32).reshape(-1, 33, 3)
    impact = table[label_columns[0]].to_numpy(dtype=np.int64) != 0
    groups = {key: int(value) for key, value in match.groupdict().items()}
    return UPFallMotionSample(
        sample_id=Path(filename).stem,
        camera=groups["camera"],
        subject=groups["subject"],
        action=groups["action"],
        trial=groups["trial"],
        pose=pose,
        impact=impact,
    )


def iter_upfall_archives(root: str | Path) -> Iterable[UPFallMotionSample]:
    root = Path(root)
    archives = sorted(root.glob("SUBJECT*.zip"))
    if not archives:
        raise FileNotFoundError(f"no SUBJECT*.zip archives under {root}")
    for archive in archives:
        with ZipFile(archive) as bundle:
            for name in sorted(bundle.namelist()):
                if name.lower().endswith(".csv"):
                    yield _read_upfall_csv(bundle.read(name), name)


def summarize_upfall(root: str | Path) -> dict[str, int]:
    root = Path(root)
    samples: list[UPFallMotionSample] = []
    skipped_schema = 0
    archives = sorted(root.glob("SUBJECT*.zip"))
    if not archives:
        raise FileNotFoundError(f"no SUBJECT*.zip archives under {root}")
    for archive in archives:
        with ZipFile(archive) as bundle:
            for name in sorted(bundle.namelist()):
                if not name.lower().endswith(".csv"):
                    continue
                try:
                    samples.append(_read_upfall_csv(bundle.read(name), name))
                except ValueError:
                    skipped_schema += 1
    return {
        "samples": len(samples),
        "subjects": len({sample.subject for sample in samples}),
        "fall_samples": sum(sample.is_fall for sample in samples),
        "impact_labeled_samples": sum(sample.impact_index is not None for sample in samples),
        "frames": sum(len(sample.pose) for sample in samples),
        "skipped_schema": skipped_schema,
    }
