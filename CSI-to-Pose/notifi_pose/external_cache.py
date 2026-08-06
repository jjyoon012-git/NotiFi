"""Build compact training caches from approved external dataset archives."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

from .external_data import _interp_axis, canonicalize_csi


_MMFI_SEQUENCE = re.compile(r"^(E\d{2})/(S\d{2})/(A\d{2})$")


def _mmfi_sequences(
    bundle: ZipFile,
) -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    sequences: dict[str, tuple[str, str, str]] = {}
    csi_frames: dict[str, list[str]] = {}
    for info in bundle.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if path.name == "ground_truth.npy":
            directory = str(path.parent)
            # Some archives add one outer directory. Match the final E/S/A suffix.
            parts = path.parent.parts
            if len(parts) < 3:
                continue
            suffix = "/".join(parts[-3:])
            match = _MMFI_SEQUENCE.match(suffix)
            if match:
                sequences[directory] = match.groups()
            continue
        if path.suffix.lower() == ".mat" and path.parent.name == "wifi-csi":
            csi_frames.setdefault(str(path.parent.parent), []).append(name)
    return sorted(
        (
            directory, *identity, tuple(sorted(csi_frames.get(directory, ())))
        )
        for directory, identity in sequences.items()
    )


def _load_mmfi_sequence(
    bundle: ZipFile, directory: str, frames: tuple[str, ...], *, target_frames: int,
    target_subcarriers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError(f"MM-Fi sequence has no CSI frames: {directory}")
    csi_frames = []
    for name in frames:
        payload = loadmat(BytesIO(bundle.read(name)))
        if "CSIamp" not in payload:
            raise ValueError(f"MM-Fi frame lacks CSIamp: {name}")
        csi_frames.append(np.asarray(payload["CSIamp"], dtype=np.float32))
    source = np.stack(csi_frames)
    view = canonicalize_csi(
        source,
        "mmfi",
        target_frames=target_frames,
        target_subcarriers=target_subcarriers,
    )

    with bundle.open(f"{directory}/ground_truth.npy") as stream:
        pose = np.load(BytesIO(stream.read()), allow_pickle=False)
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[-1] not in (2, 3):
        raise ValueError(
            f"unsupported MM-Fi ground truth shape {pose.shape} in {directory}"
        )
    pose = _interp_axis(pose, target_frames, axis=0).astype(np.float32)
    return view.values, view.link_mask, pose


def build_mmfi_zip_cache(
    archive: str | Path,
    output: str | Path,
    *,
    target_frames: int = 304,
    target_subcarriers: int = 114,
    max_sequences: int | None = None,
) -> dict:
    """Read only WiFi CSI and pose GT from an MM-Fi environment ZIP."""

    archive = Path(archive).resolve()
    output = Path(output).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite external cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial-{uuid4().hex[:8]}"
    partial.mkdir()
    try:
        with ZipFile(archive) as bundle:
            sequences = _mmfi_sequences(bundle)
            if max_sequences is not None:
                sequences = sequences[:max_sequences]
            if not sequences:
                raise ValueError("no MM-Fi E/S/A sequences found in archive")

            first = _load_mmfi_sequence(
                bundle, sequences[0][0], sequences[0][4],
                target_frames=target_frames,
                target_subcarriers=target_subcarriers,
            )
            count = len(sequences)
            csi_shape = (count, *first[0].shape)
            mask_shape = (count, *first[1].shape)
            pose_shape = (count, *first[2].shape)
            csi_mem = np.lib.format.open_memmap(
                partial / "csi_dynamic.npy", mode="w+", dtype=np.float16,
                shape=csi_shape,
            )
            mask_mem = np.lib.format.open_memmap(
                partial / "link_mask.npy", mode="w+", dtype=np.bool_,
                shape=mask_shape,
            )
            pose_mem = np.lib.format.open_memmap(
                partial / "pose_native.npy", mode="w+", dtype=np.float32,
                shape=pose_shape,
            )
            metadata = []
            for index, (directory, environment, subject, action, frames) in enumerate(sequences):
                if index == 0:
                    csi, mask, pose = first
                else:
                    csi, mask, pose = _load_mmfi_sequence(
                        bundle, directory, frames,
                        target_frames=target_frames,
                        target_subcarriers=target_subcarriers,
                    )
                if csi.shape != csi_shape[1:] or pose.shape != pose_shape[1:]:
                    raise ValueError(f"inconsistent MM-Fi shape in {directory}")
                csi_mem[index] = csi
                mask_mem[index] = mask
                pose_mem[index] = pose
                metadata.append({
                    "row": index,
                    "environment": environment,
                    "subject": subject,
                    "action": action,
                    "action_id": int(action[1:]) - 1,
                    "source_path": directory,
                })
            csi_mem.flush()
            mask_mem.flush()
            pose_mem.flush()
            # Windows keeps memory-mapped files locked until every mapping is
            # released, which otherwise prevents the transactional rename.
            del csi_mem, mask_mem, pose_mem

        pd.DataFrame(metadata).to_csv(partial / "metadata.csv", index=False)
        manifest = {
            "format": "notifi_external_csi_cache_v1",
            "dataset": "mmfi",
            "source_archive": str(archive),
            "source_bytes": archive.stat().st_size,
            "sequences": len(metadata),
            "target_frames": target_frames,
            "target_subcarriers": target_subcarriers,
            "csi_shape": list(csi_shape),
            "pose_shape": list(pose_shape),
            "csi_representation": "robust_log_amplitude_and_delta",
            "pose_representation": "native_mmfi_keypoints_resampled_only",
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        partial.replace(output)
        return manifest
    except Exception:
        if partial.exists() and partial.parent == output.parent:
            shutil.rmtree(partial)
        raise


class ExternalCSICacheDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("format") != "notifi_external_csi_cache_v1":
            raise ValueError("unsupported external CSI cache")
        self.metadata = pd.read_csv(self.root / "metadata.csv")
        self.csi = np.load(self.root / "csi_dynamic.npy", mmap_mode="r")
        self.mask = np.load(self.root / "link_mask.npy", mmap_mode="r")
        self.pose = np.load(self.root / "pose_native.npy", mmap_mode="r")
        if len(self.metadata) != len(self.csi):
            raise ValueError("external cache metadata/array length mismatch")

    def __len__(self) -> int:
        return len(self.metadata)

    def close(self) -> None:
        for array in (self.csi, self.mask, self.pose):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def __del__(self):
        self.close()

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]
        return {
            "csi": torch.from_numpy(np.array(self.csi[index], dtype=np.float32)),
            "link_mask": torch.from_numpy(np.array(self.mask[index], dtype=bool)),
            "pose_native": torch.from_numpy(np.array(self.pose[index], dtype=np.float32)),
            "action_id": torch.tensor(int(row.action_id), dtype=torch.long),
            "row": torch.tensor(int(row.row), dtype=torch.long),
            "dataset_id": str(self.manifest["dataset"]),
        }
