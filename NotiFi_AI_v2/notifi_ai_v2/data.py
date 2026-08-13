"""Read the locked v3 cache without importing legacy training code."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .constants import EXCLUDED_SITES, SEALED_SITE, SOURCE_SITES


PROMPT_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)


@dataclass(frozen=True)
class CacheRecord:
    row: int
    trial_id: str
    subject: str
    environment: str
    task: str
    action_id: int
    risk_id: int
    role: str
    cache_ok: bool
    time_method: str

    @property
    def site(self) -> str:
        return f"{self.subject}_{self.environment}"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


class CacheIndex:
    """Load source metadata while keeping the sealed subject inaccessible."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        meta_path = self.root / "cache_meta.json"
        index_path = self.root / "cache_index.csv"
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"invalid cache root: {self.root}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("preproc_version") != "v3.0.0":
            raise RuntimeError("NotiFi AI v2 M1 requires cache preproc v3.0.0")
        if self.meta.get("csi_representation") != "amp_phase":
            raise RuntimeError("M1 cache must contain amplitude and sanitized phase")

        records: list[CacheRecord] = []
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row, values in enumerate(csv.DictReader(handle)):
                records.append(CacheRecord(
                    row=row,
                    trial_id=values["trial_id"],
                    subject=values["subject"],
                    environment=values["environment"],
                    task=values["task"],
                    action_id=int(values["class_id"]),
                    risk_id=int(values["risk_id"]),
                    role=values["role"],
                    cache_ok=_as_bool(values["cache_ok"]),
                    time_method=values["time_method"],
                ))
        if len(records) != int(self.meta["n_trials"]):
            raise RuntimeError("cache metadata and index row count disagree")
        self.records = records

    def source_development_records(self) -> list[CacheRecord]:
        """Return only clean source pose trials from the historical train role."""

        selected = [
            record for record in self.records
            if (record.subject, record.environment) in SOURCE_SITES
            and record.task == "pose_and_action"
            and record.action_id != 6
            and record.role == "train"
            and record.cache_ok
        ]
        if any(
            (record.subject, record.environment) == SEALED_SITE
            for record in selected
        ):
            raise RuntimeError("sealed yja/E02 entered source development")
        if any(
            (record.subject, record.environment) in EXCLUDED_SITES
            for record in selected
        ):
            raise RuntimeError("excluded lmh GT entered source development")
        actual = {record.site for record in selected}
        expected = {f"{subject}_{environment}" for subject, environment in SOURCE_SITES}
        if actual != expected:
            raise RuntimeError(f"unexpected source sites: {sorted(actual)}")
        return selected


def reserve_support(
    records: Iterable[CacheRecord],
    seed: int = 17017,
    shots_per_class: int = 2,
) -> tuple[list[CacheRecord], list[CacheRecord]]:
    """Reserve fixed basic-action support and return disjoint query records."""

    records = list(records)
    by_site: dict[str, list[CacheRecord]] = {}
    for record in records:
        by_site.setdefault(record.site, []).append(record)
    support_ids: set[str] = set()
    for site in sorted(by_site):
        rng = np.random.default_rng(seed)
        for action_id in PROMPT_CLASSES:
            candidates = sorted(
                (row for row in by_site[site] if row.action_id == action_id),
                key=lambda row: row.trial_id,
            )
            if len(candidates) < shots_per_class:
                raise RuntimeError(f"{site} class {action_id} lacks support trials")
            order = rng.permutation(len(candidates))[:shots_per_class]
            support_ids.update(candidates[int(index)].trial_id for index in order)
    support = [row for row in records if row.trial_id in support_ids]
    query = [row for row in records if row.trial_id not in support_ids]
    if {row.trial_id for row in support} & {row.trial_id for row in query}:
        raise RuntimeError("support-query overlap")
    return support, query


def nested_source_split(
    records: Iterable[CacheRecord], held_out_subject: str,
) -> tuple[list[str], list[str], list[str]]:
    """Create source-inner site validation and outer subject LOSO sites."""

    sites = sorted({row.site for row in records})
    outer = [site for site in sites if site.startswith(f"{held_out_subject}_")]
    if not outer:
        raise ValueError(f"unknown held-out subject: {held_out_subject}")
    candidates = [site for site in sites if site not in outer]
    validation = [site for site in candidates if site.endswith("_E03")]
    if not validation:
        subjects = sorted({site.split("_")[0] for site in candidates})
        multi_site = [
            subject for subject in subjects
            if sum(site.startswith(f"{subject}_") for site in candidates) > 1
        ]
        if not multi_site:
            raise RuntimeError("nested validation requires a multi-site subject")
        validation = [
            sorted(site for site in candidates if site.startswith(f"{multi_site[0]}_"))[-1]
        ]
    train = [site for site in candidates if site not in validation]
    if len(train) < 2:
        raise RuntimeError("nested source training requires at least two sites")
    return train, validation, outer


def read_link_quality(root: str | Path) -> dict[str, np.ndarray]:
    """Load the frozen per-trial link QC table when it is available."""

    report = Path(root).resolve().parent / "reports" / "link_quality.csv"
    if not report.exists():
        return {}
    table: dict[str, np.ndarray] = {}
    temporary: dict[str, dict[str, bool]] = {}
    with report.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error", "").strip():
                continue
            try:
                dead = (
                    float(row["live_sc"]) <= 0
                    or float(row["pkt_max_med"]) <= 1.0
                    or float(row.get("allzero_ratio") or 0.0) > 0.9
                    or float(row.get("frame_corr") or 0.0) < 0.65
                )
            except (KeyError, ValueError):
                dead = True
            temporary.setdefault(row["trial_id"], {})[row["tx"]] = not dead
    for trial_id, links in temporary.items():
        table[trial_id] = np.asarray(
            [links.get(name, False) for name in ("TX1", "TX2", "TX3")],
            dtype=bool,
        )
    return table


class CsiPoseDataset(Dataset):
    """Read fixed cache rows and apply frame and trial link masks."""

    def __init__(
        self,
        cache_root: str | Path,
        records: Iterable[CacheRecord],
        link_quality: dict[str, np.ndarray] | None = None,
    ):
        self.root = Path(cache_root).resolve()
        self.records = list(records)
        self.link_quality = link_quality or {}
        self.csi = np.load(self.root / "csi_iq.npy", mmap_mode="r")
        self.link_mask = np.load(self.root / "link_mask.npy", mmap_mode="r")
        self.pose = np.load(self.root / "pose_rel.npy", mmap_mode="r")
        self.root_position = np.load(self.root / "root.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        row = record.row
        csi = np.array(self.csi[row], dtype=np.float32)
        mask = np.array(self.link_mask[row], dtype=bool)
        trial_links = self.link_quality.get(record.trial_id)
        if trial_links is not None:
            mask &= trial_links[None]
        csi *= mask[..., None, None].astype(np.float32)
        return {
            "csi": torch.from_numpy(csi),
            "link_mask": torch.from_numpy(mask),
            "pose_rel": torch.from_numpy(np.array(self.pose[row], dtype=np.float32)),
            "root": torch.from_numpy(np.array(self.root_position[row], dtype=np.float32)),
            "valid": torch.from_numpy(np.array(self.valid[row], dtype=bool)),
            "action_id": torch.tensor(record.action_id, dtype=torch.long),
            "risk_id": torch.tensor(record.risk_id, dtype=torch.long),
            "row": torch.tensor(row, dtype=torch.long),
        }


class CrossSiteClassBatchSampler(Sampler[list[int]]):
    """Place same-action trials from different source sites in each batch."""

    def __init__(
        self,
        dataset: CsiPoseDataset,
        batch_size: int,
        seed: int,
    ):
        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.groups: dict[int, dict[str, list[int]]] = {}
        for index, row in enumerate(dataset.records):
            self.groups.setdefault(row.action_id, {}).setdefault(row.site, []).append(index)
        self.pairable = sorted(
            action for action, sites in self.groups.items() if len(sites) >= 2
        )
        if not self.pairable:
            raise RuntimeError("no cross-site action pairs are available")

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1009)
        self.epoch += 1
        for _ in range(len(self)):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                action = int(rng.choice(self.pairable))
                sites = sorted(self.groups[action])
                left, right = rng.choice(sites, size=2, replace=False)
                batch.append(int(rng.choice(self.groups[action][str(left)])))
                if len(batch) < self.batch_size:
                    batch.append(int(rng.choice(self.groups[action][str(right)])))
            yield batch
