"""source-only calibration 학습과 평가가 공유하는 cache episode 도구."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from notifi_pose import contract as C
from notifi_pose.linkqc import link_mask_per_trial
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
PROMPT_SHOTS = {class_id: 2 for class_id in MOTION_PROMPT_CLASSES}
ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES


def select_source_rows(index: pd.DataFrame) -> np.ndarray:
    """cache index에서 고정 source train protocol 행을 과거 checkpoint 없이 복원한다."""
    source_subject = index.subject.isin(("ajh", "mhw", "lmh"))
    allowed_environment = ~(
        (index.subject == "lmh") & (index.environment != "E01")
    )
    selected = (
        source_subject
        & allowed_environment
        & (index.task == C.TASK_POSE)
        & (index.class_id != 6)
        & index.cache_ok
        & (index.role == "train")
    )
    return np.flatnonzero(selected.to_numpy()).astype(np.int64)


class RawStore:
    """필요한 cache 행만 RAM에 올리고 trial 링크 품질 마스크를 적용한다."""

    def __init__(self, index: pd.DataFrame, rows: np.ndarray):
        rows = np.asarray(sorted(set(int(row) for row in rows)), dtype=np.int64)
        csi = np.load(WORK / "cache/csi_iq.npy", mmap_mode="r")
        mask = np.load(WORK / "cache/link_mask.npy", mmap_mode="r")
        self.rows = rows
        self.position = {int(row): position for position, row in enumerate(rows)}
        print(f"[cache] loading {len(rows)} raw trials into RAM", flush=True)
        self.csi = torch.from_numpy(np.array(csi[rows], dtype=np.float16))
        self.mask = torch.from_numpy(np.array(mask[rows], dtype=bool))
        trial_quality = link_mask_per_trial()
        usable = np.zeros((len(rows), C.N_LINKS), dtype=bool)
        for local, row in enumerate(rows):
            trial_id = str(index.iloc[row].trial_id)
            if trial_id in trial_quality.index:
                usable[local] = trial_quality.loc[trial_id].to_numpy(dtype=bool)
            else:
                usable[local] = True
        self.mask &= torch.from_numpy(usable)[:, None]

    def get(
        self,
        rows: torch.Tensor | np.ndarray | list[int],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """전역 cache 행 번호를 모델 입력 tensor로 변환한다."""
        local = torch.tensor([
            self.position[int(row)] for row in np.asarray(rows, dtype=np.int64)
        ]).long()
        return (
            self.csi[local].to(device, non_blocking=True).float(),
            self.mask[local].to(device, non_blocking=True),
        )


def select_support(
    rows: np.ndarray,
    index: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """기본동작별 고정 수의 trial을 seed 순서로 선택한다."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in ACTIVE_PROMPT_CLASSES:
        candidates = rows[index.class_id.iloc[rows].to_numpy() == class_id]
        candidates = candidates[
            np.argsort(index.trial_id.iloc[candidates].to_numpy())
        ]
        shots = PROMPT_SHOTS[class_id]
        if len(candidates) < shots:
            raise RuntimeError(f"class {class_id} has too few support trials")
        selected.extend(rng.permutation(candidates)[:shots].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def select_absence(
    site: str,
    index: pd.DataFrame,
    seed: int,
    trials: int = 2,
) -> np.ndarray:
    """현재 site의 absence trial에서 지정한 수만큼 빈 공간 기준선을 고른다."""
    if trials < 1:
        raise ValueError("absence trial count must be positive")
    subject, environment = site.split("_")
    keep = (
        (index.subject == subject)
        & (index.environment == environment)
        & (index.task == C.TASK_CLS)
        & (index.class_id == 6)
        & index.cache_ok
    )
    candidates = np.flatnonzero(keep.to_numpy())
    candidates = candidates[
        np.argsort(index.trial_id.iloc[candidates].to_numpy())
    ]
    if len(candidates) < trials:
        raise RuntimeError(
            f"{site} has fewer than {trials} absence trials"
        )
    return np.random.default_rng(seed).permutation(candidates)[:trials]


def balanced_batches(
    rows: np.ndarray,
    index: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """각 action class가 epoch마다 비슷한 빈도로 등장하도록 재표집한다."""
    labels = torch.tensor(index.class_id.iloc[rows].to_numpy()).long()
    generator = torch.Generator().manual_seed(seed)
    frequency = torch.bincount(
        labels, minlength=C.N_CLASSES
    ).float().clamp_min(1.0)
    draw = torch.multinomial(
        (1.0 / frequency)[labels], len(rows), replacement=True,
        generator=generator,
    ).numpy()
    shuffled = rows[draw]
    return [
        shuffled[start:start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]


def augment_site(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """한 site의 absence/support/query에 동일한 합성 RF 변화를 적용한다."""
    device = tensors[0][0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    gain = torch.exp(0.25 * torch.randn(
        C.N_LINKS, generator=generator, device=device
    ))
    curvature = 0.30 * torch.randn(
        C.N_LINKS, generator=generator, device=device
    )
    ripple = 0.15 * torch.randn(
        C.N_LINKS, generator=generator, device=device
    )
    frequency = torch.linspace(
        -1.0, 1.0, C.N_LIVE_SUBCARRIERS, device=device
    )
    phase_shift = (
        curvature[:, None]
        * (frequency.square() - frequency.square().mean())[None]
        + ripple[:, None] * torch.sin(math.pi * frequency)[None]
    )
    drop = None
    if float(torch.rand((), generator=generator, device=device)) < 0.30:
        drop = int(torch.randint(
            C.N_LINKS, (1,), generator=generator, device=device
        ).item())
    augmented = []
    for csi, mask in tensors:
        values = csi.clone()
        local_mask = mask.clone()
        values[..., 0] *= gain[None, None, :, None]
        values[..., 1] += phase_shift[None, None]
        if drop is not None:
            local_mask[:, :, drop] = False
        values *= local_mask[..., None, None].to(values.dtype)
        augmented.append((values, local_mask))
    return augmented


def site_rows(
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
) -> np.ndarray:
    """선택된 source protocol에서 특정 site의 cache 행을 반환한다."""
    return selected_rows[sites == site]
