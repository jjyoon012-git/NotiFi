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


def mask_subcarrier_band(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    fraction: float = 0.12,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Hide one shared contiguous RF band without changing the episode action."""
    if not tensors:
        raise ValueError("subcarrier masking requires at least one tensor")
    if not 0.0 < fraction < 1.0:
        raise ValueError("subcarrier mask fraction must be between 0 and 1")
    reference_csi, reference_mask = tensors[0]
    if reference_csi.ndim != 5 or reference_mask.shape != reference_csi.shape[:3]:
        raise ValueError("expected CSI [B,T,L,S,2] and mask [B,T,L]")
    subcarriers = int(reference_csi.shape[3])
    width = max(1, min(subcarriers - 1, int(round(subcarriers * fraction))))
    generator = torch.Generator(device=reference_csi.device).manual_seed(seed)
    start = int(torch.randint(
        0, subcarriers - width + 1, (1,), generator=generator,
        device=reference_csi.device,
    ).item())
    stop = start + width

    amplitude, phase, available = _static_reference(
        reference_csi, reference_mask
    )
    masked = []
    for csi, mask in tensors:
        if csi.shape[2:] != reference_csi.shape[2:]:
            raise ValueError("all episode tensors must share link/subcarrier shape")
        values = csi.clone()
        values[:, :, :, start:stop, 0] = amplitude[
            None, None, :, start:stop
        ]
        values[:, :, :, start:stop, 1] = phase[
            None, None, :, start:stop
        ]
        local_mask = mask.clone()
        local_mask[..., ~available] = False
        values *= local_mask[..., None, None].to(values.dtype)
        masked.append((values, local_mask))
    return masked


def reflect_east_west(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """TX2(West)와 TX3(East)를 함께 바꿔 좌우 반사 episode를 만든다."""
    reflected = []
    for csi, mask in tensors:
        if csi.ndim != 5 or mask.ndim != 3 or csi.shape[2] != C.N_LINKS:
            raise ValueError("expected CSI [B,T,3,S,2] and mask [B,T,3]")
        order = torch.tensor((0, 2, 1), device=csi.device)
        reflected.append((
            csi.index_select(2, order), mask.index_select(2, order),
        ))
    return reflected


def drop_episode_link(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """absence·support·query에서 같은 TX 하나를 가려 링크 손실 episode를 만든다."""
    if not tensors:
        raise ValueError("link dropout requires at least one tensor")
    reference_csi, reference_mask = tensors[0]
    if (
        reference_csi.ndim != 5
        or reference_mask.shape != reference_csi.shape[:3]
        or reference_csi.shape[2] != C.N_LINKS
    ):
        raise ValueError("expected CSI [B,T,3,S,2] and mask [B,T,3]")
    generator = torch.Generator(device=reference_csi.device).manual_seed(seed)
    link = int(torch.randint(
        0, C.N_LINKS, (1,), generator=generator,
        device=reference_csi.device,
    ).item())
    dropped = []
    for csi, mask in tensors:
        if csi.shape[2:] != reference_csi.shape[2:]:
            raise ValueError("all episode tensors must share link/subcarrier shape")
        values = csi.clone()
        local_mask = mask.clone()
        values[:, :, link] = 0.0
        local_mask[:, :, link] = False
        dropped.append((values, local_mask))
    return dropped


def temporal_warp_trials(
    csi: torch.Tensor,
    mask: torch.Tensor,
    seed: int,
    strength: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """동작 순서를 보존한 채 trial별 수행 속도만 단조롭게 재표집한다."""
    if csi.ndim != 5 or mask.shape != csi.shape[:3]:
        raise ValueError("expected CSI [B,T,L,S,2] and mask [B,T,L]")
    if strength < 0.0:
        raise ValueError("temporal warp strength cannot be negative")
    batch, frames = csi.shape[:2]
    generator = torch.Generator(device=csi.device).manual_seed(seed)
    exponent = torch.exp(
        (torch.rand(batch, generator=generator, device=csi.device) * 2.0 - 1.0)
        * float(strength)
    )
    timeline = torch.linspace(0.0, 1.0, frames, device=csi.device)
    source = (timeline[None].pow(exponent[:, None]) * (frames - 1)).round().long()
    csi_index = source[:, :, None, None, None].expand(
        -1, -1, csi.shape[2], csi.shape[3], csi.shape[4],
    )
    mask_index = source[:, :, None].expand(-1, -1, mask.shape[2])
    return torch.gather(csi, 1, csi_index), torch.gather(mask, 1, mask_index)


def _static_reference(
    csi: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """absence CSI에서 link/subcarrier별 진폭과 원형 위상 기준선을 계산한다."""
    weight = mask.to(csi.dtype)[..., None]
    denominator = weight.sum((0, 1)).clamp_min(1.0)
    amplitude = (csi[..., 0] * weight).sum((0, 1)) / denominator
    sine = (torch.sin(csi[..., 1]) * weight).sum((0, 1)) / denominator
    cosine = (torch.cos(csi[..., 1]) * weight).sum((0, 1)) / denominator
    phase = torch.atan2(sine, cosine)
    available = mask.any((0, 1))
    return amplitude.clamp_min(1e-4), phase, available


def transfer_site_style(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    donor_absence: tuple[torch.Tensor, torch.Tensor],
    strength: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """동작 잔차는 보존하고 한 episode의 정적 반사 기준선만 donor site로 옮긴다."""
    if not tensors:
        raise ValueError("site style transfer requires at least one tensor")
    source_amplitude, source_phase, source_available = _static_reference(
        tensors[0][0], tensors[0][1]
    )
    donor_amplitude, donor_phase, donor_available = _static_reference(
        donor_absence[0], donor_absence[1]
    )
    usable = source_available & donor_available
    amplitude_ratio = (donor_amplitude / source_amplitude).clamp(0.5, 2.0)
    amplitude_ratio = torch.where(
        usable[:, None], amplitude_ratio, torch.ones_like(amplitude_ratio)
    ).pow(float(strength))
    phase_delta = torch.atan2(
        torch.sin(donor_phase - source_phase),
        torch.cos(donor_phase - source_phase),
    )
    phase_delta = torch.where(
        usable[:, None], phase_delta, torch.zeros_like(phase_delta)
    ) * float(strength)
    transferred = []
    for csi, mask in tensors:
        values = csi.clone()
        values[..., 0] *= amplitude_ratio[None, None]
        values[..., 1] += phase_delta[None, None]
        values *= mask[..., None, None].to(values.dtype)
        transferred.append((values, mask.clone()))
    return transferred


def site_rows(
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
) -> np.ndarray:
    """선택된 source protocol에서 특정 site의 cache 행을 반환한다."""
    return selected_rows[sites == site]
