"""KP v2 설계 전에 CSI의 정적 domain 정보와 동적 행동 정보를 분리해 측정한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))


def source_rows(index: pd.DataFrame) -> np.ndarray:
    """봉인 yja와 lmh E02/E03을 제외한 고정 source pose query만 선택한다."""
    keep = (
        index.subject.isin(("ajh", "mhw", "lmh"))
        & ~((index.subject == "lmh") & (index.environment != "E01"))
        & (index.task == "pose_and_action")
        & (index.class_id != 6)
        & index.cache_ok
        & (index.role == "train")
    )
    rows = np.flatnonzero(keep.to_numpy()).astype(np.int64)
    if "yja" in set(index.subject.iloc[rows].astype(str)):
        raise RuntimeError("sealed yja entered KP v2 diagnosis")
    return rows


def _masked_moments(
    values: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """시간축의 유효 프레임만 사용해 링크·subcarrier 평균과 표준편차를 구한다."""
    weight = mask[..., None].astype(np.float32)
    count = weight.sum(1).clip(1.0)
    mean = (values * weight).sum(1) / count
    variance = (np.square(values - mean[:, None]) * weight).sum(1) / count
    return mean, np.sqrt(np.maximum(variance, 0.0))


def extract_views(
    csi_path: Path,
    mask_path: Path,
    rows: np.ndarray,
    batch_size: int = 16,
    progress_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """정적 RF, 미분 동역학, 시간순서 보존 움직임 profile을 각각 추출한다."""
    csi = np.load(csi_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    static_parts: list[np.ndarray] = []
    dynamic_parts: list[np.ndarray] = []
    progress_parts: list[np.ndarray] = []
    invariant_parts: list[np.ndarray] = []
    invariant_amp_parts: list[np.ndarray] = []
    invariant_phase_parts: list[np.ndarray] = []

    for start in range(0, len(rows), batch_size):
        current = rows[start:start + batch_size]
        values = np.asarray(csi[current], dtype=np.float32)
        mask = np.asarray(masks[current], dtype=bool)
        amplitude = values[..., 0]
        phase = values[..., 1]

        amp_mean, amp_std = _masked_moments(amplitude, mask)
        phase_sin, _ = _masked_moments(np.sin(phase), mask)
        phase_cos, _ = _masked_moments(np.cos(phase), mask)
        static_parts.append(np.concatenate((
            amp_mean, amp_std, phase_sin, phase_cos,
        ), axis=-1).reshape(len(current), -1))

        pair = mask[:, 1:] & mask[:, :-1]
        pair_weight = pair[..., None].astype(np.float32)
        pair_count = pair_weight.sum(1).clip(1.0)
        scale = np.maximum(np.abs(amp_mean), 1e-3)
        amplitude_delta = (
            amplitude[:, 1:] - amplitude[:, :-1]
        ) / scale[:, None]
        phase_delta = np.arctan2(
            np.sin(phase[:, 1:] - phase[:, :-1]),
            np.cos(phase[:, 1:] - phase[:, :-1]),
        )

        def derivative_summary(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            absolute = (np.abs(delta) * pair_weight).sum(1) / pair_count
            rms = np.sqrt(
                (np.square(delta) * pair_weight).sum(1) / pair_count
            )
            return absolute, rms

        amp_abs, amp_rms = derivative_summary(amplitude_delta)
        phase_abs, phase_rms = derivative_summary(phase_delta)
        dynamic_parts.append(np.concatenate((
            amp_abs, amp_rms, phase_abs, phase_rms,
        ), axis=-1).reshape(len(current), -1))

        amp_energy = np.sqrt(np.mean(np.square(amplitude_delta), axis=-1))
        phase_energy = np.sqrt(np.mean(np.square(phase_delta), axis=-1))
        energy = np.stack((amp_energy, phase_energy), axis=-1)
        energy *= pair[..., None].astype(np.float32)
        acceleration = np.zeros_like(amp_energy)
        acceleration[:, 1:] = np.abs(amp_energy[:, 1:] - amp_energy[:, :-1])
        profile = np.concatenate((energy, acceleration[..., None]), axis=-1)
        profile = profile.transpose(0, 3, 2, 1).reshape(
            len(current), -1, profile.shape[1]
        )
        pooled = F.interpolate(
            torch.from_numpy(profile), size=progress_bins,
            mode="linear", align_corners=True,
        ).numpy()
        progress_parts.append(pooled.reshape(len(current), -1))

        profile_btlc = profile.reshape(
            len(current), 3, mask.shape[2], profile.shape[-1]
        ).transpose(0, 3, 2, 1)
        profile_mask = pair[..., None].astype(np.float32)
        profile_mean = (
            (profile_btlc * profile_mask).sum(1, keepdims=True)
            / profile_mask.sum(1, keepdims=True).clip(1.0)
        )
        shape_profile = np.log1p(
            np.clip(profile_btlc / np.maximum(profile_mean, 1e-4), 0.0, 20.0)
        ) * profile_mask
        shape_tensor = torch.from_numpy(shape_profile)
        activity = shape_tensor[..., :2].sum((2, 3))
        activity = activity + 1e-4 * torch.from_numpy(pair.any(-1)).float()
        coordinate = activity.cumsum(1)
        coordinate = coordinate / coordinate[:, -1:].clamp_min(1e-6)
        centers = torch.linspace(0.0, 1.0, progress_bins)
        bandwidth = 0.75 / max(progress_bins - 1, 1)
        weight = torch.exp(
            -0.5 * ((coordinate[..., None] - centers[None, None]) / bandwidth).square()
        )
        weight *= torch.from_numpy(pair.any(-1)).float()[..., None]
        weight /= weight.sum(1, keepdim=True).clamp_min(1e-6)
        progress_shape = torch.einsum(
            "btp,btlc->bplc", weight, shape_tensor
        ).flatten(1)
        spectrum = torch.fft.rfft(shape_tensor, dim=1).abs()[:, 1:17]
        spectrum /= spectrum.sum(1, keepdim=True).clamp_min(1e-6)
        invariant_parts.append(torch.cat((
            progress_shape, spectrum.flatten(1),
        ), dim=1).numpy())
        amp_channels = torch.tensor((0, 2), dtype=torch.long)
        phase_channels = torch.tensor((1,), dtype=torch.long)
        invariant_amp_parts.append(torch.cat((
            progress_shape.reshape(
                len(current), progress_bins, mask.shape[2], 3
            ).index_select(-1, amp_channels).flatten(1),
            spectrum.index_select(-1, amp_channels).flatten(1),
        ), dim=1).numpy())
        invariant_phase_parts.append(torch.cat((
            progress_shape.reshape(
                len(current), progress_bins, mask.shape[2], 3
            ).index_select(-1, phase_channels).flatten(1),
            spectrum.index_select(-1, phase_channels).flatten(1),
        ), dim=1).numpy())

    return tuple(
        np.concatenate(parts).astype(np.float32)
        for parts in (
            static_parts, dynamic_parts, progress_parts, invariant_parts,
            invariant_amp_parts, invariant_phase_parts,
        )
    )


def classifier(seed: int) -> ExtraTreesClassifier:
    """작은 데이터에서 view의 분리 가능성만 재는 고정 비선형 probe를 만든다."""
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def random_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """동일 source domain이 섞인 5-fold에서 표현의 정보량을 측정한다."""
    split = StratifiedKFold(5, shuffle=True, random_state=seed)
    prediction = cross_val_predict(
        classifier(seed), features, labels, cv=split, n_jobs=1,
    )
    return {
        "accuracy": 100.0 * accuracy_score(labels, prediction),
        "macro_f1": 100.0 * f1_score(labels, prediction, average="macro"),
    }


def subject_loso_probe(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    seed: int,
) -> dict:
    """한 사람 전체를 숨겨 행동 표현의 실제 cross-person 전이를 측정한다."""
    folds = {}
    predictions = np.full(len(labels), -1, dtype=np.int64)
    for number, subject in enumerate(sorted(set(subjects.tolist()))):
        test = subjects == subject
        train = ~test
        model = classifier(seed + number)
        model.fit(features[train], labels[train])
        prediction = model.predict(features[test])
        predictions[test] = prediction
        folds[str(subject)] = {
            "accuracy": 100.0 * accuracy_score(labels[test], prediction),
            "macro_f1": 100.0 * f1_score(
                labels[test], prediction, average="macro",
            ),
            "trials": int(test.sum()),
        }
    return {
        "accuracy": 100.0 * accuracy_score(labels, predictions),
        "macro_f1": 100.0 * f1_score(labels, predictions, average="macro"),
        "folds": folds,
    }


def main() -> None:
    """세 CSI view의 domain 누출과 행동 전이를 같은 protocol로 보고한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26081)
    parser.add_argument("--batch-size", type=int, default=16)
    options = parser.parse_args()

    index = pd.read_csv(WORK / "cache/cache_index.csv")
    rows = source_rows(index)
    selected = index.iloc[rows].reset_index(drop=True)
    static, dynamic, progress, invariant, invariant_amp, invariant_phase = extract_views(
        WORK / "cache/csi_iq.npy", WORK / "cache/link_mask.npy",
        rows, options.batch_size,
    )
    views = {
        "static_rf": static,
        "derivative_motion": dynamic,
        "ordered_progress": progress,
        "scale_free_progress_frequency": invariant,
        "scale_free_amplitude_only": invariant_amp,
        "scale_free_phase_only": invariant_phase,
        "motion_combined": np.concatenate((dynamic, progress), axis=1),
    }
    action = selected.class_id.to_numpy(dtype=np.int64)
    risk = selected.risk_id.to_numpy(dtype=np.int64)
    subjects = selected.subject.astype(str).to_numpy()
    sites = (
        selected.subject.astype(str) + "_" + selected.environment.astype(str)
    ).to_numpy()
    subject_ids = pd.factorize(subjects, sort=True)[0]
    site_ids = pd.factorize(sites, sort=True)[0]

    result = {
        "protocol": "source-only signal view diagnosis; yja sealed",
        "trials": len(rows),
        "subjects": sorted(set(subjects.tolist())),
        "sites": sorted(set(sites.tolist())),
        "feature_dimensions": {name: value.shape[1] for name, value in views.items()},
        "views": {},
        "sealed_yja_used": False,
        "target_subject_used": False,
    }
    for number, (name, features) in enumerate(views.items()):
        seed = options.seed + number * 101
        result["views"][name] = {
            "random_action": random_probe(features, action, seed),
            "subject_loso_action": subject_loso_probe(
                features, action, subjects, seed,
            ),
            "subject_loso_risk": subject_loso_probe(
                features, risk, subjects, seed,
            ),
            "random_subject_probe": random_probe(features, subject_ids, seed),
            "random_site_probe": random_probe(features, site_ids, seed),
        }
        print(name, json.dumps(result["views"][name], indent=2), flush=True)

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"wrote {options.output}")


if __name__ == "__main__":
    main()
