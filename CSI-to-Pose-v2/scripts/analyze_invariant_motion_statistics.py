"""장비 이득을 제거한 CSI motion statistics의 source subject-LOSO 상한을 측정한다."""

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
from sklearn.metrics import accuracy_score, f1_score, recall_score


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))


def source_rows(index: pd.DataFrame) -> np.ndarray:
    """yja 및 lmh E02/E03을 제외한 source 7-site의 16-action query만 선택한다."""
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
        raise RuntimeError("sealed yja entered invariant statistics diagnosis")
    return rows


def interpolate_profile(profile: np.ndarray, bins: int) -> np.ndarray:
    """[B,T,L,C] profile을 고정 시간 bin으로 선형 보간한다."""
    tensor = torch.from_numpy(profile).permute(0, 2, 3, 1).flatten(1, 2)
    pooled = F.interpolate(tensor, size=bins, mode="linear", align_corners=True)
    return pooled.numpy().reshape(len(profile), profile.shape[2], profile.shape[3], bins)


def progress_profile(
    profile: np.ndarray, valid: np.ndarray, bins: int,
) -> np.ndarray:
    """누적 motion energy 좌표에서 profile을 Gaussian pooling한다."""
    tensor = torch.from_numpy(profile)
    valid_tensor = torch.from_numpy(valid).float()
    activity = tensor.sum((2, 3)) * valid_tensor
    activity = activity + 1e-4 * valid_tensor
    coordinate = activity.cumsum(1)
    coordinate = coordinate / coordinate[:, -1:].clamp_min(1e-6)
    centers = torch.linspace(0.0, 1.0, bins)
    bandwidth = 0.75 / max(bins - 1, 1)
    weight = torch.exp(
        -0.5 * ((coordinate[..., None] - centers[None, None]) / bandwidth).square()
    ) * valid_tensor[..., None]
    weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("btp,btlc->blcp", weight, tensor).numpy()


def autocorrelation(profile: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """trial-normalized motion curve의 여러 lag 자기상관을 계산한다."""
    centered = profile - profile.mean(1, keepdims=True)
    scale = np.sqrt(np.square(centered).mean(1, keepdims=True)).clip(1e-5)
    centered = centered / scale
    values = []
    for lag in lags:
        values.append((centered[:, lag:] * centered[:, :-lag]).mean(1))
    return np.stack(values, axis=-1)


def frequency_ratios(profile: np.ndarray, bands: tuple[tuple[int, int], ...]) -> np.ndarray:
    """각 링크·channel의 temporal FFT 에너지를 전체 에너지 비율로 바꾼다."""
    spectrum = np.square(np.abs(np.fft.rfft(profile, axis=1)))
    spectrum[:, 0] = 0.0
    total = spectrum.sum(1, keepdims=True).clip(1e-8)
    return np.stack([
        spectrum[:, left:right].sum(1) / total[:, 0]
        for left, right in bands
    ], axis=-1).astype(np.float32)


def cross_link_correlation(profile: np.ndarray) -> np.ndarray:
    """세 link에서 관측된 동작 에너지 곡선의 pairwise 상관을 계산한다."""
    activity = profile.sum(-1)
    activity = activity - activity.mean(1, keepdims=True)
    activity /= np.sqrt(np.square(activity).mean(1, keepdims=True)).clip(1e-5)
    return np.stack([
        (activity[..., left] * activity[..., right]).mean(1)
        for left, right in ((0, 1), (0, 2), (1, 2))
    ], axis=-1)


def extract_features(
    csi_path: Path, mask_path: Path, rows: np.ndarray, batch_size: int, bins: int,
) -> np.ndarray:
    """상대 I/Q 변화에서 robust motion curve와 scale-free 통계만 추출한다."""
    csi = np.load(csi_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    features = []
    quantiles = (0.25, 0.50, 0.75, 0.90)
    for start in range(0, len(rows), batch_size):
        current = rows[start:start + batch_size]
        values = np.asarray(csi[current], dtype=np.float32)
        mask = np.asarray(masks[current], dtype=bool)
        amplitude = values[..., 0]
        phase = values[..., 1]
        valid = (mask[:, 1:] & mask[:, :-1]).any(-1)
        link_valid = mask[:, 1:] & mask[:, :-1]

        amp_level = np.median(np.abs(amplitude), axis=1).clip(1e-3)
        amp_delta = np.abs(amplitude[:, 1:] - amplitude[:, :-1]) / amp_level[:, None]
        phase_delta = np.abs(np.arctan2(
            np.sin(phase[:, 1:] - phase[:, :-1]),
            np.cos(phase[:, 1:] - phase[:, :-1]),
        ))
        amp_q = np.quantile(amp_delta, quantiles, axis=-1).transpose(1, 2, 3, 0)
        phase_q = np.quantile(phase_delta, quantiles, axis=-1).transpose(1, 2, 3, 0)
        profile = np.concatenate((amp_q, phase_q), axis=-1).astype(np.float32)
        profile *= link_valid[..., None].astype(np.float32)

        # 절대 RF 크기를 버리고 trial 내부의 motion shape만 남긴다.
        level = np.quantile(profile, 0.75, axis=1, keepdims=True).clip(1e-4)
        normalized = np.log1p(np.clip(profile / level, 0.0, 30.0))
        normalized *= link_valid[..., None].astype(np.float32)
        clock = interpolate_profile(normalized, bins)
        progress = progress_profile(normalized, valid, bins)
        auto = autocorrelation(normalized, (1, 2, 4, 8, 16, 32))
        frequency = frequency_ratios(
            normalized, ((1, 3), (3, 7), (7, 15), (15, 31), (31, 61)),
        )
        synchrony = cross_link_correlation(normalized)

        activity = normalized.sum((2, 3)) * valid.astype(np.float32)
        activity_sum = activity.sum(1).clip(1e-6)
        timeline = np.linspace(0.0, 1.0, activity.shape[1], dtype=np.float32)
        centroid = (activity * timeline[None]).sum(1) / activity_sum
        spread = np.sqrt(
            (activity * np.square(timeline[None] - centroid[:, None])).sum(1)
            / activity_sum
        )
        maximum = activity.max(1) / np.quantile(activity, 0.75, axis=1).clip(1e-4)
        global_shape = np.stack((centroid, spread, maximum), axis=-1)
        features.append(np.concatenate((
            clock.reshape(len(current), -1),
            progress.reshape(len(current), -1),
            auto.reshape(len(current), -1),
            frequency.reshape(len(current), -1),
            synchrony.reshape(len(current), -1),
            global_shape,
        ), axis=-1).astype(np.float32))
    return np.concatenate(features)


def classifier(seed: int) -> ExtraTreesClassifier:
    """표현의 비선형 분리 가능성을 재는 고정 balanced probe를 만든다."""
    return ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )


def grouped_probe(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int,
    danger_label: int | None = None,
) -> dict:
    """한 사람 전체를 숨긴 outer fold별 예측과 macro 지표를 계산한다."""
    prediction = np.full(len(labels), -1, dtype=np.int64)
    folds = {}
    for number, group in enumerate(sorted(set(groups.tolist()))):
        test = groups == group
        model = classifier(seed + number)
        model.fit(features[~test], labels[~test])
        current = model.predict(features[test])
        prediction[test] = current
        folds[str(group)] = {
            "accuracy": 100.0 * accuracy_score(labels[test], current),
            "macro_f1": 100.0 * f1_score(labels[test], current, average="macro"),
        }
    result = {
        "accuracy": 100.0 * accuracy_score(labels, prediction),
        "macro_f1": 100.0 * f1_score(labels, prediction, average="macro"),
        "folds": folds,
    }
    if danger_label is not None:
        result["danger_recall"] = 100.0 * recall_score(
            labels == danger_label, prediction == danger_label, zero_division=0,
        )
    return result


def main() -> None:
    """새 통계 표현의 action/risk 및 subject/site 지문을 source-only로 보고한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26089)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bins", type=int, default=24)
    options = parser.parse_args()
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    rows = source_rows(index)
    selected = index.iloc[rows].reset_index(drop=True)
    features = extract_features(
        WORK / "cache/csi_iq.npy", WORK / "cache/link_mask.npy",
        rows, options.batch_size, options.bins,
    )
    subject = selected.subject.astype(str).to_numpy()
    site = (selected.subject.astype(str) + "_" + selected.environment.astype(str)).to_numpy()
    action = selected.class_id.to_numpy(dtype=np.int64)
    risk = selected.risk_id.to_numpy(dtype=np.int64)
    subject_id = pd.factorize(subject, sort=True)[0]
    site_id = pd.factorize(site, sort=True)[0]
    result = {
        "protocol": "source-only scale-free motion-statistics diagnosis; subject LOSO; yja sealed",
        "trials": len(rows),
        "feature_dimension": int(features.shape[1]),
        "action": grouped_probe(features, action, subject, options.seed),
        "risk": grouped_probe(
            features, risk, subject, options.seed + 101, danger_label=2,
        ),
        "subject_probe_site_loso": grouped_probe(features, subject_id, site, options.seed + 202),
        "site_probe_subject_loso": grouped_probe(features, site_id, subject, options.seed + 303),
        "target_subject_used": False,
        "sealed_yja_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
