"""CAL12 설계 전에 CSI의 환경·사람·동작·잡음 성분을 분해한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import source_calibration_data as base


PROMPT_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)
SOURCE_SITES = (
    "ajh_E01", "ajh_E02", "ajh_E03",
    "mhw_E01", "mhw_E02", "mhw_E03", "lmh_E01",
)
TARGET_SITE = "yja_E02"
N_LINKS = 3
FPS = 30.0


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Boolean mask를 적용한 평균을 0으로 나누지 않고 계산한다."""
    weight = mask.astype(np.float32)
    return (values * weight).sum(axis) / np.maximum(weight.sum(axis), 1.0)


def masked_std(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Boolean mask를 적용한 표준편차를 계산한다."""
    mean = np.expand_dims(masked_mean(values, mask, axis), axis)
    weight = mask.astype(np.float32)
    variance = ((values - mean) ** 2 * weight).sum(axis)
    variance /= np.maximum(weight.sum(axis), 1.0)
    return np.sqrt(np.maximum(variance, 0.0))


def robust_trial_features(csi: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """한 묶음의 raw CSI를 물리 의미가 다른 정적·동적·잡음 특징으로 요약한다."""
    amplitude = np.log1p(np.maximum(csi[..., 0].astype(np.float32), 0.0))
    phase = csi[..., 1].astype(np.float32)
    batch, frames, links, subcarriers = amplitude.shape
    names: list[str] = []
    columns: list[np.ndarray] = []

    for link in range(links):
        valid = mask[:, :, link]
        sample_mask = valid[:, :, None]
        amp = amplitude[:, :, link]
        pha = phase[:, :, link]

        # 시간 평균 subcarrier profile은 벽·가구·설치 구조의 정적 반사를 강하게 담는다.
        profile = masked_mean(amp, sample_mask, axis=1)
        profile_centered = profile - profile.mean(1, keepdims=True)
        static_mean = profile.mean(1)
        static_shape = profile_centered.std(1)
        static_slope = np.mean(
            profile_centered * np.linspace(-1.0, 1.0, subcarriers)[None], axis=1
        )

        # trial 내부 중심을 제거한 변화량은 사람의 움직임에 더 가깝다.
        centered = amp - profile[:, None]
        centered *= sample_mask
        dynamic_std = masked_std(centered, sample_mask, axis=(1, 2))
        delta_mask = valid[:, 1:] & valid[:, :-1]
        delta_amp = np.diff(centered, axis=1)
        delta_phase = np.angle(np.exp(1j * np.diff(pha, axis=1))).astype(np.float32)
        delta_sample_mask = delta_mask[:, :, None]
        amp_velocity = masked_mean(np.abs(delta_amp), delta_sample_mask, axis=(1, 2))
        phase_velocity = masked_mean(
            np.abs(delta_phase), delta_sample_mask, axis=(1, 2)
        )
        amp_acceleration = np.diff(delta_amp, axis=1)
        acceleration_mask = delta_mask[:, 1:] & delta_mask[:, :-1]
        amp_acceleration = masked_mean(
            np.abs(amp_acceleration), acceleration_mask[:, :, None], axis=(1, 2)
        )

        # 고정 30 Hz 격자에서 주파수 대역별 에너지를 계산한다.
        centered = centered - centered.mean(1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        frequency = np.fft.rfftfreq(frames, d=1.0 / FPS)
        total = spectrum[:, 1:].mean((1, 2)) + 1e-6
        band_values = []
        for low, high in ((0.10, 0.75), (0.75, 2.0), (2.0, 5.0), (5.0, 12.0)):
            keep = (frequency >= low) & (frequency < high)
            band_values.append(spectrum[:, keep].mean((1, 2)) / total)

        # 링크 가용률과 frame gap은 모델이 동작으로 오인하면 안 되는 수집 잡음이다.
        coverage = valid.mean(1)
        transition = np.not_equal(valid[:, 1:], valid[:, :-1]).mean(1)
        values = (
            static_mean, static_shape, static_slope, dynamic_std,
            amp_velocity, phase_velocity, amp_acceleration,
            *band_values, coverage, transition,
        )
        labels = (
            "static_mean", "static_shape", "static_slope", "dynamic_std",
            "amp_velocity", "phase_velocity", "amp_acceleration",
            "band_0p1_0p75", "band_0p75_2", "band_2_5", "band_5_12",
            "coverage", "mask_transition",
        )
        columns.extend(values)
        names.extend([f"tx{link + 1}_{label}" for label in labels])

    # 링크 간 운동 에너지의 비와 상관은 설치 방향보다 몸 움직임의 공간 패턴을 나타낸다.
    frame_energy = []
    for link in range(N_LINKS):
        profile = masked_mean(
            amplitude[:, :, link], mask[:, :, link, None], axis=1
        )
        residual = amplitude[:, :, link] - profile[:, None]
        frame_energy.append(np.sqrt(np.mean(residual ** 2, axis=2) + 1e-8))
    frame_energy = np.stack(frame_energy, axis=-1)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        numerator = np.sum(
            (frame_energy[:, :, left] - frame_energy[:, :, left].mean(1, keepdims=True))
            * (frame_energy[:, :, right] - frame_energy[:, :, right].mean(1, keepdims=True)),
            axis=1,
        )
        denominator = (
            np.linalg.norm(
                frame_energy[:, :, left] - frame_energy[:, :, left].mean(1, keepdims=True),
                axis=1,
            )
            * np.linalg.norm(
                frame_energy[:, :, right] - frame_energy[:, :, right].mean(1, keepdims=True),
                axis=1,
            )
            + 1e-6
        )
        columns.append(numerator / denominator)
        names.append(f"motion_corr_tx{left + 1}_tx{right + 1}")

    return np.stack(columns, axis=1).astype(np.float32), names


def grouped_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """한 사람 또는 환경 전체를 holdout하여 선형 probe 일반화 성능을 측정한다."""
    fold_metrics = []
    predictions = np.full(len(labels), -1, dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced"),
        )
        model.fit(features[train], labels[train])
        predictions[test] = model.predict(features[test])
        fold_metrics.append({
            "held_out": str(group),
            "accuracy": float(accuracy_score(labels[test], predictions[test])),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels[test], predictions[test])
            ),
            "macro_f1": float(f1_score(
                labels[test], predictions[test], average="macro", zero_division=0
            )),
            "samples": int(test.sum()),
        })
    valid = predictions >= 0
    return {
        "folds": fold_metrics,
        "pooled_accuracy": float(accuracy_score(labels[valid], predictions[valid])),
        "pooled_balanced_accuracy": float(
            balanced_accuracy_score(labels[valid], predictions[valid])
        ),
        "pooled_macro_f1": float(f1_score(
            labels[valid], predictions[valid], average="macro", zero_division=0
        )),
    }


def random_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict:
    """특징 안에 subject/site 지문이 얼마나 쉽게 남는지 반복 holdout으로 측정한다."""
    rng = np.random.default_rng(seed)
    metrics = []
    for repeat in range(5):
        order = rng.permutation(len(labels))
        split = int(round(len(order) * 0.75))
        train, test = order[:split], order[split:]
        model = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=0.1, random_state=seed + repeat,
        )
        model.fit(features[train], labels[train])
        prediction = model.predict(features[test])
        metrics.append({
            "accuracy": float(accuracy_score(labels[test], prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels[test], prediction)
            ),
        })
    return {
        "accuracy_mean": float(np.mean([item["accuracy"] for item in metrics])),
        "balanced_accuracy_mean": float(np.mean([
            item["balanced_accuracy"] for item in metrics
        ])),
        "repeats": metrics,
    }


def eta_squared(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """각 특징 분산 중 범주 집단 평균 차이가 설명하는 비율을 계산한다."""
    overall = features.mean(0)
    total = np.square(features - overall).sum(0) + 1e-8
    between = np.zeros(features.shape[1], dtype=np.float64)
    for label in sorted(set(labels.tolist())):
        values = features[labels == label]
        between += len(values) * np.square(values.mean(0) - overall)
    return (between / total).astype(np.float32)


def choose_prompt_rows(index: pd.DataFrame, site: str) -> np.ndarray:
    """배포 때 사용자가 수행한다고 가정한 기본 자세·전환 trial만 고정 선택한다."""
    rows = np.flatnonzero(
        ((index.subject + "_" + index.environment) == site)
        & (index.task == "pose_and_action")
        & index.class_id.isin(PROMPT_CLASSES)
        & index.cache_ok
    )
    selected = []
    for class_id in PROMPT_CLASSES:
        candidates = rows[index.class_id.iloc[rows].to_numpy() == class_id]
        candidates = candidates[np.argsort(index.trial_id.iloc[candidates].to_numpy())]
        selected.extend(candidates[:2].tolist())
    return np.asarray(selected, dtype=np.int64)


def select_absence_rows(index: pd.DataFrame, sites: tuple[str, ...]) -> np.ndarray:
    """각 현장의 classification-only absence trial 행을 반환한다."""
    site = index.subject + "_" + index.environment
    return np.flatnonzero(
        site.isin(sites)
        & (index.task == "classification_only")
        & (index.class_id == 6)
        & index.cache_ok
    )


def site_summary(
    features: np.ndarray,
    rows: np.ndarray,
    index: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """현장별 특징 평균과 표준편차를 긴 형식 표로 만든다."""
    metadata = index.iloc[rows]
    sites = (metadata.subject + "_" + metadata.environment).to_numpy()
    records = []
    for site in sorted(set(sites.tolist())):
        values = features[sites == site]
        for column, name in enumerate(feature_names):
            records.append({
                "site": site,
                "feature": name,
                "mean": float(values[:, column].mean()),
                "std": float(values[:, column].std()),
                "samples": int(len(values)),
            })
    return pd.DataFrame(records)


def main() -> None:
    """분해 보고서와 재현 가능한 feature table을 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12012)
    options = parser.parse_args()
    work = options.work_root
    output = options.output
    output.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(work / "cache/cache_index.csv")
    csi = np.load(work / "cache/csi_iq.npy", mmap_mode="r")
    mask = np.load(work / "cache/link_mask.npy", mmap_mode="r")
    source_rows = base.select_source_rows(index)
    source_meta = index.iloc[source_rows]
    source_sites = (source_meta.subject + "_" + source_meta.environment).to_numpy()
    if set(source_sites.tolist()) != set(SOURCE_SITES):
        raise RuntimeError(f"unexpected source protocol: {sorted(set(source_sites))}")
    if "yja" in set(source_meta.subject.astype(str)):
        raise RuntimeError("sealed target must not enter supervised source analysis")

    # yja에서는 사전에 정한 calibration prompt와 absence만 사용한다.
    target_prompt_rows = choose_prompt_rows(index, TARGET_SITE)
    absence_rows = select_absence_rows(index, SOURCE_SITES + (TARGET_SITE,))
    analysis_rows = np.unique(np.concatenate((
        source_rows, target_prompt_rows, absence_rows
    ))).astype(np.int64)

    batches = []
    feature_names = None
    for start in range(0, len(analysis_rows), options.batch_size):
        rows = analysis_rows[start:start + options.batch_size]
        values, names = robust_trial_features(
            np.asarray(csi[rows]), np.asarray(mask[rows])
        )
        batches.append(values)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("feature schema changed between batches")
    features = np.concatenate(batches)
    feature_names = feature_names or []
    row_to_local = {int(row): local for local, row in enumerate(analysis_rows)}
    source_local = np.asarray([row_to_local[int(row)] for row in source_rows])
    source_features = features[source_local]

    source_subject = source_meta.subject.to_numpy()
    source_environment = source_meta.environment.to_numpy()
    source_action = source_meta.class_id.to_numpy(dtype=np.int64)
    balanced_environment = source_meta.subject.isin(("ajh", "mhw")).to_numpy()
    balanced_subject = (source_environment == "E01")
    static_columns = np.asarray([
        name.endswith(("static_mean", "static_shape", "static_slope"))
        for name in feature_names
    ])
    noise_columns = np.asarray([
        name.endswith(("coverage", "mask_transition")) for name in feature_names
    ])
    dynamic_columns = ~(static_columns | noise_columns)

    report = {
        "protocol": {
            "source_sites": list(SOURCE_SITES),
            "source_rows": int(len(source_rows)),
            "target_site": TARGET_SITE,
            "target_use": "absence_and_predeclared_calibration_prompts_only",
            "target_query_labels_or_gt_used": False,
            "target_prompt_rows": int(len(target_prompt_rows)),
            "absence_rows": int(len(absence_rows)),
        },
        "feature_schema": feature_names,
        "variance_eta_squared": {
            "subject_all": dict(zip(
                feature_names,
                eta_squared(source_features, source_subject).astype(float).tolist()
            )),
            "subject_at_E01": dict(zip(
                feature_names,
                eta_squared(
                    source_features[balanced_subject],
                    source_subject[balanced_subject],
                ).astype(float).tolist()
            )),
            "environment_ajh_mhw": dict(zip(
                feature_names,
                eta_squared(
                    source_features[balanced_environment],
                    source_environment[balanced_environment],
                ).astype(float).tolist()
            )),
            "action": dict(zip(
                feature_names,
                eta_squared(source_features, source_action).astype(float).tolist()
            )),
        },
        "fingerprint_probes": {
            "subject_all": random_probe(source_features, source_subject, options.seed),
            "subject_static": random_probe(
                source_features[:, static_columns], source_subject, options.seed + 1
            ),
            "subject_dynamic": random_probe(
                source_features[:, dynamic_columns], source_subject, options.seed + 2
            ),
            "site_all": random_probe(source_features, source_sites, options.seed + 3),
            "site_static": random_probe(
                source_features[:, static_columns], source_sites, options.seed + 4
            ),
            "site_dynamic": random_probe(
                source_features[:, dynamic_columns], source_sites, options.seed + 5
            ),
        },
        "action_cross_domain_probes": {
            "all_leave_subject_out": grouped_probe(
                source_features, source_action, source_subject
            ),
            "dynamic_leave_subject_out": grouped_probe(
                source_features[:, dynamic_columns], source_action, source_subject
            ),
            "all_leave_site_out": grouped_probe(
                source_features, source_action, source_sites
            ),
            "dynamic_leave_site_out": grouped_probe(
                source_features[:, dynamic_columns], source_action, source_sites
            ),
        },
    }

    # Source absence/prompt 중심에서 target calibration support가 얼마나 이동했는지 측정한다.
    target_local = np.asarray([row_to_local[int(row)] for row in target_prompt_rows])
    absence_local = np.asarray([row_to_local[int(row)] for row in absence_rows])
    absence_meta = index.iloc[absence_rows]
    absence_sites = (absence_meta.subject + "_" + absence_meta.environment).to_numpy()
    source_absence = features[absence_local][absence_sites != TARGET_SITE]
    target_absence = features[absence_local][absence_sites == TARGET_SITE]
    source_prompt = source_features[np.isin(source_action, PROMPT_CLASSES)]
    source_center = source_prompt.mean(0)
    source_scale = source_prompt.std(0) + 1e-6
    report["target_calibration_shift"] = {
        "absence_standardized_distance": float(np.linalg.norm(
            (target_absence.mean(0) - source_absence.mean(0))
            / (source_absence.std(0) + 1e-6)
        ) / np.sqrt(features.shape[1])),
        "prompt_standardized_distance": float(np.linalg.norm(
            (features[target_local].mean(0) - source_center) / source_scale
        ) / np.sqrt(features.shape[1])),
        "static_prompt_distance": float(np.linalg.norm(
            (features[target_local].mean(0)[static_columns]
             - source_center[static_columns]) / source_scale[static_columns]
        ) / np.sqrt(static_columns.sum())),
        "dynamic_prompt_distance": float(np.linalg.norm(
            (features[target_local].mean(0)[dynamic_columns]
             - source_center[dynamic_columns]) / source_scale[dynamic_columns]
        ) / np.sqrt(dynamic_columns.sum())),
    }

    pd.DataFrame(features, columns=feature_names).assign(
        cache_row=analysis_rows,
        trial_id=index.trial_id.iloc[analysis_rows].to_numpy(),
        subject=index.subject.iloc[analysis_rows].to_numpy(),
        environment=index.environment.iloc[analysis_rows].to_numpy(),
        task=index.task.iloc[analysis_rows].to_numpy(),
        calibration_role=np.where(
            np.isin(analysis_rows, target_prompt_rows), "target_prompt",
            np.where(np.isin(analysis_rows, absence_rows), "absence", "source")
        ),
    ).to_csv(output / "trial_features.csv", index=False)
    site_summary(
        features, analysis_rows, index, feature_names
    ).to_csv(output / "site_summary.csv", index=False)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
