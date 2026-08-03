"""trial 인덱스 로딩.

데이터셋에는 단일 `training_index.csv` 가 없고 fold 별 인덱스 12개
(`folds/{fold}/{split}_index.csv`)만 있다. 4개 fold 는 모두 동일한 3,299 trial
universe 를 덮으므로, 아무 fold 의 train+val+test 를 union 하면 전체 인덱스가 된다.

이 모듈은 그 union 을 만들고 timestamp 품질 정보를 붙여 준다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import contract as C

#: 인덱스 CSV 가 가진 상대경로 컬럼. 절대경로로 확장해서 `*_path` 로 추가한다.
_PATH_COLS = ("csi", "original_video", "gt_pose")


def _resolve_paths(df: pd.DataFrame) -> pd.DataFrame:
    """`data/...` 상대경로를 DATASET_ROOT 기준 절대경로 컬럼으로 확장."""
    for col in _PATH_COLS:
        if col not in df.columns:
            continue
        # pathlib 이 posix 구분자를 알아서 정규화하므로 문자열 치환 불필요.
        df[f"{col}_path"] = df[col].map(
            lambda p: None if pd.isna(p) else C.DATASET_ROOT / str(p)
        )
    return df


def load_fold(fold: str, split: str, resolve: bool = True) -> pd.DataFrame:
    """단일 fold/split 인덱스를 읽는다.

    Args:
        fold: `test_ajh` 등. C.LOSO_FOLDS 참조.
        split: train | val | test
        resolve: True 면 `*_path` 절대경로 컬럼을 추가한다.
    """
    if fold not in C.LOSO_FOLDS:
        raise ValueError(f"unknown fold {fold!r}; expected one of {C.LOSO_FOLDS}")
    if split not in C.SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {C.SPLITS}")

    path = C.fold_index_path(fold, split)
    if not path.exists():
        raise FileNotFoundError(
            f"fold index not found: {path}\n"
            f"  → NOTIFI_DATASET_ROOT 가 올바른지 확인하세요 (현재: {C.DATASET_ROOT})"
        )
    df = pd.read_csv(path)
    df["fold"] = fold
    df["split"] = split
    return _resolve_paths(df) if resolve else df


def load_timestamp_quality() -> pd.DataFrame:
    """timestamp_quality.csv — trial 별 시간 정렬 방식과 근거.

    핵심 컬럼:
      time_method: `timestamps` (행 k = 프레임 k) 또는 `uniform_30fps` (행 누락 → k/30)
      usable:      정렬 가능 여부
    """
    path = C.TIMESTAMP_QUALITY_CSV
    if path.exists():
        return pd.read_csv(path)

    generated = C.INDEX_DIR / "timestamp_quality.csv"
    if generated.exists():
        return pd.read_csv(generated)
    if not C.TIMESTAMP_MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"timestamp quality and manifest not found: {path}, "
            f"{C.TIMESTAMP_MANIFEST_CSV}"
        )

    manifest = pd.read_csv(C.TIMESTAMP_MANIFEST_CSV)
    reference_frames = manifest["gt_frames"].fillna(manifest["video_frames"]).fillna(300)
    rows = manifest["timestamp_rows"].fillna(0).astype(int)
    exact = rows >= reference_frames.astype(int)
    quality = pd.DataFrame({
        "trial_id": manifest["trial_id"],
        "time_method": exact.map({True: C.TIME_METHOD_TIMESTAMPS,
                                  False: C.TIME_METHOD_UNIFORM}),
        "time_method_reason": exact.map({
            True: "complete recorded frame timestamps",
            False: "partial recorded timestamps scaled to full frame axis",
        }),
        "ts_rows": rows,
        "video_frames": manifest["video_frames"].fillna(reference_frames).astype(int),
        "usable": rows >= 2,
    })
    if quality.trial_id.duplicated().any():
        raise ValueError("timestamp manifest contains duplicate trial_id values")
    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    quality.to_csv(generated, index=False, encoding="utf-8")
    return quality


def load_labels() -> pd.DataFrame:
    """labels.csv — class_id ↔ scenario/risk/detail_label 매핑."""
    return pd.read_csv(C.LABELS_CSV)


def build_all_index(reference_fold: str = "test_ajh", resolve: bool = True) -> pd.DataFrame:
    """전체 trial 인덱스(3,299행)를 만든다.

    reference_fold 하나의 train+val+test 를 union 하고, 나머지 fold 의 split
    소속을 `fold_{name}` 컬럼으로 붙인다. timestamp 품질도 병합한다.

    반환 컬럼(주요):
        trial_id, subject, environment, task, class_id, risk_id, scenario_id,
        detail_label, gt_schema, csi_path, gt_pose_path, original_video_path,
        time_method, timestamp_usable, fold_test_ajh, fold_test_lmh, ...
    """
    base = pd.concat(
        [load_fold(reference_fold, s, resolve=False) for s in C.SPLITS],
        ignore_index=True,
    ).drop(columns=["fold", "split"])

    # 각 fold 에서의 split 소속
    for fold in C.LOSO_FOLDS:
        membership = pd.concat(
            [
                load_fold(fold, s, resolve=False)[["trial_id"]].assign(**{f"fold_{fold}": s})
                for s in C.SPLITS
            ],
            ignore_index=True,
        )
        base = base.merge(membership, on="trial_id", how="left", validate="one_to_one")

    tq = load_timestamp_quality()[
        ["trial_id", "time_method", "time_method_reason", "ts_rows", "video_frames", "usable"]
    ].rename(columns={"usable": "timestamp_usable"})
    base = base.merge(tq, on="trial_id", how="left", validate="one_to_one")

    base = base.sort_values("trial_id", ignore_index=True)
    return _resolve_paths(base) if resolve else base


def all_index_path() -> Path:
    return C.INDEX_DIR / "all_index.csv"


def load_all_index(resolve: bool = True) -> pd.DataFrame:
    """미리 만들어 둔 all_index.csv 를 읽는다. 없으면 즉석에서 만든다."""
    path = all_index_path()
    if path.exists():
        df = pd.read_csv(path)
        return _resolve_paths(df) if resolve else df
    return build_all_index(resolve=resolve)


def pose_only(df: pd.DataFrame) -> pd.DataFrame:
    """pose GT 가 있는 trial 만. (classification_only 144개 제외)"""
    return df[df["task"] == C.TASK_POSE].reset_index(drop=True)
