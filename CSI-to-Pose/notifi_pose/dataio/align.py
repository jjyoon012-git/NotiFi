"""CSI ↔ 영상/GT 시간 정렬.

CSI 의 `pc_elapsed_s` 와 `video_timestamps.csv` 의 `pc_elapsed_s` 는 둘 다 같은
`pc_monotonic_ns` 시계에서 파생되므로 **오프셋 추정이 원리적으로 불필요**하다.
남는 문제는 "GT 프레임 k 가 몇 초인가" 하나뿐이고, 그건 subject 마다 다르다.

    ajh / mhw / yja : video_timestamps 의 행 k = 영상 프레임 k  → 그 값을 그대로 쓴다.
    lmh             : 타임스탬프 로그에 행 누락이 있다. 기록된 실제 시각의 시작/끝과
                      불규칙 간격을 보존한 채 전체 GT 프레임 축에 보간한다.

어느 쪽인지는 데이터셋의 `timestamp_quality.csv` 가 `time_method` 컬럼으로 알려준다
(`timestamps` / `uniform_30fps`). 기존 `uniform_30fps` 이름은 호환 때문에 유지하지만
더 이상 k/30을 뜻하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import contract as C


class AlignmentError(RuntimeError):
    pass


def video_timestamps_path(csi_path: Path) -> Path:
    csi_path = Path(csi_path)
    sibling = csi_path.parent / "video_timestamps.csv"
    if sibling.exists():
        return sibling

    try:
        relative = csi_path.relative_to(C.DATASET_ROOT)
    except ValueError:
        return sibling
    # Dataset: data/<task>/<subject>/<environment>/.../<trial>/csi.csv
    # Staging:          <subject>/<environment>/.../<trial>/video_timestamps.csv
    parts = relative.parts
    if len(parts) >= 5 and parts[0] == "data":
        external = C.TIMESTAMP_ROOT.joinpath(*parts[2:-1], "video_timestamps.csv")
        if external.exists():
            return external
    return sibling


def _make_strictly_increasing(t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """중복 타임스탬프를 없앤다.

    `video_timestamps.csv` 는 밀리초 해상도로 반올림되어 있어 연속 프레임이 같은
    값을 갖는 경우가 있다(표본 200 trial 중 64개, trial 당 평균 1.4개, 최대 15개).
    **시간 역행은 한 건도 관측되지 않았다** — 순수한 반올림 충돌이다.

    중복 구간은 양옆의 유효 시각 사이를 선형 보간해서 메우고, 마지막에 엄격 단조를
    보장한다(u_i = t_i - i·eps 의 누적 최대 트릭).
    """
    t = np.asarray(t, dtype=np.float64).copy()
    n = len(t)
    if n < 2:
        return t

    good = np.concatenate([[True], np.diff(t) > 0])
    if not good.all() and good.sum() >= 2:
        pos = np.flatnonzero(good)
        t = np.interp(np.arange(n, dtype=np.float64), pos.astype(np.float64), t[pos])

    ramp = np.arange(n, dtype=np.float64) * eps
    return np.maximum.accumulate(t - ramp) + ramp


def frame_times(
    csi_path: Path,
    n_frames: int,
    time_method: str,
) -> np.ndarray:
    """GT 프레임 0..n_frames-1 의 trial 상대 시각(초)을 반환한다. shape [n_frames].

    Args:
        csi_path: trial 의 csi.csv 경로 (형제 파일 video_timestamps.csv 를 찾는다)
        n_frames: GT 프레임 수 (gt_pose.npz 의 frame_index 길이)
        time_method: contract.TIME_METHOD_* 중 하나. all_index.csv 의 값을 넘긴다.
    """
    ts_path = video_timestamps_path(csi_path)
    if not ts_path.exists():
        raise AlignmentError(f"video_timestamps.csv not found: {ts_path}")
    ts = pd.read_csv(ts_path, usecols=["pc_elapsed_s"])
    elapsed = ts["pc_elapsed_s"].to_numpy(dtype=np.float64)
    if len(elapsed) == 0:
        raise AlignmentError(f"empty video_timestamps.csv: {ts_path}")

    if time_method == C.TIME_METHOD_TIMESTAMPS:
        if len(elapsed) < n_frames:
            raise AlignmentError(
                f"{ts_path.parent.name}: time_method=timestamps 인데 행 수가 부족하다 "
                f"({len(elapsed)} < {n_frames}). all_index 의 time_method 를 확인하라."
            )
        return _make_strictly_increasing(elapsed[:n_frames]).astype(np.float32)

    if time_method == C.TIME_METHOD_UNIFORM:
        if len(elapsed) < 2:
            raise AlignmentError(
                f"{ts_path.parent.name}: partial timestamp rows < 2 ({len(elapsed)})"
            )
        # 관측 행을 전체 프레임 축에 배치해 촬영 시작/끝, 평균 FPS, 프레임 드롭으로
        # 생긴 불규칙성을 보존한다. t0 + k/30보다 관측 사실을 더 많이 사용한다.
        source_axis = np.linspace(0.0, n_frames - 1.0, len(elapsed), dtype=np.float64)
        target_axis = np.arange(n_frames, dtype=np.float64)
        observed = _make_strictly_increasing(elapsed)
        return np.interp(target_axis, source_axis, observed).astype(np.float32)

    raise AlignmentError(f"unknown time_method {time_method!r}")


def motion_energy_peak(times: np.ndarray, iq: np.ndarray, link_mask: np.ndarray) -> float:
    """CSI 진폭 변화량이 최대인 시각. 정렬 검증용(학습에는 쓰지 않는다).

    Args:
        iq: [T, L, S, 2]
        link_mask: [T, L]
    """
    amp = np.linalg.norm(iq, axis=-1)                       # [T, L, S]
    diff = np.abs(np.diff(amp, axis=0)).mean(axis=-1)       # [T-1, L]
    valid = (link_mask[1:] & link_mask[:-1]).astype(np.float32)
    energy = (diff * valid).sum(axis=-1) / np.maximum(valid.sum(axis=-1), 1e-6)
    if len(energy) == 0 or not np.isfinite(energy).any():
        return float("nan")
    return float(times[1:][int(np.nanargmax(energy))])


def joint_velocity_peak(times: np.ndarray, joints: np.ndarray) -> float:
    """GT 관절 속도가 최대인 시각. 정렬 검증용.

    Args:
        joints: [T, J, 3]
    """
    dt = np.diff(times)
    dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 1.0 / C.TARGET_FPS
    vel = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=-1) / dt
    if len(vel) == 0 or not np.isfinite(vel).any():
        return float("nan")
    return float(times[1:][int(np.nanargmax(vel))])
