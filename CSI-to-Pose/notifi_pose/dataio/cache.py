"""학습용 캐시 — memmap 기반.

매 epoch 마다 csi.csv 를 파싱하면 학습이 I/O 에 묶인다(trial 당 80ms x 2,749 x epoch).
한 번만 파싱해서 이진 배열로 구워두고, 학습은 그걸 읽는다.

**이 캐시는 학습 전용이다.** 배포에서는 시리얼 스트림이 곧바로
`csi_parser.packets_to_grid()` 로 들어가므로 캐시가 없다. 두 경로가 공유하는 것은
`packets_to_grid()` 이고, 캐시는 그 결과를 미리 저장해 둔 것일 뿐이다.

저장 형식: trial 마다 파일을 만들면 Windows 에서 epoch 당 수천 번 파일을 여느라 느리다.
배열 하나에 전부 담고 memmap 으로 연다. 필요한 부분만 OS 가 페이지 단위로 올려주고,
1.3GB 정도라 사실상 전부 OS 캐시에 상주한다.

    csi_iq     [N, 304, 3, 114, 2] float16  ~1.14 GB
    link_mask  [N, 304, 3]         bool
    pose_rel   [N, 304, 22, 3]     float32  ~0.22 GB
    root       [N, 304, 3]         float32
    valid      [N, 304]            bool

float16 을 쓰는 이유: 원본 I/Q 는 정수이고 실측 최대가 738 이다. float16 은 2048 까지
정수를 오차 없이 표현하고, 값 100 근처의 해상도가 0.06 으로 원본 양자화 단위(1)보다
16배 촘촘하다. 정밀도 손실이 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import contract as C

#: 배열 이름 -> (프레임 뒤 차원, dtype)
ARRAY_SPEC = {
    "csi_iq":    ((C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2), np.float16),
    "link_mask": ((C.N_LINKS,), np.bool_),
    "pose_rel":  ((C.N_JOINTS, 3), np.float32),
    "root":      ((3,), np.float32),
    "valid":     ((), np.bool_),
}

INDEX_NAME = "cache_index.csv"
META_NAME = "cache_meta.json"


def array_path(name: str, root: Path | None = None) -> Path:
    return (root or C.CACHE_DIR) / f"{name}.npy"


def index_path(root: Path | None = None) -> Path:
    return (root or C.CACHE_DIR) / INDEX_NAME


def meta_path(root: Path | None = None) -> Path:
    return (root or C.CACHE_DIR) / META_NAME


def estimated_bytes(n: int) -> int:
    total = 0
    for shape, dtype in ARRAY_SPEC.values():
        per = C.CACHE_FRAMES * int(np.prod(shape, dtype=np.int64)) if shape else C.CACHE_FRAMES
        total += n * per * np.dtype(dtype).itemsize
    return int(total)


def create(n: int, root: Path | None = None) -> dict[str, np.memmap]:
    """빈 캐시 배열들을 만든다(쓰기 모드)."""
    root = root or C.CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (shape, dtype) in ARRAY_SPEC.items():
        full = (n, C.CACHE_FRAMES) + shape
        out[name] = np.lib.format.open_memmap(
            array_path(name, root), mode="w+", dtype=dtype, shape=full)
    return out


def open_arrays(root: Path | None = None, mode: str = "r") -> dict[str, np.memmap]:
    """기존 캐시를 연다. mode='r' 읽기전용, 'r+' 수정 가능."""
    root = root or C.CACHE_DIR
    missing = [n for n in ARRAY_SPEC if not array_path(n, root).exists()]
    if missing:
        raise FileNotFoundError(
            f"캐시 배열 없음 {missing} (경로 {root})\n"
            f"  -> python -m notifi_pose.tools.build_cache")
    return {n: np.load(array_path(n, root), mmap_mode=mode) for n in ARRAY_SPEC}


def write_meta(n: int, root: Path | None = None, **extra) -> None:
    meta = {
        "preproc_version": C.PREPROC_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_trials": int(n),
        "frames": C.CACHE_FRAMES,
        "target_fps": C.TARGET_FPS,
        "links": list(C.LINKS),
        "csi_representation": C.CSI_REPRESENTATION,
        "csi_channels": (["amplitude", "sanitized_phase"]
                         if C.CSI_REPRESENTATION == "amp_phase" else ["I", "Q"]),
        "n_live_subcarriers": C.N_LIVE_SUBCARRIERS,
        "guard_subcarriers": list(C.GUARD_SUBCARRIERS),
        "max_gap_s": C.MAX_GAP_S,
        "joint_names": list(C.JOINT_NAMES),
        "root_joint": C.ROOT_JOINT,
        "up_axis": C.UP_AXIS,
        "arrays": {k: {"shape": [C.CACHE_FRAMES, *v[0]], "dtype": np.dtype(v[1]).name}
                   for k, v in ARRAY_SPEC.items()},
        **extra,
    }
    meta_path(root).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def read_meta(root: Path | None = None) -> dict:
    p = meta_path(root)
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음. build_cache 를 먼저 실행하라.")
    return json.loads(p.read_text(encoding="utf-8"))


@dataclass
class Cache:
    """캐시 핸들. 배열(memmap) + 행 인덱스."""
    arrays: dict[str, np.memmap]
    index: pd.DataFrame
    meta: dict

    def __len__(self) -> int:
        return len(self.index)

    def row_of(self, trial_id: str) -> int:
        hit = self.index.index[self.index.trial_id == trial_id]
        if len(hit) == 0:
            raise KeyError(trial_id)
        return int(hit[0])

    def get(self, row: int) -> dict:
        """한 trial 을 dict 로. n_frames 로 잘라 패딩을 제거한다."""
        n = int(self.index.n_frames.iloc[row])
        out = {k: np.asarray(v[row, :n]) for k, v in self.arrays.items()}
        out["trial_id"] = self.index.trial_id.iloc[row]
        out["n_frames"] = n
        return out


def open_cache(root: Path | None = None, mode: str = "r") -> Cache:
    """캐시를 연다. 전처리 버전이 어긋나면 즉시 실패시킨다.

    스키마가 바뀐 낡은 캐시로 학습하면 원인 찾기 어려운 버그가 되므로,
    조용히 넘어가지 않는다.
    """
    root = root or C.CACHE_DIR
    meta = read_meta(root)
    if meta.get("preproc_version") != C.PREPROC_VERSION:
        raise RuntimeError(
            f"캐시 전처리 버전 불일치: 캐시 {meta.get('preproc_version')} "
            f"!= 코드 {C.PREPROC_VERSION}\n"
            f"  -> python -m notifi_pose.tools.build_cache --rebuild")
    if meta.get("n_live_subcarriers") != C.N_LIVE_SUBCARRIERS:
        raise RuntimeError("캐시 subcarrier 수 불일치. --rebuild 하라.")
    idx = pd.read_csv(index_path(root))
    return Cache(arrays=open_arrays(root, mode), index=idx, meta=meta)
