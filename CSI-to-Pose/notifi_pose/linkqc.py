"""링크 사용 가능 판정 — 단일 진실 공급원.

link_quality.py 가 저장한 원지표(`work_v2/reports/link_quality.csv`)에서 판정만
다시 계산한다. 임계를 바꿔도 3분짜리 전수 스캔을 다시 돌릴 필요가 없다.

판정 기준(전수 실측 근거):

  1) live_sc == 0            살아있는 subcarrier 가 없음
  2) allzero_ratio > 0.9     패킷 대부분이 완전히 0
  3) pkt_max_med <= 1        값이 0/±1 뿐 (1 LSB)
  4) frame_corr < 0.65       인접 프레임 프로파일이 안 닮음 = 채널이 아니라 잡음

4번이 가장 강력한 판별자다. 실제 CSI 라면 33ms 간격의 두 프레임이 서로 닮아야 한다
(그 사이 채널이 통째로 바뀌지 않으므로). 진폭만 보면 놓친다 — yja E01 TX3 는
진폭 6(정상의 1/5 수준이지만 0은 아님)이라 진폭 기준으로는 살아 보이는데,
프레임간 상관이 0.52 라 실제로는 잡음이다.

임계 0.65 의 근거:
  확실히 죽은 링크(yja E01/E03)   : 중앙 0.135, 99%tile 0.619
  확실히 정상인 링크(mhw E01/E02) : 1%tile 0.721, 중앙 0.913
  → 0.62~0.72 에 빈 구간. 그 중앙을 쓴다.
혼동 요인도 확인했다. 패킷율(<10Hz 0.892 vs 40Hz+ 0.837)과 동작 격렬도
(danger 0.892 / safe 0.882 / warning 0.869)에 거의 영향받지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import contract as C

MIN_FRAME_CORR = 0.65
MIN_PKT_MAX = 1.0
MAX_ALLZERO_RATIO = 0.9


def load_link_metrics() -> pd.DataFrame:
    path = C.REPORT_DIR / "link_quality.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. 먼저 실행하라:\n"
            f"  python -m notifi_pose.tools.link_quality"
        )
    d = pd.read_csv(path)
    return d[d.error.isna() | (d.error == "")].copy()


def is_dead(d: pd.DataFrame) -> pd.Series:
    """링크 단위 사망 판정. d 는 link_quality.csv 의 행들."""
    dead = (d.live_sc <= 0) | (d.pkt_max_med <= MIN_PKT_MAX)
    if "allzero_ratio" in d:
        dead |= d.allzero_ratio.fillna(0) > MAX_ALLZERO_RATIO
    if "frame_corr" in d:
        # 측정 불가(NaN)는 살아있다고 보지 않는다 — live_sc 가 이미 걸렀을 것
        dead |= d.frame_corr.fillna(0.0) < MIN_FRAME_CORR
    return dead


def alive_links_per_trial() -> pd.Series:
    """trial_id → 살아있는 링크 수(0~3)."""
    d = load_link_metrics()
    d["dead"] = is_dead(d)
    return (d.assign(alive=~d.dead)
             .groupby("trial_id").alive.sum().astype(int).rename("n_alive"))


def link_mask_per_trial() -> pd.DataFrame:
    """trial_id × TX → 사용 가능 여부. 학습 시 link_mask 와 AND 로 결합한다."""
    d = load_link_metrics()
    d["usable"] = ~is_dead(d)
    return d.pivot_table(index="trial_id", columns="tx", values="usable",
                         aggfunc="first").reindex(columns=list(C.LINKS)).fillna(False)
