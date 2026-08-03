"""사이트별 빈방 기준선 계산 (Phase 2).

각 사이트(subject x environment)의 absence trial — 사람이 없는 10초 — 로부터
링크·subcarrier·채널별 기준선 통계를 만든다. 학습 시에는 이걸 빼서 "빈방 대비 편차"를
입력으로 쓰고, 배포 시에는 새 집에서 빈방 10초를 녹음해 같은 연산을 한다.

왜 필요한가 (실측):
  - 같은 TX1 이라도 사이트마다 평균 진폭이 4.5~71.1 로 15.9배 다르다.
  - 진폭 프로파일의 '모양'도 사이트마다 다르다(TX2 사이트 간 상호상관 0.271).
    이 모양이 곧 방의 기하 = 사이트 지문이다.
  - 정지한 사람은 도플러를 만들지 않으므로, 누움/서있음은 "빈방 대비 정적 다중경로가
    얼마나 바뀌었나"로만 구분된다. 기준선이 없으면 원리적으로 풀 수 없다.
    (실측: 같은 사이트 TX2 에서 |서있음-빈방|=40 vs |누움-빈방|=174)

Phase 1(진폭+정제위상) 단독으로는 in-domain 만 좋아지고 LOSO 는 오히려 나빠졌다
(class 0.071 -> 0.035). 신호가 선명해진 만큼 지문 외우기도 쉬워졌기 때문이다.
기준선 제거가 그 지문을 걷어낸다.

저장 통계 두 종류:
  mu0, sd0    absence 만으로 계산. **라벨도 사람도 필요 없어 배포에서 그대로 가능**
  rm, rs      (x - mu0) 잔차의 사이트별 평균/표준편차. 사람이 있는 데이터가 필요하지만
              **라벨은 필요 없다**(설치 후 몇 분간의 무라벨 관측으로 얻을 수 있다).
              분석상 판별력이 더 높다(0.939 vs 0.788).

누수 주의: 잔차 통계는 **train 역할 trial 로만** 계산한다. LOSO 의 held-out subject 는
자기 absence(무라벨)만 쓰고 잔차 통계는 absence 로 대체한다 — 배포에서 가능한 것과
정확히 같은 조건을 유지하기 위해서다.

실행:
    python -m notifi_pose.tools.build_site_baseline
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio import cache as cache_mod

BASELINE_NAME = "site_baseline.npz"

#: 기준선을 신뢰할 수 있는 링크의 조건 — **실제 신호가 있었는가**.
#:
#: 주의: `frame_corr`(인접 프레임 상관)로 판정하면 안 된다. 그건 "프레임 단위로 믿을 수
#: 있나"의 지표이지 "평균이 유효한가"의 지표가 아니다. 패킷율이 낮은 링크는 간격이 길어
#: 상관이 자연히 떨어진다(ajh E02 TX2 는 8Hz 라 corr 0.58 이지만 진폭 9.49 로 멀쩡하다).
#: 기준선은 absence 12 trial x 수십~수백 패킷의 **평균**이라 프레임 잡음은 상쇄된다.
#:
#: 이 기준으로 걸리는 것은 개발셋에서 lmh E03 TX1/TX2 뿐이고, 둘 다 linkqc 가 이미
#: 죽었다고 판정해 학습에서 마스크되는 링크다 — 즉 잃는 데이터가 없다.
MIN_BASELINE_PKT_MAX = 4.0
MIN_BASELINE_LIVE_SC = 50


def baseline_path():
    return C.CACHE_DIR / BASELINE_NAME


def _link_validity(site_prefix: str) -> np.ndarray:
    """absence trial 의 링크 품질로 (3,) 기준선 신뢰 플래그를 만든다."""
    path = C.REPORT_DIR / "link_quality.csv"
    if not path.exists():
        return np.ones(C.N_LINKS, dtype=bool)
    q = pd.read_csv(path)
    q = q[(q.error.isna() | (q.error == "")) & (q.scenario_id == "S07")]
    q = q[q.trial_id.str.startswith(site_prefix + "_")]
    out = np.zeros(C.N_LINKS, dtype=bool)
    for li, tx in enumerate(C.LINKS):
        s = q[q.tx == tx]
        if len(s) == 0:
            continue
        out[li] = (float(s.pkt_max_med.median()) >= MIN_BASELINE_PKT_MAX
                   and float(s.live_sc.median()) >= MIN_BASELINE_LIVE_SC)
    return out


def _accumulate(cache, rows) -> tuple[np.ndarray, np.ndarray, int]:
    """행들의 (합, 제곱합, 개수)를 링크·subcarrier·채널별로 모은다."""
    shape = (C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2)
    s = np.zeros(shape, dtype=np.float64)
    s2 = np.zeros(shape, dtype=np.float64)
    cnt = np.zeros((C.N_LINKS, 1, 1), dtype=np.float64)
    for r in rows:
        n = int(cache.index.n_frames.iloc[r])
        if n <= 0:
            continue
        x = np.asarray(cache.arrays["csi_iq"][r, :n], dtype=np.float64)   # [n,L,S,2]
        m = np.asarray(cache.arrays["link_mask"][r, :n])                  # [n,L]
        w = m[:, :, None, None]
        s += (x * w).sum(0)
        s2 += ((x ** 2) * w).sum(0)
        cnt += m.sum(0)[:, None, None]
    return s, s2, cnt


def _mean_std(s, s2, cnt, eps=1e-3):
    c = np.maximum(cnt, 1.0)
    mu = s / c
    var = np.maximum(s2 / c - mu ** 2, 0.0)
    return mu.astype(np.float32), np.maximum(np.sqrt(var), eps).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-frames", type=int, default=300,
                    help="사이트당 최소 유효 프레임 수 (미달이면 경고)")
    args = ap.parse_args()

    cache = cache_mod.open_cache()
    ix = cache.index
    sites = sorted({(s, e) for s, e in zip(ix.subject, ix.environment)})
    print(f"[baseline] 사이트 {len(sites)}개, 캐시 {len(ix)} trial")

    mu0, sd0, rm, rs, valid = {}, {}, {}, {}, {}
    report = {}
    for (s, e) in sites:
        base = (ix.subject == s) & (ix.environment == e) & ix.cache_ok
        ab_rows = np.flatnonzero((base & (ix.task == C.TASK_CLS)).to_numpy())
        if len(ab_rows) == 0:
            print(f"  [건너뜀] {s} {e}: absence trial 없음")
            continue

        key = f"{s}_{e}"
        S, S2, CNT = _accumulate(cache, ab_rows)
        m0, s0 = _mean_std(S, S2, CNT)
        mu0[key] = m0
        sd0[key] = s0
        valid[key] = _link_validity(f"{s}_{e}")

        # 잔차 통계: 사람이 있는 train 역할 trial 로만. 없으면 absence 로 대체(배포 조건).
        tr_rows = np.flatnonzero(
            (base & (ix.task == C.TASK_POSE) & (ix.role == "train")).to_numpy())
        src = tr_rows if len(tr_rows) else ab_rows
        rS = np.zeros_like(S)
        rS2 = np.zeros_like(S2)
        rC = np.zeros_like(CNT)
        for r in src:
            n = int(ix.n_frames.iloc[r])
            if n <= 0:
                continue
            x = np.asarray(cache.arrays["csi_iq"][r, :n], dtype=np.float64) - m0
            m = np.asarray(cache.arrays["link_mask"][r, :n])
            w = m[:, :, None, None]
            rS += (x * w).sum(0)
            rS2 += ((x ** 2) * w).sum(0)
            rC += m.sum(0)[:, None, None]
        a, b = _mean_std(rS, rS2, rC)
        rm[key] = a
        rs[key] = b

        report[key] = {
            "baseline_valid_links": [bool(x) for x in valid[key]],
            "n_absence": int(len(ab_rows)),
            "n_residual_src": int(len(src)),
            "residual_from": "train_pose" if len(tr_rows) else "absence(배포 조건)",
            "amp_mu0_median": float(np.median(m0[..., 0])),
            "amp_sd0_median": float(np.median(s0[..., 0])),
            "phase_sd0_median": float(np.median(s0[..., 1])),
            "resid_sd_median": float(np.median(b[..., 0])),
            "frames": int(CNT.max()),
        }

    if not mu0:
        print("[baseline] 만들 수 있는 사이트가 없다")
        return 1

    np.savez(baseline_path(),
             preproc_version=C.PREPROC_VERSION,
             representation=C.CSI_REPRESENTATION,
             sites=np.array(sorted(mu0.keys())),
             **{f"mu0__{k}": v for k, v in mu0.items()},
             **{f"sd0__{k}": v for k, v in sd0.items()},
             **{f"rm__{k}": v for k, v in rm.items()},
             **{f"rs__{k}": v for k, v in rs.items()},
             **{f"valid__{k}": v for k, v in valid.items()})

    print("\n=== 사이트별 빈방 기준선 ===")
    print(f"{'site':<10} {'absence':>8} {'잔차원본':>9} {'진폭 mu0':>10} {'진폭 sd0':>10} "
          f"{'위상 sd0':>10} {'잔차 sd':>9}")
    for k, v in report.items():
        print(f"{k:<10} {v['n_absence']:>8} {v['n_residual_src']:>9} "
              f"{v['amp_mu0_median']:>10.2f} {v['amp_sd0_median']:>10.2f} "
              f"{v['phase_sd0_median']:>10.3f} {v['resid_sd_median']:>9.2f}")

    amps = [v["amp_mu0_median"] for v in report.values()]
    print(f"\n  사이트 간 빈방 진폭 범위 {min(amps):.2f} ~ {max(amps):.2f} "
          f"({max(amps)/max(min(amps),1e-9):.1f}배)  <- 이만큼의 지문을 제거한다")
    print(f"  잔차 통계 출처: "
          f"{sum(1 for v in report.values() if v['residual_from']=='train_pose')} 사이트는 train, "
          f"{sum(1 for v in report.values() if v['residual_from']!='train_pose')} 사이트는 absence(배포 조건)")

    (C.REPORT_DIR / "site_baseline.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {baseline_path()}")
    print(f"  wrote {C.REPORT_DIR / 'site_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
