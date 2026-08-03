"""CSI - GT 시간 정렬 검증.

원리적으로는 오프셋이 0이어야 한다. CSI 의 `pc_elapsed_s` 와 video_timestamps 의
`pc_elapsed_s` 가 같은 `pc_monotonic_ns` 시계에서 파생되기 때문이다. 문제는 lmh 인데,
타임스탬프 로그에 행 누락이 있어 프레임 시각을 k/30 으로 만들어야 한다(align.py 참조).
그 가정이 맞는지 실측으로 확인하는 것이 이 도구의 목적이다.

방법: GT 관절 속도와 CSI 모션 에너지의 cross-correlation lag 을 잰다.

이전 시도가 실패한 이유(중요):
  링크별 진폭 스케일이 10배 이상 다른데 세 링크를 그냥 평균 내서 에너지를 만들었다.
  스케일 큰 링크가 지배해 버리고, 마스크 경계에서 계단이 생겨 신호가 뭉개졌다.
  → 링크별·subcarrier별로 그 trial 안에서 z 정규화한 뒤 합친다.
  → 죽은 링크는 아예 뺀다(linkqc).

해석 기준: 지표 자체의 잡음 바닥을 알아야 lag 숫자를 해석할 수 있으므로,
**다른 trial 의 GT 와 짝지은 대조군(shuffled)** 을 같이 잰다. 정렬이 잘 맞는다면
정상 짝의 |lag| 가 대조군보다 뚜렷하게 작아야 한다.

실행:
    python -m notifi_pose.tools.verify_alignment
    python -m notifi_pose.tools.verify_alignment --n 40 --risk danger
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import contract as C
from .. import linkqc
from ..dataio import align, csi_parser, index as idx, targets

SMOOTH_K = 5          # 이동평균 창 (프레임). 30fps 에서 약 167ms
MAX_LAG_FRAMES = 60   # 탐색 범위 ±2초


def _smooth(x: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)


def gt_speed(times: np.ndarray, joints_abs: np.ndarray) -> np.ndarray:
    """[T-1] 관절 평균 속도(m/s)."""
    dt = np.diff(times.astype(np.float64))
    dt[dt <= 0] = 1.0 / C.TARGET_FPS
    return _smooth(np.linalg.norm(np.diff(joints_abs, axis=0), axis=-1).mean(-1) / dt)


def csi_motion(csi, usable_links: np.ndarray) -> np.ndarray:
    """[T-1] CSI 모션 에너지.

    링크별·subcarrier별로 그 trial 안에서 z 정규화한 뒤 프레임 차분의 크기를 잰다.
    링크 간 스케일 차이(10배 이상)를 제거해야 약한 링크의 신호가 살아남는다.
    """
    amp = np.linalg.norm(csi.iq, axis=-1)              # [T, L, S]
    per_link = []
    for li in range(C.N_LINKS):
        if not usable_links[li]:
            continue
        v = csi.link_mask[:, li]
        if v.sum() < 30:
            continue
        a = amp[:, li].astype(np.float64)
        mu = a[v].mean(0, keepdims=True)
        sd = a[v].std(0, keepdims=True) + 1e-9
        z = (a - mu) / sd
        e = np.abs(np.diff(z, axis=0)).mean(-1)        # [T-1]
        pair = v[1:] & v[:-1]
        per_link.append(np.where(pair, e, np.nan))
    if not per_link:
        return np.zeros(0)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(np.stack(per_link, axis=1), axis=1)
    return _smooth(np.nan_to_num(e, nan=0.0))


def xcorr_curve(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """±MAX_LAG_FRAMES 구간의 상관곡선 [2L+1]. 가운데가 lag 0.

    곡선을 그대로 돌려주는 이유: trial 하나의 argmax 는 잡음이 지배해서 쓸 수 없고,
    수백 trial 의 곡선을 더해야 참 지연이 드러나기 때문이다(main 참조).
    """
    n = min(len(a), len(b))
    if n < 30:
        return None
    a, b = _z(a[:n]), _z(b[:n])
    full = np.correlate(a, b, mode="full") / n
    center = n - 1
    L = MAX_LAG_FRAMES
    if center - L < 0 or center + L + 1 > len(full):
        return None
    return full[center - L: center + L + 1]


def curve_lag_ms(curve: np.ndarray) -> float:
    """상관곡선의 봉우리 위치(ms)."""
    return (int(np.argmax(curve)) - MAX_LAG_FRAMES) / C.TARGET_FPS * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="subject 당 표본 trial 수")
    ap.add_argument("--risk", default="danger", choices=["danger", "warning", "safe", "all"],
                    help="어떤 동작으로 볼 것인가 (움직임이 큰 danger 가 기본)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    sealed = pd.read_csv(C.SPLIT_DIR / "sealed_index.csv")
    pool = pd.concat([dev, sealed], ignore_index=True)
    pool = pool[(pool.task == C.TASK_POSE) & (pool.n_alive == 3)]
    if args.risk != "all":
        pool = pool[pool.risk == args.risk]

    usable = linkqc.link_mask_per_trial()

    sample = pool.groupby("subject", group_keys=False).apply(
        lambda g: g.sample(min(args.n, len(g)), random_state=args.seed))
    print(f"[verify_alignment] {len(sample)} trials  (risk={args.risk})")

    loaded = []
    for r in sample.itertuples():
        csi_path = C.DATASET_ROOT / r.csi
        try:
            tgt = targets.load_pose_target(C.DATASET_ROOT / r.gt_pose)
            ft = align.frame_times(csi_path, tgt.n_frames, r.time_method)
            csi = csi_parser.load_csi_trial(csi_path, grid_times=ft)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {r.trial_id}: {type(exc).__name__}: {exc}")
            continue
        ul = usable.loc[r.trial_id].to_numpy() if r.trial_id in usable.index \
            else np.ones(C.N_LINKS, dtype=bool)
        g = gt_speed(ft, tgt.absolute())
        e = csi_motion(csi, ul)
        if len(e) == 0:
            continue
        loaded.append({"trial_id": r.trial_id, "subject": r.subject,
                       "environment": r.environment, "time_method": r.time_method,
                       "gt": g, "csi": e})

    rows = []
    curves: dict[str, list[np.ndarray]] = {}      # 그룹 -> 상관곡선 목록
    shuffled_curves: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    for i, x in enumerate(loaded):
        cur = xcorr_curve(x["gt"], x["csi"])
        if cur is None:
            continue
        # 대조군: 다른 trial 의 CSI 와 짝지어 지표의 잡음 바닥을 잰다
        j = int(rng.integers(0, len(loaded)))
        while j == i and len(loaded) > 1:
            j = int(rng.integers(0, len(loaded)))
        scur = xcorr_curve(x["gt"], loaded[j]["csi"])

        for key in ("ALL", f"time_method:{x['time_method']}", f"subject:{x['subject']}"):
            curves.setdefault(key, []).append(cur)
        if scur is not None:
            shuffled_curves.append(scur)

        rows.append({k: x[k] for k in ("trial_id", "subject", "environment", "time_method")}
                    | {"lag_ms": curve_lag_ms(cur), "corr0": float(cur[MAX_LAG_FRAMES]),
                       "shuffled_lag_ms": curve_lag_ms(scur) if scur is not None else np.nan,
                       "shuffled_corr0": float(scur[MAX_LAG_FRAMES]) if scur is not None else np.nan})

    d = pd.DataFrame(rows)
    if d.empty:
        print("[verify_alignment] 표본 없음")
        return 1

    # ---- 합산 상관곡선: 판정의 근거 ----
    agg = {k: np.mean(v, axis=0) for k, v in curves.items()}
    agg_shuffled = np.mean(shuffled_curves, axis=0) if shuffled_curves else None

    print("\n=== 합산 상관곡선 (판정 근거) ===")
    print("   trial 하나의 argmax 는 잡음이 지배한다. 곡선을 전부 더하면 잡음이 상쇄되고")
    print("   참 지연만 남는다. 봉우리가 0ms 에 있어야 정렬이 맞은 것이다.\n")
    probe = [-30, -15, -6, -3, 0, 3, 6, 15, 30]      # 프레임 오프셋
    hdr = "  ".join(f"{p / C.TARGET_FPS * 1000:+7.0f}" for p in probe)
    print(f"  {'그룹':<26} {'n':>4} {'봉우리':>8}   {hdr}")
    for key in sorted(agg, key=lambda k: (k != "ALL", k)):
        m = agg[key]
        vals = "  ".join(f"{m[MAX_LAG_FRAMES + p]:+7.4f}" for p in probe)
        print(f"  {key:<26} {len(curves[key]):>4} {curve_lag_ms(m):>+7.0f}ms   {vals}")
    if agg_shuffled is not None:
        vals = "  ".join(f"{agg_shuffled[MAX_LAG_FRAMES + p]:+7.4f}" for p in probe)
        print(f"  {'[대조] 엉뚱한 trial 끼리':<26} {len(shuffled_curves):>4} "
              f"{curve_lag_ms(agg_shuffled):>+7.0f}ms   {vals}")

    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "|lag| 중앙(ms)": round(float(g.lag_ms.abs().median()), 1),
            "lag 중앙(ms)": round(float(g.lag_ms.median()), 1),
            "|lag|<=100ms": round(float((g.lag_ms.abs() <= 100).mean()), 3),
            "|lag|<=200ms": round(float((g.lag_ms.abs() <= 200).mean()), 3),
            "lag0 상관 중앙": round(float(g.corr0.median()), 3),
            "[대조] |lag| 중앙": round(float(g.shuffled_lag_ms.abs().median()), 1),
            "[대조] |lag|<=100ms": round(float((g.shuffled_lag_ms.abs() <= 100).mean()), 3),
            "[대조] lag0 상관": round(float(g.shuffled_corr0.median()), 3),
        })

    print("\n=== trial 별 통계 (참고용 - 판정에는 쓰지 않는다) ===")
    print("   trial 하나의 argmax 는 잡음이 신호보다 커서 흩어진다. 정상이다.")
    print(d.groupby("time_method").apply(summarize, include_groups=False).T.to_string())

    # ---- 판정 ----
    # 두 조건을 모두 만족해야 한다.
    #   (1) 합산 곡선의 봉우리가 0ms 근처(±1프레임=33ms)에 있다  → 계통 오프셋 없음
    #   (2) lag0 상관이 대조군보다 뚜렷하게 높다                  → 우연이 아님
    # time_method 별로도 따로 본다. lmh 의 k/30 가정이 여기서 검증된다.
    tol_ms = 1.0 / C.TARGET_FPS * 1000.0 * 1.5        # ±1.5 프레임 = 50ms
    ctl0 = float(agg_shuffled[MAX_LAG_FRAMES]) if agg_shuffled is not None else 0.0

    verdicts = {}
    for key in [k for k in agg if k == "ALL" or k.startswith("time_method:")]:
        m = agg[key]
        lag = curve_lag_ms(m)
        peak_ok = abs(lag) <= tol_ms
        signal_ok = m[MAX_LAG_FRAMES] > max(ctl0 * 3.0, 0.01)
        verdicts[key] = {"peak_lag_ms": lag, "corr0": float(m[MAX_LAG_FRAMES]),
                         "peak_ok": bool(peak_ok), "signal_ok": bool(signal_ok),
                         "passed": bool(peak_ok and signal_ok)}
    passed = all(v["passed"] for v in verdicts.values())

    print(f"\n판정 (허용 오차 ±{tol_ms:.0f}ms, 대조군 lag0 상관 {ctl0:.4f})")
    for key, v in verdicts.items():
        mark = "OK " if v["passed"] else "NG "
        print(f"  {mark} {key:<26} 봉우리 {v['peak_lag_ms']:+.0f}ms  "
              f"lag0 상관 {v['corr0']:.4f}  "
              f"(대조군의 {v['corr0'] / max(ctl0, 1e-9):.1f}배)")
    print("  ->", "정렬 확인됨" if passed else "정렬 미확인")

    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.REPORT_DIR / "alignment_verify.json"
    out.write_text(json.dumps({
        "risk": args.risk, "n": len(d),
        "tolerance_ms": tol_ms,
        "verdicts": verdicts,
        "aggregate_curves": {k: [float(x) for x in v] for k, v in agg.items()},
        "shuffled_curve": [float(x) for x in agg_shuffled] if agg_shuffled is not None else None,
        "per_trial_overall": summarize(d).to_dict(),
        "per_trial_by_time_method": {k: summarize(g).to_dict()
                                     for k, g in d.groupby("time_method")},
        "passed": bool(passed),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    d.to_csv(C.REPORT_DIR / "alignment_verify.csv", index=False, encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
