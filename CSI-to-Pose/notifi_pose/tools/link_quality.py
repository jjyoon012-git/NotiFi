"""링크 품질 전수 조사.

패킷율(coverage)만으로는 잡히지 않는 고장 유형이 있다: **패킷은 정상 속도로 들어오는데
CSI 진폭이 0에 가까운 링크**. 표본 조사에서 yja 의 E01/E03 에 광범위하게 나타났다.
이런 링크를 걸러내지 않으면 모델이 '내용 없는 입력'을 학습하게 된다.

실행:
    python -m notifi_pose.tools.link_quality              # 전체 3,299 trial
    python -m notifi_pose.tools.link_quality --limit 200  # 표본
    python -m notifi_pose.tools.link_quality --workers 8

산출:
    work_v2/reports/link_quality.csv    trial x link 단위 지표
    work_v2/reports/link_quality.json   집계 요약
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio import align, csi_parser, index as idx, targets

#: 링크 등급 임계 — `pkt_max_med`(패킷별 최대 |I/Q| 의 중앙값) 기준.
#:
#: 주의: 이전 버전은 `median(sqrt(I^2+Q^2))` 을 **전체 subcarrier** 에 대해 계산했다.
#: 128 subcarrier 중 14개는 guard band 라 항상 0이므로, 신호가 작으면 중앙값이 0으로
#: 깔려 '진폭 0 = 완전 사망' 으로 오판했다. 실제로는 값이 들어와 있고 크기만 작은
#: 경우가 많다. 그래서 지표를 패킷별 최대값 + 살아있는 subcarrier 수 + RSSI 로 바꿨다.
#:
#: 정상 링크는 pkt_max_med 12~120. 값이 1~3 이면 양자화 단계가 2~3개뿐이라
#: 사람 움직임에 의한 수 % 변조를 표현할 수 없다.
DEAD_MAX = 1.0     # 이하: 정보 없음 (1 LSB 또는 전부 0)
SEVERE_MAX = 4.0   # 이하: 심각 (2비트 미만)
WEAK_MAX = 12.0    # 이하: 약함


#: 인접 프레임 간 subcarrier 진폭 프로파일 상관.
#: 실제 CSI 라면 33ms 간격의 두 프레임이 서로 닮아야 한다(채널이 그 사이 통째로 바뀌지
#: 않으므로). 정상 링크는 0.90~0.95, 잡음만 있는 링크는 0.5 근처로 확연히 갈린다.
#: **진폭 크기보다 훨씬 나은 판별자다** — yja E01 TX3 는 진폭이 6(weak 수준)이라
#: 진폭 기준으로는 살아있어 보이지만, 프레임간 상관 0.52 로 실제로는 잡음이다.
MIN_FRAME_CORR = 0.75

#: subcarrier 간 진폭 프로파일의 표준편차. 실내 다중경로 채널이면 주파수 선택적
#: 페이딩 때문에 subcarrier 마다 진폭이 크게 다르다(정상 10~15). 평평하면 신호가 아니다.
MIN_PROFILE_STD = 2.0


def grade(pkt_max_med: float, live_sc: int, allzero_ratio: float,
          frame_corr: float = float("nan"), profile_std: float = float("nan")) -> str:
    if live_sc == 0 or allzero_ratio > 0.9 or pkt_max_med <= DEAD_MAX:
        return "dead"
    # 진폭은 있으나 채널 구조가 없는 경우 — 잡음을 신호로 오인하지 않도록 여기서 잡는다
    if np.isfinite(frame_corr) and frame_corr < MIN_FRAME_CORR:
        return "dead"
    if np.isfinite(profile_std) and profile_std < MIN_PROFILE_STD:
        return "dead"
    if pkt_max_med <= SEVERE_MAX:
        return "severe"
    if pkt_max_med <= WEAK_MAX:
        return "weak"
    return "ok"


def _measure(args) -> list[dict]:
    """원본 패킷 단위로 측정한다. 리샘플·마스크를 거치지 않으므로 보간 아티팩트가 없다."""
    trial_id, subject, environment, scenario_id, task, csi_rel, gt_rel, time_method = args
    csi_path = C.DATASET_ROOT / csi_rel
    base = {"trial_id": trial_id, "subject": subject, "environment": environment,
            "scenario_id": scenario_id}
    try:
        df, meta = csi_parser.read_csi_packets(csi_path)
    except Exception as exc:  # noqa: BLE001 — 실패도 리포트에 남긴다
        return [{**base, "tx": tx, "error": f"{type(exc).__name__}: {exc}"} for tx in C.LINKS]

    duration = float(df["pc_elapsed_s"].max() - df["pc_elapsed_s"].min()) or 1.0
    out = []
    for tx in C.LINKS:
        s = df[df["sender_id"] == tx]
        row = {**base, "tx": tx, "error": "", "n_packets": len(s),
               "hz": len(s) / duration, "bad_row_ratio": meta["bad_row_ratio"]}
        if len(s) == 0:
            out.append({**row, "pkt_max_med": 0.0, "pkt_max_p95": 0.0, "live_sc": 0,
                        "allzero_ratio": 1.0, "rssi_med": np.nan, "amp_cv": 0.0,
                        "grade": "dead", "max_gap_s": np.nan})
            continue

        v = np.array(",".join(s["csi_data"].str.strip().str.strip("[]").tolist()).split(","),
                     dtype=np.int32).reshape(len(s), C.CSI_RAW_LEN)
        pkt_max = np.abs(v).max(axis=1)
        amp = np.sqrt((v[:, 0::2].astype(np.float32) ** 2
                       + v[:, 1::2].astype(np.float32) ** 2))     # [n, 128]
        # guard band 를 빼기 위해, 절반 이상의 패킷에서 0이 아닌 subcarrier 만 '살아있다'고 본다
        live = (amp > 0).mean(axis=0) > 0.5
        n_live = int(live.sum())
        cv = frame_corr = profile_std = float("nan")
        if n_live >= 5:
            a = amp[:, live].astype(np.float64)
            cv = float(np.median(a.std(0) / (a.mean(0) + 1e-9)))
            profile_std = float(a.mean(0).std())
            # 인접 프레임 프로파일 상관의 중앙 (최대 200쌍 표본)
            k = min(len(a) - 1, 200)
            if k > 0:
                x, y = a[:k], a[1:k + 1]
                xc = x - x.mean(1, keepdims=True)
                yc = y - y.mean(1, keepdims=True)
                den = np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = (xc * yc).sum(1) / den
                frame_corr = float(np.nanmedian(corr))
        allzero = float((pkt_max == 0).mean())
        pm = float(np.median(pkt_max))
        t = np.sort(s["pc_elapsed_s"].to_numpy())
        out.append({**row,
                    "pkt_max_med": pm,
                    "pkt_max_p95": float(np.percentile(pkt_max, 95)),
                    "live_sc": n_live,
                    "allzero_ratio": allzero,
                    "rssi_med": float(s["rssi"].median()),
                    "amp_cv": cv,
                    "frame_corr": frame_corr,
                    "profile_std": profile_std,
                    "max_gap_s": float(np.diff(t).max()) if len(t) > 1 else np.nan,
                    "grade": grade(pm, n_live, allzero, frame_corr, profile_std)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="표본 trial 수 (기본: 전체)")
    ap.add_argument("--workers", type=int, default=min(mp.cpu_count(), 8))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = idx.load_all_index(resolve=False)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed)
    print(f"[link_quality] {len(df):,} trials, workers={args.workers}")

    jobs = [
        (r.trial_id, r.subject, r.environment, r.scenario_id, r.task,
         r.csi, r.gt_pose, r.time_method)
        for r in df.itertuples()
    ]

    rows = []
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_measure, jobs, chunksize=8), 1):
            rows.extend(res)
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    d = pd.DataFrame(rows)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = C.REPORT_DIR / "link_quality.csv"
    d.to_csv(out_csv, index=False, encoding="utf-8")

    errs = d[d.error != ""]
    ok = d[d.error == ""].copy()

    print("\n=== 신호 크기 pkt_max_med (subject x environment) ===")
    pivot = ok.pivot_table(index="subject", columns="environment",
                           values="pkt_max_med", aggfunc="median").round(1)
    print(pivot.to_string())

    print("\n=== RSSI 중앙값 (subject x environment) ===")
    print(ok.pivot_table(index="subject", columns="environment",
                         values="rssi_med", aggfunc="median").round(0).to_string())

    print("\n=== 링크 등급 분포 ===")
    gp = ok.pivot_table(index=["subject", "environment"], columns="grade",
                        values="trial_id", aggfunc="count").fillna(0).astype(int)
    for g in ["dead", "severe", "weak", "ok"]:
        if g not in gp:
            gp[g] = 0
    print(gp[["dead", "severe", "weak", "ok"]].to_string())

    ok["usable"] = ok.grade.isin(["weak", "ok"])
    per_trial = ok.groupby(["trial_id", "subject", "environment"]).agg(
        n_usable=("usable", "sum"),
        n_ok=("grade", lambda s: int((s == "ok").sum())),
    ).reset_index()
    print("\n=== trial 당 쓸 만한 링크 수 (weak 이상) ===")
    print(per_trial.pivot_table(index="subject", columns="n_usable",
                                values="trial_id", aggfunc="count")
                   .fillna(0).astype(int).to_string())

    n_zero = int((per_trial.n_usable == 0).sum())
    print(f"\n  쓸 만한 링크가 없는 trial: {n_zero:,} / {len(per_trial):,} "
          f"({n_zero / max(len(per_trial), 1):.1%})")
    print(f"  3링크 모두 weak 이상: {int((per_trial.n_usable == 3).sum()):,}")
    print(f"  3링크 모두 ok:        {int((per_trial.n_ok == 3).sum()):,}")
    if len(errs):
        print(f"  로딩 실패 trial: {errs.trial_id.nunique()}")
        for t in errs.trial_id.unique()[:5]:
            print(f"    - {t}: {errs[errs.trial_id == t].error.iloc[0]}")

    summary = {
        "thresholds": {"dead": DEAD_MAX, "severe": SEVERE_MAX, "weak": WEAK_MAX},
        "n_trials": int(df.shape[0]),
        "n_links_measured": int(len(ok)),
        "grade_counts": ok.grade.value_counts().to_dict(),
        "pkt_max_med_by_subject_env": pivot.reset_index().to_dict(orient="records"),
        "grade_by_subject_env": (
            gp[["dead", "severe", "weak", "ok"]]
            .reset_index().to_dict(orient="records")
        ),
        "trials_with_no_usable_link": n_zero,
        "trials_all3_usable": int((per_trial.n_usable == 3).sum()),
        "trials_all3_ok": int((per_trial.n_ok == 3).sum()),
        "n_failed_trials": int(errs.trial_id.nunique()) if len(errs) else 0,
    }
    out_json = C.REPORT_DIR / "link_quality.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\n[link_quality] wrote {out_csv}")
    print(f"[link_quality] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
