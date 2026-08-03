"""GT 품질 감사 — 사람이 공중에 떠 있는 trial 을 찾아 눈으로 확인하게 만든다.

문제: GVHMR 이 지면을 잘못 잡으면(예: 침대 상판을 바닥으로 인식) 바닥에 선 사람이
통째로 공중에 뜬다. 실측 사례 lmh E01 D01: 시작 시 발 높이 0.669m — 같은 사이트
침대 높이(0.41~0.64m)와 일치한다. 이런 trial 의 root 오차는 모델 탓이 아니다.

판정: 시나리오마다 '시작 시 사람이 어디 있어야 하는가'가 정해져 있다.
  바닥에서 시작   D01 D02 S01 S02 S06 S09 W01 W02  -> 시작 발 높이 ≈ 0
  침대/의자에서   D03 D04 D05 S03 S04 S05 W03      -> 떠 있는 게 정상, 검사 제외
낙상(D01~D05)은 끝에 바닥에 있어야 하므로 '끝 발 높이'도 함께 본다.

**영상 위에 골격을 직접 겹쳐 그릴 수는 없다.** 개발셋 GT(gvhmr_joints_v1)에는
카메라 내부·외부 파라미터가 없어서 3D 관절을 화면 좌표로 투영할 수 없다
(yja 의 gvhmr_smpl_full_v1 에만 K_fullimg 가 있다). 대신 영상과 GT 골격을
위아래로 붙이고 바닥선·발높이를 같이 표시해 눈으로 대조할 수 있게 한다.

실행:
    python -m notifi_pose.tools.gt_audit                 # 조사만
    python -m notifi_pose.tools.gt_audit --render        # 의심 trial 영상 생성
    python -m notifi_pose.tools.gt_audit --render --max 60
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio import cache as cache_mod

#: 사람이 바닥에서 시작하는 시나리오. 시작 시 발 높이가 0 이어야 한다.
FLOOR_START = ("D01", "D02", "S01", "S02", "S06", "S09", "W01", "W02")
#: 낙상 — 끝에는 바닥에 있어야 한다.
FLOOR_END = ("D01", "D02", "D03", "D04", "D05")

#: 판정 문턱. 정상 사이트의 시작 발 높이 중앙값이 0.01~0.11m 라 0.20m 면 확실히 이상.
START_TOL = 0.20
END_TOL = 0.25

FEET = [C.JOINT_INDEX["left_foot"], C.JOINT_INDEX["right_foot"]]
HEAD = C.JOINT_INDEX["head"]

# ---------------------------------------------------------------- 조사


def audit(cache) -> pd.DataFrame:
    ix, a = cache.index, cache.arrays
    rows = []
    for r in ix.index:
        md = ix.iloc[r]
        if not md.cache_ok or md.task != C.TASK_POSE:
            continue
        T = int(md.n_frames)
        v = np.asarray(a["valid"][r, :T]).astype(bool)
        if v.sum() < 60:
            continue
        A = (np.asarray(a["pose_rel"][r, :T], dtype=np.float32)
             + np.asarray(a["root"][r, :T], dtype=np.float32)[:, None, :])
        idx = np.flatnonzero(v)
        i0, i1 = idx[:30], idx[-30:]
        fs = float(A[i0][:, FEET, 1].mean())
        fe = float(A[i1][:, FEET, 1].mean())
        sc = md.scenario_id
        bad_start = sc in FLOOR_START and fs > START_TOL
        bad_end = sc in FLOOR_END and fe > END_TOL
        rows.append({
            "trial_id": md.trial_id, "subject": md.subject,
            "environment": md.environment, "scenario_id": sc,
            "label": md.detail_label, "role": md.role,
            "split_group": md.split_group,
            "foot_start": round(fs, 3), "foot_end": round(fe, 3),
            "foot_min": round(float(A[v][:, FEET, 1].min()), 3),
            "pelvis_start": round(float(A[i0][:, C.ROOT_JOINT, 1].mean()), 3),
            "bad_start": bad_start, "bad_end": bad_end,
            "bad": bool(bad_start or bad_end),
        })
    return pd.DataFrame(rows)


def report(d: pd.DataFrame) -> None:
    print(f"=== 검사 대상 {len(d)} pose trial ===")
    print(f"  바닥에서 시작하는 시나리오 {int(d.scenario_id.isin(FLOOR_START).sum())}")
    print(f"  이상 판정 {int(d.bad.sum())} "
          f"(시작 {int(d.bad_start.sum())}, 끝 {int(d.bad_end.sum())})\n")

    print("=== 시나리오 폴더 단위 집계 (이상 trial 이 있는 폴더만) ===")
    bad = d[d.bad]
    if bad.empty:
        print("  없음")
        return
    g = bad.groupby(["subject", "environment", "scenario_id", "label"]).agg(
        이상=("bad", "size"),
        발높이_시작중앙=("foot_start", "median"),
        발높이_끝중앙=("foot_end", "median")).reset_index()
    tot = d.groupby(["subject", "environment", "scenario_id"]).size().rename("전체")
    g = g.join(tot, on=["subject", "environment", "scenario_id"])
    g["비율"] = (g.이상 / g.전체 * 100).round(0).astype(int).astype(str) + "%"
    print(g.sort_values("이상", ascending=False).to_string(index=False))

    print("\n=== 폴더 경로 (확인·삭제용) ===")
    dev = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    sealed = pd.read_csv(C.SPLIT_DIR / "sealed_index.csv")
    pool = pd.concat([dev, sealed], ignore_index=True)
    path_of = dict(zip(pool.trial_id, pool.gt_pose))
    seen = set()
    for _, r in g.sort_values(["subject", "environment", "scenario_id"]).iterrows():
        sample = bad[(bad.subject == r.subject) & (bad.environment == r.environment)
                     & (bad.scenario_id == r.scenario_id)].trial_id.iloc[0]
        p = path_of.get(sample)
        if p is None:
            continue
        folder = str((C.DATASET_ROOT / p).parent.parent)
        if folder in seen:
            continue
        seen.add(folder)
        print(f"  [{r.이상}/{r.전체}] {folder}")


# ---------------------------------------------------------------- 렌더


VW, VH = 900, 506
PW, PH = 900, 420
FLOOR_Y = (-0.2, 2.2)


def _panel(A: np.ndarray, t: int, foot_h: float, ok: bool) -> np.ndarray:
    """GT 골격 정면뷰 + 바닥선. A [T,J,3]"""
    canvas = np.full((PH, PW, 3), 250, np.uint8)
    scale = PH / (FLOOR_Y[1] - FLOOR_Y[0])
    cx = float(np.median(A[:, :, 0]))

    def proj(P):
        return np.stack([PW / 2 + (P[:, 0] - cx) * scale,
                         PH - (P[:, 1] - FLOOR_Y[0]) * scale], -1)

    for m in np.arange(0.0, FLOOR_Y[1], 0.5):
        y = int(PH - (m - FLOOR_Y[0]) * scale)
        col = (90, 90, 90) if m == 0 else (222, 222, 222)
        cv2.line(canvas, (0, y), (PW, y), col, 3 if m == 0 else 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{m:.1f}m", (6, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "FLOOR (y=0)", (PW - 130, int(PH - (0 - FLOOR_Y[0]) * scale) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    xy = proj(A[t])
    col = (60, 170, 60) if ok else (50, 50, 235)
    for a_, b_ in C.SKELETON_EDGES:
        cv2.line(canvas, tuple(xy[a_].astype(int)), tuple(xy[b_].astype(int)),
                 col, 3, cv2.LINE_AA)
    for j in range(len(xy)):
        cv2.circle(canvas, tuple(xy[j].astype(int)), 4, col, -1, cv2.LINE_AA)

    # 발 높이 표시
    fy = int(PH - (foot_h - FLOOR_Y[0]) * scale)
    y0 = int(PH - (0 - FLOOR_Y[0]) * scale)
    if abs(fy - y0) > 3:
        cv2.arrowedLine(canvas, (60, y0), (60, fy), col, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(canvas, f"foot {foot_h*100:.0f}cm", (70, (fy + y0) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return canvas


def render_one(cache, row: pd.Series, video: str | None, out_path) -> bool:
    ix, a = cache.index, cache.arrays
    r = cache.row_of(row.trial_id)
    T = int(ix.n_frames.iloc[r])
    A = (np.asarray(a["pose_rel"][r, :T], dtype=np.float32)
         + np.asarray(a["root"][r, :T], dtype=np.float32)[:, None, :])
    foot = A[:, FEET, 1].mean(1)

    cap = cv2.VideoCapture(str(video)) if video else None
    nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        C.TARGET_FPS, (VW, VH + PH))
    for t in range(T):
        cv = np.full((VH + PH, VW, 3), 250, np.uint8)
        if cap is not None and t < nv:
            ok, f = cap.read()
            if ok:
                cv[0:VH, 0:VW] = cv2.resize(f, (VW, VH))
        good = foot[t] < START_TOL
        cv[VH:VH + PH] = _panel(A, t, float(foot[t]), good)
        band = cv[0:78, 0:VW]
        cv[0:78, 0:VW] = cv2.addWeighted(band, 0.4, np.zeros_like(band), 0.6, 0)
        cv2.putText(cv, f"{row.trial_id}  {row.label}  t={t/C.TARGET_FPS:5.2f}s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(cv, f"foot start {row.foot_start*100:.0f}cm  end {row.foot_end*100:.0f}cm"
                        f"   [{'START' if row.bad_start else ''}"
                        f"{' END' if row.bad_end else ''} 이상]",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (110, 110, 255), 2)
        w.write(cv)
    w.release()
    if cap is not None:
        cap.release()
    return True


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true", help="의심 trial 영상 생성")
    ap.add_argument("--max", type=int, default=40, help="렌더 최대 개수")
    ap.add_argument("--out", default=None, help="기본: work_v2/reports/gt_audit")
    args = ap.parse_args()

    cache = cache_mod.open_cache()
    d = audit(cache)
    report(d)

    out_dir = (C.WORK_ROOT / "reports" / "gt_audit") if args.out is None else args.out
    out_dir = out_dir if hasattr(out_dir, "mkdir") else __import__("pathlib").Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.sort_values(["bad", "foot_start"], ascending=[False, False]).to_csv(
        out_dir / "gt_audit.csv", index=False, encoding="utf-8")
    print(f"\n  wrote {out_dir / 'gt_audit.csv'}")

    if not args.render:
        print("  영상까지 만들려면 --render")
        return 0

    dev = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    sealed = pd.read_csv(C.SPLIT_DIR / "sealed_index.csv")
    pool = pd.concat([dev, sealed], ignore_index=True)
    vid = dict(zip(pool.trial_id, pool.original_video))

    bad = d[d.bad].sort_values("foot_start", ascending=False).head(args.max)
    print(f"\n[gt_audit] 렌더 {len(bad)}개 -> {out_dir}")
    for k, (_, row) in enumerate(bad.iterrows(), 1):
        rel = vid.get(row.trial_id)
        path = C.DATASET_ROOT / rel if isinstance(rel, str) else None
        name = (f"{row.foot_start*100:03.0f}cm_{row.subject}_{row.environment}_"
                f"{row.scenario_id}_{row.trial_id}.mp4")
        render_one(cache, row, path if path and path.exists() else None,
                   out_dir / name)
        print(f"  [{k}/{len(bad)}] {name}")
    print(f"\n  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
