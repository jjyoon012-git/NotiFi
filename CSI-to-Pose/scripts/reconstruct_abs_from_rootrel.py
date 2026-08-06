"""
reconstruct_abs_from_rootrel.py

root-relative 실험(예: cross_trainlmh_testmhw_rootrel)의 예측은 골반 기준 상대좌표라
render_video_overlay.py(=[0,1] 이미지 좌표 가정)로 바로 못 그린다.

이 스크립트는 상대좌표 예측을 "실제 위치에 놓인 절대좌표"로 되돌려
render_video_overlay가 읽을 수 있는 predictions 폴더를 새로 만든다.

  pred_abs(joint) = pred_relative(joint) + GT_root      ← GT 골반 위치를 빌려 배치
  gt_abs(joint)   = 원본 절대 GT (절대좌표 실험 폴더에서 가져옴)

즉 "모델이 맞힌 자세 모양"을 GT와 같은 위치에 놓아, 영상 속 사람 위에서 모양만 비교한다.
(모델은 root를 예측하지 않으므로 GT root로 배치하는 것이 표준 시각화 방식.)

실행:
    python reconstruct_abs_from_rootrel.py \
        --rootrel_dir experiments/cross_trainlmh_testmhw_rootrel/ALL \
        --abs_dir     experiments/cross_trainlmh_testmhw_AFTER1/ALL \
        --out_dir     experiments/cross_trainlmh_testmhw_rootrel_absviz/ALL
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

JOINTS = [
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
AXES = ("x", "y", "z")


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def gt_root_per_frame(abs_df):
    """절대좌표 GT에서 프레임별 골반 중점(root)을 구한다. 반환 (T,3) dict-of-axis."""
    root = {}
    for ax in AXES:
        lh = abs_df[f"gt_left_hip_{ax}"].to_numpy(dtype=np.float32)
        rh = abs_df[f"gt_right_hip_{ax}"].to_numpy(dtype=np.float32)
        root[ax] = (lh + rh) / 2.0
    return root


def reconstruct_one(rootrel_csv, abs_csv, out_csv, fixed_root=None):
    rr = pd.read_csv(rootrel_csv)
    ab = pd.read_csv(abs_csv)
    n = min(len(rr), len(ab))
    rr, ab = rr.iloc[:n].reset_index(drop=True), ab.iloc[:n].reset_index(drop=True)

    if fixed_root is not None:
        # 고정점 배치: "위치 정보가 없는 모델이 실전 추론에서 주는 그대로" 시각화.
        # 예측은 고정점에 못박히고, GT는 실제 위치 → 모양은 비교되고 위치 부재가 드러남.
        root = {ax: np.full(n, v, dtype=np.float32) for ax, v in zip(AXES, fixed_root)}
    elif "pred_root_x" in rr.columns:
        # root-head 모델: 모델이 예측한 root로 배치 (진짜 end-to-end 시각화)
        root = {ax: rr[f"pred_root_{ax}"].to_numpy(dtype=np.float32) for ax in AXES}
    else:
        root = gt_root_per_frame(ab)
    out = pd.DataFrame()
    out["trial"] = rr["trial"] if "trial" in rr else ab["trial"]
    out["timestamp_sec"] = rr["timestamp_sec"]

    for j in JOINTS:
        for ax in AXES:
            # 예측: 상대좌표 + GT root  → 실제 위치에 배치
            out[f"pred_{j}_{ax}"] = rr[f"pred_{j}_{ax}"].to_numpy(dtype=np.float32) + root[ax]
            # GT: 원본 절대좌표 그대로 (영상 속 사람과 정렬됨)
            out[f"gt_{j}_{ax}"] = ab[f"gt_{j}_{ax}"].to_numpy(dtype=np.float32)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rootrel_dir", required=True, help="root-relative 실험 폴더 (ALL 포함)")
    ap.add_argument("--abs_dir", required=True, help="절대좌표 실험 폴더 (GT root 출처)")
    ap.add_argument("--out_dir", required=True, help="복원된 predictions 저장 폴더")
    ap.add_argument("--fixed_root", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="예측을 고정점에 배치 (예: 0.5 0.6 0.3). 위치정보 없는 모델의 실전 추론 모습 시각화")
    args = ap.parse_args()

    rr_pred = resolve(args.rootrel_dir) / "predictions"
    ab_pred = resolve(args.abs_dir) / "predictions"
    out_pred = resolve(args.out_dir) / "predictions"

    files = sorted(rr_pred.glob("*_prediction.csv"))
    print(f"복원 대상 {len(files)}개")
    done = miss = 0
    for f in files:
        ab_f = ab_pred / f.name
        if not ab_f.exists():
            print(f"  [SKIP] 절대좌표 짝 없음: {f.name}")
            miss += 1
            continue
        reconstruct_one(f, ab_f, out_pred / f.name, fixed_root=args.fixed_root)
        done += 1
    print(f"완료: 복원 {done}개 | 스킵 {miss}개")
    print(f"→ {out_pred}")
    print("\n이제 렌더:")
    print(f"  python scripts/render_video_overlay.py --subject mhw --label ALL "
          f"--exp_dir {args.out_dir}")


if __name__ == "__main__":
    main()
