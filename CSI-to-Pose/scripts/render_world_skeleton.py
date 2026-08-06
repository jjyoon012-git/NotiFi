"""
render_world_skeleton.py

world 좌표(미터·골반원점) 예측을 눈으로 검증하는 2단 레이아웃 영상 생성.
  [ 위:  원본 영상 (가로 전체) ]
  [ 아래: GT 골격(정면뷰) | 예측 골격(정면뷰) ]
- GT의 root 위치를 빌리지 않는다. 각 골격을 자기 골반 중심으로 그린다(중심정렬+자동스케일).
- 정면뷰 = world (x=좌우, y=상하). y음수=위(머리)/y양수=아래(발) → 서면 세로/누우면 가로.

실행 (11라벨 × 4 test trial 전부):
    python scripts/render_world_skeleton.py \
      --exp_dir experiments/loso_test_mhw_rootrel_vismask_vel_rf16_world/ALL --subject mhw
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JOINTS = ["head","left_shoulder","right_shoulder","left_elbow","right_elbow",
          "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee",
          "left_ankle","right_ankle"]
EDGES = [("head","left_shoulder"),("head","right_shoulder"),("left_shoulder","right_shoulder"),
         ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
         ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
         ("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip"),
         ("left_hip","left_knee"),("left_knee","left_ankle"),
         ("right_hip","right_knee"),("right_knee","right_ankle")]
GT_COLOR, PRED_COLOR = (0,200,0), (0,60,230)   # BGR
ALL_LABELS = ["standing_still","sitting_still","lying_still","walking","sit_to_stand",
              "stand_to_sit","lie_to_stand","stand_to_lie_normal","bed_exit_failed",
              "unstable_walking","post_fall_inactive"]
TEST_TRIALS = ["t007","t013","t019","t025"]
LABEL_VIDEO_PATH = {
    "standing_still":"safe/posture/standing_still","sitting_still":"safe/posture/sitting_still",
    "lying_still":"safe/posture/lying_still","walking":"safe/motion/walking",
    "sit_to_stand":"safe/transition/sit_to_stand","stand_to_sit":"safe/transition/stand_to_sit",
    "lie_to_stand":"safe/transition/lie_to_stand","stand_to_lie_normal":"safe/transition/stand_to_lie_normal",
    "bed_exit_failed":"warning/bed_exit/bed_exit_failed","unstable_walking":"warning/gait/unstable_walking",
    "post_fall_inactive":"danger/fall/post_fall_inactive"}

PW, PH = 400, 560            # 골격 패널 (아래 2칸)
VW = PW * 2                  # 영상 폭 = 골격 2칸 폭
VH = int(VW * 9 / 16)        # 16:9 → letterbox 없이 꽉 참
CW, CH = VW, VH + PH         # 최종 캔버스


def compute_scale(df):
    xs = np.concatenate([df[f"{p}{j}_x"].values for p in ("gt_","pred_") for j in JOINTS])
    ys = np.concatenate([df[f"{p}{j}_y"].values for p in ("gt_","pred_") for j in JOINTS])
    cx, cy = (np.nanmin(xs)+np.nanmax(xs))/2, (np.nanmin(ys)+np.nanmax(ys))/2
    span = max(np.nanmax(xs)-np.nanmin(xs), np.nanmax(ys)-np.nanmin(ys), 0.5) * 1.3
    return cx, cy, span


def draw_panel(row, pfx, color, cx, cy, span, title):
    canvas = np.full((PH, PW, 3), 255, np.uint8)
    pts = {}
    for j in JOINTS:
        x, y = row.get(f"{pfx}{j}_x"), row.get(f"{pfx}{j}_y")
        if x is None or (isinstance(x, float) and np.isnan(x)):
            pts[j] = None
        else:
            px = int((x - cx)/span * PW + PW/2)
            py = int((y - cy)/span * PH + PH/2)   # y양수=아래 → 화면 y와 동일
            pts[j] = (px, py)
    for a,b in EDGES:
        if pts[a] and pts[b]:
            cv2.line(canvas, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for p in pts.values():
        if p:
            cv2.circle(canvas, p, 5, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return canvas


def compose(video_frame, row, cx, cy, span, label, trial, t_sec):
    canvas = np.full((CH, CW, 3), 255, np.uint8)
    # 상단: 영상 (꽉 채움)
    canvas[0:VH, 0:VW] = cv2.resize(video_frame, (VW, VH))
    cv2.putText(canvas, f"{label} {trial}  {t_sec:.1f}s", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    # 하단: GT | Pred
    canvas[VH:VH+PH, 0:PW]      = draw_panel(row, "gt_",   GT_COLOR,   cx, cy, span, "GT (world)")
    canvas[VH:VH+PH, PW:PW*2]   = draw_panel(row, "pred_", PRED_COLOR, cx, cy, span, "Pred (CSI)")
    cv2.line(canvas, (PW, VH), (PW, CH), (210,210,210), 1)
    cv2.line(canvas, (0, VH), (CW, VH), (180,180,180), 1)
    return canvas


def render_one(pred_csv, subject, label, trial, out_dir):
    df = pd.read_csv(pred_csv)
    cx, cy, span = compute_scale(df)
    pred_ts = df["timestamp_sec"].to_numpy()

    vdir = ROOT / f"csi_to_pose_{subject}" / "videos" / LABEL_VIDEO_PATH[label] / subject
    vpath = vdir / f"{subject}_{label}_{trial}.avi"
    ts_path = vdir / f"{subject}_{label}_{trial}_frame_timestamps.csv"
    if not (vpath.exists() and ts_path.exists()):
        print(f"  [SKIP] 영상 없음: {vpath.name}"); return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{label}_{trial}_world.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (CW, CH))

    cap = cv2.VideoCapture(str(vpath))
    fts = pd.read_csv(ts_path)["timestamp_sec"].to_numpy()
    idx_map = np.clip(np.searchsorted(pred_ts, fts), 0, len(df)-1)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        j = idx_map[fi] if fi < len(idx_map) else len(df)-1
        row = df.iloc[j]
        writer.write(compose(frame, row, cx, cy, span, label, trial, float(row["timestamp_sec"])))
        fi += 1
    cap.release(); writer.release()
    print(f"  [OK] {out_path.name}  ({fi} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--subject", default="mhw")
    ap.add_argument("--labels", nargs="+", default=ALL_LABELS)
    ap.add_argument("--trials", nargs="+", default=TEST_TRIALS)
    args = ap.parse_args()

    exp = ROOT / args.exp_dir if not Path(args.exp_dir).is_absolute() else Path(args.exp_dir)
    pred_dir = exp / "predictions"
    out_dir = exp / "world_view"
    n = 0
    for label in args.labels:
        for trial in args.trials:
            f = pred_dir / f"{args.subject}_{label}_{trial}_prediction.csv"
            if not f.exists():
                print(f"  [SKIP] 예측 없음: {f.name}"); continue
            render_one(f, args.subject, label, trial, out_dir); n += 1
    print(f"\n완료 {n}개 → {out_dir}")


if __name__ == "__main__":
    main()
