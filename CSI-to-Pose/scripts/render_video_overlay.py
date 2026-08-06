"""
render_video_overlay.py

trial 하나당 두 가지 영상을 생성한다.

  1. skeleton_only  : 흰 배경, 왼쪽=GT / 오른쪽=Pred  (항상 생성)
  2. video_overlay  : 원본 .avi 위에 GT+Pred 오버레이 (영상 있을 때만)

실행:
    python render_video_overlay.py --subject mhw --label post_fall_inactive --trials t007 t013 t019 t025
    python render_video_overlay.py --subject mhw --label post_fall_inactive  # 전체 prediction 자동 탐색
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

JOINTS = [
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

EDGES = [
    ("head", "left_shoulder"), ("head", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

GT_COLOR   = (0, 200, 0)    # BGR 초록
PRED_COLOR = (0, 60, 230)   # BGR 빨강


ALL_LABELS = [
    "standing_still", "sitting_still", "lying_still", "walking",
    "sit_to_stand", "stand_to_sit", "lie_to_stand", "stand_to_lie_normal",
    "bed_exit_failed", "unstable_walking", "post_fall_inactive",
]

LABEL_VIDEO_PATH = {
    "standing_still":      "safe/posture/standing_still",
    "sitting_still":       "safe/posture/sitting_still",
    "lying_still":         "safe/posture/lying_still",
    "walking":             "safe/motion/walking",
    "sit_to_stand":        "safe/transition/sit_to_stand",
    "stand_to_sit":        "safe/transition/stand_to_sit",
    "lie_to_stand":        "safe/transition/lie_to_stand",
    "stand_to_lie_normal": "safe/transition/stand_to_lie_normal",
    "bed_exit_failed":     "warning/bed_exit/bed_exit_failed",
    "unstable_walking":    "warning/gait/unstable_walking",
    "post_fall_inactive":  "danger/fall/post_fall_inactive",
}


def subject_prefix(subject):
    return subject.upper() if subject.lower() == "ajh" else subject


def draw_skeleton_on(canvas, joints_xy, color, offset_x=0, thickness=2, dot_r=5):
    """canvas 위에 skeleton을 그린다. offset_x로 패널 오프셋 지정."""
    h, w_half = canvas.shape[0], canvas.shape[1] // 2
    pts = {}
    for joint in JOINTS:
        x = joints_xy.get(f"{joint}_x")
        y = joints_xy.get(f"{joint}_y")
        if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
            pts[joint] = None
        else:
            px = int(float(x) * w_half) + offset_x
            py = int(float(y) * h)
            pts[joint] = (px, py)

    for a, b in EDGES:
        pa, pb = pts.get(a), pts.get(b)
        if pa and pb:
            cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)

    for joint, p in pts.items():
        if p:
            cv2.circle(canvas, p, dot_r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, dot_r, (255, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton_full(frame, joints_xy, color, thickness=2, dot_r=5):
    """원본 영상 전체 해상도 위에 skeleton 오버레이."""
    h, w = frame.shape[:2]
    pts = {}
    for joint in JOINTS:
        x = joints_xy.get(f"{joint}_x")
        y = joints_xy.get(f"{joint}_y")
        if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
            pts[joint] = None
        else:
            pts[joint] = (int(float(x) * w), int(float(y) * h))

    for a, b in EDGES:
        pa, pb = pts.get(a), pts.get(b)
        if pa and pb:
            cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)

    for joint, p in pts.items():
        if p:
            cv2.circle(frame, p, dot_r, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p, dot_r, (255, 255, 255), 1, cv2.LINE_AA)


# ── 버전 1: 흰 배경 skeleton-only ─────────────────────────────────────────────

def render_skeleton_only(pred_df, out_path, canvas_w=1280, canvas_h=720, fps=30):
    """흰 배경, 왼쪽=GT / 오른쪽=Pred."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))
    half = canvas_w // 2

    for _, row in pred_df.iterrows():
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        # 중앙 구분선
        cv2.line(canvas, (half, 0), (half, canvas_h), (200, 200, 200), 1)

        gt_xy   = {f"{j}_{ax}": row[f"gt_{j}_{ax}"]   for j in JOINTS for ax in ("x", "y")}
        pred_xy = {f"{j}_{ax}": row[f"pred_{j}_{ax}"] for j in JOINTS for ax in ("x", "y")}

        draw_skeleton_on(canvas, gt_xy,   GT_COLOR,   offset_x=0,    thickness=2)
        draw_skeleton_on(canvas, pred_xy, PRED_COLOR, offset_x=half, thickness=2)

        t_sec = row["timestamp_sec"]
        cv2.putText(canvas, f"GT   {t_sec:.2f}s",   (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GT_COLOR,   2, cv2.LINE_AA)
        cv2.putText(canvas, f"Pred {t_sec:.2f}s", (half + 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, PRED_COLOR, 2, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()


# ── 버전 2: 원본 영상 오버레이 ────────────────────────────────────────────────

def render_video_overlay(pred_df, frame_ts_path, video_path, out_path):
    """원본 .avi 프레임 위에 GT(초록) + Pred(빨강) skeleton 오버레이."""
    frame_ts = pd.read_csv(frame_ts_path)
    pred_ts  = pred_df["timestamp_sec"].to_numpy()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    ts_arr       = frame_ts["timestamp_sec"].to_numpy()
    pred_indices = np.clip(np.searchsorted(pred_ts, ts_arr, side="left"), 0, len(pred_df) - 1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(pred_indices):
            row = pred_df.iloc[pred_indices[frame_idx]]
            gt_xy   = {f"{j}_{ax}": row[f"gt_{j}_{ax}"]   for j in JOINTS for ax in ("x", "y")}
            pred_xy = {f"{j}_{ax}": row[f"pred_{j}_{ax}"] for j in JOINTS for ax in ("x", "y")}

            draw_skeleton_full(frame, gt_xy,   GT_COLOR,   thickness=2)
            draw_skeleton_full(frame, pred_xy, PRED_COLOR, thickness=2)

            t_sec = float(row["timestamp_sec"])
            cv2.putText(frame, "GT",   (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GT_COLOR,   2, cv2.LINE_AA)
            cv2.putText(frame, "Pred", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, PRED_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{t_sec:.2f}s", (w - 110, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


# ── trial 단위 실행 ───────────────────────────────────────────────────────────

def render_trial(subject, label, trial, out_dir, exp_name=None, base_dir=None):
    prefix    = subject_prefix(subject)
    if base_dir:
        pred_path = Path(base_dir) / "predictions" / f"{subject}_{label}_{trial}_prediction.csv"
    else:
        exp       = exp_name if exp_name else label
        pred_path = ROOT / "experiments" / subject / exp / "predictions" / \
                    f"{prefix}_{label}_{trial}_prediction.csv"

    if not pred_path.exists():
        print(f"  [SKIP] prediction 없음: {pred_path.name}")
        return

    pred_df = pd.read_csv(pred_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 버전 1: 항상 생성 ──────────────────────────────────────────────────
    out_skeleton = out_dir / f"{prefix}_{label}_{trial}_skeleton_only.mp4"
    render_skeleton_only(pred_df, out_skeleton)
    print(f"  [skeleton_only] {out_skeleton.name}")

    # ── 버전 2: 영상 있을 때만 ────────────────────────────────────────────
    video_subpath = LABEL_VIDEO_PATH.get(label, "")
    video_dir  = ROOT / f"csi_to_pose_{subject}" / "videos" / video_subpath / prefix
    video_path = video_dir / f"{prefix}_{label}_{trial}.avi"
    ts_path    = video_dir / f"{prefix}_{label}_{trial}_frame_timestamps.csv"

    if video_path.exists() and ts_path.exists():
        out_overlay = out_dir / f"{prefix}_{label}_{trial}_video_overlay.mp4"
        render_video_overlay(pred_df, ts_path, video_path, out_overlay)
        print(f"  [video_overlay] {out_overlay.name}")
    else:
        print(f"  [video_overlay] 영상 없음 → skeleton_only만 생성")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  default="mhw")
    parser.add_argument("--label",    default="post_fall_inactive")
    parser.add_argument("--exp_name", default=None,
                        help="실험 폴더명 (기본값=label). ALL 모드: --exp_name ALL")
    parser.add_argument("--exp_dir",  default=None,
                        help="실험 폴더 전체 경로 (cross 실험 등: experiments/cross_trainlmh_testmhw/ALL)")
    parser.add_argument("--trial",    default=None)
    parser.add_argument("--trials",   nargs="+", default=None)
    args = parser.parse_args()

    if args.exp_dir:
        base_dir = ROOT / args.exp_dir if not Path(args.exp_dir).is_absolute() else Path(args.exp_dir)
        pred_dir_root = base_dir / "predictions"
        out_dir       = base_dir / "video_overlay"
    else:
        base_dir      = None
        exp_name      = args.exp_name if args.exp_name else args.label
        pred_dir_root = ROOT / "experiments" / args.subject / exp_name / "predictions"
        out_dir       = ROOT / "experiments" / args.subject / exp_name / "video_overlay"

    labels = ALL_LABELS if args.label == "ALL" else [args.label]

    if args.trials:
        trials = args.trials
    elif args.trial:
        trials = [args.trial]
    else:
        trials = None  # 라벨별로 자동 탐색

    print(f"\n[render] subject={args.subject}  label={args.label}")

    for label in labels:
        if trials is None:
            if not pred_dir_root.exists():
                print(f"prediction 폴더 없음: {pred_dir_root}")
                sys.exit(1)
            label_trials = sorted(set(
                p.stem.replace(f"{args.subject}_{label}_", "").replace("_prediction", "")
                for p in pred_dir_root.glob(f"{args.subject}_{label}_*_prediction.csv")
            ))
        else:
            label_trials = trials

        print(f"\n[label={label}]  trials={label_trials}")
        for trial in label_trials:
            print(f"  trial={trial}")
            render_trial(args.subject, label, trial, out_dir, base_dir=base_dir, exp_name=args.exp_name)

    print("\n완료")


if __name__ == "__main__":
    main()
