import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "csi_to_pose"
POSE_ROOT = EXPERIMENT_ROOT / "pose_gt"
PLOT_ROOT = EXPERIMENT_ROOT / "pose_plots"
OVERLAY_ROOT = EXPERIMENT_ROOT / "pose_overlays"

PROXY_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MP_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def landmark_to_dict(lm):
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(lm.visibility),
    }


def mean_points(points):
    valid = [p for p in points if p is not None and not np.isnan(p["x"])]
    if not valid:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "visibility": 0.0}
    return {
        "x": float(np.mean([p["x"] for p in valid])),
        "y": float(np.mean([p["y"] for p in valid])),
        "z": float(np.mean([p["z"] for p in valid])),
        "visibility": float(np.mean([p["visibility"] for p in valid])),
    }


def point_distance(a, b):
    if a is None or b is None:
        return np.nan
    vals = [a["x"], a["y"], a["z"], b["x"], b["y"], b["z"]]
    if any(np.isnan(v) for v in vals):
        return np.nan
    return float(
        math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)
    )


def xy_angle_degrees(a, b):
    if a is None or b is None:
        return np.nan
    vals = [a["x"], a["y"], b["x"], b["y"]]
    if any(np.isnan(v) for v in vals):
        return np.nan
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return float(math.degrees(math.atan2(dy, dx)))


def empty_point():
    return {"x": np.nan, "y": np.nan, "z": np.nan, "visibility": 0.0}


def build_proxy(full):
    def p(name):
        return full.get(name, empty_point())

    head_candidates = [
        p("nose"),
        p("left_eye"),
        p("right_eye"),
        p("left_ear"),
        p("right_ear"),
    ]
    return {
        "head": mean_points(head_candidates),
        "left_shoulder": p("left_shoulder"),
        "right_shoulder": p("right_shoulder"),
        "left_elbow": p("left_elbow"),
        "right_elbow": p("right_elbow"),
        "left_wrist": p("left_wrist"),
        "right_wrist": p("right_wrist"),
        "left_hip": p("left_hip"),
        "right_hip": p("right_hip"),
        "left_knee": p("left_knee"),
        "right_knee": p("right_knee"),
        "left_ankle": p("left_ankle"),
        "right_ankle": p("right_ankle"),
    }


def build_derived(proxy, prev_proxy, sample_rate):
    shoulder_center = mean_points([proxy["left_shoulder"], proxy["right_shoulder"]])
    hip_center = mean_points([proxy["left_hip"], proxy["right_hip"]])
    knee_center = mean_points([proxy["left_knee"], proxy["right_knee"]])
    ankle_center = mean_points([proxy["left_ankle"], proxy["right_ankle"]])
    body_center = mean_points([shoulder_center, hip_center])

    body_height = point_distance(proxy["head"], ankle_center)
    torso_angle = xy_angle_degrees(shoulder_center, hip_center)

    prev_body_center = None
    if prev_proxy is not None:
        prev_shoulder = mean_points([prev_proxy["left_shoulder"], prev_proxy["right_shoulder"]])
        prev_hip = mean_points([prev_proxy["left_hip"], prev_proxy["right_hip"]])
        prev_body_center = mean_points([prev_shoulder, prev_hip])
        prev_ankle = mean_points([prev_proxy["left_ankle"], prev_proxy["right_ankle"]])
        prev_body_height = point_distance(prev_proxy["head"], prev_ankle)
    else:
        prev_body_height = np.nan

    motion_velocity = point_distance(body_center, prev_body_center) * sample_rate if prev_proxy else np.nan
    height_velocity = (body_height - prev_body_height) * sample_rate if prev_proxy else np.nan
    wrist_motion = (
        np.nanmean(
            [
                point_distance(proxy["left_wrist"], prev_proxy["left_wrist"]),
                point_distance(proxy["right_wrist"], prev_proxy["right_wrist"]),
            ]
        )
        * sample_rate
        if prev_proxy
        else np.nan
    )
    knee_motion = (
        np.nanmean(
            [
                point_distance(proxy["left_knee"], prev_proxy["left_knee"]),
                point_distance(proxy["right_knee"], prev_proxy["right_knee"]),
            ]
        )
        * sample_rate
        if prev_proxy
        else np.nan
    )
    ankle_motion = (
        np.nanmean(
            [
                point_distance(proxy["left_ankle"], prev_proxy["left_ankle"]),
                point_distance(proxy["right_ankle"], prev_proxy["right_ankle"]),
            ]
        )
        * sample_rate
        if prev_proxy
        else np.nan
    )

    return {
        "shoulder_center_x": shoulder_center["x"],
        "shoulder_center_y": shoulder_center["y"],
        "shoulder_center_z": shoulder_center["z"],
        "hip_center_x": hip_center["x"],
        "hip_center_y": hip_center["y"],
        "hip_center_z": hip_center["z"],
        "knee_center_x": knee_center["x"],
        "knee_center_y": knee_center["y"],
        "knee_center_z": knee_center["z"],
        "ankle_center_x": ankle_center["x"],
        "ankle_center_y": ankle_center["y"],
        "ankle_center_z": ankle_center["z"],
        "body_center_x": body_center["x"],
        "body_center_y": body_center["y"],
        "body_center_z": body_center["z"],
        "body_height": body_height,
        "torso_angle": torso_angle,
        "motion_velocity": motion_velocity,
        "height_velocity": height_velocity,
        "knee_motion": knee_motion,
        "wrist_motion": wrist_motion,
        "ankle_motion": ankle_motion,
    }


def flatten_landmarks(prefix, items):
    row = {}
    for name, point in items.items():
        for key in ("x", "y", "z", "visibility"):
            row[f"{prefix}{name}_{key}"] = point[key]
    return row


def infer_label_parts(video_path, label, subject, trial):
    stem = Path(video_path).stem
    parts = stem.split("_")
    if subject is None and parts:
        subject = parts[0]
    if trial is None:
        for part in reversed(parts):
            if part.startswith("t") and part[1:].isdigit():
                trial = part
                break
    if label is None and subject and trial and stem.startswith(subject + "_"):
        label = stem[len(subject) + 1 :]
        if label.endswith("_" + trial):
            label = label[: -(len(trial) + 1)]
    return label or "unknown_label", subject or "unknown_subject", trial or "unknown_trial"


def load_frame_timestamps(video_path):
    timestamp_path = Path(video_path).with_name(f"{Path(video_path).stem}_frame_timestamps.csv")
    if not timestamp_path.exists():
        return None
    df = pd.read_csv(timestamp_path)
    if "timestamp_sec" not in df.columns:
        return None
    return pd.to_numeric(df["timestamp_sec"], errors="coerce").to_numpy()


def extract(video_path, label, subject, trial, make_overlay=True):
    video_path = Path(video_path)
    label, subject, trial = infer_label_parts(video_path, label, subject, trial)
    base = f"{subject}_{label}_{trial}"

    full_out = POSE_ROOT / "full_33_landmarks" / label / subject / f"{base}_full33.csv"
    proxy_out = POSE_ROOT / "proxy_13" / label / subject / f"{base}_proxy13.csv"
    feature_plot = PLOT_ROOT / label / subject / f"{base}_derived_features.png"
    overlay_out = OVERLAY_ROOT / label / subject / f"{base}_overlay.mp4"

    for path in (full_out, proxy_out, feature_plot, overlay_out):
        ensure_parent(path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    frame_timestamps = load_frame_timestamps(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if make_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_out), fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    full_rows = []
    proxy_rows = []
    prev_proxy = None
    prev_timestamp_sec = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if frame_timestamps is not None and frame_idx < len(frame_timestamps):
                timestamp_sec = float(frame_timestamps[frame_idx])
            else:
                timestamp_sec = frame_idx / fps

            if result.pose_landmarks:
                full = {
                    name: landmark_to_dict(result.pose_landmarks.landmark[idx])
                    for name, idx in MP_INDEX.items()
                }
                proxy = build_proxy(full)
                if prev_timestamp_sec is not None:
                    dt = timestamp_sec - prev_timestamp_sec
                    sample_rate = 1.0 / dt if dt > 1e-6 else fps
                else:
                    sample_rate = fps
                derived = build_derived(proxy, prev_proxy, sample_rate)
                prev_proxy = proxy
                prev_timestamp_sec = timestamp_sec

                full_row = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "subject": subject,
                    "label": label,
                    "trial": trial,
                    "pose_detected": 1,
                }
                full_row.update(flatten_landmarks("", full))
                full_rows.append(full_row)

                proxy_row = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "subject": subject,
                    "label": label,
                    "trial": trial,
                    "pose_detected": 1,
                }
                proxy_row.update(flatten_landmarks("", proxy))
                proxy_row.update(derived)
                proxy_rows.append(proxy_row)

                if writer:
                    mp_draw.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
            else:
                full_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "subject": subject,
                        "label": label,
                        "trial": trial,
                        "pose_detected": 0,
                    }
                )
                proxy_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "subject": subject,
                        "label": label,
                        "trial": trial,
                        "pose_detected": 0,
                    }
                )

            if writer:
                writer.write(frame)
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    pd.DataFrame(full_rows).to_csv(full_out, index=False)
    proxy_df = pd.DataFrame(proxy_rows)
    proxy_df.to_csv(proxy_out, index=False)
    make_feature_plot(proxy_df, feature_plot)
    print_summary(proxy_df, video_path, proxy_out, feature_plot, overlay_out if make_overlay else None)


def make_feature_plot(df, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [
        "body_center_x",
        "body_center_y",
        "body_height",
        "torso_angle",
        "motion_velocity",
        "knee_motion",
        "ankle_motion",
        "wrist_motion",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    x = df["timestamp_sec"] if "timestamp_sec" in df.columns else df.index
    for ax, col in zip(axes, available):
        ax.plot(x, df[col], linewidth=1.2)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (sec)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_summary(df, video_path, proxy_out, feature_plot, overlay_out):
    detected = int(df.get("pose_detected", pd.Series(dtype=int)).sum())
    total = len(df)
    ratio = detected / total * 100 if total else 0
    print("\n[Pose Feature Extraction]")
    print(f"video: {video_path}")
    print(f"frames: {total}")
    print(f"pose detected: {detected}/{total} ({ratio:.1f}%)")
    print(f"proxy csv: {proxy_out}")
    print(f"plot: {feature_plot}")
    if overlay_out:
        print(f"overlay: {overlay_out}")
    for col in ("body_height", "torso_angle", "motion_velocity", "knee_motion", "ankle_motion", "wrist_motion"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                print(
                    f"{col}: min={series.min():.4f}, mean={series.mean():.4f}, max={series.max():.4f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe 13-point skeleton proxy and derived features from video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--label")
    parser.add_argument("--subject")
    parser.add_argument("--trial")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()
    extract(
        video_path=args.video,
        label=args.label,
        subject=args.subject,
        trial=args.trial,
        make_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
