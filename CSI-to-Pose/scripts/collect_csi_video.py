import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from save_csi_raw import LABEL_MAP, next_trial


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "csi_to_pose"
DATA_ROOT = EXPERIMENT_ROOT / "data"
VIDEO_ROOT = EXPERIMENT_ROOT / "videos"
METADATA_PATH = EXPERIMENT_ROOT / "collection_logs" / "csi_to_pose_trials.csv"

SOUNDS = {
    "start": "/System/Library/Sounds/Ping.aiff",
    "trial_done": "/System/Library/Sounds/Pop.aiff",
    "set_done": "/System/Library/Sounds/Glass.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
}


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def play_sound(name, enabled=True):
    if not enabled:
        return
    sound_path = SOUNDS.get(name)
    if not sound_path or not os.path.exists(sound_path):
        return
    subprocess.Popen(
        ["afplay", sound_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_metadata(row):
    ensure_parent(METADATA_PATH)
    exists = METADATA_PATH.exists()
    fields = [
        "created_at",
        "subject",
        "risk",
        "domain",
        "label",
        "trial",
        "ambient",
        "duration",
        "camera_index",
        "fps",
        "effective_fps",
        "video_path",
        "frame_timestamp_path",
        "csi_expected_path",
        "note",
    ]
    with METADATA_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def open_camera(camera_index, width, height, fps):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "macOS 카메라 권한 또는 camera index를 확인하세요."
        )

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width or 1280
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height or 720
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if not actual_fps or actual_fps <= 1:
        actual_fps = fps or 30

    return cap, actual_width, actual_height, actual_fps


def record_video(video_path, camera_index, duration, width, height, fps, preview_pose=False):
    cap, actual_width, actual_height, actual_fps = open_camera(
        camera_index, width, height, fps
    )
    ensure_parent(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path), fourcc, float(actual_fps), (actual_width, actual_height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {video_path}")

    frame_count = 0
    frame_timestamps = []
    start = time.time()
    pose = None
    mp_pose = None
    mp_draw = None
    window_name = "NotiFi CSI-to-Pose Live MediaPipe Preview"
    if preview_pose:
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    try:
        while time.time() - start < duration:
            ok, frame = cap.read()
            if not ok:
                continue
            writer.write(frame)
            elapsed = time.time() - start
            frame_timestamps.append(elapsed)
            frame_count += 1

            if preview_pose:
                shown = frame.copy()
                rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        shown,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    status = "POSE DETECTED"
                    color = (0, 220, 0)
                else:
                    status = "NO POSE"
                    color = (0, 0, 255)

                cv2.putText(
                    shown,
                    f"{status} | {elapsed:04.1f}/{duration:.1f}s | camera={camera_index}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    preview_pose = False
                    cv2.destroyWindow(window_name)
    finally:
        if pose is not None:
            pose.close()
        if preview_pose:
            cv2.destroyWindow(window_name)

    writer.release()
    cap.release()
    timestamp_path = video_path.with_name(f"{video_path.stem}_frame_timestamps.csv")
    ensure_parent(timestamp_path)
    with timestamp_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame_idx", "timestamp_sec"])
        for idx, ts in enumerate(frame_timestamps):
            writer_csv.writerow([idx, f"{ts:.6f}"])
    effective_fps = frame_count / frame_timestamps[-1] if frame_timestamps else 0.0
    return frame_count, actual_fps, effective_fps, timestamp_path


def run_trial(args, trial):
    risk, domain = LABEL_MAP[args.label]
    filename = f"{args.subject}_{args.label}_{trial}"
    video_path = VIDEO_ROOT / risk / domain / args.label / args.subject / f"{filename}.mp4"
    csi_path = DATA_ROOT / risk / domain / args.label / args.subject / f"{filename}.csv"

    if video_path.exists() and not args.overwrite:
        raise FileExistsError(f"Video already exists: {video_path}")
    if csi_path.exists() and not args.overwrite:
        raise FileExistsError(f"CSI already exists: {csi_path}")

    print(f"\n[CSI-to-Pose] {args.label}/{args.ambient} {trial}")
    print(f"video: {video_path}")
    print(f"csi:   {csi_path}")
    print("[SYNC] 녹화가 시작되면 바로 손을 크게 들거나 박수 1회를 해주세요.")

    save_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "save_csi_raw.py"),
        "--port",
        args.port,
        "--baud",
        str(args.baud),
        "--subject",
        args.subject,
        "--label",
        args.label,
        "--trial",
        trial,
        "--duration",
        str(args.duration),
        "--repeat",
        "1",
        "--delay",
        "0",
        "--break_sec",
        "0",
        "--data_root",
        str(DATA_ROOT),
    ]

    csi_proc = subprocess.Popen(save_cmd)
    try:
        frames, actual_fps, effective_fps, timestamp_path = record_video(
            video_path=video_path,
            camera_index=args.camera,
            duration=args.duration,
            width=args.width,
            height=args.height,
            fps=args.fps,
            preview_pose=args.preview_pose,
        )
    finally:
        csi_return = csi_proc.wait()

    if csi_return != 0:
        raise RuntimeError(f"CSI collection failed with exit code {csi_return}")

    write_metadata(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "subject": args.subject,
            "risk": risk,
            "domain": domain,
            "label": args.label,
            "trial": trial,
            "ambient": args.ambient,
            "duration": args.duration,
            "camera_index": args.camera,
            "fps": round(float(actual_fps), 3),
            "effective_fps": round(float(effective_fps), 3),
            "video_path": str(video_path),
            "frame_timestamp_path": str(timestamp_path),
            "csi_expected_path": str(csi_path),
            "note": args.note,
        }
    )

    print(f"[DONE] video frames={frames}, writer_fps={actual_fps:.2f}, effective_fps={effective_fps:.2f}")
    print(f"[DONE] frame timestamps: {timestamp_path}")
    return video_path, csi_path


def countdown(seconds):
    if seconds <= 0:
        return
    print(f"[READY] {seconds}s 후 수집 시작. 위치로 이동하세요.")
    for remain in range(seconds, 0, -1):
        print(f"  starting in {remain}s...", end="\r")
        time.sleep(1)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect paired CSI CSV and video for CSI-to-Pose experiments."
    )
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--subject", default="yja")
    parser.add_argument("--label", required=True, choices=list(LABEL_MAP.keys()))
    parser.add_argument("--trial", default="t001")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--break_sec", type=float, default=1.5)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--ambient", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_sound", action="store_true")
    parser.add_argument(
        "--preview_pose",
        action="store_true",
        help="수집 중 MediaPipe skeleton overlay 창을 실시간으로 띄웁니다. q를 누르면 preview만 닫힙니다.",
    )
    args = parser.parse_args()
    sound_enabled = not args.no_sound

    print("start sound: Ping.aiff")
    print("trial done sound: Pop.aiff")
    print("set done sound: Glass.aiff")
    print("error sound: Basso.aiff")

    countdown(args.delay)
    trial = args.trial
    try:
        for idx in range(args.repeat):
            print(f"\n[TRIAL {idx + 1}/{args.repeat}]")
            play_sound("start", sound_enabled)
            run_trial(args, trial)
            play_sound("trial_done", sound_enabled)
            trial = next_trial(trial)
            if idx < args.repeat - 1 and args.break_sec > 0:
                print(f"[BREAK] {args.break_sec}s")
                time.sleep(args.break_sec)

        play_sound("set_done", sound_enabled)
        print("\n[ALL DONE] paired CSI/video collection complete.")
    except Exception:
        play_sound("error", sound_enabled)
        raise


if __name__ == "__main__":
    main()
