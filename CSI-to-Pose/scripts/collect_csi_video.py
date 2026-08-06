import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from save_csi_raw import LABEL_MAP, next_trial

if platform.system() == "Windows":
    import winsound

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "csi_to_pose"
DATA_ROOT = EXPERIMENT_ROOT / "csi"
VIDEO_ROOT = EXPERIMENT_ROOT / "videos"
METADATA_PATH = EXPERIMENT_ROOT / "collection_logs" / "csi_to_pose_trials.csv"

_SOUNDS_MAC = {
    "ready":      "/System/Library/Sounds/Tink.aiff",
    "start":      "/System/Library/Sounds/Ping.aiff",
    "trial_done": "/System/Library/Sounds/Pop.aiff",
    "set_done":   "/System/Library/Sounds/Glass.aiff",
    "error":      "/System/Library/Sounds/Basso.aiff",
}

_SOUNDS_WIN = {
    "ready":      (660,  150),
    "start":      (880,  200),
    "trial_done": (1047, 300),
    "set_done":   (1319, 500),
    "error":      (440,  600),
}

AMBIENT_SCHEDULE = {
    25: [("quiet", 7), ("aircon", 6), ("tv", 6), ("music", 6)],
    15: [("quiet", 4), ("aircon", 4), ("tv", 4), ("music", 3)],
    10: [("quiet", 3), ("aircon", 3), ("tv", 2), ("music", 2)],
}


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def get_ambient_for_trial(trial_str, total):
    num = int(trial_str[1:])  # "t007" -> 7
    schedule = AMBIENT_SCHEDULE.get(total)
    if not schedule:
        return None
    cumulative = 0
    for ambient, count in schedule:
        cumulative += count
        if num <= cumulative:
            return ambient
    return None


def play_sound(name, enabled=True):
    if not enabled:
        return
    if platform.system() == "Windows":
        params = _SOUNDS_WIN.get(name)
        if params:
            winsound.Beep(*params)
    elif platform.system() == "Darwin":
        sound_path = _SOUNDS_MAC.get(name)
        if sound_path and os.path.exists(sound_path):
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
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "카메라 권한 또는 camera index를 확인하세요."
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

    print(f"[CAM] 해상도: {actual_width}x{actual_height}, FPS: {actual_fps:.1f}")
    return cap, actual_width, actual_height, actual_fps


def record_video(video_path, cap, actual_width, actual_height, actual_fps, duration, preview_pose=False):
    ensure_parent(video_path)

    if platform.system() == "Windows":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path), fourcc, float(actual_fps), (actual_width, actual_height)
    )
    if not writer.isOpened():
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
                    f"{status} | {elapsed:04.1f}/{duration:.1f}s",
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
    timestamp_path = video_path.with_name(f"{video_path.stem}_frame_timestamps.csv")
    ensure_parent(timestamp_path)
    with timestamp_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame_idx", "timestamp_sec"])
        for idx, ts in enumerate(frame_timestamps):
            writer_csv.writerow([idx, f"{ts:.6f}"])
    effective_fps = frame_count / frame_timestamps[-1] if frame_timestamps else 0.0
    return frame_count, actual_fps, effective_fps, timestamp_path


def run_trial(args, trial, ambient, sound_enabled=True):
    risk, domain = LABEL_MAP[args.label]
    filename = f"{args.subject}_{args.label}_{trial}"
    ext = "avi" if platform.system() == "Windows" else "mp4"
    video_path = VIDEO_ROOT / risk / domain / args.label / args.subject / f"{filename}.{ext}"
    csi_path = DATA_ROOT / risk / domain / args.label / args.subject / f"{filename}.csv"

    if video_path.exists() and not args.overwrite:
        raise FileExistsError(f"Video already exists: {video_path}")
    if csi_path.exists() and not args.overwrite:
        raise FileExistsError(f"CSI already exists: {csi_path}")

    print(f"\n[CSI-to-Pose] {args.label}/{ambient} {trial}")
    print(f"video: {video_path}")
    print(f"csi:   {csi_path}")

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

    # 1. 카메라 먼저 초기화 (CSI 시작 전)
    print("[CAM] 카메라 초기화 중...")
    cap, actual_width, actual_height, actual_fps = open_camera(
        args.camera, args.width, args.height, args.fps
    )

    try:
        # 2. CSI 수집 시작
        csi_proc = subprocess.Popen(save_cmd)

        # 3. 삑 (카메라+CSI 모두 준비 완료)
        play_sound("start", sound_enabled)
        print("[REC] 녹화 시작!")

        # 4. 녹화 (카메라 이미 열려있음)
        try:
            frames, actual_fps, effective_fps, timestamp_path = record_video(
                video_path=video_path,
                cap=cap,
                actual_width=actual_width,
                actual_height=actual_height,
                actual_fps=actual_fps,
                duration=args.duration,
                preview_pose=args.preview_pose,
            )
        finally:
            csi_return = csi_proc.wait()
    finally:
        cap.release()

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
            "ambient": ambient,
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


def countdown(seconds, sound_enabled=True):
    if seconds <= 0:
        return
    play_sound("ready", sound_enabled)
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
    parser.add_argument("--break_sec", type=float, default=3.0)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--ambient", default=None, help="ambient 환경 (quiet/aircon/tv/music). --total 사용 시 자동 결정.")
    parser.add_argument("--total", type=int, choices=[25, 15, 10], default=None,
                        help="총 수집 횟수 (25/15/10). trial 번호 기준으로 ambient를 자동 결정합니다.")
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

    if args.ambient is None and args.total is None:
        parser.error("--ambient 또는 --total 중 하나는 필수입니다.")

    sound_enabled = not args.no_sound

    if platform.system() == "Windows":
        print("start sound: 880Hz beep")
        print("trial done sound: 1047Hz beep")
        print("set done sound: 1319Hz beep")
        print("error sound: 440Hz beep")
    else:
        print("start sound: Ping.aiff")
        print("trial done sound: Pop.aiff")
        print("set done sound: Glass.aiff")
        print("error sound: Basso.aiff")

    countdown(args.delay, sound_enabled)
    trial = args.trial
    try:
        for idx in range(args.repeat):
            if args.total:
                ambient = get_ambient_for_trial(trial, args.total)
            else:
                ambient = args.ambient
            print(f"\n[TRIAL {idx + 1}/{args.repeat}] ambient={ambient}")
            run_trial(args, trial, ambient, sound_enabled)
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
