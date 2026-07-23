from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.files import write_trial_checksums


PROXY_JOINTS = (
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
)

MP_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def load_metadata(path: Path) -> tuple[Path, dict]:
    if path.is_dir():
        matches = list(path.glob("*_meta.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one *_meta.json in {path}")
        path = matches[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def mean_landmark(landmarks, names: tuple[str, ...]) -> tuple[float, float, float, float]:
    points = [landmarks[MP_INDEX[name]] for name in names]
    return (
        float(np.mean([point.x for point in points])),
        float(np.mean([point.y for point in points])),
        float(np.mean([point.z for point in points])),
        float(np.mean([point.visibility for point in points])),
    )


def proxy_from_landmarks(landmarks) -> dict[str, tuple[float, float, float, float]]:
    proxy = {
        "head": mean_landmark(
            landmarks,
            ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
        )
    }
    for name in PROXY_JOINTS[1:]:
        point = landmarks[MP_INDEX[name]]
        proxy[name] = (
            float(point.x),
            float(point.y),
            float(point.z),
            float(point.visibility),
        )
    return proxy


def nearest_sync_p95_ms(video_ns: list[int], csi_ns: list[int]) -> float | None:
    if not video_ns or not csi_ns:
        return None
    residuals = []
    for value in video_ns:
        index = bisect.bisect_left(csi_ns, value)
        candidates = []
        if index < len(csi_ns):
            candidates.append(abs(csi_ns[index] - value))
        if index > 0:
            candidates.append(abs(csi_ns[index - 1] - value))
        if candidates:
            residuals.append(min(candidates) / 1e6)
    return float(np.percentile(residuals, 95)) if residuals else None


def read_video_timestamps(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [int(row["pc_monotonic_ns"]) for row in csv.DictReader(handle)]


def read_csi_timestamps(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [int(row["pc_monotonic_ns"]) for row in csv.DictReader(handle)]
    return sorted(values)


def process_trial(meta_path: Path, overlay: bool) -> dict:
    meta_path, meta = load_metadata(meta_path)
    uid = meta["source_trial_uid"]
    if not meta.get("valid_for_reconstruction", True):
        print(f"[SKIP] {uid}: valid_for_reconstruction=false")
        return {"source_trial_uid": uid, "status": "SKIPPED"}

    trial_dir = meta_path.parent
    video_path = trial_dir / meta["files"]["video"]
    video_ts_path = trial_dir / meta["files"]["video_timestamps"]
    csi_path = trial_dir / meta["files"]["csi"]
    pose_path = trial_dir / f"{uid}_pose13.csv"
    pose_qc_path = trial_dir / f"{uid}_pose_qc.json"
    overlay_path = trial_dir / f"{uid}_pose_overlay.mp4"

    video_times_ns = read_video_timestamps(video_ts_path)
    csi_times_ns = read_csi_timestamps(csi_path)
    sync_p95_ms = nearest_sync_p95_ms(video_times_ns, csi_times_ns)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    overlay_writer = None
    if overlay:
        overlay_writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    mp_pose = mp.solutions.pose
    drawer = mp.solutions.drawing_utils
    fieldnames = ["frame_index", "pc_monotonic_ns", "pc_elapsed_s", "pose_valid"]
    for joint in PROXY_JOINTS:
        fieldnames.extend(
            (
                f"{joint}_x",
                f"{joint}_y",
                f"{joint}_z",
                f"{joint}_visibility",
                f"{joint}_valid",
            )
        )

    total_frames = 0
    detected_frames = 0
    ankle_visible_frames = 0
    rows: list[dict] = []
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = total_frames
            total_frames += 1
            monotonic_ns = (
                video_times_ns[frame_index]
                if frame_index < len(video_times_ns)
                else int(meta["trial_start_monotonic_ns"] + frame_index / fps * 1e9)
            )
            elapsed_s = (monotonic_ns - int(meta["trial_start_monotonic_ns"])) / 1e9
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {
                "frame_index": frame_index,
                "pc_monotonic_ns": monotonic_ns,
                "pc_elapsed_s": f"{elapsed_s:.9f}",
                "pose_valid": 0,
            }
            if result.pose_world_landmarks is not None:
                detected_frames += 1
                row["pose_valid"] = 1
                proxy = proxy_from_landmarks(result.pose_world_landmarks.landmark)
                ankle_visibility = min(
                    proxy["left_ankle"][3],
                    proxy["right_ankle"][3],
                )
                if ankle_visibility >= 0.5:
                    ankle_visible_frames += 1
                for joint, values in proxy.items():
                    x, y, z, visibility = values
                    row.update(
                        {
                            f"{joint}_x": x,
                            f"{joint}_y": y,
                            f"{joint}_z": z,
                            f"{joint}_visibility": visibility,
                            f"{joint}_valid": int(visibility >= 0.5),
                        }
                    )
            else:
                for joint in PROXY_JOINTS:
                    row.update(
                        {
                            f"{joint}_x": "",
                            f"{joint}_y": "",
                            f"{joint}_z": "",
                            f"{joint}_visibility": 0.0,
                            f"{joint}_valid": 0,
                        }
                    )
            rows.append(row)

            if overlay_writer is not None:
                if result.pose_landmarks is not None:
                    drawer.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                overlay_writer.write(frame)
    finally:
        pose.close()
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()

    with pose_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pose_valid_ratio = detected_frames / total_frames if total_frames else 0.0
    ankle_visible_ratio = ankle_visible_frames / total_frames if total_frames else 0.0
    if pose_valid_ratio < 0.95:
        status = "REJECT"
    elif sync_p95_ms is not None and sync_p95_ms > 100:
        status = "REJECT"
    elif sync_p95_ms is not None and sync_p95_ms > 50:
        status = "REVIEW"
    else:
        status = "PASS"
    qc = {
        "source_trial_uid": uid,
        "status": status,
        "total_video_frames": total_frames,
        "pose_detected_frames": detected_frames,
        "pose_valid_ratio": pose_valid_ratio,
        "ankle_visible_ratio": ankle_visible_ratio,
        "sync_residual_p95_ms": sync_p95_ms,
        "pose_csv": pose_path.name,
        "overlay_video": overlay_path.name if overlay else None,
    }
    pose_qc_path.write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta["pose_qc"] = qc
    meta["files"]["pose13"] = pose_path.name
    meta["files"]["pose_qc"] = pose_qc_path.name
    if overlay:
        meta["files"]["pose_overlay"] = overlay_path.name
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_trial_checksums(trial_dir)
    print(
        f"[POSE] {uid}: detected={pose_valid_ratio:.1%}, "
        f"ankles={ankle_visible_ratio:.1%}, sync_p95={sync_p95_ms}, status={status}"
    )
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 13-point pose teacher GT.")
    parser.add_argument("target", nargs="?", type=Path, help="Trial directory or *_meta.json")
    parser.add_argument("--root", type=Path, help="Process every trial under this root")
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()
    if (args.target is None) == (args.root is None):
        raise ValueError("Provide exactly one target or --root")

    if args.target is not None:
        process_trial(args.target, args.overlay)
        return
    meta_files = sorted(args.root.rglob("*_meta.json"))
    print(f"[POSE] processing {len(meta_files)} trial(s)")
    failed = 0
    for meta_path in meta_files:
        try:
            process_trial(meta_path, args.overlay)
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {meta_path}: {exc}")
    if failed:
        raise SystemExit(f"{failed} trial(s) failed")


if __name__ == "__main__":
    main()
