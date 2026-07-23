from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import serial

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.csi import EXPECTED_TX, parse_csi_line
from notifi_collection.files import write_trial_checksums
from notifi_collection.labels import (
    LABELS,
    TRIAL_DURATION_S,
    cue_for_trial,
    get_label,
    variant_for_trial,
)
from notifi_collection.session import DEFAULT_CONFIG_PATH, SessionConfig, load_session
from notifi_collection.sound import play_sound


CSV_FIELDS = (
    "pc_monotonic_ns",
    "pc_elapsed_s",
    "pc_wall_time_iso",
    "sender_id",
    "sender_mac",
    "sequence_number",
    "firmware_timestamp_us",
    "rssi",
    "rate",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "sig_len",
    "rx_state",
    "csi_len",
    "first_word_invalid",
    "csi_data",
    "subject_id",
    "environment_id",
    "session_id",
    "source_trial_uid",
    "risk_label",
    "scenario_label",
    "planned_cue_s",
    "planned_variant",
    "raw_line",
)

MANIFEST_FIELDS = (
    "created_at",
    "source_trial_uid",
    "subject_id",
    "environment_id",
    "session_id",
    "scenario_id",
    "scenario_label",
    "risk_label",
    "context",
    "trial_number",
    "duration_s",
    "planned_cue_s",
    "planned_variant",
    "outcome",
    "contact_side",
    "first_contact_region",
    "actual_onset_s",
    "impact_s",
    "action_end_s",
    "manual_qc",
    "automatic_qc",
    "tx1_frames",
    "tx2_frames",
    "tx3_frames",
    "unknown_mac_frames",
    "video_frames",
    "effective_video_fps",
    "csi_path",
    "video_path",
    "metadata_path",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_camera(config: SessionConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.camera_index}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Camera {config.camera_index} opened but returned no frame")
    return cap


def next_trial_number(label_root: Path) -> int:
    numbers: list[int] = []
    if label_root.exists():
        for child in label_root.iterdir():
            if not child.is_dir():
                continue
            marker = child.name.rsplit("_t", 1)
            if len(marker) == 2 and marker[1].isdigit():
                numbers.append(int(marker[1]))
    return max(numbers, default=0) + 1


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_manifest(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def validate_session_limits(
    rows: list[dict[str, str]],
    config: SessionConfig,
    risk: str,
    requested: int,
) -> None:
    session_rows = [
        row
        for row in rows
        if row.get("subject_id") == config.subject
        and row.get("environment_id") == config.environment
        and row.get("session_id") == config.session_id
    ]
    if len(session_rows) + requested > 60:
        raise ValueError(
            f"Session accepted/captured limit would exceed 60: "
            f"existing={len(session_rows)}, requested={requested}"
        )
    danger_count = sum(row.get("risk_label") == "DANGER" for row in session_rows)
    if risk == "DANGER" and danger_count + requested > 10:
        raise ValueError(
            f"Session danger limit would exceed 10: existing={danger_count}, requested={requested}"
        )


def csi_worker(
    ser: serial.Serial,
    path: Path,
    start_event: threading.Event,
    stop_event: threading.Event,
    clock: dict,
    context: dict,
    result: dict,
    errors: list[BaseException],
) -> None:
    counts: Counter[str] = Counter()
    total = 0
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            start_event.wait()
            while not stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                frame = parse_csi_line(line)
                if frame is None:
                    continue
                monotonic_ns = time.monotonic_ns()
                counts[frame.sender_id] += 1
                total += 1
                writer.writerow(
                    {
                        "pc_monotonic_ns": monotonic_ns,
                        "pc_elapsed_s": f"{(monotonic_ns - clock['trial_start_ns']) / 1e9:.9f}",
                        "pc_wall_time_iso": utc_now(),
                        "sender_id": frame.sender_id,
                        "sender_mac": frame.sender_mac,
                        "sequence_number": frame.sequence_number,
                        "firmware_timestamp_us": frame.firmware_timestamp_us,
                        "rssi": frame.rssi,
                        "rate": frame.rate,
                        "noise_floor": frame.noise_floor,
                        "fft_gain": frame.fft_gain,
                        "agc_gain": frame.agc_gain,
                        "channel": frame.channel,
                        "sig_len": frame.sig_len,
                        "rx_state": frame.rx_state,
                        "csi_len": frame.csi_len,
                        "first_word_invalid": frame.first_word_invalid,
                        "csi_data": frame.csi_data,
                        **context,
                        "raw_line": frame.raw_line,
                    }
                )
    except BaseException as exc:
        errors.append(exc)
        stop_event.set()
    finally:
        result.update({"counts": dict(counts), "total": total})


def video_worker(
    cap: cv2.VideoCapture,
    video_path: Path,
    timestamp_path: Path,
    config: SessionConfig,
    start_event: threading.Event,
    stop_event: threading.Event,
    clock: dict,
    result: dict,
    errors: list[BaseException],
    preview: bool,
) -> None:
    writer = None
    frame_rows: list[tuple[int, int, float]] = []
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.camera_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.camera_height
        requested_fps = float(config.camera_fps)
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            requested_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {video_path}")

        start_event.wait()
        frame_index = 0
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            monotonic_ns = time.monotonic_ns()
            elapsed_s = (monotonic_ns - clock["trial_start_ns"]) / 1e9
            writer.write(frame)
            frame_rows.append((frame_index, monotonic_ns, elapsed_s))
            frame_index += 1

            if preview:
                shown = frame.copy()
                cv2.putText(
                    shown,
                    f"REC {elapsed_s:04.1f}s",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("NotiFi collection preview", shown)
                cv2.waitKey(1)
    except BaseException as exc:
        errors.append(exc)
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyWindow("NotiFi collection preview")
        with timestamp_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(("frame_index", "pc_monotonic_ns", "pc_elapsed_s"))
            csv_writer.writerows(
                (index, monotonic_ns, f"{elapsed_s:.9f}")
                for index, monotonic_ns, elapsed_s in frame_rows
            )
        last_elapsed = frame_rows[-1][2] if frame_rows else 0.0
        result.update(
            {
                "frames": len(frame_rows),
                "effective_fps": len(frame_rows) / last_elapsed if last_elapsed > 0 else 0.0,
            }
        )


def countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print(f"[READY] First trial starts in {seconds}s.")
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}s ", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20, end="\r")


def default_outcome(risk: str) -> str:
    return {"SAFE": "normal", "WARNING": "recovered", "DANGER": "ground_contact"}[risk]


def default_contact_side(variant: str) -> str:
    if "left" in variant:
        return "left"
    if "right" in variant:
        return "right"
    return "none"


def first_contact_region(label: str) -> str:
    return {
        "fall_from_standing": "hip_then_forearm",
        "fall_while_walking": "knee_then_forearm",
        "fall_collapse": "knee_then_hip_then_forearm",
        "bed_fall": "hip_then_forearm",
        "chair_fall": "hip_then_forearm",
    }.get(label, "none")


def collect_one(
    ser: serial.Serial,
    cap: cv2.VideoCapture,
    config: SessionConfig,
    spec,
    trial_number: int,
    output_root: Path,
    sound_enabled: bool,
    preview: bool,
    min_link_ratio: float,
) -> tuple[dict, bool]:
    cue_s = cue_for_trial(spec, trial_number)
    variant = variant_for_trial(spec, trial_number)
    trial_token = f"t{trial_number:03d}"
    source_uid = (
        f"{config.subject}_{config.environment}_{config.session_id}_"
        f"{spec.label}_{trial_token}"
    )
    trial_dir = (
        output_root
        / config.subject
        / config.environment
        / config.session_id
        / spec.risk.lower()
        / spec.label
        / source_uid
    )
    if trial_dir.exists():
        raise FileExistsError(f"Trial directory already exists: {trial_dir}")
    trial_dir.mkdir(parents=True)

    csi_path = trial_dir / f"{source_uid}_csi.csv"
    video_path = trial_dir / f"{source_uid}_video.mp4"
    video_ts_path = trial_dir / f"{source_uid}_video_timestamps.csv"
    meta_path = trial_dir / f"{source_uid}_meta.json"

    context = {
        "subject_id": config.subject,
        "environment_id": config.environment,
        "session_id": config.session_id,
        "source_trial_uid": source_uid,
        "risk_label": spec.risk,
        "scenario_label": spec.label,
        "planned_cue_s": "" if cue_s is None else cue_s,
        "planned_variant": variant,
    }
    clock: dict[str, int] = {}
    start_event = threading.Event()
    stop_event = threading.Event()
    errors: list[BaseException] = []
    csi_result: dict = {}
    video_result: dict = {}

    ser.reset_input_buffer()
    csi_thread = threading.Thread(
        target=csi_worker,
        args=(
            ser,
            csi_path,
            start_event,
            stop_event,
            clock,
            context,
            csi_result,
            errors,
        ),
        daemon=True,
    )
    video_thread = threading.Thread(
        target=video_worker,
        args=(
            cap,
            video_path,
            video_ts_path,
            config,
            start_event,
            stop_event,
            clock,
            video_result,
            errors,
            preview,
        ),
        daemon=True,
    )
    csi_thread.start()
    video_thread.start()

    clock["trial_start_ns"] = time.monotonic_ns()
    start_wall = utc_now()
    start_event.set()
    play_sound("trial_start", sound_enabled)
    print(
        f"[START] {source_uid} | label={spec.label} | variant={variant} | "
        f"cue={'none' if cue_s is None else f'{cue_s:.1f}s'}"
    )

    cue_monotonic_ns = None
    while (time.monotonic_ns() - clock["trial_start_ns"]) / 1e9 < TRIAL_DURATION_S:
        if errors:
            break
        elapsed = (time.monotonic_ns() - clock["trial_start_ns"]) / 1e9
        if cue_s is not None and cue_monotonic_ns is None and elapsed >= cue_s:
            cue_monotonic_ns = time.monotonic_ns()
            play_sound("action_cue", sound_enabled)
            print(f"[ACTION CUE] {cue_s:.1f}s - perform the action once.")
        time.sleep(0.005)

    trial_end_ns = time.monotonic_ns()
    stop_event.set()
    csi_thread.join(timeout=3)
    video_thread.join(timeout=3)
    play_sound("trial_end", sound_enabled)

    if csi_thread.is_alive() or video_thread.is_alive():
        raise RuntimeError("Collection worker did not stop cleanly")
    if errors:
        raise RuntimeError(f"Collection worker failed: {errors[0]}")

    counts = csi_result.get("counts", {})
    tx_counts = {
        "TX1": int(counts.get("TX1", 0)),
        "TX2": int(counts.get("TX2", 0)),
        "TX3": int(counts.get("TX3", 0)),
    }
    unknown_count = int(counts.get("UNKNOWN", 0))
    expected_per_link = config.packet_rate * TRIAL_DURATION_S
    minimum_per_link = int(expected_per_link * min_link_ratio)
    links_ok = all(value >= minimum_per_link for value in tx_counts.values())
    video_frames = int(video_result.get("frames", 0))
    video_ok = video_frames >= int(config.camera_fps * TRIAL_DURATION_S * 0.8)
    automatic_qc = "AUTO_PASS_PENDING_MANUAL_QC" if links_ok and video_ok else "AUTO_REJECT"

    meta = {
        "dataset_version": "v2.0",
        "created_at": start_wall,
        "source_trial_uid": source_uid,
        "subject_id": config.subject,
        "environment_id": config.environment,
        "session_id": config.session_id,
        "scenario_id": spec.id,
        "scenario_label": spec.label,
        "risk_label": spec.risk,
        "presence_label": "absent" if spec.label == "absence" else "present",
        "fall_event": int(spec.risk == "DANGER"),
        "context": spec.context,
        "outcome": default_outcome(spec.risk),
        "phase_labels": ["pre", "action", "contact", "post"],
        "contact_side": default_contact_side(variant),
        "first_contact_region": first_contact_region(spec.label),
        "trial_number": trial_number,
        "duration_s": TRIAL_DURATION_S,
        "planned_cue_s": cue_s,
        "planned_variant": variant,
        "actual_onset_s": None,
        "impact_s": None,
        "action_end_s": None,
        "annotation_status": "PENDING",
        "manual_qc": "PENDING",
        "valid_for_reconstruction": spec.valid_for_reconstruction,
        "shared_pc_monotonic_clock": True,
        "trial_start_monotonic_ns": clock["trial_start_ns"],
        "action_cue_monotonic_ns": cue_monotonic_ns,
        "trial_end_monotonic_ns": trial_end_ns,
        "trial_elapsed_s": (trial_end_ns - clock["trial_start_ns"]) / 1e9,
        "firmware_commit": config.firmware_commit,
        "device_config_id": config.device_config_id,
        "channel": config.channel,
        "packet_rate_per_tx": config.packet_rate,
        "baud": config.baud,
        "tx_mac_map": EXPECTED_TX,
        "device_positions": {
            "RX": config.rx_position,
            "TX1": config.tx1_position,
            "TX2": config.tx2_position,
            "TX3": config.tx3_position,
        },
        "camera": {
            "index": config.camera_index,
            "requested_width": config.camera_width,
            "requested_height": config.camera_height,
            "requested_fps": config.camera_fps,
            "frame_count": video_frames,
            "effective_fps": video_result.get("effective_fps", 0.0),
        },
        "automatic_qc": {
            "status": automatic_qc,
            "minimum_link_frames": minimum_per_link,
            "link_counts": tx_counts,
            "unknown_mac_frames": unknown_count,
            "video_ok": video_ok,
        },
        "files": {
            "csi": csi_path.name,
            "video": video_path.name,
            "video_timestamps": video_ts_path.name,
        },
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_trial_checksums(trial_dir)

    manifest_row = {
        "created_at": start_wall,
        "source_trial_uid": source_uid,
        "subject_id": config.subject,
        "environment_id": config.environment,
        "session_id": config.session_id,
        "scenario_id": spec.id,
        "scenario_label": spec.label,
        "risk_label": spec.risk,
        "context": spec.context,
        "trial_number": trial_number,
        "duration_s": TRIAL_DURATION_S,
        "planned_cue_s": "" if cue_s is None else cue_s,
        "planned_variant": variant,
        "outcome": default_outcome(spec.risk),
        "contact_side": default_contact_side(variant),
        "first_contact_region": first_contact_region(spec.label),
        "actual_onset_s": "",
        "impact_s": "",
        "action_end_s": "",
        "manual_qc": "PENDING",
        "automatic_qc": automatic_qc,
        "tx1_frames": tx_counts["TX1"],
        "tx2_frames": tx_counts["TX2"],
        "tx3_frames": tx_counts["TX3"],
        "unknown_mac_frames": unknown_count,
        "video_frames": video_frames,
        "effective_video_fps": f"{video_result.get('effective_fps', 0.0):.3f}",
        "csi_path": str(csi_path.relative_to(output_root)),
        "video_path": str(video_path.relative_to(output_root)),
        "metadata_path": str(meta_path.relative_to(output_root)),
    }
    print(
        f"[QC] TX1={tx_counts['TX1']} TX2={tx_counts['TX2']} "
        f"TX3={tx_counts['TX3']} minimum={minimum_per_link} "
        f"video={video_frames} -> {automatic_qc}"
    )
    return manifest_row, automatic_qc != "AUTO_REJECT"


def print_plan(config: SessionConfig, spec, repeat: int, start_trial: int) -> None:
    print("\n[NotiFi v2.0 Collection Plan]")
    print(
        f"subject={config.subject} environment={config.environment} "
        f"session={config.session_id}"
    )
    print(
        f"label={spec.label} ({spec.korean}) risk={spec.risk} "
        f"repeat={repeat} duration={TRIAL_DURATION_S:.0f}s"
    )
    print(
        f"trial_range=t{start_trial:03d}-t{start_trial + repeat - 1:03d} "
        f"default_break=2.0s"
    )
    print(f"start_pose: {spec.start_pose}")
    for phase, action in spec.timeline:
        print(f"  {phase}: {action}")
    print(f"end_pose: {spec.end_pose}")
    print(f"recollect_if: {spec.recollect}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect synchronized 3TX+1RX CSI and RGB video for NotiFi v2.0."
    )
    parser.add_argument("--label", required=True, choices=tuple(LABELS))
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "collection_data" / "v2",
    )
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--break-sec", type=float, default=2.0)
    parser.add_argument("--min-link-ratio", type=float, default=0.90)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    parser.add_argument("--continue-on-qc-fail", action="store_true")
    parser.add_argument(
        "--safety-confirmed",
        action="store_true",
        help="Required for DANGER trials after mat, spotter, chair, and participant checks.",
    )
    args = parser.parse_args()

    config = load_session(args.config)
    spec = get_label(args.label)
    repeat = args.repeat if args.repeat is not None else spec.repeats_per_environment
    if repeat <= 0:
        raise ValueError("--repeat must be positive")
    if spec.risk == "DANGER" and not args.safety_confirmed:
        raise ValueError(
            "DANGER collection requires --safety-confirmed after completing the safety checklist."
        )
    if not 0 < args.min_link_ratio <= 1:
        raise ValueError("--min-link-ratio must be within (0, 1]")

    manifest_path = args.output_root / "manifests" / "trials.csv"
    existing_rows = load_manifest_rows(manifest_path)
    label_rows = [
        row
        for row in existing_rows
        if row.get("subject_id") == config.subject
        and row.get("environment_id") == config.environment
        and row.get("scenario_label") == spec.label
    ]
    recorded_numbers = [
        int(row["trial_number"])
        for row in label_rows
        if str(row.get("trial_number", "")).isdigit()
    ]
    start_trial = max(recorded_numbers, default=0) + 1
    accepted_count = sum(
        row.get("automatic_qc") != "AUTO_REJECT"
        and row.get("manual_qc") != "REJECT"
        for row in label_rows
    )
    if not args.allow_extra and accepted_count + repeat > spec.repeats_per_environment:
        remaining = max(0, spec.repeats_per_environment - accepted_count)
        raise ValueError(
            f"{spec.label} target is {spec.repeats_per_environment} per environment. "
            f"Only {remaining} trial(s) remain. Use --repeat {remaining}."
        )

    validate_session_limits(existing_rows, config, spec.risk, repeat)
    print_plan(config, spec, repeat, start_trial)
    if args.dry_run:
        print("[DRY RUN] Hardware was not opened and no files were written.")
        return

    countdown(args.delay)
    cap = None
    ser = None
    sound_enabled = not args.no_sound
    completed = 0
    try:
        cap = ensure_camera(config)
        ser = serial.Serial(config.port, config.baud, timeout=0.1)
        time.sleep(0.5)
        for offset in range(repeat):
            trial_number = start_trial + offset
            row, passed = collect_one(
                ser=ser,
                cap=cap,
                config=config,
                spec=spec,
                trial_number=trial_number,
                output_root=args.output_root,
                sound_enabled=sound_enabled,
                preview=args.preview,
                min_link_ratio=args.min_link_ratio,
            )
            append_manifest(manifest_path, row)
            completed += 1
            if not passed and not args.continue_on_qc_fail:
                raise RuntimeError(
                    "Automatic QC failed. Collection stopped; inspect TX power, antenna, "
                    "camera, and serial port before retrying with a new session ID."
                )
            if offset < repeat - 1:
                if spec.risk == "DANGER" and completed == 5:
                    print("[SAFETY BREAK] Five DANGER trials completed. Rest for 180 seconds.")
                    time.sleep(180)
                else:
                    print(f"[BREAK] {args.break_sec:.1f}s")
                    time.sleep(args.break_sec)

        play_sound("set_done", sound_enabled)
        print(f"[ALL DONE] {completed}/{repeat} trials completed.")
        if spec.risk == "DANGER" and completed >= 10:
            print("[SAFETY] Rest for at least 10 minutes before another DANGER set.")
    except Exception:
        play_sound("error", sound_enabled)
        raise
    finally:
        if ser is not None and ser.is_open:
            ser.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
