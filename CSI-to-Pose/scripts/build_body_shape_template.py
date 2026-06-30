from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CSI_TO_POSE_ROOT = ROOT / "csi_to_pose"

JOINTS = [
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

SEGMENTS = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_lower_arm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_lower_arm": ("right_elbow", "right_wrist"),
    "left_upper_leg": ("left_hip", "left_knee"),
    "left_lower_leg": ("left_knee", "left_ankle"),
    "right_upper_leg": ("right_hip", "right_knee"),
    "right_lower_leg": ("right_knee", "right_ankle"),
}


def distance(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    cols = [f"{a}_x", f"{a}_y", f"{a}_z", f"{b}_x", f"{b}_y", f"{b}_z"]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return np.sqrt(
        (values[f"{a}_x"] - values[f"{b}_x"]) ** 2
        + (values[f"{a}_y"] - values[f"{b}_y"]) ** 2
        + (values[f"{a}_z"] - values[f"{b}_z"]) ** 2
    )


def median_finite(values: pd.Series, fallback: float = 0.05) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return fallback
    return float(np.median(arr))


def load_proxy_frames(label: str, subject: str, trials: list[str]) -> pd.DataFrame:
    frames = []
    for trial in trials:
        path = (
            CSI_TO_POSE_ROOT
            / "pose_gt"
            / "proxy_13"
            / label
            / subject
            / f"{subject}_{label}_{trial}_proxy13.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["source_trial"] = trial
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_template(df: pd.DataFrame, *, subject: str, label: str, trials: list[str]) -> dict:
    detected = df[df.get("pose_detected", True).astype(bool)].copy() if "pose_detected" in df.columns else df.copy()
    if detected.empty:
        detected = df.copy()

    segment_lengths = {name: median_finite(distance(detected, a, b)) for name, (a, b) in SEGMENTS.items()}
    shoulder_width = segment_lengths["shoulder_width"]
    hip_width = segment_lengths["hip_width"]
    torso_length = median_finite(
        np.sqrt(
            (detected["shoulder_center_x"] - detected["hip_center_x"]) ** 2
            + (detected["shoulder_center_y"] - detected["hip_center_y"]) ** 2
            + (detected["shoulder_center_z"] - detected["hip_center_z"]) ** 2
        )
        if {"shoulder_center_x", "hip_center_x"}.issubset(detected.columns)
        else distance(detected, "left_shoulder", "left_hip")
    )
    body_height = median_finite(detected["body_height"]) if "body_height" in detected.columns else torso_length * 2.4
    body_height = max(body_height, 1e-3)

    # Radii are relative to the observed body proportions. They are not a true
    # surface scan, but they make the CSI pose readable as a person-shaped proxy.
    radii = {
        "head": max(shoulder_width * 0.24, body_height * 0.035),
        "torso": max((shoulder_width + hip_width) * 0.16, body_height * 0.035),
        "upper_arm": max(shoulder_width * 0.075, body_height * 0.014),
        "lower_arm": max(shoulder_width * 0.06, body_height * 0.012),
        "upper_leg": max(hip_width * 0.12, body_height * 0.018),
        "lower_leg": max(hip_width * 0.09, body_height * 0.014),
    }

    return {
        "schema": "notifi_body_shape_template_v1",
        "subject": subject,
        "label_source": label,
        "trials": trials,
        "coordinate_space": "mediapipe_normalized_world_proxy",
        "body_height": body_height,
        "torso_length": torso_length,
        "segment_lengths": segment_lengths,
        "body_widths": {
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
        },
        "radii": radii,
        "rendering": {
            "body_color": "#8fb8ff",
            "body_alpha": 0.42,
            "skeleton_color": "#2a2f3a",
            "joint_color": "#1d4ed8",
        },
        "notes": [
            "This is a personalized body-volume proxy derived from 13-point MediaPipe pose GT.",
            "It is not a photorealistic mesh or SMPL body. Use it as a readable shape prior under CSI-predicted skeletons.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a personal body shape template for CSI-to-Pose rendering.")
    parser.add_argument("--label", default="unstable_walking")
    parser.add_argument("--subject", default="yja")
    parser.add_argument("--trials", nargs="+", default=[f"t{i:03d}" for i in range(1, 11)])
    parser.add_argument("--out")
    args = parser.parse_args()

    df = load_proxy_frames(args.label, args.subject, args.trials)
    template = build_template(df, subject=args.subject, label=args.label, trials=args.trials)
    out = Path(args.out) if args.out else CSI_TO_POSE_ROOT / "body_templates" / args.subject / f"{args.subject}_body_shape_template.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"body_template: {out}")
    print(json.dumps({k: template[k] for k in ["body_height", "torso_length", "body_widths", "radii"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
