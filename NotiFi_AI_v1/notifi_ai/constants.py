"""Stable public labels and deployment tensor dimensions."""

from __future__ import annotations

MODEL_NAME = "NotiFi_AI_v1"
SCHEMA_VERSION = 1
TARGET_FPS = 30.0
MAX_FRAMES = 304
N_LINKS = 3
N_SUBCARRIERS = 114
N_CHANNELS = 2
N_JOINTS = 22

LINK_ORDER = ("TX1", "TX2", "TX3")
LINK_DIRECTIONS = {
    "RX": "North",
    "TX1": "South",
    "TX2": "West",
    "TX3": "East",
}

ACTION_LABELS = (
    "walking",
    "standing_still",
    "sitting_still",
    "lying_still",
    "lie_to_stand",
    "stand_to_lie_normal",
    "absence",
    "sit_to_stand",
    "stand_to_sit",
    "unstable_walking",
    "stumble_recover",
    "bed_exit_failed",
    "fall_from_standing",
    "fall_while_walking",
    "bed_exit_fall",
    "bed_fall",
    "chair_exit_fall",
)

RISK_LABELS = ("safe", "warning", "danger")

JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

ACTION_TO_RISK = tuple(
    0 if index < 9 else 1 if index < 12 else 2
    for index in range(len(ACTION_LABELS))
)

BASIC_CALIBRATION_ACTIONS = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
)

OPTIONAL_DANGER_CALIBRATION_ACTIONS = (12, 13, 14, 15, 16)

