"""데이터 계약 — 경로, 컬럼 화이트리스트, 관절 정의, 파싱 규칙.

이 파일의 상수는 전 파이프라인의 단일 진실 공급원이다. 여기 없는 컬럼명·관절명·
매직넘버를 다른 모듈에 직접 써넣지 말 것.

값의 근거는 실제 데이터 전수/표본 검사이며, 각 상수 옆에 근거를 남긴다.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 경로

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 데이터셋 패키지 루트. 실제 데이터는 환경변수로 명시하며 저장소에 넣지 않는다.
DATASET_ROOT = Path(
    os.environ.get("NOTIFI_DATASET_ROOT", PROJECT_ROOT / "data")
)

# Timestamp files are intentionally stored outside the training ZIP. The relative
# hierarchy below this root matches data/<task>/ after dropping the task component.
TIMESTAMP_ROOT = Path(
    os.environ.get(
        "NOTIFI_TIMESTAMP_ROOT",
        PROJECT_ROOT / "timestamps",
    )
)
TIMESTAMP_MANIFEST_CSV = Path(
    os.environ.get(
        "NOTIFI_TIMESTAMP_MANIFEST",
        TIMESTAMP_ROOT / "timestamp_manifest.csv",
    )
)

#: 파생물(인덱스, 캐시, 리포트) 저장 위치. 데이터셋 원본은 읽기 전용으로 다룬다.
WORK_ROOT = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT_ROOT / "work_v2"))
INDEX_DIR = WORK_ROOT / "index"
CACHE_DIR = WORK_ROOT / "cache"
REPORT_DIR = WORK_ROOT / "reports"
SPLIT_DIR = WORK_ROOT / "splits"


def fold_index_path(fold: str, split: str) -> Path:
    """`folds/test_ajh/train_index.csv` 같은 경로."""
    return DATASET_ROOT / "folds" / fold / f"{split}_index.csv"


def fold_quality_path(fold: str, split: str) -> Path:
    return DATASET_ROOT / "folds" / fold / f"{split}_sample_quality.csv"


TIMESTAMP_QUALITY_CSV = DATASET_ROOT / "timestamp_quality.csv"
LABELS_CSV = DATASET_ROOT / "labels.csv"
CSI_SCHEMA_CSV = DATASET_ROOT / "csi_feature_schema.csv"

# ---------------------------------------------------------------- 축

SUBJECTS = ("ajh", "lmh", "mhw", "yja")
ENVIRONMENTS = ("E01", "E02", "E03")
LOSO_FOLDS = tuple(f"test_{s}" for s in SUBJECTS)
SPLITS = ("train", "val", "test")

TASK_POSE = "pose_and_action"
TASK_CLS = "classification_only"

N_CLASSES = 17
N_RISK = 3

#: action/risk 출력 ID의 단일 해석 계약. class 6은 absence calibration 전용이며
#: 실제 행동 query 평가에서는 제외한다.
ACTION_NAMES = (
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
RISK_NAMES = ("safe", "warning", "danger")
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}
RISK_TO_ID = {name: index for index, name in enumerate(RISK_NAMES)}
CALIBRATION_PROMPT_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)

assert len(ACTION_NAMES) == N_CLASSES
assert len(RISK_NAMES) == N_RISK

#: 전수 확인된 trial 수 (verify_dataset 의 기준값)
EXPECTED_TRIALS_TOTAL = 3299
EXPECTED_TRIALS_POSE = 3155
EXPECTED_TRIALS_CLS = 144

# ---------------------------------------------------------------- CSI

#: 모델 입력으로 허용된 컬럼. 데이터셋의 csi_feature_schema.csv 에서
#: use_for_model in {yes, optional} 인 것만. 나머지(subject_id, session_id,
#: risk_label, sender_mac, pc_wall_time_iso ...)는 **읽지도 않는다** — 실수로
#: 피처에 섞이는 경로를 원천 차단하기 위해 usecols 로 걸러낸다.
CSI_USECOLS = (
    "pc_elapsed_s",       # 시간 정렬 (trial 상대 시각, 초)
    "sender_id",          # 링크 식별 TX1/TX2/TX3
    "csi_data",           # I/Q 페이로드
    "csi_len",            # 품질 검사 (파손 행 탐지)
    "first_word_invalid",  # 품질 검사
    "rssi",               # 라디오 메타 (선택 입력)
)

#: 절대 로드하면 안 되는 컬럼. 테스트에서 CSI_USECOLS 와 교집합이 0인지 확인한다.
CSI_FORBIDDEN_COLS = (
    "subject_id", "environment_id", "session_id", "source_trial_uid",
    "risk_label", "scenario_label", "sender_mac", "pc_wall_time_iso",
    "pc_monotonic_ns", "planned_cue_s", "planned_variant", "raw_line",
)

#: TX 문자열 → 링크 인덱스. 순서 고정 (캐시 호환성에 영향).
LINKS = ("TX1", "TX2", "TX3")
LINK_INDEX = {tx: i for i, tx in enumerate(LINKS)}
N_LINKS = len(LINKS)

# Installation bearings are fixed across environments. North is +y and east
# is +x. Distances and heights are intentionally omitted because they are not
# fixed deployment invariants.
BOARD_BEARINGS = {
    "RX": (0.0, 1.0),
    "TX1": (0.0, -1.0),
    "TX2": (-1.0, 0.0),
    "TX3": (1.0, 0.0),
}

# Per-link geometry: transmitter bearing followed by the unit TX->RX vector.
# The row order is contractually identical to LINKS.
INV_SQRT_2 = 2.0 ** -0.5
LINK_GEOMETRY = (
    (0.0, -1.0, 0.0, 1.0),
    (-1.0, 0.0, INV_SQRT_2, INV_SQRT_2),
    (1.0, 0.0, -INV_SQRT_2, INV_SQRT_2),
)

#: csi_data 배열 길이. subcarrier 128개 × (I, Q).
CSI_RAW_LEN = 256
N_SUBCARRIERS = CSI_RAW_LEN // 2

#: 항상 0인 subcarrier — 802.11 OFDM 의 DC null(중심)과 대역 가장자리 guard band.
#: 27 trial 19,630 패킷 전수 확인. 모든 subject·환경에서 예외 없이 0이다.
#: 저장·학습에서 제외한다. 남겨두면 정규화에서 0/0 이 나고 파라미터만 낭비된다.
GUARD_SUBCARRIERS = (0, 1, 2, 3, 4, 5, 63, 64, 65, 123, 124, 125, 126, 127)
LIVE_SUBCARRIERS = tuple(i for i in range(N_SUBCARRIERS) if i not in GUARD_SUBCARRIERS)
N_LIVE_SUBCARRIERS = len(LIVE_SUBCARRIERS)   # 114

#: 파손 행 판정 — 시리얼 로그가 잘려 CSV 필드가 밀린 행이 존재한다
#: (source 표본 실측: mhw 0.76%, ajh 0.29%, lmh 0.01%).
#: 원본 라인 자체가 truncate 되어 복구 불가능하므로 드롭한다.
#: `csi_len == 256` 이고 csi_data 가 정확히 256개 정수로 파싱되는 행만 채택.
def is_valid_csi_len(value) -> bool:
    try:
        return int(value) == CSI_RAW_LEN
    except (TypeError, ValueError):
        return False


#: trial 당 파손 행 비율이 이 값을 넘으면 품질 경고. (드롭 자체는 항상 수행)
MAX_BAD_ROW_RATIO = 0.05

# ---------------------------------------------------------------- 시간축

#: 공통 리샘플 격자. GT 영상이 30fps 300프레임(10초)이므로 동일 격자를 쓴다.
TARGET_FPS = 30.0
NOMINAL_TRIAL_FRAMES = 300

#: 캐시의 고정 프레임 길이. 실측 분포는 198~304 이고, 304 로 잡으면 잘리는 프레임이
#: 0개다(300 으로 잡으면 860 trial 에서 1,670 프레임이 잘린다). 300 대비 용량 차이는
#: 0.01 GB 뿐이라 손실 없는 쪽을 택한다. 짧은 trial 은 뒤를 0 으로 채우고
#: valid/link_mask 를 False 로 둔다.
CACHE_FRAMES = 304

#: CSI 입력 표현. 캐시 마지막 축(크기 2)의 의미를 정한다.
#:
#:   "amp_phase" (기본, v3.0.0~)
#:       ch0 = 진폭 |H|
#:       ch1 = 정제 위상 — 패킷 내 subcarrier 축의 선형성분(랜덤 타이밍·CFO)을 제거한 잔차
#:
#:   "iq" (v2.0.0 재현용, 쓰지 말 것)
#:       ch0 = I, ch1 = Q
#:
#: 왜 바꿨는가: ESP32 CSI 는 패킷마다 위상이 랜덤이라(공통 위상회전 std 1.79 rad,
#: 균등랜덤 이론값 1.81) 생 I/Q 를 선형보간하면 진폭이 깎인다. 방향이 제각각인 두
#: 화살표의 중간점을 잡는 것과 같다. 실측 손실 중앙 12%, 하위 10% 는 52%.
#: 30fps 격자점의 78% 가 패킷 사이라 입력 대부분이 오염됐었다.
CSI_REPRESENTATION = "amp_phase"

#: 리샘플 시각 주변 이 간격 안에 실측 패킷이 없으면 link_mask=False.
#: 링크별 패킷율 표본 실측: p5 ≈ 10.6 Hz, 중앙 ≈ 25.7 Hz, p95 ≈ 40 Hz.
#: 최저 링크(≈6.7 Hz, 간격 150ms)까지 살리면 보간이 과해지므로 100ms 로 자른다.
MAX_GAP_S = 0.100

#: timestamp_quality.csv 의 time_method 값. lmh의 부분 로그는 이름은 그대로
#: 유지하되 실제 기록 구간을 전체 GT 프레임 축에 보간한다. k/30 가정은 쓰지 않는다.
TIME_METHOD_TIMESTAMPS = "timestamps"
TIME_METHOD_UNIFORM = "uniform_30fps"

# ---------------------------------------------------------------- 관절

#: GVHMR joints_world 의 관절 순서 = SMPL body-22.
#:
#: 검증 방법(추측 아님): 아래 JOINT_PARENTS 트리로 뼈 길이를 재면 프레임 간
#: 상대표준편차가 0.0000 (4 subject 전부). 무작위로 섞은 트리는 0.012~0.039.
#: 실측 뼈 길이도 대퇴 0.392m / 경골 0.414m / 상완 0.271m / 전완 0.260m 로
#: 해부학적으로 타당하며, 정지 서기에서 y 순위가 head→neck→shoulder ...
#: →ankle→foot 로 정렬된다.
JOINT_NAMES = (
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
)
N_JOINTS = len(JOINT_NAMES)

JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

#: SMPL kinematic tree. 루트(pelvis)의 부모는 -1.
JOINT_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

#: bone loss 용 엣지 21개 = 트리 그대로.
SKELETON_EDGES = tuple(
    (parent, child) for child, parent in enumerate(JOINT_PARENTS) if parent >= 0
)

#: root 관절. **transl 이 아니다.** transl 은 SMPL 파라미터의 pelvis translation 이라
#: joints_world[:, 0] 과 y축으로 약 -0.35m 어긋난다(표본 실측 -0.349 ~ -0.381).
#: 절대 skeleton 을 복원할 때는 반드시 joints_world[:, 0] 을 기준으로 쓴다.
ROOT_JOINT = 0

#: 중력 정렬 축 (Y-up, 바닥 y≈0). 정지 서기 실측: pelvis 0.877, head 1.537, 발 0.005.
UP_AXIS = 1

#: 좌/우 대칭쌍 (augmentation·평가에서 사용).
SYMMETRIC_PAIRS = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
)

#: 부위별 관절 그룹 (충돌 부위 라벨링·부위별 MPJPE 리포트용).
JOINT_GROUPS = {
    "head": (12, 15),
    "torso": (0, 3, 6, 9, 13, 14),
    "left_arm": (16, 18, 20),
    "right_arm": (17, 19, 21),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}

#: GT 스키마. 일부 파일에만 SMPL 파라미터가 추가될 수 있다.
#: **공통 계약은 joints_world / transl / frame_index 3개뿐이다.** 선택 필드를
#: 학습 필수 경로에 넣으면 데이터 생산 버전이 다른 fold에서 코드가 깨진다.
GT_SCHEMA_JOINTS = "gvhmr_joints_v1"
GT_SCHEMA_SMPL = "gvhmr_smpl_full_v1"
GT_REQUIRED_KEYS = ("joints_world", "transl", "frame_index")

# ---------------------------------------------------------------- 버전

#: 전처리 규약 버전. CSI 파싱/리샘플/마스크 규칙이 바뀌면 올린다.
#: 캐시 메타에 기록되어 스키마 변경 시 자동 무효화된다.
#:   v2.0.0  생 I/Q 를 보간
#:   v3.0.0  보간 전에 (진폭, 정제위상) 으로 변환  <- 현재
PREPROC_VERSION = "v3.0.0"
