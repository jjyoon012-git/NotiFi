import argparse
import ast
import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "split_manifest" / "manifest_full.csv"

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

SKELETON_EDGES = [
    ("head", "left_shoulder"),
    ("head", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

DERIVED_COLS = [
    "body_center_x",
    "body_center_y",
    "body_height",
    "torso_angle",
    "motion_velocity",
    "knee_motion",
    "wrist_motion",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_csi_array(raw_line):
    start = raw_line.rfind("[")
    end = raw_line.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        arr = np.array(ast.literal_eval(raw_line[start : end + 1]), dtype=np.float32)
    except Exception:
        return None
    if len(arr) < 2:
        return None
    if len(arr) % 2 == 1:
        arr = arr[:-1]
    real = arr[0::2]
    imag = arr[1::2]
    amp = np.sqrt(real * real + imag * imag)
    return amp


def sanitize_phase(real, imag, amp=None):
    """패킷 1개의 I/Q → 새니타이즈된 위상 잔차의 (cos, sin).  (C-MambaPose 위상 전처리 이식)

    ESP32 위상은 CFO/SFO(송수신 클럭 오프셋)가 subcarrier축 선형 성분으로 오염됨.
    ① subcarrier축 unwrap (±π 감김 풀기)
    ② 유효 subcarrier에 최소제곱 직선 피팅 → 빼기 (하드웨어 오프셋 제거)
    ③ 잔차(다중경로 기하 성분)를 cos/sin 2채널로 (각도 불연속 제거 + [-1,1] 정규화)
    null/guard subcarrier(진폭~0)는 위상이 정의 안 됨 → 0 마스크.
    """
    if amp is None:
        amp = np.sqrt(real * real + imag * imag)
    S = len(amp)
    cosr = np.zeros(S, dtype=np.float32)
    sinr = np.zeros(S, dtype=np.float32)
    idx = np.nonzero(amp > 1e-6)[0]
    if len(idx) < 8:                      # 유효 subcarrier가 너무 적으면 위상 신뢰 불가
        return cosr, sinr
    ph = np.arctan2(imag[idx], real[idx]).astype(np.float64)
    u = np.unwrap(ph)
    coef = np.polyfit(idx.astype(np.float64), u, 1)
    resid = u - (coef[0] * idx + coef[1])
    cosr[idx] = np.cos(resid)
    sinr[idx] = np.sin(resid)
    return cosr, sinr


def parse_csi_array_phase(raw_line):
    """parse_csi_array와 동일 파싱이되 (amp, cos, sin)을 반환."""
    start = raw_line.rfind("[")
    end = raw_line.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        arr = np.array(ast.literal_eval(raw_line[start : end + 1]), dtype=np.float32)
    except Exception:
        return None
    if len(arr) < 2:
        return None
    if len(arr) % 2 == 1:
        arr = arr[:-1]
    real = arr[0::2]
    imag = arr[1::2]
    amp = np.sqrt(real * real + imag * imag)
    cosr, sinr = sanitize_phase(real, imag, amp)
    return amp, cosr, sinr


def vel_spectrum(e_norm, win=64, bands=16):
    """움직임 에너지 시계열(균일 시간격, 스케일-프리)에서 변동 스펙트럼 (P, bands).
    낙하의 물리 신호(짧은 버스트 → 침묵)를 입력에 명시 (Widar3.0 BVP/DeFall 계열).
    ★ 반드시 균일 재샘플된 e를 넣을 것 — 불균일 패킷 간격 그대로 FFT하면
      스펙트럼이 뒤섞여 낙하 정렬이 깨짐 (실측: 정렬 1/4 → 균일화 후 5/6)."""
    P = len(e_norm)
    if P < 2:
        return np.zeros((P, bands), dtype=np.float32)
    e = e_norm.astype(np.float32)
    epad = np.concatenate([np.full(win - 1, e[0], dtype=np.float32), e])
    seg = np.lib.stride_tricks.sliding_window_view(epad, win)         # (P, win)
    seg = seg * np.hanning(win).astype(np.float32)
    power = np.abs(np.fft.rfft(seg, axis=1)) ** 2                     # (P, win//2+1)
    power = power[:, 1:]                                              # DC 제거
    n_per = power.shape[1] // bands
    banded = power[:, : n_per * bands].reshape(P, bands, n_per).mean(axis=2)
    return np.log1p(banded).astype(np.float32)


def build_v2_features(times, amp_raw, grid_ms=14.0):
    """raw 진폭 시계열 → 균일 그리드 재샘플 → [unit-norm amp + 통계 4 + 변동 스펙트럼 16].
    반환: (균일 그리드 times[초], features (P,148))
    - 움직임 에너지 e는 raw amp의 균일 diff에서 계산 (unit-norm은 파워 변화를 지워 e를 오염시킴)
    - e를 trial 중앙값으로 나눠 링크 게인과 무관한 상대량으로 만든 뒤 스펙트럼."""
    t = np.asarray(times, dtype=np.float64)
    tu = np.arange(t[0], t[-1], grid_ms / 1000.0)
    A = np.stack([np.interp(tu, t, amp_raw[:, d]) for d in range(amp_raw.shape[1])], axis=1)
    # 움직임 에너지 (raw, 균일격자) → 스케일-프리
    e = np.linalg.norm(np.diff(A, axis=0), axis=1)
    e = np.concatenate([[e[0] if len(e) else 0.0], e])
    e = e / (np.median(e) + 1e-8)
    spec = vel_spectrum(e)
    # 기존 132차원 특징 (unit-norm amp + 통계) — 균일 격자 위에서
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    stats = np.stack([An.mean(axis=1), An.std(axis=1), An.max(axis=1), An.min(axis=1)], axis=1)
    feats = np.concatenate([An, stats, spec], axis=1).astype(np.float32)
    return tu.astype(np.float32), feats


def parse_csi_csv(csi_path):
    """CSV → (times[초], raw amp (P,S)). 캐시 생성용 공통 파서."""
    df = pd.read_csv(csi_path)
    rel_t = (pd.to_numeric(df["pc_time_ms"], errors="coerce") - df["pc_time_ms"].iloc[0]) / 1000.0
    amps, times = [], []
    for _, row in df.iterrows():
        amp = parse_csi_array(row["raw_line"])
        if amp is None:
            continue
        amps.append(np.nan_to_num(amp.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        times.append(float(rel_t.loc[row.name]))
    return np.asarray(times, dtype=np.float64), np.stack(amps)


def parse_csi_csv_iq(csi_path):
    """CSV → (times, raw amp (P,S), 새니타이즈 위상 cos (P,S), sin (P,S)).
    위상 detrend는 패킷 단위 연산이라 여기(원본 패킷)에서 수행 — 시간 보간은 cos/sin에서 안전."""
    df = pd.read_csv(csi_path)
    rel_t = (pd.to_numeric(df["pc_time_ms"], errors="coerce") - df["pc_time_ms"].iloc[0]) / 1000.0
    amps, coss, sins, times = [], [], [], []
    for _, row in df.iterrows():
        parsed = parse_csi_array_phase(row["raw_line"])
        if parsed is None:
            continue
        amp, cosr, sinr = parsed
        amps.append(np.nan_to_num(amp.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        coss.append(cosr)
        sins.append(sinr)
        times.append(float(rel_t.loc[row.name]))
    return (np.asarray(times, dtype=np.float64), np.stack(amps),
            np.stack(coss), np.stack(sins))


def build_v3_features(times, amp_raw, cos_raw, sin_raw, grid_ms=14.0):
    """v3 = v2(148d) + 새니타이즈 위상 cos/sin (128×2 = 256d) = 404d.
    amp 경로는 build_v2_features 그대로 (동일 균일 그리드), 위상은 cos/sin을 시간 보간."""
    tu, feats = build_v2_features(times, amp_raw, grid_ms)
    t = np.asarray(times, dtype=np.float64)
    C = np.stack([np.interp(tu, t, cos_raw[:, d]) for d in range(cos_raw.shape[1])], axis=1)
    Sn = np.stack([np.interp(tu, t, sin_raw[:, d]) for d in range(sin_raw.shape[1])], axis=1)
    return tu, np.concatenate([feats, C, Sn], axis=1).astype(np.float32)


def load_csi_features(csi_path, vel_spec=False, phase=False):
    csi_path = Path(csi_path)
    cache_path = csi_path.with_suffix(".npz")

    if phase:
        # v3 캐시: v2(148d) + 새니타이즈 위상 cos/sin(256d) = 404d. 기존 캐시와 공존.
        v3_path = csi_path.with_name(csi_path.stem + "_v3.npz")
        if v3_path.exists():
            data = np.load(v3_path)
            return data["times"], data["features"]
        raw_t, raw_amp, raw_cos, raw_sin = parse_csi_csv_iq(csi_path)
        times3, feats3 = build_v3_features(raw_t, raw_amp, raw_cos, raw_sin)
        np.savez(v3_path, times=times3, features=feats3)
        return times3, feats3

    if vel_spec:
        # v2 캐시: 균일 재샘플 + [amp 128 + 통계 4 + 변동 스펙트럼 16] = 148차원. 기존 캐시와 공존.
        v2_path = csi_path.with_name(csi_path.stem + "_v2.npz")
        if v2_path.exists():
            data = np.load(v2_path)
            return data["times"], data["features"]
        raw_t, raw_amp = parse_csi_csv(csi_path)
        times2, feats2 = build_v2_features(raw_t, raw_amp)
        np.savez(v2_path, times=times2, features=feats2)
        return times2, feats2

    if cache_path.exists():
        data = np.load(cache_path)
        return data["times"], data["features"]

    df = pd.read_csv(csi_path)
    rel_t = (pd.to_numeric(df["pc_time_ms"], errors="coerce") - df["pc_time_ms"].iloc[0]) / 1000.0
    features = []
    times = []
    for _, row in df.iterrows():
        amp = parse_csi_array(row["raw_line"])
        if amp is None:
            continue
        amp = amp.astype(np.float32)
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        # per-packet 스케일 정규화: AGC/Tx power/거리 게인(전체 배율)을 제거하고
        # subcarrier 프로파일의 "모양"(방향)만 남긴다. → cross-domain 전이용.
        amp = amp / (np.linalg.norm(amp) + 1e-6)
        feature = np.concatenate(
            [
                amp,
                np.array(
                    [
                        np.nanmean(amp),
                        np.nanstd(amp),
                        np.nanmax(amp),
                        np.nanmin(amp),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        features.append(feature)
        times.append(float(rel_t.loc[row.name]))

    times_arr    = np.asarray(times,    dtype=np.float32)
    features_arr = np.asarray(features, dtype=np.float32)
    np.savez(cache_path, times=times_arr, features=features_arr)
    return times_arr, features_arr


def pose_target_columns():
    return [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]


# proxy13 관절 ↔ full33 landmark 매핑 (extract_pose_features.build_proxy와 동일)
VIS_SOURCES = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    **{j: [j] for j in JOINTS if j != "head"},
}


def load_visibility(full33_path):
    """full33 CSV에서 proxy13 13관절의 visibility (T,13)를 읽는다."""
    cols = sorted({f"{lm}_visibility" for lms in VIS_SOURCES.values() for lm in lms})
    df = pd.read_csv(full33_path, usecols=cols)
    vis = np.stack(
        [df[[f"{lm}_visibility" for lm in VIS_SOURCES[j]]].mean(axis=1).to_numpy(dtype=np.float32)
         for j in JOINTS],
        axis=1,
    )
    return np.nan_to_num(vis, nan=0.0)


def attach_vis_weights(seq, full33_path, predict_root=False, thr=0.5, soft_root=False):
    """visibility<thr인 관절-프레임을 loss에서 제외하는 가중치 행렬 seq['w']를 만든다.
    MediaPipe가 가려진 관절에 대해 '추측'으로 출력한 좌표(오답 정답지)를 채점에서 빼는 것.

    root(+3차원):
      soft_root=False (v1): 양쪽 hip이 모두 보일 때만 채점 (이진)
                            → 낙상/옆눕기(한쪽 hip 가려짐)에서 root 감독이 소실되는 문제
      soft_root=True  (v2): 양쪽 hip visibility의 평균을 연속 가중치로 사용
                            → 낙상 프레임의 root 감독을 신뢰도만큼 유지.
                            (골반 중점은 몸통 문맥 제약이 강해 가려져도 GT가 크게 안 틀림)"""
    T_frames, dim = seq["y"].shape
    w = np.ones((T_frames, dim), dtype=np.float32)
    full33_path = Path(full33_path)
    if not full33_path.exists():
        seq["w"] = w
        return seq
    vis = load_visibility(full33_path)
    n = min(T_frames, len(vis))
    mask13 = (vis[:n] >= thr).astype(np.float32)              # (n,13)
    w[:n, : len(JOINTS) * 3] = np.repeat(mask13, 3, axis=1)   # 관절 마스크 → x,y,z 3열 복제
    if predict_root and dim > len(JOINTS) * 3:
        lh, rh = JOINTS.index("left_hip"), JOINTS.index("right_hip")
        if soft_root:
            root_w = ((vis[:n, lh] + vis[:n, rh]) / 2.0)[:, None]   # 연속 가중치 (v2)
        else:
            root_w = (mask13[:, lh] * mask13[:, rh])[:, None]       # 이진 (v1)
        w[:n, len(JOINTS) * 3:] = root_w
    seq["w"] = w
    return seq


def load_pose_targets(pose_path):
    df = pd.read_csv(pose_path)
    cols = pose_target_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing pose columns in {pose_path}: {missing[:5]}")
    time = pd.to_numeric(df["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float32)
    target = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    derived = df[[c for c in DERIVED_COLS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return time, target, derived, df


def resample_csi_to_pose(csi_t, csi_x, pose_t):
    if len(csi_t) == 0:
        raise ValueError("No CSI features")
    out = []
    for dim in range(csi_x.shape[1]):
        out.append(np.interp(pose_t, csi_t, csi_x[:, dim]))
    return np.stack(out, axis=1).astype(np.float32)


def to_root_relative(y):
    """
    절대 좌표 골격을 root(골반=left_hip·right_hip 중점) 기준 상대 좌표로 변환.
    골격 전체의 평행이동(카메라 위치/피험자 위치 차이로 생기는 상수 오프셋)을 제거한다.
    반환: (y_rel, root)  root는 별도 저장 → 필요 시 절대 높이 복원용.
    y: (T, 39)  = 13 joints x (x,y,z)
    """
    T = y.shape[0]
    yj = y.reshape(T, len(JOINTS), 3)
    lh = JOINTS.index("left_hip")
    rh = JOINTS.index("right_hip")
    root = (yj[:, lh, :] + yj[:, rh, :]) / 2.0            # (T, 3)
    yj_rel = yj - root[:, None, :]
    return yj_rel.reshape(T, len(JOINTS) * 3).astype(np.float32), root.astype(np.float32)


def apply_torso_norm(y_rel):
    """
    root-relative 골격을 torso 길이(어깨중심~골반)로 나눠 몸 "크기"를 통일한다.
    사람마다 다른 신체 크기(팔다리 길이=CSI로 예측 불가한 양)를 타깃에서 제거 →
    모델은 크기 대신 "비율(모양)"만 학습. root-relative가 위치를 없앤 것과 같은 논리.

    torso 길이는 프레임별이 아니라 시퀀스 중앙값(스칼라)을 쓴다:
      몸 크기는 일정한데 프레임별 torso는 자세(눕기 등 단축)로 흔들려서,
      프레임별로 나누면 자세 자체가 왜곡되기 때문.
    반환: (y_norm, torso_scale)  torso_scale은 절대 크기 복원용.
    """
    T = y_rel.shape[0]
    yj = y_rel.reshape(T, len(JOINTS), 3)
    ls = JOINTS.index("left_shoulder")
    rs = JOINTS.index("right_shoulder")
    # root-relative에서 골반중심=0 이므로 torso 벡터 = 어깨중심 - 0 = 어깨중심
    shoulder_center = (yj[:, ls, :] + yj[:, rs, :]) / 2.0     # (T, 3)
    torso = np.linalg.norm(shoulder_center, axis=1)           # (T,)
    scale = float(np.median(torso))
    scale = max(scale, 1e-3)
    return (y_rel / scale).astype(np.float32), scale


# ── 자세 상태 (upright / sitting / lying) 라벨링 ─────────────────────────────
# 고정 임계값 대신 "그 subject 자신의 정적 클래스 GT 분포"를 기준으로 최근접 라벨링.
# → 카메라 각도·체형이 바뀌어도 기준이 함께 이동 (매직 넘버 없음).

STATE_NAMES = ["upright", "sitting", "lying"]
STATE_REF_LABELS = {  # 기준 분포를 만들 정적 클래스 (전부 safe → 배포 캘리브레이션에서도 확보 가능)
    "upright": ["standing_still", "walking"],
    "sitting": ["sitting_still"],
    "lying": ["lying_still"],
}
EXPECTED_STATE = {  # 검증 게이트용: 정적 클래스라면 이 상태여야 한다
    "standing_still": "upright", "walking": "upright", "unstable_walking": "upright",
    "sitting_still": "sitting",
    "lying_still": "lying", "post_fall_inactive": "lying",
}


def posture_feats(y_pose):
    """root-relative pose (T,>=39) → 프레임별 (θ 몸통수평각[deg], κ/ℓ 무릎상대높이). 스케일-프리."""
    yj = y_pose[:, : len(JOINTS) * 3].reshape(len(y_pose), len(JOINTS), 3)
    ls, rs = JOINTS.index("left_shoulder"), JOINTS.index("right_shoulder")
    lk, rk = JOINTS.index("left_knee"), JOINTS.index("right_knee")
    sc = (yj[:, ls] + yj[:, rs]) / 2.0
    ell = np.linalg.norm(sc, axis=1) + 1e-8
    theta = np.degrees(np.arccos(np.clip(np.abs(sc[:, 1]) / ell, 0, 1)))
    kappa = ((yj[:, lk, 1] + yj[:, rk, 1]) / 2.0) / ell
    return np.stack([theta, kappa], axis=1).astype(np.float32)


def build_state_refs(subject):
    """subject의 train split 정적 trial GT로 상태별 (평균, 표준편차) 기준 분포를 만든다."""
    refs = {}
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["subject"] == subject and r["valid"] == "True"
                and r["proxy13_exists"] == "True" and r["split_mode1"] == "train"]
    for state, labels in STATE_REF_LABELS.items():
        feats = []
        for r in rows:
            if r["label"] not in labels:
                continue
            _, y, _, _ = load_pose_targets(ROOT / r["proxy13_path"])
            y_rel, _ = to_root_relative(np.nan_to_num(y))
            feats.append(posture_feats(y_rel))
        if not feats:
            raise RuntimeError(f"state 기준 분포 생성 실패: {subject}/{state} — 정적 trial 없음")
        F = np.concatenate(feats)
        refs[state] = (F.mean(axis=0), F.std(axis=0) + 1e-3)
    return refs


def attach_state_labels(seq, refs, ambig_ratio=0.7):
    """프레임별 상태 라벨: 그 subject의 기준 분포 3개 중 최근접(정규화 거리).
    1등/2등 거리가 비슷하면(ratio>ambig_ratio) 모호 → CE 가중 0 (이행 중간을 못 박지 않음)."""
    F = posture_feats(seq["y"])                                 # (T,2)
    dists = np.stack(
        [np.linalg.norm((F - mu) / sd, axis=1) for s in STATE_NAMES for mu, sd in [refs[s]]],
        axis=1,
    )                                                           # (T,3)
    order = np.sort(dists, axis=1)
    seq["state"] = np.argmin(dists, axis=1).astype(np.int64)
    seq["state_w"] = (order[:, 0] / (order[:, 1] + 1e-8) <= ambig_ratio).astype(np.float32)
    return seq


def validate_state_labels(seqs, tag=""):
    """검증 게이트: 정적 클래스 trial의 라벨이 기대 상태와 ≥95% 일치하는지."""
    from collections import defaultdict as dd
    agg = dd(lambda: [0, 0])
    for seq in seqs:
        exp = EXPECTED_STATE.get(seq.get("label"))
        if exp is None:
            continue
        m = seq["state_w"] > 0
        agg[seq["label"]][0] += int((seq["state"][m] == STATE_NAMES.index(exp)).sum())
        agg[seq["label"]][1] += int(m.sum())
    warn = False
    for lab, (ok, n) in sorted(agg.items()):
        rate = ok / max(n, 1)
        flag = "" if rate >= 0.95 else "  ⚠ 95% 미달 — 데이터 확인 필요"
        if rate < 0.95:
            warn = True
        print(f"  [state gate{tag}] {lab:22s} {100*rate:5.1f}% ({n}프레임){flag}")
    return not warn


def attach_motion_weights(seq, alpha, cap=10.0):
    """움직임이 큰 프레임에 loss 가중치를 곱한다.
    낙하/이행 프레임은 전체의 ~1-5%라 균등 평균에서 묻히는 문제(학습 신호 희석)를 해소.
      w(t) = 1 + alpha * min(|Δy(t)| / mean|Δy|, cap)
    낙하 프레임은 움직임이 평균의 ~5-10배 → 가중 ~6-11배 → loss 기여가 의미 있게 상승.
    vis_mask 가중치(seq['w'])가 있으면 곱해서 결합한다."""
    y = seq["y"]
    m = np.linalg.norm(np.diff(y, axis=0), axis=1)
    m = np.concatenate([[m[0] if len(m) else 0.0], m])          # 첫 프레임 보정
    ratio = np.clip(m / (m.mean() + 1e-8), 0.0, cap)
    wf = (1.0 + alpha * ratio).astype(np.float32)[:, None]      # (T,1) → 프레임 단위 가중
    if "w" in seq:
        seq["w"] = seq["w"] * wf
    else:
        seq["w"] = np.ones_like(y, dtype=np.float32) * wf
    return seq


def load_sequence(csi_path, pose_path, trial, vel_spec=False, phase=False):
    csi_path = Path(csi_path)
    pose_path = Path(pose_path)
    if not csi_path.exists():
        raise FileNotFoundError(csi_path)
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    csi_t, csi_x = load_csi_features(csi_path, vel_spec=vel_spec, phase=phase)
    pose_t, y, derived, pose_df = load_pose_targets(pose_path)
    x = resample_csi_to_pose(csi_t, csi_x, pose_t)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "trial": trial,
        "x": x,
        "y": y,
        "time": pose_t,
        "derived": derived,
        "pose_df": pose_df,
        "csi_path": csi_path,
        "pose_path": pose_path,
    }


def load_manifest_rows(label, subject=None, test_subject=None, train_subject=None):
    """
    Mode1:   subject="mhw"
      → mhw의 split_mode1 기준으로 train/val/test

    Cross:   train_subject="lmh", test_subject="mhw"
      → lmh만 train/val (split_mode1 기준), mhw 전체 test

    LOSO:    test_subject="ajh"  (train_subject 없음)
      → ajh 제외 전원 train/val, ajh 전체 test
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST_PATH}\n  → scripts/build_manifest.py 를 먼저 실행하세요.")

    train_rows, val_rows, test_rows = [], [], []

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if label != "ALL" and row["label"] != label:
                continue
            if row["valid"] != "True":
                continue
            if row["csi_exists"] != "True" or row["proxy13_exists"] != "True":
                continue

            subj = row["subject"]

            if train_subject and test_subject:
                # Cross 모드: 특정 1명 train, 특정 1명 test (test split만)
                if subj == test_subject and row["split_mode1"] == "test":
                    test_rows.append(row)
                elif subj == train_subject:
                    if row["split_mode1"] == "train":
                        train_rows.append(row)
                    elif row["split_mode1"] == "val":
                        val_rows.append(row)
            elif test_subject:
                # LOSO 모드: test_subject 제외 전원 train
                if subj == test_subject:
                    test_rows.append(row)
                else:
                    if row["split_mode1"] == "train":
                        train_rows.append(row)
                    elif row["split_mode1"] == "val":
                        val_rows.append(row)
            else:
                # Mode1: 단일 subject
                if subj != subject:
                    continue
                if row["split_mode1"] == "train":
                    train_rows.append(row)
                elif row["split_mode1"] == "val":
                    val_rows.append(row)
                else:
                    test_rows.append(row)

    return train_rows, val_rows, test_rows


class WindowDataset(Dataset):
    def __init__(self, sequences, win=32, step=4, x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.items = []
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        for seq_idx, seq in enumerate(sequences):
            n = len(seq["x"])
            for start in range(0, max(1, n - win + 1), step):
                end = start + win
                if end <= n:
                    self.items.append((seq_idx, start, end))
        self.sequences = sequences

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq_idx, start, end = self.items[idx]
        seq = self.sequences[seq_idx]
        x = seq["x"][start:end]
        y = seq["y"][start:end]
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        if "w" in seq:
            w = seq["w"][start:end]
        else:
            w = np.ones_like(y, dtype=np.float32)
        if "state" in seq:
            st = seq["state"][start:end]
            sw = seq["state_w"][start:end]
        else:
            st = np.zeros(end - start, dtype=np.int64)
            sw = np.zeros(end - start, dtype=np.float32)
        return (torch.from_numpy(x.T).float(), torch.from_numpy(y).float(),
                torch.from_numpy(w).float(), torch.from_numpy(st).long(), torch.from_numpy(sw).float())


class TinyTCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=96, dilations=(1, 2, 4), state_classes=0):
        """수용영역 = 1 + (k-1)·Σdilations.  (1,2,4)→29프레임(~1초), (1,2,4,8,16)→125프레임(~4초).
        낙하 추론('버스트 후 정적 = 쓰러짐')은 버스트와 현재를 한 시야에 담아야 가능."""
        super().__init__()
        k = 5
        layers = []
        ch = in_dim
        for d in dilations:
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=k, padding=(k - 1) // 2 * d, dilation=d),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            ]
            ch = hidden
        layers.append(nn.Conv1d(hidden, out_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)
        # 상태 분류 헤드 (upright/sitting/lying): 몸통 특징을 공유, pose 헤드와 병렬.
        # CE loss가 "정적 상태 혼동"에 직접 벌점 → 몸통이 상태별 특징을 분리하도록 강제
        # → pose 헤드의 '전부 서있음으로 붕괴' 문제를 표현 수준에서 해소.
        self.state_head = nn.Conv1d(hidden, state_classes, kernel_size=1) if state_classes > 0 else None

    def forward(self, x, return_state=False):
        # x: batch, feature, time
        h = x
        for layer in list(self.net)[:-1]:      # 몸통 (마지막 1x1 conv = pose 헤드 제외)
            h = layer(h)
        pose = self.net[-1](h).transpose(1, 2)
        if return_state and self.state_head is not None:
            state = self.state_head(h).transpose(1, 2)   # (B, T, 3) logits
            return pose, state
        return pose


class MambaBlock(nn.Module):
    """Selective SSM (Mamba) 블록 — 순수 PyTorch 구현 (mamba-minimal 계열).

    공식 mamba-ssm은 CUDA 커널 필수 + Windows 빌드 불가 → 우리 규모(hidden 96,
    window 32)에서는 시간축 for-loop 스캔으로 충분히 실용적.
    구성은 C-MambaPose와 동일: d_state=16, d_conv=4, expand=2.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)   # causal depthwise
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        T = x.shape[1]
        xi, z = self.in_proj(x).chunk(2, dim=-1)                     # (B,T,din) ×2
        xc = self.conv1d(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xc = F.silu(xc)
        y = self._ssm(xc)
        return self.out_proj(y * F.silu(z))

    def _ssm(self, x):
        # selective scan: h[t] = exp(Δt·A)·h[t-1] + Δt·B[t]·x[t],  y[t] = C[t]·h[t]
        A = -torch.exp(self.A_log)                                   # (din, N)
        d_in, N = A.shape
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                            # (B,T,din)
        dA = torch.exp(dt.unsqueeze(-1) * A)                         # (B,T,din,N)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)   # (B,T,din,N)
        h = x.new_zeros(x.shape[0], d_in, N)
        ys = []
        for t in range(x.shape[1]):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
        return torch.stack(ys, dim=1) + x * self.D


class TinyMamba(nn.Module):
    """TinyTCN의 몸통(dilated conv)만 Mamba로 교체한 대응 모델. 입출력 규약 동일.

    TCN은 대칭 padding으로 비인과(양방향 시야) → 공정 비교 위해 Mamba도 양방향:
    각 층에서 정방향 블록 + 역방향 블록(시간 뒤집기)의 합을 residual로 더한다.
    (배포 스트리밍 시엔 정방향만 쓰는 인과 모드로 전환 가능 — 이벤트 브랜치용)
    """

    def __init__(self, in_dim, out_dim, hidden=96, n_blocks=2, d_state=16,
                 bidir=True, state_classes=0):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])
        self.fwd = nn.ModuleList([MambaBlock(hidden, d_state) for _ in range(n_blocks)])
        self.bwd = nn.ModuleList([MambaBlock(hidden, d_state) for _ in range(n_blocks)]) if bidir else None
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.state_head = nn.Linear(hidden, state_classes) if state_classes > 0 else None

    def forward(self, x, return_state=False):
        # x: (B, feature, T)  — TinyTCN과 동일 입력 규약
        h = self.inp(x.transpose(1, 2))                              # (B,T,D)
        for i, blk in enumerate(self.fwd):
            hn = self.norms[i](h)
            upd = blk(hn)
            if self.bwd is not None:
                upd = upd + torch.flip(self.bwd[i](torch.flip(hn, [1])), [1])
            h = h + upd
        h = self.out_norm(h)
        pose = self.head(h)                                          # (B,T,out)
        if return_state and self.state_head is not None:
            return pose, self.state_head(h)
        return pose


def build_model(arch, in_dim, out_dim, hidden=96, dilations=(1, 2, 4), state_classes=0):
    """arch 문자열로 TinyTCN/TinyMamba 생성 (train·few_shot_calib 공용)."""
    if arch == "mamba":
        return TinyMamba(in_dim, out_dim, hidden=hidden, state_classes=state_classes)
    return TinyTCN(in_dim, out_dim, hidden=hidden, dilations=dilations, state_classes=state_classes)


def make_bone_loss(y_mean, y_std, device):
    """뼈 길이 일관성 loss (C-MambaPose의 L_bone 이식).

    예측·GT 각각에서 SKELETON_EDGES 14개 뼈의 길이를 재고, 그 차이의 제곱에 벌점을 준다.
    → 관절별 loss가 놓치는 "팔·다리가 늘거나 쪼그라드는" 해부학적 붕괴를 직접 억제.

    주의:
    - y는 축별(mean/std) 정규화라 축마다 스케일이 달라 정규화 공간의 '길이'는 왜곡됨
      → 실좌표(미터)로 역정규화한 뒤 계산한다.
    - predict_root(42d)여도 pose 앞 39차원만 사용.
    - vis_mask 가중치(wb)는 양 끝 관절 가중치의 곱으로 뼈 단위 반영
      (한쪽이라도 가려짐 추측 GT면 그 뼈는 감독하지 않음).
    """
    n_pose = len(JOINTS) * 3
    mean_t = torch.from_numpy(np.asarray(y_mean, dtype=np.float32)[..., :n_pose]).to(device)
    std_t = torch.from_numpy(np.asarray(y_std, dtype=np.float32)[..., :n_pose]).to(device)
    idx_a = torch.tensor([JOINTS.index(a) for a, _ in SKELETON_EDGES], device=device)
    idx_b = torch.tensor([JOINTS.index(b) for _, b in SKELETON_EDGES], device=device)

    def bone_loss(pred, yb, wb):
        B, T = pred.shape[0], pred.shape[1]
        pj = (pred[..., :n_pose] * std_t + mean_t).reshape(B, T, len(JOINTS), 3)
        yj = (yb[..., :n_pose] * std_t + mean_t).reshape(B, T, len(JOINTS), 3)
        len_pred = torch.linalg.norm(pj[:, :, idx_a] - pj[:, :, idx_b], dim=-1)   # (B, T, 14)
        len_gt = torch.linalg.norm(yj[:, :, idx_a] - yj[:, :, idx_b], dim=-1)
        wj = wb[..., 0:n_pose:3]                       # 관절별 가중 (x열 대표, 3열 동일값)
        bw = wj[:, :, idx_a] * wj[:, :, idx_b]         # 뼈 가중 = 양끝 관절 모두 유효
        return ((len_pred - len_gt) ** 2 * bw).sum() / bw.sum().clamp(min=1.0)

    return bone_loss


def compute_norm(sequences):
    x = np.concatenate([s["x"] for s in sequences], axis=0)
    y = np.concatenate([s["y"] for s in sequences], axis=0)
    x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (y.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return x_mean, x_std, y_mean, y_std


def train_model(train_seqs, val_seqs, args, out_dir):
    x_mean, x_std, y_mean, y_std = compute_norm(train_seqs)
    train_ds = WindowDataset(train_seqs, args.window, args.step, x_mean, x_std, y_mean, y_std)
    val_ds = WindowDataset(val_seqs, args.window, args.step, x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_on = bool(getattr(args, "state_head", False))
    model = build_model(getattr(args, "arch", "tcn"),
                        train_seqs[0]["x"].shape[1], train_seqs[0]["y"].shape[1], args.hidden,
                        dilations=tuple(args.dilations), state_classes=3 if state_on else 0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={type(model).__name__}  params={n_params:,}")
    ce_weight = None
    if state_on:
        # 클래스 역빈도 가중: upright 프레임 다수 → lying이 CE 안에서 묻히지 않게
        cnt = np.zeros(3)
        for sq in train_seqs:
            m = sq["state_w"] > 0
            for c in range(3):
                cnt[c] += int((sq["state"][m] == c).sum())
        ce_weight = torch.tensor(cnt.sum() / (3.0 * np.maximum(cnt, 1)), dtype=torch.float32).to(device)
        print(f"state 프레임 분포 upright/sitting/lying = {cnt.astype(int)}  CE가중 = {np.round(ce_weight.cpu().numpy(),2)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    def masked_loss(pred, yb, wb):
        l = loss_fn(pred, yb) * wb
        return l.sum() / wb.sum().clamp(min=1.0)

    ce_fn = nn.CrossEntropyLoss(weight=ce_weight, reduction="none") if state_on else None

    def state_loss(slog, sb, swb):
        l = ce_fn(slog.reshape(-1, 3), sb.reshape(-1)) * swb.reshape(-1)
        return l.sum() / swb.sum().clamp(min=1.0)

    bone_lambda = float(getattr(args, "bone_lambda", 0.0))
    bone_fn = make_bone_loss(y_mean, y_std, device) if bone_lambda > 0 else None

    history = []
    best_val = math.inf
    best_path = out_dir / "best_model.pt"
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, wb, sb, swb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            if state_on:
                pred, slog = model(xb, return_state=True)
                loss = masked_loss(pred, yb, wb) + args.state_lambda * state_loss(slog, sb.to(device), swb.to(device))
            else:
                pred = model(xb)
                loss = masked_loss(pred, yb, wb)
            if bone_fn is not None:
                loss = loss + bone_lambda * bone_fn(pred, yb, wb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, wb, sb, swb in val_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                if state_on:
                    pred, slog = model(xb, return_state=True)
                    vl = masked_loss(pred, yb, wb) + args.state_lambda * state_loss(slog, sb.to(device), swb.to(device))
                else:
                    pred = model(xb)
                    vl = masked_loss(pred, yb, wb)
                if bone_fn is not None:
                    vl = vl + bone_lambda * bone_fn(pred, yb, wb)
                val_losses.append(float(vl.item()))
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "joints": JOINTS,
                    "predict_root": getattr(args, "predict_root", False),
                    "vis_mask": getattr(args, "vis_mask", False),
                    "vis_mask_v2": getattr(args, "vis_mask_v2", False),
                    "motion_weight": getattr(args, "motion_weight", 0.0),
                    "vel_spec": getattr(args, "vel_spec", False),
                    "phase": getattr(args, "phase", False),
                    "dilations": tuple(getattr(args, "dilations", (1, 2, 4))),
                    "window": getattr(args, "window", 32),
                    "state_head": getattr(args, "state_head", False),
                    "state_lambda": getattr(args, "state_lambda", 0.3),
                    "bone_lambda": bone_lambda,
                    "arch": getattr(args, "arch", "tcn"),
                    "hidden": getattr(args, "hidden", 96),
                },
                best_path,
            )
        else:
            patience_counter += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} | train {train_loss:.5f} | val {val_loss:.5f} | patience {patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            print(f"early stop at epoch {epoch}  (best val {best_val:.5f})")
            break

    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"]).to_csv(
        out_dir / "training_history.csv", index=False
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint, device


def predict_sequence(model, checkpoint, device, seq, return_state=False):
    x = (seq["x"] - checkpoint["x_mean"]) / checkpoint["x_std"]
    xb = torch.from_numpy(x.T[None]).float().to(device)
    model.eval()
    with torch.no_grad():
        if return_state and getattr(model, "state_head", None) is not None:
            pred, slog = model(xb, return_state=True)
            pred = pred.cpu().numpy()[0]
            state = slog.argmax(dim=2).cpu().numpy()[0]
            return (pred * checkpoint["y_std"] + checkpoint["y_mean"]).astype(np.float32), state
        pred = model(xb).cpu().numpy()[0]
    pred = pred * checkpoint["y_std"] + checkpoint["y_mean"]
    return pred.astype(np.float32)


def save_prediction_csv(seq, pred, out_path):
    cols = pose_target_columns()
    if pred.shape[1] > len(cols):          # predict_root: root 절대위치 3차원 추가
        cols = cols + ["root_x", "root_y", "root_z"]
    df = pd.DataFrame(pred, columns=[f"pred_{c}" for c in cols])
    df.insert(0, "timestamp_sec", seq["time"])
    df.insert(0, "trial", seq["trial"])
    gt = pd.DataFrame(seq["y"], columns=[f"gt_{c}" for c in cols])
    out = pd.concat([df, gt], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def plot_summary(seq, pred, out_path):
    gt = seq["y"]
    t = seq["time"]
    cols = pose_target_columns()
    gt_df = pd.DataFrame(gt, columns=cols)
    pr_df = pd.DataFrame(pred, columns=cols)
    metrics = {
        "head_y": ("head_y", "Head Y"),
        "hip_y": ("left_hip_y", "Left Hip Y"),
        "left_knee_y": ("left_knee_y", "Left Knee Y"),
        "right_knee_y": ("right_knee_y", "Right Knee Y"),
    }
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    for ax, (_, (col, title)) in zip(axes, metrics.items()):
        ax.plot(t, gt_df[col], label="GT", linewidth=1.6)
        ax.plot(t, pr_df[col], label="Pred", linewidth=1.2, alpha=0.85)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (sec)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_animation(seq, pred, out_path, max_frames=180):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.animation as animation

    gt = seq["y"]
    n = len(gt)
    idxs = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
    joint_index = {j: i for i, j in enumerate(JOINTS)}

    all_xyz = np.concatenate([gt.reshape(n, len(JOINTS), 3), pred.reshape(n, len(JOINTS), 3)], axis=0)
    mins = np.nanmin(all_xyz, axis=(0, 1))
    maxs = np.nanmax(all_xyz, axis=(0, 1))
    center = (mins + maxs) / 2
    span = max(float(np.max(maxs - mins)), 1e-3)

    fig = plt.figure(figsize=(10, 5))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pr = fig.add_subplot(1, 2, 2, projection="3d")

    def draw_skeleton(ax, xyz, title):
        ax.clear()
        ax.set_title(title)
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for a, b in SKELETON_EDGES:
            ia, ib = joint_index[a], joint_index[b]
            ax.plot(
                [xyz[ia, 0], xyz[ib, 0]],
                [xyz[ia, 1], xyz[ib, 1]],
                [xyz[ia, 2], xyz[ib, 2]],
                marker="o",
                linewidth=2,
            )

    def update(frame_no):
        idx = idxs[frame_no]
        draw_skeleton(ax_gt, gt[idx].reshape(len(JOINTS), 3), f"GT {seq['trial']} {seq['time'][idx]:.1f}s")
        draw_skeleton(ax_pr, pred[idx].reshape(len(JOINTS), 3), f"CSI Pred {seq['trial']} {seq['time'][idx]:.1f}s")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(idxs), interval=80, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(out_path, writer="ffmpeg", fps=12)
    except Exception:
        fallback = out_path.with_suffix(".gif")
        ani.save(fallback, writer="pillow", fps=12)
        out_path = fallback
    plt.close(fig)
    return out_path


def evaluate(gt, pred):
    """pose 39차원과 (있다면) root 3차원을 분리 채점.
    mae_all은 pose 39차원만 → 기존 실험 수치와 직접 비교 가능."""
    n_pose = len(JOINTS) * 3
    pose_gt, pose_pr = gt[:, :n_pose], pred[:, :n_pose]
    mae = np.nanmean(np.abs(pose_gt - pose_pr))
    per_joint = {}
    gt_j = pose_gt.reshape(len(gt), len(JOINTS), 3)
    pr_j = pose_pr.reshape(len(pred), len(JOINTS), 3)
    for idx, joint in enumerate(JOINTS):
        per_joint[joint] = float(np.nanmean(np.abs(gt_j[:, idx, :] - pr_j[:, idx, :])))
    extra = {}
    if gt.shape[1] > n_pose:
        extra["mae_root"] = float(np.nanmean(np.abs(gt[:, n_pose:] - pred[:, n_pose:])))
        extra["mae_root_y"] = float(np.nanmean(np.abs(gt[:, n_pose + 1] - pred[:, n_pose + 1])))
    return float(mae), per_joint, extra


def main():
    parser = argparse.ArgumentParser(description="Train CSI-to-3D-skeleton-proxy pilot model.")
    parser.add_argument("--label", default="unstable_walking")
    parser.add_argument("--subject", default="mhw",
                        help="Mode1용 단일 subject (--test_subject 없을 때)")
    parser.add_argument("--test_subject", default=None,
                        help="test subject. --train_subject 없으면 나머지 전원 train (LOSO)")
    parser.add_argument("--train_subject", default=None,
                        help="Cross 모드: 이 subject만 train/val, --test_subject가 test")
    parser.add_argument("--root_relative", action="store_true",
                        help="타깃을 root(골반) 기준 상대좌표로 학습 (골격 전체 위치 오프셋 제거)")
    parser.add_argument("--torso_norm", action="store_true",
                        help="root-relative 위에 torso 길이로 크기 정규화 (신체 크기 차이 제거). root_relative 자동 활성")
    parser.add_argument("--predict_root", action="store_true",
                        help="상대좌표 39 + root 절대위치 3 = 42차원 출력 (낙상 감지용 높이 복원). root_relative 자동 활성")
    parser.add_argument("--vis_mask", action="store_true",
                        help="MediaPipe visibility<0.5인 관절-프레임을 loss에서 제외 (가려짐 추측 GT 배제)")
    parser.add_argument("--vis_mask_v2", action="store_true",
                        help="vis_mask + root는 이진 제외 대신 연속 가중치(양 hip visibility 평균) — 낙상 root 감독 유지")
    parser.add_argument("--motion_weight", type=float, default=0.0,
                        help="움직임 큰 프레임 loss 가중 α (0=끔). 낙하/이행 순간이 loss에 묻히지 않게. 권장 1.0")
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4],
                        help="TCN dilation 목록. 1 2 4 8 16 → 수용영역 ~4초 (낙하 문맥)")
    parser.add_argument("--state_head", action="store_true",
                        help="자세상태(upright/sitting/lying) 분류 헤드 추가. 정적 상태 붕괴 방지 + 낙상 감지용")
    parser.add_argument("--state_lambda", type=float, default=0.3,
                        help="상태 CE loss 가중 λ")
    parser.add_argument("--arch", choices=["tcn", "mamba"], default="tcn",
                        help="시간 인코더: tcn(기본) 또는 mamba (양방향 selective SSM, C-MambaPose 구성). "
                             "Mamba vs TCN ablation용 — 나머지 조건 동일하게 몸통만 교체")
    parser.add_argument("--phase", action="store_true",
                        help="새니타이즈 위상 cos/sin 채널 추가 (v3 캐시 404d = v2 148 + 위상 256). "
                             "정적 자세의 반사 기하 정보 가설 검증용 (C-MambaPose 복소 표현 이식)")
    parser.add_argument("--bone_lambda", type=float, default=0.0,
                        help="뼈 길이 일관성 loss 가중 λ (0=끔, C-MambaPose 준용 권장 0.1). "
                             "실좌표(미터) 공간에서 계산 — pose loss(정규화 공간)와 스케일이 다르므로 "
                             "train 로그의 loss 절대값은 기존 실험과 직접 비교 불가")
    parser.add_argument("--vel_spec", action="store_true",
                        help="변동 스펙트럼 16차원 추가 (132→148). 낙하 버스트를 입력에 명시. _v2 캐시 사용")
    parser.add_argument("--manifest", default=None,
                        help="사용할 manifest 파일명 (기본: manifest_full.csv). "
                             "world 좌표 학습: manifest_full_world.csv")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.manifest:
        global MANIFEST_PATH
        mpath = Path(args.manifest)
        MANIFEST_PATH = mpath if mpath.is_absolute() else ROOT / "split_manifest" / args.manifest
        print(f"[manifest] {MANIFEST_PATH}")

    set_seed(args.seed)

    # torso_norm / predict_root는 root-relative 위에서만 의미가 있으므로 자동 활성화
    if args.torso_norm or args.predict_root:
        args.root_relative = True
    if args.vis_mask_v2:
        args.vis_mask = True

    cross = args.train_subject is not None and args.test_subject is not None
    loso  = args.test_subject is not None and not cross
    rr_suffix = ""
    if args.root_relative:
        rr_suffix += "_rootrel"
    if args.torso_norm:
        rr_suffix += "_torso"
    if args.predict_root:
        rr_suffix += "_rooty"
    if args.vis_mask_v2:
        rr_suffix += "_vismask2"
    elif args.vis_mask:
        rr_suffix += "_vismask"
    if args.motion_weight > 0:
        rr_suffix += f"_mw{args.motion_weight:g}"
    if args.vel_spec:
        rr_suffix += "_vel"
    if args.phase:
        rr_suffix += "_ph"
    if tuple(args.dilations) != (1, 2, 4):
        rr_suffix += f"_rf{max(args.dilations)}"
    if args.state_head:
        rr_suffix += "_st"
    if args.bone_lambda > 0:
        rr_suffix += f"_bone{args.bone_lambda:g}"
    if args.arch != "tcn":
        rr_suffix += f"_{args.arch}"
    if args.manifest and "world" in args.manifest:
        rr_suffix += "_world"

    if cross:
        out_dir  = ROOT / "experiments" / f"cross_train{args.train_subject}_test{args.test_subject}{rr_suffix}" / args.label
        mode_str = f"Cross  train={args.train_subject}  test={args.test_subject}"
    elif loso:
        out_dir  = ROOT / "experiments" / f"loso_test_{args.test_subject}{rr_suffix}" / args.label
        mode_str = f"LOSO  test_subject={args.test_subject}"
    else:
        out_dir  = ROOT / "experiments" / f"{args.subject}{rr_suffix}" / args.label
        mode_str = f"Mode1  subject={args.subject}"
    if args.root_relative:
        mode_str += "  [root-relative]"
    if args.torso_norm:
        mode_str += "  [torso-norm]"
    if args.predict_root:
        mode_str += "  [+root-head 42d]"
    if args.vis_mask:
        mode_str += "  [vis-mask]"
    if args.vel_spec:
        mode_str += "  [vel-spec 148d]"
    if args.phase:
        mode_str += "  [+phase 404d]"
    if tuple(args.dilations) != (1, 2, 4):
        mode_str += f"  [RF dilations={args.dilations}]"
    if args.state_head:
        mode_str += f"  [state-head λ={args.state_lambda:g}]"
    if args.bone_lambda > 0:
        mode_str += f"  [bone-loss λ={args.bone_lambda:g}]"
    if args.arch != "tcn":
        mode_str += f"  [arch={args.arch}]"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, test_rows = load_manifest_rows(
        label=args.label,
        subject=args.subject if not (loso or cross) else None,
        test_subject=args.test_subject,
        train_subject=args.train_subject,
    )
    if not train_rows:
        raise RuntimeError(f"manifest에서 train 데이터를 찾을 수 없음: {mode_str} label={args.label}")

    state_refs_map = {}
    if args.state_head:
        all_subjects = sorted({r["subject"] for r in train_rows + val_rows + test_rows})
        print(f"state 기준 분포 생성: {all_subjects} (각자의 train split 정적 trial GT 사용)")
        for subj in all_subjects:
            state_refs_map[subj] = build_state_refs(subj)

    def rows_to_seqs(rows):
        seqs = []
        for r in rows:
            seq = load_sequence(ROOT / r["csi_path"], ROOT / r["proxy13_path"], r["trial"],
                                vel_spec=args.vel_spec, phase=args.phase)
            if args.root_relative:
                seq["y"], seq["root"] = to_root_relative(seq["y"])
            if args.torso_norm:
                seq["y"], seq["torso"] = apply_torso_norm(seq["y"])
            if args.predict_root:
                # 상대좌표(모양) 39 + root 절대위치 3 = 42차원 타깃
                seq["y"] = np.concatenate([seq["y"], seq["root"]], axis=1)
            if args.vis_mask and r.get("full33_path"):
                attach_vis_weights(seq, ROOT / r["full33_path"], predict_root=args.predict_root,
                                   soft_root=args.vis_mask_v2)
            if args.motion_weight > 0:
                attach_motion_weights(seq, args.motion_weight)
            if args.state_head:
                seq["state_refs_subject"] = r["subject"]
                attach_state_labels(seq, state_refs_map[r["subject"]])
            seq["label"] = r["label"]
            seq["subject"] = r["subject"]
            seqs.append(seq)
        return seqs

    train_seqs = rows_to_seqs(train_rows)
    val_seqs   = rows_to_seqs(val_rows)
    test_seqs  = rows_to_seqs(test_rows)
    if args.state_head:
        print("[state 라벨 검증 게이트]")
        validate_state_labels(train_seqs + val_seqs, ":train+val")
        validate_state_labels(test_seqs, ":test")

    print("[CSI-to-Pose Training]")
    print(f"mode={mode_str}  label={args.label}")
    print(f"train={len(train_seqs)}개  val={len(val_seqs)}개  test={len(test_seqs)}개")
    train_subjects = sorted({r['subject'] for r in train_rows})
    test_subjects  = sorted({r['subject'] for r in test_rows})
    print(f"  train subjects: {train_subjects}")
    print(f"  test  subjects: {test_subjects}")
    print(f"target=13 joints x 3D = {len(JOINTS) * 3} values/frame")
    print(f"output={out_dir}")

    model, checkpoint, device = train_model(train_seqs, val_seqs, args, out_dir)

    summary_rows = []
    for split_name, split_seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        for seq in split_seqs:
            if args.state_head:
                pred, st_pred = predict_sequence(model, checkpoint, device, seq, return_state=True)
                m_valid = seq["state_w"] > 0
                if m_valid.any():
                    extra_state = {"state_acc": float((st_pred[m_valid] == seq["state"][m_valid]).mean())}
                    lying_m = m_valid & (seq["state"] == STATE_NAMES.index("lying"))
                    if lying_m.any():
                        extra_state["lying_recall"] = float((st_pred[lying_m] == STATE_NAMES.index("lying")).mean())
                else:
                    extra_state = {}
            else:
                pred = predict_sequence(model, checkpoint, device, seq)
                extra_state = {}
            mae, per_joint, extra = evaluate(seq["y"], pred)
            extra.update(extra_state)
            seq_label   = seq.get("label", args.label)
            seq_subject = seq.get("subject", args.subject)
            summary_rows.append(
                {
                    "split": split_name,
                    "subject": seq_subject,
                    "label": seq_label,
                    "trial": seq["trial"],
                    "mae_all": mae,
                    **extra,
                    **{f"mae_{k}": v for k, v in per_joint.items()},
                }
            )
            pred_csv = out_dir / "predictions" / f"{seq_subject}_{seq_label}_{seq['trial']}_prediction.csv"
            save_prediction_csv(seq, pred, pred_csv)
            if split_name == "test":
                plot_key = f"{seq_label}_{seq['trial']}"
                n_pose = len(JOINTS) * 3
                pose_seq = dict(seq, y=seq["y"][:, :n_pose])
                plot_summary(pose_seq, pred[:, :n_pose], out_dir / "plots" / f"{plot_key}_gt_vs_pred.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    print("\n[Metrics]")
    print(summary[["split", "label", "trial", "mae_all"]].to_string(index=False))
    if args.label == "ALL":
        print("\n[Per-label MAE (test split)]")
        test_df = summary[summary["split"] == "test"]
        per_label = test_df.groupby("label")["mae_all"].mean().sort_values()
        print(per_label.to_string())
        if "mae_root_y" in summary.columns:
            print("\n[Per-label root_y MAE (test split)] — 높이 예측 오차")
            print(test_df.groupby("label")["mae_root_y"].mean().sort_values().to_string())
    print(f"\n[DONE] model and outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
