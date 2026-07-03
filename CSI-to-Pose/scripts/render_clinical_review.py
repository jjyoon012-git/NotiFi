import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


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

JOINT_INDEX = {name: idx for idx, name in enumerate(JOINTS)}

EDGES = [
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

GROUPS = {
    "head": ["head"],
    "shoulder": ["left_shoulder", "right_shoulder"],
    "elbow": ["left_elbow", "right_elbow"],
    "wrist": ["left_wrist", "right_wrist"],
    "hip": ["left_hip", "right_hip"],
    "knee": ["left_knee", "right_knee"],
    "ankle": ["left_ankle", "right_ankle"],
}

KO_JOINT = {
    "head": "머리/목",
    "left_shoulder": "좌 어깨",
    "right_shoulder": "우 어깨",
    "left_elbow": "좌 팔꿈치",
    "right_elbow": "우 팔꿈치",
    "left_wrist": "좌 손목",
    "right_wrist": "우 손목",
    "left_hip": "좌 골반",
    "right_hip": "우 골반",
    "left_knee": "좌 무릎",
    "right_knee": "우 무릎",
    "left_ankle": "좌 발목",
    "right_ankle": "우 발목",
}

KO_GROUP = {
    "head": "머리/목",
    "shoulder": "어깨/상체",
    "elbow": "팔꿈치",
    "wrist": "손목/방어동작",
    "hip": "골반/중심",
    "knee": "무릎/체중지지",
    "ankle": "발목/보행",
}

GROUP_HINT = {
    "head": "머리 높이/목 정렬 변화",
    "shoulder": "상체 기울기 변화",
    "elbow": "팔 지지 또는 방어동작",
    "wrist": "손 짚기/방어 반응",
    "hip": "몸 중심 하강/이동",
    "knee": "체중 지지와 굴곡 변화",
    "ankle": "발목 흔들림/보행 불안정",
}


def load_fonts():
    regular_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    bold_path = Path(r"C:\Windows\Fonts\malgunbd.ttf")
    if not regular_path.exists() and not bold_path.exists():
        fallback = ImageFont.load_default()
        return {
            "title": fallback,
            "h1": fallback,
            "h2": fallback,
            "body": fallback,
            "small": fallback,
            "tiny": fallback,
            "mono": fallback,
        }
    regular = str(regular_path if regular_path.exists() else bold_path)
    bold = str(bold_path if bold_path.exists() else regular_path)
    return {
        "title": ImageFont.truetype(bold, 34),
        "h1": ImageFont.truetype(bold, 28),
        "h2": ImageFont.truetype(bold, 22),
        "body": ImageFont.truetype(regular, 20),
        "small": ImageFont.truetype(regular, 16),
        "tiny": ImageFont.truetype(regular, 14),
        "mono": ImageFont.truetype(regular, 17),
    }


def bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return b, g, r


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def load_prediction(path: Path):
    df = pd.read_csv(path)
    time = pd.to_numeric(df["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float32)
    gt_cols = [f"gt_{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]
    pr_cols = [f"pred_{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]
    gt = df[gt_cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").fillna(0).to_numpy(dtype=np.float32)
    pred = df[pr_cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").fillna(0).to_numpy(dtype=np.float32)
    meta = {
        "subject": str(df["subject"].iloc[0]),
        "label": str(df["label"].iloc[0]),
        "trial": str(df["trial"].iloc[0]),
        "ambient": str(df["ambient"].iloc[0]),
    }
    return time, gt.reshape(len(df), len(JOINTS), 3), pred.reshape(len(df), len(JOINTS), 3), meta


def derivative(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    d = np.zeros_like(x)
    d[1:] = (x[1:] - x[:-1]) / max(dt, 1e-6)
    d[0] = d[1]
    return d


def torso_angle(pose: np.ndarray) -> np.ndarray:
    shoulder = 0.5 * (pose[:, JOINT_INDEX["left_shoulder"], :2] + pose[:, JOINT_INDEX["right_shoulder"], :2])
    hip = 0.5 * (pose[:, JOINT_INDEX["left_hip"], :2] + pose[:, JOINT_INDEX["right_hip"], :2])
    vec = shoulder - hip
    return np.degrees(np.arctan2(vec[:, 0], -vec[:, 1]))


def sequence_features(time: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    valid_dt = np.diff(time)
    valid_dt = valid_dt[np.isfinite(valid_dt) & (valid_dt > 1e-4)]
    dt = float(np.median(valid_dt)) if valid_dt.size else 1 / 12

    gt_vel = np.linalg.norm(derivative(gt, dt), axis=-1)
    pred_vel = np.linalg.norm(derivative(pred, dt), axis=-1)
    pred_acc = np.abs(derivative(pred_vel, dt))
    err = np.linalg.norm(pred - gt, axis=-1)

    vel_ref = float(np.percentile(pred_vel, 95)) if pred_vel.size else 1.0
    acc_ref = float(np.percentile(pred_acc, 95)) if pred_acc.size else 1.0
    vel_ref = max(vel_ref, 1e-4)
    acc_ref = max(acc_ref, 1e-4)

    group_scores = {}
    for group, joints in GROUPS.items():
        idxs = [JOINT_INDEX[j] for j in joints]
        peak_v = float(np.percentile(pred_vel[:, idxs], 95))
        peak_a = float(np.percentile(pred_acc[:, idxs], 95))
        score = min(100.0, 100.0 * (0.62 * peak_v / vel_ref + 0.38 * peak_a / acc_ref))
        group_scores[group] = score

    pred_head_y = pred[:, JOINT_INDEX["head"], 1]
    pred_hip_y = 0.5 * (pred[:, JOINT_INDEX["left_hip"], 1] + pred[:, JOINT_INDEX["right_hip"], 1])
    head_drop = float(np.nanmax(pred_head_y) - np.nanmin(pred_head_y))
    hip_drop = float(np.nanmax(pred_hip_y) - np.nanmin(pred_hip_y))
    angle_change = float(np.nanmax(torso_angle(pred)) - np.nanmin(torso_angle(pred)))

    total_motion = pred_vel.mean(axis=1)
    post_static = 0.0
    if len(total_motion) > 8:
        tail = total_motion[int(len(total_motion) * 0.7) :]
        post_static = float(1.0 - min(1.0, np.mean(tail) / max(np.percentile(total_motion, 85), 1e-4)))

    return {
        "dt": dt,
        "gt_vel": gt_vel,
        "pred_vel": pred_vel,
        "pred_acc": pred_acc,
        "err": err,
        "vel_ref": vel_ref,
        "group_scores": group_scores,
        "head_drop": head_drop,
        "hip_drop": hip_drop,
        "angle_change": angle_change,
        "post_static": post_static,
        "mpjpe": float(err.mean()),
        "mae": float(np.abs(pred - gt).mean()),
    }


def stress_color(score: float) -> tuple[int, int, int]:
    score = float(np.clip(score, 0.0, 1.0))
    stops = [
        (0.0, np.array(bgr("#3b82f6"), dtype=np.float32)),
        (0.35, np.array(bgr("#22c55e"), dtype=np.float32)),
        (0.68, np.array(bgr("#facc15"), dtype=np.float32)),
        (1.0, np.array(bgr("#ef4444"), dtype=np.float32)),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if score <= b:
            t = (score - a) / max(b - a, 1e-6)
            c = ca * (1 - t) + cb * t
            return tuple(int(v) for v in c)
    return tuple(int(v) for v in stops[-1][1])


def draw_rounded_rect(img, xy, color, radius=14, thickness=-1):
    x1, y1, x2, y2 = map(int, xy)
    if thickness != -1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def draw_grid(img, box):
    x, y, w, h = box
    draw_rounded_rect(img, (x, y, x + w, y + h), bgr("#08111f"), radius=18)
    cv2.rectangle(img, (x, y), (x + w, y + h), bgr("#1f3b57"), 1, cv2.LINE_AA)
    for gx in range(x + 60, x + w, 80):
        cv2.line(img, (gx, y + 20), (gx, y + h - 20), bgr("#12263b"), 1, cv2.LINE_AA)
    for gy in range(y + 60, y + h, 80):
        cv2.line(img, (x + 20, gy), (x + w - 20, gy), bgr("#12263b"), 1, cv2.LINE_AA)
    cv2.line(img, (x + 40, y + h - 78), (x + w - 40, y + h - 78), bgr("#36546f"), 2, cv2.LINE_AA)


def make_screen_mapper(all_pose: np.ndarray, left_box, right_box):
    xy = all_pose[:, :, :2].reshape(-1, 2)
    center = np.nanmean(xy, axis=0)
    min_xy = np.nanmin(xy, axis=0)
    max_xy = np.nanmax(xy, axis=0)
    span = np.maximum(max_xy - min_xy, np.array([0.55, 0.9], dtype=np.float32))
    scale = min(left_box[2] * 0.56 / float(span[0]), left_box[3] * 0.62 / float(span[1]))

    def map_points(pose, box):
        x, y, w, h = box
        pts = pose[:, :2]
        out = np.zeros((len(JOINTS), 2), dtype=np.int32)
        out[:, 0] = np.round(x + w / 2 + (pts[:, 0] - center[0]) * scale).astype(np.int32)
        out[:, 1] = np.round(y + h * 0.53 + (pts[:, 1] - center[1]) * scale).astype(np.int32)
        return out

    return map_points


def draw_trail(img, poses, idx, box, mapper, joint_name, color):
    start = max(0, idx - 36)
    pts = []
    for frame_idx in range(start, idx + 1):
        screen = mapper(poses[frame_idx], box)
        pts.append(screen[JOINT_INDEX[joint_name]])
    if len(pts) < 2:
        return
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        alpha = (i + 1) / len(pts)
        c = tuple(int(v * alpha + 24 * (1 - alpha)) for v in color)
        cv2.line(img, tuple(a), tuple(b), c, 2, cv2.LINE_AA)


def draw_skeleton(img, poses, idx, box, mapper, motion, vel_ref, title_color, ghost=False):
    pose = poses[idx]
    pts = mapper(pose, box)
    draw_trail(img, poses, idx, box, mapper, "head", bgr("#f8fafc") if not ghost else bgr("#64748b"))
    draw_trail(img, poses, idx, box, mapper, "left_hip", bgr("#60a5fa") if not ghost else bgr("#64748b"))
    draw_trail(img, poses, idx, box, mapper, "right_hip", bgr("#f59e0b") if not ghost else bgr("#64748b"))

    torso = np.array(
        [
            pts[JOINT_INDEX["left_shoulder"]],
            pts[JOINT_INDEX["right_shoulder"]],
            pts[JOINT_INDEX["right_hip"]],
            pts[JOINT_INDEX["left_hip"]],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [torso], bgr("#1e3a5f") if not ghost else bgr("#334155"))

    for a, b in EDGES:
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        if ghost:
            color = bgr("#93a4b8")
            width = 4
        else:
            score = np.clip((motion[ia] + motion[ib]) / (2 * vel_ref), 0, 1)
            color = stress_color(score)
            width = 7 if score > 0.55 else 5
        cv2.line(img, tuple(pts[ia]), tuple(pts[ib]), color, width, cv2.LINE_AA)

    for joint, jidx in JOINT_INDEX.items():
        score = float(np.clip(motion[jidx] / vel_ref, 0, 1))
        radius = 12 if joint == "head" else 8
        if ghost:
            color = bgr("#cbd5e1")
            radius = 7 if joint != "head" else 10
        else:
            color = stress_color(score)
            if score > 0.72:
                cv2.circle(img, tuple(pts[jidx]), radius + 12, bgr("#7f1d1d"), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[jidx]), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[jidx]), radius, bgr("#030712"), 1, cv2.LINE_AA)

    return pts


def fit_text(draw, text, font, max_width):
    if draw.textlength(text, font=font) <= max_width:
        return text
    out = text
    while out and draw.textlength(out + "...", font=font) > max_width:
        out = out[:-1]
    return out + "..." if out else ""


def draw_texts(img, text_items, fonts):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for item in text_items:
        text, xy, font_name, color = item[:4]
        max_width = item[4] if len(item) > 4 else None
        font = fonts[font_name]
        if max_width:
            text = fit_text(draw, text, font, max_width)
        draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_bar(img, x, y, w, h, value, color, bg="#172033"):
    draw_rounded_rect(img, (x, y, x + w, y + h), bgr(bg), radius=6)
    fill = int(w * float(np.clip(value, 0, 1)))
    if fill > 0:
        draw_rounded_rect(img, (x, y, x + fill, y + h), color, radius=6)


def clinical_notes(meta, feats, idx):
    notes = []
    pred_vel = feats["pred_vel"]
    err = feats["err"]
    top_joint = JOINTS[int(np.argmax(pred_vel[idx]))]
    notes.append(f"{KO_JOINT[top_joint]} 움직임 급증: 속도 {pred_vel[idx].max():.2f}")
    if feats["head_drop"] > 0.22 or feats["hip_drop"] > 0.20:
        notes.append(f"머리/골반 하강폭 큼: head {feats['head_drop']:.2f}, hip {feats['hip_drop']:.2f}")
    if feats["angle_change"] > 28:
        notes.append(f"상체 축 변화 큼: torso Δ {feats['angle_change']:.1f}°")
    if feats["post_static"] > 0.45:
        notes.append(f"움직임 후 정지 경향: static {feats['post_static']:.2f}")
    if feats["mpjpe"] > 0.23:
        notes.append(f"복원 신뢰도 주의: MPJPE {feats['mpjpe']:.3f}")
    else:
        notes.append(f"복원-Teacher 일치도 양호: MPJPE {feats['mpjpe']:.3f}")
    if err[idx].mean() > feats["mpjpe"] * 1.25:
        notes.append("현재 frame은 평균보다 복원 오차가 큼")
    return notes[:4]


def draw_clinical_frame(time, gt, pred, meta, feats, idx, mapper, fonts, width=1920, height=1080):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = bgr("#050b14")

    header_h = 98
    left_box = (42, 130, 860, 610)
    right_box = (1018, 130, 860, 610)
    bottom_box = (42, 770, 1836, 270)

    draw_rounded_rect(img, (24, 22, width - 24, header_h), bgr("#0b1628"), radius=18)
    draw_grid(img, left_box)
    draw_grid(img, right_box)
    draw_rounded_rect(img, bottom_box, bgr("#0b1628"), radius=18)
    cv2.line(img, (960, 132), (960, 740), bgr("#24425e"), 2, cv2.LINE_AA)

    gt_motion = feats["gt_vel"][idx]
    pred_motion = feats["pred_vel"][idx]
    draw_skeleton(img, gt, idx, left_box, mapper, gt_motion, feats["vel_ref"], bgr("#38bdf8"), ghost=False)
    draw_skeleton(img, pred, idx, right_box, mapper, pred_motion, feats["vel_ref"], bgr("#f59e0b"), ghost=False)

    top_joint_idxs = np.argsort(-pred_motion)[:4]
    frame_group_scores = {}
    for group, joints in GROUPS.items():
        idxs = [JOINT_INDEX[j] for j in joints]
        frame_group_scores[group] = min(100.0, 100.0 * float(np.mean(pred_motion[idxs])) / max(feats["vel_ref"], 1e-4))
    group_rank = sorted(frame_group_scores.items(), key=lambda kv: -kv[1])[:4]
    notes = clinical_notes(meta, feats, idx)[:3]

    text = [
        ("CSI-to-Pose Clinical Motion Review", (50, 34), "title", rgb("#f8fafc")),
        ("의학적 진단이 아닌 움직임/부하 해석용 시각화", (50, 72), "small", rgb("#94a3b8")),
        (f"{meta['subject']} | {meta['label']} | {meta['trial']} | {meta['ambient']}", (1260, 42), "h2", rgb("#dbeafe")),
        (f"time {time[idx]:.2f}s   frame {idx + 1}/{len(time)}   MPJPE {feats['mpjpe']:.3f}", (1260, 74), "small", rgb("#bfdbfe")),
        ("GT: Video Pose Teacher", (68, 148), "h1", rgb("#93c5fd")),
        ("왼쪽은 학습용 영상에서 추출한 기준 skeleton", (68, 184), "small", rgb("#9fb2c7")),
        ("CSI-only Reconstruction", (1042, 148), "h1", rgb("#fbbf24")),
        ("오른쪽은 CSI 신호만으로 복원한 skeleton", (1042, 184), "small", rgb("#d8c69b")),
        ("관절 변화량 / 부하 의심 리포트", (68, 792), "h1", rgb("#f8fafc")),
        ("빠르게 움직이거나 급가속한 관절일수록 yellow/red로 표시", (68, 826), "small", rgb("#94a3b8")),
    ]

    x0, y0 = 70, 872
    for row, jidx in enumerate(top_joint_idxs):
        y = y0 + row * 38
        joint = JOINTS[int(jidx)]
        value = float(np.clip(pred_motion[jidx] / feats["vel_ref"], 0, 1))
        color = stress_color(value)
        draw_bar(img, x0 + 160, y + 2, 250, 18, value, color)
        text.append((KO_JOINT[joint], (x0, y - 2), "small", rgb("#dbeafe")))
        text.append((f"속도 {pred_motion[jidx]:.2f} | 오차 {feats['err'][idx, jidx]:.3f}", (x0 + 425, y - 3), "tiny", rgb("#cbd5e1")))

    gx, gy = 760, 870
    text.append(("주의 부위 후보", (gx, gy - 44), "h2", rgb("#fecaca")))
    for row, (group, score) in enumerate(group_rank):
        y = gy + row * 44
        value = score / 100.0
        draw_bar(img, gx + 150, y + 3, 240, 20, value, stress_color(value), bg="#1f2937")
        text.append((KO_GROUP[group], (gx, y - 2), "small", rgb("#fee2e2")))
        text.append((f"{score:4.0f}/100", (gx + 405, y - 3), "small", rgb("#fef3c7")))

    nx, ny = 1360, 862
    text.append(("판독 포인트", (nx, ny - 36), "h2", rgb("#bfdbfe")))
    for row, note in enumerate(notes):
        text.append((f"• {note}", (nx, ny + row * 42), "small", rgb("#e5e7eb"), 500))
    text.append((f"Sequence: head drop {feats['head_drop']:.2f} | hip drop {feats['hip_drop']:.2f} | torso Δ {feats['angle_change']:.1f}°", (nx, ny + 142), "tiny", rgb("#94a3b8"), 510))

    return draw_texts(img, text, fonts)


def render_video(prediction_path: Path, out_path: Path, max_frames: int, fps: int, fonts):
    time, gt, pred, meta = load_prediction(prediction_path)
    feats = sequence_features(time, gt, pred)
    all_pose = np.concatenate([gt, pred], axis=0)
    left_box = (42, 130, 860, 610)
    right_box = (1018, 130, 860, 610)
    mapper = make_screen_mapper(all_pose, left_box, right_box)

    frame_count = min(max_frames, len(time))
    idxs = np.linspace(0, len(time) - 1, frame_count).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for idx in idxs:
        frame = draw_clinical_frame(time, gt, pred, meta, feats, int(idx), mapper, fonts)
        writer.write(frame)
    writer.release()
    return {
        "source": str(prediction_path),
        "video": str(out_path),
        "subject": meta["subject"],
        "label": meta["label"],
        "trial": meta["trial"],
        "ambient": meta["ambient"],
        "mpjpe": feats["mpjpe"],
        "mae": feats["mae"],
        "top_regions": sorted(feats["group_scores"].items(), key=lambda kv: -kv[1])[:5],
    }


def representative_predictions(output_root: Path):
    report_path = output_root / "experiment_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        paths = []
        for item in report.get("rendered", []):
            label = item["label"]
            subject = item["subject"]
            trial = item["trial"]
            pred_dir = output_root / "predictions" / "test" / label
            matches = sorted(pred_dir.glob(f"{subject}_{label}_{trial}_prediction.csv"))
            paths.extend(matches)
        if paths:
            return paths

    metrics = pd.read_csv(output_root / "metrics_summary.csv")
    metrics = metrics[metrics["split"] == "test"].copy()
    chosen = []
    for label, group in metrics.groupby("label"):
        row = group.sort_values("mpjpe_all").iloc[0]
        pred_dir = output_root / "predictions" / "test" / label
        matches = sorted(pred_dir.glob(f"{row['subject']}_{label}_{row['trial']}_prediction.csv"))
        chosen.extend(matches)
    return chosen


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Render clinical-style GT vs CSI reconstruction review videos.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Training output directory. Defaults to outputs/manifest_split_all_labels.",
    )
    parser.add_argument("--all-test", action="store_true", help="Render all test prediction CSVs, not only one per label.")
    parser.add_argument("--out-subdir", default="clinical_review_videos_no_gray")
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else repo_root / "outputs" / "manifest_split_all_labels"
    fonts = load_fonts()
    if args.all_test:
        prediction_paths = sorted((output_root / "predictions" / "test").glob("*/*_prediction.csv"))
    else:
        prediction_paths = representative_predictions(output_root)

    manifest = []
    for idx, prediction_path in enumerate(prediction_paths, start=1):
        label = prediction_path.parent.name
        out_path = (
            output_root
            / args.out_subdir
            / label
            / prediction_path.name.replace("_prediction.csv", "_clinical_review.mp4")
        )
        item = render_video(prediction_path, out_path, args.max_frames, args.fps, fonts)
        manifest.append(item)
        print(f"[{idx}/{len(prediction_paths)}] {out_path}")

    manifest_path = output_root / args.out_subdir / "clinical_review_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
