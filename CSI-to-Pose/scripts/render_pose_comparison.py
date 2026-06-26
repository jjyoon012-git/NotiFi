import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Polygon


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "csi_to_pose"

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

LEFT_JOINTS = {"left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"}
RIGHT_JOINTS = {"right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"}


def load_prediction(path):
    df = pd.read_csv(path)
    gt = {}
    pred = {}
    for joint in JOINTS:
        gt[joint] = df[[f"gt_{joint}_x", f"gt_{joint}_y", f"gt_{joint}_z"]].to_numpy(dtype=float)
        pred[joint] = df[[f"pred_{joint}_x", f"pred_{joint}_y", f"pred_{joint}_z"]].to_numpy(dtype=float)
    return df["timestamp_sec"].to_numpy(dtype=float), gt, pred


def front_point(points, idx):
    # MediaPipe normalized image coordinates: y grows downward.
    # Render in a human-readable front view: x horizontal, 1-y vertical.
    x = points[idx, 0]
    y = 1.0 - points[idx, 1]
    return np.array([x, y], dtype=float)


def joint_color(joint):
    if joint in LEFT_JOINTS:
        return "#1f77b4"
    if joint in RIGHT_JOINTS:
        return "#ff7f0e"
    return "#333333"


def draw_person(ax, pose, idx, title, alpha=1.0):
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.05, 0.98)
    ax.set_ylim(-0.45, 0.98)

    pts = {joint: front_point(pose[joint], idx) for joint in JOINTS}

    # Torso shape makes the skeleton read like a human body instead of loose lines.
    torso_names = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]
    torso_pts = np.array([pts[name] for name in torso_names])
    if np.isfinite(torso_pts).all():
        ax.add_patch(
            Polygon(
                torso_pts,
                closed=True,
                facecolor="#7f7f7f",
                edgecolor="#444444",
                linewidth=1.2,
                alpha=0.18 * alpha,
            )
        )

    for a, b in EDGES:
        pa, pb = pts[a], pts[b]
        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            continue
        color = joint_color(a if a.startswith(("left", "right")) else b)
        linewidth = 4.2 if ("hip" in a or "shoulder" in a or "knee" in a or "ankle" in a) else 3.2
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            alpha=alpha,
        )

    head = pts["head"]
    if np.isfinite(head).all():
        ax.add_patch(Circle(head, 0.035, facecolor="#333333", edgecolor="white", linewidth=1.5, alpha=alpha))

    for joint, p in pts.items():
        if not np.isfinite(p).all() or joint == "head":
            continue
        ax.scatter(
            p[0],
            p[1],
            s=42,
            color=joint_color(joint),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            alpha=alpha,
        )

    # Hip trail helps show walking direction and body sway.
    if idx > 1:
        hip_trail = []
        start = max(0, idx - 25)
        for i in range(start, idx + 1):
            lh = front_point(pose["left_hip"], i)
            rh = front_point(pose["right_hip"], i)
            hip_trail.append((lh + rh) / 2)
        hip_trail = np.array(hip_trail)
        if np.isfinite(hip_trail).all():
            ax.plot(hip_trail[:, 0], hip_trail[:, 1], color="#2ca02c", linewidth=2, alpha=0.45)


def make_animation(time, gt, pred, out_path, max_frames=180):
    n = len(time)
    idxs = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.5), constrained_layout=True)

    def update(frame_no):
        idx = idxs[frame_no]
        for ax in axes:
            ax.clear()
        draw_person(axes[0], gt, idx, f"GT MediaPipe Pose\n{time[idx]:.1f}s")
        draw_person(axes[1], pred, idx, f"CSI Prediction\n{time[idx]:.1f}s")
        fig.suptitle("CSI-to-Pose: Human-readable Skeleton Proxy", fontsize=15, weight="bold")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(idxs), interval=80, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(out_path, writer="ffmpeg", fps=12)
        saved = out_path
    except Exception:
        saved = out_path.with_suffix(".gif")
        ani.save(saved, writer="pillow", fps=12)
    plt.close(fig)
    return saved


def make_contact_sheet(time, gt, pred, out_path, frames=6):
    n = len(time)
    idxs = np.linspace(0, n - 1, min(frames, n)).astype(int)
    fig, axes = plt.subplots(len(idxs), 2, figsize=(8, 3.2 * len(idxs)), constrained_layout=True)
    if len(idxs) == 1:
        axes = np.array([axes])
    for row, idx in enumerate(idxs):
        draw_person(axes[row, 0], gt, idx, f"GT {time[idx]:.1f}s")
        draw_person(axes[row, 1], pred, idx, f"Pred {time[idx]:.1f}s")
    fig.suptitle("CSI-to-Pose Contact Sheet", fontsize=15, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render human-readable CSI-to-Pose skeleton comparison.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--outdir")
    parser.add_argument("--max_frames", type=int, default=180)
    args = parser.parse_args()

    pred_path = Path(args.prediction)
    outdir = Path(args.outdir) if args.outdir else pred_path.parent.parent / "human_readable"
    time, gt, pred = load_prediction(pred_path)
    anim = make_animation(time, gt, pred, outdir / f"{pred_path.stem}_human_readable.mp4", args.max_frames)
    sheet = outdir / f"{pred_path.stem}_contact_sheet.png"
    make_contact_sheet(time, gt, pred, sheet)
    print(f"animation: {anim}")
    print(f"contact_sheet: {sheet}")


if __name__ == "__main__":
    main()
