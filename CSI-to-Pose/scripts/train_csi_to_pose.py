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
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


def load_csi_features(csi_path):
    df = pd.read_csv(csi_path)
    rel_t = (pd.to_numeric(df["pc_time_ms"], errors="coerce") - df["pc_time_ms"].iloc[0]) / 1000.0
    features = []
    times = []
    for _, row in df.iterrows():
        amp = parse_csi_array(row["raw_line"])
        if amp is None:
            continue
        raw_parts = str(row["raw_line"]).split(",")
        try:
            rssi = float(raw_parts[3])
        except Exception:
            rssi = np.nan
        amp = amp.astype(np.float32)
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        feature = np.concatenate(
            [
                amp,
                np.array(
                    [
                        np.nanmean(amp),
                        np.nanstd(amp),
                        np.nanmax(amp),
                        np.nanmin(amp),
                        rssi,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        features.append(feature)
        times.append(float(rel_t.loc[row.name]))
    return np.asarray(times, dtype=np.float32), np.asarray(features, dtype=np.float32)


def pose_target_columns():
    return [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]


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


def load_sequence(root, label, subject, trial):
    csi_path = root / "data" / "warning" / "gait" / label / subject / f"{subject}_{label}_{trial}.csv"
    pose_path = root / "pose_gt" / "proxy_13" / label / subject / f"{subject}_{label}_{trial}_proxy13.csv"
    if not csi_path.exists():
        raise FileNotFoundError(csi_path)
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    csi_t, csi_x = load_csi_features(csi_path)
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
        return torch.from_numpy(x.T).float(), torch.from_numpy(y).float()


class TinyTCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, out_dim, kernel_size=1),
        )

    def forward(self, x):
        # x: batch, feature, time
        y = self.net(x)
        return y.transpose(1, 2)


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
    model = TinyTCN(train_seqs[0]["x"].shape[1], train_seqs[0]["y"].shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    history = []
    best_val = math.inf
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).item()))
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "joints": JOINTS,
                },
                best_path,
            )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} | train {train_loss:.5f} | val {val_loss:.5f}")

    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"]).to_csv(
        out_dir / "training_history.csv", index=False
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint, device


def predict_sequence(model, checkpoint, device, seq):
    x = (seq["x"] - checkpoint["x_mean"]) / checkpoint["x_std"]
    xb = torch.from_numpy(x.T[None]).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(xb).cpu().numpy()[0]
    pred = pred * checkpoint["y_std"] + checkpoint["y_mean"]
    return pred.astype(np.float32)


def save_prediction_csv(seq, pred, out_path):
    cols = pose_target_columns()
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
    mae = np.nanmean(np.abs(gt - pred))
    per_joint = {}
    gt_j = gt.reshape(len(gt), len(JOINTS), 3)
    pr_j = pred.reshape(len(pred), len(JOINTS), 3)
    for idx, joint in enumerate(JOINTS):
        per_joint[joint] = float(np.nanmean(np.abs(gt_j[:, idx, :] - pr_j[:, idx, :])))
    return float(mae), per_joint


def main():
    parser = argparse.ArgumentParser(description="Train CSI-to-3D-skeleton-proxy pilot model.")
    parser.add_argument("--label", default="unstable_walking")
    parser.add_argument("--subject", default="yja")
    parser.add_argument("--trials", nargs="+", default=[f"t{i:03d}" for i in range(1, 11)])
    parser.add_argument("--train_trials", nargs="+", default=[f"t{i:03d}" for i in range(1, 9)])
    parser.add_argument("--val_trials", nargs="+", default=["t009"])
    parser.add_argument("--test_trials", nargs="+", default=["t010"])
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = EXPERIMENT_ROOT / "models" / args.label / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = {trial: load_sequence(EXPERIMENT_ROOT, args.label, args.subject, trial) for trial in args.trials}
    train_seqs = [seqs[t] for t in args.train_trials]
    val_seqs = [seqs[t] for t in args.val_trials]
    test_seqs = [seqs[t] for t in args.test_trials]

    print("[CSI-to-Pose Training]")
    print(f"label={args.label} subject={args.subject}")
    print(f"train={args.train_trials} val={args.val_trials} test={args.test_trials}")
    print(f"target=13 joints x 3D = {len(JOINTS) * 3} values/frame")
    print(f"output={out_dir}")

    model, checkpoint, device = train_model(train_seqs, val_seqs, args, out_dir)

    summary_rows = []
    for split_name, split_seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        for seq in split_seqs:
            pred = predict_sequence(model, checkpoint, device, seq)
            mae, per_joint = evaluate(seq["y"], pred)
            summary_rows.append(
                {
                    "split": split_name,
                    "trial": seq["trial"],
                    "mae_all": mae,
                    **{f"mae_{k}": v for k, v in per_joint.items()},
                }
            )
            pred_csv = out_dir / "predictions" / f"{args.subject}_{args.label}_{seq['trial']}_prediction.csv"
            save_prediction_csv(seq, pred, pred_csv)
            if split_name == "test":
                plot_summary(seq, pred, out_dir / "plots" / f"{seq['trial']}_gt_vs_pred.png")
                anim_path = make_animation(seq, pred, out_dir / "animations" / f"{seq['trial']}_gt_vs_pred.mp4")
                print(f"test animation: {anim_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    print("\n[Metrics]")
    print(summary[["split", "trial", "mae_all"]].to_string(index=False))
    print(f"\n[DONE] model and outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
