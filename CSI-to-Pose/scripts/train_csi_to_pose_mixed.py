import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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

LEFT_JOINTS = {name for name in JOINTS if name.startswith("left_")}
RIGHT_JOINTS = {name for name in JOINTS if name.startswith("right_")}

POSE_COLS = [f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y", "z")]


@dataclass
class Sequence:
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    meta: dict
    pose_raw: np.ndarray
    pose_scale: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def parse_csi_amp(raw_line: str, amp_dim: int) -> np.ndarray | None:
    start = raw_line.rfind("[")
    end = raw_line.rfind("]")
    if start < 0 or end <= start:
        return None
    arr = np.fromstring(raw_line[start + 1 : end], sep=",", dtype=np.float32)
    if arr.size < 2:
        return None
    if arr.size % 2 == 1:
        arr = arr[:-1]
    real = arr[0::2]
    imag = arr[1::2]
    amp = np.sqrt(real * real + imag * imag).astype(np.float32)
    if amp.size < amp_dim:
        amp = np.pad(amp, (0, amp_dim - amp.size))
    elif amp.size > amp_dim:
        amp = amp[:amp_dim]
    return amp


def load_csi_features(csi_path: Path, cache_dir: Path, amp_dim: int) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = stable_hash(str(csi_path.resolve()))
    cache_path = cache_dir / f"{cache_key}_csi.npz"
    if cache_path.exists() and cache_path.stat().st_mtime >= csi_path.stat().st_mtime:
        cached = np.load(cache_path)
        return cached["time"].astype(np.float32), cached["x"].astype(np.float32)

    df = pd.read_csv(csi_path)
    if "pc_time_ms" not in df.columns or "raw_line" not in df.columns:
        raise ValueError(f"CSI file has unexpected columns: {csi_path}")

    pc_time = pd.to_numeric(df["pc_time_ms"], errors="coerce").to_numpy(dtype=np.float64)
    rel_t = (pc_time - np.nanmin(pc_time)) / 1000.0
    features = []
    times = []

    for idx, raw in enumerate(df["raw_line"].astype(str)):
        amp = parse_csi_amp(raw, amp_dim)
        if amp is None:
            continue
        parts = raw.split(",", 5)
        try:
            rssi = float(parts[3])
        except Exception:
            rssi = 0.0
        finite_amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        feature = np.concatenate(
            [
                finite_amp,
                np.array(
                    [
                        finite_amp.mean(),
                        finite_amp.std(),
                        finite_amp.max(initial=0.0),
                        finite_amp.min(initial=0.0),
                        rssi,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        if np.isfinite(rel_t[idx]):
            features.append(feature)
            times.append(float(rel_t[idx]))

    if not features:
        raise ValueError(f"No parsable CSI frames: {csi_path}")

    time = np.asarray(times, dtype=np.float32)
    x = np.asarray(features, dtype=np.float32)
    order = np.argsort(time)
    time = time[order]
    x = x[order]
    unique_time, unique_idx = np.unique(time, return_index=True)
    time = unique_time.astype(np.float32)
    x = x[unique_idx].astype(np.float32)
    np.savez_compressed(cache_path, time=time, x=x)
    return time, x


def load_pose_targets(pose_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    df = pd.read_csv(pose_path)
    missing = [col for col in POSE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Pose file has missing columns {missing[:4]}: {pose_path}")

    time = pd.to_numeric(df["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float32)
    pose_df = df[POSE_COLS].apply(pd.to_numeric, errors="coerce")
    pose_df = pose_df.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    raw = pose_df.to_numpy(dtype=np.float32).reshape(-1, len(JOINTS), 3)

    hip_center = 0.5 * (raw[:, JOINT_INDEX["left_hip"], :] + raw[:, JOINT_INDEX["right_hip"], :])
    centered = raw - hip_center[:, None, :]

    y_span = np.nanmax(raw[:, :, 1], axis=1) - np.nanmin(raw[:, :, 1], axis=1)
    body_height = pd.to_numeric(df.get("body_height", pd.Series(y_span)), errors="coerce").to_numpy(dtype=np.float32)
    scale_candidates = np.concatenate([y_span[np.isfinite(y_span)], body_height[np.isfinite(body_height)]])
    scale_candidates = scale_candidates[scale_candidates > 1e-4]
    scale = float(np.median(scale_candidates)) if scale_candidates.size else 1.0
    if not np.isfinite(scale) or scale < 1e-4:
        scale = 1.0

    canonical = centered / scale
    canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return time, canonical.reshape(len(canonical), -1), raw.astype(np.float32), scale


def resample_csi_to_pose(csi_t: np.ndarray, csi_x: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
    if len(csi_t) == 0:
        raise ValueError("No CSI timestamps")
    pose_t = np.nan_to_num(pose_t, nan=0.0)
    out = np.empty((len(pose_t), csi_x.shape[1]), dtype=np.float32)
    for dim in range(csi_x.shape[1]):
        out[:, dim] = np.interp(pose_t, csi_t, csi_x[:, dim]).astype(np.float32)
    return out


def load_sequence(row: dict, data_root: Path, cache_dir: Path, amp_dim: int) -> Sequence:
    csi_path = data_root / row["csi_path"]
    pose_path = data_root / row["proxy13_path"]
    csi_t, csi_x = load_csi_features(csi_path, cache_dir, amp_dim)
    pose_t, y, pose_raw, pose_scale = load_pose_targets(pose_path)
    x = resample_csi_to_pose(csi_t, csi_x, pose_t)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Sequence(
        x=x,
        y=y,
        time=pose_t.astype(np.float32),
        meta={
            "subject": row["subject"],
            "prefix": row["prefix"],
            "label": row["label"],
            "trial": row["trial"],
            "ambient": row["ambient"],
            "split": row["mixed_split"],
            "csi_path": row["csi_path"],
            "proxy13_path": row["proxy13_path"],
        },
        pose_raw=pose_raw,
        pose_scale=pose_scale,
    )


def read_manifest(data_root: Path) -> list[dict]:
    manifest = data_root / "split_manifest" / "manifest_full.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [
        row
        for row in rows
        if row.get("valid") == "True"
        and row.get("csi_exists") == "True"
        and row.get("proxy13_exists") == "True"
    ]
    if not rows:
        raise RuntimeError("No valid paired CSI/proxy13 rows found in manifest.")
    return rows


def split_targets(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    train = int(round(total * ratios[0]))
    val = int(round(total * ratios[1]))
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def allocate_counts(strata: list[tuple[tuple[str, str], list[dict]]], ratios: tuple[float, float, float], seed: int) -> dict:
    split_names = ["train", "val", "test"]
    total = sum(len(rows) for _, rows in strata)
    targets = split_targets(total, ratios)
    counts = {}
    slots = {}
    current = {name: 0 for name in split_names}

    for key, rows in strata:
        n = len(rows)
        exact = {"train": n * ratios[0], "val": n * ratios[1], "test": n * ratios[2]}
        counts[key] = {name: int(math.floor(exact[name])) for name in split_names}
        used = sum(counts[key].values())
        slots[key] = n - used
        for name in split_names:
            current[name] += counts[key][name]

    needs = {name: targets[name] - current[name] for name in split_names}
    rng = random.Random(seed)
    order = [key for key, _ in strata]
    rng.shuffle(order)

    while any(value > 0 for value in needs.values()):
        progressed = False
        for key in order:
            if slots[key] <= 0:
                continue
            candidates = [name for name in split_names if needs[name] > 0]
            if not candidates:
                break
            chosen = max(candidates, key=lambda name: (needs[name], rng.random()))
            counts[key][chosen] += 1
            needs[chosen] -= 1
            slots[key] -= 1
            progressed = True
        if not progressed:
            break

    for key in order:
        while slots[key] > 0:
            chosen = min(split_names, key=lambda name: counts[key][name])
            counts[key][chosen] += 1
            slots[key] -= 1

    return counts


def choose_rows_by_ambient(rows: list[dict], count: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    by_ambient = {}
    for row in rows:
        by_ambient.setdefault(row["ambient"], []).append(row)
    ambients = sorted(by_ambient)
    rng.shuffle(ambients)
    for items in by_ambient.values():
        items.sort(key=lambda row: int(row["trial_num"]))
        rng.shuffle(items)

    chosen = []
    while len(chosen) < count and any(by_ambient.values()):
        for ambient in ambients:
            if len(chosen) >= count:
                break
            if by_ambient[ambient]:
                chosen.append(by_ambient[ambient].pop(0))
    chosen_ids = {id(row) for row in chosen}
    remaining = [row for row in rows if id(row) not in chosen_ids]
    return chosen, remaining


def build_mixed_split(rows: list[dict], out_dir: Path, ratios: tuple[float, float, float], seed: int) -> list[dict]:
    strata_map = {}
    for row in rows:
        key = (row["subject"], row["label"])
        strata_map.setdefault(key, []).append(dict(row))
    strata = sorted(strata_map.items(), key=lambda item: item[0])
    counts = allocate_counts(strata, ratios, seed)
    rng = random.Random(seed)
    split_rows = []

    for key, group_rows in strata:
        group_rows = sorted(group_rows, key=lambda row: int(row["trial_num"]))
        c = counts[key]
        test_rows, remaining = choose_rows_by_ambient(group_rows, c["test"], rng)
        val_rows, train_rows = choose_rows_by_ambient(remaining, c["val"], rng)
        for split, selected in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            for row in selected:
                row = dict(row)
                row["mixed_split"] = split
                split_rows.append(row)

    split_rows.sort(key=lambda row: (row["mixed_split"], row["subject"], row["label"], int(row["trial_num"])))
    out_path = out_dir / "split_manifest_mixed_75_15_10.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(split_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)
    return split_rows


def build_manifest_split(rows: list[dict], out_dir: Path) -> list[dict]:
    split_rows = []
    allowed = {"train", "val", "test"}
    for row in rows:
        split = row.get("split_mode1", "")
        if split not in allowed:
            raise ValueError(f"Unexpected split_mode1={split!r} in manifest row: {row}")
        row = dict(row)
        row["mixed_split"] = split
        split_rows.append(row)

    split_rows.sort(key=lambda row: (row["mixed_split"], row["subject"], row["label"], int(row["trial_num"])))
    out_path = out_dir / "split_manifest_from_manifest_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(split_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)
    return split_rows


def compute_norm(sequences: list[Sequence]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([seq.x for seq in sequences], axis=0)
    y = np.concatenate([seq.y for seq in sequences], axis=0)
    x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (y.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return x_mean, x_std, y_mean, y_std


class WindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[Sequence],
        window: int,
        step: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.items = []
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        for seq_idx, seq in enumerate(sequences):
            n = len(seq.x)
            for start in range(0, max(1, n - window + 1), step):
                end = start + window
                if end <= n:
                    self.items.append((seq_idx, start, end))
        if not self.items:
            raise RuntimeError("No training windows were generated. Try a smaller --window.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        seq_idx, start, end = self.items[idx]
        seq = self.sequences[seq_idx]
        x = (seq.x[start:end] - self.x_mean) / self.x_std
        y = (seq.y[start:end] - self.y_mean) / self.y_std
        return torch.from_numpy(x.T).float(), torch.from_numpy(y).float()


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PoseTCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden, 1, dropout),
            ResidualTCNBlock(hidden, 2, dropout),
            ResidualTCNBlock(hidden, 4, dropout),
            ResidualTCNBlock(hidden, 8, dropout),
            ResidualTCNBlock(hidden, 16, dropout),
        )
        self.head = nn.Conv1d(hidden, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.blocks(self.stem(x)))
        return y.transpose(1, 2)


class HumanPoseLoss(nn.Module):
    def __init__(self, velocity_weight: float = 0.2, bone_weight: float = 0.15) -> None:
        super().__init__()
        self.coord = nn.SmoothL1Loss()
        self.velocity_weight = velocity_weight
        self.bone_weight = bone_weight
        self.edge_idx = [(JOINT_INDEX[a], JOINT_INDEX[b]) for a, b in EDGES]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.coord(pred, target)
        if pred.shape[1] > 1:
            loss = loss + self.velocity_weight * self.coord(
                pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1]
            )
        pred_j = pred.reshape(pred.shape[0], pred.shape[1], len(JOINTS), 3)
        tgt_j = target.reshape(target.shape[0], target.shape[1], len(JOINTS), 3)
        bone_losses = []
        for ia, ib in self.edge_idx:
            pred_len = torch.linalg.norm(pred_j[:, :, ia] - pred_j[:, :, ib], dim=-1)
            tgt_len = torch.linalg.norm(tgt_j[:, :, ia] - tgt_j[:, :, ib], dim=-1)
            bone_losses.append(torch.mean(torch.abs(pred_len - tgt_len)))
        return loss + self.bone_weight * torch.stack(bone_losses).mean()


def train_model(train_seqs: list[Sequence], val_seqs: list[Sequence], args, out_dir: Path):
    x_mean, x_std, y_mean, y_std = compute_norm(train_seqs)
    train_ds = WindowDataset(train_seqs, args.window, args.step, x_mean, x_std, y_mean, y_std)
    val_ds = WindowDataset(val_seqs, args.window, args.step, x_mean, x_std, y_mean, y_std)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = PoseTCN(train_seqs[0].x.shape[1], train_seqs[0].y.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = HumanPoseLoss(args.velocity_loss_weight, args.bone_loss_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    history = []
    best_val = math.inf
    best_path = out_dir / "best_model.pt"
    patience = 0

    print(f"[train] windows train={len(train_ds)} val={len(val_ds)} device={device} in_dim={train_seqs[0].x.shape[1]}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        lr = float(opt.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "joints": JOINTS,
                    "edges": EDGES,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "amp_dim": args.amp_dim,
                    "canonical_pose": "hip_centered_sequence_scale",
                },
                best_path,
            )
        else:
            patience += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | train {train_loss:.5f} | val {val_loss:.5f} "
                f"| best {best_val:.5f} | patience {patience}/{args.patience}"
            )
        if patience >= args.patience:
            print(f"[early-stop] epoch={epoch} best_val={best_val:.5f}")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint, device, hist_df


def smooth_prediction(pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(pred) < window:
        return pred.astype(np.float32)
    pad = window // 2
    padded = np.pad(pred, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty_like(pred, dtype=np.float32)
    for dim in range(pred.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out.astype(np.float32)


def predict_sequence(model: nn.Module, checkpoint: dict, device: torch.device, seq: Sequence, smooth: int) -> np.ndarray:
    x = (seq.x - checkpoint["x_mean"]) / checkpoint["x_std"]
    xb = torch.from_numpy(x.T[None]).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(xb).cpu().numpy()[0]
    pred = pred * checkpoint["y_std"] + checkpoint["y_mean"]
    return smooth_prediction(pred.astype(np.float32), smooth)


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_j = gt.reshape(len(gt), len(JOINTS), 3)
    pr_j = pred.reshape(len(pred), len(JOINTS), 3)
    diff = pr_j - gt_j
    per_joint = np.linalg.norm(diff, axis=-1)
    mae_joint = np.mean(np.abs(diff), axis=(0, 2))
    metrics = {
        "mae_all": float(np.mean(np.abs(diff))),
        "mpjpe_all": float(np.mean(per_joint)),
    }
    if len(gt_j) > 1:
        gt_v = gt_j[1:] - gt_j[:-1]
        pr_v = pr_j[1:] - pr_j[:-1]
        metrics["velocity_mpjpe"] = float(np.mean(np.linalg.norm(pr_v - gt_v, axis=-1)))
    else:
        metrics["velocity_mpjpe"] = 0.0
    bone_errors = []
    for a, b in EDGES:
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        gt_len = np.linalg.norm(gt_j[:, ia] - gt_j[:, ib], axis=-1)
        pr_len = np.linalg.norm(pr_j[:, ia] - pr_j[:, ib], axis=-1)
        bone_errors.append(np.mean(np.abs(pr_len - gt_len)))
    metrics["bone_length_mae"] = float(np.mean(bone_errors))
    for idx, joint in enumerate(JOINTS):
        metrics[f"mpjpe_{joint}"] = float(np.mean(per_joint[:, idx]))
        metrics[f"mae_{joint}"] = float(mae_joint[idx])
    return metrics


def save_prediction(seq: Sequence, pred: np.ndarray, out_path: Path) -> None:
    pred_df = pd.DataFrame(pred, columns=[f"pred_{col}" for col in POSE_COLS])
    gt_df = pd.DataFrame(seq.y, columns=[f"gt_{col}" for col in POSE_COLS])
    out = pd.concat([pred_df, gt_df], axis=1)
    out.insert(0, "timestamp_sec", seq.time)
    out.insert(0, "ambient", seq.meta["ambient"])
    out.insert(0, "trial", seq.meta["trial"])
    out.insert(0, "label", seq.meta["label"])
    out.insert(0, "subject", seq.meta["subject"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def plot_training_history(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train", linewidth=2.0)
    ax.plot(history["epoch"], history["val_loss"], label="validation", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("human-pose loss")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("CSI-to-Pose Training Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, split_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"] == "test"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    split_summary = metrics.groupby("split")[["mae_all", "mpjpe_all", "velocity_mpjpe", "bone_length_mae"]].mean()
    split_summary.loc[["train", "val", "test"]].plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Average Metrics by Split")
    axes[0, 0].set_ylabel("canonical skeleton units")
    axes[0, 0].grid(axis="y", alpha=0.25)

    label_summary = test.groupby("label")["mpjpe_all"].mean().sort_values()
    label_summary.plot(kind="barh", ax=axes[0, 1], color="#4374b3")
    axes[0, 1].set_title("Test MPJPE by Label")
    axes[0, 1].set_xlabel("mean per-joint position error")
    axes[0, 1].grid(axis="x", alpha=0.25)

    joint_cols = [f"mpjpe_{joint}" for joint in JOINTS]
    joint_summary = test[joint_cols].mean().rename(index={f"mpjpe_{joint}": joint for joint in JOINTS})
    joint_summary.sort_values().plot(kind="barh", ax=axes[1, 0], color="#3f9f82")
    axes[1, 0].set_title("Test Error by Joint")
    axes[1, 0].set_xlabel("MPJPE")
    axes[1, 0].grid(axis="x", alpha=0.25)

    split_df = pd.DataFrame(split_rows)
    counts = split_df.groupby(["mixed_split", "subject"]).size().unstack(fill_value=0).loc[["train", "val", "test"]]
    counts.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["#5b8def", "#f4a261", "#5abf90"])
    axes[1, 1].set_title("Mixed Split Subject Balance")
    axes[1, 1].set_ylabel("paired clips")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / "metrics_overview.png", dpi=170)
    plt.close(fig)

    per_label_subject = test.pivot_table(index="label", columns="subject", values="mpjpe_all", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(per_label_subject.fillna(0).to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(per_label_subject.columns)), per_label_subject.columns)
    ax.set_yticks(np.arange(len(per_label_subject.index)), per_label_subject.index)
    ax.set_title("Test MPJPE Heatmap: Label x Subject")
    fig.colorbar(im, ax=ax, label="MPJPE")
    fig.tight_layout()
    fig.savefig(out_dir / "test_label_subject_heatmap.png", dpi=170)
    plt.close(fig)


def screen_points(pose: np.ndarray, x0: int, y0: int, width: int, height: int, scale: float | None = None):
    pts = pose.reshape(len(JOINTS), 3)
    xy = pts[:, :2]
    if scale is None:
        span_x = max(float(np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0])), 0.4)
        span_y = max(float(np.nanmax(xy[:, 1]) - np.nanmin(xy[:, 1])), 0.8)
        scale = min(width / span_x * 0.45, height / span_y * 0.45)
    cx = x0 + width // 2
    cy = y0 + int(height * 0.55)
    screen = np.zeros((len(JOINTS), 2), dtype=np.int32)
    screen[:, 0] = np.round(cx + xy[:, 0] * scale).astype(np.int32)
    screen[:, 1] = np.round(cy + xy[:, 1] * scale).astype(np.int32)
    return screen, scale


def draw_text(img, text: str, org: tuple[int, int], scale=0.55, color=(235, 238, 245), thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def joint_color_bgr(joint: str) -> tuple[int, int, int]:
    if joint in LEFT_JOINTS:
        return (230, 139, 55)
    if joint in RIGHT_JOINTS:
        return (66, 185, 244)
    return (230, 230, 230)


def draw_skeleton(
    img: np.ndarray,
    pose: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    scale: float,
    alpha: float = 1.0,
    ghost: bool = False,
) -> tuple[np.ndarray, float]:
    overlay = img.copy()
    pts, scale = screen_points(pose, x0, y0, width, height, scale)
    torso_names = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]
    torso = np.array([pts[JOINT_INDEX[name]] for name in torso_names], dtype=np.int32)
    torso_color = (86, 102, 123) if ghost else (80, 126, 166)
    cv2.fillPoly(overlay, [torso], torso_color)
    cv2.addWeighted(overlay, 0.22 if not ghost else 0.12, img, 0.78 if not ghost else 0.88, 0, img)

    line_alpha = 0.35 if ghost else alpha
    for a, b in EDGES:
        ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
        color = (132, 139, 150) if ghost else joint_color_bgr(a if a.startswith(("left_", "right_")) else b)
        line_width = 2 if ghost else 6
        cv2.line(img, tuple(pts[ia]), tuple(pts[ib]), color, line_width, cv2.LINE_AA)
        if not ghost:
            cv2.line(img, tuple(pts[ia]), tuple(pts[ib]), (20, 25, 31), 1, cv2.LINE_AA)

    for joint, idx in JOINT_INDEX.items():
        radius = 4 if ghost else (11 if joint == "head" else 7)
        color = (150, 150, 150) if ghost else joint_color_bgr(joint)
        cv2.circle(img, tuple(pts[idx]), radius, color, -1, cv2.LINE_AA)
        if not ghost:
            cv2.circle(img, tuple(pts[idx]), radius, (12, 16, 22), 1, cv2.LINE_AA)
    return pts, scale


def draw_motion_panel(
    img: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    idx: int,
    meta: dict,
    metrics: dict,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), (24, 29, 38), -1)
    cv2.line(img, (x0, y0), (x0, y0 + height), (69, 81, 100), 2)
    draw_text(img, "CSI-only 3D Skeleton Proxy", (x0 + 26, y0 + 42), 0.68, (245, 247, 252), 2)
    draw_text(img, f"{meta['subject']} | {meta['label']} | {meta['trial']} | {meta['ambient']}", (x0 + 26, y0 + 76), 0.5)
    draw_text(img, f"MPJPE {metrics['mpjpe_all']:.3f}   Bone {metrics['bone_length_mae']:.3f}", (x0 + 26, y0 + 106), 0.5, (180, 216, 255))

    pred_j = pred.reshape(len(pred), len(JOINTS), 3)
    gt_j = gt.reshape(len(gt), len(JOINTS), 3)
    if idx > 0:
        motion = np.linalg.norm(pred_j[idx] - pred_j[idx - 1], axis=-1)
    else:
        motion = np.zeros(len(JOINTS), dtype=np.float32)
    err = np.linalg.norm(pred_j[idx] - gt_j[idx], axis=-1)
    top = np.argsort(-motion)[:7]
    max_motion = max(float(np.max(motion)), 1e-4)

    draw_text(img, "Top joint movement this frame", (x0 + 26, y0 + 155), 0.55, (245, 247, 252), 1)
    base_y = y0 + 190
    for row, joint_idx in enumerate(top):
        joint = JOINTS[joint_idx]
        y = base_y + row * 42
        bar_w = int((width - 190) * float(motion[joint_idx]) / max_motion)
        color = joint_color_bgr(joint)
        draw_text(img, joint.replace("_", " "), (x0 + 26, y + 8), 0.46, (220, 226, 235), 1)
        cv2.rectangle(img, (x0 + 168, y - 8), (x0 + 168 + bar_w, y + 10), color, -1)
        draw_text(img, f"d {motion[joint_idx]:.3f}", (x0 + width - 105, y + 8), 0.43, (235, 238, 245), 1)
        draw_text(img, f"err {err[joint_idx]:.3f}", (x0 + width - 105, y + 28), 0.38, (168, 181, 199), 1)

    draw_text(img, "blue=left limbs   orange=right limbs   gray=GT ghost", (x0 + 26, y0 + height - 36), 0.43, (165, 178, 198), 1)


def render_reconstruction_video(seq: Sequence, pred: np.ndarray, metrics: dict, out_path: Path, max_frames: int, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    main_w = 850
    panel_w = width - main_w
    frame_count = min(max_frames, len(seq.time))
    idxs = np.linspace(0, len(seq.time) - 1, frame_count).astype(int)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    pred_j = pred.reshape(len(pred), len(JOINTS), 3)
    gt_j = seq.y.reshape(len(seq.y), len(JOINTS), 3)
    all_xy = np.concatenate([pred_j[:, :, :2], gt_j[:, :, :2]], axis=0).reshape(-1, 2)
    span_x = max(float(np.nanmax(all_xy[:, 0]) - np.nanmin(all_xy[:, 0])), 0.4)
    span_y = max(float(np.nanmax(all_xy[:, 1]) - np.nanmin(all_xy[:, 1])), 0.8)
    scale = min(main_w / span_x * 0.45, height / span_y * 0.42)

    trail = []
    for frame_no, idx in enumerate(idxs):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (18, 22, 29)
        cv2.rectangle(img, (0, 0), (main_w, height), (16, 20, 27), -1)
        cv2.line(img, (70, 622), (main_w - 55, 622), (55, 68, 82), 2, cv2.LINE_AA)

        hip = 0.5 * (pred_j[idx, JOINT_INDEX["left_hip"], :2] + pred_j[idx, JOINT_INDEX["right_hip"], :2])
        trail.append(hip)
        if len(trail) > 35:
            trail.pop(0)
        if len(trail) > 1:
            trail_pts = []
            for h in trail:
                screen, _ = screen_points(np.tile(np.array([h[0], h[1], 0.0], dtype=np.float32), len(JOINTS)), 0, 0, main_w, height, scale)
                trail_pts.append(screen[0])
            for a, b in zip(trail_pts[:-1], trail_pts[1:]):
                cv2.line(img, tuple(a), tuple(b), (72, 147, 101), 2, cv2.LINE_AA)

        draw_skeleton(img, seq.y[idx], 0, 0, main_w, height, scale, ghost=True)
        draw_skeleton(img, pred[idx], 0, 0, main_w, height, scale, ghost=False)
        draw_text(img, "Predicted from CSI only", (42, 48), 0.78, (246, 248, 252), 2)
        draw_text(img, f"time {seq.time[idx]:.2f}s   frame {frame_no + 1}/{len(idxs)}", (42, 82), 0.52, (177, 190, 210), 1)
        draw_motion_panel(img, pred, seq.y, idx, seq.meta, metrics, main_w, 0, panel_w, height)
        writer.write(img)
    writer.release()


def make_contact_sheet(seq: Sequence, pred: np.ndarray, metrics: dict, out_path: Path, frames: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(seq.time) - 1, min(frames, len(seq.time))).astype(int)
    fig, axes = plt.subplots(len(idxs), 2, figsize=(8, 2.6 * len(idxs)), constrained_layout=True)
    if len(idxs) == 1:
        axes = np.array([axes])
    for row, idx in enumerate(idxs):
        for col, (title, pose) in enumerate([("GT", seq.y[idx]), ("CSI Prediction", pred[idx])]):
            ax = axes[row, col]
            pts = pose.reshape(len(JOINTS), 3)
            ax.set_aspect("equal")
            ax.set_xlim(-0.65, 0.65)
            ax.set_ylim(0.78, -0.75)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.12)
            for a, b in EDGES:
                ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
                ax.plot([pts[ia, 0], pts[ib, 0]], [pts[ia, 1], pts[ib, 1]], linewidth=2.6)
            ax.scatter(pts[:, 0], pts[:, 1], s=18, color="#20252e")
            ax.set_title(f"{title} {seq.time[idx]:.1f}s", fontsize=9)
    fig.suptitle(
        f"{seq.meta['subject']} {seq.meta['label']} {seq.meta['trial']} | MPJPE {metrics['mpjpe_all']:.3f}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def summarize_split(split_rows: list[dict]) -> pd.DataFrame:
    return (
        pd.DataFrame(split_rows)
        .groupby(["mixed_split", "subject", "label"])
        .size()
        .rename("clips")
        .reset_index()
        .sort_values(["mixed_split", "subject", "label"])
    )


def load_sequences_by_split(split_rows: list[dict], data_root: Path, cache_dir: Path, amp_dim: int) -> dict[str, list[Sequence]]:
    out = {"train": [], "val": [], "test": []}
    total = len(split_rows)
    for idx, row in enumerate(split_rows, start=1):
        seq = load_sequence(row, data_root, cache_dir, amp_dim)
        out[row["mixed_split"]].append(seq)
        if idx == 1 or idx % 75 == 0 or idx == total:
            print(f"[load] {idx}/{total} sequences")
    return out


def evaluate_and_save(
    model: nn.Module,
    checkpoint: dict,
    device: torch.device,
    sequences: dict[str, list[Sequence]],
    out_dir: Path,
    smooth: int,
    save_prediction_splits: set[str],
) -> tuple[pd.DataFrame, dict[str, tuple[Sequence, np.ndarray, dict]]]:
    rows = []
    rendered_candidates: dict[str, tuple[Sequence, np.ndarray, dict]] = {}
    for split, seqs in sequences.items():
        for seq in seqs:
            pred = predict_sequence(model, checkpoint, device, seq, smooth)
            metrics = compute_metrics(seq.y, pred)
            row = {**seq.meta, **metrics}
            rows.append(row)
            if split in save_prediction_splits:
                name = f"{seq.meta['subject']}_{seq.meta['label']}_{seq.meta['trial']}_prediction.csv"
                save_prediction(seq, pred, out_dir / "predictions" / split / seq.meta["label"] / name)
            if split == "test":
                current = rendered_candidates.get(seq.meta["label"])
                if current is None or metrics["mpjpe_all"] < current[2]["mpjpe_all"]:
                    rendered_candidates[seq.meta["label"]] = (seq, pred, metrics)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    return metrics_df, rendered_candidates


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train mixed-subject CSI-to-Pose model and render reconstructions.")
    parser.add_argument("--data-root", default=None, help="CSI-to-Pose data root. Defaults to this CSI-to-Pose directory.")
    parser.add_argument("--out-root", default=None, help="Output directory. Defaults to outputs/manifest_split_all_labels.")
    parser.add_argument(
        "--split-policy",
        choices=["manifest", "mixed_ratio"],
        default="manifest",
        help="manifest uses split_mode1 from split_manifest/manifest_full.csv; mixed_ratio rebuilds a 7.5:1.5:1 split.",
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.2)
    parser.add_argument("--bone-loss-weight", type=float, default=0.15)
    parser.add_argument("--amp-dim", type=int, default=128)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--render-per-label", type=int, default=1)
    parser.add_argument("--max-render-frames", type=int, default=180)
    parser.add_argument("--render-fps", type=int, default=12)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root) if args.data_root else repo_root
    out_dir = Path(args.out_root) if args.out_root else repo_root / "outputs" / "manifest_split_all_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    rows = read_manifest(data_root)
    if args.split_policy == "manifest":
        split_rows = build_manifest_split(rows, out_dir)
        split_label = "manifest_full.csv split_mode1"
    else:
        split_rows = build_mixed_split(rows, out_dir, (0.75, 0.15, 0.10), args.seed)
        split_label = "7.5:1.5:1 mixed by subject+label"
    split_summary = summarize_split(split_rows)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)

    split_counts = pd.DataFrame(split_rows)["mixed_split"].value_counts().to_dict()
    print(f"[split] {split_label}")
    print(json.dumps(split_counts, indent=2, sort_keys=True))

    sequences = load_sequences_by_split(split_rows, data_root, cache_dir, args.amp_dim)
    model, checkpoint, device, history = train_model(sequences["train"], sequences["val"], args, out_dir)

    plot_training_history(history, out_dir / "training_history.png")
    metrics_df, render_candidates = evaluate_and_save(
        model,
        checkpoint,
        device,
        sequences,
        out_dir,
        args.smooth,
        save_prediction_splits={"test"},
    )
    plot_metrics(metrics_df, split_rows, out_dir / "plots")

    rendered = []
    if args.render_per_label > 0:
        for label, (seq, pred, metrics) in sorted(render_candidates.items()):
            safe_name = f"{seq.meta['subject']}_{label}_{seq.meta['trial']}"
            video_path = out_dir / "reconstruction_videos" / label / f"{safe_name}_csi_reconstruction.mp4"
            sheet_path = out_dir / "contact_sheets" / label / f"{safe_name}_contact_sheet.png"
            render_reconstruction_video(seq, pred, metrics, video_path, args.max_render_frames, args.render_fps)
            make_contact_sheet(seq, pred, metrics, sheet_path)
            rendered.append({"label": label, "video": str(video_path), "contact_sheet": str(sheet_path), **seq.meta, **metrics})
            print(f"[render] {label}: {video_path}")

    test_metrics = metrics_df[metrics_df["split"] == "test"]
    report = {
        "project": "NotiFi CSI-to-Pose mixed-subject all-label experiment",
        "split_policy": args.split_policy,
        "split_source": "split_manifest/manifest_full.csv split_mode1" if args.split_policy == "manifest" else "script-generated 7.5:1.5:1 mixed split",
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "total_clips": len(split_rows),
        "split_counts": split_counts,
        "subjects": sorted(pd.DataFrame(split_rows)["subject"].unique().tolist()),
        "labels": sorted(pd.DataFrame(split_rows)["label"].unique().tolist()),
        "test_mae_all": float(test_metrics["mae_all"].mean()),
        "test_mpjpe_all": float(test_metrics["mpjpe_all"].mean()),
        "test_velocity_mpjpe": float(test_metrics["velocity_mpjpe"].mean()),
        "test_bone_length_mae": float(test_metrics["bone_length_mae"].mean()),
        "rendered": rendered,
    }
    with open(out_dir / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[done]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
