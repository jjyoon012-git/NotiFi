"""Evaluate a checkpoint on the sealed yja E02 CSI-only pose set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import PoseDataset, build_datasets
from ..nets import build_model


def make_model(checkpoint: dict, device: str):
    cfg = checkpoint["cfg"]
    kwargs = {
        "hidden": cfg["hidden"],
        "n_blocks": cfg["n_blocks"],
        "dropout": cfg["dropout"],
    }
    if cfg["arch"] in {
        "graphformer", "robust_graphformer", "impact_graphformer", "latent_flow"
    }:
        kwargs.update(
            heads=cfg.get("heads", 4), graph_blocks=cfg.get("graph_blocks", 2),
            decoder=cfg.get("decoder", "tree"),
            domain_grl=cfg.get("domain_grl", 0.2),
        )
        if cfg["arch"] == "impact_graphformer":
            kwargs["refiner_joint_scale"] = cfg.get("refiner_joint_scale")
        elif cfg["arch"] == "latent_flow":
            kwargs.update(
                flow_steps=cfg.get("flow_steps", 4),
                flow_noise=cfg.get("flow_noise", 0.25),
            )
    elif cfg["arch"] == "v3":
        kwargs.update(
            heads=cfg.get("heads", 4), graph_blocks=cfg.get("graph_blocks", 2),
            frequency_tokens=cfg.get("frequency_tokens", 12),
            geometry_path=cfg.get("geometry_path"),
            domain_grl=cfg.get("domain_grl", 0.2),
        )
    else:
        kwargs.update(
            dilations=tuple(cfg["dilations"]),
            fusion=cfg["fusion"],
            film=cfg["film"],
        )
    model = build_model(cfg["arch"], **kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def speed(sequence: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(sequence[1:] - sequence[:-1], dim=-1) * C.TARGET_FPS


def finite_mean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def smooth_valid(values: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("smooth window must be odd")
    output = values.clone()
    radius = window // 2
    for item in range(len(values)):
        positions = torch.nonzero(valid[item], as_tuple=False).flatten()
        if len(positions) < window:
            continue
        selected = values[item, positions]
        shape = selected.shape
        signal = selected.reshape(len(selected), -1).transpose(0, 1)[None]
        signal = F.pad(signal, (radius, radius), mode="replicate")
        filtered = F.avg_pool1d(signal, kernel_size=window, stride=1)
        output[item, positions] = filtered[0].transpose(0, 1).reshape(shape)
    return output


def aggregate(rows: pd.DataFrame, key: str) -> dict:
    metrics = (
        "mpjpe_m", "dynamic_mpjpe_m", "distal_mpjpe_m", "head_mpjpe_m",
        "impact_mpjpe_m", "root_error_m", "pose_speed_ratio",
        "root_speed_ratio", "class_correct", "risk_correct",
    )
    return {
        str(name): {metric: finite_mean(group[metric]) for metric in metrics}
        for name, group in rows.groupby(key, sort=True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--fold", default="yja_E02")
    parser.add_argument("--dataset", choices=("sealed", "val", "test"), default="sealed")
    parser.add_argument(
        "--exp", choices=("single_split", "yja_holdout", "loso"),
        default="single_split",
        help="split protocol used when --dataset is val or test",
    )
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    baseline = checkpoint["cfg"].get("baseline", "none")
    if args.dataset == "sealed":
        selected = build_datasets(
            exp="sealed", fold=args.fold, baseline=baseline
        )["test"]
    else:
        selected = build_datasets(
            exp=args.exp, fold=args.fold if args.exp == "loso" else None,
            baseline=baseline,
        )[args.dataset]
    pose_positions = np.flatnonzero(selected.index.task.to_numpy() == C.TASK_POSE)
    dataset = PoseDataset(
        selected.rows[pose_positions], selected.cache, selected.link_ok,
        train=False, seed=0, baseline=selected.baseline,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = make_model(checkpoint, device)
    model.eval()

    rows = []
    joint_sum = np.zeros(C.N_JOINTS, dtype=np.float64)
    joint_count = np.zeros(C.N_JOINTS, dtype=np.int64)
    all_pose, all_root, all_valid = [], [], []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            csi = batch["csi"].to(device)
            link_mask = batch["link_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(csi, link_mask)
            pred_pose = output["pose_rel"].float().cpu()
            pred_root = output["root"].float().cpu()
            gt_pose = batch["pose_rel"].float()
            gt_root = batch["root"].float()
            valid = batch["valid"].bool()
            pred_pose = smooth_valid(pred_pose, valid, args.smooth_window)
            pred_root = smooth_valid(pred_root, valid, args.smooth_window)
            class_pred = output["class_logits"].argmax(-1).cpu()
            risk_pred = output["risk_logits"].argmax(-1).cpu()

            all_pose.append(pred_pose.numpy())
            all_root.append(pred_root.numpy())
            all_valid.append(valid.numpy())
            for item in range(len(csi)):
                meta = dataset.index.iloc[cursor]
                mask = valid[item]
                pose_error = torch.linalg.vector_norm(
                    pred_pose[item] - gt_pose[item], dim=-1
                )
                root_error = torch.linalg.vector_norm(
                    pred_root[item] - gt_root[item], dim=-1
                )
                pair = mask[1:] & mask[:-1]
                gt_pose_speed = speed(gt_pose[item]).mean(-1)
                pred_pose_speed = speed(pred_pose[item]).mean(-1)
                gt_root_speed = speed(gt_root[item])
                pred_root_speed = speed(pred_root[item])
                dynamic = pair & (gt_pose_speed > 0.25)
                impact = L.impact_window(
                    gt_pose[item:item + 1], gt_root[item:item + 1],
                    valid[item:item + 1], batch["risk_id"][item:item + 1],
                )[0]
                absolute_error = torch.linalg.vector_norm(
                    (pred_pose[item] + pred_root[item, :, None])
                    - (gt_pose[item] + gt_root[item, :, None]),
                    dim=-1,
                )

                valid_error = pose_error[mask]
                joint_sum += valid_error.sum(0).numpy()
                joint_count += int(mask.sum())
                gt_pose_motion = float(gt_pose_speed[pair].mean()) if pair.any() else np.nan
                gt_root_motion = float(gt_root_speed[pair].mean()) if pair.any() else np.nan
                rows.append({
                    "trial_id": meta.trial_id,
                    "subject": meta.subject,
                    "environment": meta.environment,
                    "scenario_id": meta.scenario_id,
                    "detail_label": meta.detail_label,
                    "risk": meta.risk,
                    "n_valid_frames": int(mask.sum()),
                    "mpjpe_m": float(valid_error.mean()),
                    "dynamic_mpjpe_m": (
                        float(pose_error[1:][dynamic].mean()) if dynamic.any() else np.nan
                    ),
                    "distal_mpjpe_m": float(
                        pose_error[mask][:, L.DISTAL_JOINTS].mean()
                    ),
                    "head_mpjpe_m": float(
                        pose_error[mask, C.JOINT_INDEX["head"]].mean()
                    ),
                    "impact_mpjpe_m": (
                        float(absolute_error[impact].mean()) if impact.any() else np.nan
                    ),
                    "root_error_m": float(root_error[mask].mean()),
                    "gt_pose_speed_mps": gt_pose_motion,
                    "pred_pose_speed_mps": (
                        float(pred_pose_speed[pair].mean()) if pair.any() else np.nan
                    ),
                    "pose_speed_ratio": (
                        float(pred_pose_speed[pair].mean()) / max(gt_pose_motion, 1e-6)
                        if pair.any() else np.nan
                    ),
                    "gt_root_speed_mps": gt_root_motion,
                    "pred_root_speed_mps": (
                        float(pred_root_speed[pair].mean()) if pair.any() else np.nan
                    ),
                    "root_speed_ratio": (
                        float(pred_root_speed[pair].mean()) / max(gt_root_motion, 1e-6)
                        if pair.any() else np.nan
                    ),
                    "class_pred": int(class_pred[item]),
                    "class_target": int(batch["class_id"][item]),
                    "class_correct": int(class_pred[item] == batch["class_id"][item]),
                    "risk_pred": int(risk_pred[item]),
                    "risk_target": int(batch["risk_id"][item]),
                    "risk_correct": int(risk_pred[item] == batch["risk_id"][item]),
                })
                cursor += 1

    frame = pd.DataFrame(rows)
    evaluation_name = (
        args.fold if args.dataset == "sealed"
        else f"{args.exp}_{args.fold or args.dataset}_{args.dataset}"
    )
    output_dir = args.output_dir or args.checkpoint.parent / (
        f"eval_{evaluation_name}_smooth{args.smooth_window}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "trial_metrics.csv", index=False, encoding="utf-8")
    joint_mpjpe = joint_sum / np.maximum(joint_count, 1)
    summary = {
        "checkpoint": str(args.checkpoint),
        "protocol": args.exp if args.dataset != "sealed" else "sealed",
        "split": args.dataset,
        "fold": args.fold,
        "calibration": baseline,
        "smooth_window": args.smooth_window,
        "subjects": sorted(frame.subject.unique().tolist()),
        "environments": sorted(frame.environment.unique().tolist()),
        "pose_trials": len(frame),
        "absence_excluded": int(len(selected) - len(dataset)),
        "overall": {
            "mpjpe_m": finite_mean(frame.mpjpe_m),
            "dynamic_mpjpe_m": finite_mean(frame.dynamic_mpjpe_m),
            "distal_mpjpe_m": finite_mean(frame.distal_mpjpe_m),
            "head_mpjpe_m": finite_mean(frame.head_mpjpe_m),
            "impact_mpjpe_m": finite_mean(frame.impact_mpjpe_m),
            "root_error_m": finite_mean(frame.root_error_m),
            "pose_speed_ratio": finite_mean(frame.pose_speed_ratio),
            "root_speed_ratio": finite_mean(frame.root_speed_ratio),
            "class_accuracy": finite_mean(frame.class_correct),
            "risk_accuracy": finite_mean(frame.risk_correct),
        },
        "by_risk": aggregate(frame, "risk"),
        "by_label": aggregate(frame, "detail_label"),
        "joint_mpjpe_m": dict(zip(C.JOINT_NAMES, joint_mpjpe.tolist())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        trial_id=dataset.index.trial_id.to_numpy(dtype=str),
        pose_rel=np.concatenate(all_pose),
        root=np.concatenate(all_root),
        valid=np.concatenate(all_valid),
    )
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
