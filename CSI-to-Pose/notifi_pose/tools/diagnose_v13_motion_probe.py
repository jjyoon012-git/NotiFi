"""Probe motion information in locked V12 temporal features on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..trainer import set_seed
from ..hybrid_v10 import build_residual_hybrid
from .calibrate_v11_residual_temporal import _checked_checkpoint
from .evaluate_sealed import make_model
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import make_loaders


def _ridge_fit(features: torch.Tensor, targets: torch.Tensor,
               ridge: float) -> dict:
    mean = features.mean(0)
    scale = features.std(0).clamp_min(1e-4)
    normalized = (features - mean) / scale
    design = torch.cat((
        normalized, torch.ones(len(normalized), 1, dtype=normalized.dtype)
    ), dim=1)
    identity = torch.eye(design.shape[1], dtype=design.dtype)
    identity[-1, -1] = 0.0
    weight = torch.linalg.solve(
        design.T @ design + float(ridge) * identity,
        design.T @ targets,
    )
    return {"mean": mean, "scale": scale, "weight": weight}


def _ridge_predict(probe: dict, features: torch.Tensor) -> torch.Tensor:
    normalized = (features - probe["mean"]) / probe["scale"]
    design = torch.cat((
        normalized, torch.ones(len(normalized), 1, dtype=normalized.dtype)
    ), dim=1)
    return design @ probe["weight"]


def _regression_metrics(predicted: torch.Tensor,
                        target: torch.Tensor) -> dict:
    residual = (predicted - target).square().sum(0)
    total = (target - target.mean(0)).square().sum(0).clamp_min(1e-8)
    r2 = 1.0 - residual / total
    left = predicted - predicted.mean(0)
    right = target - target.mean(0)
    correlation = (left * right).sum(0) / (
        left.square().sum(0).sqrt() * right.square().sum(0).sqrt()
    ).clamp_min(1e-8)
    return {
        "r2": [float(value) for value in r2],
        "r2_mean": float(r2.mean()),
        "correlation": [float(value) for value in correlation],
        "correlation_mean": float(correlation.mean()),
        "mae": [float(value) for value in (predicted - target).abs().mean(0)],
    }


@torch.no_grad()
def _extract(model, dataset, device: str, batch_size: int,
             lag: int, frames_per_trial: int, seed: int) -> dict:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    rng = np.random.default_rng(seed)
    frame_feature = []
    root_velocity = []
    pose_speed = []
    trial_feature = []
    root_displacement = []
    direct_root_velocity = []
    direct_pose_speed = []
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        feature = output["temporal_features_v10"].float().cpu()
        valid = batch["valid"].bool()
        interval = valid[:, lag:] & valid[:, :-lag]
        scale = C.TARGET_FPS / lag
        target_root_velocity = (
            batch["root"][:, lag:] - batch["root"][:, :-lag]
        ) * scale
        target_pose_speed = torch.linalg.vector_norm(
            (batch["pose_rel"][:, lag:] - batch["pose_rel"][:, :-lag]) * scale,
            dim=-1,
        ).mean(-1, keepdim=True)
        for item in range(len(feature)):
            positions = torch.nonzero(interval[item], as_tuple=False).flatten()
            if len(positions) > frames_per_trial:
                selected = rng.choice(
                    positions.numpy(), size=frames_per_trial, replace=False
                )
                positions = torch.from_numpy(np.sort(selected)).long()
            if len(positions):
                frame_feature.append(feature[item, positions + lag])
                root_velocity.append(target_root_velocity[item, positions])
                pose_speed.append(target_pose_speed[item, positions])
                if "root_velocity_observation_v13" in output:
                    direct_root_velocity.append(
                        output["root_velocity_observation_v13"][
                            item, positions + lag
                        ].float().cpu()
                    )
                    direct_pose_speed.append(
                        output["pose_speed_observation_v13"][
                            item, positions + lag
                        ].float().cpu()[:, None]
                    )

            frames = torch.nonzero(valid[item], as_tuple=False).flatten()
            if len(frames):
                trial_feature.append(feature[item, frames].mean(0, keepdim=True))
                displacement = (
                    batch["root"][item, frames[-1]]
                    - batch["root"][item, frames[0]]
                )
                root_displacement.append(displacement[None])
    values = {
        "frame_feature": torch.cat(frame_feature),
        "root_velocity": torch.cat(root_velocity),
        "pose_speed": torch.cat(pose_speed),
        "trial_feature": torch.cat(trial_feature),
        "root_displacement": torch.cat(root_displacement),
    }
    if direct_root_velocity:
        values["direct_root_velocity"] = torch.cat(direct_root_velocity)
        values["direct_pose_speed"] = torch.cat(direct_pose_speed)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument(
        "--hybrid-checkpoint", type=Path, default=None,
        help="probe one validation-trained hybrid instead of the locked ensemble",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--frames-per-trial", type=int, default=48)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, _ = make_loaders(args, device)
    train, validation, _ = datasets
    if args.hybrid_checkpoint is None:
        root_lock = _read_locked(args.root_calibration, args.exp)
        class_lock = _read_locked(args.classification_calibration, args.exp)
        model, configuration = build_locked_model(
            args, device, root_lock, class_lock
        )
    else:
        p2 = torch.load(
            args.p2_checkpoint, map_location=device, weights_only=False
        )
        checkpoint = _checked_checkpoint(
            args.hybrid_checkpoint, device, args.exp
        )
        model = build_residual_hybrid(
            make_model(p2, device), checkpoint.get("residual_decoder", "subcarrier")
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.set_calibration(0.0, 1.0, 0.0, 0.0)
        configuration = {
            "p2_checkpoint": str(args.p2_checkpoint),
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "residual_decoder": checkpoint.get("residual_decoder", "subcarrier"),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
    model.eval()
    train_values = _extract(
        model, train, device, args.batch_size, args.lag,
        args.frames_per_trial, args.seed,
    )
    validation_values = _extract(
        model, validation, device, args.batch_size, args.lag,
        args.frames_per_trial, args.seed + 1,
    )

    tasks = {
        "root_velocity": ("frame_feature", "root_velocity"),
        "pose_speed": ("frame_feature", "pose_speed"),
        "trial_root_displacement": ("trial_feature", "root_displacement"),
    }
    results = {}
    for name, (feature_key, target_key) in tasks.items():
        probe = _ridge_fit(
            train_values[feature_key], train_values[target_key], args.ridge
        )
        predicted = _ridge_predict(probe, validation_values[feature_key])
        results[name] = _regression_metrics(
            predicted, validation_values[target_key]
        )
        results[name]["train_samples"] = len(train_values[feature_key])
        results[name]["validation_samples"] = len(
            validation_values[feature_key]
        )
    if "direct_root_velocity" in validation_values:
        results["direct_motion_head"] = {
            "root_velocity": _regression_metrics(
                validation_values["direct_root_velocity"],
                validation_values["root_velocity"],
            ),
            "pose_speed": _regression_metrics(
                validation_values["direct_pose_speed"],
                validation_values["pose_speed"],
            ),
        }

    report = {
        "run": "v13_v12_motion_feature_probe",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "lag": args.lag,
        "frames_per_trial": args.frames_per_trial,
        "ridge": args.ridge,
        "configuration": configuration,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
