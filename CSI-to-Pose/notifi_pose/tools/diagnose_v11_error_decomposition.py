"""Decompose V11 validation errors into anchor, trajectory and local pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..trainer import set_seed
from .calibrate_v11_residual_temporal import _build_model
from .evaluate_sealed import smooth_valid
from .train_seen_v4_trajectory import DISTAL_JOINTS, make_loaders


def _finite_mean(rows: list[dict], key: str) -> float:
    values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
    return float(values[torch.isfinite(values)].mean())


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {key: _finite_mean(rows, key) for key in rows[0]}


def _configure(args, locked: dict, device: str):
    source = locked["source"]
    args.pose_strength = float(source["pose_strength"])
    args.root_strength = float(source["root_strength"])
    args.bone_blend = float(source["bone_blend"])
    args.bone_symmetric = bool(source["bone_symmetric"])
    model = _build_model(args, device)
    model.base.set_calibration(
        int(locked["selected"]["window"]),
        float(locked["selected"]["blend"]),
        source.get("risk_adaptive", "none"),
        float(source.get("danger_logit_bias", 0.0)),
    )
    model.base.set_root_calibration(
        int(locked["selected"].get("root_window", 1)),
        float(locked["selected"].get("root_blend", 0.0)),
    )
    return model.eval()


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = json.loads(args.calibration.read_text(encoding="utf-8"))
    if locked.get("protocol") != args.exp:
        raise RuntimeError("calibration protocol mismatch")
    if locked.get("test_used_for_selection") is not False:
        raise RuntimeError("test split was not proven sealed")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model = _configure(args, locked, device).to(device)
    rows = []
    by_class: dict[int, list[dict]] = {}
    for batch in loaders["val"]:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
        )
        valid = batch["valid"].bool()
        pose = smooth_valid(output["pose_rel"].float().cpu(), valid, 5)
        root = smooth_valid(output["root"].float().cpu(), valid, 5)
        target_pose = batch["pose_rel"].float()
        target_root = batch["root"].float()
        for index in range(len(pose)):
            if int(batch["risk_id"][index]) != 2:
                continue
            mask = valid[index]
            frames = torch.nonzero(mask, as_tuple=False).flatten()
            if not len(frames):
                continue
            first = int(frames[0])
            endpoint = frames[-15:]
            local_error = torch.linalg.vector_norm(
                pose[index] - target_pose[index], dim=-1,
            )
            root_delta = root[index] - target_root[index]
            root_error = torch.linalg.vector_norm(root_delta, dim=-1)
            predicted_displacement = root[index] - root[index, first]
            target_displacement = target_root[index] - target_root[index, first]
            displacement_error = torch.linalg.vector_norm(
                predicted_displacement - target_displacement, dim=-1,
            )
            translation_aligned = (
                pose[index] + predicted_displacement[:, None]
                - target_pose[index] - target_displacement[:, None]
            )
            aligned_error = torch.linalg.vector_norm(
                translation_aligned, dim=-1,
            )
            row = {
                "local_pose_mpjpe_m": float(local_error[mask].mean()),
                "local_distal_mpjpe_m": float(
                    local_error[mask][:, DISTAL_JOINTS].mean()
                ),
                "local_endpoint_mpjpe_m": float(local_error[endpoint].mean()),
                "root_error_m": float(root_error[mask].mean()),
                "root_anchor_error_m": float(root_error[first]),
                "root_displacement_error_m": float(
                    displacement_error[mask].mean()
                ),
                "translation_aligned_mpjpe_m": float(
                    aligned_error[mask].mean()
                ),
                "translation_aligned_endpoint_mpjpe_m": float(
                    aligned_error[endpoint].mean()
                ),
                "root_x_mae_m": float(root_delta[mask, 0].abs().mean()),
                "root_y_mae_m": float(root_delta[mask, 1].abs().mean()),
                "root_z_mae_m": float(root_delta[mask, 2].abs().mean()),
            }
            rows.append(row)
            by_class.setdefault(int(batch["class_id"][index]), []).append(row)

    report = {
        "run": "p2_v11_error_decomposition",
        "protocol": args.exp,
        "split": "validation",
        "test_used": False,
        "danger_trials": len(rows),
        "danger": _aggregate(rows),
        "danger_by_class": {
            str(key): _aggregate(value) | {"trials": len(value)}
            for key, value in sorted(by_class.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
