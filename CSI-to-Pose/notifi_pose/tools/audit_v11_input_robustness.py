"""Validation-only CSI perturbation audit for a locked V11 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .. import contract as C
from ..trainer import set_seed
from .calibrate_v11_residual_temporal import _build_model
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


class PerturbedDataset(Dataset):
    """Apply deterministic hardware/timing perturbations to validation CSI."""

    def __init__(self, target: Dataset, mode: str):
        self.target = target
        self.mode = mode

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict:
        sample = self.target[index]
        sample = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        csi = sample["csi"]
        mask = sample["link_mask"]
        if self.mode.startswith("drop_link_burst"):
            alive = torch.nonzero(mask.any(0), as_tuple=False).flatten()
            if len(alive) >= 2:
                selected = int(alive[index % len(alive)])
                width = len(mask) // 2
                if self.mode == "drop_link_burst_early":
                    start = 0
                elif self.mode == "drop_link_burst_late":
                    start = len(mask) - width
                elif self.mode == "drop_link_burst_shifted":
                    start = (index * 37) % (len(mask) - width + 1)
                elif self.mode in {"drop_link_burst", "drop_link_burst_middle"}:
                    start = (len(mask) - width) // 2
                else:
                    raise ValueError(f"unknown perturbation: {self.mode}")
                stop = start + width
                mask[start:stop, selected] = False
                csi[start:stop, selected] = 0.0
        elif self.mode == "drop_one_link" or self.mode.startswith("drop_link_"):
            alive = torch.nonzero(mask.any(0), as_tuple=False).flatten()
            if len(alive) >= 2:
                if self.mode == "drop_one_link":
                    selected = int(alive[index % len(alive)])
                else:
                    selected = int(self.mode.rsplit("_", 1)[1])
                if selected in alive.tolist():
                    mask[:, selected] = False
                    csi[:, selected] = 0.0
        elif self.mode == "subcarrier_band":
            width = min(12, csi.shape[2])
            start = (csi.shape[2] - width) // 2
            csi[:, :, start:start + width] = 0.0
        elif self.mode in {
            "gain_phase", "gain_phase_alt", "gain_phase_trial",
        }:
            if self.mode == "gain_phase":
                scale_values = (0.82, 1.18, 0.93)
                phase_values = (0.22, -0.17, 0.11)
            elif self.mode == "gain_phase_alt":
                scale_values = (1.12, 0.78, 1.24)
                phase_values = (-0.28, 0.13, 0.31)
            else:
                offset = (index % 11 - 5) / 5.0
                scale_values = (
                    1.0 + 0.18 * offset,
                    1.0 - 0.14 * offset,
                    1.0 + 0.10 * offset,
                )
                phase_values = (
                    0.25 * offset,
                    -0.20 * offset,
                    0.15 * offset,
                )
            scale = csi.new_tensor(scale_values)[:csi.shape[1]]
            phase = csi.new_tensor(phase_values)[:csi.shape[1]]
            if C.CSI_REPRESENTATION == "amp_phase":
                frequency = torch.linspace(
                    -1.0, 1.0, csi.shape[2],
                    dtype=csi.dtype, device=csi.device,
                )
                curvature = frequency.square() - frequency.square().mean()
                ripple = phase[:, None] * curvature[None]
                csi[..., 0] *= scale[None, :, None]
                csi[..., 1] += ripple[None]
            else:
                cosine, sine = torch.cos(phase), torch.sin(phase)
                real, imag = csi[..., 0].clone(), csi[..., 1].clone()
                csi[..., 0] = scale[None, :, None] * (
                    real * cosine[None, :, None] - imag * sine[None, :, None]
                )
                csi[..., 1] = scale[None, :, None] * (
                    real * sine[None, :, None] + imag * cosine[None, :, None]
                )
        elif self.mode == "time_jitter_2":
            shift = 2 if index % 2 == 0 else -2
            csi = torch.roll(csi, shift, dims=0)
            mask = torch.roll(mask, shift, dims=0)
            if shift > 0:
                csi[:shift] = 0.0
                mask[:shift] = False
            else:
                csi[shift:] = 0.0
                mask[shift:] = False
            sample["csi"] = csi
            sample["link_mask"] = mask
        elif self.mode != "clean":
            raise ValueError(f"unknown perturbation: {self.mode}")
        return sample


def _summary(metrics: dict) -> dict:
    keys = (
        "mpjpe_m", "dynamic_mpjpe_m", "root_error_m", "pose_speed_ratio",
        "danger_mpjpe_m", "danger_distal_mpjpe_m",
        "danger_endpoint_mpjpe_m", "danger_pose_mpjpe_m",
        "danger_pose_distal_mpjpe_m", "danger_pose_endpoint_mpjpe_m",
        "danger_speed_correlation",
    )
    return {key: metrics[key] for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = json.loads(args.calibration.read_text(encoding="utf-8"))
    if locked.get("protocol") != args.exp:
        raise RuntimeError("calibration protocol mismatch")
    if locked.get("test_used_for_selection") is not False:
        raise RuntimeError("test split was not proven sealed")
    source = locked["source"]
    args.pose_strength = float(source["pose_strength"])
    args.root_strength = float(source["root_strength"])
    args.bone_blend = float(source["bone_blend"])
    args.bone_symmetric = bool(source["bone_symmetric"])

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
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
    model.eval()

    modes = ("clean", "drop_one_link", "subcarrier_band", "gain_phase",
             "time_jitter_2")
    results = {}
    for mode in modes:
        pose_loader = DataLoader(
            PerturbedDataset(loaders["val"].dataset, mode),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
        )
        class_loader = DataLoader(
            PerturbedDataset(loaders["val_class"].dataset, mode),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
        )
        trajectory = evaluate_trajectory(
            model, pose_loader, device, args.max_shift
        )
        classification = evaluate_classification(
            model, class_loader, device,
            float(source.get("danger_logit_bias", 0.0)),
        )
        results[mode] = {
            "trajectory": _summary(trajectory),
            "class_accuracy": classification["class"]["accuracy"],
            "class_macro_f1": classification["class"]["macro_f1"],
            "risk_accuracy": classification["risk"]["accuracy"],
            "risk_macro_f1": classification["risk"]["macro_f1"],
            "danger_recall": classification["risk"]["danger_recall"],
            "safe_to_danger": classification["risk"]["safe_to_danger"],
        }
    clean = results["clean"]["trajectory"]
    for mode in modes[1:]:
        metrics = results[mode]["trajectory"]
        results[mode]["delta"] = {
            "mpjpe_m": metrics["mpjpe_m"] - clean["mpjpe_m"],
            "danger_mpjpe_m": (
                metrics["danger_mpjpe_m"] - clean["danger_mpjpe_m"]
            ),
        }
    report = {
        "run": "p2_v11_input_robustness_audit",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "calibration": str(args.calibration),
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
