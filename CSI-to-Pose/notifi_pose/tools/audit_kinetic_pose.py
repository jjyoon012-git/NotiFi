"""Audit whether KP1-EXP01 actually uses temporal CSI evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..kinetic_pose import KineticPoseResidual
from ..quality import QualityWeightedDataset
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .train_kinetic_pose import (
    CoarsePoseStore,
    evaluate_strengths,
)


class SignalCounterfactualDataset(Dataset):
    """Change CSI only while preserving the target pose and coarse V13S row."""

    def __init__(self, target: Dataset, mode: str, seed: int):
        self.target = target
        self.mode = mode
        self.index = target.index
        self.permutation = np.arange(len(target), dtype=np.int64)
        if mode == "matched_shuffle":
            rng = np.random.default_rng(seed)
            columns = ["subject", "environment", "class_id"]
            groups = self.index.groupby(columns, sort=False).indices
            for positions in groups.values():
                positions = np.asarray(positions, dtype=np.int64)
                if len(positions) > 1:
                    shuffled = rng.permutation(positions)
                    if np.all(shuffled == positions):
                        shuffled = np.roll(shuffled, 1)
                    self.permutation[positions] = shuffled

    def __len__(self) -> int:
        return len(self.target)

    @staticmethod
    def _reverse(csi: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        csi = csi.clone()
        mask = mask.clone()
        for link in range(mask.shape[1]):
            positions = torch.nonzero(mask[:, link], as_tuple=False).flatten()
            if len(positions) > 1:
                csi[positions, link] = csi[positions.flip(0), link]
        return csi, mask

    @staticmethod
    def _mean(csi: torch.Tensor, mask: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        csi = csi.clone()
        for link in range(mask.shape[1]):
            positions = torch.nonzero(mask[:, link], as_tuple=False).flatten()
            if len(positions):
                mean = csi[positions, link].mean(0, keepdim=True)
                csi[positions, link] = mean
        return csi, mask

    def __getitem__(self, index: int) -> dict:
        sample = self.target[index]
        if self.mode == "matched_shuffle":
            signal = self.target[int(self.permutation[index])]
            sample["csi"] = signal["csi"]
            sample["link_mask"] = signal["link_mask"]
        elif self.mode == "temporal_reverse":
            sample["csi"], sample["link_mask"] = self._reverse(
                sample["csi"], sample["link_mask"]
            )
        elif self.mode == "temporal_mean":
            sample["csi"], sample["link_mask"] = self._mean(
                sample["csi"], sample["link_mask"]
            )
        elif self.mode != "clean":
            raise ValueError(f"unknown counterfactual mode {self.mode!r}")
        return sample


def delta(clean: dict, changed: dict) -> dict:
    keys = (
        "mpjpe_m", "dynamic_mpjpe_m", "high_motion_mpjpe_m",
        "danger_pose_mpjpe_m", "danger_distal_mpjpe_m",
        "danger_high_motion_mpjpe_m", "danger_endpoint_mpjpe_m",
        "speed_correlation", "danger_speed_correlation",
    )
    return {key: float(changed[key] - clean[key]) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_exp01_dynamic_pose" / "best_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_exp01_dynamic_pose" / "counterfactual.json",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["source"]
    p2_checkpoint = torch.load(
        C.PROJECT_ROOT / source["p2_checkpoint"],
        map_location=device, weights_only=False,
    )
    p2_model = make_model(p2_checkpoint, device)
    architecture = checkpoint["architecture"]
    model = KineticPoseResidual(
        None, p2_model.norm,
        hidden=int(architecture["hidden"]),
        temporal_layers=int(architecture["temporal_layers"]),
        max_delta=float(architecture["max_delta_m"]),
        condition_on_coarse=bool(architecture.get("condition_on_coarse", True)),
        activity_floor=float(architecture.get("activity_floor", 0.15)),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.set_activity_threshold(float(checkpoint.get("activity_threshold", 0.0)))
    del p2_model

    cached = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    store = CoarsePoseStore(cached["rows"], cached["pose"])
    selected = QualityWeightedDataset(
        pose_only(build_datasets(exp=args.exp, baseline="sub")["test"])
    )
    strength = float(checkpoint["residual_strength"])
    metrics = {}
    for mode in ("clean", "matched_shuffle", "temporal_reverse", "temporal_mean"):
        loader = DataLoader(
            SignalCounterfactualDataset(selected, mode, args.seed),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        metrics[mode] = evaluate_strengths(
            model, loader, [strength], device, store
        )[strength]
    result = {
        "run": f"{checkpoint['run']}-counterfactual",
        "protocol": args.exp,
        "split": "test",
        "checkpoint": report_path(args.checkpoint),
        "residual_strength": strength,
        "clean": metrics["clean"],
        "counterfactuals": {
            mode: {
                "metrics": metrics[mode],
                "delta_from_clean": delta(metrics["clean"], metrics[mode]),
            }
            for mode in ("matched_shuffle", "temporal_reverse", "temporal_mean")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
