"""Validation-only trial observability audit for the locked V12 model.

The perturbations preserve the dataset, targets, and split.  They only replace
CSI at loader time so the report can distinguish class-template prediction from
trial-specific motion reconstruction without opening the sealed test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..trainer import set_seed
from .audit_v11_input_robustness import _summary
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


class ObservabilityPerturbation(Dataset):
    """Deterministically remove trial identity or temporal motion from CSI."""

    MODES = (
        "clean",
        "same_site_class_shuffle",
        "same_class_shuffle",
        "global_shuffle",
        "time_reverse",
        "time_mean",
        "time_shift_30",
    )

    def __init__(self, target: Dataset, mode: str, seed: int = 17):
        if mode not in self.MODES:
            raise ValueError(f"unknown observability perturbation: {mode}")
        self.target = target
        self.mode = mode
        self.seed = int(seed)
        self.permutation = self._make_permutation() if "shuffle" in mode else None

    def __len__(self) -> int:
        return len(self.target)

    def _make_permutation(self) -> np.ndarray:
        index = self.target.index.reset_index(drop=True)
        rng = np.random.default_rng(self.seed)
        permutation = np.arange(len(index), dtype=np.int64)
        if self.mode == "same_site_class_shuffle":
            columns = ["subject", "environment", "class_id"]
        elif self.mode == "same_class_shuffle":
            columns = ["class_id"]
        else:
            columns = []

        groups = [np.arange(len(index), dtype=np.int64)]
        if columns:
            groups = [
                np.asarray(list(positions), dtype=np.int64)
                for positions in index.groupby(columns, sort=True).indices.values()
            ]
        for positions in groups:
            if len(positions) <= 1:
                continue
            donors = positions.copy()
            rng.shuffle(donors)
            if np.any(donors == positions):
                donors = np.roll(positions, 1)
            permutation[positions] = donors
        return permutation

    @staticmethod
    def _clone(sample: dict) -> dict:
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }

    def __getitem__(self, position: int) -> dict:
        sample = self._clone(self.target[position])
        if self.mode == "clean":
            return sample
        if self.permutation is not None:
            donor = self.target[int(self.permutation[position])]
            sample["csi"] = donor["csi"].clone()
            sample["link_mask"] = donor["link_mask"].clone()
            return sample

        csi = sample["csi"]
        link_mask = sample["link_mask"]
        active = link_mask.any(dim=1)
        active_positions = torch.nonzero(active, as_tuple=False).flatten()
        if len(active_positions) == 0:
            return sample
        start = int(active_positions[0])
        stop = int(active_positions[-1]) + 1

        if self.mode == "time_reverse":
            csi[start:stop] = torch.flip(csi[start:stop], dims=(0,))
            link_mask[start:stop] = torch.flip(link_mask[start:stop], dims=(0,))
        elif self.mode == "time_mean":
            for link in range(csi.shape[1]):
                valid = link_mask[start:stop, link]
                if valid.any():
                    mean = csi[start:stop, link][valid].mean(dim=0)
                    csi[start:stop, link][valid] = mean
        elif self.mode == "time_shift_30":
            width = stop - start
            if width > 1:
                shift = min(30, width - 1)
                csi[start:stop] = torch.roll(csi[start:stop], shift, dims=0)
                link_mask[start:stop] = torch.roll(
                    link_mask[start:stop], shift, dims=0
                )
        else:
            raise ValueError(f"unknown observability perturbation: {self.mode}")
        return sample


def _result(trajectory: dict, classification: dict) -> dict:
    return {
        "trajectory": _summary(trajectory),
        "class_accuracy": classification["class"]["accuracy"],
        "class_macro_f1": classification["class"]["macro_f1"],
        "risk_accuracy": classification["risk"]["accuracy"],
        "risk_macro_f1": classification["risk"]["macro_f1"],
        "danger_recall": classification["risk"]["danger_recall"],
        "danger_precision": classification["risk"]["danger_precision"],
        "safe_to_danger": classification["risk"]["safe_to_danger"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    model.eval()

    results = {}
    for mode in ObservabilityPerturbation.MODES:
        pose_loader = DataLoader(
            ObservabilityPerturbation(loaders["val"].dataset, mode, args.seed),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        class_loader = DataLoader(
            ObservabilityPerturbation(
                loaders["val_class"].dataset, mode, args.seed
            ),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        trajectory = evaluate_trajectory(
            model, pose_loader, device, args.max_shift
        )
        classification = evaluate_classification(
            model, class_loader, device, 0.0
        )
        results[mode] = _result(trajectory, classification)

    clean = results["clean"]
    for mode, metrics in results.items():
        if mode == "clean":
            continue
        metrics["delta"] = {
            key: metrics["trajectory"][key] - clean["trajectory"][key]
            for key in (
                "mpjpe_m",
                "dynamic_mpjpe_m",
                "root_error_m",
                "danger_mpjpe_m",
                "danger_endpoint_mpjpe_m",
            )
        }
        metrics["delta"].update({
            "class_accuracy": metrics["class_accuracy"] - clean["class_accuracy"],
            "risk_accuracy": metrics["risk_accuracy"] - clean["risk_accuracy"],
            "danger_recall": metrics["danger_recall"] - clean["danger_recall"],
        })

    report = {
        "run": "v13_observability_gate_v12",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "configuration": configuration,
        "modes": list(ObservabilityPerturbation.MODES),
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
