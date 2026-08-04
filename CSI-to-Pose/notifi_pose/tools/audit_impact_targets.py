"""Audit physical impact proxy targets before event-localizer training."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataio.dataset import DropoutConfig, build_datasets
from ..impact_event import physical_impact_targets
from ..seen_v2 import (
    INJURY_JOINT_NAMES, N_INJURY_JOINTS, injury_targets,
)
from .diagnose_observability import pose_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    dataset = pose_only(build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=7,
    )[args.split])
    counts = torch.zeros(N_INJURY_JOINTS, dtype=torch.long)
    legacy_counts = torch.zeros(N_INJURY_JOINTS, dtype=torch.long)
    frames = []
    edge_examples = []
    cursor = 0
    for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        target = physical_impact_targets(
            batch["pose_rel"], batch["root"], batch["valid"].bool(),
            batch["risk_id"],
        )
        selected = target["event_valid"]
        counts += torch.bincount(
            target["event_joint"][selected], minlength=N_INJURY_JOINTS
        )
        legacy = injury_targets(
            batch["pose_rel"], batch["root"], batch["valid"].bool(),
            batch["risk_id"],
        )
        legacy_selected = legacy["first_contact_valid"]
        legacy_counts += torch.bincount(
            legacy["first_contact"][legacy_selected], minlength=N_INJURY_JOINTS
        )
        frames.extend(target["event_frame"][selected].tolist())
        batch_trial_ids = dataset.index.iloc[
            cursor:cursor + len(batch["class_id"])
        ].trial_id.to_numpy()
        cursor += len(batch["class_id"])
        for trial_id, frame, joint in zip(
            batch_trial_ids[selected.cpu().numpy()].tolist(),
            target["event_frame"][selected].tolist(),
            target["event_joint"][selected].tolist(),
        ):
            if frame <= 10 or frame >= 290:
                edge_examples.append({
                    "trial_id": trial_id,
                    "frame": frame,
                    "joint": INJURY_JOINT_NAMES[joint],
                })
    quantiles = (
        np.quantile(frames, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
        if frames else []
    )
    print(json.dumps({
        "split": args.split,
        "event_trials": len(frames),
        "joint_counts": dict(zip(INJURY_JOINT_NAMES, counts.tolist())),
        "legacy_joint_counts": dict(
            zip(INJURY_JOINT_NAMES, legacy_counts.tolist())
        ),
        "frame_quantiles": quantiles,
        "edge_examples": edge_examples,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
