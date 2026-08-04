"""Prove shared-backbone V12 is output-equivalent on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..trainer import set_seed
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import make_loaders, move_batch


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

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    _, loaders = make_loaders(args, device)
    separate, separate_config = build_locked_model(
        args, device, root_lock, class_lock, share_backbone=False
    )
    shared, shared_config = build_locked_model(
        args, device, root_lock, class_lock, share_backbone=True
    )
    separate.eval()
    shared.eval()

    keys = ("pose_rel", "root", "class_logits", "risk_logits")
    maximum = {key: 0.0 for key in keys}
    examples = 0
    with torch.no_grad():
        for batch in loaders["val_class"]:
            batch = move_batch(batch, device)
            expected = separate(batch["csi"], batch["link_mask"])
            actual = shared(batch["csi"], batch["link_mask"])
            examples += len(batch["csi"])
            for key in keys:
                difference = float((expected[key] - actual[key]).abs().max())
                maximum[key] = max(maximum[key], difference)
    report = {
        "run": "p2_v12_shared_backbone_equivalence",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used": False,
        "examples": examples,
        "max_absolute_difference": maximum,
        "bitwise_equivalent": all(value == 0.0 for value in maximum.values()),
        "separate_parameters": sum(
            parameter.numel() for parameter in separate.parameters()
        ),
        "shared_parameters": sum(
            parameter.numel() for parameter in shared.parameters()
        ),
        "separate_configuration": separate_config,
        "shared_configuration": shared_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["bitwise_equivalent"]:
        raise SystemExit("shared and separate V12 outputs differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
