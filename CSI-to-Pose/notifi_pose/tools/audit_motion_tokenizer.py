"""Separate continuous-latent and quantized KP2-B tokenizer error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_tokens import (
    FactorizedMotionTokenizer,
    KinematicMotionTokenizer,
    trial_bone_lengths,
)
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import _aggregate_rows, _pose_rows


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, device: str) -> dict:
    model.eval()
    quantized_rows = []
    continuous_rows = []
    latent_error = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        pose = batch["pose_rel"].to(device)
        valid = batch["valid"].to(device).bool()
        encoded = model.encode(pose, valid)
        lengths = trial_bone_lengths(pose, valid)
        quantized = model.decode(
            encoded["quantized"], lengths, pose.shape[1], valid
        )["pose_rel"].cpu()
        continuous = model.decode(
            encoded["latent"], lengths, pose.shape[1], valid
        )["pose_rel"].cpu()
        quantized_rows.extend(_pose_rows(quantized, batch))
        continuous_rows.extend(_pose_rows(continuous, batch))
        selected = encoded["token_mask"]
        latent_error.append(torch.linalg.vector_norm(
            encoded["latent"][selected] - encoded["quantized"][selected], dim=-1
        ).cpu())
    return {
        "quantized": _aggregate_rows(quantized_rows),
        "continuous": _aggregate_rows(continuous_rows),
        "latent_quantization_l2": float(torch.cat(latent_error).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2b_motion_tokenizer" / "best_model.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    common = {
        "hidden": int(architecture["hidden"]),
        "code_dim": int(architecture["code_dim"]),
        "codes": int(architecture["codes"]),
        "dropout": float(architecture["dropout"]),
        "commitment": float(architecture["commitment"]),
        "downsample": int(architecture.get("downsample", 4)),
    }
    model = (
        FactorizedMotionTokenizer(
            **common, part_hidden=int(architecture.get("part_hidden", 96))
        )
        if architecture.get("kind") == "factorized"
        else KinematicMotionTokenizer(
            **common,
            quantizer_levels=int(architecture.get("quantizer_levels", 1)),
            continuous=architecture.get("kind") == "continuous",
        )
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    datasets = build_datasets(exp=args.exp, baseline="none")
    result = {
        "run": "KP2-B-TOKENIZER-EXP01-bottleneck-audit",
        "checkpoint": report_path(args.checkpoint),
        "validation": evaluate(
            model, pose_only(datasets["val"]), args.batch_size, device
        ),
        "test": evaluate(
            model, pose_only(datasets["test"]), args.batch_size, device
        ),
    }
    output = args.output or args.checkpoint.parent / "bottleneck_audit.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
