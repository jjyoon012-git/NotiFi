"""Rebuild and evaluate a validation-locked V11 candidate.

The test split remains inaccessible unless --open-test is supplied. All model
and calibration settings are read from the validation artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..trainer import set_seed
from .calibrate_v11_residual_temporal import _build_model
from .evaluate_sealed import smooth_valid
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


def _same_path(left: str | Path, right: Path) -> bool:
    return Path(left).resolve() == right.resolve()


def _similarity_aligned_mpjpe(predicted: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
    """Per-frame Protocol-2 MPJPE after similarity Procrustes alignment."""
    predicted_mean = predicted.mean(1, keepdim=True)
    target_mean = target.mean(1, keepdim=True)
    left = predicted - predicted_mean
    right = target - target_mean
    covariance = left.transpose(1, 2) @ right
    u, singular, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    reflected = torch.linalg.det(rotation) < 0
    if reflected.any():
        u = u.clone()
        singular = singular.clone()
        u[reflected, :, -1] *= -1
        singular[reflected, -1] *= -1
        rotation = u @ vh
    scale = singular.sum(-1) / left.square().sum((1, 2)).clamp_min(1e-8)
    aligned = scale[:, None, None] * (left @ rotation) + target_mean
    return torch.linalg.vector_norm(aligned - target, dim=-1).mean(-1)


@torch.no_grad()
def evaluate_pa_mpjpe(model, loader, device: str) -> float:
    values = []
    model.eval()
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        valid = batch["valid"].bool()
        predicted = smooth_valid(output["pose_rel"].float().cpu(), valid, 5)
        target = batch["pose_rel"].float()
        values.append(_similarity_aligned_mpjpe(
            predicted[valid], target[valid]
        ))
    return float(torch.cat(values).mean())


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
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = json.loads(args.calibration.read_text(encoding="utf-8"))
    if locked.get("protocol") != args.exp:
        raise RuntimeError(
            f"calibration protocol mismatch: {locked.get('protocol')} != {args.exp}"
        )
    if locked.get("selection_split") != "validation":
        raise RuntimeError("final calibration must be selected on validation")
    if locked.get("test_used_for_selection") is not False:
        raise RuntimeError("calibration does not prove that test stayed sealed")
    source = locked["source"]
    if not _same_path(source["hybrid_checkpoint"], args.hybrid_checkpoint):
        raise RuntimeError("hybrid checkpoint differs from calibration source")
    if not _same_path(
        source["root_expert_checkpoint"], args.root_expert_checkpoint
    ):
        raise RuntimeError("root checkpoint differs from calibration source")

    args.pose_strength = float(source["pose_strength"])
    args.root_strength = float(source["root_strength"])
    args.bone_blend = float(source["bone_blend"])
    args.bone_symmetric = bool(source["bone_symmetric"])
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model = _build_model(args, device)
    temporal = model.base
    temporal.set_calibration(
        int(locked["selected"]["window"]),
        float(locked["selected"]["blend"]),
        source.get("risk_adaptive", "none"),
        float(source.get("danger_logit_bias", 0.0)),
    )
    temporal.set_root_calibration(
        int(locked["selected"].get("root_window", 1)),
        float(locked["selected"].get("root_blend", 0.0)),
    )
    model.eval()

    danger_bias = float(source.get("danger_logit_bias", 0.0))
    validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    validation["pa_mpjpe_m"] = evaluate_pa_mpjpe(
        model, loaders["val"], device
    )
    validation_classification = evaluate_classification(
        model, loaders["val_class"], device, danger_bias
    )
    result = {
        "run": "p2_v11_final_locked_evaluation",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_opened": bool(args.open_test),
        "calibration": str(args.calibration),
        "configuration": {
            **source,
            "residual_window": int(locked["selected"]["window"]),
            "residual_blend": float(locked["selected"]["blend"]),
            "root_residual_window": int(
                locked["selected"].get("root_window", 1)
            ),
            "root_residual_blend": float(
                locked["selected"].get("root_blend", 0.0)
            ),
        },
        "validation": validation,
        "validation_classification": validation_classification,
    }
    if args.open_test:
        result["test"] = evaluate_trajectory(
            model, loaders["test"], device, args.max_shift
        )
        result["test"]["pa_mpjpe_m"] = evaluate_pa_mpjpe(
            model, loaders["test"], device
        )
        result["test_classification"] = evaluate_classification(
            model, loaders["test_class"], device, danger_bias
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "protocol": args.exp,
        "configuration": result["configuration"],
        "source_calibration": str(args.calibration),
        "validation": validation,
        "validation_classification": validation_classification,
        **({
            "test": result["test"],
            "test_classification": result["test_classification"],
        } if args.open_test else {}),
    }, args.output.with_name("final_model.pt"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
