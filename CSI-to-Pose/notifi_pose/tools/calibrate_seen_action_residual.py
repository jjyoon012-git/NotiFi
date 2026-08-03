"""Select the seen action-residual strength on validation, then test once."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import DropoutConfig, build_datasets
from ..trainer import evaluate, set_seed
from .diagnose_observability import evaluate_model, pose_only, report_path
from .train_seen_action_residual import make_model


def calibration_score(metrics: dict) -> float:
    """Balance position accuracy with physically plausible motion amplitude."""
    speed_ratio = max(float(metrics["pose_speed_ratio"]), 1e-3)
    return (
        float(metrics["mpjpe_m"])
        + 0.10 * float(metrics["dynamic_mpjpe_m"])
        + 0.10 * float(metrics["distal_mpjpe"])
        + 0.10 * float(metrics["impact_mpjpe"])
        + 0.10 * abs(math.log(speed_ratio))
    )


def measure(model, dataset, loader, criterion, device: str,
            batch_size: int, smooth_window: int) -> dict:
    return {
        **evaluate(model, loader, criterion, device),
        **evaluate_model(
            model, dataset, device, batch_size, smooth_window
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
    )
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "calibration.json",
    )
    args = parser.parse_args()
    if not args.scales:
        raise ValueError("at least one residual scale is required")
    if any(scale < 0.0 or scale > 1.0 for scale in args.scales):
        raise ValueError("all residual scales must be between 0 and 1")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=args.seed,
    )
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    loaders = {
        "val": DataLoader(
            validation, batch_size=args.batch_size, shuffle=False, num_workers=0
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size, shuffle=False, num_workers=0
        ),
    }
    model = make_model(args.baseline_checkpoint, args.motion_checkpoint, device)
    checkpoint = torch.load(
        args.residual_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    criterion = L.PoseLoss(
        lambda_root=1.0, lambda_bone=0.1, lambda_cls=0.0, lambda_risk=0.0,
        lambda_velocity=0.1, lambda_impact=0.2,
        lambda_displacement=0.1, motion_weight=3.0, device=device,
    ).to(device)

    candidates = []
    for scale in args.scales:
        model.set_residual_scale(scale)
        metrics = measure(
            model, validation, loaders["val"], criterion, device,
            args.batch_size, args.smooth_window,
        )
        score = calibration_score(metrics)
        candidates.append({
            "scale": scale, "calibration_score": score, "validation": metrics,
        })
        print(
            f"scale={scale:.2f} score={score:.4f} "
            f"mpjpe={metrics['mpjpe_m'] * 100:.2f}cm "
            f"dynamic={metrics['dynamic_mpjpe_m'] * 100:.2f}cm "
            f"speed_ratio={metrics['pose_speed_ratio']:.3f}"
        )

    selected = min(candidates, key=lambda item: item["calibration_score"])
    model.set_residual_scale(float(selected["scale"]))
    test_metrics = measure(
        model, test, loaders["test"], criterion, device,
        args.batch_size, args.smooth_window,
    )
    result = {
        "run": "calibrated_action_motion_residual_single_split",
        "protocol": "single_split",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "score_definition": (
            "mpjpe + 0.10*dynamic + 0.10*distal + 0.10*impact "
            "+ 0.10*abs(log(pose_speed_ratio))"
        ),
        "selected_scale": selected["scale"],
        "selected_validation": selected["validation"],
        "candidates": candidates,
        "test": test_metrics,
        "checkpoints": {
            "baseline": report_path(args.baseline_checkpoint),
            "motion": report_path(args.motion_checkpoint),
            "residual": report_path(args.residual_checkpoint),
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
