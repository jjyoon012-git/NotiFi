"""Validation-only calibration of sequence-consistent predicted bone lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..hybrid_v10 import SequenceBoneCalibration
from ..trainer import set_seed
from .calibrate_v11_temporal import _load_model
from .train_p2_v9_hybrid import pose_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument(
        "--hybrid-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v9_hybrid_v10_clean" / "best_model.pt",
    )
    parser.add_argument(
        "--hybrid-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v9_hybrid_v10_clean" / "recalibrated_results.json",
    )
    parser.add_argument(
        "--root-expert-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v4_v9a_lmh_e01_multitask_recall93" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--root-expert-kind", choices=("v9", "p2_hybrid"), default="v9"
    )
    parser.add_argument("--allow-unverified-root-protocol", action="store_true")
    parser.add_argument(
        "--v3-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_v3_contact_root" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--v2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibrated_model.pt",
    )
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
    )
    parser.add_argument(
        "--motion-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--pose-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
    )
    parser.add_argument(
        "--root-residual-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--root-expert-strength", type=float, default=0.5)
    parser.add_argument("--pose-strength-override", type=float, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v11_bone_calibration" / "validation.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    base, source = _load_model(args, device)
    model = SequenceBoneCalibration(base).to(device)
    candidates = []
    for symmetric in (False, True):
        for blend in (0.0, 0.25, 0.50, 0.75, 1.0):
            model.set_calibration(blend, symmetric)
            metrics = evaluate_trajectory(
                model, loaders["val"], device, args.max_shift
            )
            candidates.append({
                "blend": blend,
                "symmetric": symmetric,
                "score": pose_selection_score(metrics),
                "validation": metrics,
            })
    baseline = candidates[0]
    feasible = [
        item for item in candidates
        if item["validation"]["mpjpe_m"] <= baseline["validation"]["mpjpe_m"]
        and item["validation"]["dynamic_mpjpe_m"]
        <= baseline["validation"]["dynamic_mpjpe_m"] * 1.01
        and item["validation"]["danger_endpoint_mpjpe_m"]
        <= baseline["validation"]["danger_endpoint_mpjpe_m"] * 1.02
    ]
    selected = min(feasible or candidates, key=lambda item: item["score"])
    result = {
        "run": "p2_v11_sequence_bone_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_calibration": source,
        "selected": {
            "blend": selected["blend"],
            "symmetric": selected["symmetric"],
        },
        "baseline_validation": baseline["validation"],
        "selected_validation": selected["validation"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
