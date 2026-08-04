"""Evaluate a fixed equal-weight residual-seed ensemble on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .. import contract as C
from ..hybrid_v10 import (
    PoseModelEnsemble,
    RootExpertBlend,
    SequenceBoneCalibration,
    build_residual_hybrid,
)
from ..seen_v4 import AlignmentRobustTrajectoryNet
from ..trainer import set_seed
from .evaluate_sealed import make_model
from .train_seen_v4_trajectory import evaluate_trajectory, load_v3, make_loaders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument("--hybrid-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--pose-strength", type=float, default=0.35)
    parser.add_argument(
        "--root-expert-checkpoint", type=Path,
        required=True,
        help="clean-protocol root checkpoint; legacy untagged checkpoints are rejected",
    )
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
    parser.add_argument("--root-strength", type=float, default=0.5)
    parser.add_argument("--bone-blend", type=float, default=0.25)
    parser.add_argument("--bone-symmetric", action="store_true")
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    hybrids = []
    for path in args.hybrid_checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_residual_hybrid(
            make_model(p2_checkpoint, device),
            checkpoint.get("residual_decoder", "dense"),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.set_calibration(args.pose_strength, 0.0, 0.0, 0.0)
        model.eval()
        hybrids.append(model)

    ensemble = PoseModelEnsemble(hybrids).to(device)
    root_expert = AlignmentRobustTrajectoryNet(load_v3(args, device)).to(device)
    root_checkpoint = torch.load(
        args.root_expert_checkpoint, map_location=device, weights_only=False
    )
    if root_checkpoint.get("protocol") != args.exp:
        raise RuntimeError(
            "root expert is missing the requested clean protocol tag: "
            f"{root_checkpoint.get('protocol')!r} != {args.exp!r}"
        )
    missing, unexpected = root_expert.load_state_dict(
        root_checkpoint["model"], strict=False
    )
    invalid_missing = [
        key for key in missing
        if not key.startswith(("class_head.", "risk_head."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "incompatible root expert checkpoint: "
            f"missing={invalid_missing}, unexpected={list(unexpected)}"
        )
    root_expert.set_calibration(
        float(root_checkpoint.get("pose_strength", 0.0)),
        float(root_checkpoint.get("root_strength", 0.0)),
    )
    root_expert.eval()
    rooted = RootExpertBlend(ensemble, root_expert).to(device)
    rooted.set_root_strength(args.root_strength)
    model = SequenceBoneCalibration(
        rooted, blend=args.bone_blend, symmetric=args.bone_symmetric
    ).to(device)
    model.eval()
    validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    result = {
        "run": "p2_v11_equal_seed_ensemble",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "hybrid_checkpoints": [str(path) for path in args.hybrid_checkpoints],
        "selected": {
            "weights": [1.0 / len(hybrids)] * len(hybrids),
            "pose_strength": args.pose_strength,
            "root_strength": args.root_strength,
            "bone_blend": args.bone_blend,
            "bone_symmetric": args.bone_symmetric,
        },
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "hybrid_checkpoints": result["hybrid_checkpoints"],
        "selected": result["selected"],
        "validation": validation,
    }, args.output.with_name("ensemble_model.pt"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
