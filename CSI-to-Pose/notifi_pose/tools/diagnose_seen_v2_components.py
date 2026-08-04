"""Compare coarse, rotation-only, and full seen-v2 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from ..seen_v2 import SeenReconstructionV2Net
from .diagnose_observability import aggregate, evaluate_predictions, pose_only
from .train_seen_v2 import load_motion, make_final_seen_backbone


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "best_model.pt",
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
    parser.add_argument("--dataset", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = QualityWeightedDataset(pose_only(build_datasets(
        exp="single_split", baseline="sub"
    )[args.dataset]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    backbone = make_final_seen_backbone(
        args.baseline_checkpoint, args.motion_checkpoint,
        args.pose_residual_checkpoint, args.root_residual_checkpoint,
        0.5, device,
    )
    model = SeenReconstructionV2Net(
        backbone, load_motion(args.motion_checkpoint, device),
        hidden=128,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.set_partial_finetune(False)
    model.eval()
    rows = {key: [] for key in ("coarse", "rotation_only", "full_pose", "full")}
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        variants = {
            "coarse": (output["pose_coarse"], output["root_coarse"]),
            "rotation_only": (output["pose_low"], output["root_coarse"]),
            "full_pose": (output["pose_rel"], output["root_coarse"]),
            "full": (output["pose_rel"], output["root"]),
        }
        for key, (pose, root) in variants.items():
            rows[key].extend(evaluate_predictions(
                pose, root, batch, args.smooth_window
            ))
    result = {key: aggregate(value) for key, value in rows.items()}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
