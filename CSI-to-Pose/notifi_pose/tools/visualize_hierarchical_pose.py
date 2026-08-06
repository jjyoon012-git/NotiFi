"""Render GVHMR-surface overlays for validation-locked KP2-DH predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .. import contract as C
from ..continuous_pose import CSILatentPoseRegressor
from ..dataio.dataset import build_datasets
from ..hierarchical_pose import HierarchicalCSIPoseRegressor
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_sealed import make_model
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_continuous_pose import load_teacher
from .visualize_v13s_seen import (
    SmplSurfaceFitter,
    predict,
    render_trial,
    select_representatives,
)


class HierarchicalPoseBlend(nn.Module):
    """Use V13S outputs while replacing pose with the locked KP2-DH blend."""

    def __init__(self, coarse_model: nn.Module, candidate_model: nn.Module,
                 strength: float):
        super().__init__()
        if not 0.0 <= strength <= 1.0:
            raise ValueError("blend strength must be between zero and one")
        self.coarse_model = coarse_model
        self.candidate_model = candidate_model
        self.strength = float(strength)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        coarse = self.coarse_model(csi, link_mask)
        candidate = self.candidate_model(csi, link_mask)
        pose = coarse["pose_rel"] + self.strength * (
            candidate["pose_rel"] - coarse["pose_rel"]
        )
        return {
            **coarse,
            "pose_rel": pose,
            "pose_coarse": coarse["pose_rel"],
            "pose_candidate": candidate["pose_rel"],
        }


def build_hierarchical_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = checkpoint["source"]
    kp2c_path = C.PROJECT_ROOT / source["kp2c_checkpoint"]
    p2_path = C.PROJECT_ROOT / source["p2_checkpoint"]
    motion_path = C.PROJECT_ROOT / source["motion_checkpoint"]
    kp2c_checkpoint = torch.load(
        kp2c_path, map_location="cpu", weights_only=False
    )
    teacher, motion_architecture = load_teacher(motion_path, device)
    p2_checkpoint = torch.load(p2_path, map_location=device, weights_only=False)
    base_model = make_model(p2_checkpoint, device)
    architecture = checkpoint["architecture"]
    source_architecture = kp2c_checkpoint["architecture"]
    backbone = CSILatentPoseRegressor(
        base_model,
        teacher.decoder,
        checkpoint["latent_mean"],
        checkpoint["latent_std"],
        checkpoint["bone_lengths"],
        hidden=int(architecture.get(
            "hidden", source_architecture["hidden"]
        )),
        code_dim=int(architecture.get(
            "code_dim", source_architecture.get(
                "code_dim", motion_architecture["code_dim"]
            )
        )),
        temporal_layers=int(architecture.get(
            "temporal_layers", source_architecture.get("temporal_layers", 2)
        )),
        heads=int(architecture.get(
            "heads", source_architecture.get("heads", 4)
        )),
        dropout=float(architecture.get(
            "dropout", source_architecture.get("dropout", 0.08)
        )),
    ).to(device)
    model = HierarchicalCSIPoseRegressor(
        backbone,
        direction_scale=float(architecture.get("direction_scale", 1.0)),
        endpoint_scale=float(architecture.get("endpoint_scale", 0.40)),
        dropout=float(architecture.get("dropout", 0.08)),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.eval()
    return model, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dh_hierarchical_pose" / "best_model.pt",
    )
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=C.PROJECT_ROOT / "docs" / "results" / "v13s_pruned_pose_root_ensemble.json",
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v12w_robust_classification_ensemble" / "validation.json",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--trial-id", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fps", type=float, default=C.TARGET_FPS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--smpl-model", type=Path,
        default=Path(r"C:\Users\jjeong\Desktop\NotiFi-3D\SMPLX\SMPL_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--video-root", type=Path,
        default=Path(
            r"C:\Users\jjeong\Desktop\NotiFi-3D\NotiFi-CSI-Pose-Dataset"
            r"\TRAINING_DATA"
        ),
    )
    parser.add_argument(
        "--out", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2dh_seen_fall_overlay",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate, checkpoint = build_hierarchical_model(args.checkpoint, device)
    if checkpoint.get("protocol") != args.exp:
        raise RuntimeError("KP2-DH checkpoint protocol mismatch")
    strength = float(checkpoint["blend_selection"]["strength"])
    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    coarse, coarse_configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    model = HierarchicalPoseBlend(coarse, candidate, strength).to(device)
    model.eval()

    dataset = pose_only(build_datasets(
        exp=args.exp, baseline="sub", seed=args.seed
    )["test"])
    prediction = predict(model, dataset, device, args.batch_size)
    positions = select_representatives(dataset, prediction, args.trial_id)
    split_index = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    video_by_trial = dict(zip(
        split_index.trial_id.astype(str), split_index.original_video
    ))

    args.out.mkdir(parents=True, exist_ok=True)
    fitter = SmplSurfaceFitter(args.smpl_model)
    results = []
    for count, position in enumerate(positions, 1):
        row = dataset.index.iloc[position].copy()
        row["original_video"] = video_by_trial.get(str(row.trial_id), "")
        item = dataset[position]
        output = args.out / f"{row.trial_id}_kp2dh_gvhmr.mp4"
        print(
            f"[{count}/{len(positions)}] {row.trial_id} "
            f"{row.risk}/{row.detail_label}", flush=True,
        )
        results.append(render_trial(
            row, item, prediction["pose"][position],
            prediction["root"][position],
            int(prediction["class"][position]),
            int(prediction["risk"][position]),
            "gvhmr", output, args.fps, fitter, args.video_root,
        ))

    report = {
        "run": f"{checkpoint['run']}-fall-overlay",
        "protocol": args.exp,
        "model": "validation-locked KP2-DH blend",
        "checkpoint": str(args.checkpoint),
        "blend_formula": (
            f"V13S + {strength:.2f} * (KP2-DH - V13S)"
        ),
        "selection": (
            "explicit trial ids" if args.trial_id
            else "default representative selection"
        ),
        "selection_uses_test_gt": args.trial_id is None,
        "inference_input": ["csi", "link_mask"],
        "coarse_configuration": coarse_configuration,
        "results": results,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(
        args.out / "selected_trial.csv", index=False, encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.out),
        "trials": [dataset.index.iloc[position].trial_id for position in positions],
        "blend_strength": strength,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
