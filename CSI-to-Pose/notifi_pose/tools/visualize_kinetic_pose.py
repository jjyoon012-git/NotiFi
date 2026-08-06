"""Render representative seen-test overlays for a KineticPose checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..kinetic_pose import KineticPoseResidual
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_sealed import make_model
from .evaluate_v12_final import _read_locked, build_locked_model
from .visualize_v13s_seen import (
    SmplSurfaceFitter,
    contact_sheet,
    predict,
    render_trial,
    select_representatives,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_exp03_activity_calibrated" / "best_model.pt",
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
        default=C.WORK_ROOT / "runs" / "kp1_exp03_seen_overlays",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    baseline, baseline_configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    p2_checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    p2_model = make_model(p2_checkpoint, device)
    architecture = checkpoint["architecture"]
    model = KineticPoseResidual(
        baseline, p2_model.norm,
        hidden=int(architecture["hidden"]),
        temporal_layers=int(architecture["temporal_layers"]),
        max_delta=float(architecture["max_delta_m"]),
        condition_on_coarse=bool(architecture.get("condition_on_coarse", True)),
        activity_floor=float(architecture.get("activity_floor", 0.15)),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.set_residual_strength(float(checkpoint["residual_strength"]))
    model.set_activity_threshold(float(checkpoint.get("activity_threshold", 0.0)))
    model.eval()
    del p2_model

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
    (args.out / "stickman").mkdir(exist_ok=True)
    (args.out / "gvhmr").mkdir(exist_ok=True)
    fitter = SmplSurfaceFitter(args.smpl_model)
    results = []
    for count, position in enumerate(positions, 1):
        row = dataset.index.iloc[position].copy()
        row["original_video"] = video_by_trial.get(str(row.trial_id), "")
        item = dataset[position]
        print(
            f"[{count}/{len(positions)}] {row.trial_id} "
            f"{row.risk}/{row.detail_label}", flush=True
        )
        for mode in ("stickman", "gvhmr"):
            output = args.out / mode / f"{row.trial_id}_{mode}.mp4"
            results.append(render_trial(
                row, item, prediction["pose"][position],
                prediction["root"][position],
                int(prediction["class"][position]),
                int(prediction["risk"][position]),
                mode, output, args.fps, fitter, args.video_root,
            ))
    contact_sheet(results, args.out / "preview_contact_sheet.png")
    report = {
        "run": f"{checkpoint['run']}-seen-overlays",
        "protocol": args.exp,
        "model": checkpoint["run"],
        "checkpoint": str(args.checkpoint),
        "selection": (
            "median-MPJPE walking/ajh, stumble/mhw, fall-walking/lmh"
            if args.trial_id is None else "explicit trial ids"
        ),
        "selection_uses_test_gt": True,
        "selection_note": "visualization only; no model or metric selection",
        "baseline_configuration": baseline_configuration,
        "results": results,
        "preview_contact_sheet": str(args.out / "preview_contact_sheet.png"),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(
        args.out / "selected_trials.csv", index=False, encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.out),
        "preview": str(args.out / "preview_contact_sheet.png"),
        "trials": [dataset.index.iloc[position].trial_id for position in positions],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
