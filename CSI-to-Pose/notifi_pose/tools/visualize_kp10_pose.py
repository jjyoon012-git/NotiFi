"""Render deterministic walking and fall overlays for locked KP10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from .audit_kp10_paired_bootstrap import kp10_prediction
from .calibrate_independent_risk_fusion import classifier_outputs
from .calibrate_part_motion_profile_reranking import prepare
from .calibrate_predicted_action_retrieval import add_action_arguments
from .diagnose_observability import pose_only
from .visualize_v13s_seen import SmplSurfaceFitter, render_trial


def deterministic_representatives(index: pd.DataFrame) -> list[int]:
    positions = []
    for class_id in (0, 13):
        candidates = index.index[index.class_id == class_id].tolist()
        candidates.sort(key=lambda item: str(index.iloc[item].trial_id))
        if not candidates:
            raise RuntimeError(f"no fixed-test trials for class {class_id}")
        positions.append(candidates[len(candidates) // 2])
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_action_arguments(parser)
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_classifier_seed181"
        / "best_model.pt",
    )
    parser.add_argument(
        "--profile-ranker-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp8_profile_candidate_ranker_seed127"
        / "best_model.pt",
    )
    parser.add_argument(
        "--strength-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "kp10_action_strength"
        / "calibration.json",
    )
    parser.add_argument("--fps", type=float, default=C.TARGET_FPS)
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
        default=C.WORK_ROOT / "runs" / "kp10_representative_overlays",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare(args, "test", device)
    predicted = kp10_prediction(data, args, device).cpu().numpy()

    checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=False
    )
    classifier = TemporalMotionSelector(**checkpoint["model_config"]).to(device)
    classifier.load_state_dict(checkpoint["model"])
    extra, _ = classifier_outputs(classifier, data["cache"], device)
    action = (1.50 * data["base_action_logits"] + 0.75 * extra).argmax(-1)
    risk = data["risk_probability"].argmax(-1)

    dataset = pose_only(build_datasets(
        exp=args.exp, baseline="sub", seed=17
    )["test"])
    if len(dataset) != len(predicted):
        raise RuntimeError("prediction and fixed-test dataset order differ")
    positions = deterministic_representatives(dataset.index.reset_index(drop=True))
    split_index = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    videos = dict(zip(
        split_index.trial_id.astype(str), split_index.original_video
    ))
    args.out.mkdir(parents=True, exist_ok=True)
    fitter = SmplSurfaceFitter(args.smpl_model)
    results = []
    previews = []
    for position in positions:
        row = dataset.index.iloc[position].copy()
        row["original_video"] = videos.get(str(row.trial_id), "")
        output = args.out / f"{row.trial_id}_kp10_gvhmr.mp4"
        result = render_trial(
            row, dataset[position], predicted[position],
            dataset[position]["root"].numpy(), int(action[position]),
            int(risk[position]), "gvhmr", output, args.fps, fitter,
            args.video_root, pelvis_align_prediction=True,
        )
        results.append(result)
        previews.append(cv2.resize(cv2.imread(result["preview"]), (960, 540)))
    sheet = args.out / "walking_fall_contact_sheet.png"
    cv2.imwrite(str(sheet), cv2.vconcat(previews))
    report = {
        "model": "KP10-ACTION-FUSED-45",
        "protocol": args.exp,
        "selection": "middle trial_id within class 0 and class 13",
        "selection_uses_pose_error": False,
        "selection_uses_test_action_label": True,
        "inference_inputs": ["CSI", "link mask"],
        "visualization_alignment": (
            "GT root shared only for pelvis-aligned pose visualization; "
            "absolute root is not evaluated"
        ),
        "results": results,
        "contact_sheet": str(sheet),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
