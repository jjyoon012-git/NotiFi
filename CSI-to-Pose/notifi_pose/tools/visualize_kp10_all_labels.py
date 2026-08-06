"""Render one deterministic KP10 stick and SMPL video per pose label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
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


POSE_CLASS_IDS = tuple(class_id for class_id in range(C.N_CLASSES) if class_id != 6)


def deterministic_label_representatives(
    index: pd.DataFrame, subject: str,
) -> list[int]:
    """Choose the middle sorted test trial per label without looking at GT error."""
    index = index.reset_index(drop=True)
    positions = []
    missing = []
    for class_id in POSE_CLASS_IDS:
        candidates = index.index[
            (index.subject == subject) & (index.class_id == class_id)
        ].tolist()
        candidates.sort(key=lambda position: str(index.iloc[position].trial_id))
        if not candidates:
            missing.append(class_id)
            continue
        positions.append(candidates[len(candidates) // 2])
    if missing:
        raise RuntimeError(
            f"subject {subject!r} has no fixed-test pose trial for classes {missing}"
        )
    return positions


def _contact_sheet(results: list[dict], mode: str, destination: Path) -> None:
    matches = [result for result in results if result["mode"] == mode]
    tiles = []
    for result in matches:
        image = cv2.imread(result["preview"])
        if image is None:
            raise RuntimeError(f"cannot read preview: {result['preview']}")
        tiles.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    columns = 4
    rows = []
    blank = np.full_like(tiles[0], 245)
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(destination), np.concatenate(rows, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_action_arguments(parser)
    parser.add_argument("--subject", default="mhw", choices=C.SUBJECTS)
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
        default=C.WORK_ROOT / "runs" / "kp10_mhw_all_label_overlays",
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
    classifier.eval()
    extra, _ = classifier_outputs(classifier, data["cache"], device)
    action = (1.50 * data["base_action_logits"] + 0.75 * extra).argmax(-1)
    risk = data["risk_probability"].argmax(-1)

    dataset = pose_only(build_datasets(
        exp=args.exp, baseline="sub", seed=17
    )["test"])
    if len(dataset) != len(predicted):
        raise RuntimeError("prediction and fixed-test dataset order differ")
    positions = deterministic_label_representatives(dataset.index, args.subject)

    split_index = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    videos = dict(zip(
        split_index.trial_id.astype(str), split_index.original_video
    ))
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in ("stickman", "gvhmr"):
        (args.out / mode).mkdir(exist_ok=True)
    fitter = SmplSurfaceFitter(args.smpl_model, face_stride=12)

    results = []
    for count, position in enumerate(positions, 1):
        row = dataset.index.iloc[position].copy()
        row["original_video"] = videos.get(str(row.trial_id), "")
        scenario = str(row.scenario_id)
        stem = f"{scenario}_{row.detail_label}__{row.trial_id}"
        print(
            f"[{count:02d}/{len(positions)}] {scenario} {row.detail_label} "
            f"({row.trial_id})", flush=True,
        )
        for mode in ("stickman", "gvhmr"):
            output = args.out / mode / f"{stem}__{mode}.mp4"
            result = render_trial(
                row, dataset[position], predicted[position],
                dataset[position]["root"].numpy(), int(action[position]),
                int(risk[position]), mode, output, args.fps, fitter,
                args.video_root, pelvis_align_prediction=True,
            )
            result["scenario_id"] = scenario
            result["class_id"] = int(row.class_id)
            results.append(result)
            print(f"  wrote {output}", flush=True)

    sheets = {}
    for mode in ("stickman", "gvhmr"):
        destination = args.out / f"preview_{mode}_contact_sheet.png"
        _contact_sheet(results, mode, destination)
        sheets[mode] = str(destination)

    report = {
        "model": "KP10-ACTION-FUSED-45",
        "protocol": args.exp,
        "subject": args.subject,
        "selection": "middle sorted fixed-test trial_id per pose class",
        "selection_uses_pose_error": False,
        "selection_uses_test_label_for_display_only": True,
        "inference_inputs": ["CSI", "link mask"],
        "pose_alignment": "GT pelvis shared for pose visualization; root not evaluated",
        "mesh": "neutral SMPL surface fitted to GVHMR-22 joints",
        "excluded_label": {
            "class_id": 6,
            "label": "absence",
            "reason": "classification-only trial without GVHMR pose target",
        },
        "results": results,
        "contact_sheets": sheets,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(
        args.out / "selected_trials.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({
        "output": str(args.out),
        "videos": len(results),
        "labels": len(positions),
        "contact_sheets": sheets,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
