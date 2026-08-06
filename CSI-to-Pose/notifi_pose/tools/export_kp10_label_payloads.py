"""Export deterministic mhw KP10 GT/prediction trajectories for rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from .visualize_kp10_all_labels import deterministic_label_representatives


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
    parser.add_argument("--out", type=Path, required=True)
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
    action = (
        1.50 * data["base_action_logits"] + 0.75 * extra
    ).argmax(-1).cpu().numpy()
    risk = data["risk_probability"].argmax(-1).cpu().numpy()

    dataset = pose_only(build_datasets(
        exp=args.exp, baseline="sub", seed=17
    )["test"])
    if len(dataset) != len(predicted):
        raise RuntimeError("prediction and fixed-test dataset order differ")
    positions = deterministic_label_representatives(dataset.index, args.subject)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for count, position in enumerate(positions, 1):
        row = dataset.index.iloc[position]
        item = dataset[position]
        frames = int(row.n_frames)
        root = item["root"].numpy()[:frames]
        target_relative = item["pose_rel"].numpy()[:frames]
        predicted_relative = predicted[position, :frames]
        target_absolute = target_relative + root[:, None]
        predicted_absolute = predicted_relative + root[:, None]
        valid = item["valid"].numpy().astype(bool)[:frames]
        motion = np.zeros(frames, np.float32)
        motion[1:] = np.linalg.norm(
            target_absolute[1:] - target_absolute[:-1], axis=-1
        ).mean(-1)
        scenario = str(row.scenario_id)
        stem = f"{scenario}_{row.detail_label}__{row.trial_id}"
        destination = args.out / f"{stem}.npz"
        np.savez_compressed(
            destination,
            target_absolute=target_absolute.astype(np.float32),
            predicted_absolute=predicted_absolute.astype(np.float32),
            valid=valid,
            preview_frame=np.asarray(int(np.argmax(motion)), np.int32),
            class_prediction=np.asarray(int(action[position]), np.int32),
            risk_prediction=np.asarray(int(risk[position]), np.int32),
            class_id=np.asarray(int(row.class_id), np.int32),
            trial_id=np.asarray(str(row.trial_id)),
            scenario_id=np.asarray(scenario),
            detail_label=np.asarray(str(row.detail_label)),
            risk=np.asarray(str(row.risk)),
        )
        rows.append({
            "class_id": int(row.class_id),
            "scenario_id": scenario,
            "detail_label": str(row.detail_label),
            "risk": str(row.risk),
            "trial_id": str(row.trial_id),
            "frames": frames,
            "preview_frame": int(np.argmax(motion)),
            "class_prediction": int(action[position]),
            "risk_prediction": int(risk[position]),
            "payload": str(destination),
        })
        print(f"[{count:02d}/{len(positions)}] wrote {destination}", flush=True)

    pd.DataFrame(rows).to_csv(
        args.out / "payload_manifest.csv", index=False, encoding="utf-8-sig"
    )
    (args.out / "payload_manifest.json").write_text(
        json.dumps({
            "model": "KP10-ACTION-FUSED-45",
            "protocol": args.exp,
            "subject": args.subject,
            "selection": "middle sorted fixed-test trial_id per pose class",
            "pelvis_alignment": "prediction and GT share GT root for pose display",
            "rows": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.out), "payloads": len(rows)
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
