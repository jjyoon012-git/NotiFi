"""CAL32의 5-seed source-LOSO action/risk confusion을 누수 없이 집계한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import class_prototypes, embed_site  # noqa: E402
from calibrate_cal28_dual_transport import blended_actions  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


ACTION_NAMES = (
    "walking", "standing_still", "sitting_still", "lying_still",
    "lie_to_stand", "stand_to_lie_normal", "absence", "sit_to_stand",
    "stand_to_sit", "unstable_walking", "stumble_recover",
    "bed_exit_failed", "fall_from_standing", "fall_while_walking",
    "bed_exit_fall", "bed_fall", "chair_exit_fall",
)
RISK_NAMES = ("safe", "warning", "danger")


def add_confusion(
    matrix: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor,
) -> None:
    """target-prediction 쌍을 정수 confusion matrix에 누적한다."""
    flat = target.long() * matrix.shape[1] + prediction.long()
    matrix += torch.bincount(
        flat, minlength=matrix.numel(),
    ).reshape_as(matrix)


def summarize_action(matrix: torch.Tensor) -> dict:
    """class recall과 대각선을 제외한 최다 혼동 쌍을 계산한다."""
    support = matrix.sum(1)
    recall = matrix.diag().float() / support.clamp_min(1)
    mistakes = []
    for target in range(C.N_CLASSES):
        for prediction in range(C.N_CLASSES):
            if target != prediction and int(matrix[target, prediction]) > 0:
                mistakes.append({
                    "target": ACTION_NAMES[target],
                    "prediction": ACTION_NAMES[prediction],
                    "count": int(matrix[target, prediction]),
                })
    mistakes.sort(key=lambda row: row["count"], reverse=True)
    return {
        "class_recall": {
            ACTION_NAMES[index]: float(recall[index])
            for index in range(C.N_CLASSES) if int(support[index]) > 0
        },
        "support": {
            ACTION_NAMES[index]: int(support[index])
            for index in range(C.N_CLASSES) if int(support[index]) > 0
        },
        "top_confusions": mistakes[:20],
    }


def main() -> None:
    """잠긴 CAL32 config로 5개 support seed의 outer prediction confusion을 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    options = parser.parse_args()
    locked_result = json.loads(options.config_result.read_text(encoding="utf-8"))
    if any(locked_result.get(key) is not False for key in (
        "target_subject_used", "sealed_yja_used",
        "query_labels_or_pose_gt_at_inference", "outer_used_for_selection",
    )):
        raise RuntimeError("CAL32 config result is not source-clean")
    locked = locked_result["fixed_configs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL32 diagnosis")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    if set(sites.tolist()) != SOURCE_SITES:
        raise RuntimeError("unexpected source site contract")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS) & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in sorted(SOURCE_SITES)
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    training = json.loads((options.run_dir / "result.json").read_text(encoding="utf-8"))
    action_confusion = torch.zeros(C.N_CLASSES, C.N_CLASSES, dtype=torch.long)
    conservative_risk = torch.zeros(C.N_RISK, C.N_RISK, dtype=torch.long)
    safety_risk = torch.zeros(C.N_RISK, C.N_RISK, dtype=torch.long)
    danger_subtype = torch.zeros(5, 6, dtype=torch.long)
    site_rows = []
    for held_out in ("ajh", "mhw", "lmh"):
        checkpoint = torch.load(
            options.run_dir / f"selection_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        info = training["fold_results"][held_out]
        train_sites = list(info["train_sites"])
        outer_sites = list(info["outer_test_sites"])
        config = locked[held_out]
        for seed in options.support_seeds:
            embedded = {
                site: embed_site(
                    model, store, index, selected_rows, sites, site, device,
                    seed, seed + 1, 2, None, options.absence_trials,
                )
                for site in train_sites + outer_sites
            }
            library = [{
                "site": site,
                "classes": class_prototypes(embedded[site]),
                "anchors": embedded[site]["anchors"],
            } for site in train_sites]
            targets = [embedded[site] for site in outer_sites]
            actions = blended_actions(
                targets, library, tuple(config["linear_config"]),
                tuple(config["kernel_config"]), config["kernel_weight"],
            )
            for site, target, action in zip(outer_sites, targets, actions):
                labels = target["labels"]
                risks = target["risks"]
                action_prediction = action.argmax(-1)
                native_prediction = target["risk"].argmax(-1)
                safety_logits = target["risk"].clone()
                safety_logits[:, 2] += float(config["native_risk_danger_bias"])
                safety_prediction = safety_logits.argmax(-1)
                add_confusion(action_confusion, labels, action_prediction)
                add_confusion(conservative_risk, risks, native_prediction)
                add_confusion(safety_risk, risks, safety_prediction)
                danger = risks == 2
                subtype_target = labels[danger] - 12
                subtype_prediction = action_prediction[danger]
                subtype_prediction = torch.where(
                    (subtype_prediction >= 12) & (subtype_prediction <= 16),
                    subtype_prediction - 12,
                    torch.full_like(subtype_prediction, 5),
                )
                add_confusion(danger_subtype, subtype_target, subtype_prediction)
                site_rows.append({
                    "site": site,
                    "support_seed": seed,
                    "trials": len(labels),
                })
    result = {
        "run": "CAL32-POOLED-CONFUSION-DIAGNOSIS",
        "action_names": ACTION_NAMES,
        "risk_names": RISK_NAMES,
        "action_confusion": action_confusion.tolist(),
        "action_summary": summarize_action(action_confusion),
        "conservative_risk_confusion": conservative_risk.tolist(),
        "safety_risk_confusion": safety_risk.tolist(),
        "danger_subtype_columns": (*ACTION_NAMES[12:17], "other_action"),
        "danger_subtype_confusion": danger_subtype.tolist(),
        "site_seed_evaluations": site_rows,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "outer_used_for_selection": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "class_recall": result["action_summary"]["class_recall"],
        "top_confusions": result["action_summary"]["top_confusions"][:10],
        "danger_subtype_confusion": result["danger_subtype_confusion"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
