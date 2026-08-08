"""KP-v2 source outer subject의 class별 혼동을 target 없이 진단한다."""

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
from calibrate_cal17_style_transport import embed_site  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def _confusion(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """정답 행·예측 열의 17x17 confusion matrix를 만든다."""
    matrix = torch.zeros(C.N_CLASSES, C.N_CLASSES, dtype=torch.long)
    for truth, prediction in zip(target.tolist(), predicted.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def main() -> None:
    """각 outer source 사람의 query를 고정 support로 한 번씩 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja entered KP-v2 outer diagnosis")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS)
            & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in all_sites
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    names = {
        int(class_id): str(group.detail_label.iloc[0])
        for class_id, group in selected.groupby("class_id")
    }
    results = {}
    total_confusion = torch.zeros(C.N_CLASSES, C.N_CLASSES, dtype=torch.long)
    all_actions, all_risks, all_labels, all_risk_targets = [], [], [], []
    for held_out in ("ajh", "mhw", "lmh"):
        checkpoint = torch.load(
            options.run_dir / f"selection_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        if checkpoint.get("outer_holdout_used_for_selection") is not False:
            raise RuntimeError("outer-contaminated checkpoint")
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        site_results = {}
        for site in sorted(value for value in all_sites if value.startswith(held_out)):
            payload = embed_site(
                model, store, index, selected_rows, sites, site, device,
                support_seed=options.support_seed,
            )
            prediction = payload["action"].argmax(-1)
            matrix = _confusion(prediction, payload["labels"])
            total_confusion += matrix
            class_rows = {}
            for class_id in range(C.N_CLASSES):
                count = int(matrix[class_id].sum())
                if not count:
                    continue
                top = matrix[class_id].topk(3)
                class_rows[str(class_id)] = {
                    "name": names.get(class_id, "absence"),
                    "trials": count,
                    "recall": float(matrix[class_id, class_id] / count),
                    "top_predictions": [
                        {
                            "class_id": int(predicted_id),
                            "name": names.get(int(predicted_id), "absence"),
                            "count": int(value),
                        }
                        for value, predicted_id in zip(
                            top.values.tolist(), top.indices.tolist()
                        )
                    ],
                }
            metrics = classification_metrics(
                payload["action"], payload["direct_risk"],
                payload["labels"], payload["risks"],
            )
            metrics["absence_predictions_on_active_queries"] = int(
                (prediction == 6).sum()
            )
            site_results[site] = {"metrics": metrics, "classes": class_rows}
            all_actions.append(payload["action"])
            all_risks.append(payload["direct_risk"])
            all_labels.append(payload["labels"])
            all_risk_targets.append(payload["risks"])
        results[held_out] = site_results
    aggregate = classification_metrics(
        torch.cat(all_actions), torch.cat(all_risks),
        torch.cat(all_labels), torch.cat(all_risk_targets),
    )
    aggregate["absence_predictions_on_active_queries"] = int(
        total_confusion[:, 6].sum()
    )
    result = {
        "run": options.run_dir.name,
        "protocol": "source_outer_subject_diagnosis_only",
        "support_seed": options.support_seed,
        "aggregate": aggregate,
        "confusion": total_confusion.tolist(),
        "folds": results,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
