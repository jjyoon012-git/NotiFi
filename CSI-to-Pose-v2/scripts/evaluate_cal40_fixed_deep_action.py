"""두 encoder의 CAL17 action을 고정 결합하고 CAL32 safety risk를 유지한다."""

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
from calibrate_cal17_style_transport import (  # noqa: E402
    class_prototypes, embed_site, evaluate_action_config,
)
from calibrate_cal62_deep_ensemble import (  # noqa: E402
    load_fold_model, probability_ensemble,
)
from diagnose_cal32_confusions import add_confusion, summarize_action  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


METRIC_KEYS = (
    "action_accuracy", "action_macro_f1", "risk_accuracy",
    "risk_macro_f1", "danger_recall", "danger_action_accuracy",
)
ACTION_RANGES = ((0, 9), (9, 12), (12, 17))


def hierarchy_diagnostics(
    action: torch.Tensor,
    risk: torch.Tensor,
    labels: torch.Tensor,
    risks: torch.Tensor,
) -> dict[str, int | float]:
    """정답 risk oracle과 예측 risk routing으로 계층 분류 병목만 진단한다."""
    if action.ndim != 2 or action.shape[-1] != C.N_CLASSES:
        raise ValueError("action logits must have shape [B,17]")
    if risk.shape != (len(action), C.N_RISK):
        raise ValueError("risk logits must have shape [B,3]")
    oracle = action.new_full(action.shape, -torch.inf)
    predicted = action.new_full(action.shape, -torch.inf)
    predicted_risk = risk.argmax(-1)
    for risk_id, (start, stop) in enumerate(ACTION_RANGES):
        oracle[risks == risk_id, start:stop] = action[
            risks == risk_id, start:stop
        ]
        predicted[predicted_risk == risk_id, start:stop] = action[
            predicted_risk == risk_id, start:stop
        ]
    danger = risks == 2
    danger_logits = action[danger, 12:17]
    danger_labels = labels[danger] - 12
    top2 = danger_logits.topk(min(2, danger_logits.shape[-1]), dim=-1).indices
    return {
        "oracle_risk_action_correct": int((oracle.argmax(-1) == labels).sum()),
        "predicted_risk_action_correct": int(
            (predicted.argmax(-1) == labels).sum()
        ),
        "hierarchy_trials": int(len(labels)),
        "danger_within_group_correct": int(
            (danger_logits.argmax(-1) == danger_labels).sum()
        ),
        "danger_within_group_top2_correct": int(
            (top2 == danger_labels[:, None]).any(-1).sum()
        ),
        "danger_within_group_total": int(danger.sum()),
    }


def locked_linear_configs(paths: list[Path]) -> dict[str, list[list[float]]]:
    """여러 source-inner report의 두 모델 설정을 fold별 중앙값으로 잠근다."""
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, report in zip(paths, reports):
        for key in (
            "outer_holdout_used_for_selection", "target_subject_used",
            "sealed_yja_used", "query_labels_or_pose_gt_at_inference",
        ):
            if report.get(key) is not False:
                raise RuntimeError(f"contaminated CAL39 report {path}: {key}")
        if any(
            fold.get("outer_used_for_selection") is not False
            for fold in report["folds"].values()
        ):
            raise RuntimeError(f"outer-selected CAL39 report: {path}")
    locked = {}
    for held_out in ("ajh", "mhw", "lmh"):
        values = np.asarray([
            report["folds"][held_out]["linear_configs"]
            for report in reports
        ], dtype=np.float64)
        locked[held_out] = np.median(values, axis=0).tolist()
    return locked


def aggregate(rows: list[dict]) -> dict:
    """outer 7개 site를 trial 수로 가중해 하나의 성능 표로 집계한다."""
    weights = np.asarray([row["trials"] for row in rows], dtype=np.float64)
    result = {
        key: float(np.average([row[key] for row in rows], weights=weights))
        for key in METRIC_KEYS
    }
    result["safe_to_danger_rate"] = (
        sum(row["safe_to_danger"] for row in rows)
        / max(sum(row["safe_total"] for row in rows), 1)
    )
    result["worst_site_action"] = float(min(
        row["action_accuracy"] for row in rows
    ))
    hierarchy_trials = sum(row["hierarchy_trials"] for row in rows)
    danger_total = sum(row["danger_within_group_total"] for row in rows)
    result["oracle_risk_action_accuracy"] = (
        sum(row["oracle_risk_action_correct"] for row in rows)
        / max(hierarchy_trials, 1)
    )
    result["predicted_risk_action_accuracy"] = (
        sum(row["predicted_risk_action_correct"] for row in rows)
        / max(hierarchy_trials, 1)
    )
    result["danger_within_group_accuracy"] = (
        sum(row["danger_within_group_correct"] for row in rows)
        / max(danger_total, 1)
    )
    result["danger_within_group_top2_accuracy"] = (
        sum(row["danger_within_group_top2_correct"] for row in rows)
        / max(danger_total, 1)
    )
    return result


def main() -> None:
    """고정 설정 CAL40을 5개 support seed의 source nested-LOSO로 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--selection-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--danger-bias", type=float, default=1.5)
    parser.add_argument("--include-confusion", action="store_true")
    options = parser.parse_args()
    if options.run_dir_a.resolve() == options.run_dir_b.resolve():
        raise ValueError("CAL40 requires two different source models")
    locked = locked_linear_configs(options.selection_results)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL40")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
    absence_rows = np.concatenate([
        np.flatnonzero(((index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS) & (index.class_id == 6)
            & index.cache_ok).to_numpy())
        for site in all_sites
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    training = [
        json.loads((run / "result.json").read_text(encoding="utf-8"))
        for run in (options.run_dir_a, options.run_dir_b)
    ]
    models = {
        held_out: [
            load_fold_model(run, held_out, device)
            for run in (options.run_dir_a, options.run_dir_b)
        ]
        for held_out in ("ajh", "mhw", "lmh")
    }

    seeds = []
    action_confusion = torch.zeros(
        C.N_CLASSES, C.N_CLASSES, dtype=torch.long
    )
    risk_confusion = torch.zeros(C.N_RISK, C.N_RISK, dtype=torch.long)
    danger_subtype = torch.zeros(5, 6, dtype=torch.long)
    for seed in options.support_seeds:
        all_outer = []
        folds = {}
        for held_out in ("ajh", "mhw", "lmh"):
            info = training[0]["fold_results"][held_out]
            other = training[1]["fold_results"][held_out]
            for key in ("train_sites", "outer_test_sites"):
                if list(info[key]) != list(other[key]):
                    raise RuntimeError(f"split mismatch: {held_out}/{key}")
            train_sites = list(info["train_sites"])
            outer_sites = list(info["outer_test_sites"])
            requested = train_sites + outer_sites
            embedded = [{
                site: embed_site(
                    model, store, index, selected_rows, sites, site, device,
                    seed, seed + 1, 2, None, options.absence_trials,
                )
                for site in requested
            } for model in models[held_out]]
            libraries = [[{
                "site": site,
                "classes": class_prototypes(payload[site]),
                "anchors": payload[site]["anchors"],
            } for site in train_sites] for payload in embedded]
            actions = []
            for payload, library, config in zip(
                embedded, libraries, locked[held_out]
            ):
                _, current = evaluate_action_config(
                    [payload[site] for site in outer_sites],
                    library, tuple(config),
                )
                actions.append(current)
            rows = []
            for number, site in enumerate(outer_sites):
                left = embedded[0][site]
                right = embedded[1][site]
                if not torch.equal(left["labels"], right["labels"]):
                    raise RuntimeError("ensemble query order mismatch")
                action = probability_ensemble(
                    actions[0][number], actions[1][number]
                )
                risk = left["risk"].clone()
                risk[:, 2] += float(options.danger_bias)
                row = classification_metrics(
                    action, risk, left["labels"], left["risks"]
                )
                row.update(hierarchy_diagnostics(
                    action, risk, left["labels"], left["risks"]
                ))
                if options.include_confusion:
                    action_prediction = action.argmax(-1)
                    risk_prediction = risk.argmax(-1)
                    add_confusion(
                        action_confusion, left["labels"], action_prediction
                    )
                    add_confusion(
                        risk_confusion, left["risks"], risk_prediction
                    )
                    danger = left["risks"] == 2
                    subtype_target = left["labels"][danger] - 12
                    subtype_prediction = action_prediction[danger]
                    subtype_prediction = torch.where(
                        (subtype_prediction >= 12)
                        & (subtype_prediction <= 16),
                        subtype_prediction - 12,
                        torch.full_like(subtype_prediction, 5),
                    )
                    add_confusion(
                        danger_subtype, subtype_target, subtype_prediction
                    )
                rows.append(row)
            all_outer.extend(rows)
            folds[held_out] = {
                "outer_sites": outer_sites,
                "outer_metrics": rows,
                "outer_used_for_selection": False,
            }
        seeds.append({
            "support_seed": int(seed),
            "metrics": aggregate(all_outer),
            "folds": folds,
        })
        print(f"finished support seed {seed}", flush=True)

    keys = (
        *METRIC_KEYS, "safe_to_danger_rate", "worst_site_action",
        "oracle_risk_action_accuracy", "predicted_risk_action_accuracy",
        "danger_within_group_accuracy", "danger_within_group_top2_accuracy",
    )
    result = {
        "run": "CAL40-FIXED-DEEP-ACTION-SAFETY-RISK",
        "protocol": "source nested-LOSO; five support seeds; fixed configs",
        "members": [options.run_dir_a.name, options.run_dir_b.name],
        "fixed_linear_configs": locked,
        "action_ensemble_weights": [0.5, 0.5],
        "risk_source_member": 0,
        "danger_bias": options.danger_bias,
        "seeds": seeds,
        "aggregate": {
            key: {
                "mean": float(np.mean([seed["metrics"][key] for seed in seeds])),
                "std": float(np.std([seed["metrics"][key] for seed in seeds])),
            }
            for key in keys
        },
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "diagnostics_use_query_risk_labels": True,
        "diagnostics_used_for_model_selection": False,
    }
    if options.include_confusion:
        result.update({
            "diagnostic_confusion": True,
            "action_names": C.ACTION_NAMES,
            "risk_names": C.RISK_NAMES,
            "action_confusion": action_confusion.tolist(),
            "action_summary": summarize_action(action_confusion),
            "risk_confusion": risk_confusion.tolist(),
            "danger_subtype_columns": (*C.ACTION_NAMES[12:17], "other_action"),
            "danger_subtype_confusion": danger_subtype.tolist(),
        })
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
