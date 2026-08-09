"""CAL40에 통제된 낙상 support를 더한 CAL44를 source LOSO로 평가한다."""

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
    class_prototypes,
    embed_site,
    evaluate_action_config,
)
from calibrate_cal62_deep_ensemble import (  # noqa: E402
    load_fold_model,
    probability_ensemble,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.danger_support import (  # noqa: E402
    apply_danger_support,
    class_prototypes as danger_class_prototypes,
    support_evidence,
)
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import (  # noqa: E402
    SOURCE_SITES,
    nested_site_split,
)


def select_danger_positions(
    payload: dict[str, torch.Tensor],
    index: pd.DataFrame,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """각 danger class에서 1개를 뽑고 나머지 query 위치를 반환한다."""
    rows = payload["query_rows"].numpy()
    labels = index.class_id.iloc[rows].to_numpy()
    trial_ids = index.trial_id.iloc[rows].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in C.DANGER_CALIBRATION_CLASSES:
        positions = np.flatnonzero(labels == class_id)
        positions = positions[np.argsort(trial_ids[positions])]
        if len(positions) < 2:
            raise RuntimeError(f"danger class {class_id} has too few trials")
        selected.append(int(rng.permutation(positions)[0]))
    support = torch.tensor(selected, dtype=torch.long)
    keep = torch.ones(len(rows), dtype=torch.bool)
    keep[support] = False
    return support, keep


def prepare_target(
    payload: dict[str, torch.Tensor],
    index: pd.DataFrame,
    seed: int,
) -> dict[str, torch.Tensor]:
    """낙상 support prototype을 만들고 support와 겹치지 않는 query를 보관한다."""
    support, keep = select_danger_positions(payload, index, seed)
    labels = payload["labels"][support]
    return {
        "keep": keep,
        "danger_prototypes": danger_class_prototypes(
            payload["embedding"][support], labels,
            C.DANGER_CALIBRATION_CLASSES,
        ),
    }


def summarize(rows: list[dict]) -> dict:
    """CAL40 공통 집계에 support가 제외된 trial 수를 추가한다."""
    weights = np.asarray([row["trials"] for row in rows], dtype=np.float64)
    metric_keys = (
        "action_accuracy", "action_macro_f1", "risk_accuracy",
        "risk_macro_f1", "danger_recall", "danger_action_accuracy",
    )
    result = {
        key: float(np.average([row[key] for row in rows], weights=weights))
        for key in metric_keys
    }
    result["safe_to_danger_rate"] = (
        sum(row["safe_to_danger"] for row in rows)
        / max(sum(row["safe_total"] for row in rows), 1)
    )
    result["worst_site_action"] = float(min(
        row["action_accuracy"] for row in rows
    ))
    result["trials"] = int(sum(row["trials"] for row in rows))
    return result


def main() -> None:
    """봉인 target 없이 고정 CAL44 설정을 여러 support seed에서 검증한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--cal40-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--danger-bias", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--subtype-weight", type=float, default=0.50)
    options = parser.parse_args()
    locked_result = json.loads(
        options.cal40_result.read_text(encoding="utf-8")
    )
    locked = locked_result["fixed_linear_configs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL44")
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
    models = {
        held_out: [
            load_fold_model(run, held_out, device)
            for run in (options.run_dir_a, options.run_dir_b)
        ]
        for held_out in ("ajh", "mhw", "lmh")
    }
    config = {
        "temperature": float(options.temperature),
        "subtype_weight": float(options.subtype_weight),
        "action_margin_gain": 0.0,
        "action_bias": 0.0,
        "risk_margin_gain": 0.0,
        "risk_bias": 0.0,
    }
    seed_results = []
    for seed in options.support_seeds:
        baseline_rows = []
        calibrated_rows = []
        folds = {}
        for held_out in ("ajh", "mhw", "lmh"):
            train_sites, _, outer_sites = nested_site_split(all_sites, held_out)
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
            for payload, library, member_config in zip(
                embedded, libraries, locked[held_out]
            ):
                _, member_actions = evaluate_action_config(
                    [payload[site] for site in outer_sites],
                    library, tuple(member_config),
                )
                actions.append(member_actions)
            fold_baseline = []
            fold_calibrated = []
            for number, site in enumerate(outer_sites):
                left = embedded[0][site]
                right = embedded[1][site]
                if not torch.equal(left["query_rows"], right["query_rows"]):
                    raise RuntimeError("ensemble query order mismatch")
                left_target = prepare_target(left, index, seed + 1000)
                right_target = prepare_target(right, index, seed + 1000)
                keep = left_target["keep"]
                if not torch.equal(keep, right_target["keep"]):
                    raise RuntimeError("danger support selection mismatch")
                action = probability_ensemble(
                    actions[0][number][keep], actions[1][number][keep]
                )
                risk = left["risk"][keep].clone()
                risk[:, 2] += float(options.danger_bias)
                labels = left["labels"][keep]
                risks = left["risks"][keep]
                baseline = classification_metrics(
                    action, risk, labels, risks
                )
                evidence = [
                    support_evidence(
                        left["embedding"][keep], left["anchors"][:-1],
                        left_target["danger_prototypes"], options.temperature,
                    ),
                    support_evidence(
                        right["embedding"][keep], right["anchors"][:-1],
                        right_target["danger_prototypes"], options.temperature,
                    ),
                ]
                adjusted_action, adjusted_risk, _ = apply_danger_support(
                    action, risk, evidence, config
                )
                calibrated = classification_metrics(
                    adjusted_action, adjusted_risk, labels, risks
                )
                baseline_rows.append(baseline)
                calibrated_rows.append(calibrated)
                fold_baseline.append(baseline)
                fold_calibrated.append(calibrated)
            folds[held_out] = {
                "outer_sites": outer_sites,
                "baseline": fold_baseline,
                "cal44": fold_calibrated,
                "outer_used_for_selection": False,
            }
        seed_results.append({
            "support_seed": int(seed),
            "baseline": summarize(baseline_rows),
            "cal44": summarize(calibrated_rows),
            "folds": folds,
        })
        print(f"finished support seed {seed}", flush=True)
    keys = tuple(seed_results[0]["cal44"])
    aggregate_result = {}
    for model_name in ("baseline", "cal44"):
        aggregate_result[model_name] = {
            key: {
                "mean": float(np.mean([
                    row[model_name][key] for row in seed_results
                ])),
                "std": float(np.std([
                    row[model_name][key] for row in seed_results
                ])),
            }
            for key in keys if key != "trials"
        }
        aggregate_result[model_name]["trials_per_seed"] = int(
            seed_results[0][model_name]["trials"]
        )
    result = {
        "run": "CAL44-FALL-SUPPORT-SINGLE-BUNDLE",
        "protocol": "source nested-LOSO; fixed config; danger support excluded",
        "danger_support_classes": list(C.DANGER_CALIBRATION_CLASSES),
        "danger_support_shots_per_class": 1,
        "config": config,
        "seeds": seed_results,
        "aggregate": aggregate_result,
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate_result, indent=2))


if __name__ == "__main__":
    main()
