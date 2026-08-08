"""여러 source-inner run의 중앙 설정 하나로 CAL28의 support-seed 강건성을 평가한다."""

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
    choose_risk_config, class_prototypes, embed_site,
)
from calibrate_cal28_dual_transport import (  # noqa: E402
    blended_actions, final_metrics,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


METRIC_KEYS = (
    "action_accuracy", "action_macro_f1", "risk_accuracy",
    "risk_macro_f1", "danger_recall", "danger_action_accuracy",
)


def fixed_configs(paths: list[Path]) -> dict:
    """query와 outer를 사용하지 않은 여러 run의 fold별 설정 중앙값을 잠근다."""
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, report in zip(paths, reports):
        if report.get("target_subject_used") is not False:
            raise RuntimeError(f"target subject contamination in {path}")
        if report.get("sealed_yja_used") is not False:
            raise RuntimeError(f"sealed yja contamination in {path}")
        if report.get("query_labels_or_pose_gt_at_inference") is not False:
            raise RuntimeError(f"query supervision contamination in {path}")
        if any(
            fold.get("outer_used_for_selection") is not False
            for fold in report["folds"].values()
        ):
            raise RuntimeError(f"outer selection contamination in {path}")
    locked = {}
    for held_out in ("ajh", "mhw", "lmh"):
        folds = [report["folds"][held_out] for report in reports]
        locked[held_out] = {
            "linear_config": np.median(np.asarray([
                fold["linear_config"] for fold in folds
            ], dtype=np.float64), axis=0).tolist(),
            "kernel_config": np.median(np.asarray([
                fold["kernel_config"] for fold in folds
            ], dtype=np.float64), axis=0).tolist(),
            "kernel_weight": float(np.median([
                fold["kernel_weight"] for fold in folds
            ])),
            "risk_config": {
                key: float(np.median([
                    fold["risk_config"][key] for fold in folds
                ]))
                for key in ("safe_weight", "fusion", "danger_bias")
            },
            "selection_source": "median of source-inner-only CAL28 runs",
        }
    return locked


def aggregate(rows: list[dict]) -> dict:
    """7개 outer site를 trial 가중 지표와 pooled 오류율로 집계한다."""
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
    return result


def choose_episode_blend(
    episodes: list[tuple[dict, list[dict]]],
    linear_config: tuple[float, ...],
    kernel_config: tuple[float, ...],
) -> float:
    """여러 support seed의 source-inner episode에서 고정 dual 비율 하나를 선택한다."""
    scored = []
    for weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        utilities = []
        for target, library in episodes:
            action = blended_actions(
                [target], library, linear_config, kernel_config, weight,
            )[0]
            row = classification_metrics(
                action, target["direct_risk"], target["labels"], target["risks"],
            )
            utilities.append(
                0.55 * row["action_macro_f1"]
                + 0.20 * row["action_accuracy"]
                + 0.25 * row["danger_action_accuracy"]
            )
        scored.append((
            0.5 * float(np.mean(utilities)) + 0.5 * float(np.min(utilities)),
            weight,
        ))
    return max(scored)[1]


def native_risk_metrics(
    targets: list[dict[str, torch.Tensor]],
    library: list[dict[str, torch.Tensor]],
    linear_config: tuple[float, ...],
    kernel_config: tuple[float, ...],
    kernel_weight: float,
    danger_bias: float = 0.0,
) -> list[dict]:
    """transport action과 CAL60의 학습된 독립 risk 출력을 각자 평가한다."""
    actions = blended_actions(
        targets, library, linear_config, kernel_config, kernel_weight,
    )
    rows = []
    for target, action in zip(targets, actions):
        risk = target["risk"].clone()
        risk[:, 2] += float(danger_bias)
        rows.append(classification_metrics(
            action, risk, target["labels"], target["risks"],
        ))
    return rows


def choose_native_danger_bias(targets: list[dict[str, torch.Tensor]]) -> float:
    """source-inner 위험 성능·danger recall·오경보를 함께 고려해 bias 하나를 고른다."""
    scored = []
    for bias in (-1.0, -0.5, 0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50):
        utilities = []
        for target in targets:
            risk = target["risk"].clone()
            risk[:, 2] += bias
            row = classification_metrics(
                target["action"], risk, target["labels"], target["risks"],
            )
            false_rate = row["safe_to_danger"] / max(row["safe_total"], 1)
            utilities.append(
                0.35 * row["risk_macro_f1"]
                + 0.30 * row["danger_recall"]
                + 0.20 * row["risk_accuracy"]
                + 0.15 * (1.0 - false_rate)
            )
        scored.append((
            0.5 * float(np.mean(utilities)) + 0.5 * float(np.min(utilities)),
            -abs(bias),
            bias,
        ))
    return max(scored)[2]


def main() -> None:
    """잠긴 fold별 설정으로 5개 support seed의 outer 결과만 계산한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--aggregate-inner-selection", action="store_true")
    parser.add_argument("--native-risk", action="store_true")
    parser.add_argument("--select-native-risk-bias", action="store_true")
    options = parser.parse_args()
    locked = fixed_configs(options.selection_results)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL29")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS) & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in all_sites
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    training = json.loads((options.run_dir / "result.json").read_text(encoding="utf-8"))
    checkpoints = {}
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
        checkpoints[held_out] = model

    if options.aggregate_inner_selection:
        for held_out in ("ajh", "mhw", "lmh"):
            model = checkpoints[held_out]
            info = training["fold_results"][held_out]
            train_sites = list(info["train_sites"])
            inner_sites = list(info["inner_validation_sites"])
            episodes = []
            for seed in options.support_seeds:
                embedded = {
                    site: embed_site(
                        model, store, index, selected_rows, sites, site, device,
                        seed, seed + 1, 2, None, options.absence_trials,
                    )
                    for site in train_sites + inner_sites
                }
                library = [{
                    "site": site,
                    "classes": class_prototypes(embedded[site]),
                    "anchors": embedded[site]["anchors"],
                } for site in train_sites]
                episodes.extend((embedded[site], library) for site in inner_sites)
            config = locked[held_out]
            linear = tuple(config["linear_config"])
            kernel = tuple(config["kernel_config"])
            weight = choose_episode_blend(episodes, linear, kernel)
            targets = [episode[0] for episode in episodes]
            actions = [
                blended_actions(
                    [target], library, linear, kernel, weight,
                )[0]
                for target, library in episodes
            ]
            risk = choose_risk_config(model, targets, actions)
            config["kernel_weight"] = weight
            config["risk_config"] = dict(zip((
                "safe_weight", "fusion", "danger_bias",
            ), risk))
            config["selection_source"] = (
                "joint source-inner selection across all declared support seeds"
            )

    if options.select_native_risk_bias:
        if not options.native_risk:
            raise ValueError("native risk bias selection requires --native-risk")
        for held_out in ("ajh", "mhw", "lmh"):
            model = checkpoints[held_out]
            info = training["fold_results"][held_out]
            inner_sites = list(info["inner_validation_sites"])
            inner_targets = []
            for seed in options.support_seeds:
                inner_targets.extend(
                    embed_site(
                        model, store, index, selected_rows, sites, site, device,
                        seed, seed + 1, 2, None, options.absence_trials,
                    )
                    for site in inner_sites
                )
            locked[held_out]["native_risk_danger_bias"] = (
                choose_native_danger_bias(inner_targets)
            )
            locked[held_out]["native_risk_selection_source"] = (
                "joint source-inner selection across all declared support seeds"
            )

    seeds = []
    for seed in options.support_seeds:
        all_outer = []
        fold_rows = {}
        for held_out in ("ajh", "mhw", "lmh"):
            model = checkpoints[held_out]
            info = training["fold_results"][held_out]
            train_sites = list(info["train_sites"])
            outer_sites = list(info["outer_test_sites"])
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
            config = locked[held_out]
            risk = tuple(config["risk_config"][key] for key in (
                "safe_weight", "fusion", "danger_bias",
            ))
            targets = [embedded[site] for site in outer_sites]
            if options.native_risk:
                metrics = native_risk_metrics(
                    targets, library,
                    tuple(config["linear_config"]),
                    tuple(config["kernel_config"]),
                    config["kernel_weight"],
                    config.get("native_risk_danger_bias", 0.0),
                )
            else:
                metrics = final_metrics(
                    model, targets, library,
                    tuple(config["linear_config"]), tuple(config["kernel_config"]),
                    config["kernel_weight"], risk,
                )
            all_outer.extend(metrics)
            fold_rows[held_out] = {
                "outer_sites": outer_sites,
                "outer_metrics": metrics,
                "outer_used_for_selection": False,
            }
        seeds.append({
            "support_seed": int(seed),
            "metrics": aggregate(all_outer),
            "folds": fold_rows,
        })
        print(f"finished support seed {seed}", flush=True)
    keys = (*METRIC_KEYS, "safe_to_danger_rate", "worst_site_action")
    result = {
        "run": (
            "CAL32-FIXED-ACTION-SAFETY-RISK"
            if options.select_native_risk_bias
            else "CAL31-FIXED-ACTION-NATIVE-RISK"
            if options.native_risk
            else "CAL30-JOINT-FIXED-DUAL-TRANSPORT"
            if options.aggregate_inner_selection
            else "CAL29-FIXED-DUAL-TRANSPORT"
        ),
        "protocol": "source-nested LOSO; fixed fold configs across five support seeds",
        "fixed_configs": locked,
        "selection_results": [str(path.resolve()) for path in options.selection_results],
        "seeds": seeds,
        "aggregate": {
            key: {
                "mean": float(np.mean([seed["metrics"][key] for seed in seeds])),
                "std": float(np.std([seed["metrics"][key] for seed in seeds])),
            }
            for key in keys
        },
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "outer_used_for_selection": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
