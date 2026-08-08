"""Safe-support margin으로 global danger bias를 부분 수축해 site 편향을 줄인다."""

from __future__ import annotations

import argparse
import itertools
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
from calibrate_cal62_deep_ensemble import load_fold_model  # noqa: E402
from evaluate_cal42_safe_anchor_risk import RISK_KEYS, pooled  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def observed_margin(payload: dict[str, torch.Tensor], quantile: float) -> float:
    """모두 safe인 calibration CSI의 danger-vs-nondanger margin을 요약한다."""
    calibration = payload["calibration_safe_risk"]
    margin = calibration[:, 2] - calibration[:, :2].max(-1).values
    return float(torch.quantile(margin, float(quantile)))


def shrinkage_metrics(
    payload: dict[str, torch.Tensor],
    quantile: float,
    base_bias: float,
    shrinkage: float,
    reference_margin: float,
) -> dict[str, float | int]:
    """global bias에서 target safe-margin 편차의 일부만 빼고 risk를 평가한다."""
    target_margin = observed_margin(payload, quantile)
    adjustment = float(base_bias) - float(shrinkage) * (
        target_margin - float(reference_margin)
    )
    risk = payload["risk"].clone()
    risk[:, 2] += adjustment
    metrics = classification_metrics(
        payload["action"], risk, payload["labels"], payload["risks"]
    )
    return {
        **metrics,
        "safe_to_danger_rate": (
            metrics["safe_to_danger"] / max(metrics["safe_total"], 1)
        ),
        "danger_adjustment": adjustment,
        "observed_safe_margin": target_margin,
        "reference_safe_margin": float(reference_margin),
    }


def choose_config(
    payloads: list[dict[str, torch.Tensor]],
) -> tuple[float, float, float, float]:
    """inner query만으로 quantile·base bias·수축률을 선택한다."""
    best = None
    for quantile, base_bias, shrinkage in itertools.product(
        (0.50, 0.75, 0.90, 0.95),
        (1.0, 1.5, 2.0),
        (0.0, 0.25, 0.50, 0.75, 1.0),
    ):
        reference = float(np.median([
            observed_margin(payload, quantile) for payload in payloads
        ]))
        rows = [
            shrinkage_metrics(
                payload, quantile, base_bias, shrinkage, reference
            )
            for payload in payloads
        ]
        false_alarm = np.asarray(
            [row["safe_to_danger_rate"] for row in rows], dtype=np.float64
        )
        utility = np.asarray([
            0.40 * row["risk_macro_f1"]
            + 0.35 * row["danger_recall"]
            + 0.15 * row["risk_accuracy"]
            - 0.40 * row["safe_to_danger_rate"]
            for row in rows
        ])
        score = float(utility.mean() + 0.25 * utility.min())
        if false_alarm.mean() > 0.25 or false_alarm.max() > 0.40:
            score -= 1.0
        candidate = (
            score, -float(false_alarm.mean()),
            (quantile, base_bias, shrinkage, reference),
        )
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[2]


def main() -> None:
    """CAL43 partial shrinkage를 5-seed source nested-LOSO로 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--baseline-danger-bias", type=float, default=1.5)
    options = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL43")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    if set(sites.tolist()) != SOURCE_SITES:
        raise RuntimeError("unexpected source site contract")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS)
            & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in sorted(SOURCE_SITES)
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    training = json.loads((options.run_dir / "result.json").read_text(
        encoding="utf-8"
    ))
    models = {
        subject: load_fold_model(options.run_dir, subject, device)
        for subject in ("ajh", "mhw", "lmh")
    }

    seeds = []
    for seed in options.support_seeds:
        selected_outer = []
        baseline_outer = []
        folds = {}
        for held_out, model in models.items():
            info = training["fold_results"][held_out]
            inner_sites = list(info["inner_validation_sites"])
            outer_sites = list(info["outer_test_sites"])
            embedded = {
                site: embed_site(
                    model, store, index, selected_rows, sites, site, device,
                    seed, seed + 1, 2, None, options.absence_trials,
                )
                for site in inner_sites + outer_sites
            }
            config = choose_config([embedded[site] for site in inner_sites])
            quantile, base_bias, shrinkage, reference = config
            inner_rows = [
                shrinkage_metrics(embedded[site], *config)
                for site in inner_sites
            ]
            outer_rows = [
                shrinkage_metrics(embedded[site], *config)
                for site in outer_sites
            ]
            baseline_rows = []
            for site in outer_sites:
                payload = embedded[site]
                risk = payload["risk"].clone()
                risk[:, 2] += float(options.baseline_danger_bias)
                metrics = classification_metrics(
                    payload["action"], risk,
                    payload["labels"], payload["risks"],
                )
                metrics["safe_to_danger_rate"] = (
                    metrics["safe_to_danger"] / max(metrics["safe_total"], 1)
                )
                baseline_rows.append(metrics)
            selected_outer.extend(outer_rows)
            baseline_outer.extend(baseline_rows)
            folds[held_out] = {
                "inner_sites": inner_sites,
                "outer_sites": outer_sites,
                "selected_config": {
                    "safe_margin_quantile": quantile,
                    "base_danger_bias": base_bias,
                    "shrinkage": shrinkage,
                    "reference_margin": reference,
                },
                "inner_metrics": inner_rows,
                "outer_metrics": outer_rows,
                "baseline_outer_metrics": baseline_rows,
                "outer_used_for_selection": False,
            }
        seeds.append({
            "support_seed": int(seed),
            "cal43": pooled(selected_outer),
            "baseline": pooled(baseline_outer),
            "folds": folds,
        })
        print(f"finished support seed {seed}", flush=True)

    metric_keys = (*RISK_KEYS, "safe_to_danger_rate")
    result = {
        "run": "A57-CAL43-SAFE-ANCHOR-PARTIAL-SHRINKAGE",
        "protocol": "source nested-LOSO; config selected on inner sites only",
        "seeds": seeds,
        "aggregate": {
            name: {
                key: {
                    "mean": float(np.mean([seed[name][key] for seed in seeds])),
                    "std": float(np.std([seed[name][key] for seed in seeds])),
                }
                for key in metric_keys
            }
            for name in ("cal43", "baseline")
        },
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
