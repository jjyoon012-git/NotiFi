"""CAL40 5-seed 결과를 source site별 운영 신뢰도로 다시 집계한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "action_accuracy",
    "action_macro_f1",
    "risk_accuracy",
    "risk_macro_f1",
    "danger_recall",
    "danger_action_accuracy",
    "safe_to_danger",
)


def main() -> None:
    """A44 fold 원시 수치를 site별 평균·표준편차와 범위로 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = json.loads(options.input.read_text(encoding="utf-8"))
    if report.get("sealed_yja_used") is not False:
        raise RuntimeError("site reliability input must exclude sealed yja")

    values: dict[str, dict[str, list[float]]] = {}
    for seed in report["seeds"]:
        for fold in seed["folds"].values():
            if fold.get("outer_used_for_selection") is not False:
                raise RuntimeError("outer result was used for model selection")
            for site, metrics in zip(fold["outer_sites"], fold["outer_metrics"]):
                site_values = values.setdefault(
                    site, {metric: [] for metric in METRICS}
                )
                for metric in METRICS:
                    if metric == "safe_to_danger":
                        value = metrics[metric] / max(metrics["safe_total"], 1)
                    else:
                        value = metrics[metric]
                    site_values[metric].append(float(value))

    sites = {}
    for site, metrics in sorted(values.items()):
        sites[site] = {
            metric: {
                "mean": float(np.mean(numbers)),
                "std": float(np.std(numbers)),
                "minimum": float(np.min(numbers)),
                "maximum": float(np.max(numbers)),
            }
            for metric, numbers in metrics.items()
        }
    result = {
        "run": "A55-CAL40-SITE-RELIABILITY",
        "source": str(options.input.name),
        "support_seeds": [int(seed["support_seed"]) for seed in report["seeds"]],
        "sites": sites,
        "worst_mean_site": {
            metric: min(sites, key=lambda site: sites[site][metric]["mean"])
            for metric in METRICS
        },
        "best_mean_site": {
            metric: max(sites, key=lambda site: sites[site][metric]["mean"])
            for metric in METRICS
        },
        "diagnostic_only": True,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        site: {
            "action": metrics["action_accuracy"]["mean"],
            "danger_recall": metrics["danger_recall"]["mean"],
            "safe_to_danger": metrics["safe_to_danger"]["mean"],
        }
        for site, metrics in sites.items()
    }, indent=2))


if __name__ == "__main__":
    main()
