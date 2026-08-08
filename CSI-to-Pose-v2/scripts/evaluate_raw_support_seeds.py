"""Evaluate model-native logits over repeated source calibration supports."""

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
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from train_cal20_source_folds import evaluate_site  # noqa: E402


def _aggregate(metrics: list[dict]) -> dict:
    """Pool one support seed over its seven outer source sites."""
    weights = np.asarray([row["trials"] for row in metrics], dtype=np.float64)
    keys = (
        "action_accuracy", "action_macro_f1", "risk_accuracy",
        "risk_macro_f1", "danger_recall", "danger_action_accuracy",
    )
    result = {
        key: float(np.average([row[key] for row in metrics], weights=weights))
        for key in keys
    }
    result.update({
        "safe_to_danger_rate": (
            sum(row["safe_to_danger"] for row in metrics)
            / max(sum(row["safe_total"] for row in metrics), 1)
        ),
        "worst_site_action": float(min(
            row["action_accuracy"] for row in metrics
        )),
    })
    return result


def main() -> None:
    """Load clean folds, vary support only, and save model-native metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--support-seeds", type=int, nargs="+",
        default=(17017, 17027, 17037, 17047, 17057),
    )
    parser.add_argument("--absence-trials", type=int, default=12)
    options = parser.parse_args()

    base.ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
    base.PROMPT_SHOTS = {class_id: 2 for class_id in MOTION_PROMPT_CLASSES}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja must not appear in source rows")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
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

    checkpoints = {}
    for subject in ("ajh", "mhw", "lmh"):
        path = options.run_dir / f"selection_{subject}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if any(checkpoint.get(key) is not False for key in (
            "outer_holdout_used_for_selection", "target_subject_used",
            "sealed_yja_used",
        )):
            raise RuntimeError(f"unclean source checkpoint: {path}")
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        checkpoints[subject] = (model, list(checkpoint["outer_test"]))

    seed_results = []
    for seed in options.support_seeds:
        metrics = []
        for subject, (model, outer_sites) in checkpoints.items():
            if any(not site.startswith(f"{subject}_") for site in outer_sites):
                raise RuntimeError(f"invalid outer sites for {subject}")
            metrics.extend([
                evaluate_site(
                    model, store, index, selected_rows, sites, site,
                    device, seed=seed, absence_trials=options.absence_trials,
                )
                for site in outer_sites
            ])
        seed_results.append({
            "support_seed": int(seed),
            "metrics": _aggregate(metrics),
            "sites": metrics,
        })
    metric_keys = tuple(seed_results[0]["metrics"])
    aggregate = {
        key: {
            "mean": float(np.mean([
                row["metrics"][key] for row in seed_results
            ])),
            "std": float(np.std([
                row["metrics"][key] for row in seed_results
            ])),
        }
        for key in metric_keys
    }
    report = {
        "run_dir": str(options.run_dir.resolve()),
        "protocol": "model_native_source_loso_repeated_support_v1",
        "absence_trials": options.absence_trials,
        "aggregate": aggregate,
        "seeds": seed_results,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    options.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
