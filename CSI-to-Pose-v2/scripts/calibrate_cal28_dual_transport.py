"""CAL17 prototype 이동과 CAL27 query 이동을 source-inner에서만 결합한다."""

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
from calibrate_cal17_style_transport import (  # noqa: E402
    choose_action_config as choose_linear_config,
    choose_risk_config,
    class_prototypes,
    embed_site,
    evaluate_action_config as evaluate_linear,
)
from calibrate_cal27_kernel_transport import (  # noqa: E402
    choose_action_config as choose_kernel_config,
    evaluate_action_config as evaluate_kernel,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def blended_actions(
    targets: list[dict[str, torch.Tensor]],
    library: list[dict[str, torch.Tensor]],
    linear_config: tuple[float, ...],
    kernel_config: tuple[float, ...],
    kernel_weight: float,
) -> list[torch.Tensor]:
    """두 calibration의 정규화 log probability를 고정 비율로 결합한다."""
    _, linear = evaluate_linear(targets, library, linear_config)
    _, kernel = evaluate_kernel(targets, library, kernel_config)
    return [
        (1.0 - kernel_weight) * left.log_softmax(-1)
        + kernel_weight * right.log_softmax(-1)
        for left, right in zip(linear, kernel)
    ]


def choose_blend(
    targets: list[dict[str, torch.Tensor]],
    library: list[dict[str, torch.Tensor]],
    linear_config: tuple[float, ...],
    kernel_config: tuple[float, ...],
) -> float:
    """outer를 보지 않고 inner 평균·최악 action utility로 결합률을 고른다."""
    scored = []
    for weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        actions = blended_actions(
            targets, library, linear_config, kernel_config, weight,
        )
        utilities = []
        for target, action in zip(targets, actions):
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


def final_metrics(
    model,
    targets: list[dict[str, torch.Tensor]],
    library: list[dict[str, torch.Tensor]],
    linear_config: tuple[float, ...],
    kernel_config: tuple[float, ...],
    kernel_weight: float,
    risk_config: tuple[float, float, float],
) -> list[dict]:
    """inner에서 잠근 dual transport와 risk 설정으로 query를 평가한다."""
    actions = blended_actions(
        targets, library, linear_config, kernel_config, kernel_weight,
    )
    safe_weight, fusion, danger_bias = risk_config
    rows = []
    for target, action in zip(targets, actions):
        direct = target["direct_risk"].clone()
        similarity = torch.nn.functional.normalize(
            target["embedding"], dim=-1,
        ) @ target["anchors"].transpose(0, 1)
        top2 = similarity.topk(2, dim=-1).values
        direct[:, 0] += safe_weight * (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
        direct[:, 2] += danger_bias
        risk = (1.0 - fusion) * direct + fusion * model.action_to_risk(action)
        rows.append(classification_metrics(
            action, risk, target["labels"], target["risks"],
        ))
    return rows


def main() -> None:
    """세 outer 사람에서 dual transport를 nested source-only로 검증한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-seed", type=int, default=17018)
    parser.add_argument("--shots-per-prompt", type=int, default=2)
    parser.add_argument("--absence-trials", type=int, default=12)
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL28")
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
    folds = {}
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
        info = training["fold_results"][held_out]
        train_sites = list(info["train_sites"])
        inner_sites = list(info["inner_validation_sites"])
        outer_sites = list(info["outer_test_sites"])
        embedded = {
            site: embed_site(
                model, store, index, selected_rows, sites, site, device,
                options.support_seed, options.absence_seed,
                options.shots_per_prompt, None, options.absence_trials,
            )
            for site in train_sites + inner_sites + outer_sites
        }
        library = [{
            "site": site,
            "classes": class_prototypes(embedded[site]),
            "anchors": embedded[site]["anchors"],
        } for site in train_sites]
        inner_targets = [embedded[site] for site in inner_sites]
        linear_config = choose_linear_config(inner_targets, library)
        kernel_config = choose_kernel_config(inner_targets, library)
        kernel_weight = choose_blend(
            inner_targets, library, linear_config, kernel_config,
        )
        inner_actions = blended_actions(
            inner_targets, library, linear_config, kernel_config, kernel_weight,
        )
        risk_config = choose_risk_config(model, inner_targets, inner_actions)
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_sites": inner_sites,
            "outer_sites": outer_sites,
            "linear_config": list(linear_config),
            "kernel_config": list(kernel_config),
            "kernel_weight": kernel_weight,
            "risk_config": dict(zip((
                "safe_weight", "fusion", "danger_bias",
            ), risk_config)),
            "inner_metrics": final_metrics(
                model, inner_targets, library, linear_config, kernel_config,
                kernel_weight, risk_config,
            ),
            "outer_metrics": final_metrics(
                model, [embedded[site] for site in outer_sites], library,
                linear_config, kernel_config, kernel_weight, risk_config,
            ),
            "outer_used_for_selection": False,
        }
    result = {
        "run": "CAL28-INNER-SELECTED-DUAL-TRANSPORT",
        "folds": folds,
        "support_seed": options.support_seed,
        "absence_trials": options.absence_trials,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
