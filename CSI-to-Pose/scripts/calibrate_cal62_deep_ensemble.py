"""두 source-clean CAL60 모델을 source-inner calibration으로 결합한다."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import (  # noqa: E402
    choose_action_config,
    class_prototypes,
    embed_site,
    evaluate_action_config,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal12 import CLASS_RANGES  # noqa: E402
from notifi_pose.cal17 import ANCHOR_CLASSES  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402
from train_cal20_source_folds import (  # noqa: E402
    SOURCE_SITES,
    cal12_site_selection_score,
)


def probability_ensemble(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """두 모델의 정규화 확률을 같은 비율로 합쳐 logit으로 반환한다."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("ensemble logits must have the same [B,C] shape")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("ensemble logits must be finite")
    return torch.logaddexp(
        left.log_softmax(-1), right.log_softmax(-1)
    ) - math.log(2.0)


def target_components(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    left_action: torch.Tensor,
    right_action: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """행동·위험 확률과 기본자세 상대 신뢰도를 두 모델에서 평균한다."""
    if not torch.equal(left["labels"], right["labels"]):
        raise RuntimeError("ensemble members have different query label order")
    if not torch.equal(left["risks"], right["risks"]):
        raise RuntimeError("ensemble members have different query risk order")
    margins = []
    for target in (left, right):
        similarity = F.normalize(target["embedding"], dim=-1) @ target[
            "anchors"
        ].transpose(0, 1)
        top2 = similarity.topk(2, dim=-1).values
        margins.append((top2[:, 0] - top2[:, 1]).clamp_min(0.0))
    return {
        "action": probability_ensemble(left_action, right_action),
        "direct": probability_ensemble(
            left["direct_risk"], right["direct_risk"]
        ),
        "safe_margin": 0.5 * (margins[0] + margins[1]),
        "labels": left["labels"],
        "risks": left["risks"],
    }


def risk_metrics(
    targets: list[dict[str, torch.Tensor]],
    config: tuple[float, float, float],
) -> list[dict]:
    """고정된 ensemble 위험 운용점의 분류 지표를 계산한다."""
    results = []
    for target in targets:
        risk = ensemble_risk_logits(target, config)
        results.append(classification_metrics(
            target["action"], risk, target["labels"], target["risks"]
        ))
    return results


def ensemble_risk_logits(
    target: dict[str, torch.Tensor],
    config: tuple[float, float, float] | list[float],
) -> torch.Tensor:
    """평균 direct risk와 행동 계층 확률을 고정 운용점에서 결합한다."""
    safe_weight, fusion, danger_bias = config
    direct = target["direct"].clone()
    direct[:, 0] += safe_weight * target["safe_margin"]
    direct[:, 2] += danger_bias
    action_risk = torch.stack([
        torch.logsumexp(target["action"][:, start:stop], dim=-1)
        for start, stop in CLASS_RANGES
    ], dim=-1)
    return (1.0 - fusion) * direct + fusion * action_risk


def choose_risk_config(
    targets: list[dict[str, torch.Tensor]],
) -> tuple[float, float, float]:
    """source-inner 평균과 최악값만으로 ensemble 위험 운용점을 선택한다."""
    scored = []
    for config in itertools.product(
        (0.0, 1.0, 2.0, 4.0, 6.0),
        (0.0, 0.15, 0.30, 0.50),
        (-1.0, -0.5, 0.0, 0.5, 1.0),
    ):
        utilities = []
        for metrics in risk_metrics(targets, config):
            diagnostic = cal12_site_selection_score(metrics)
            utilities.append(
                0.35 * metrics["risk_macro_f1"]
                + 0.35 * diagnostic["danger_balance"]
                + 0.30 * metrics["danger_recall"]
            )
        score = 0.5 * float(np.mean(utilities)) + 0.5 * float(np.min(utilities))
        scored.append((score, config))
    return max(scored, key=lambda item: item[0])[1]


def load_fold_model(run: Path, held_out: str, device: str):
    """outer 선택과 target 사용이 없는 fold checkpoint만 복원한다."""
    checkpoint = torch.load(
        run / f"selection_{held_out}.pt", map_location="cpu", weights_only=False
    )
    required_false = (
        "outer_holdout_used_for_selection", "target_subject_used", "sealed_yja_used"
    )
    if any(checkpoint.get(key) is not False for key in required_false):
        raise RuntimeError(f"unclean ensemble checkpoint: {run}")
    model = build_calibration_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main() -> None:
    """두 CAL60 fold를 독립 보정한 뒤 source-inner에서만 ensemble을 선택한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-name", default="CAL62-SOURCE-CLEAN-DEEP-ENSEMBLE",
        help="결과 JSON에 기록할 검증된 ensemble 버전 이름",
    )
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-seed", type=int, default=17018)
    parser.add_argument("--absence-trials", type=int, default=12)
    options = parser.parse_args()
    if options.run_dir_a.resolve() == options.run_dir_b.resolve():
        raise ValueError("deep ensemble requires two different source runs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL62")
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
    training = [json.loads((run / "result.json").read_text(encoding="utf-8"))
                for run in (options.run_dir_a, options.run_dir_b)]

    folds = {}
    for held_out in ("ajh", "mhw", "lmh"):
        models = [
            load_fold_model(run, held_out, device)
            for run in (options.run_dir_a, options.run_dir_b)
        ]
        info = training[0]["fold_results"][held_out]
        other_info = training[1]["fold_results"][held_out]
        for key in (
            "train_sites", "inner_validation_sites", "outer_test_sites"
        ):
            if list(info[key]) != list(other_info[key]):
                raise RuntimeError(f"ensemble fold split mismatch: {held_out}/{key}")
        train_sites = list(info["train_sites"])
        inner_sites = list(info["inner_validation_sites"])
        outer_sites = list(info["outer_test_sites"])
        requested_sites = train_sites + inner_sites + outer_sites
        embedded = [{
            site: embed_site(
                model, store, index, selected_rows, sites, site, device,
                options.support_seed, options.absence_seed, 2, None,
                options.absence_trials,
            ) for site in requested_sites
        } for model in models]
        libraries = [[{
            "site": site,
            "classes": class_prototypes(payload[site]),
            "anchors": payload[site]["anchors"],
        } for site in train_sites] for payload in embedded]
        action_configs = [
            choose_action_config([payload[site] for site in inner_sites], library)
            for payload, library in zip(embedded, libraries)
        ]

        def components_for(site_names: list[str]) -> list[dict[str, torch.Tensor]]:
            actions = []
            for payload, library, config in zip(
                embedded, libraries, action_configs
            ):
                _, current = evaluate_action_config(
                    [payload[site] for site in site_names], library, config
                )
                actions.append(current)
            return [
                target_components(
                    embedded[0][site], embedded[1][site],
                    actions[0][number], actions[1][number],
                ) for number, site in enumerate(site_names)
            ]

        inner = components_for(inner_sites)
        risk_config = choose_risk_config(inner)
        outer = components_for(outer_sites)
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_sites": inner_sites,
            "outer_sites": outer_sites,
            "action_configs": [list(config) for config in action_configs],
            "risk_config": list(risk_config),
            "inner_metrics": risk_metrics(inner, risk_config),
            "outer_metrics": risk_metrics(outer, risk_config),
            "outer_used_for_selection": False,
        }
        del models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    result = {
        "run": options.run_name,
        "folds": folds,
        "members": [str(options.run_dir_a), str(options.run_dir_b)],
        "anchor_classes": list(ANCHOR_CLASSES),
        "support_seed": options.support_seed,
        "absence_seed": options.absence_seed,
        "absence_trials": options.absence_trials,
        "ensemble_probability_weights": [0.5, 0.5],
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
