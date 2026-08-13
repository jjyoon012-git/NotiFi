"""CAL44에 support-supervised ridge alignment를 붙여 source nested-LOSO로 평가한다."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import (  # noqa: E402
    class_prototypes,
    evaluate_action_config,
)
from calibrate_cal62_deep_ensemble import (  # noqa: E402
    load_fold_model,
    probability_ensemble,
)
from evaluate_cal44_fall_support import prepare_target, summarize  # noqa: E402
from evaluate_cal44_phase_risk import embed_fold  # noqa: E402
from notifi_ai_v2.support_alignment import aligned_logits  # noqa: E402
from notifi_ai_v2.support_alignment import action_to_risk_log_probability  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.danger_support import (  # noqa: E402
    apply_danger_support,
    support_evidence,
)
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402


WARNING_CLASSES = (9, 10, 11)


def warning_support(
    payload: dict[str, torch.Tensor],
    index: pd.DataFrame,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """warning 3종에서 각각 한 trial을 support로 뽑고 query mask를 반환한다."""
    rows = payload["query_rows"].numpy()
    labels = index.class_id.iloc[rows].to_numpy()
    trial_ids = index.trial_id.iloc[rows].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in WARNING_CLASSES:
        positions = np.flatnonzero(labels == class_id)
        positions = positions[np.argsort(trial_ids[positions])]
        if len(positions) < 2:
            raise RuntimeError(f"warning class {class_id} has too few trials")
        selected.append(int(rng.permutation(positions)[0]))
    support = torch.tensor(selected, dtype=torch.long)
    keep = torch.ones(len(rows), dtype=torch.bool)
    keep[support] = False
    labels_at_support = payload["labels"][support]
    prototypes = torch.stack([
        payload["embedding"][support][labels_at_support == class_id].mean(0)
        for class_id in WARNING_CLASSES
    ])
    return prototypes, keep


def evaluate_sites(
    embedded: list[dict[str, dict[str, torch.Tensor]]],
    source_sites: list[str],
    target_sites: list[str],
    configs: list[list[float]],
    index: pd.DataFrame,
    support_seed: int,
    regularization: float,
    prototype_temperature: float,
    site_temperature: float,
    mixture: float,
    direction: str,
    danger_bias: float = 1.5,
    danger_temperature: float = 0.10,
    danger_subtype_weight: float = 0.50,
    deployment_profiles: list[list[list[float]]] | None = None,
    use_warning_support: bool = False,
    risk_fusion: float = 0.0,
) -> list[dict]:
    """라벨 support로 맞춘 ridge 행동 좌표와 고정 risk 경로를 평가한다."""
    libraries = [
        [
            {
                "site": site,
                "classes": class_prototypes(payload[site]),
                "anchors": payload[site]["anchors"],
            }
            for site in source_sites
        ]
        for payload in embedded
    ]
    base_actions = []
    deployment_actions = None
    if deployment_profiles is None:
        for payload, library, config in zip(embedded, libraries, configs):
            _, actions = evaluate_action_config(
                [payload[site] for site in target_sites], library, tuple(config)
            )
            base_actions.append(actions)
    else:
        deployment_actions = []
        for profile in deployment_profiles:
            members = []
            for payload, library, config in zip(embedded, libraries, profile):
                _, actions = evaluate_action_config(
                    [payload[site] for site in target_sites], library, tuple(config)
                )
                members.append(actions)
            deployment_actions.append([
                probability_ensemble(left, right)
                for left, right in zip(members[0], members[1])
            ])

    danger_config = {
        "temperature": float(danger_temperature),
        "subtype_weight": float(danger_subtype_weight),
        "action_margin_gain": 0.0,
        "action_bias": 0.0,
        "risk_margin_gain": 0.0,
        "risk_bias": 0.0,
    }
    rows = []
    for number, site in enumerate(target_sites):
        left = embedded[0][site]
        right = embedded[1][site]
        left_target = prepare_target(left, index, support_seed + 1000)
        right_target = prepare_target(right, index, support_seed + 1000)
        if not torch.equal(left_target["keep"], right_target["keep"]):
            raise RuntimeError("danger support selection mismatch")
        left_warning = right_warning = None
        warning_keep = torch.ones_like(left_target["keep"])
        if use_warning_support:
            left_warning, warning_keep = warning_support(
                left, index, support_seed + 2000
            )
            right_warning, right_warning_keep = warning_support(
                right, index, support_seed + 2000
            )
            if not torch.equal(warning_keep, right_warning_keep):
                raise RuntimeError("warning support selection mismatch")
        keep = left_target["keep"] & warning_keep
        ridge_actions = []
        for member, payload, target, library in zip(
            range(2), (left, right), (left_target, right_target), libraries
        ):
            ridge = aligned_logits(
                payload["embedding"][keep],
                payload["anchors"],
                target["danger_prototypes"],
                library,
                regularization,
                prototype_temperature,
                site_temperature,
                direction,
                target_warning=(left_warning, right_warning)[member],
            )
            ridge_actions.append(ridge)
        ridge_action = probability_ensemble(ridge_actions[0], ridge_actions[1])
        if deployment_actions is None:
            base_action = probability_ensemble(
                base_actions[0][number][keep], base_actions[1][number][keep]
            )
        else:
            profile_actions = [
                profile[number][keep].log_softmax(-1)
                for profile in deployment_actions
            ]
            base_action = torch.logsumexp(
                torch.stack(profile_actions), dim=0
            ) - math.log(len(profile_actions))
        if mixture <= 0.0:
            action = base_action
        elif mixture >= 1.0:
            action = ridge_action
        else:
            action = torch.logaddexp(
                base_action.log_softmax(-1) + math.log1p(-mixture),
                ridge_action.log_softmax(-1) + math.log(mixture),
            )
        risk = left["risk"][keep].clone()
        risk[:, 2] += float(danger_bias)
        evidence = [
            support_evidence(
                payload["embedding"][keep],
                payload["anchors"][:-1],
                target["danger_prototypes"],
                danger_temperature,
            )
            for payload, target in ((left, left_target), (right, right_target))
        ]
        action, risk, _ = apply_danger_support(
            action, risk, evidence, danger_config
        )
        if risk_fusion > 0.0:
            if not 0.0 <= risk_fusion <= 1.0:
                raise ValueError("risk fusion must be in [0,1]")
            action_risk = action_to_risk_log_probability(action)
            if risk_fusion >= 1.0:
                risk = action_risk
            else:
                risk = torch.logaddexp(
                    risk.log_softmax(-1) + math.log1p(-risk_fusion),
                    action_risk + math.log(risk_fusion),
                )
        metric = classification_metrics(
            action, risk, left["labels"][keep], left["risks"][keep]
        )
        metric["site"] = site
        rows.append(metric)
    return rows


def score(metrics: dict) -> float:
    """행동 macro-F1을 중심으로 전체 정확도와 danger subtype도 함께 선택한다."""
    return float(
        0.70 * metrics["action_macro_f1"]
        + 0.20 * metrics["action_accuracy"]
        + 0.10 * metrics["danger_action_accuracy"]
    )


def risk_score(metrics: dict) -> float:
    """위험 macro-F1과 danger recall을 보되 safe 오경보를 함께 벌점으로 준다."""
    return float(
        metrics["risk_macro_f1"]
        + 0.25 * metrics["danger_recall"]
        - 0.25 * metrics["safe_to_danger_rate"]
    )


def main() -> None:
    """inner source 환경에서 ridge 설정을 고르고 outer 사람에게 한 번 적용한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--cal40-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument(
        "--fixed-config",
        type=Path,
        help="5-seed source-inner 탐색으로 미리 고정한 fold별 ridge 설정",
    )
    parser.add_argument(
        "--regularizations", type=float, nargs="+", default=(0.05, 0.2, 1.0, 5.0)
    )
    parser.add_argument(
        "--prototype-temperatures", type=float, nargs="+", default=(0.05, 0.10, 0.20)
    )
    parser.add_argument(
        "--site-temperatures", type=float, nargs="+", default=(0.02, 0.10, 0.25)
    )
    parser.add_argument(
        "--mixtures", type=float, nargs="+", default=(0.25, 0.50, 0.75, 1.0)
    )
    parser.add_argument(
        "--directions", nargs="+", default=("source_to_target", "target_to_source")
    )
    parser.add_argument(
        "--average-action-profiles",
        action="store_true",
        help="실제 단일 bundle처럼 세 fold의 고정 action 설정을 평균한다",
    )
    parser.add_argument(
        "--warning-support",
        action="store_true",
        help="warning 3종을 각각 1회 calibration support로 사용한다",
    )
    parser.add_argument(
        "--risk-fusions", type=float, nargs="+", default=(0.0,)
    )
    parser.add_argument(
        "--danger-biases", type=float, nargs="+", default=(1.5,)
    )
    options = parser.parse_args()

    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    locked = json.loads(options.cal40_result.read_text(encoding="utf-8"))[
        "fixed_linear_configs"
    ]
    fixed_configs = (
        json.loads(options.fixed_config.read_text(encoding="utf-8"))["fold_configs"]
        if options.fixed_config is not None
        else None
    )
    deployment_profiles = (
        [locked[name] for name in ("ajh", "mhw", "lmh")]
        if options.average_action_profiles else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter support ridge selection")
    row_sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(row_sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
    absence_rows = np.concatenate(
        [
            np.flatnonzero(
                (
                    (index.subject == site.split("_")[0])
                    & (index.environment == site.split("_")[1])
                    & (index.task == C.TASK_CLS)
                    & (index.class_id == 6)
                    & index.cache_ok
                ).to_numpy()
            )
            for site in all_sites
        ]
    )
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))

    folds = {}
    selected_rows_metrics = []
    baseline_rows = []
    for held_out in ("ajh", "mhw", "lmh"):
        train_sites, validation_sites, outer_sites = nested_site_split(
            all_sites, held_out
        )
        models = [
            load_fold_model(run, held_out, device)
            for run in (options.run_dir_a, options.run_dir_b)
        ]
        requested = train_sites + validation_sites + outer_sites
        embedded = embed_fold(
            models,
            requested,
            store,
            index,
            selected_rows,
            row_sites,
            device,
            options.support_seed,
            options.absence_trials,
        )
        if fixed_configs is not None:
            chosen = dict(fixed_configs[held_out])
            risk_candidates = []
            for risk_fusion in options.risk_fusions:
                for danger_bias in options.danger_biases:
                    metrics = summarize(evaluate_sites(
                        embedded,
                        train_sites,
                        validation_sites,
                        locked[held_out],
                        index,
                        options.support_seed,
                        chosen["regularization"],
                        chosen["prototype_temperature"],
                        chosen["site_temperature"],
                        chosen["mixture"],
                        chosen["direction"],
                        danger_bias=danger_bias,
                        deployment_profiles=deployment_profiles,
                        use_warning_support=options.warning_support,
                        risk_fusion=risk_fusion,
                    ))
                    risk_candidates.append({
                        "risk_fusion": float(risk_fusion),
                        "danger_bias": float(danger_bias),
                        "metrics": metrics,
                        "score": risk_score(metrics),
                    })
            selected_risk = max(
                risk_candidates,
                key=lambda item: (
                    item["score"],
                    item["metrics"]["risk_macro_f1"],
                    -item["metrics"]["safe_to_danger_rate"],
                ),
            )
            chosen["risk_fusion"] = selected_risk["risk_fusion"]
            chosen["danger_bias"] = selected_risk["danger_bias"]
            validation_metrics = selected_risk["metrics"]
            chosen["validation"] = validation_metrics
            chosen["selection_score"] = score(validation_metrics)
        else:
            candidates = []
            for direction in options.directions:
                for regularization in options.regularizations:
                    for prototype_temperature in options.prototype_temperatures:
                        for site_temperature in options.site_temperatures:
                            for mixture in options.mixtures:
                                metrics = summarize(evaluate_sites(
                                    embedded,
                                    train_sites,
                                    validation_sites,
                                    locked[held_out],
                                    index,
                                    options.support_seed,
                                    regularization,
                                    prototype_temperature,
                                    site_temperature,
                                    mixture,
                                    direction,
                                    deployment_profiles=deployment_profiles,
                                    use_warning_support=options.warning_support,
                                ))
                                candidates.append({
                                    "direction": direction,
                                    "regularization": float(regularization),
                                    "prototype_temperature": float(prototype_temperature),
                                    "site_temperature": float(site_temperature),
                                    "mixture": float(mixture),
                                    "validation": metrics,
                                    "selection_score": score(metrics),
                                })
            chosen = max(
                candidates,
                key=lambda item: (
                    item["selection_score"],
                    item["validation"]["action_macro_f1"],
                    -item["mixture"],
                ),
            )
        outer = evaluate_sites(
            embedded,
            train_sites,
            outer_sites,
            locked[held_out],
            index,
            options.support_seed,
            chosen["regularization"],
            chosen["prototype_temperature"],
            chosen["site_temperature"],
            chosen["mixture"],
            chosen["direction"],
            danger_bias=chosen.get("danger_bias", 1.5),
            deployment_profiles=deployment_profiles,
            use_warning_support=options.warning_support,
            risk_fusion=chosen.get("risk_fusion", 0.0),
        )
        baseline = evaluate_sites(
            embedded,
            train_sites,
            outer_sites,
            locked[held_out],
            index,
            options.support_seed,
            1.0,
            0.1,
            0.1,
            0.0,
            "source_to_target",
            danger_bias=1.5,
            deployment_profiles=deployment_profiles,
            use_warning_support=options.warning_support,
            risk_fusion=0.0,
        )
        selected_rows_metrics.extend(outer)
        baseline_rows.extend(baseline)
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": validation_sites,
            "outer_test_sites": outer_sites,
            "selected": chosen,
            "outer": summarize(outer),
            "baseline": summarize(baseline),
            "outer_used_for_selection": False,
        }
        print(
            f"{held_out}: {chosen['direction']} reg={chosen['regularization']:.2g} "
            f"temp={chosen['prototype_temperature']:.2g} mix={chosen['mixture']:.2f}",
            flush=True,
        )

    result = {
        "run": "CAL44-SUPPORT-RIDGE-ALIGNMENT",
        "protocol": "source nested-LOSO; inner-site selection; support excluded",
        "support_seed": int(options.support_seed),
        "configuration_source": (
            str(options.fixed_config) if options.fixed_config is not None
            else "source-inner grid for this support seed"
        ),
        "action_profile_mode": (
            "deployment_average" if options.average_action_profiles
            else "nested_fold_profile"
        ),
        "baseline": summarize(baseline_rows),
        "selected": summarize(selected_rows_metrics),
        "folds": folds,
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "warning_support": bool(options.warning_support),
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"baseline": result["baseline"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
