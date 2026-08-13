"""CAL44의 위상 사용량과 위험 경계를 엄격한 source nested-LOSO로 선택한다."""

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
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))

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
from evaluate_cal44_fall_support import prepare_target, summarize  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.danger_support import (  # noqa: E402
    apply_danger_support,
    support_evidence,
)
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402


SAFE_ANCHOR_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8, 6)


def set_phase_strength(models: list[torch.nn.Module], strength: float) -> None:
    """학습된 모델이 추론에서 사용할 시간 변화 위상의 비율을 설정한다."""
    for model in models:
        model.canonicalizer.phase_strength = float(strength)


def embed_fold(
    models: list[torch.nn.Module],
    sites: list[str],
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    row_sites: np.ndarray,
    device: str,
    support_seed: int,
    absence_trials: int,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """두 CAL44 인코더로 지정된 source 사이트를 같은 support 조건에서 임베딩한다."""
    return [
        {
            site: embed_site(
                model,
                store,
                index,
                selected_rows,
                row_sites,
                site,
                device,
                support_seed,
                support_seed + 1,
                2,
                None,
                absence_trials,
            )
            for site in sites
        }
        for model in models
    ]


def evaluate_sites(
    embedded: list[dict[str, dict[str, torch.Tensor]]],
    source_sites: list[str],
    target_sites: list[str],
    configs: list[list[float]],
    index: pd.DataFrame,
    support_seed: int,
    danger_bias: float,
    temperature: float,
    subtype_weight: float,
    safe_support_weight: float = 0.0,
    safe_support_temperature: float = 0.10,
    risk_embedded: list[dict[str, dict[str, torch.Tensor]]] | None = None,
) -> list[dict]:
    """source library와 겹치지 않는 target query에서 행동과 위험 지표를 계산한다."""
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
    member_actions = []
    for payload, library, config in zip(embedded, libraries, configs):
        _, actions = evaluate_action_config(
            [payload[site] for site in target_sites], library, tuple(config)
        )
        member_actions.append(actions)

    danger_config = {
        "temperature": float(temperature),
        "subtype_weight": float(subtype_weight),
        "action_margin_gain": 0.0,
        "action_bias": 0.0,
        "risk_margin_gain": 0.0,
        "risk_bias": 0.0,
    }
    rows = []
    for number, site in enumerate(target_sites):
        left = embedded[0][site]
        right = embedded[1][site]
        if not torch.equal(left["query_rows"], right["query_rows"]):
            raise RuntimeError("CAL44 ensemble query order mismatch")
        left_target = prepare_target(left, index, support_seed + 1000)
        right_target = prepare_target(right, index, support_seed + 1000)
        keep = left_target["keep"]
        if not torch.equal(keep, right_target["keep"]):
            raise RuntimeError("danger support selection mismatch")
        action = probability_ensemble(
            member_actions[0][number][keep], member_actions[1][number][keep]
        )
        if safe_support_weight > 0.0:
            safe_members = []
            for payload in (left, right):
                similarity = F.normalize(
                    payload["embedding"][keep], dim=-1
                ) @ F.normalize(payload["anchors"], dim=-1).transpose(0, 1)
                safe_members.append(
                    (similarity / max(safe_support_temperature, 1e-4)).log_softmax(-1)
                )
            safe_support = torch.logaddexp(
                safe_members[0], safe_members[1]
            ) - math.log(2.0)
            ordered_safe = action.new_full((len(action), 9), -float("inf"))
            for anchor_position, class_id in enumerate(SAFE_ANCHOR_CLASSES):
                ordered_safe[:, class_id] = safe_support[:, anchor_position]
            base_safe = action[:, :9].log_softmax(-1)
            weight = float(safe_support_weight)
            safe_conditional = torch.logaddexp(
                base_safe + math.log1p(-weight),
                ordered_safe + math.log(weight),
            )
            safe_mass = torch.logsumexp(action[:, :9], dim=-1, keepdim=True)
            action = action.clone()
            action[:, :9] = safe_mass + safe_conditional
        risk_source = risk_embedded[0][site] if risk_embedded is not None else left
        if not torch.equal(left["query_rows"], risk_source["query_rows"]):
            raise RuntimeError("action and risk query order mismatch")
        risk = risk_source["risk"][keep].clone()
        risk[:, 2] += float(danger_bias)
        labels = left["labels"][keep]
        risks = left["risks"][keep]
        evidence = [
            support_evidence(
                left["embedding"][keep],
                left["anchors"][:-1],
                left_target["danger_prototypes"],
                temperature,
            ),
            support_evidence(
                right["embedding"][keep],
                right["anchors"][:-1],
                right_target["danger_prototypes"],
                temperature,
            ),
        ]
        action, risk, _ = apply_danger_support(
            action, risk, evidence, danger_config
        )
        metric = classification_metrics(action, risk, labels, risks)
        metric["site"] = site
        rows.append(metric)
    return rows


def selection_score(metrics: dict) -> float:
    """위험 경로와 분리된 행동 경로의 세부 동작 판별력을 평가한다."""
    return float(
        0.70 * metrics["action_macro_f1"]
        + 0.20 * metrics["action_accuracy"]
        + 0.10 * metrics["danger_action_accuracy"]
    )


def candidate_key(candidate: dict) -> tuple[float, float, float, float]:
    """동률이면 오탐과 원본 위상에서 덜 벗어나는 후보를 우선한다."""
    metrics = candidate["validation"]
    return (
        float(candidate["selection_score"]),
        -float(metrics["safe_to_danger_rate"]),
        float(metrics["risk_macro_f1"]),
        float(candidate["phase_strength"]),
    )


def main() -> None:
    """inner site로만 위상과 위험 경계를 고르고 outer subject를 한 번 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--cal40-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument(
        "--phase-strengths", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0)
    )
    parser.add_argument(
        "--danger-biases", type=float, nargs="+", default=(0.0, 0.5, 1.0, 1.5, 2.0)
    )
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--subtype-weight", type=float, default=0.50)
    parser.add_argument(
        "--safe-support-weights", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75)
    )
    parser.add_argument(
        "--safe-support-temperatures", type=float, nargs="+", default=(0.05, 0.10, 0.20)
    )
    options = parser.parse_args()

    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    locked = json.loads(options.cal40_result.read_text(encoding="utf-8"))[
        "fixed_linear_configs"
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter phase-risk selection")
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
    outer_rows = []
    baseline_rows = []
    for held_out in ("ajh", "mhw", "lmh"):
        train_sites, validation_sites, outer_sites = nested_site_split(
            all_sites, held_out
        )
        models = [
            load_fold_model(run, held_out, device)
            for run in (options.run_dir_a, options.run_dir_b)
        ]
        candidates = []
        phase_payloads = {}
        requested = train_sites + validation_sites + outer_sites
        for phase_strength in options.phase_strengths:
            set_phase_strength(models, phase_strength)
            phase_payloads[float(phase_strength)] = embed_fold(
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
        if 1.0 not in phase_payloads:
            raise ValueError("decoupled risk evaluation requires phase strength 1.0")
        for phase_strength in options.phase_strengths:
            for danger_bias in options.danger_biases:
                for safe_weight in options.safe_support_weights:
                    temperatures = (
                        (options.safe_support_temperatures[0],)
                        if safe_weight == 0.0
                        else options.safe_support_temperatures
                    )
                    for safe_temperature in temperatures:
                        metrics = summarize(
                            evaluate_sites(
                                phase_payloads[float(phase_strength)],
                                train_sites,
                                validation_sites,
                                locked[held_out],
                                index,
                                options.support_seed,
                                danger_bias,
                                options.temperature,
                            options.subtype_weight,
                            safe_weight,
                            safe_temperature,
                            phase_payloads[1.0],
                            )
                        )
                        candidate = {
                            "phase_strength": float(phase_strength),
                            "danger_bias": float(danger_bias),
                            "safe_support_weight": float(safe_weight),
                            "safe_support_temperature": float(safe_temperature),
                            "validation": metrics,
                        }
                        candidate["selection_score"] = selection_score(metrics)
                        candidates.append(candidate)
        selected_candidate = max(candidates, key=candidate_key)
        selected_payload = phase_payloads[selected_candidate["phase_strength"]]
        outer = evaluate_sites(
            selected_payload,
            train_sites,
            outer_sites,
            locked[held_out],
            index,
            options.support_seed,
            selected_candidate["danger_bias"],
            options.temperature,
            options.subtype_weight,
            selected_candidate["safe_support_weight"],
            selected_candidate["safe_support_temperature"],
            phase_payloads[1.0],
        )
        baseline = evaluate_sites(
            phase_payloads[1.0],
            train_sites,
            outer_sites,
            locked[held_out],
            index,
            options.support_seed,
            1.5,
            options.temperature,
            options.subtype_weight,
        )
        outer_rows.extend(outer)
        baseline_rows.extend(baseline)
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": validation_sites,
            "outer_test_sites": outer_sites,
            "selected": selected_candidate,
            "outer": summarize(outer),
            "baseline": summarize(baseline),
            "candidates": sorted(candidates, key=candidate_key, reverse=True),
            "outer_used_for_selection": False,
        }
        print(
            f"{held_out}: phase={selected_candidate['phase_strength']:.2f} "
            f"danger_bias={selected_candidate['danger_bias']:.2f} "
            f"safe_weight={selected_candidate['safe_support_weight']:.2f}",
            flush=True,
        )

    result = {
        "run": "CAL44-PHASE-RISK-NESTED-SELECTION",
        "protocol": "source nested-LOSO; inner-site selection; danger support excluded",
        "support_seed": int(options.support_seed),
        "selection_objective": (
            "0.70 action_macro_f1 + 0.20 action_accuracy "
            "+ 0.10 danger_action_accuracy; risk uses locked phase-1 path"
        ),
        "baseline": summarize(baseline_rows),
        "selected": summarize(outer_rows),
        "folds": folds,
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"baseline": result["baseline"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
