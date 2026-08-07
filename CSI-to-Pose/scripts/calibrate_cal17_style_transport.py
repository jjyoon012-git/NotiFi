"""CAL17 style transport를 source nested fold에서만 선택하고 평가한다."""

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
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal14 import CAL14InvariantCosine  # noqa: E402
from notifi_pose.cal17 import (  # noqa: E402
    ANCHOR_CLASSES,
    cal17_action,
    cal17_risk,
    transported_logits,
)
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402
from train_cal20_source_folds import (  # noqa: E402
    SOURCE_SITES,
    cal12_site_selection_score,
)


def select_support_shots(
    rows: np.ndarray,
    index: pd.DataFrame,
    seed: int,
    shots_per_prompt: int,
) -> np.ndarray:
    """각 기본동작에서 동일 seed 순서의 지정된 개수만 calibration으로 고른다."""
    if shots_per_prompt < 1:
        raise ValueError("shots_per_prompt must be positive")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in MOTION_PROMPT_CLASSES:
        candidates = rows[index.class_id.iloc[rows].to_numpy() == class_id]
        candidates = candidates[
            np.argsort(index.trial_id.iloc[candidates].to_numpy())
        ]
        if len(candidates) < shots_per_prompt:
            raise RuntimeError(f"class {class_id} has too few support trials")
        selected.extend(
            rng.permutation(candidates)[:shots_per_prompt].tolist()
        )
    return np.asarray(sorted(selected), dtype=np.int64)


@torch.no_grad()
def embed_site(
    model: CAL14InvariantCosine,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
    device: str,
    support_seed: int = 17017,
    absence_seed: int = 17018,
    shots_per_prompt: int = 2,
    query_exclusion_shots: int | None = None,
    absence_trials: int = 2,
) -> dict[str, torch.Tensor]:
    """한 site의 고정 support와 모든 query를 GT가 없는 추론 방식으로 embedding한다."""
    rows = base.site_rows(selected_rows, sites, site)
    support = select_support_shots(
        rows, index, support_seed, shots_per_prompt
    )
    absence = base.select_absence(
        site, index, absence_seed, trials=absence_trials
    )
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    support_labels = torch.tensor(
        index.class_id.iloc[support].to_numpy(), device=device
    )
    support_output = model(
        support_csi, support_mask,
        support_csi, support_mask, support_labels,
        absence_csi, absence_mask,
    )
    support_embedding = support_output["embedding"].cpu()
    absence_output = model(
        absence_csi, absence_mask,
        support_csi, support_mask, support_labels,
        absence_csi, absence_mask,
    )
    absence_embedding = absence_output["embedding"].cpu()
    anchor_prototypes = torch.stack([
        F.normalize(absence_embedding.mean(0), dim=0)
        if class_id == 6 else F.normalize(
            support_embedding[support_labels.cpu() == class_id].mean(0), dim=0
        )
        for class_id in ANCHOR_CLASSES
    ])
    excluded_support = select_support_shots(
        rows, index, support_seed,
        query_exclusion_shots or shots_per_prompt,
    )
    support_set = set(excluded_support.tolist())
    query = np.asarray(
        [row for row in rows if row not in support_set], dtype=np.int64
    )
    payload = {
        "action": [], "direct_risk": [], "embedding": [], "motion": [],
        "features": [], "frame_mask": [],
    }
    for start in range(0, len(query), 24):
        batch = query[start:start + 24]
        query_csi, query_mask = store.get(batch, device)
        output = model(
            query_csi, query_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        for key, output_key in (
            ("action", "action_logits"),
            ("direct_risk", "direct_risk_logits"),
            ("embedding", "embedding"),
            ("motion", "pose_motion"),
            ("features", "query_features"),
            ("frame_mask", "query_frame_mask"),
        ):
            payload[key].append(output[output_key].cpu())
    return {
        **{key: torch.cat(value) for key, value in payload.items()},
        "anchors": anchor_prototypes,
        "absence_embedding": absence_embedding,
        "labels": torch.tensor(index.class_id.iloc[query].to_numpy()).long(),
        "risks": torch.tensor(index.risk_id.iloc[query].to_numpy()).long(),
        "query_rows": torch.from_numpy(query),
    }


def class_prototypes(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    """source label로 각 행동의 평균 embedding을 만들어 배포 library에 저장한다."""
    prototypes = []
    for class_id in range(C.N_CLASSES):
        keep = payload["labels"] == class_id
        if class_id == 6:
            prototypes.append(F.normalize(
                payload["absence_embedding"].mean(0), dim=0
            ))
        elif not bool(keep.any()):
            raise RuntimeError(f"source library has no class {class_id}")
        else:
            prototypes.append(F.normalize(
                payload["embedding"][keep].mean(0), dim=0
            ))
    return torch.stack(prototypes)


def evaluate_action_config(
    targets: list[dict[str, torch.Tensor]],
    source_library: list[dict[str, torch.Tensor]],
    config: tuple[float, float, float, float, float],
) -> tuple[list[dict], list[torch.Tensor]]:
    """base와 transport log-probability를 섞은 행동 성능을 계산한다."""
    strength, anchor_temp, proto_temp, site_temp, mixture = config
    metrics = []
    actions = []
    for target in targets:
        transport = transported_logits(
            target, source_library, strength, anchor_temp,
            proto_temp, site_temp,
        )
        action = (
            (1.0 - mixture) * target["action"].log_softmax(-1)
            + mixture * transport.log_softmax(-1)
        )
        actions.append(action)
        metrics.append(classification_metrics(
            action, target["direct_risk"], target["labels"], target["risks"]
        ))
    return metrics, actions


def choose_action_config(
    targets: list[dict[str, torch.Tensor]],
    source_library: list[dict[str, torch.Tensor]],
) -> tuple[float, float, float, float, float]:
    """inner site의 평균과 최악 action utility로 transport 강도만 선택한다."""
    candidates = itertools.product(
        (0.0, 0.5, 1.0, 1.5),
        (0.05, 0.10, 0.20, 0.40),
        (0.05, 0.10, 0.20),
        (0.02, 0.05, 0.10, 0.25),
        (0.0, 0.25, 0.50, 0.75, 1.0),
    )
    scored = []
    for config in candidates:
        metrics, _ = evaluate_action_config(targets, source_library, config)
        utility = [
            0.55 * item["action_macro_f1"]
            + 0.20 * item["action_accuracy"]
            + 0.25 * item["danger_action_accuracy"]
            for item in metrics
        ]
        score = 0.5 * float(np.mean(utility)) + 0.5 * float(np.min(utility))
        scored.append((score, config))
    return max(scored, key=lambda item: item[0])[1]


def choose_risk_config(
    model: CAL14InvariantCosine,
    targets: list[dict[str, torch.Tensor]],
    actions: list[torch.Tensor],
) -> tuple[float, float, float]:
    """inner site에서 direct risk와 action-derived risk의 혼합 및 danger bias를 선택한다."""
    scored = []
    for safe_weight, fusion, danger_bias in itertools.product(
        (0.0, 1.0, 2.0, 4.0, 6.0),
        (0.0, 0.15, 0.30, 0.50),
        (-1.0, -0.5, 0.0, 0.5, 1.0),
    ):
        utilities = []
        for target, action in zip(targets, actions):
            direct = target["direct_risk"].clone()
            similarity = F.normalize(target["embedding"], dim=-1) @ target[
                "anchors"
            ].transpose(0, 1)
            top2 = similarity.topk(2, dim=-1).values
            direct[:, 0] += safe_weight * (
                top2[:, 0] - top2[:, 1]
            ).clamp_min(0.0)
            direct[:, 2] += danger_bias
            risk = (1.0 - fusion) * direct + fusion * model.action_to_risk(action)
            metrics = classification_metrics(
                action, risk, target["labels"], target["risks"]
            )
            diagnostic = cal12_site_selection_score(metrics)
            utilities.append(
                0.35 * metrics["risk_macro_f1"]
                + 0.35 * diagnostic["danger_balance"]
                + 0.30 * metrics["danger_recall"]
            )
        score = 0.5 * float(np.mean(utilities)) + 0.5 * float(np.min(utilities))
        scored.append((score, safe_weight, fusion, danger_bias))
    _, safe_weight, fusion, danger_bias = max(scored)
    return safe_weight, fusion, danger_bias


def final_metrics(
    model: CAL14InvariantCosine,
    targets: list[dict[str, torch.Tensor]],
    source_library: list[dict[str, torch.Tensor]],
    action_config: tuple[float, float, float, float, float],
    risk_config: tuple[float, float, float],
) -> list[dict]:
    """고정된 inner 설정으로 outer query를 한 번만 평가한다."""
    _, actions = evaluate_action_config(targets, source_library, action_config)
    safe_weight, fusion, danger_bias = risk_config
    results = []
    for target, action in zip(targets, actions):
        direct = target["direct_risk"].clone()
        similarity = F.normalize(target["embedding"], dim=-1) @ target[
            "anchors"
        ].transpose(0, 1)
        top2 = similarity.topk(2, dim=-1).values
        direct[:, 0] += safe_weight * (
            top2[:, 0] - top2[:, 1]
        ).clamp_min(0.0)
        direct[:, 2] += danger_bias
        risk = (1.0 - fusion) * direct + fusion * model.action_to_risk(action)
        results.append(classification_metrics(
            action, risk, target["labels"], target["risks"]
        ))
    return results


def main() -> None:
    """각 outer subject를 숨긴 채 source 행동 prototype 이동을 선택하고 검증한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-seed", type=int, default=17018)
    parser.add_argument("--shots-per-prompt", type=int, default=2)
    parser.add_argument("--query-exclusion-shots", type=int)
    parser.add_argument("--absence-trials", type=int, default=2)
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL17 selection")
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
    training_result = json.loads((options.run_dir / "result.json").read_text(
        encoding="utf-8"
    ))
    folds = {}
    for held_out in ("ajh", "mhw", "lmh"):
        checkpoint = torch.load(
            options.run_dir / f"selection_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        if checkpoint.get("outer_holdout_used_for_selection") is not False:
            raise RuntimeError("checkpoint is not nested-source clean")
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        fold_info = training_result["fold_results"][held_out]
        train_sites = list(fold_info["train_sites"])
        inner_sites = list(fold_info["inner_validation_sites"])
        outer_sites = list(fold_info["outer_test_sites"])
        embedded = {
            site: embed_site(
                model, store, index, selected_rows, sites, site, device,
                options.support_seed, options.absence_seed,
                options.shots_per_prompt, options.query_exclusion_shots,
                options.absence_trials,
            )
            for site in train_sites + inner_sites + outer_sites
        }
        library = [{
            "site": site,
            "classes": class_prototypes(embedded[site]),
            "anchors": embedded[site]["anchors"],
        } for site in train_sites]
        inner_targets = [embedded[site] for site in inner_sites]
        action_config = choose_action_config(inner_targets, library)
        _, inner_actions = evaluate_action_config(
            inner_targets, library, action_config
        )
        risk_config = choose_risk_config(model, inner_targets, inner_actions)
        inner_metrics = final_metrics(
            model, inner_targets, library, action_config, risk_config
        )
        outer_metrics = final_metrics(
            model, [embedded[site] for site in outer_sites], library,
            action_config, risk_config,
        )
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_sites": inner_sites,
            "outer_sites": outer_sites,
            "action_config": {
                key: value for key, value in zip((
                    "strength", "anchor_temperature", "prototype_temperature",
                    "site_temperature", "mixture",
                ), action_config)
            },
            "risk_config": {
                "safe_weight": risk_config[0],
                "fusion": risk_config[1],
                "danger_bias": risk_config[2],
            },
            "inner_metrics": inner_metrics,
            "outer_metrics": outer_metrics,
            "outer_used_for_selection": False,
        }
    result = {
        "run": "CAL17-SAFE-STYLE-TRANSPORT",
        "folds": folds,
        "anchor_classes": list(ANCHOR_CLASSES),
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "support_seed": options.support_seed,
        "absence_seed": options.absence_seed,
        "absence_trials": options.absence_trials,
        "shots_per_prompt": options.shots_per_prompt,
        "query_exclusion_shots": (
            options.query_exclusion_shots or options.shots_per_prompt
        ),
    }
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
