"""Select static/dynamic calibration head blending on source-held-out lmh only."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_cal8_raw_source as train
from notifi_pose.meta_calibration import (
    MOTION_PROMPT_CLASSES,
    PROMPT_CLASSES,
    RawSupportConditionedModel,
)
from notifi_pose.tools.train_dynamic_motion import classification_metrics


PROJECT = Path(__file__).resolve().parents[1]
STATIC_SHOTS = {0: 2, 1: 2, 2: 2, 3: 2}
DYNAMIC_SHOTS = {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 7: 1, 8: 1}


def select_support(
    rows: np.ndarray,
    index: pd.DataFrame,
    shots: dict[int, int],
    seed: int,
) -> np.ndarray:
    """고정 seed와 class별 shot 수만으로 calibration support를 선택한다."""
    rng = np.random.default_rng(seed)
    selected = []
    for class_id, count in shots.items():
        candidates = rows[index.class_id.iloc[rows].to_numpy() == class_id]
        candidates = candidates[np.argsort(index.trial_id.iloc[candidates].to_numpy())]
        selected.extend(rng.permutation(candidates)[:count].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def load_model(
    path: Path,
    prompt_classes: tuple[int, ...],
    device: str,
) -> RawSupportConditionedModel:
    """과거 checkpoint에도 명시적으로 prompt contract를 붙여 모델을 복원한다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    config["prompt_classes"] = prompt_classes
    model = RawSupportConditionedModel(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: RawSupportConditionedModel,
    store: train.RawStore,
    index: pd.DataFrame,
    support: np.ndarray,
    absence: np.ndarray,
    query: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """한 calibration support를 고정한 채 query 전체의 두 head logit을 계산한다."""
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    support_labels = torch.tensor(
        index.class_id.iloc[support].to_numpy(), device=device
    )
    action, risk = [], []
    for start in range(0, len(query), 32):
        batch = query[start:start + 32]
        query_csi, query_mask = store.get(batch, device)
        output = model(
            query_csi, query_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        action.append(output["action_logits"].cpu())
        risk.append(output["risk_logits"].cpu())
    return torch.cat(action), torch.cat(risk)


def blend(left: torch.Tensor, right: torch.Tensor, alpha: float) -> torch.Tensor:
    """서로 logit 척도가 다른 모델을 확률 공간에서 안전하게 결합한다."""
    probability = (
        (1.0 - alpha) * left.softmax(-1) + alpha * right.softmax(-1)
    )
    return probability.clamp_min(1e-8).log()


def danger_balance(metrics: dict) -> float:
    """danger recall과 safe specificity 중 한쪽으로 쏠린 예측을 벌점 처리한다."""
    specificity = 1.0 - metrics["safe_to_danger"] / max(metrics["safe_total"], 1)
    recall = metrics["danger_recall"]
    return 2.0 * recall * specificity / max(recall + specificity, 1e-8)


def main() -> None:
    """yja를 열지 않고 source-held-out validation에서 ensemble 비율을 잠근다."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-checkpoint", type=Path,
        default=train.WORK / "runs/cal8_raw_source_anchor/selection_model.pt",
    )
    parser.add_argument(
        "--dynamic-checkpoint", type=Path,
        default=train.WORK / "runs/cal8_raw_source_dynamic_anchor_v2/selection_model.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=train.WORK / "runs/cal8_source_ensemble",
    )
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(train.WORK / "cache/cache_index.csv")
    feature_cache = torch.load(
        train.WORK / "runs/kp5_mpr_selector_seed17/train_features.pt",
        map_location="cpu", weights_only=False,
    )
    selected_rows = feature_cache["rows"].numpy().astype(np.int64)
    selected_index = index.iloc[selected_rows]
    sites = (selected_index.subject + "_" + selected_index.environment).to_numpy()
    all_sites = sorted(set(sites))
    absence_rows = np.concatenate([
        np.flatnonzero(((index.subject == site.split("_")[0])
                        & (index.environment == site.split("_")[1])
                        & (index.task == train.C.TASK_CLS)
                        & (index.class_id == 6) & index.cache_ok).to_numpy())
        for site in all_sites
    ])
    store = train.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    site = "lmh_E01"
    rows = selected_rows[sites == site]
    static_support = select_support(rows, index, STATIC_SHOTS, seed=17017)
    dynamic_support = select_support(rows, index, DYNAMIC_SHOTS, seed=17017)
    absence = train.select_absence(site, index, seed=17018)
    support_set = set(dynamic_support.tolist())
    query = np.asarray([row for row in rows if row not in support_set], dtype=np.int64)

    static_model = load_model(
        options.static_checkpoint,
        PROMPT_CLASSES, device,
    )
    dynamic_model = load_model(
        options.dynamic_checkpoint,
        MOTION_PROMPT_CLASSES, device,
    )
    static_action, static_risk = predict(
        static_model, store, index, static_support, absence, query, device
    )
    dynamic_action, dynamic_risk = predict(
        dynamic_model, store, index, dynamic_support, absence, query, device
    )
    labels = torch.tensor(index.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(index.risk_id.iloc[query].to_numpy()).long()
    candidates = []
    for action_alpha in np.linspace(0.0, 1.0, 5):
        action_logits = blend(static_action, dynamic_action, float(action_alpha))
        for risk_alpha in np.linspace(0.0, 1.0, 11):
            risk_logits = blend(static_risk, dynamic_risk, float(risk_alpha))
            metrics = classification_metrics(action_logits, risk_logits, labels, risks)
            metrics["danger_balance"] = danger_balance(metrics)
            score = (
                metrics["action_macro_f1"]
                + 0.75 * metrics["risk_macro_f1"]
                + metrics["danger_balance"]
                + 0.25 * metrics["danger_action_accuracy"]
            )
            candidates.append({
                "action_dynamic_weight": float(action_alpha),
                "risk_dynamic_weight": float(risk_alpha),
                "selection_score": float(score),
                "metrics": metrics,
            })
    best = max(candidates, key=lambda item: item["selection_score"])
    options.run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run": "CAL8-SOURCE-HELDOUT-ENSEMBLE-SELECTION",
        "validation_site": site,
        "target_subject_used": False,
        "static_support_trials": len(static_support),
        "dynamic_support_trials": len(dynamic_support),
        "best": best,
        "static_only": next(
            item for item in candidates
            if item["action_dynamic_weight"] == 0.0
            and item["risk_dynamic_weight"] == 0.0
        ),
        "dynamic_only": next(
            item for item in candidates
            if item["action_dynamic_weight"] == 1.0
            and item["risk_dynamic_weight"] == 1.0
        ),
    }
    (options.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
