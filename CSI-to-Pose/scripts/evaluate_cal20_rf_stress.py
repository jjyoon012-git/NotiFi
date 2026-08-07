"""고정 CAL20+CAL17을 링크 gain/phase 변화와 링크 손실에 대해 검증한다."""

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
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import (  # noqa: E402
    class_prototypes,
    embed_site,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal17 import cal17_action, cal17_risk  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


class TargetShiftStore:
    """지정한 target site 행에만 동일한 합성 RF 변화를 적용한다."""

    def __init__(
        self,
        source: base.RawStore,
        affected_rows: set[int],
        gain_phase: bool,
        dropped_link: int | None,
    ) -> None:
        self.source = source
        self.affected_rows = affected_rows
        self.gain_phase = gain_phase
        self.dropped_link = dropped_link

    def get(
        self,
        rows: torch.Tensor | np.ndarray | list[int],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """원본 tensor를 읽고 target 행만 복사해 결정적 RF shift를 가한다."""
        values, mask = self.source.get(rows, device)
        row_array = np.asarray(rows, dtype=np.int64)
        affected = torch.tensor(
            [int(row) in self.affected_rows for row in row_array],
            device=values.device,
        )
        if not bool(affected.any()):
            return values, mask
        values = values.clone()
        mask = mask.clone()
        if self.gain_phase:
            gain = values.new_tensor((0.55, 1.65, 0.80))
            frequency = torch.linspace(
                -1.0, 1.0, C.N_LIVE_SUBCARRIERS,
                device=values.device, dtype=values.dtype,
            )
            curvature = values.new_tensor((0.65, -0.45, 0.35))
            ripple = values.new_tensor((0.30, -0.20, 0.25))
            phase = (
                curvature[:, None]
                * (frequency.square() - frequency.square().mean())[None]
                + ripple[:, None] * torch.sin(math.pi * frequency)[None]
            )
            values[affected, ..., 0] *= gain[None, None, :, None]
            values[affected, ..., 1] += phase[None, None]
        if self.dropped_link is not None:
            mask[affected, :, self.dropped_link] = False
        values[affected] *= mask[affected, ..., None, None].to(values.dtype)
        return values, mask


def target_rows(index: pd.DataFrame, site: str) -> set[int]:
    """target site의 pose query와 absence cache 행을 모두 반환한다."""
    subject, environment = site.split("_")
    keep = (
        (index.subject == subject)
        & (index.environment == environment)
        & index.cache_ok
    )
    return set(np.flatnonzero(keep.to_numpy()).tolist())


@torch.no_grad()
def evaluate_condition(
    condition: dict,
    run_dir: Path,
    calibration: dict,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    training: dict,
    device: str,
    absence_trials: int,
) -> dict[str, float]:
    """fold 설정을 다시 고르지 않고 모든 outer site의 합성 shift 성능을 모은다."""
    actions = []
    risks = []
    labels = []
    risk_labels = []
    for held_out in ("ajh", "mhw", "lmh"):
        checkpoint = torch.load(
            run_dir / f"selection_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        fold = training["fold_results"][held_out]
        train_sites = list(fold["train_sites"])
        outer_sites = list(fold["outer_test_sites"])
        source_payload = {
            site: embed_site(
                model, store, index, selected_rows, sites, site, device,
                absence_trials=absence_trials,
            )
            for site in train_sites
        }
        library = [{
            "classes": class_prototypes(source_payload[site]),
            "anchors": source_payload[site]["anchors"],
        } for site in train_sites]
        action_config = calibration["folds"][held_out]["action_config"]
        risk_config = calibration["folds"][held_out]["risk_config"]
        for site in outer_sites:
            shifted = TargetShiftStore(
                store, target_rows(index, site),
                condition["gain_phase"], condition["dropped_link"],
            )
            payload = embed_site(
                model, shifted, index, selected_rows, sites, site, device,
                absence_trials=absence_trials,
            )
            action = cal17_action(payload, library, action_config)
            risk = cal17_risk(model, payload, action, risk_config)
            actions.append(action)
            risks.append(risk)
            labels.append(payload["labels"])
            risk_labels.append(payload["risks"])
    metrics = classification_metrics(
        torch.cat(actions), torch.cat(risks),
        torch.cat(labels), torch.cat(risk_labels),
    )
    metrics["non_danger_specificity"] = float(
        1.0 - metrics["safe_to_danger"] / max(metrics["safe_total"], 1)
    )
    metrics["danger_subtype_accuracy"] = metrics[
        "danger_action_accuracy"
    ]
    return metrics


def main() -> None:
    """고정 source-LOSO checkpoint에 배포형 RF 스트레스 조건을 적용한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absence-trials", type=int, default=2)
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
    base.PROMPT_SHOTS = {
        class_id: 2 for class_id in MOTION_PROMPT_CLASSES
    }
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    feature_cache = torch.load(
        WORK / "runs/kp5_mpr_selector_seed17/train_features.pt",
        map_location="cpu", weights_only=False,
    )
    selected_rows = feature_cache["rows"].numpy().astype(np.int64)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter RF stress evaluation")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
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
    store = base.RawStore(
        index, np.concatenate((selected_rows, absence_rows))
    )
    training = json.loads(
        (options.run_dir / "result.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        options.calibration.read_text(encoding="utf-8")
    )
    conditions = {
        "clean": {"gain_phase": False, "dropped_link": None},
        "gain_phase": {"gain_phase": True, "dropped_link": None},
        "drop_tx1": {"gain_phase": False, "dropped_link": 0},
        "drop_tx2": {"gain_phase": False, "dropped_link": 1},
        "drop_tx3": {"gain_phase": False, "dropped_link": 2},
        "gain_phase_drop_tx1": {"gain_phase": True, "dropped_link": 0},
    }
    result = {
        "run": "CAL20-CAL17-RF-STRESS",
        "conditions": {
            name: evaluate_condition(
                condition, options.run_dir, calibration, store, index,
                selected_rows, sites, training, device,
                options.absence_trials,
            )
            for name, condition in conditions.items()
        },
        "target_subject_used": False,
        "sealed_yja_used": False,
        "outer_labels_used_for_selection": False,
        "absence_trials": options.absence_trials,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
