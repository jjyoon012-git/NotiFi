"""카메라 없는 균일 30Hz CSI grid에서 고정 CAL20+CAL17을 검증한다."""

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
from calibrate_cal17_style_transport import (  # noqa: E402
    choose_action_config,
    choose_risk_config,
    class_prototypes,
    embed_site,
    evaluate_action_config,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal17 import cal17_action, cal17_risk  # noqa: E402
from notifi_pose.deployment import load_csi_csv  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


class UniformGridStore:
    """cache 행 번호를 raw CSV에서 다시 만든 균일 30Hz tensor에 연결한다."""

    def __init__(
        self,
        cache_index: pd.DataFrame,
        all_index: pd.DataFrame,
        rows: np.ndarray,
        dataset_root: Path,
    ) -> None:
        by_trial = all_index.set_index("trial_id")["csi"].to_dict()
        rows = np.asarray(sorted(set(int(row) for row in rows)), dtype=np.int64)
        self.position = {int(row): local for local, row in enumerate(rows)}
        csi = []
        mask = []
        for number, row in enumerate(rows, start=1):
            trial_id = str(cache_index.iloc[row].trial_id)
            relative = by_trial.get(trial_id)
            if not relative:
                raise RuntimeError(f"raw CSI path is missing: {trial_id}")
            values, valid, _ = load_csi_csv(dataset_root / str(relative))
            csi.append(values.half())
            mask.append(valid)
            if number % 50 == 0 or number == len(rows):
                print(
                    f"[uniform-grid] parsed {number}/{len(rows)}",
                    flush=True,
                )
        self.csi = torch.stack(csi)
        self.mask = torch.stack(mask)

    def get(
        self,
        rows: torch.Tensor | np.ndarray | list[int],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """전역 cache 행 번호의 균일 grid tensor를 model device로 보낸다."""
        local = torch.tensor([
            self.position[int(row)] for row in np.asarray(rows, dtype=np.int64)
        ]).long()
        return (
            self.csi[local].to(device, non_blocking=True).float(),
            self.mask[local].to(device, non_blocking=True),
        )


def site_absence_rows(index: pd.DataFrame, site: str) -> np.ndarray:
    """한 site의 cache 가능한 classification-only absence 행을 반환한다."""
    subject, environment = site.split("_")
    return np.flatnonzero((
        (index.subject == subject)
        & (index.environment == environment)
        & (index.task == C.TASK_CLS)
        & (index.class_id == 6)
        & index.cache_ok
    ).to_numpy())


@torch.no_grad()
def main() -> None:
    """source library는 학습 grid, outer target만 배포 grid로 고정 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retune-on-inner", action="store_true")
    parser.add_argument("--retune-risk-on-inner", action="store_true")
    parser.add_argument("--absence-trials", type=int, default=2)
    parser.add_argument(
        "--force-risk-config", type=float, nargs=3,
        metavar=("SAFE_WEIGHT", "FUSION", "DANGER_BIAS"),
    )
    options = parser.parse_args()
    modes = sum((
        bool(options.retune_on_inner),
        bool(options.retune_risk_on_inner),
        options.force_risk_config is not None,
    ))
    if modes > 1:
        parser.error("choose only one risk/config selection mode")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
    base.PROMPT_SHOTS = {
        class_id: 2 for class_id in MOTION_PROMPT_CLASSES
    }
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    all_index = pd.read_csv(WORK / "index/all_index.csv")
    feature_cache = torch.load(
        WORK / "runs/kp5_mpr_selector_seed17/train_features.pt",
        map_location="cpu", weights_only=False,
    )
    selected_rows = feature_cache["rows"].numpy().astype(np.int64)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter uniform-grid evaluation")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
    all_absence = np.concatenate([
        site_absence_rows(index, site) for site in all_sites
    ])
    cache_store = base.RawStore(
        index, np.concatenate((selected_rows, all_absence))
    )
    training = json.loads(
        (options.run_dir / "result.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        options.calibration.read_text(encoding="utf-8")
    )
    actions = []
    risks = []
    labels = []
    risk_labels = []
    site_metrics = {}
    selected_configs = {}
    for held_out in ("ajh", "mhw", "lmh"):
        checkpoint = torch.load(
            options.run_dir / f"selection_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        fold = training["fold_results"][held_out]
        train_sites = list(fold["train_sites"])
        inner_sites = list(fold["inner_validation_sites"])
        outer_sites = list(fold["outer_test_sites"])
        source_payload = {
            site: embed_site(
                model, cache_store, index, selected_rows, sites, site, device,
                absence_trials=options.absence_trials,
            )
            for site in train_sites
        }
        library = [{
            "classes": class_prototypes(source_payload[site]),
            "anchors": source_payload[site]["anchors"],
        } for site in train_sites]
        target_payload = {}
        target_sites = (
            inner_sites + outer_sites
            if options.retune_on_inner or options.retune_risk_on_inner
            else outer_sites
        )
        for site in target_sites:
            rows = selected_rows[sites == site]
            uniform = UniformGridStore(
                index, all_index,
                np.concatenate((rows, site_absence_rows(index, site))),
                options.dataset_root,
            )
            target_payload[site] = embed_site(
                model, uniform, index, selected_rows, sites, site, device,
                absence_trials=options.absence_trials,
            )
        if options.force_risk_config is not None:
            action_config = calibration["folds"][held_out]["action_config"]
            risk_config = dict(zip((
                "safe_weight", "fusion", "danger_bias",
            ), options.force_risk_config))
        elif options.retune_on_inner:
            inner_payload = [target_payload[site] for site in inner_sites]
            action_tuple = choose_action_config(inner_payload, library)
            _, inner_actions = evaluate_action_config(
                inner_payload, library, action_tuple
            )
            risk_tuple = choose_risk_config(
                model, inner_payload, inner_actions
            )
            action_config = dict(zip((
                "strength", "anchor_temperature", "prototype_temperature",
                "site_temperature", "mixture",
            ), action_tuple))
            risk_config = dict(zip((
                "safe_weight", "fusion", "danger_bias",
            ), risk_tuple))
        elif options.retune_risk_on_inner:
            action_config = calibration["folds"][held_out]["action_config"]
            inner_payload = [target_payload[site] for site in inner_sites]
            inner_actions = [
                cal17_action(target, library, action_config)
                for target in inner_payload
            ]
            risk_tuple = choose_risk_config(
                model, inner_payload, inner_actions
            )
            risk_config = dict(zip((
                "safe_weight", "fusion", "danger_bias",
            ), risk_tuple))
        else:
            action_config = calibration["folds"][held_out]["action_config"]
            risk_config = calibration["folds"][held_out]["risk_config"]
        selected_configs[held_out] = {
            "action_config": action_config,
            "risk_config": risk_config,
            "inner_sites": inner_sites,
            "outer_used_for_selection": False,
        }
        for site in outer_sites:
            payload = target_payload[site]
            action = cal17_action(payload, library, action_config)
            risk = cal17_risk(model, payload, action, risk_config)
            metric = classification_metrics(
                action, risk, payload["labels"], payload["risks"]
            )
            site_metrics[site] = metric
            actions.append(action)
            risks.append(risk)
            labels.append(payload["labels"])
            risk_labels.append(payload["risks"])
    pooled = classification_metrics(
        torch.cat(actions), torch.cat(risks),
        torch.cat(labels), torch.cat(risk_labels),
    )
    pooled["non_danger_specificity"] = float(
        1.0 - pooled["safe_to_danger"] / max(pooled["safe_total"], 1)
    )
    result = {
        "run": "CAL20-CAL17-UNIFORM-30HZ-DEPLOYMENT-GRID",
        "pooled": pooled,
        "sites": site_metrics,
        "selected_configs": selected_configs,
        "inner_grid_config_selection": bool(options.retune_on_inner),
        "inner_grid_risk_only_selection": bool(
            options.retune_risk_on_inner
        ),
        "forced_risk_config": options.force_risk_config,
        "source_library_grid": "timestamp_aligned_training_cache",
        "outer_target_grid": "uniform_30hz_304_frames_from_raw_csi",
        "target_subject_used": False,
        "sealed_yja_used": False,
        "absence_trials": options.absence_trials,
        "outer_labels_used_for_selection": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
