"""source nested LOSO에서 calibration anchor geometry gate를 검증한다."""

from __future__ import annotations

import argparse
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
from notifi_pose.cal17 import ANCHOR_CLASSES, anchor_geometry_error  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


@torch.no_grad()
def site_anchors(
    model,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
    device: str,
    absence_trials: int,
) -> torch.Tensor:
    """query를 읽지 않고 기본동작 16개와 absence 2개로 anchor만 만든다."""
    rows = base.site_rows(selected_rows, sites, site)
    support = base.select_support(rows, index, seed=17017)
    absence = base.select_absence(
        site, index, seed=17018, trials=absence_trials
    )
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    labels = torch.tensor(
        index.class_id.iloc[support].to_numpy(), device=device
    )
    support_output = model(
        support_csi, support_mask,
        support_csi, support_mask, labels,
        absence_csi, absence_mask,
    )
    absence_output = model(
        absence_csi, absence_mask,
        support_csi, support_mask, labels,
        absence_csi, absence_mask,
    )
    support_embedding = support_output["embedding"]
    absence_embedding = absence_output["embedding"]
    return torch.stack([
        F.normalize(absence_embedding.mean(0), dim=0)
        if class_id == 6 else F.normalize(
            support_embedding[labels == class_id].mean(0), dim=0
        )
        for class_id in ANCHOR_CLASSES
    ]).cpu()


def leave_one_site_threshold(anchors: list[torch.Tensor]) -> float:
    """train site끼리 최근접 거리의 최악값에 1.5배 여유를 둔다."""
    if len(anchors) < 2:
        raise ValueError("at least two source sites are required")
    nearest = [
        min(
            float(anchor_geometry_error(left, right))
            for other, right in enumerate(anchors)
            if other != index
        )
        for index, left in enumerate(anchors)
    ]
    return 1.5 * max(nearest)


def main() -> None:
    """fold-local source threshold로 숨긴 사람의 calibration 통과율을 계산한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
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
        raise RuntimeError("sealed yja cannot enter geometry gate evaluation")
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
    folds = {}
    passed = 0
    total = 0
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
        outer_sites = list(fold["outer_test_sites"])
        anchors = {
            site: site_anchors(
                model, store, index, selected_rows, sites, site, device,
                options.absence_trials,
            )
            for site in train_sites + outer_sites
        }
        threshold = leave_one_site_threshold([
            anchors[site] for site in train_sites
        ])
        outer = {}
        for site in outer_sites:
            distance = min(
                float(anchor_geometry_error(
                    anchors[source], anchors[site]
                ))
                for source in train_sites
            )
            accepted = distance <= threshold
            passed += int(accepted)
            total += 1
            outer[site] = {
                "nearest_geometry_error": distance,
                "accepted": accepted,
            }
        folds[held_out] = {
            "train_sites": train_sites,
            "outer_sites": outer_sites,
            "source_only_threshold": threshold,
            "outer": outer,
            "outer_used_for_threshold": False,
        }
    result = {
        "run": "CAL20-CALIBRATION-GEOMETRY-GATE",
        "folds": folds,
        "accepted_sites": passed,
        "total_outer_sites": total,
        "acceptance_rate": passed / max(total, 1),
        "query_csi_or_labels_used": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "absence_trials": options.absence_trials,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
