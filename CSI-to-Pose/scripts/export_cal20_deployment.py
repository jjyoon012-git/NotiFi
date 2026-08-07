"""전체 source 모델과 CAL17/CAL23 library를 단일 배포 bundle로 내보낸다."""

from __future__ import annotations

import argparse
import hashlib
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
    class_prototypes,
    embed_site,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal13 import (  # noqa: E402
    pose_motion_descriptor,
    temporal_motion_signature,
)
from notifi_pose.cal17 import anchor_geometry_error  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def sha256(path: Path) -> str:
    """배포 bundle 입력 파일의 내용 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_source_clean(
    result: dict, name: str, require_query_flag: bool = True,
) -> None:
    """배포 입력 결과가 target subject나 query 정답을 사용하지 않았는지 검증한다."""
    if result.get("target_subject_used") is not False:
        raise RuntimeError(f"{name} does not explicitly exclude target subjects")
    if result.get("sealed_yja_used") is not False:
        raise RuntimeError(f"{name} does not explicitly exclude sealed yja")
    if (
        require_query_flag
        and result.get("query_labels_or_pose_gt_at_inference") is not False
    ):
        raise RuntimeError(f"{name} does not explicitly exclude query labels and pose GT")


def median_config(configs: list[dict]) -> dict:
    """nested source fold에서 고른 수치 설정의 중앙값을 deployment에 고정한다."""
    keys = configs[0].keys()
    return {
        key: float(np.median([float(config[key]) for config in configs]))
        for key in keys
    }


def main() -> None:
    """yja 없이 학습된 checkpoint와 source GT library를 결합해 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--pose-result", type=Path, required=True)
    parser.add_argument("--uniform-grid-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absence-trials", type=int, default=2)
    options = parser.parse_args()
    calibration = json.loads(
        options.calibration.read_text(encoding="utf-8")
    )
    if int(calibration.get("absence_trials", 2)) != options.absence_trials:
        raise RuntimeError("calibration absence contract does not match export")
    require_source_clean(calibration, "calibration result")
    uniform = None
    if options.uniform_grid_result is not None:
        uniform = json.loads(
            options.uniform_grid_result.read_text(encoding="utf-8")
        )
        if int(uniform.get("absence_trials", 2)) != options.absence_trials:
            raise RuntimeError("uniform-grid absence contract does not match export")
        require_source_clean(
            uniform, "uniform-grid result", require_query_flag=False,
        )
        if (
            uniform.get("outer_labels_used_for_selection") is not False
            or uniform.get("inner_grid_risk_only_selection") is not True
        ):
            raise RuntimeError("uniform-grid risk result is not source-inner clean")
    pose_result = json.loads(
        options.pose_result.read_text(encoding="utf-8")
    )
    if int(pose_result.get("absence_trials", 2)) != options.absence_trials:
        raise RuntimeError("pose absence contract does not match export")
    require_source_clean(pose_result, "pose result")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
    base.PROMPT_SHOTS = {
        class_id: 2 for class_id in MOTION_PROMPT_CLASSES
    }
    checkpoint = torch.load(
        options.run_dir / "deployment_model.pt",
        map_location="cpu", weights_only=False,
    )
    if checkpoint.get("sealed_yja_used") is not False:
        raise RuntimeError("deployment checkpoint is not sealed-target clean")
    model = build_calibration_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter deployment library")
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
    embedded = {
        site: embed_site(
            model, store, index, selected_rows, sites, site, device,
            absence_trials=options.absence_trials,
        )
        for site in all_sites
    }
    source_library = [{
        "site": site,
        "classes": class_prototypes(embedded[site]).cpu(),
        "anchors": embedded[site]["anchors"].cpu(),
    } for site in all_sites]
    nearest_geometry = []
    for left, source in enumerate(source_library):
        nearest_geometry.append(min(
            float(anchor_geometry_error(
                source["anchors"], other["anchors"]
            ))
            for right, other in enumerate(source_library)
            if right != left
        ))
    geometry_threshold = 1.5 * max(nearest_geometry)
    action_config = median_config([
        fold["action_config"] for fold in calibration["folds"].values()
    ])
    risk_config = median_config([
        fold["risk_config"] for fold in calibration["folds"].values()
    ])
    risk_config_source = "timestamp_grid_nested_source_folds"
    if uniform is not None:
        risk_config = median_config([
            fold["risk_config"]
            for fold in uniform["selected_configs"].values()
        ])
        risk_config_source = "uniform_30hz_source_inner_risk_only"
    pose_configs = [
        fold["selected"] for fold in pose_result["folds"].values()
    ]
    pose_config = median_config(pose_configs)
    pose_config["neighbors"] = int(round(pose_config["neighbors"]))
    pose_array = np.load(WORK / "cache/pose_rel.npy", mmap_mode="r")
    valid_array = np.load(WORK / "cache/valid.npy", mmap_mode="r")
    pose = torch.from_numpy(np.asarray(pose_array[selected_rows]).copy()).float()
    valid = torch.from_numpy(
        np.asarray(valid_array[selected_rows]).copy()
    ).bool()
    descriptors = pose_motion_descriptor(pose, valid)
    signatures = temporal_motion_signature(descriptors, valid)
    center = signatures.mean(0)
    scale = signatures.std(0).clamp_min(0.05)
    pose_library = {
        "pose": pose.half(),
        "valid": valid,
        "descriptors": descriptors.half(),
        "normalized_signatures": ((signatures - center) / scale).float(),
        "signature_center": center.float(),
        "signature_scale": scale.float(),
        "labels": torch.tensor(
            index.class_id.iloc[selected_rows].to_numpy()
        ).long(),
        "trial_ids": index.trial_id.iloc[selected_rows].astype(str).tolist(),
    }
    support_contract = dict(checkpoint["support_contract"])
    support_contract["absence_trials"] = options.absence_trials
    bundle = {
        "bundle_version": "cal20_cal17_cal23_v4",
        "model": checkpoint["model"],
        "model_config": checkpoint["model_config"],
        "support_contract": support_contract,
        "source_sites": all_sites,
        "source_library": source_library,
        "action_config": action_config,
        "risk_config": risk_config,
        "risk_config_source": risk_config_source,
        "pose_config": pose_config,
        "pose_library": pose_library,
        "calibration_geometry_threshold": geometry_threshold,
        "source_leave_one_site_geometry": nearest_geometry,
        "config_aggregation": "median_of_nested_source_fold_selections",
        "source_pose_gt_training_and_library_only": True,
        "query_labels_or_pose_gt_used": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "absence_trials": options.absence_trials,
        "provenance": {
            "deployment_model_sha256": sha256(
                options.run_dir / "deployment_model.pt"
            ),
            "calibration_result_sha256": sha256(options.calibration),
            "pose_result_sha256": sha256(options.pose_result),
            "uniform_grid_result_sha256": (
                sha256(options.uniform_grid_result)
                if options.uniform_grid_result is not None else None
            ),
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, options.output)
    print(json.dumps({
        "output": str(options.output),
        "source_sites": all_sites,
        "source_prototype_libraries": len(source_library),
        "pose_candidates": len(pose),
        "action_config": action_config,
        "risk_config": risk_config,
        "risk_config_source": risk_config_source,
        "pose_config": pose_config,
        "calibration_geometry_threshold": geometry_threshold,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "absence_trials": options.absence_trials,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
