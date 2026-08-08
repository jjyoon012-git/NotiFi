"""CAL23: inner-selected train-pose neighbor ensemble로 CSI-only 3D simulation을 안정화한다."""

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
    class_prototypes,
    embed_site,
)
from notifi_pose.pose_simulation import (  # noqa: E402
    best_motion_shift,
    fill_pose_gaps,
    retrieval_metrics,
    shift_pose,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal13 import (  # noqa: E402
    pose_motion_descriptor,
    temporal_motion_signature,
)
from notifi_pose.cal17 import cal17_action, cal17_risk  # noqa: E402
from notifi_pose.hybrid_v10 import sequence_bone_projection  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def retrieval_payload(
    payload: dict,
    action: torch.Tensor,
    candidate_pose: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_descriptor: torch.Tensor,
    normalized_candidates: torch.Tensor,
    candidate_labels: torch.Tensor,
    signature_center: torch.Tensor,
    signature_scale: torch.Tensor,
    pose_array: np.ndarray,
    valid_array: np.ndarray,
    max_neighbors: int = 5,
) -> dict:
    """GT를 보지 않고 CSI action·motion으로 상위 train trajectory와 shift를 고정한다."""
    csi_valid = payload["frame_mask"].bool()
    query_signature = temporal_motion_signature(payload["motion"], csi_valid)
    normalized_query = (query_signature - signature_center) / signature_scale
    distance = torch.cdist(normalized_query, normalized_candidates).square()
    probability = action.softmax(-1)
    top_actions = probability.topk(3, dim=-1).indices
    allowed = (
        candidate_labels[None, :, None] == top_actions[:, None, :]
    ).any(-1)
    action_penalty = -0.25 * torch.log(
        probability[:, candidate_labels].clamp_min(1e-8)
    )
    score = (distance + action_penalty).masked_fill(~allowed, torch.inf)
    top_score, top_index = score.topk(max_neighbors, largest=False, dim=-1)
    hypotheses = []
    for number in range(len(top_index)):
        current = []
        for candidate_number in top_index[number].tolist():
            shift = best_motion_shift(
                payload["motion"][number],
                candidate_descriptor[candidate_number],
                csi_valid[number] & candidate_valid[candidate_number],
            )
            current.append(shift_pose(
                candidate_pose[candidate_number], shift
            ))
        hypotheses.append(torch.stack(current))
    query_rows = payload["query_rows"].numpy()
    return {
        "hypotheses": torch.stack(hypotheses),
        "scores": top_score,
        "candidate_labels": candidate_labels[top_index],
        "csi_valid": csi_valid,
        "target_pose": torch.from_numpy(
            np.asarray(pose_array[query_rows]).copy()
        ),
        "target_valid": torch.from_numpy(
            np.asarray(valid_array[query_rows]).copy()
        ).bool(),
        "labels": payload["labels"],
        "risks": payload["risks"],
    }


def ensemble_prediction(
    payloads: list[dict],
    neighbors: int,
    temperature: float,
    bone_blend: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """상위 실제 궤적을 확률 가중하고 trial bone length consistency를 적용한다."""
    predictions = []
    targets = []
    valids = []
    risks = []
    for payload in payloads:
        score = payload["scores"][:, :neighbors]
        if neighbors == 1:
            weight = torch.ones_like(score)
        else:
            weight = torch.softmax(
                -(score - score[:, :1]) / max(float(temperature), 1e-4), dim=-1
            )
        hypotheses = payload["hypotheses"][:, :neighbors]
        prediction = (
            hypotheses
            * weight[:, :, None, None, None]
        ).sum(1)
        if bone_blend > 0.0:
            projected = sequence_bone_projection(
                prediction, payload["csi_valid"], symmetric=True
            )
            prediction = prediction + bone_blend * (projected - prediction)
        predictions.append(prediction)
        targets.append(payload["target_pose"])
        valids.append(payload["target_valid"])
        risks.append(payload["risks"])
    return (
        torch.cat(predictions), torch.cat(targets),
        torch.cat(valids), torch.cat(risks),
    )


def selection_score(metrics: dict) -> float:
    """전체·말단·danger 오차를 함께 낮추는 v1 inner simulation 점수를 계산한다."""
    return float(
        0.35 * metrics["pose_cm"]
        + 0.20 * metrics["distal_cm"]
        + 0.25 * metrics["danger_pose_cm"]
        + 0.20 * metrics["danger_distal_cm"]
    )


def main() -> None:
    """각 fold의 inner site로 ensemble 설정을 고르고 outer trajectory를 한 번 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absence-trials", type=int, default=2)
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter CAL23 selection")
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
    pose_array = np.load(WORK / "cache/pose_rel.npy", mmap_mode="r")
    valid_array = np.load(WORK / "cache/valid.npy", mmap_mode="r")
    training = json.loads((options.run_dir / "result.json").read_text(
        encoding="utf-8"
    ))
    calibration = json.loads(options.calibration.read_text(encoding="utf-8"))
    folds = {}
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
        embedded = {
            site: embed_site(
                model, store, index, selected_rows, sites, site, device,
                absence_trials=options.absence_trials,
            )
            for site in train_sites + inner_sites + outer_sites
        }
        library = [{
            "classes": class_prototypes(embedded[site]),
            "anchors": embedded[site]["anchors"],
        } for site in train_sites]
        candidate_rows = selected_rows[np.isin(sites, train_sites)]
        candidate_pose = torch.from_numpy(
            np.asarray(pose_array[candidate_rows]).copy()
        )
        candidate_valid = torch.from_numpy(
            np.asarray(valid_array[candidate_rows]).copy()
        ).bool()
        candidate_valid &= torch.isfinite(candidate_pose).all(-1).all(-1)
        candidate_pose = fill_pose_gaps(candidate_pose, candidate_valid)
        candidate_descriptor = pose_motion_descriptor(
            candidate_pose, candidate_valid
        )
        candidate_signature = temporal_motion_signature(
            candidate_descriptor, candidate_valid
        )
        center = candidate_signature.mean(0)
        scale = candidate_signature.std(0).clamp_min(0.05)
        normalized_candidates = (candidate_signature - center) / scale
        candidate_labels = torch.tensor(
            index.class_id.iloc[candidate_rows].to_numpy()
        ).long()
        action_config = calibration["folds"][held_out]["action_config"]
        risk_config = calibration["folds"][held_out]["risk_config"]

        def build(site_names: list[str]) -> tuple[list[dict], list[torch.Tensor], list[torch.Tensor]]:
            """고정 모델로 site별 retrieval 후보와 분류 logit을 구성한다."""
            retrievals = []
            actions = []
            risks = []
            for site in site_names:
                payload = embedded[site]
                action = cal17_action(payload, library, action_config)
                risk = cal17_risk(model, payload, action, risk_config)
                retrievals.append(retrieval_payload(
                    payload, action, candidate_pose, candidate_valid,
                    candidate_descriptor, normalized_candidates,
                    candidate_labels, center, scale, pose_array, valid_array,
                ))
                actions.append(action)
                risks.append(risk)
            return retrievals, actions, risks

        inner_payloads, _, _ = build(inner_sites)
        candidates = []
        for neighbors, temperature, bone_blend in itertools.product(
            (1, 3, 5), (0.05, 0.10, 0.25, 0.50), (0.0, 0.5, 1.0),
        ):
            if neighbors == 1 and temperature != 0.05:
                continue
            prediction, target, valid, risk = ensemble_prediction(
                inner_payloads, neighbors, temperature, bone_blend,
            )
            metrics = retrieval_metrics(prediction, target, valid, risk)
            candidates.append({
                "neighbors": neighbors,
                "temperature": temperature,
                "bone_blend": bone_blend,
                "score": selection_score(metrics),
                "inner_metrics": metrics,
            })
        selected_config = min(candidates, key=lambda item: item["score"])
        outer_payloads, outer_actions, outer_risks = build(outer_sites)
        prediction, target, valid, risk = ensemble_prediction(
            outer_payloads,
            selected_config["neighbors"],
            selected_config["temperature"],
            selected_config["bone_blend"],
        )
        labels = torch.cat([payload["labels"] for payload in outer_payloads])
        selected_labels = torch.cat([
            payload["candidate_labels"][:, 0] for payload in outer_payloads
        ])
        folds[held_out] = {
            "selected": {
                key: selected_config[key]
                for key in (
                    "neighbors", "temperature", "bone_blend",
                )
            },
            "selected_inner_metrics": selected_config["inner_metrics"],
            "outer_pose": retrieval_metrics(
                prediction, target, valid, risk
            ),
            "classification": classification_metrics(
                torch.cat(outer_actions), torch.cat(outer_risks), labels, risk
            ),
            "retrieval_action_match": float(
                (selected_labels == labels).float().mean()
            ),
            "outer_used_for_selection": False,
            "query_pose_gt_used_for_retrieval": False,
            "query_gt_valid_mask_used_for_retrieval": False,
        }
    result = {
        "run": "CAL23-INNER-SELECTED-POSE-ENSEMBLE",
        "folds": folds,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "absence_trials": options.absence_trials,
    }
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
