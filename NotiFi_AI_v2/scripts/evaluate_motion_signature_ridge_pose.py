"""Calibration CSI motion을 source GT motion 좌표에 맞춰 3D 검색을 개선한다."""

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
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import (  # noqa: E402
    class_prototypes,
    embed_site,
    select_support_shots,
)
from evaluate_cal44_fall_support import select_danger_positions  # noqa: E402
from notifi_ai_v2.support_alignment import (  # noqa: E402
    apply_affine_map,
    identity_ridge_map,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal13 import (  # noqa: E402
    pose_motion_descriptor,
    temporal_motion_signature,
)
from notifi_pose.cal17 import cal17_action, cal17_risk  # noqa: E402
from notifi_pose.hybrid_v10 import sequence_bone_projection  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.pose_simulation import (  # noqa: E402
    best_motion_shift,
    fill_pose_gaps,
    retrieval_metrics,
    shift_pose,
)
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402


CALIBRATION_MOTION_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16)


def class_means(
    values: torch.Tensor,
    labels: torch.Tensor,
    classes: tuple[int, ...] = CALIBRATION_MOTION_CLASSES,
) -> torch.Tensor:
    """고정 클래스 순서대로 signature prototype을 만든다."""
    prototypes = []
    for class_id in classes:
        keep = labels == class_id
        if not bool(keep.any()):
            raise ValueError(f"motion support is missing class {class_id}")
        prototypes.append(values[keep].mean(0))
    return torch.stack(prototypes)


@torch.no_grad()
def target_support_motion(
    model: torch.nn.Module,
    payload: dict[str, torch.Tensor],
    site: str,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    row_sites: np.ndarray,
    device: str,
    support_seed: int,
    absence_trials: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """기본·danger calibration CSI의 예측 motion signature와 query keep을 만든다."""
    site_rows = base.site_rows(selected_rows, row_sites, site)
    basic_rows = select_support_shots(site_rows, index, support_seed, 2)
    absence_rows = base.select_absence(
        site, index, support_seed + 1, trials=absence_trials
    )
    danger_positions, keep = select_danger_positions(
        payload, index, support_seed + 1000
    )
    danger_rows = payload["query_rows"][danger_positions].numpy()
    motion_rows = np.concatenate((basic_rows, danger_rows))

    query_csi, query_mask = store.get(motion_rows, device)
    support_csi, support_mask = store.get(basic_rows, device)
    absence_csi, absence_mask = store.get(absence_rows, device)
    support_labels = torch.tensor(
        index.class_id.iloc[basic_rows].to_numpy(), device=device
    ).long()
    output = model(
        query_csi,
        query_mask,
        support_csi,
        support_mask,
        support_labels,
        absence_csi,
        absence_mask,
    )
    signatures = temporal_motion_signature(
        output["pose_motion"].cpu(), output["query_frame_mask"].cpu().bool()
    )
    labels = torch.tensor(index.class_id.iloc[motion_rows].to_numpy()).long()
    return class_means(signatures, labels), keep


def calibrated_query_signature(
    payload: dict[str, torch.Tensor],
    target_support: torch.Tensor,
    source_support: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    regularization: float,
    mixture: float,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """target CSI motion signature를 source GT descriptor 좌표로 보수적으로 이동한다."""
    query = temporal_motion_signature(
        payload["motion"], payload["frame_mask"].bool()
    )
    normalized_query = (query - center) / scale
    normalized_target = (target_support - center) / scale
    mapping = identity_ridge_map(
        normalized_target,
        source_support,
        regularization,
        normalize_inputs=False,
    )
    mapped = apply_affine_map(
        normalized_query, mapping, normalize_output=False
    )
    amount = float(mixture)
    if gate is not None:
        amount = amount * gate[:, None]
    return normalized_query + amount * (mapped - normalized_query)


def danger_gate(risk: torch.Tensor, mode: str) -> torch.Tensor:
    """예측 위험 확률로 motion 보정이 필요한 query의 강도를 정한다."""
    probability = risk.softmax(-1)[:, 2]
    if mode == "all":
        return torch.ones_like(probability)
    if mode == "risk_soft":
        return probability
    if mode == "risk_sqrt":
        return probability.clamp_min(0.0).sqrt()
    if mode == "risk_hard":
        return (risk.argmax(-1) == 2).to(risk.dtype)
    raise ValueError(f"unknown danger gate: {mode}")


def retrieval_payload(
    payload: dict[str, torch.Tensor],
    keep: torch.Tensor,
    action: torch.Tensor,
    query_signature: torch.Tensor,
    candidate_pose: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_descriptor: torch.Tensor,
    normalized_candidates: torch.Tensor,
    candidate_labels: torch.Tensor,
    pose_array: np.ndarray,
    valid_array: np.ndarray,
    max_neighbors: int = 5,
) -> dict:
    """보정 signature로 source trajectory를 찾고 calibration support를 평가에서 뺀다."""
    action = action[keep]
    query_signature = query_signature[keep]
    csi_motion = payload["motion"][keep]
    csi_valid = payload["frame_mask"][keep].bool()
    distance = torch.cdist(query_signature, normalized_candidates).square()
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
                csi_motion[number],
                candidate_descriptor[candidate_number],
                csi_valid[number] & candidate_valid[candidate_number],
            )
            aligned_pose = shift_pose(candidate_pose[candidate_number], shift)
            current.append(aligned_pose)
        hypotheses.append(torch.stack(current))
    query_rows = payload["query_rows"][keep].numpy()
    return {
        "hypotheses": torch.stack(hypotheses),
        "scores": top_score,
        "csi_valid": csi_valid,
        "target_pose": torch.from_numpy(np.asarray(pose_array[query_rows]).copy()),
        "target_valid": torch.from_numpy(np.asarray(valid_array[query_rows]).copy()).bool(),
        "risks": payload["risks"][keep],
        "candidate_labels": candidate_labels[top_index],
    }


def ensemble_prediction(
    payloads: list[dict],
    neighbors: int = 5,
    temperature: float = 0.5,
    bone_blend: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """top-k source motion을 거리 가중 평균하고 선택적으로 뼈 길이를 투영한다."""
    predictions = []
    targets = []
    valids = []
    risks = []
    for payload in payloads:
        score = payload["scores"][:, :neighbors]
        weight = torch.softmax(
            -(score - score[:, :1]) / max(float(temperature), 1e-4), dim=-1
        )
        prediction = (
            payload["hypotheses"][:, :neighbors]
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
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(valids),
        torch.cat(risks),
    )


def pose_score(metrics: dict) -> float:
    """전체·사지·danger 오차를 함께 낮추는 source-inner 선택 점수를 계산한다."""
    return float(
        0.30 * metrics["pose_cm"]
        + 0.20 * metrics["distal_cm"]
        + 0.30 * metrics["danger_pose_cm"]
        + 0.20 * metrics["danger_distal_cm"]
    )


def main() -> None:
    """motion ridge 설정을 inner site에서 고른 뒤 held-out 사람의 pose를 평가한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument(
        "--regularizations", type=float, nargs="+", default=(1.0, 10.0, 100.0)
    )
    parser.add_argument(
        "--mixtures", type=float, nargs="+", default=(0.25, 0.50, 0.75, 1.0)
    )
    parser.add_argument(
        "--gates", nargs="+", default=("all", "risk_soft", "risk_sqrt", "risk_hard")
    )
    options = parser.parse_args()

    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter motion alignment")
    row_sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(row_sites.tolist()))
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
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    pose_array = np.load(work / "cache" / "pose_rel.npy", mmap_mode="r")
    valid_array = np.load(work / "cache" / "valid.npy", mmap_mode="r")
    calibration = json.loads(options.calibration.read_text(encoding="utf-8"))

    folds = {}
    for held_out in ("ajh", "mhw", "lmh"):
        train_sites, inner_sites, outer_sites = nested_site_split(all_sites, held_out)
        checkpoint = torch.load(
            options.run_dir / f"selection_{held_out}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = build_calibration_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        embedded = {
            site: embed_site(
                model,
                store,
                index,
                selected_rows,
                row_sites,
                site,
                device,
                options.support_seed,
                options.support_seed + 1,
                2,
                None,
                options.absence_trials,
            )
            for site in train_sites + inner_sites + outer_sites
        }
        library = [
            {
                "classes": class_prototypes(embedded[site]),
                "anchors": embedded[site]["anchors"],
            }
            for site in train_sites
        ]
        candidate_rows = selected_rows[np.isin(row_sites, train_sites)]
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
        source_support = class_means(normalized_candidates, candidate_labels)
        action_config = calibration["folds"][held_out]["action_config"]
        risk_config = calibration["folds"][held_out]["risk_config"]
        site_support = {}
        site_keep = {}
        site_action = {}
        site_risk = {}
        for site in inner_sites + outer_sites:
            site_support[site], site_keep[site] = target_support_motion(
                model,
                embedded[site],
                site,
                store,
                index,
                selected_rows,
                row_sites,
                device,
                options.support_seed,
                options.absence_trials,
            )
            site_action[site] = cal17_action(
                embedded[site], library, action_config
            )
            site_risk[site] = cal17_risk(
                model, embedded[site], site_action[site], risk_config
            )

        def build(
            site_names: list[str],
            regularization: float,
            mixture: float,
            gate_mode: str,
        ) -> list[dict]:
            """주어진 motion 보정으로 여러 site의 retrieval payload를 만든다."""
            rows = []
            for site in site_names:
                query_signature = calibrated_query_signature(
                    embedded[site],
                    site_support[site],
                    source_support,
                    center,
                    scale,
                    regularization,
                    mixture,
                    danger_gate(site_risk[site], gate_mode),
                )
                rows.append(retrieval_payload(
                    embedded[site],
                    site_keep[site],
                    site_action[site],
                    query_signature,
                    candidate_pose,
                    candidate_valid,
                    candidate_descriptor,
                    normalized_candidates,
                    candidate_labels,
                    pose_array,
                    valid_array,
                ))
            return rows

        baseline_payload = build(outer_sites, 1.0, 0.0, "all")
        baseline_prediction = ensemble_prediction(baseline_payload)
        baseline_metrics = retrieval_metrics(*baseline_prediction)
        candidates = []
        for regularization in options.regularizations:
            for mixture in options.mixtures:
                for gate_mode in options.gates:
                    inner_payload = build(
                        inner_sites, regularization, mixture, gate_mode
                    )
                    prediction = ensemble_prediction(inner_payload)
                    metrics = retrieval_metrics(*prediction)
                    candidates.append({
                        "regularization": float(regularization),
                        "mixture": float(mixture),
                        "gate": gate_mode,
                        "metrics": metrics,
                        "score": pose_score(metrics),
                    })
        selected_config = min(candidates, key=lambda item: item["score"])
        outer_payload = build(
            outer_sites,
            selected_config["regularization"],
            selected_config["mixture"],
            selected_config["gate"],
        )
        outer_prediction = ensemble_prediction(outer_payload)
        outer_metrics = retrieval_metrics(*outer_prediction)
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": inner_sites,
            "outer_test_sites": outer_sites,
            "selected": selected_config,
            "baseline": baseline_metrics,
            "outer": outer_metrics,
            "outer_used_for_selection": False,
        }
        print(
            f"{held_out}: reg={selected_config['regularization']:.2g} "
            f"mix={selected_config['mixture']:.2f} gate={selected_config['gate']} "
            f"pose={baseline_metrics['pose_cm']:.2f}->{outer_metrics['pose_cm']:.2f}",
            flush=True,
        )

    weights = np.asarray([
        3 if held_out in ("ajh", "mhw") else 1
        for held_out in ("ajh", "mhw", "lmh")
    ], dtype=np.float64)
    metric_names = (
        "pose_cm", "distal_cm", "pa_pose_cm", "danger_pose_cm", "danger_distal_cm"
    )
    aggregate = {
        side: {
            key: float(np.average([
                folds[held_out][side][key] for held_out in ("ajh", "mhw", "lmh")
            ], weights=weights))
            for key in metric_names
        }
        for side in ("baseline", "outer")
    }
    result = {
        "run": "NOTIFI-AI-V2-MOTION-SIGNATURE-RIDGE-POSE",
        "protocol": "source nested-LOSO; target calibration CSI labels only",
        "support_seed": int(options.support_seed),
        "folds": folds,
        "aggregate": aggregate,
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "calibration_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
