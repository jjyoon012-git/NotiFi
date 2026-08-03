"""Calibrate impact-refiner strength per joint using validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import losses as L
from ..dataio.dataset import build_datasets
from .evaluate_sealed import make_model


def masked_trial_mean(distance: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.to(distance.dtype)
    count = weight.sum(1)
    valid_trial = count > 0
    if not valid_trial.any():
        return float("nan")
    value = (distance * weight).sum(1) / count.clamp_min(1.0)
    return float(value[valid_trial].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--exp", choices=("single_split", "yja_holdout", "loso"), required=True
    )
    parser.add_argument("--fold", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument("--impact-weight", type=float, default=0.25)
    parser.add_argument("--max-regression-cm", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["cfg"]["arch"] != "impact_graphformer":
        raise ValueError("calibration requires an impact_graphformer checkpoint")
    baseline = checkpoint["cfg"].get("baseline", "none")
    validation = build_datasets(
        exp=args.exp, fold=args.fold, baseline=baseline
    )["val"]
    loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = make_model(checkpoint, device)
    model.eval()

    coarse_all = []
    delta_all = []
    pred_root_all = []
    target_pose_all = []
    target_root_all = []
    valid_all = []
    risk_all = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["csi"].to(device), batch["link_mask"].to(device)
            )
            coarse = output["pose_coarse"].float().cpu()
            refined = output["pose_rel"].float().cpu()
            coarse_all.append(coarse)
            delta_all.append(refined - coarse)
            pred_root_all.append(output["root"].float().cpu())
            target_pose_all.append(batch["pose_rel"].float())
            target_root_all.append(batch["root"].float())
            valid_all.append(batch["valid"].bool())
            risk_all.append(batch["risk_id"].long())

    coarse = torch.cat(coarse_all)
    delta = torch.cat(delta_all)
    pred_root = torch.cat(pred_root_all)
    target_pose = torch.cat(target_pose_all)
    target_root = torch.cat(target_root_all)
    valid = torch.cat(valid_all)
    risk = torch.cat(risk_all)
    impact = L.impact_window(target_pose, target_root, valid, risk)

    max_regression = args.max_regression_cm / 100.0
    selected_scales = []
    rows = []
    for joint, joint_name in enumerate(C.JOINT_NAMES):
        candidates = []
        for scale in args.scales:
            prediction = coarse[:, :, joint] + float(scale) * delta[:, :, joint]
            general_distance = torch.linalg.vector_norm(
                prediction - target_pose[:, :, joint], dim=-1
            )
            absolute_distance = torch.linalg.vector_norm(
                prediction + pred_root - target_pose[:, :, joint] - target_root,
                dim=-1,
            )
            general_error = masked_trial_mean(general_distance, valid)
            impact_error = masked_trial_mean(absolute_distance, impact)
            score = general_error + args.impact_weight * impact_error
            candidates.append({
                "scale": float(scale), "mpjpe_m": general_error,
                "impact_mpjpe_m": impact_error, "score": score,
            })

        baseline_error = candidates[0]["mpjpe_m"]
        feasible = [
            candidate for candidate in candidates
            if candidate["mpjpe_m"] <= baseline_error + max_regression
        ]
        selected = min(feasible, key=lambda item: (item["score"], item["scale"]))
        selected_scales.append(selected["scale"])
        rows.append({
            "joint": joint_name,
            "baseline_mpjpe_m": baseline_error,
            "selected": selected,
            "candidates": candidates,
        })

    calibrated = dict(checkpoint)
    calibrated["cfg"] = dict(checkpoint["cfg"])
    calibrated["cfg"]["refiner_joint_scale"] = selected_scales
    calibrated["refiner_calibration"] = {
        "split": "val", "exp": args.exp, "fold": args.fold,
        "impact_weight": args.impact_weight,
        "max_regression_cm": args.max_regression_cm,
        "scales": list(args.scales), "joints": rows,
    }
    output = args.output or args.checkpoint.with_name("calibrated_model.pt")
    torch.save(calibrated, output)
    report = output.with_suffix(".json")
    report.write_text(
        json.dumps(calibrated["refiner_calibration"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selected = {
        joint: scale for joint, scale in zip(C.JOINT_NAMES, selected_scales)
    }
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
