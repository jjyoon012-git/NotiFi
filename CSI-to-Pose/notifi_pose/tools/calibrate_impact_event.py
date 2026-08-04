"""Select event/contact/speed branch strengths on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import DropoutConfig, build_datasets
from ..impact_event import ImpactEventLocalizer
from ..quality import QualityWeightedDataset
from .diagnose_observability import pose_only, report_path
from .train_impact_event import evaluate_event, load_v3, selection_score
from .train_seen_v2 import evaluate_injury


def make_args(checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        v3_checkpoint=C.WORK_ROOT / "runs" / "seen_v3_contact_root" / "calibrated_model.pt",
        v2_checkpoint=C.WORK_ROOT / "runs" / "seen_reconstruction_v2" / "calibrated_model.pt",
        baseline_checkpoint=C.WORK_ROOT / "runs" / "graphformer_hybrid_dynamic_v1" / "best_model.pt",
        motion_checkpoint=C.WORK_ROOT / "runs" / "motion_first_seen" / "best_model.pt",
        pose_residual_checkpoint=C.WORK_ROOT / "runs" / "action_motion_residual_seen" / "best_model.pt",
        root_residual_checkpoint=C.WORK_ROOT / "runs" / "keyframe_root_residual_seen" / "best_model.pt",
        event_checkpoint=checkpoint,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "impact_event_v8c_raw" / "best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "impact_event_v8c_raw" / "calibration.json",
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = make_args(args.checkpoint)
    model = ImpactEventLocalizer(load_v3(model_args, device)).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    datasets = build_datasets(
        exp="single_split", baseline="sub",
        dropout=DropoutConfig(p=0.0, rf_augment=False), seed=7,
    )
    validation = QualityWeightedDataset(pose_only(datasets["val"]))
    test = QualityWeightedDataset(pose_only(datasets["test"]))
    loaders = {
        "val": DataLoader(
            validation, batch_size=args.batch_size, shuffle=False, num_workers=0
        ),
        "test": DataLoader(
            test, batch_size=args.batch_size, shuffle=False, num_workers=0
        ),
    }

    model.set_calibration(event=0, joint=0, contact=0, speed=0)
    baseline_event = evaluate_event(model, loaders["val"], device)
    baseline_injury = evaluate_injury(model, loaders["val"], device)

    contact_candidates = []
    for strength in args.strengths:
        model.set_calibration(contact=strength)
        metrics = evaluate_injury(model, loaders["val"], device)
        contact_candidates.append({"strength": strength, "validation": metrics})
    selected_contact = max(
        contact_candidates,
        key=lambda item: item["validation"]["injury_contact_f1"],
    )
    model.set_calibration(contact=selected_contact["strength"])

    joint_candidates = []
    for strength in args.strengths:
        model.set_calibration(event=strength, joint=strength)
        event = evaluate_event(model, loaders["val"], device)
        injury = evaluate_injury(model, loaders["val"], device)
        gate = (
            event["timing_mae_frames"] <= baseline_event["timing_mae_frames"]
            and event["region_accuracy"] >= baseline_event["region_accuracy"] - 0.01
            and event["joint_accuracy"] >= baseline_event["joint_accuracy"] - 0.01
            and injury["first_contact_accuracy"]
            >= baseline_injury["first_contact_accuracy"] - 0.01
        )
        joint_candidates.append({
            "strength": strength, "gate": gate,
            "score": selection_score(event),
            "event_validation": event, "injury_validation": injury,
        })
    eligible = [item for item in joint_candidates if item["gate"]]
    selected_joint = min(eligible, key=lambda item: item["score"])
    model.set_calibration(
        event=selected_joint["strength"], joint=selected_joint["strength"]
    )

    speed_candidates = []
    for strength in args.strengths:
        model.set_calibration(speed=strength)
        metrics = evaluate_injury(model, loaders["val"], device)
        speed_candidates.append({"strength": strength, "validation": metrics})
    selected_speed = min(
        speed_candidates,
        key=lambda item: item["validation"]["impact_joint_speed_mae_mps"],
    )
    model.set_calibration(speed=selected_speed["strength"])

    selected = {
        "event_strength": selected_joint["strength"],
        "joint_strength": selected_joint["strength"],
        "contact_strength": selected_contact["strength"],
        "speed_strength": selected_speed["strength"],
    }
    result = {
        "run": "impact_event_v8_branch_calibration",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_checkpoint": report_path(args.checkpoint),
        "selected": selected,
        "baseline_validation": {
            "event": baseline_event, "injury": baseline_injury,
        },
        "selected_validation": {
            "event": evaluate_event(model, loaders["val"], device),
            "injury": evaluate_injury(model, loaders["val"], device),
        },
        "test": {
            "event": evaluate_event(model, loaders["test"], device),
            "injury": evaluate_injury(model, loaders["test"], device),
        },
        "contact_candidates": contact_candidates,
        "joint_candidates": joint_candidates,
        "speed_candidates": speed_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(), "calibration": selected,
        "validation": result["selected_validation"], "test": result["test"],
    }, args.output.with_name("calibrated_model.pt"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
