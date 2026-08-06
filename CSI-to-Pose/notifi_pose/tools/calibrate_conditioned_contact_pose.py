"""Select a validation-only deployment strength for a trained KP4-DCC run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .. import contract as C
from ..conditioned_contact_pose import DirectionalConditionedContactPose
from .train_geometry_phase_pose import build_model as build_source_model
from .train_kinetic_pose import (
    build_components,
    evaluate_strengths,
    load_or_create_coarse_store,
    make_loaders,
    pose_selection_score,
)


def _path_config(config: dict) -> SimpleNamespace:
    values = dict(config)
    for key in (
        "hierarchical_checkpoint", "p2_checkpoint", "root_calibration",
        "classification_calibration", "external_checkpoint", "coarse_cache",
        "run_dir",
    ):
        if key in values and values[key] is not None:
            values[key] = C.PROJECT_ROOT / Path(values[key])
    return SimpleNamespace(**values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp4_dcc_staged_seed17",
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=tuple(round(index / 10, 1) for index in range(11)),
    )
    cli = parser.parse_args()
    result_path = cli.run_dir / "result.json"
    checkpoint_path = cli.run_dir / "best_model.pt"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    args = _path_config(result["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_checkpoint = torch.load(
        args.hierarchical_checkpoint, map_location="cpu", weights_only=False
    )
    source_model, _, architecture, _ = build_source_model(
        source_checkpoint, device
    )
    model = DirectionalConditionedContactPose(
        source_model, dropout=float(architecture.get("dropout", 0.08))
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])

    datasets, loaders = make_loaders(args, device)
    baseline, _, _ = build_components(args, device)
    coarse_store = load_or_create_coarse_store(
        baseline, datasets, args.coarse_cache, device,
        args.batch_size, args.exp,
    )
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    strengths = [float(value) for value in cli.strengths]
    validation = evaluate_strengths(
        model, loaders["val"], strengths, device, coarse_store
    )
    scores = {
        strength: pose_selection_score(metrics)
        for strength, metrics in validation.items()
    }
    selected = min(scores, key=scores.get)
    test = evaluate_strengths(
        model, loaders["test"], [0.0, selected], device, coarse_store
    )
    critical = (
        "mpjpe_m", "danger_pose_mpjpe_m", "danger_distal_mpjpe_m",
        "danger_high_motion_mpjpe_m",
    )
    deployment_gate = all(
        test[selected][key] <= test[0.0][key] for key in critical
    )
    calibration = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selected_strength": selected,
        "selected_validation_score": scores[selected],
        "validation_candidates": {
            str(strength): {"score": scores[strength], "metrics": validation[strength]}
            for strength in strengths
        },
        "test_baseline": test[0.0],
        "test_selected": test[selected],
        "critical_metrics": list(critical),
        "deployment_gate_passed": deployment_gate,
    }
    (cli.run_dir / "deployment_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result["deployment_calibration"] = calibration
    result["promotion_status"] = (
        "pose_and_multitask_deployment"
        if deployment_gate and selected > 0.0 else
        "multitask_heads_pose_locked"
        if selected == 0.0 else "experimental"
    )
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        **checkpoint,
        "deployment_strength": selected,
        "deployment_calibration": calibration,
    }, cli.run_dir / "deployment_model.pt")
    print(json.dumps(calibration, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
