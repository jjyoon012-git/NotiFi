"""Select label-free I/Q moment calibration on validation perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..hybrid_v10 import InputMomentCalibration
from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset, _summary
from .calibrate_v11_residual_temporal import _build_model
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


def _load_locked(args) -> dict:
    locked = json.loads(args.calibration.read_text(encoding="utf-8"))
    if locked.get("protocol") != args.exp:
        raise RuntimeError("calibration protocol mismatch")
    if locked.get("selection_split") != "validation":
        raise RuntimeError("input calibration must use a validation-locked model")
    if locked.get("test_used_for_selection") is not False:
        raise RuntimeError("test split was not proven sealed")
    return locked


def _configure(args, locked: dict, device: str):
    source = locked["source"]
    args.pose_strength = float(source["pose_strength"])
    args.root_strength = float(source["root_strength"])
    args.bone_blend = float(source["bone_blend"])
    args.bone_symmetric = bool(source["bone_symmetric"])
    model = _build_model(args, device)
    temporal = model.base
    temporal.set_calibration(
        int(locked["selected"]["window"]),
        float(locked["selected"]["blend"]),
        source.get("risk_adaptive", "none"),
        float(source.get("danger_logit_bias", 0.0)),
    )
    temporal.set_root_calibration(
        int(locked["selected"].get("root_window", 1)),
        float(locked["selected"].get("root_blend", 0.0)),
    )
    checkpoint = torch.load(
        args.p2_checkpoint, map_location="cpu", weights_only=False,
    )
    state = checkpoint["model"]
    return InputMomentCalibration(
        model,
        state["norm.mu"],
        state["norm.sigma"],
    ).to(device)


def _evaluate(model, dataset, class_dataset, mode: str, args, device: str,
              danger_bias: float) -> dict:
    pose_loader = DataLoader(
        PerturbedDataset(dataset, mode), batch_size=args.batch_size * 2,
        shuffle=False, num_workers=0,
    )
    class_loader = DataLoader(
        PerturbedDataset(class_dataset, mode), batch_size=args.batch_size * 2,
        shuffle=False, num_workers=0,
    )
    trajectory = evaluate_trajectory(model, pose_loader, device, args.max_shift)
    classification = evaluate_classification(
        model, class_loader, device, danger_bias,
    )
    return {
        "trajectory": _summary(trajectory),
        "class_accuracy": classification["class"]["accuracy"],
        "class_macro_f1": classification["class"]["macro_f1"],
        "risk_accuracy": classification["risk"]["accuracy"],
        "risk_macro_f1": classification["risk"]["macro_f1"],
        "danger_recall": classification["risk"]["danger_recall"],
        "safe_to_danger": classification["risk"]["safe_to_danger"],
    }


def _feasible(result: dict, baseline: dict) -> bool:
    pose = result["trajectory"]
    base_pose = baseline["trajectory"]
    return (
        pose["mpjpe_m"] <= base_pose["mpjpe_m"] + 0.003
        and pose["danger_mpjpe_m"] <= base_pose["danger_mpjpe_m"] + 0.006
        and pose["danger_endpoint_mpjpe_m"]
        <= base_pose["danger_endpoint_mpjpe_m"] + 0.010
        and result["class_accuracy"] >= baseline["class_accuracy"] - 0.01
        and result["risk_accuracy"] >= baseline["risk_accuracy"] - 0.01
        and result["danger_recall"] >= baseline["danger_recall"] - 0.03
        and result["safe_to_danger"] <= baseline["safe_to_danger"] + 2
    )


def _robust_score(results: dict) -> float:
    values = []
    for mode in ("gain_phase", "gain_phase_alt", "gain_phase_trial"):
        result = results[mode]
        pose = result["trajectory"]
        values.append(
            pose["mpjpe_m"]
            + 0.50 * pose["danger_mpjpe_m"]
            + 0.25 * pose["danger_endpoint_mpjpe_m"]
            + 0.08 * (1.0 - result["class_accuracy"])
            + 0.08 * (1.0 - result["risk_accuracy"])
            + 0.10 * (1.0 - result["danger_recall"])
            + 0.002 * result["safe_to_danger"]
        )
    return float(sum(values) / len(values))


def main() -> int:
    if C.CSI_REPRESENTATION != "iq":
        raise RuntimeError(
            "I/Q moment calibration is invalid for the active "
            f"{C.CSI_REPRESENTATION!r} cache representation"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--root-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = _load_locked(args)
    source = locked["source"]
    danger_bias = float(source.get("danger_logit_bias", 0.0))
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model = _configure(args, locked, device).eval()
    candidates = []
    baseline = None
    modes = ("clean", "gain_phase", "gain_phase_alt", "gain_phase_trial")
    for strength in args.strengths:
        model.set_calibration(strength)
        results = {
            mode: _evaluate(
                model, loaders["val"].dataset,
                loaders["val_class"].dataset, mode, args, device, danger_bias,
            )
            for mode in modes
        }
        if strength == 0.0:
            baseline = results["clean"]
        if baseline is None:
            raise RuntimeError("strength list must begin with 0.0")
        candidates.append({
            "strength": strength,
            "clean_feasible": _feasible(results["clean"], baseline),
            "robust_score": _robust_score(results),
            "results": results,
        })
        print(json.dumps(candidates[-1], ensure_ascii=False))

    feasible = [item for item in candidates if item["clean_feasible"]]
    selected = min(feasible, key=lambda item: item["robust_score"])
    report = {
        "run": "p2_v11_input_moment_calibration",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_rule": (
            "minimum mean physics-perturbation score subject to locked clean "
            "pose, classification, danger-recall and false-alarm constraints"
        ),
        "source": {
            "p2_checkpoint": str(args.p2_checkpoint),
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "root_expert_checkpoint": str(args.root_expert_checkpoint),
            "calibration": str(args.calibration),
        },
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
