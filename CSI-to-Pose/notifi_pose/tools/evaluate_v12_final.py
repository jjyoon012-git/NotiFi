"""Rebuild the validation-locked V12 multi-expert model and optionally open test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..hybrid_v10 import (
    ClassificationExpertBlend,
    HierarchicalRiskCalibration,
    PoseModelEnsemble,
    RootExpertBlend,
    SequenceBoneCalibration,
    SharedBackboneCache,
    SharedBackboneExecution,
    build_residual_hybrid,
)
from ..trainer import set_seed
from .calibrate_v11_residual_temporal import (
    ResidualTemporalCalibration,
    _checked_checkpoint,
)
from .evaluate_sealed import make_model
from .evaluate_v11_final import _same_path, evaluate_pa_mpjpe
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
    make_loaders,
)


def _read_locked(path: Path, protocol: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != protocol:
        raise RuntimeError(f"protocol mismatch in {path}")
    if report.get("selection_split") != "validation":
        raise RuntimeError(f"non-validation selection in {path}")
    if report.get("test_used_for_selection") is not False:
        raise RuntimeError(f"test was not proven sealed in {path}")
    return report


def _load_hybrid(p2: dict, path: Path, protocol: str, device: str,
                 pose_strength: float, root_strength: float,
                 shared_backbone: SharedBackboneCache | None = None):
    checkpoint = _checked_checkpoint(path, device, protocol)
    model = build_residual_hybrid(
        make_model(p2, device), checkpoint.get("residual_decoder", "subcarrier")
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    if shared_backbone is not None:
        reference = shared_backbone.base.state_dict()
        candidate = model.base.state_dict()
        identical = (
            reference.keys() == candidate.keys()
            and all(torch.equal(reference[key], candidate[key]) for key in reference)
        )
        if not identical:
            raise RuntimeError(f"expert backbone differs and cannot be shared: {path}")
        model.base = shared_backbone
    model.set_calibration(pose_strength, root_strength, 0.0, 0.0)
    return model, checkpoint


def build_locked_model(args, device: str, root_lock: dict,
                       class_lock: dict, share_backbone: bool = True):
    source = root_lock["source"]
    if not _same_path(source["p2_checkpoint"], args.p2_checkpoint):
        raise RuntimeError("P2 checkpoint differs from root calibration source")
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)

    pose_paths = [Path(path) for path in source["pose_checkpoints"]]
    shared_backbone = None
    if share_backbone:
        first_pose, _ = _load_hybrid(
            p2, pose_paths[0], args.exp, device, 1.0, 0.0
        )
        shared_backbone = SharedBackboneCache(first_pose.base).to(device)
        first_pose.base = shared_backbone
        pose_models = [first_pose]
        pose_models.extend(
            _load_hybrid(
                p2, path, args.exp, device, 1.0, 0.0, shared_backbone
            )[0]
            for path in pose_paths[1:]
        )
    else:
        pose_models = [
            _load_hybrid(p2, path, args.exp, device, 1.0, 0.0)[0]
            for path in pose_paths
        ]
    pose = PoseModelEnsemble(
        pose_models, list(source["pose_weights"])
    ).to(device)

    root_paths = [Path(path) for path in source["root_checkpoints"]]
    root_models = []
    for path in root_paths:
        model, checkpoint = _load_hybrid(
            p2, path, args.exp, device, 0.0, 1.0, shared_backbone
        )
        if checkpoint.get("objective") != "root_only":
            raise RuntimeError(f"non-root checkpoint in root ensemble: {path}")
        root_models.append(model)
    root = PoseModelEnsemble(
        root_models, list(root_lock["selected"]["root_weights"])
    ).to(device)

    rooted = RootExpertBlend(pose, root).to(device)
    rooted.set_root_strength(1.0)
    temporal = ResidualTemporalCalibration(rooted).to(device)
    temporal.set_calibration(
        int(source["window"]), float(source["blend"]), "probability",
        float(source["danger_logit_bias"]),
    )
    geometric = SequenceBoneCalibration(
        temporal, blend=float(source["bone_blend"]),
        symmetric=bool(source["bone_symmetric"]),
    ).to(device)

    class_source = class_lock["source"]
    if not _same_path(class_source["p2_checkpoint"], args.p2_checkpoint):
        raise RuntimeError("classification calibration uses a different P2")
    if "classification_expert_checkpoints" in class_source:
        class_paths = [
            Path(path) for path in class_source["classification_expert_checkpoints"]
        ]
        class_models = []
        for path in class_paths:
            model, checkpoint = _load_hybrid(
                p2, path, args.exp, device, 0.0, 0.0, shared_backbone
            )
            if checkpoint.get("objective") != "classification_only":
                raise RuntimeError("classification expert has the wrong objective")
            model.set_calibration(0.0, 0.0, 1.0, 1.0)
            class_models.append(model)
        class_expert = PoseModelEnsemble(
            class_models, list(class_source["classification_expert_weights"])
        ).to(device)
    else:
        class_paths = [Path(class_source["classification_expert_checkpoint"])]
        class_expert, checkpoint = _load_hybrid(
            p2, class_paths[0], args.exp, device, 0.0, 0.0,
            shared_backbone
        )
        if checkpoint.get("objective") != "classification_only":
            raise RuntimeError("classification expert has the wrong objective")
        class_expert.set_calibration(0.0, 0.0, 1.0, 1.0)
    classified = ClassificationExpertBlend(geometric, class_expert).to(device)
    classified.set_calibration(
        float(class_source.get("class_expert_strength", 1.0)), 0.0
    )
    selected_class = class_lock["selected"]
    calibrated = HierarchicalRiskCalibration(
        classified,
        class_weight=float(selected_class["class_weight"]),
        danger_logit_bias=float(selected_class["danger_logit_bias"]),
    ).to(device)
    final = (
        SharedBackboneExecution(calibrated, shared_backbone).to(device)
        if shared_backbone is not None else calibrated
    )
    configuration = {
        "p2_checkpoint": str(args.p2_checkpoint),
        "pose_checkpoints": [str(path) for path in pose_paths],
        "pose_weights": list(source["pose_weights"]),
        "root_checkpoints": [str(path) for path in root_paths],
        "root_weights": list(root_lock["selected"]["root_weights"]),
        "residual_window": int(source["window"]),
        "residual_blend": float(source["blend"]),
        "bone_blend": float(source["bone_blend"]),
        "bone_symmetric": bool(source["bone_symmetric"]),
        "classification_expert_checkpoints": [
            str(path) for path in class_paths
        ],
        "classification_expert_weights": list(
            class_source.get("classification_expert_weights", [1.0])
        ),
        "classification_expert_strength": float(
            class_source.get("class_expert_strength", 1.0)
        ),
        "hierarchical_class_weight": float(selected_class["class_weight"]),
        "final_danger_logit_bias": float(selected_class["danger_logit_bias"]),
        "shared_backbone_execution": share_backbone,
    }
    return final, configuration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    model, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    model.eval()

    validation = evaluate_trajectory(
        model, loaders["val"], device, args.max_shift
    )
    validation["pa_mpjpe_m"] = evaluate_pa_mpjpe(
        model, loaders["val"], device
    )
    validation_classification = evaluate_classification(
        model, loaders["val_class"], device, 0.0
    )
    result = {
        "run": "p2_v12_final_locked_evaluation",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_opened": bool(args.open_test),
        "root_calibration": str(args.root_calibration),
        "classification_calibration": str(args.classification_calibration),
        "configuration": configuration,
        "validation": validation,
        "validation_classification": validation_classification,
    }
    if args.open_test:
        result["test"] = evaluate_trajectory(
            model, loaders["test"], device, args.max_shift
        )
        result["test"]["pa_mpjpe_m"] = evaluate_pa_mpjpe(
            model, loaders["test"], device
        )
        result["test_classification"] = evaluate_classification(
            model, loaders["test_class"], device, 0.0
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "model": model.state_dict(),
        "protocol": args.exp,
        "configuration": configuration,
        "root_calibration": str(args.root_calibration),
        "classification_calibration": str(args.classification_calibration),
        "validation": validation,
        "validation_classification": validation_classification,
        **({
            "test": result["test"],
            "test_classification": result["test_classification"],
        } if args.open_test else {}),
    }, args.output.with_name("final_model.pt"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
