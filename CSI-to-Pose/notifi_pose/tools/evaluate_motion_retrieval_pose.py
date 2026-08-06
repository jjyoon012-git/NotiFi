"""Evaluate deployable train-bank retrieval over the locked KP2-DH pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..conditioned_contact_pose import DirectionalConditionedContactPose
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from .audit_motion_retrieval_oracle import (
    PARTS,
    _canonicalize,
    _load_coarse,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .diagnose_observability import pose_only, report_path
from .train_geometry_phase_pose import build_model as build_source_model
from .train_kinetic_pose import CoarsePoseStore, pose_selection_score


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else C.PROJECT_ROOT / path


def _load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source_path = _resolve(checkpoint["source"]["hierarchical_checkpoint"])
    source_checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    source_model, _, source_architecture, _ = build_source_model(
        source_checkpoint, device
    )
    model = DirectionalConditionedContactPose(
        source_model, dropout=float(source_architecture.get("dropout", 0.08))
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.set_residual_strength(float(checkpoint.get("deployment_strength", 0.0)))
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _predict(model, loader, coarse_store: CoarsePoseStore,
             device: str) -> tuple[torch.Tensor, ...]:
    pose, action_logits, risks, rows = [], [], [], []
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            coarse_pose=coarse_store.lookup(batch["row"], device),
        )
        pose.append(output["pose_rel"].float().cpu())
        action_logits.append(output["action_logits"].float().cpu())
        risks.append(output["risk_logits"].float().cpu())
        rows.append(batch["row"].long())
    return (
        torch.cat(pose), torch.cat(action_logits),
        torch.cat(risks), torch.cat(rows),
    )


def _retrieve(query: torch.Tensor, train_bank: torch.Tensor,
              train_class: torch.Tensor, class_ids: list[int] | None,
              partwise: bool) -> torch.Tensor:
    if class_ids:
        keep = torch.zeros_like(train_class, dtype=torch.bool)
        for class_id in class_ids:
            keep |= train_class == int(class_id)
        members = train_bank[keep]
    else:
        members = train_bank
    error = torch.linalg.vector_norm(members - query[None], dim=-1)
    whole = members[error.mean((1, 2)).argmin()]
    if not partwise:
        return whole
    result = whole.clone()
    for joints in PARTS.values():
        part_error = error[:, :, joints].mean((1, 2))
        result[:, joints] = members[part_error.argmin(), :, joints]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--test-selector", choices=(
            "global", "predicted_class", "predicted_top2",
            "predicted_class_part",
        ), default="predicted_top2",
    )
    parser.add_argument("--test-strength", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp4_dcc_staged_seed17"
        / "deployment_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_motion_retrieval"
        / "validation.json",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("test evaluation requires explicit --allow-test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    target = QualityWeightedDataset(pose_only(datasets[args.split]), audit)
    train_pose, train_valid, train_class, _ = _load_pose_arrays(train)
    target_pose, target_valid, target_class, target_risk = _load_pose_arrays(target)
    train_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(train_pose, train_valid)
    ])

    model, checkpoint = _load_model(args.checkpoint, device)
    raw_cache = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    coarse_store = CoarsePoseStore(raw_cache["rows"], raw_cache["pose"])
    loader = DataLoader(
        target, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    baseline, action_logits, risk_logits, predicted_rows = _predict(
        model, loader, coarse_store, device
    )
    expected_rows = torch.from_numpy(target.target.rows).long()
    if not torch.equal(predicted_rows, expected_rows):
        raise RuntimeError("prediction order differs from target split order")
    predicted_class = action_logits.argmax(-1)
    top2_class = action_logits.topk(2, dim=-1).indices
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, target_valid)
    ])

    names = (
        "global", "predicted_class", "predicted_top2", "predicted_class_part"
    )
    if args.split == "test":
        names = (args.test_selector,)
    retrieved = {name: [] for name in names}
    for index, (query, valid) in enumerate(zip(baseline_bank, target_valid)):
        choices = {
            "global": (None, False),
            "predicted_class": ([int(predicted_class[index])], False),
            "predicted_top2": (top2_class[index].tolist(), False),
            "predicted_class_part": ([int(predicted_class[index])], True),
        }
        for name in names:
            classes, partwise = choices[name]
            motion = _retrieve(
                query, train_bank, train_class, classes, partwise
            )
            retrieved[name].append(
                _render(motion, valid, C.CACHE_FRAMES)
            )

    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    strengths = (
        (args.test_strength,) if args.split == "test"
        else (0.25, 0.50, 0.75, 1.0)
    )
    for name, values in retrieved.items():
        candidate = torch.stack(values)
        for strength in strengths:
            key = f"{name}_blend_{int(strength * 100):03d}"
            metrics[key] = _metric_batch(
                (1.0 - strength) * baseline + strength * candidate,
                target_pose, target_valid, target_risk,
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = (
        f"{args.test_selector}_blend_{int(args.test_strength * 100):03d}"
        if args.split == "test" else min(scores, key=scores.get)
    )
    result = {
        "status": "deployable_validation_candidate" if args.split == "val"
        else "promoted_candidate_test_confirmation",
        "protocol": args.exp,
        "split": args.split,
        "test_used_for_selection": False,
        "train_trials": len(train),
        "target_trials": len(target),
        "action_accuracy": float((predicted_class == target_class).float().mean()),
        "risk_accuracy": float(
            (risk_logits.argmax(-1) == target_risk).float().mean()
        ),
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "architecture": {
            "query": "locked KP2-DH 30 percent CSI pose",
            "bank": "train-only normalized-phase GT trajectories",
            "selectors": ["global", "predicted action", "predicted top-2 action"],
            "parts": list(PARTS),
        },
        "checkpoint": report_path(args.checkpoint),
        "source_run": checkpoint.get("run"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
