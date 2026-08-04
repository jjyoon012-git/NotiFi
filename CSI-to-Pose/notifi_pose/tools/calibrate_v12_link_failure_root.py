"""Select a direct-root missing-link specialist on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..hybrid_v10 import ConditionalLinkFailureRootBlend
from ..trainer import set_seed
from .audit_v11_input_robustness import PerturbedDataset
from .evaluate_v12_final import _load_hybrid, _read_locked, build_locked_model
from .train_p2_v9_hybrid import root_selection_score
from .train_seen_v4_trajectory import evaluate_trajectory, make_loaders


def _root_summary(metrics: dict) -> dict:
    return {
        key: metrics[key] for key in (
            "root_error_m", "danger_root_error_m",
            "danger_root_drop_mae_m", "danger_mpjpe_m",
            "danger_endpoint_mpjpe_m",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loaders = make_loaders(args, device)
    primary, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    p2 = torch.load(args.p2_checkpoint, map_location=device, weights_only=False)
    expert, checkpoint = _load_hybrid(
        p2, args.expert_checkpoint, args.exp, device, 0.0, 1.0
    )
    if checkpoint.get("objective") != "root_only":
        raise RuntimeError("link-failure root expert must be root-only")
    model = ConditionalLinkFailureRootBlend(primary, expert).to(device).eval()
    clean_loader = loaders["val"]
    failure_loader = DataLoader(
        PerturbedDataset(clean_loader.dataset, "drop_one_link"),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    candidates = []
    baseline_clean = None
    for strength in args.strengths:
        model.set_strength(float(strength))
        clean = evaluate_trajectory(model, clean_loader, device, args.max_shift)
        failure = evaluate_trajectory(
            model, failure_loader, device, args.max_shift
        )
        if baseline_clean is None:
            baseline_clean = clean
        feasible = (
            clean["root_error_m"] <= baseline_clean["root_error_m"] + 0.003
            and clean["danger_root_error_m"]
            <= baseline_clean["danger_root_error_m"] + 0.005
        )
        candidates.append({
            "strength": float(strength),
            "feasible": bool(feasible),
            "failure_score": root_selection_score(failure),
            "clean": _root_summary(clean),
            "drop_one_link": _root_summary(failure),
        })
        print(
            f"strength={strength:.2f} clean_root={clean['root_error_m'] * 100:.2f}cm "
            f"drop_root={failure['root_error_m'] * 100:.2f}cm "
            f"danger_root={failure['danger_root_error_m'] * 100:.2f}cm "
            f"feasible={feasible}"
        )
    feasible = [item for item in candidates if item["feasible"]]
    selected = min(feasible or candidates, key=lambda item: item["failure_score"])
    report = {
        "run": "p2_v12_link_failure_root_calibration",
        "protocol": args.exp,
        "selection_split": "validation_drop_one_link",
        "test_used_for_selection": False,
        "source_configuration": configuration,
        "expert_checkpoint": str(args.expert_checkpoint),
        "expert_training_config": checkpoint.get("training_config", {}),
        "selection_rule": {
            "clean_root_tolerance_m": 0.003,
            "clean_danger_root_tolerance_m": 0.005,
        },
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
