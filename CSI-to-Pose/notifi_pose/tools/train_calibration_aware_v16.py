"""Train support moment alignment plus early FiLM, then evaluate sealed yja/E02."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..calibration_aware import MomentAlignedSupportConditionedP2
from ..dataio.dataset import build_datasets
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_sealed import make_model
from .train_calibration_aware_v14 import (
    ActiveSupportModel,
    META_VALIDATION_SITES,
    adapter_state,
    build_profiles,
    evaluate_sites,
    profile_normalization,
    selection_score,
    split_support_queries,
    subset_dataset,
    support_profile,
)
from .train_calibration_aware_v15 import (
    build_v13s_with_conditioned_p2,
    evaluate_target,
    evaluate_v13s_sites,
    train_conditioner,
)


def reference_statistics(
    profiles: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    moments = [
        MomentAlignedSupportConditionedP2.profile_moments(profile)
        for profile in profiles.values()
    ]
    means = torch.stack([item[0] for item in moments])
    stds = torch.stack([item[1] for item in moments])
    mean = means.mean(dim=0)
    second = (stds.square() + means.square()).mean(dim=0)
    std = (second - mean.square()).clamp_min(1e-8).sqrt()
    return mean, std


def build_model(args, device: str, support_mean: torch.Tensor,
                support_std: torch.Tensor, reference_mean: torch.Tensor,
                reference_std: torch.Tensor):
    checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    base = make_model(checkpoint, device)
    model = MomentAlignedSupportConditionedP2(
        base, support_hidden=args.support_hidden
    ).to(device)
    model.set_support_normalization(
        support_mean.to(device), support_std.to(device)
    )
    model.set_reference_statistics(
        reference_mean.to(device), reference_std.to(device)
    )
    return model, checkpoint["cfg"]


def select_strengths(model, validation, profiles, sites, args, device):
    candidates = []
    for alignment in (0.0, 0.25, 0.5, 0.75, 1.0):
        for film in (0.0, 0.5, 1.0):
            model.set_alignment_strength(alignment)
            model.set_strength(film)
            result = evaluate_sites(
                model, validation, profiles, sites,
                args.batch_size, device, args.max_shift,
            )
            candidates.append({
                "alignment_strength": alignment,
                "film_strength": film,
                "score": selection_score(result),
                "result": result,
            })
    selected = min(candidates, key=lambda item: item["score"])
    model.set_alignment_strength(selected["alignment_strength"])
    model.set_strength(selected["film_strength"])
    return selected, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=Path("work_v2/runs/p2_sub_single_clean_finetune/best_model.pt"),
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=Path("docs/results/v13s_pruned_pose_root_ensemble.json"),
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=Path(
            "work_v2/runs/p2_v12w_robust_classification_ensemble/validation.json"
        ),
    )
    parser.add_argument("--source-exp", default="single_split_lmh_e01")
    parser.add_argument("--target-fold", default="yja_E02")
    parser.add_argument("--support-per-pose", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--support-hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("work_v2/runs/v16_moment_aligned_p2"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("docs/results/v16_moment_aligned_p2_yja.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = build_datasets(
        exp=args.source_exp, baseline="sub", seed=args.seed
    )
    source_train = pose_only(source["train"])
    source_val = pose_only(source["val"])
    source_support, source_query = split_support_queries(
        source_train, args.support_per_pose, args.seed
    )
    source_profiles = build_profiles(source_train, source_support)
    support_mean, support_std = profile_normalization(source_profiles)
    reference_mean, reference_std = reference_statistics(source_profiles)
    all_sites = tuple(sorted(source_profiles))
    meta_val_sites = tuple(
        site for site in META_VALIDATION_SITES if site in all_sites
    )
    meta_train_sites = tuple(
        site for site in all_sites if site not in meta_val_sites
    )

    model, p2_configuration = build_model(
        args, device, support_mean, support_std,
        reference_mean, reference_std,
    )
    meta_best, meta_history = train_conditioner(
        model, source_train, source_query, source_profiles,
        meta_train_sites, args.epochs, args, device,
        validation_dataset=source_val,
        validation_sites=meta_val_sites,
    )
    strength, strength_candidates = select_strengths(
        model, source_val, source_profiles, meta_val_sites, args, device
    )
    selected_epoch = int(meta_best["epoch"])

    del model
    torch.cuda.empty_cache()
    set_seed(args.seed + 1)
    final_p2, _ = build_model(
        args, device, support_mean, support_std,
        reference_mean, reference_std,
    )
    _, final_history = train_conditioner(
        final_p2, source_train, source_query, source_profiles,
        all_sites, selected_epoch, args, device,
    )
    final_p2.set_alignment_strength(strength["alignment_strength"])
    final_p2.set_strength(strength["film_strength"])
    p2_source_validation = evaluate_sites(
        final_p2, source_val, source_profiles, all_sites,
        args.batch_size, device, args.max_shift,
    )

    target = pose_only(build_datasets(
        exp="sealed", fold=args.target_fold, baseline="sub", seed=args.seed
    )["test"])
    target_support, target_query = split_support_queries(
        target, args.support_per_pose, args.seed
    )
    target_site = args.target_fold
    target_profile = support_profile(
        target, target_support[target_site], target_site
    )
    target_test = subset_dataset(target, target_query[target_site])
    target_loader = DataLoader(
        target_test, batch_size=args.batch_size, shuffle=False
    )
    active_p2 = ActiveSupportModel(final_p2).to(device).eval()
    active_p2.set_profile(target_profile)
    final_p2.set_alignment_strength(0.0)
    final_p2.set_strength(0.0)
    frozen_p2_target = evaluate_target(
        active_p2, target_loader, device, args.max_shift
    )
    final_p2.set_alignment_strength(strength["alignment_strength"])
    final_p2.set_strength(strength["film_strength"])
    conditioned_p2_target = evaluate_target(
        active_p2, target_loader, device, args.max_shift
    )

    v13s, v13s_configuration = build_v13s_with_conditioned_p2(
        args, final_p2, device
    )
    final_p2.set_profile(target_profile)
    final_p2.set_alignment_strength(0.0)
    final_p2.set_strength(0.0)
    frozen_v13s_target = evaluate_target(
        v13s, target_loader, device, args.max_shift
    )
    final_p2.set_alignment_strength(strength["alignment_strength"])
    final_p2.set_strength(strength["film_strength"])
    conditioned_v13s_target = evaluate_target(
        v13s, target_loader, device, args.max_shift
    )
    v13s_source_validation = evaluate_v13s_sites(
        v13s, final_p2, source_val, source_profiles, all_sites,
        args.batch_size, device, args.max_shift,
    )

    report = {
        "run": "v16_moment_aligned_support_conditioned_p2",
        "source_protocol": args.source_exp,
        "target_protocol": f"sealed/{args.target_fold}",
        "selection_protocol": {
            "meta_train_sites": list(meta_train_sites),
            "meta_validation_sites": list(meta_val_sites),
            "target_used_for_selection": False,
        },
        "support_protocol": {
            "classes": {"1": "standing", "2": "sitting", "3": "lying"},
            "trials_per_pose": args.support_per_pose,
            "warning_or_danger_support": 0,
            "target_gt_used_for_calibration": False,
            "target_support_trials": len(target_support[target_site]),
            "target_test_trials": len(target_test),
            "target_danger_test_trials": int(
                (target_test.index.risk_id == 2).sum()
            ),
        },
        "p2_configuration": p2_configuration,
        "v13s_configuration": v13s_configuration,
        "selected_epoch": selected_epoch,
        "selected_alignment_strength": strength["alignment_strength"],
        "selected_film_strength": strength["film_strength"],
        "meta_history": meta_history,
        "strength_candidates": strength_candidates,
        "final_history": final_history,
        "source_validation": {
            "conditioned_p2": p2_source_validation,
            "conditioned_v13s": v13s_source_validation,
        },
        "target_test": {
            "frozen_p2": frozen_p2_target,
            "conditioned_p2": conditioned_p2_target,
            "frozen_v13s": frozen_v13s_target,
            "conditioned_v13s": conditioned_v13s_target,
        },
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "run": report["run"],
        "source_protocol": args.source_exp,
        "p2_configuration": p2_configuration,
        "support_mean": support_mean,
        "support_std": support_std,
        "reference_mean": reference_mean,
        "reference_std": reference_std,
        "selected_epoch": selected_epoch,
        "selected_alignment_strength": strength["alignment_strength"],
        "selected_film_strength": strength["film_strength"],
        "adapter": adapter_state(final_p2),
    }, args.run_dir / "best_model.pt")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "report": str(args.report),
        "checkpoint": str(args.run_dir / "best_model.pt"),
        "selected_epoch": selected_epoch,
        "selected_alignment_strength": strength["alignment_strength"],
        "selected_film_strength": strength["film_strength"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
