"""Train support-conditioned FiLM before P2 temporal encoding and test V13S."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..calibration_aware import SupportConditionedP2
from ..dataio.dataset import build_datasets
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_sealed import make_model
from .evaluate_v11_final import evaluate_pa_mpjpe
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_calibration_aware_v14 import (
    ActiveSupportModel,
    META_VALIDATION_SITES,
    _weights,
    adapter_state,
    build_profiles,
    calibration_loss,
    combine_classification,
    combine_trajectory,
    episode_augment,
    evaluate_sites,
    load_adapter_state,
    profile_normalization,
    selection_score,
    site_name,
    split_support_queries,
    subset_dataset,
    support_profile,
)
from .train_seen_v4_trajectory import (
    evaluate_classification,
    evaluate_trajectory,
)


def build_conditioned_p2(args, device: str, support_mean: torch.Tensor,
                         support_std: torch.Tensor) -> tuple[SupportConditionedP2, dict]:
    checkpoint = torch.load(
        args.p2_checkpoint, map_location=device, weights_only=False
    )
    base = make_model(checkpoint, device)
    model = SupportConditionedP2(
        base, support_hidden=args.support_hidden
    ).to(device)
    model.set_support_normalization(
        support_mean.to(device), support_std.to(device)
    )
    return model, checkpoint["cfg"]


def train_conditioner(model: SupportConditionedP2, train_dataset,
                      queries: dict[str, np.ndarray],
                      profiles: dict[str, torch.Tensor],
                      sites: tuple[str, ...], epochs: int, args, device: str,
                      validation_dataset=None,
                      validation_sites: tuple[str, ...] = ()) -> tuple[dict, list[dict]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    positions = np.concatenate([queries[site] for site in sites])
    weight_dataset = subset_dataset(train_dataset, positions)
    class_weight = _weights(weight_dataset, "class_id", C.N_CLASSES, device)
    risk_weight = _weights(
        weight_dataset, "risk_id", C.N_RISK, device, danger_boost=2.0
    )
    best_state = adapter_state(model)
    best_score = math.inf
    best_epoch = 0
    history = []
    model.set_strength(1.0)
    if validation_dataset is not None:
        model.set_strength(0.0)
        baseline = evaluate_sites(
            model, validation_dataset, profiles, validation_sites,
            args.batch_size, device, args.max_shift,
        )
        model.set_strength(1.0)
        best_score = selection_score(baseline)
        history.append({"epoch": 0, "validation": baseline, "score": best_score})
    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        rng = np.random.default_rng(args.seed + epoch * 1009)
        ordered_sites = list(sites)
        rng.shuffle(ordered_sites)
        for site in ordered_sites:
            query_dataset = subset_dataset(train_dataset, queries[site], train=True)
            query_dataset.set_epoch(epoch)
            loader = DataLoader(
                query_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            raw_profile = profiles[site].to(device)
            for batch in loader:
                csi = batch["csi"].to(device, non_blocking=True)
                mask = batch["link_mask"].to(device, non_blocking=True)
                csi, profile = episode_augment(csi, raw_profile)
                moved = {
                    key: value.to(device, non_blocking=True)
                    if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                output = model(csi, mask, profile)
                loss, parts = calibration_loss(
                    output, moved, class_weight, risk_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                batches += 1
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + value
        row = {
            "epoch": epoch,
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
        }
        if validation_dataset is not None:
            model.eval()
            validation = evaluate_sites(
                model, validation_dataset, profiles, validation_sites,
                args.batch_size, device, args.max_shift,
            )
            score = selection_score(validation)
            row.update({"validation": validation, "score": score})
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = adapter_state(model)
        else:
            best_epoch = epoch
            best_state = adapter_state(model)
        history.append(row)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": row["train"].get("loss"),
            "validation_score": row.get("score"),
            "best_epoch": best_epoch,
        }), flush=True)
    load_adapter_state(model, best_state)
    return {
        "state": best_state, "epoch": best_epoch, "score": best_score
    }, history


def select_conditioning_strength(model: SupportConditionedP2, validation,
                                 profiles, sites, args, device):
    candidates = []
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        model.set_strength(strength)
        result = evaluate_sites(
            model, validation, profiles, sites,
            args.batch_size, device, args.max_shift,
        )
        candidates.append({
            "strength": strength,
            "score": selection_score(result),
            "result": result,
        })
    selected = min(candidates, key=lambda item: item["score"])
    model.set_strength(selected["strength"])
    return selected, candidates


@torch.no_grad()
def evaluate_v13s_sites(v13s, conditioned_p2: SupportConditionedP2,
                        dataset, profiles, sites, batch_size, device, max_shift):
    names = site_name(dataset.index)
    trajectories = []
    classifications = []
    for site in sites:
        positions = np.flatnonzero(names == site)
        selected = subset_dataset(dataset, positions)
        loader = DataLoader(selected, batch_size=batch_size, shuffle=False)
        conditioned_p2.set_profile(profiles[site])
        trajectories.append((len(selected), evaluate_trajectory(
            v13s, loader, device, max_shift
        )))
        classifications.append(evaluate_classification(v13s, loader, device, 0.0))
    return {
        "trajectory": combine_trajectory(trajectories),
        "classification": combine_classification(classifications),
    }


def build_v13s_with_conditioned_p2(args, conditioned_p2, device):
    root_lock = _read_locked(args.root_calibration, args.source_exp)
    class_lock = _read_locked(args.classification_calibration, args.source_exp)
    base_args = argparse.Namespace(
        p2_checkpoint=args.p2_checkpoint, exp=args.source_exp
    )
    model, configuration = build_locked_model(
        base_args, device, root_lock, class_lock
    )
    if not hasattr(model, "backbone"):
        raise RuntimeError("V13S shared backbone was not created")
    model.backbone.base = conditioned_p2
    return model.to(device).eval(), configuration


def evaluate_target(model, loader, device, max_shift, pa: bool = True):
    result = {
        "trajectory": evaluate_trajectory(model, loader, device, max_shift),
        "classification": evaluate_classification(model, loader, device, 0.0),
    }
    if pa:
        result["trajectory"]["pa_mpjpe_m"] = evaluate_pa_mpjpe(
            model, loader, device
        )
    return result


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
        default=Path("work_v2/runs/p2_v12w_robust_classification_ensemble/validation.json"),
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
        default=Path("work_v2/runs/v15_support_conditioned_p2"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("docs/results/v15_support_conditioned_p2_yja.json"),
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
    all_sites = tuple(sorted(source_profiles))
    meta_val_sites = tuple(site for site in META_VALIDATION_SITES if site in all_sites)
    meta_train_sites = tuple(site for site in all_sites if site not in meta_val_sites)

    model, p2_configuration = build_conditioned_p2(
        args, device, support_mean, support_std
    )
    meta_best, meta_history = train_conditioner(
        model, source_train, source_query, source_profiles,
        meta_train_sites, args.epochs, args, device,
        validation_dataset=source_val,
        validation_sites=meta_val_sites,
    )
    strength, strength_candidates = select_conditioning_strength(
        model, source_val, source_profiles, meta_val_sites, args, device
    )
    selected_epoch = int(meta_best["epoch"])

    del model
    torch.cuda.empty_cache()
    set_seed(args.seed + 1)
    final_p2, _ = build_conditioned_p2(
        args, device, support_mean, support_std
    )
    _, final_history = train_conditioner(
        final_p2, source_train, source_query, source_profiles,
        all_sites, selected_epoch, args, device,
    )
    final_p2.set_strength(float(strength["strength"]))
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
    target_profile = support_profile(
        target, target_support["yja_E02"], "yja_E02"
    )
    target_test = subset_dataset(target, target_query["yja_E02"])
    target_loader = DataLoader(
        target_test, batch_size=args.batch_size, shuffle=False
    )
    active_p2 = ActiveSupportModel(final_p2).to(device).eval()
    active_p2.set_profile(target_profile)
    final_p2.set_strength(0.0)
    frozen_p2_target = evaluate_target(
        active_p2, target_loader, device, args.max_shift
    )
    final_p2.set_strength(float(strength["strength"]))
    conditioned_p2_target = evaluate_target(
        active_p2, target_loader, device, args.max_shift
    )

    v13s, v13s_configuration = build_v13s_with_conditioned_p2(
        args, final_p2, device
    )
    final_p2.set_profile(target_profile)
    final_p2.set_strength(0.0)
    frozen_v13s_target = evaluate_target(
        v13s, target_loader, device, args.max_shift
    )
    final_p2.set_strength(float(strength["strength"]))
    conditioned_v13s_target = evaluate_target(
        v13s, target_loader, device, args.max_shift
    )
    v13s_source_validation = evaluate_v13s_sites(
        v13s, final_p2, source_val, source_profiles, all_sites,
        args.batch_size, device, args.max_shift,
    )

    report = {
        "run": "v15_support_conditioned_p2",
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
            "target_support_trials": len(target_support["yja_E02"]),
            "target_test_trials": len(target_test),
            "target_danger_test_trials": int((target_test.index.risk_id == 2).sum()),
        },
        "p2_configuration": p2_configuration,
        "v13s_configuration": v13s_configuration,
        "selected_epoch": selected_epoch,
        "selected_strength": float(strength["strength"]),
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
        "selected_epoch": selected_epoch,
        "selected_strength": float(strength["strength"]),
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
        "selected_strength": strength["strength"],
        "source_validation": report["source_validation"],
        "target_test": report["target_test"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
