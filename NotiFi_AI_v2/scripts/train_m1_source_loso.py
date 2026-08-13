"""Train and evaluate the M1 encoder with nested source-only LOSO."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notifi_ai_v2.data import (  # noqa: E402
    CacheIndex,
    CrossSiteClassBatchSampler,
    CsiPoseDataset,
    nested_source_split,
    read_link_quality,
    reserve_support,
)
from notifi_ai_v2.metrics import selection_score  # noqa: E402
from notifi_ai_v2.model import (  # noqa: E402
    MotionCalibratedEncoder,
    MotionEncoderConfig,
)
from notifi_ai_v2.training import (  # noqa: E402
    augment_amp_phase,
    cpu_state_dict,
    evaluate,
    set_seed,
    training_loss,
)


def records_at_sites(records, sites):
    allowed = set(sites)
    return [record for record in records if record.site in allowed]


def datasets_by_site(cache_root, records, link_quality):
    sites = sorted({record.site for record in records})
    return [
        CsiPoseDataset(
            cache_root,
            [record for record in records if record.site == site],
            link_quality,
        )
        for site in sites
    ]


def train_fold(options, held_out, query_records, link_quality, device, fold_number):
    train_sites, validation_sites, outer_sites = nested_source_split(
        query_records, held_out
    )
    train_dataset = CsiPoseDataset(
        options.cache_root,
        records_at_sites(query_records, train_sites),
        link_quality,
    )
    validation = datasets_by_site(
        options.cache_root,
        records_at_sites(query_records, validation_sites),
        link_quality,
    )
    outer = datasets_by_site(
        options.cache_root,
        records_at_sites(query_records, outer_sites),
        link_quality,
    )
    fold_seed = options.seed + fold_number * 1009
    set_seed(fold_seed)
    model_config = MotionEncoderConfig(
        hidden=options.hidden,
        temporal_layers=options.temporal_layers,
        dropout=options.dropout,
        motion_targets=8,
    )
    model = MotionCalibratedEncoder(model_config).to(device)
    sampler = CrossSiteClassBatchSampler(
        train_dataset, options.batch_size, fold_seed
    )
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=0,
        pin_memory=device == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    total_steps = max(1, options.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.08 + 0.92 * 0.5 * (
            1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)
        ),
    )
    risk_weight = torch.ones(3, device=device)
    risk_weight /= risk_weight.mean()
    generator = torch.Generator(device=device).manual_seed(fold_seed + 77)
    best = None
    history = []
    stale = 0
    for epoch in range(1, options.epochs + 1):
        model.train()
        rows = []
        started = time.perf_counter()
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            augmented_csi, augmented_mask = augment_amp_phase(
                batch["csi"], batch["link_mask"], generator
            )
            clean = model(
                batch["csi"], batch["link_mask"], representation="amp_phase"
            )
            augmented = model(
                augmented_csi, augmented_mask, representation="amp_phase"
            )
            loss, parts = training_loss(clean, augmented, batch, risk_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            rows.append(parts)
        site_metrics, macro = evaluate(
            model, validation, device, options.eval_batch_size
        )
        score = selection_score(macro)
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": {
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
            },
            "inner_validation": macro,
            "inner_validation_sites": site_metrics,
            "selection_score": score,
        }
        history.append(record)
        print(json.dumps({
            "held_out": held_out,
            "epoch": epoch,
            "score": score,
            "action_f1": macro["action_macro_f1"],
            "risk_f1": macro["risk_macro_f1"],
            "danger_recall": macro["danger_recall"],
            "safe_to_danger": macro["safe_to_danger_rate"],
            "seconds": record["seconds"],
        }), flush=True)
        if best is None or score > best["score"] + 1e-5:
            best = {
                "epoch": epoch,
                "score": score,
                "state": cpu_state_dict(model),
                "inner_validation": macro,
            }
            stale = 0
        else:
            stale += 1
            if epoch >= options.minimum_epochs and stale >= options.patience:
                break
    model.load_state_dict(best["state"])
    outer_sites_metrics, outer_macro = evaluate(
        model, outer, device, options.eval_batch_size
    )
    checkpoint = {
        "model": best["state"],
        "model_config": model_config.__dict__,
        "held_out_subject": held_out,
        "train_sites": train_sites,
        "inner_validation_sites": validation_sites,
        "outer_test_sites": outer_sites,
        "selected_epoch": best["epoch"],
        "inner_validation": best["inner_validation"],
        "outer_test": outer_sites_metrics,
        "protocol_version": "notifi_ai_v2_m1_source_loso_v1",
        "outer_holdout_used_for_selection": False,
        "sealed_yja_used": False,
        "support_used_by_model": False,
    }
    torch.save(checkpoint, options.run_dir / f"m1_{held_out}.pt")
    return {
        key: value for key, value in checkpoint.items() if key not in {"model"}
    } | {"history": history, "outer_macro": outer_macro}


def aggregate(folds):
    rows = [
        site
        for fold in folds.values()
        for site in fold["outer_test"].values()
    ]
    keys = (
        "action_accuracy", "action_macro_f1", "risk_accuracy",
        "risk_macro_f1", "danger_recall", "danger_action_accuracy",
        "safe_to_danger_rate", "motion_mae",
    )
    output = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    output["worst_site_action_accuracy"] = float(
        min(row["action_accuracy"] for row in rows)
    )
    output["sites"] = len(rows)
    output["trials"] = sum(int(row["trials"]) for row in rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", choices=("ajh", "mhw", "lmh"), default=("ajh", "mhw", "lmh"))
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--minimum-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=24017)
    options = parser.parse_args()
    options.cache_root = options.cache_root.resolve()
    options.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = CacheIndex(options.cache_root)
    source = index.source_development_records()
    support, query = reserve_support(source)
    if any(record.subject == "yja" for record in support + query):
        raise RuntimeError("sealed yja appeared after source split")
    link_quality = read_link_quality(options.cache_root)
    fold_results = {}
    for number, held_out in enumerate(options.folds):
        fold_results[held_out] = train_fold(
            options, held_out, query, link_quality, device, number
        )
    result = {
        "run": "NotiFi_AI_v2_M1_motion_encoder",
        "device": device,
        "cache_root": str(options.cache_root),
        "source_trials_before_support_reserve": len(source),
        "reserved_support_trials": len(support),
        "query_trials": len(query),
        "folds": fold_results,
        "outer_macro": aggregate(fold_results),
        "selection_protocol": "nested_source_subject_loso_m1_v1",
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "support_used_by_model": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(options).items()
        },
    }
    (options.run_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["outer_macro"], indent=2), flush=True)


if __name__ == "__main__":
    main()
