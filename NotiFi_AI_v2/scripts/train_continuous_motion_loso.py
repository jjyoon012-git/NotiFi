"""Train a retrieval-free continuous CSI-to-pose generator with nested LOSO."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import class_prototypes, embed_site  # noqa: E402
from evaluate_motion_signature_ridge_pose import target_support_motion  # noqa: E402
from notifi_ai_v2.continuous_motion import (  # noqa: E402
    ContinuousMotionGenerator,
    continuous_motion_loss,
)
from notifi_ai_v2.motion_residual import local_bones  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal17 import cal17_action, cal17_risk  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.pose_simulation import retrieval_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402


DISTAL_JOINTS = (7, 8, 10, 11, 20, 21)


def seed_everything(seed: int) -> None:
    """Make initialization and weighted sampling deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def danger_sampler(risk: torch.Tensor, seed: int) -> WeightedRandomSampler:
    """Oversample source danger trials without changing validation data."""
    weight = torch.where(risk == 2, torch.tensor(3.0), torch.tensor(1.0))
    return WeightedRandomSampler(
        weight, len(weight), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class DirectMotionDataset(Dataset):
    """Store calibrated frame features and source-only pose supervision."""

    def __init__(self, payloads: list[dict[str, torch.Tensor]]) -> None:
        keys = (
            "features", "target", "frame_mask", "valid",
            "action_probability", "risk_probability", "risk",
        )
        self.data = {
            key: torch.cat([payload[key] for payload in payloads]) for key in keys
        }

    def __len__(self) -> int:
        return len(self.data["risk"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


def direct_payload(
    site: str,
    source_sites: list[str],
    model: torch.nn.Module,
    embedded: dict[str, dict[str, torch.Tensor]],
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    row_sites: np.ndarray,
    pose_array: np.ndarray,
    valid_array: np.ndarray,
    action_config: dict,
    risk_config: dict,
    device: str,
    support_seed: int,
    absence_trials: int,
) -> dict[str, torch.Tensor]:
    """Prepare direct targets while excluding every calibration support trial."""
    library = [
        {
            "classes": class_prototypes(embedded[name]),
            "anchors": embedded[name]["anchors"],
        }
        for name in source_sites
    ]
    action = cal17_action(embedded[site], library, action_config)
    risk = cal17_risk(model, embedded[site], action, risk_config)
    _, keep = target_support_motion(
        model, embedded[site], site, store, index, selected_rows, row_sites,
        device, support_seed, absence_trials,
    )
    query_rows = embedded[site]["query_rows"][keep].numpy()
    target = torch.from_numpy(np.asarray(pose_array[query_rows]).copy()).float()
    valid = torch.from_numpy(np.asarray(valid_array[query_rows]).copy()).bool()
    valid &= torch.isfinite(target).all(-1).all(-1)
    target = torch.nan_to_num(target)
    frame_mask = embedded[site]["frame_mask"][keep].bool()
    return {
        "features": embedded[site]["features"][keep].float(),
        "query_rows": torch.from_numpy(query_rows.copy()).long(),
        "target": target,
        "frame_mask": frame_mask,
        "valid": valid & frame_mask,
        "action_probability": action[keep].softmax(-1).float(),
        "risk_probability": risk[keep].softmax(-1).float(),
        "label": embedded[site]["labels"][keep].long(),
        "risk": embedded[site]["risks"][keep].long(),
    }


def fit_skeleton(dataset: DirectMotionDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit body lengths and a neutral direction prior on training folds only."""
    pose = dataset.data["target"]
    valid = dataset.data["valid"]
    bones = local_bones(pose)
    lengths = torch.linalg.vector_norm(bones, dim=-1)
    fitted_lengths = []
    directions = []
    for joint in range(C.N_JOINTS):
        if joint == C.ROOT_JOINT:
            fitted_lengths.append(torch.tensor(0.0))
            directions.append(torch.tensor((0.0, 1.0, 0.0)))
            continue
        values = lengths[:, :, joint][valid]
        fitted_lengths.append(values.median())
        unit = F.normalize(bones[:, :, joint][valid], dim=-1)
        direction = F.normalize(unit.mean(0), dim=-1)
        if not bool(torch.isfinite(direction).all()):
            direction = torch.tensor((0.0, 1.0, 0.0))
        directions.append(direction)
    return torch.stack(fitted_lengths), torch.stack(directions)


def selection_score(metrics: dict[str, float]) -> float:
    """Balance absolute, shape-aligned, danger, and distal reconstruction."""
    return float(
        0.20 * metrics["pose_cm"]
        + 0.15 * metrics["distal_cm"]
        + 0.20 * metrics["pa_pose_cm"]
        + 0.20 * metrics["danger_pose_cm"]
        + 0.25 * metrics["danger_distal_cm"]
    )


@torch.no_grad()
def evaluate(
    model: ContinuousMotionGenerator,
    dataset: DirectMotionDataset,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate direct generation without labels or pose GT in model inputs."""
    model.eval()
    predictions, targets, valids, risks = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(
            batch["features"].to(device), batch["frame_mask"].to(device),
            batch["action_probability"].to(device),
            batch["risk_probability"].to(device),
        )
        predictions.append(output["pose_rel"].cpu())
        targets.append(batch["target"])
        valids.append(batch["valid"])
        risks.append(batch["risk"])
    return retrieval_metrics(
        torch.cat(predictions), torch.cat(targets), torch.cat(valids),
        torch.cat(risks),
    )


def train_generator(
    training: DirectMotionDataset,
    validation: DirectMotionDataset,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[ContinuousMotionGenerator, dict]:
    """Select a direct generator epoch using inner environments only."""
    seed_everything(seed)
    lengths, directions = fit_skeleton(training)
    model = ContinuousMotionGenerator(
        training.data["features"].shape[-1], lengths, directions
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    sampler = danger_sampler(training.data["risk"], seed)
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler)
    best = {"score": math.inf, "epoch": 0, "state": None, "metrics": None}
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rows = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["features"].to(device),
                batch["frame_mask"].to(device),
                batch["action_probability"].to(device),
                batch["risk_probability"].to(device),
            )
            loss, parts = continuous_motion_loss(
                output, batch["target"].to(device),
                batch["valid"].to(device), batch["risk"].to(device),
                DISTAL_JOINTS,
            )
            loss.backward()
            invalid = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if invalid:
                raise FloatingPointError(
                    "continuous generator produced non-finite gradients: "
                    + ", ".join(invalid)
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rows.append(parts)
        scheduler.step()
        metrics = evaluate(model, validation, device, batch_size * 2)
        score = selection_score(metrics)
        record = {
            "epoch": epoch,
            "train": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "validation": metrics,
            "score": score,
        }
        history.append(record)
        print(
            f"  epoch={epoch:02d} pose={metrics['pose_cm']:.2f} "
            f"pa={metrics['pa_pose_cm']:.2f} danger={metrics['danger_pose_cm']:.2f}",
            flush=True,
        )
        if score < best["score"] - 1e-4:
            best = {
                "score": score,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "metrics": metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    if best["state"] is None:
        raise RuntimeError("continuous generator produced no checkpoint")
    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "validation": best["metrics"],
        "history": history,
    }


def aggregate(folds: dict, key: str) -> dict[str, float]:
    """Aggregate subject folds in proportion to available environments."""
    subjects = ("ajh", "mhw", "lmh")
    weights = np.asarray((3, 3, 1), dtype=np.float64)
    names = (
        "pose_cm", "distal_cm", "pa_pose_cm", "danger_pose_cm",
        "danger_distal_cm",
    )
    return {
        name: float(np.average(
            [folds[subject][key][name] for subject in subjects], weights=weights
        ))
        for name in names
    }


def main() -> None:
    """Run retrieval-free source nested-LOSO training and outer evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--retrieval-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--training-seed", type=int, default=31081)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed target cannot enter direct generator development")
    row_sites = (selected.subject + "_" + selected.environment).to_numpy()
    all_sites = sorted(set(row_sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected source sites: {all_sites}")
    absence_rows = np.concatenate([
        np.flatnonzero(((index.subject == site.split("_")[0])
                        & (index.environment == site.split("_")[1])
                        & (index.task == C.TASK_CLS) & (index.class_id == 6)
                        & index.cache_ok).to_numpy())
        for site in all_sites
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    pose_array = np.load(work / "cache" / "pose_rel.npy", mmap_mode="r")
    valid_array = np.load(work / "cache" / "valid.npy", mmap_mode="r")
    calibration = json.loads(options.calibration.read_text(encoding="utf-8"))
    baseline = json.loads(options.retrieval_baseline.read_text(encoding="utf-8"))
    folds = {}
    for fold_number, held_out in enumerate(("ajh", "mhw", "lmh")):
        print(f"fold={held_out}", flush=True)
        train_sites, inner_sites, outer_sites = nested_site_split(all_sites, held_out)
        checkpoint = torch.load(
            options.run_dir / f"selection_{held_out}.pt", map_location="cpu",
            weights_only=False,
        )
        encoder = build_calibration_model(checkpoint["model_config"]).to(device)
        encoder.load_state_dict(checkpoint["model"])
        encoder.eval()
        embedded = {
            site: embed_site(
                encoder, store, index, selected_rows, row_sites, site, device,
                options.support_seed, options.support_seed + 1, 2, None,
                options.absence_trials,
            )
            for site in train_sites + inner_sites + outer_sites
        }
        config = calibration["folds"][held_out]
        common = {
            "model": encoder,
            "embedded": embedded,
            "store": store,
            "index": index,
            "selected_rows": selected_rows,
            "row_sites": row_sites,
            "pose_array": pose_array,
            "valid_array": valid_array,
            "action_config": config["action_config"],
            "risk_config": config["risk_config"],
            "device": device,
            "support_seed": options.support_seed,
            "absence_trials": options.absence_trials,
        }
        training_payloads = [direct_payload(
            site=site,
            source_sites=[name for name in train_sites if name != site],
            **common,
        ) for site in train_sites]
        validation_payloads = [direct_payload(
            site=site, source_sites=train_sites, **common,
        ) for site in inner_sites]
        outer_payloads = [direct_payload(
            site=site, source_sites=train_sites, **common,
        ) for site in outer_sites]
        training = DirectMotionDataset(training_payloads)
        validation = DirectMotionDataset(validation_payloads)
        outer = DirectMotionDataset(outer_payloads)
        generator, selection = train_generator(
            training, validation, device,
            options.training_seed + fold_number,
            options.epochs, options.batch_size,
        )
        outer_metrics = evaluate(generator, outer, device, options.batch_size * 2)
        retrieval = baseline["folds"][held_out]["outer"]
        torch.save({
            "model_config": generator.config(),
            "model": generator.state_dict(),
            "held_out_subject": held_out,
            "selection": selection,
            "outer_holdout_used_for_selection": False,
        }, options.output_dir / f"continuous_{held_out}.pt")
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": inner_sites,
            "outer_test_sites": outer_sites,
            "training_trials": len(training),
            "validation_trials": len(validation),
            "outer_trials": len(outer),
            "selection": selection,
            "retrieval_baseline": retrieval,
            "outer": outer_metrics,
            "outer_used_for_selection": False,
        }
        print(
            f"  outer retrieval={retrieval['pose_cm']:.2f}/"
            f"{retrieval['danger_pose_cm']:.2f} direct="
            f"{outer_metrics['pose_cm']:.2f}/{outer_metrics['danger_pose_cm']:.2f}",
            flush=True,
        )
        del encoder, generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "run": "NOTIFI-AI-V3-DIRECT-CONTINUOUS-MOTION",
        "protocol": "source nested-LOSO; retrieval-free pose generation",
        "folds": folds,
        "aggregate": {
            "retrieval_baseline": aggregate(folds, "retrieval_baseline"),
            "direct": aggregate(folds, "outer"),
        },
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_target_used": False,
        "motion_bank_used_for_pose": False,
        "query_labels_or_pose_gt_at_inference": False,
        "source_pose_gt_training_only": True,
    }
    (options.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
