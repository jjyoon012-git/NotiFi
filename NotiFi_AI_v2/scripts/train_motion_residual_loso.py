"""Train and evaluate a CSI-conditioned residual pose decoder with nested LOSO."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
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
from evaluate_motion_signature_ridge_pose import (  # noqa: E402
    calibrated_query_signature,
    class_means,
    danger_gate,
    ensemble_prediction,
    pose_score,
    retrieval_payload,
    target_support_motion,
)
from notifi_ai_v2.motion_residual import (  # noqa: E402
    MotionResidualDecoder,
    motion_residual_loss,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal13 import pose_motion_descriptor, temporal_motion_signature  # noqa: E402
from notifi_pose.cal17 import cal17_action, cal17_risk  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.pose_simulation import fill_pose_gaps, retrieval_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402


DISTAL_JOINTS = (7, 8, 10, 11, 20, 21)


def seed_everything(seed: int) -> None:
    """Make decoder initialization and sampling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MotionDataset(Dataset):
    """Keep aligned CSI features, retrieval pose, and source-only GT in memory."""

    def __init__(self, payloads: list[dict[str, torch.Tensor]]) -> None:
        keys = (
            "features", "coarse", "target", "frame_mask", "valid",
            "action_probability", "risk_probability", "risk",
        )
        self.data = {
            key: torch.cat([payload[key] for payload in payloads]) for key in keys
        }

    def __len__(self) -> int:
        return len(self.data["risk"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


def candidate_bank(
    candidate_rows: np.ndarray,
    index: pd.DataFrame,
    pose_array: np.ndarray,
    valid_array: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Build a GVHMR bank and normalized motion signatures from allowed rows."""
    pose = torch.from_numpy(np.asarray(pose_array[candidate_rows]).copy())
    valid = torch.from_numpy(np.asarray(valid_array[candidate_rows]).copy()).bool()
    valid &= torch.isfinite(pose).all(-1).all(-1)
    pose = fill_pose_gaps(pose, valid)
    descriptor = pose_motion_descriptor(pose, valid)
    signature = temporal_motion_signature(descriptor, valid)
    center = signature.mean(0)
    scale = signature.std(0).clamp_min(0.05)
    normalized = (signature - center) / scale
    labels = torch.tensor(index.class_id.iloc[candidate_rows].to_numpy()).long()
    return {
        "pose": pose,
        "valid": valid,
        "descriptor": descriptor,
        "center": center,
        "scale": scale,
        "normalized": normalized,
        "labels": labels,
        "support": class_means(normalized, labels),
    }


def make_reconstruction_payload(
    site: str,
    model: torch.nn.Module,
    embedded: dict[str, dict[str, torch.Tensor]],
    source_sites: list[str],
    bank: dict[str, torch.Tensor],
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    row_sites: np.ndarray,
    pose_array: np.ndarray,
    valid_array: np.ndarray,
    action_config: list[float],
    risk_config: list[float],
    device: str,
    support_seed: int,
    absence_trials: int,
    regularization: float,
    mixture: float,
    gate_mode: str,
) -> dict[str, torch.Tensor]:
    """Create one site without using query labels or query GT for inference."""
    library = [
        {"classes": class_prototypes(embedded[name]),
         "anchors": embedded[name]["anchors"]}
        for name in source_sites
    ]
    action = cal17_action(embedded[site], library, action_config)
    risk = cal17_risk(model, embedded[site], action, risk_config)
    target_support, keep = target_support_motion(
        model, embedded[site], site, store, index, selected_rows, row_sites,
        device, support_seed, absence_trials,
    )
    signature = calibrated_query_signature(
        embedded[site], target_support, bank["support"], bank["center"],
        bank["scale"], regularization, mixture, danger_gate(risk, gate_mode),
    )
    retrieval = retrieval_payload(
        embedded[site], keep, action, signature, bank["pose"], bank["valid"],
        bank["descriptor"], bank["normalized"], bank["labels"], pose_array,
        valid_array,
    )
    coarse, target, valid, risks = ensemble_prediction([retrieval])
    frame_mask = embedded[site]["frame_mask"][keep].bool()
    return {
        "features": embedded[site]["features"][keep].float(),
        "coarse": coarse.float(),
        "target": target.float(),
        "frame_mask": frame_mask,
        "valid": valid.bool() & frame_mask,
        "action_probability": action[keep].softmax(-1).float(),
        "risk_probability": risk[keep].softmax(-1).float(),
        "risk": risks.long(),
    }


@torch.no_grad()
def evaluate(
    model: MotionResidualDecoder,
    dataset: MotionDataset,
    device: str,
    strength: float,
    batch_size: int,
) -> dict[str, float]:
    """Measure root-relative reconstruction metrics without changing weights."""
    model.eval()
    predictions, targets, valids, risks = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(
            batch["features"].to(device), batch["coarse"].to(device),
            batch["frame_mask"].to(device),
            batch["action_probability"].to(device),
            batch["risk_probability"].to(device), strength,
        )
        predictions.append(output["pose_rel"].cpu())
        targets.append(batch["target"])
        valids.append(batch["valid"])
        risks.append(batch["risk"])
    return retrieval_metrics(
        torch.cat(predictions), torch.cat(targets), torch.cat(valids),
        torch.cat(risks),
    )


def train_decoder(
    training: MotionDataset,
    validation: MotionDataset,
    feature_dim: int,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[MotionResidualDecoder, dict]:
    """Select decoder epoch and inference strength using inner validation only."""
    seed_everything(seed)
    model = MotionResidualDecoder(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    weights = torch.where(
        training.data["risk"] == 2,
        torch.tensor(2.5),
        torch.tensor(1.0),
    )
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler)
    best_state = copy.deepcopy(model.state_dict())
    best_score = pose_score(evaluate(model, validation, device, 0.0, batch_size))
    best_epoch, stale, history = 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["features"].to(device), batch["coarse"].to(device),
                batch["frame_mask"].to(device),
                batch["action_probability"].to(device),
                batch["risk_probability"].to(device), 1.0,
            )
            loss, _ = motion_residual_loss(
                output, batch["target"].to(device), batch["valid"].to(device),
                batch["risk"].to(device), DISTAL_JOINTS,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation, device, 1.0, batch_size)
        score = pose_score(metrics)
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)),
            "validation": metrics, "score": score,
        })
        print(f"  epoch={epoch:02d} loss={np.mean(losses):.4f} "
              f"inner={metrics['pose_cm']:.2f}/{metrics['danger_pose_cm']:.2f}",
              flush=True)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    strengths = []
    for strength in (0.0, 0.25, 0.50, 0.75, 1.0):
        metrics = evaluate(model, validation, device, strength, batch_size)
        strengths.append({
            "strength": strength, "metrics": metrics, "score": pose_score(metrics)
        })
    selected = min(strengths, key=lambda item: item["score"])
    return model, {
        "best_epoch": best_epoch,
        "selected_strength": selected,
        "strength_candidates": strengths,
        "history": history,
    }


def aggregate(folds: dict, key: str) -> dict[str, float]:
    """Aggregate subject folds in proportion to their number of environments."""
    subjects = ("ajh", "mhw", "lmh")
    weights = np.asarray((3, 3, 1), dtype=np.float64)
    names = (
        "pose_cm", "distal_cm", "pa_pose_cm", "danger_pose_cm",
        "danger_distal_cm",
    )
    return {
        name: float(np.average([folds[item][key][name] for item in subjects],
                               weights=weights))
        for name in names
    }


def main() -> None:
    """Run leakage-controlled nested source LOSO and save fold decoders."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--training-seed", type=int, default=24081)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed target subject cannot enter decoder development")
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
    pose_config = json.loads(options.pose_config.read_text(encoding="utf-8"))
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
        configs = calibration["folds"][held_out]
        selected_pose = pose_config["folds"][held_out]["selected"]
        common = {
            "model": encoder, "embedded": embedded, "store": store,
            "index": index, "selected_rows": selected_rows,
            "row_sites": row_sites, "pose_array": pose_array,
            "valid_array": valid_array,
            "action_config": configs["action_config"],
            "risk_config": configs["risk_config"], "device": device,
            "support_seed": options.support_seed,
            "absence_trials": options.absence_trials,
            "regularization": selected_pose["regularization"],
            "mixture": selected_pose["mixture"], "gate_mode": selected_pose["gate"],
        }
        training_payloads = []
        for site in train_sites:
            source_sites = [name for name in train_sites if name != site]
            rows = selected_rows[np.isin(row_sites, source_sites)]
            bank = candidate_bank(rows, index, pose_array, valid_array)
            training_payloads.append(make_reconstruction_payload(
                site=site, source_sites=source_sites, bank=bank, **common
            ))
        train_bank = candidate_bank(
            selected_rows[np.isin(row_sites, train_sites)], index,
            pose_array, valid_array,
        )
        validation_payloads = [make_reconstruction_payload(
            site=site, source_sites=train_sites, bank=train_bank, **common
        ) for site in inner_sites]
        outer_payloads = [make_reconstruction_payload(
            site=site, source_sites=train_sites, bank=train_bank, **common
        ) for site in outer_sites]
        training = MotionDataset(training_payloads)
        validation = MotionDataset(validation_payloads)
        outer = MotionDataset(outer_payloads)
        baseline = evaluate(
            MotionResidualDecoder(training.data["features"].shape[-1]).to(device),
            outer, device, 0.0, options.batch_size,
        )
        decoder, selection = train_decoder(
            training, validation, training.data["features"].shape[-1], device,
            options.training_seed + fold_number, options.epochs,
            options.batch_size,
        )
        strength = selection["selected_strength"]["strength"]
        outer_metrics = evaluate(
            decoder, outer, device, strength, options.batch_size,
        )
        torch.save({
            "model_config": decoder.config(), "model": decoder.state_dict(),
            "strength": strength, "held_out_subject": held_out,
            "outer_holdout_used_for_selection": False,
        }, options.output_dir / f"residual_{held_out}.pt")
        folds[held_out] = {
            "train_sites": train_sites, "inner_validation_sites": inner_sites,
            "outer_test_sites": outer_sites, "training_trials": len(training),
            "validation_trials": len(validation), "outer_trials": len(outer),
            "selection": selection, "baseline": baseline, "outer": outer_metrics,
            "outer_used_for_selection": False,
        }
        print(f"  outer pose={baseline['pose_cm']:.2f}->{outer_metrics['pose_cm']:.2f} "
              f"danger={baseline['danger_pose_cm']:.2f}->{outer_metrics['danger_pose_cm']:.2f}",
              flush=True)
        del encoder, decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "run": "NOTIFI-AI-V2-CSI-CONTINUOUS-MOTION-RESIDUAL",
        "protocol": "source nested-LOSO; leave-site-out training retrieval",
        "folds": folds,
        "aggregate": {"baseline": aggregate(folds, "baseline"),
                      "outer": aggregate(folds, "outer")},
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_target_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "source_pose_gt_training_only": True,
    }
    (options.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
