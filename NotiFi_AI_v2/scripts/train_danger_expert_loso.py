"""Specialize a continuous fall decoder and route with CSI-predicted risk."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import embed_site  # noqa: E402
from notifi_ai_v2.continuous_motion import (  # noqa: E402
    ContinuousMotionGenerator,
    continuous_motion_loss,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.pose_simulation import retrieval_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402
import train_continuous_motion_loso as direct  # noqa: E402
import train_pose_specific_encoder_loso as pose_path  # noqa: E402


class DangerDataset(Dataset):
    """Expose only true source danger trials while keeping predicted conditions."""

    def __init__(self, source: pose_path.PoseSpecificDataset) -> None:
        keep = source.data["risk"] == 2
        if not bool(keep.any()):
            raise RuntimeError("danger expert received no source danger trials")
        self.data = {key: value[keep] for key, value in source.data.items()}

    def __len__(self) -> int:
        return len(self.data["risk"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


@torch.no_grad()
def evaluate_routed(
    base_model: pose_path.PoseSpecificModel,
    danger_generator: ContinuousMotionGenerator,
    dataset: pose_path.PoseSpecificDataset,
    threshold: float,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Route queries using predicted danger probability and never true risk."""
    base_model.eval()
    danger_generator.eval()
    predictions, targets, valids, risks = [], [], [], []
    routed = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        motion = batch["calibrated_motion"].float().to(device)
        link_mask = batch["link_mask"].to(device)
        action = batch["action_probability"].to(device)
        risk_probability = batch["risk_probability"].to(device)
        features, frame_mask, _, _ = base_model.motion_encoder(motion, link_mask)
        base = base_model.generator(features, frame_mask, action, risk_probability)
        danger = danger_generator(features, frame_mask, action, risk_probability)
        gate = risk_probability[:, 2] >= float(threshold)
        routed += int(gate.sum())
        pose = torch.where(
            gate[:, None, None, None], danger["pose_rel"], base["pose_rel"],
        )
        predictions.append(pose.cpu())
        targets.append(batch["target"])
        valids.append(batch["valid"])
        risks.append(batch["risk"])
    metrics = retrieval_metrics(
        torch.cat(predictions), torch.cat(targets), torch.cat(valids),
        torch.cat(risks),
    )
    metrics["routed_trials"] = routed
    metrics["routed_fraction"] = routed / len(dataset)
    return metrics


def expert_score(metrics: dict[str, float]) -> float:
    """Prioritize danger limbs while retaining whole-dataset safety."""
    return float(
        0.12 * metrics["pose_cm"]
        + 0.08 * metrics["distal_cm"]
        + 0.10 * metrics["pa_pose_cm"]
        + 0.25 * metrics["danger_pose_cm"]
        + 0.45 * metrics["danger_distal_cm"]
    )


def train_expert(
    base_model: pose_path.PoseSpecificModel,
    training: pose_path.PoseSpecificDataset,
    validation: pose_path.PoseSpecificDataset,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[ContinuousMotionGenerator, dict]:
    """Fine-tune a fall-only decoder and select routing on inner source sites."""
    direct.seed_everything(seed)
    base_model = base_model.to(device).eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    expert = copy.deepcopy(base_model.generator).to(device)
    for parameter in expert.parameters():
        parameter.requires_grad_(True)
    danger = DangerDataset(training)
    loader = DataLoader(danger, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=8e-6,
    )
    thresholds = (0.30, 0.45, 0.60, 0.75)
    best = {
        "score": math.inf, "epoch": 0, "state": None,
        "threshold": None, "metrics": None,
    }
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        expert.train()
        losses = []
        for batch in loader:
            motion = batch["calibrated_motion"].float().to(device)
            link_mask = batch["link_mask"].to(device)
            motion, link_mask = pose_path.augment_motion(motion, link_mask)
            with torch.no_grad():
                features, frame_mask, _, _ = base_model.motion_encoder(
                    motion, link_mask,
                )
            output = expert(
                features, frame_mask,
                batch["action_probability"].to(device),
                batch["risk_probability"].to(device),
            )
            loss, _ = continuous_motion_loss(
                output, batch["target"].to(device),
                batch["valid"].to(device), batch["risk"].to(device),
                direct.DISTAL_JOINTS, danger_weight=1.0,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        candidates = []
        for threshold in thresholds:
            metrics = evaluate_routed(
                base_model, expert, validation, threshold, device, batch_size,
            )
            candidates.append({
                "threshold": threshold, "metrics": metrics,
                "score": expert_score(metrics),
            })
        candidates.sort(key=lambda row: row["score"])
        selected = candidates[0]
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "threshold_candidates": candidates,
        })
        metrics = selected["metrics"]
        print(
            f"  epoch={epoch:02d} threshold={selected['threshold']:.2f} "
            f"pose={metrics['pose_cm']:.2f} danger={metrics['danger_pose_cm']:.2f} "
            f"danger_distal={metrics['danger_distal_cm']:.2f}",
            flush=True,
        )
        if selected["score"] < best["score"] - 1e-4:
            best = {
                "score": selected["score"], "epoch": epoch,
                "state": copy.deepcopy(expert.state_dict()),
                "threshold": selected["threshold"], "metrics": metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= 7:
                break
    if best["state"] is None:
        raise RuntimeError("danger expert produced no checkpoint")
    expert.load_state_dict(best["state"])
    return expert, {
        "best_epoch": best["epoch"], "best_score": best["score"],
        "threshold": best["threshold"], "validation": best["metrics"],
        "history": history,
    }


def main() -> None:
    """Run predicted-risk danger expert development under source nested LOSO."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--retrieval-baseline", type=Path, required=True)
    parser.add_argument("--base-pose-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--training-seed", type=int, default=91081)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed target cannot enter danger-expert development")
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
            options.run_dir / f"selection_{held_out}.pt",
            map_location="cpu", weights_only=False,
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
            "model": encoder, "embedded": embedded, "store": store,
            "index": index, "selected_rows": selected_rows,
            "row_sites": row_sites, "pose_array": pose_array,
            "valid_array": valid_array,
            "action_config": config["action_config"],
            "risk_config": config["risk_config"], "device": device,
            "support_seed": options.support_seed,
            "absence_trials": options.absence_trials,
        }
        def payload(site: str, source_sites: list[str]) -> dict[str, torch.Tensor]:
            values = direct.direct_payload(site=site, source_sites=source_sites, **common)
            return pose_path.add_calibrated_motion(
                values, encoder, site, store, index, selected_rows,
                row_sites, device, options.support_seed, options.absence_trials,
            )

        training = pose_path.PoseSpecificDataset([
            payload(site, [name for name in train_sites if name != site])
            for site in train_sites
        ])
        validation = pose_path.PoseSpecificDataset([
            payload(site, train_sites) for site in inner_sites
        ])
        outer = pose_path.PoseSpecificDataset([
            payload(site, train_sites) for site in outer_sites
        ])
        base_checkpoint = torch.load(
            options.base_pose_dir / f"pose_specific_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        motion_encoder = copy.deepcopy(encoder.motion_encoder).cpu()
        motion_encoder.load_state_dict(base_checkpoint["motion_encoder"])
        generator = ContinuousMotionGenerator(**base_checkpoint["generator_config"])
        generator.load_state_dict(base_checkpoint["generator"])
        base_model = pose_path.PoseSpecificModel(motion_encoder, generator)
        expert, selection = train_expert(
            base_model, training, validation, device,
            options.training_seed + fold_number,
            options.epochs, options.batch_size,
        )
        outer_metrics = evaluate_routed(
            base_model.to(device), expert, outer,
            float(selection["threshold"]), device, options.batch_size,
        )
        retrieval = baseline["folds"][held_out]["outer"]
        torch.save({
            "base_motion_encoder": base_model.motion_encoder.state_dict(),
            "base_generator_config": base_model.generator.config(),
            "base_generator": base_model.generator.state_dict(),
            "danger_generator_config": expert.config(),
            "danger_generator": expert.state_dict(),
            "risk_threshold": selection["threshold"],
            "selection": selection,
            "outer_holdout_used_for_selection": False,
        }, options.output_dir / f"danger_expert_{held_out}.pt")
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": inner_sites,
            "outer_test_sites": outer_sites,
            "training_danger_trials": len(DangerDataset(training)),
            "validation_trials": len(validation), "outer_trials": len(outer),
            "selection": selection,
            "retrieval_baseline": retrieval,
            "outer": outer_metrics,
            "outer_used_for_selection": False,
        }
        print(
            f"  outer retrieval={retrieval['pose_cm']:.2f}/"
            f"{retrieval['danger_pose_cm']:.2f} expert="
            f"{outer_metrics['pose_cm']:.2f}/{outer_metrics['danger_pose_cm']:.2f}",
            flush=True,
        )
        del encoder, embedded, training, validation, outer, base_model, expert
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "run": "NOTIFI-AI-V3-PREDICTED-RISK-DANGER-EXPERT",
        "protocol": "source nested-LOSO; CSI-risk routed continuous fall decoder",
        "folds": folds,
        "aggregate": {
            "retrieval_baseline": direct.aggregate(folds, "retrieval_baseline"),
            "danger_expert": direct.aggregate(folds, "outer"),
        },
        "routing_source": "CSI predicted risk probability; no GT risk",
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_target_used": False,
        "motion_bank_used_for_pose": False,
        "query_labels_or_pose_gt_at_inference": False,
        "source_pose_gt_training_only": True,
    }
    (options.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
