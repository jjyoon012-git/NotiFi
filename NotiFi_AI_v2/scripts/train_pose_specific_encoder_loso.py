"""Fine-tune a pose-specific CSI encoder without changing classification heads."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import (  # noqa: E402
    embed_site,
    select_support_shots,
)
from notifi_ai_v2.continuous_motion import (  # noqa: E402
    ContinuousMotionGenerator,
    continuous_motion_loss,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from notifi_pose.pose_simulation import retrieval_metrics  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES, nested_site_split  # noqa: E402
import train_continuous_motion_loso as direct  # noqa: E402


class PoseSpecificModel(nn.Module):
    """Map calibrated per-link CSI motion to continuous body-22 trajectories."""

    def __init__(
        self,
        motion_encoder: nn.Module,
        generator: ContinuousMotionGenerator,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder
        self.generator = generator

    def forward(
        self,
        calibrated_motion: torch.Tensor,
        link_mask: torch.Tensor,
        action_probability: torch.Tensor,
        risk_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate pose while retaining link geometry and temporal CSI detail."""
        features, frame_mask, _, _ = self.motion_encoder(
            calibrated_motion, link_mask,
        )
        return self.generator(
            features, frame_mask, action_probability, risk_probability,
        )


class PoseSpecificDataset(Dataset):
    """Keep fold-local calibrated CSI motion and source pose supervision."""

    def __init__(self, payloads: list[dict[str, torch.Tensor]]) -> None:
        keys = (
            "calibrated_motion", "link_mask", "target", "valid",
            "action_probability", "risk_probability", "label", "risk",
        )
        self.data = {
            key: torch.cat([payload[key] for payload in payloads]) for key in keys
        }

    def __len__(self) -> int:
        return len(self.data["risk"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


@torch.no_grad()
def calibrated_motion_for_rows(
    model: nn.Module,
    site: str,
    rows: np.ndarray,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    row_sites: np.ndarray,
    device: str,
    support_seed: int,
    absence_trials: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply deployment calibration and retain its per-link temporal motion."""
    site_rows = base.site_rows(selected_rows, row_sites, site)
    support_rows = select_support_shots(
        site_rows, index, support_seed, shots_per_prompt=2,
    )
    absence_rows = base.select_absence(
        site, index, support_seed + 1, trials=absence_trials,
    )
    support_csi, support_mask = store.get(support_rows, device)
    absence_csi, absence_mask = store.get(absence_rows, device)
    support_labels = torch.tensor(
        index.class_id.iloc[support_rows].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    motions, masks = [], []
    for start in range(0, len(rows), 12):
        batch = rows[start:start + 12]
        query_csi, query_mask = store.get(batch, device)
        prepared = model.canonicalizer.prepare(
            query_csi, query_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        motions.append(prepared["query_motion"].cpu().half())
        masks.append(prepared["query_mask"].cpu().bool())
    return torch.cat(motions), torch.cat(masks)


def add_calibrated_motion(
    payload: dict[str, torch.Tensor],
    model: nn.Module,
    site: str,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    row_sites: np.ndarray,
    device: str,
    support_seed: int,
    absence_trials: int,
) -> dict[str, torch.Tensor]:
    """Attach CSI-native input that matches every retained query row exactly."""
    motion, mask = calibrated_motion_for_rows(
        model, site, payload["query_rows"].numpy(), store, index,
        selected_rows, row_sites, device, support_seed, absence_trials,
    )
    if len(motion) != len(payload["target"]):
        raise RuntimeError("calibrated motion and pose target order diverged")
    return {**payload, "calibrated_motion": motion, "link_mask": mask}


def augment_motion(
    motion: torch.Tensor,
    link_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb residual RF scale and occasionally hide one transmitter link."""
    if not torch.is_grad_enabled():
        return motion, link_mask
    batch, _, links = motion.shape[:3]
    gain = torch.exp(0.06 * torch.randn(
        batch, 1, links, 1, 1, device=motion.device,
    ))
    motion = motion * gain
    motion = motion + 0.008 * torch.randn_like(motion)
    drop = torch.rand(batch, device=motion.device) < 0.08
    if bool(drop.any()):
        chosen = torch.randint(0, links, (batch,), device=motion.device)
        keep = torch.ones(batch, links, dtype=torch.bool, device=motion.device)
        keep[torch.arange(batch, device=motion.device), chosen] = ~drop
        link_mask = link_mask & keep[:, None]
        motion = motion * keep[:, None, :, None, None].to(motion.dtype)
    return motion.clamp(-12.0, 12.0), link_mask


@torch.no_grad()
def evaluate(
    model: PoseSpecificModel,
    dataset: PoseSpecificDataset,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Measure query-only reconstruction without exposing query pose to inputs."""
    model.eval()
    predictions, targets, valids, risks = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(
            batch["calibrated_motion"].float().to(device),
            batch["link_mask"].to(device),
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


def train_model(
    motion_encoder: nn.Module,
    generator: ContinuousMotionGenerator,
    training: PoseSpecificDataset,
    validation: PoseSpecificDataset,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[PoseSpecificModel, dict]:
    """Tune only the pose path and select an epoch on inner source sites."""
    direct.seed_everything(seed)
    model = PoseSpecificModel(motion_encoder, generator).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.motion_encoder.parameters(), "lr": 6e-5},
        {"params": model.generator.parameters(), "lr": 2e-4},
    ], weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5,
    )
    sample_weight = torch.where(
        training.data["risk"] == 2, torch.tensor(3.0), torch.tensor(1.0),
    )
    sampler = WeightedRandomSampler(
        sample_weight, len(sample_weight), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler)
    best = {"score": math.inf, "epoch": 0, "state": None, "metrics": None}
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        for batch in loader:
            motion = batch["calibrated_motion"].float().to(device)
            link_mask = batch["link_mask"].to(device)
            motion, link_mask = augment_motion(motion, link_mask)
            output = model(
                motion, link_mask,
                batch["action_probability"].to(device),
                batch["risk_probability"].to(device),
            )
            loss, _ = continuous_motion_loss(
                output, batch["target"].to(device),
                batch["valid"].to(device), batch["risk"].to(device),
                direct.DISTAL_JOINTS,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            invalid = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if invalid:
                raise FloatingPointError("non-finite pose gradients: " + ", ".join(invalid))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.append(float(loss.detach()))
        scheduler.step()
        metrics = evaluate(model, validation, device, batch_size)
        score = direct.selection_score(metrics)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_loss)),
            "validation": metrics,
            "score": score,
        })
        print(
            f"  epoch={epoch:02d} loss={np.mean(train_loss):.4f} "
            f"pose={metrics['pose_cm']:.2f} pa={metrics['pa_pose_cm']:.2f} "
            f"danger={metrics['danger_pose_cm']:.2f}",
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
            if stale >= 7:
                break
    if best["state"] is None:
        raise RuntimeError("pose-specific training produced no checkpoint")
    model.load_state_dict(best["state"])
    return model, {
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "validation": best["metrics"],
        "history": history,
    }


def main() -> None:
    """Run pose-specific encoder fine-tuning under leakage-safe nested LOSO."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--retrieval-baseline", type=Path, required=True)
    parser.add_argument("--generator-init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--training-seed", type=int, default=41081)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed target cannot enter pose-specific development")
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
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
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
            values = direct.direct_payload(
                site=site, source_sites=source_sites, **common,
            )
            return add_calibrated_motion(
                values, encoder, site, store, index, selected_rows,
                row_sites, device, options.support_seed, options.absence_trials,
            )

        training = PoseSpecificDataset([
            payload(site, [name for name in train_sites if name != site])
            for site in train_sites
        ])
        validation = PoseSpecificDataset([
            payload(site, train_sites) for site in inner_sites
        ])
        outer = PoseSpecificDataset([
            payload(site, train_sites) for site in outer_sites
        ])
        initial = torch.load(
            options.generator_init_dir / f"continuous_{held_out}.pt",
            map_location="cpu", weights_only=False,
        )
        generator = ContinuousMotionGenerator(**initial["model_config"])
        generator.load_state_dict(initial["model"])
        model, selection = train_model(
            copy.deepcopy(encoder.motion_encoder).cpu(), generator,
            training, validation, device,
            options.training_seed + fold_number,
            options.epochs, options.batch_size,
        )
        outer_metrics = evaluate(model, outer, device, options.batch_size)
        retrieval = baseline["folds"][held_out]["outer"]
        frozen_direct = initial["selection"]["validation"]
        torch.save({
            "motion_encoder": model.motion_encoder.state_dict(),
            "generator_config": model.generator.config(),
            "generator": model.generator.state_dict(),
            "held_out_subject": held_out,
            "selection": selection,
            "classification_encoder_changed": False,
            "outer_holdout_used_for_selection": False,
        }, options.output_dir / f"pose_specific_{held_out}.pt")
        folds[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": inner_sites,
            "outer_test_sites": outer_sites,
            "training_trials": len(training),
            "validation_trials": len(validation),
            "outer_trials": len(outer),
            "selection": selection,
            "frozen_generator_inner_reference": frozen_direct,
            "retrieval_baseline": retrieval,
            "outer": outer_metrics,
            "outer_used_for_selection": False,
        }
        print(
            f"  outer retrieval={retrieval['pose_cm']:.2f}/"
            f"{retrieval['danger_pose_cm']:.2f} pose-specific="
            f"{outer_metrics['pose_cm']:.2f}/{outer_metrics['danger_pose_cm']:.2f}",
            flush=True,
        )
        del encoder, model, generator, training, validation, outer, embedded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "run": "NOTIFI-AI-V3-POSE-SPECIFIC-CSI-ENCODER",
        "protocol": "source nested-LOSO; calibrated CSI-native pose path",
        "folds": folds,
        "aggregate": {
            "retrieval_baseline": direct.aggregate(folds, "retrieval_baseline"),
            "pose_specific": direct.aggregate(folds, "outer"),
        },
        "classification_encoder_changed": False,
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
