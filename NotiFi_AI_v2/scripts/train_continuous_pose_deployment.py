"""Train the selected continuous pose base and danger expert on all source sites."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


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
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402
import train_continuous_motion_loso as direct  # noqa: E402
import train_danger_expert_loso as danger_path  # noqa: E402
import train_pose_specific_encoder_loso as pose_path  # noqa: E402


def train_generator_stage(
    dataset: direct.DirectMotionDataset,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[ContinuousMotionGenerator, list[dict]]:
    """Fit the continuous generator to frozen calibrated CSI features."""
    direct.seed_everything(seed)
    lengths, directions = direct.fit_skeleton(dataset)
    model = ContinuousMotionGenerator(
        dataset.data["features"].shape[-1], lengths, directions,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=3e-4)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=direct.danger_sampler(dataset.data["risk"], seed),
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            output = model(
                batch["features"].to(device), batch["frame_mask"].to(device),
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(record)
        print(f"generator {json.dumps(record)}", flush=True)
    return model, history


def train_pose_encoder_stage(
    motion_encoder: torch.nn.Module,
    generator: ContinuousMotionGenerator,
    dataset: pose_path.PoseSpecificDataset,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[pose_path.PoseSpecificModel, list[dict]]:
    """Adapt the pose-only CSI encoder while preserving classification weights."""
    direct.seed_everything(seed)
    model = pose_path.PoseSpecificModel(motion_encoder, generator).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.motion_encoder.parameters(), "lr": 6e-5},
        {"params": model.generator.parameters(), "lr": 2e-4},
    ], weight_decay=4e-4)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=direct.danger_sampler(dataset.data["risk"], seed),
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            motion = batch["calibrated_motion"].float().to(device)
            link_mask = batch["link_mask"].to(device)
            motion, link_mask = pose_path.augment_motion(motion, link_mask)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(record)
        print(f"pose_encoder {json.dumps(record)}", flush=True)
    return model, history


def train_danger_stage(
    base_model: pose_path.PoseSpecificModel,
    dataset: pose_path.PoseSpecificDataset,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[ContinuousMotionGenerator, list[dict]]:
    """Specialize a decoder on source danger motion with the pose encoder frozen."""
    direct.seed_everything(seed)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    expert = copy.deepcopy(base_model.generator).to(device)
    for parameter in expert.parameters():
        parameter.requires_grad_(True)
    danger = danger_path.DangerDataset(dataset)
    loader = DataLoader(danger, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-4, weight_decay=5e-4)
    history = []
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
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(record)
        print(f"danger_expert {json.dumps(record)}", flush=True)
    return expert, history


def main() -> None:
    """Export one source-only artifact after protocol-fixed full-source training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--training-seed", type=int, default=101081)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--generator-epochs", type=int, default=1)
    parser.add_argument("--pose-encoder-epochs", type=int, default=1)
    parser.add_argument("--danger-epochs", type=int, default=1)
    parser.add_argument("--risk-threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=8)
    options = parser.parse_args()
    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torch.load(options.artifact, map_location="cpu", weights_only=False)
    if bundle.get("sealed_yja_used") is not False:
        raise RuntimeError("source artifact does not keep sealed target closed")
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed target cannot enter deployment pose training")
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
    encoder = build_calibration_model(bundle["model_config"]).to(device)
    encoder.load_state_dict(bundle["model"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    embedded = {
        site: embed_site(
            encoder, store, index, selected_rows, row_sites, site, device,
            options.support_seed, options.support_seed + 1, 2, None,
            options.absence_trials,
        )
        for site in all_sites
    }
    common = {
        "model": encoder, "embedded": embedded, "store": store,
        "index": index, "selected_rows": selected_rows,
        "row_sites": row_sites, "pose_array": pose_array,
        "valid_array": valid_array,
        "action_config": bundle["action_config"],
        "risk_config": bundle["risk_config"], "device": device,
        "support_seed": options.support_seed,
        "absence_trials": options.absence_trials,
    }
    payloads = []
    for site in all_sites:
        values = direct.direct_payload(
            site=site,
            source_sites=[name for name in all_sites if name != site],
            **common,
        )
        payloads.append(pose_path.add_calibrated_motion(
            values, encoder, site, store, index, selected_rows,
            row_sites, device, options.support_seed, options.absence_trials,
        ))
    direct_dataset = direct.DirectMotionDataset(payloads)
    pose_dataset = pose_path.PoseSpecificDataset(payloads)
    generator, generator_history = train_generator_stage(
        direct_dataset, device, options.training_seed,
        options.generator_epochs, options.batch_size,
    )
    base_model, pose_history = train_pose_encoder_stage(
        copy.deepcopy(encoder.motion_encoder).cpu(), generator,
        pose_dataset, device, options.training_seed + 1,
        options.pose_encoder_epochs, options.batch_size,
    )
    expert, danger_history = train_danger_stage(
        base_model, pose_dataset, device, options.training_seed + 2,
        options.danger_epochs, options.batch_size,
    )
    promoted = copy.deepcopy(bundle)
    promoted["bundle_version"] = "notifi-ai-v2-continuous-danger-expert-v1"
    promoted["continuous_pose"] = {
        "motion_encoder": {
            key: value.detach().cpu()
            for key, value in base_model.motion_encoder.state_dict().items()
        },
        "base_generator_config": base_model.generator.config(),
        "base_generator": {
            key: value.detach().cpu()
            for key, value in base_model.generator.state_dict().items()
        },
        "danger_generator_config": expert.config(),
        "danger_generator": {
            key: value.detach().cpu() for key, value in expert.state_dict().items()
        },
        "risk_threshold": float(options.risk_threshold),
        "source_nested_loso_result": "runs/danger_expert_loso_20260814/result.json",
        "motion_bank_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "source_pose_gt_training_only": True,
        "sealed_target_used": False,
    }
    promoted.setdefault("v2_provenance", {})["continuous_pose"] = {
        "architecture": "calibrated CSI pose encoder plus continuous danger expert",
        "epoch_protocol": {
            "generator": int(options.generator_epochs),
            "pose_encoder": int(options.pose_encoder_epochs),
            "danger_expert": int(options.danger_epochs),
        },
        "risk_threshold": float(options.risk_threshold),
        "outer_holdout_used_for_selection": False,
        "sealed_target_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(promoted, options.output)
    summary = {
        "artifact": str(options.output.resolve()),
        "training_trials": len(pose_dataset),
        "danger_training_trials": len(danger_path.DangerDataset(pose_dataset)),
        "source_sites": all_sites,
        "generator_history": generator_history,
        "pose_encoder_history": pose_history,
        "danger_history": danger_history,
        "risk_threshold": float(options.risk_threshold),
        "target_subject_used": False,
        "sealed_target_used": False,
    }
    options.summary.parent.mkdir(parents=True, exist_ok=True)
    options.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
