"""Train the promoted residual decoder on every source site and export a bundle."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import embed_site  # noqa: E402
from notifi_ai_v2.motion_residual import (  # noqa: E402
    MotionResidualDecoder,
    motion_residual_loss,
)
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402
from train_motion_residual_loso import (  # noqa: E402
    DISTAL_JOINTS,
    MotionDataset,
    candidate_bank,
    make_reconstruction_payload,
    seed_everything,
)


def main() -> None:
    """Fit a fixed source-only decoder and append it to a copy of the bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--training-seed", type=int, default=26081)
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=12)
    options = parser.parse_args()
    seed_everything(options.training_seed)
    work = Path(os.environ.get("NOTIFI_WORK_ROOT", LEGACY / "work_v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torch.load(options.artifact, map_location="cpu", weights_only=False)
    if bundle.get("sealed_yja_used") is not False:
        raise RuntimeError("sealed target must be excluded from the source bundle")
    index = pd.read_csv(work / "cache" / "cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed target cannot enter deployment decoder training")
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
    embedded = {
        site: embed_site(
            encoder, store, index, selected_rows, row_sites, site, device,
            options.support_seed, options.support_seed + 1, 2, None,
            options.absence_trials,
        )
        for site in all_sites
    }
    motion_config = bundle["motion_ridge_config"]
    common = {
        "model": encoder, "embedded": embedded, "store": store,
        "index": index, "selected_rows": selected_rows,
        "row_sites": row_sites, "pose_array": pose_array,
        "valid_array": valid_array, "action_config": bundle["action_config"],
        "risk_config": bundle["risk_config"], "device": device,
        "support_seed": options.support_seed,
        "absence_trials": options.absence_trials,
        "regularization": float(motion_config["regularization"]),
        "mixture": float(motion_config["mixture"]),
        "gate_mode": motion_config["gate"],
    }
    payloads = []
    for site in all_sites:
        source_sites = [name for name in all_sites if name != site]
        bank = candidate_bank(
            selected_rows[np.isin(row_sites, source_sites)], index,
            pose_array, valid_array,
        )
        payloads.append(make_reconstruction_payload(
            site=site, source_sites=source_sites, bank=bank, **common
        ))
    training = MotionDataset(payloads)
    decoder = MotionResidualDecoder(
        feature_dim=training.data["features"].shape[-1],
        bone_length_blend=0.0,
        risk_gate_floor=0.25,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=8e-4, weight_decay=2e-4
    )
    weights = torch.where(
        training.data["risk"] == 2, torch.tensor(2.5), torch.tensor(1.0)
    )
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(training, batch_size=options.batch_size, sampler=sampler)
    history = []
    for epoch in range(1, options.epochs + 1):
        decoder.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = decoder(
                batch["features"].to(device), batch["coarse"].to(device),
                batch["frame_mask"].to(device),
                batch["action_probability"].to(device),
                batch["risk_probability"].to(device), 1.0,
            )
            loss, components = motion_residual_loss(
                output, batch["target"].to(device), batch["valid"].to(device),
                batch["risk"].to(device), DISTAL_JOINTS,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record), flush=True)
    promoted = copy.deepcopy(bundle)
    promoted["bundle_version"] = "notifi-ai-v2-motion-residual-v1"
    promoted["motion_residual"] = {
        "model_config": decoder.config(),
        "model": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "strength": float(options.strength),
        "training_protocol": "source-only leave-one-site-out retrieval",
        "training_seed": int(options.training_seed),
        "epochs": int(options.epochs),
        "outer_holdout_used_for_selection": False,
        "sealed_target_used": False,
    }
    promoted.setdefault("v2_provenance", {})["motion_residual"] = {
        "architecture": "CSI-conditioned dilated temporal bone residual",
        "selection": "source nested-LOSO five-seed",
        "source_pose_gt_training_only": True,
        "query_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(promoted, options.output)
    summary = {
        "artifact": str(options.output.resolve()),
        "training_trials": len(training),
        "source_sites": all_sites,
        "epochs": int(options.epochs),
        "strength": float(options.strength),
        "training_seed": int(options.training_seed),
        "history": history,
        "target_subject_used": False,
        "sealed_target_used": False,
    }
    options.summary.parent.mkdir(parents=True, exist_ok=True)
    options.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
