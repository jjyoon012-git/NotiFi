"""Audit GT-length leakage in the KP2 continuous motion decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_tokens import KinematicMotionTokenizer, trial_bone_lengths
from .diagnose_observability import pose_only, report_path
from .train_kinetic_pose import _aggregate_rows, _pose_rows


@torch.no_grad()
def fit_train_length_priors(dataset, batch_size: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Fit robust skeleton lengths using train GT only."""
    all_lengths: list[torch.Tensor] = []
    subject_lengths: dict[int, list[torch.Tensor]] = {}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        lengths = trial_bone_lengths(batch["pose_rel"], batch["valid"].bool())
        all_lengths.append(lengths)
        for subject, value in zip(batch["subject_id"].tolist(), lengths):
            subject_lengths.setdefault(int(subject), []).append(value)
    global_prior = torch.cat(all_lengths).median(dim=0).values
    per_subject = {
        subject: torch.stack(values).median(dim=0).values
        for subject, values in subject_lengths.items()
    }
    return global_prior, per_subject


@torch.no_grad()
def evaluate(model, dataset, global_prior: torch.Tensor,
             subject_priors: dict[int, torch.Tensor], batch_size: int,
             device: str) -> dict:
    model.eval()
    rows = {"oracle_trial": [], "train_global": [], "train_subject": []}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        pose = batch["pose_rel"].to(device)
        valid = batch["valid"].to(device).bool()
        latent = model.encode(pose, valid)["quantized"]
        oracle = trial_bone_lengths(pose, valid)
        global_lengths = global_prior.to(device)[None].expand(len(pose), -1)
        subject_lengths = torch.stack([
            subject_priors.get(int(subject), global_prior)
            for subject in batch["subject_id"].tolist()
        ]).to(device)
        for name, lengths in (
            ("oracle_trial", oracle),
            ("train_global", global_lengths),
            ("train_subject", subject_lengths),
        ):
            prediction = model.decode(
                latent, lengths, pose.shape[1], valid
            )["pose_rel"].cpu()
            rows[name].extend(_pose_rows(prediction, batch))
    return {name: _aggregate_rows(value) for name, value in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2b_continuous_motion_autoencoder" / "best_model.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if architecture.get("kind") != "continuous":
        raise ValueError("length-prior audit requires a continuous checkpoint")
    model = KinematicMotionTokenizer(
        hidden=int(architecture["hidden"]),
        code_dim=int(architecture["code_dim"]),
        codes=int(architecture["codes"]),
        dropout=float(architecture["dropout"]),
        commitment=float(architecture["commitment"]),
        downsample=int(architecture["downsample"]),
        quantizer_levels=int(architecture["quantizer_levels"]),
        continuous=True,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    datasets = build_datasets(exp=args.exp, baseline="none")
    train = pose_only(datasets["train"])
    global_prior, subject_priors = fit_train_length_priors(train, args.batch_size)
    result = {
        "run": "KP2-B-CONTINUOUS-length-prior-audit",
        "checkpoint": report_path(args.checkpoint),
        "protocol": args.exp,
        "fit_split": "train_only",
        "global_prior_uses_test_gt": False,
        "subject_prior_uses_test_gt": False,
        "validation": evaluate(
            model, pose_only(datasets["val"]), global_prior,
            subject_priors, args.batch_size, device,
        ),
        "test": evaluate(
            model, pose_only(datasets["test"]), global_prior,
            subject_priors, args.batch_size, device,
        ),
    }
    output = args.output or args.checkpoint.parent / "length_prior_audit.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
