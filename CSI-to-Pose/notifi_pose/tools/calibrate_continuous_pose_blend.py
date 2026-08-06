"""Select a validation-only V13S/KP2-C pose blend and evaluate test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..continuous_pose import CSILatentPoseRegressor
from ..dataio.dataset import build_datasets
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .train_continuous_pose import load_teacher
from .train_kinetic_pose import CoarsePoseStore, _aggregate_rows, _pose_rows, pose_selection_score


def load_model(path: Path, device: str) -> tuple[CSILatentPoseRegressor, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    p2_path = C.PROJECT_ROOT / checkpoint["source"]["p2_checkpoint"]
    motion_path = C.PROJECT_ROOT / checkpoint["source"]["motion_checkpoint"]
    p2_checkpoint = torch.load(p2_path, map_location=device, weights_only=False)
    base = make_model(p2_checkpoint, device)
    teacher, motion_architecture = load_teacher(motion_path, device)
    architecture = checkpoint["architecture"]
    model = CSILatentPoseRegressor(
        base, teacher.decoder, checkpoint["latent_mean"],
        checkpoint["latent_std"], checkpoint["bone_lengths"],
        hidden=int(architecture["hidden"]),
        code_dim=int(motion_architecture["code_dim"]),
        temporal_layers=int(architecture.get("temporal_layers", 2)),
        heads=int(architecture.get("heads", 4)),
        dropout=float(architecture.get("dropout", 0.08)),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(model, dataset, coarse_store: CoarsePoseStore,
             strengths: list[float], batch_size: int, device: str) -> dict:
    rows = {strength: [] for strength in strengths}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )["pose_rel"].cpu()
        coarse = coarse_store.lookup(batch["row"], "cpu").float()
        for strength in strengths:
            pose = coarse + strength * (output - coarse)
            pose -= pose[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]
            rows[strength].extend(_pose_rows(pose, batch))
    return {strength: _aggregate_rows(value) for strength, value in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2c_continuous_csi_pose" / "best_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_model(args.checkpoint, device)
    cached = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    if cached.get("protocol") != args.exp:
        raise RuntimeError("coarse cache protocol mismatch")
    coarse_store = CoarsePoseStore(cached["rows"], cached["pose"])
    datasets = build_datasets(exp=args.exp, baseline="sub")
    validation = pose_only(datasets["val"])
    test = pose_only(datasets["test"])
    strengths = [float(value) for value in args.strengths]
    validation_candidates = evaluate(
        model, validation, coarse_store, strengths, args.batch_size, device
    )
    scores = {
        strength: pose_selection_score(metrics)
        for strength, metrics in validation_candidates.items()
    }
    selected = min(scores, key=scores.get)
    test_candidates = evaluate(
        model, test, coarse_store, [0.0, selected], args.batch_size, device
    )
    result = {
        "run": "KP2-C-EXP01-validation-blend",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "note": "Test had already been opened by the preceding standalone KP2-C experiment.",
        "checkpoint": report_path(args.checkpoint),
        "coarse_cache": report_path(args.coarse_cache),
        "selection": {
            "strength": selected, "score": scores[selected],
        },
        "validation_candidates": {
            str(strength): {
                "score": scores[strength], "metrics": metrics,
            }
            for strength, metrics in validation_candidates.items()
        },
        "test": {
            "v13s_strength_0": test_candidates[0.0],
            "selected_blend": test_candidates[selected],
        },
        "source_selection": checkpoint["selection"],
    }
    output = args.output or args.checkpoint.parent / "validation_blend.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
