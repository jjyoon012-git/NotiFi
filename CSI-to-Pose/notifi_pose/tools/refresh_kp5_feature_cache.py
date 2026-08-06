"""Refresh frozen KP4 feature caches after their schema is extended."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from .diagnose_observability import pose_only
from .evaluate_motion_retrieval_pose import _load_model
from .train_kinetic_pose import CoarsePoseStore
from .train_motion_retrieval_selector import extract_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument(
        "--source-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp4_dcc_staged_seed17"
        / "deployment_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _load_model(args.source_checkpoint, device)
    raw = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    coarse = CoarsePoseStore(raw["rows"], raw["pose"])
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=17)
    splits = ["train", "val"] + (["test"] if args.include_test else [])
    for split in splits:
        dataset = QualityWeightedDataset(
            pose_only(datasets[split]), protocol_audit_path(args.exp)
        )
        result = extract_features(
            model, dataset, coarse, args.cache_dir / f"{split}_features.pt",
            device, args.batch_size, args.exp,
        )
        print(f"{split}: {len(result['rows'])} rows, schema extended")


if __name__ == "__main__":
    main()
