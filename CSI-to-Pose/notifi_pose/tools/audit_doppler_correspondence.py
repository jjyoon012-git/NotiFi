"""Audit held-out CSI-to-GT trial retrieval for a KP2-A checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..doppler_pose import DopplerPoseResidual
from ..quality import QualityWeightedDataset
from .audit_kinetic_pose import SignalCounterfactualDataset
from .diagnose_observability import pose_only, report_path
from .evaluate_sealed import make_model
from .train_kinetic_pose import CoarsePoseStore


@torch.no_grad()
def collect_embeddings(model: DopplerPoseResidual, dataset,
                       store: CoarsePoseStore, device: str,
                       batch_size: int) -> dict[str, torch.Tensor]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {}
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            coarse_pose=store.lookup(batch["row"], device),
        )
        target = model.encode_target_motion(
            batch["pose_rel"].to(device), batch["valid"].to(device).bool()
        )
        values = {
            "csi": output["csi_motion_embedding"],
            "pose": target,
            "row": batch["row"],
            "class_id": batch["class_id"],
            "domain_id": batch["domain_id"],
        }
        for key, value in values.items():
            collected.setdefault(key, []).append(value.detach().cpu())
    return {key: torch.cat(values) for key, values in collected.items()}


def retrieval_metrics(values: dict[str, torch.Tensor]) -> dict[str, float]:
    similarity = values["csi"] @ values["pose"].T
    positive = values["row"][:, None] == values["row"][None, :]
    same_group = (
        (values["class_id"][:, None] == values["class_id"][None, :])
        & (values["domain_id"][:, None] == values["domain_id"][None, :])
    )
    exact = positive.float().argmax(1)
    top1 = similarity.argmax(1)
    grouped = similarity.masked_fill(~same_group, -torch.inf).argmax(1)
    order = similarity.argsort(dim=1, descending=True)
    rank = (order == exact[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    group_order = similarity.masked_fill(~same_group, -torch.inf).argsort(
        dim=1, descending=True
    )
    group_rank = (group_order == exact[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    hard_negative = same_group & ~positive
    return {
        "trials": int(len(similarity)),
        "top1_all": float(positive.gather(1, top1[:, None]).float().mean()),
        "top1_same_class_site": float(
            positive.gather(1, grouped[:, None]).float().mean()
        ),
        "median_rank_all": float(rank.float().median()),
        "median_rank_same_class_site": float(group_rank.float().median()),
        "positive_similarity": float(similarity[positive].mean()),
        "same_class_site_negative_similarity": (
            float(similarity[hard_negative].mean())
            if hard_negative.any() else math.nan
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2a_exp01_doppler_correspondence" / "best_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["source"]
    p2_checkpoint = torch.load(
        C.PROJECT_ROOT / source["p2_checkpoint"],
        map_location=device, weights_only=False,
    )
    p2_model = make_model(p2_checkpoint, device)
    architecture = checkpoint["architecture"]
    model = DopplerPoseResidual(
        None, p2_model.norm,
        hidden=int(architecture["hidden"]),
        temporal_layers=int(architecture["temporal_layers"]),
        heads=int(architecture.get("heads", 4)),
        dropout=float(architecture.get("dropout", 0.08)),
        max_delta=float(architecture["max_delta_m"]),
        condition_on_coarse=bool(architecture["condition_on_coarse"]),
        activity_floor=float(architecture["activity_floor"]),
        embedding_dim=int(architecture["embedding_dim"]),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    del p2_model

    cached = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    store = CoarsePoseStore(cached["rows"], cached["pose"])
    datasets = build_datasets(exp=args.exp, baseline="sub")
    selected = {
        split: QualityWeightedDataset(pose_only(datasets[split]))
        for split in ("val", "test")
    }
    validation = retrieval_metrics(collect_embeddings(
        model, selected["val"], store, device, args.batch_size
    ))
    test_modes = {}
    for mode in ("clean", "matched_shuffle", "temporal_reverse", "temporal_mean"):
        changed = SignalCounterfactualDataset(
            selected["test"], mode, args.seed
        )
        test_modes[mode] = retrieval_metrics(collect_embeddings(
            model, changed, store, device, args.batch_size
        ))
    clean = test_modes["clean"]
    result = {
        "run": f"{checkpoint['run']}-held-out-correspondence-audit",
        "checkpoint": report_path(args.checkpoint),
        "protocol": args.exp,
        "validation": validation,
        "test": test_modes,
        "counterfactual_delta": {
            mode: {
                key: float(metrics[key] - clean[key])
                for key in ("top1_all", "top1_same_class_site",
                            "median_rank_all", "positive_similarity")
            }
            for mode, metrics in test_modes.items() if mode != "clean"
        },
    }
    output = args.output or args.checkpoint.parent / "correspondence_audit.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
