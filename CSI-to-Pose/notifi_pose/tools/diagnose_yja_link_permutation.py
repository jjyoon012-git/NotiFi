"""Audit whether target TX ordering disagrees with the fixed link geometry."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import configure_work_root, split_support_query
from .train_kinetic_pose import CoarsePoseStore


def main() -> None:
    work_root = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=work_root)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--support-per-class", type=int, default=2)
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=work_root / "runs/cal2_kp10_seed223_danger_gate"
        / "yja_e02_v13s_coarse.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=work_root / "runs/cal3_kp10_seed239"
        / "yja_link_permutation_audit.json",
    )
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline="sub", seed=args.seed
    )["test"]
    support, _ = split_support_query(
        sealed.index, ("yja_E02",),
        args.support_per_class, args.seed + 15,
    )
    positions = support["yja_E02"]
    samples = [sealed[int(position)] for position in positions]
    csi = torch.stack([sample["csi"] for sample in samples]).to(device)
    mask = torch.stack([sample["link_mask"] for sample in samples]).to(device)
    rows = torch.stack([sample["row"] for sample in samples]).long()
    labels = torch.stack([sample["class_id"] for sample in samples]).long().to(device)
    cached = torch.load(
        args.coarse_cache, map_location="cpu", weights_only=False
    )
    coarse = CoarsePoseStore(cached["rows"], cached["pose"])
    coarse_pose = coarse.lookup(rows, device)
    kp4, _ = _load_model(
        args.work_root / "runs/kp4_dcc_staged_seed17/deployment_model.pt",
        device,
    )
    classifier_checkpoint = torch.load(
        args.work_root / "runs/kp10_action_classifier_seed181/best_model.pt",
        map_location="cpu", weights_only=False,
    )
    classifier = TemporalMotionSelector(
        **classifier_checkpoint["model_config"]
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    selector_checkpoint = torch.load(
        args.work_root / "runs/kp5_mpr_selector_seed17/best_model.pt",
        map_location="cpu", weights_only=False,
    )
    selector = TemporalMotionSelector(
        **selector_checkpoint["model_config"]
    ).to(device)
    selector.load_state_dict(selector_checkpoint["model"])
    for model in (kp4, classifier, selector):
        model.eval()

    results = []
    with torch.no_grad():
        for permutation in itertools.permutations(range(C.N_LINKS)):
            order = torch.tensor(permutation, device=device)
            current_csi = csi.index_select(2, order)
            current_mask = mask.index_select(2, order)
            output = kp4(current_csi, current_mask, coarse_pose)
            valid = current_mask.any(-1)
            classified = classifier(output["conditioned_features"], valid)
            selected = selector(output["conditioned_features"], valid)
            logits = (
                1.50 * output["action_logits"]
                + 0.75 * classified["action_logits"]
            )
            results.append({
                "permutation": [int(value) for value in permutation],
                "physical_mapping": {
                    "model_TX1_south_reads": f"input_TX{permutation[0] + 1}",
                    "model_TX2_west_reads": f"input_TX{permutation[1] + 1}",
                    "model_TX3_east_reads": f"input_TX{permutation[2] + 1}",
                },
                "fused_accuracy": float((logits.argmax(-1) == labels).float().mean()),
                "classifier_accuracy": float((
                    classified["action_logits"].argmax(-1) == labels
                ).float().mean()),
                "selector_accuracy": float((
                    selected["action_logits"].argmax(-1) == labels
                ).float().mean()),
                "fused_cross_entropy": float(F.cross_entropy(logits, labels)),
                "classifier_cross_entropy": float(F.cross_entropy(
                    classified["action_logits"], labels
                )),
            })
    results.sort(key=lambda value: (
        -value["fused_accuracy"], value["fused_cross_entropy"]
    ))
    report = {
        "target": "yja/E02 calibration support",
        "support_trials": len(positions),
        "fixed_geometry": {
            "RX": "north", "TX1": "south",
            "TX2": "west", "TX3": "east",
        },
        "best": results[0],
        "identity": next(
            item for item in results if item["permutation"] == [0, 1, 2]
        ),
        "all": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
