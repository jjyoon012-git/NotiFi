"""Audit frozen backbone identity across the final V12 expert checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def _base_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("base."): value.detach().cpu()
        for key, value in checkpoint["model"].items()
        if key.startswith("base.")
    }
    if not state:
        raise RuntimeError(f"checkpoint has no base state: {path}")
    return state, checkpoint


def _digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(bytes(value.numpy()))
    return digest.hexdigest()


def _relative_l2(first: dict[str, torch.Tensor],
                 second: dict[str, torch.Tensor]) -> float:
    if first.keys() != second.keys():
        return math.inf
    delta_square = 0.0
    reference_square = 0.0
    for key in first:
        left = first[key].double()
        right = second[key].double()
        delta_square += float((left - right).square().sum())
        reference_square += float(left.square().sum())
    return math.sqrt(delta_square / max(reference_square, 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    states = []
    for path in args.checkpoints:
        state, checkpoint = _base_state(path)
        states.append(state)
        records.append({
            "path": str(path),
            "objective": checkpoint.get("objective"),
            "initial_source": checkpoint.get("initial_source"),
            "base_sha256": _digest(state),
            "base_parameters": sum(value.numel() for value in state.values()),
        })
    comparisons = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            comparisons.append({
                "left": str(args.checkpoints[left]),
                "right": str(args.checkpoints[right]),
                "same_hash": (
                    records[left]["base_sha256"]
                    == records[right]["base_sha256"]
                ),
                "relative_l2": _relative_l2(states[left], states[right]),
            })
    report = {
        "run": "p2_v12_backbone_diversity_audit",
        "test_used": False,
        "experts": records,
        "pairwise": comparisons,
        "unique_backbone_hashes": len({
            record["base_sha256"] for record in records
        }),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
