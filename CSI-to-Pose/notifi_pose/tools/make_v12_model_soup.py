"""Create a protocol-checked weight-space soup of compatible pose experts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def average_state_dicts(states: list[dict[str, torch.Tensor]],
                        weights: list[float]) -> dict[str, torch.Tensor]:
    if not states or len(states) != len(weights) or sum(weights) <= 0:
        raise ValueError("states and positive-sum weights must have equal length")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("model state dictionaries have different keys")
    normalized = [weight / sum(weights) for weight in weights]
    result = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"incompatible tensor shape for {key}")
        if tensors[0].is_floating_point():
            result[key] = sum(
                weight * tensor for weight, tensor in zip(normalized, tensors)
            )
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating state differs for {key}")
            result[key] = tensors[0].clone()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--protocol", default="single_split_lmh_e01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.weights):
        raise ValueError("checkpoint and weight counts differ")

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoints
    ]
    expected = (
        args.protocol,
        checkpoints[0].get("residual_decoder"),
        checkpoints[0].get("objective"),
    )
    for path, checkpoint in zip(args.checkpoints, checkpoints):
        actual = (
            checkpoint.get("protocol"),
            checkpoint.get("residual_decoder"),
            checkpoint.get("objective"),
        )
        if actual != expected or actual[0] != args.protocol:
            raise RuntimeError(
                f"incompatible checkpoint metadata for {path}: {actual} != {expected}"
            )

    normalized = [weight / sum(args.weights) for weight in args.weights]
    state = average_state_dicts(
        [checkpoint["model"] for checkpoint in checkpoints], args.weights
    )
    artifact = {
        "model": state,
        "epoch": 0,
        "protocol": args.protocol,
        "residual_decoder": expected[1],
        "objective": expected[2],
        "training_config": {
            "method": "model_soup",
            "sources": [str(path) for path in args.checkpoints],
            "weights": normalized,
            "selection_split": "validation",
            "test_used_for_selection": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print({
        "output": str(args.output),
        "sources": artifact["training_config"]["sources"],
        "weights": normalized,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
