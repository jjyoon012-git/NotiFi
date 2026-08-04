"""Benchmark the validation-locked V12 inference graph without opening test."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ..trainer import set_seed
from .evaluate_v12_final import _read_locked, build_locked_model
from .train_seen_v4_trajectory import make_loaders


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--root-calibration", type=Path, required=True)
    parser.add_argument("--classification-calibration", type=Path, required=True)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--danger-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    _, loaders = make_loaders(args, device)
    model, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    model.eval()
    batch = next(iter(loaders["val"]))
    csi = batch["csi"][:1].to(device)
    link_mask = batch["link_mask"][:1].to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(csi, link_mask)
        _synchronize(device)
        latencies = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model(csi, link_mask)
            _synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "run": "p2_v12_final_inference_benchmark",
        "protocol": args.exp,
        "split_used_for_input_shape": "validation",
        "test_used": False,
        "device": device,
        "input_shape": list(csi.shape),
        "parameters": parameters,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": ordered[p95_index],
            "min": min(latencies),
            "max": max(latencies),
        },
        "configuration": configuration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
