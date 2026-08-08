"""CAL60 encoder의 calibration 및 query 추론 지연시간을 재현 가능하게 측정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402


def synchronize(device: torch.device) -> None:
    """CUDA 비동기 실행을 끝내 wall-clock 측정 경계를 맞춘다."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], probability: float) -> float:
    """외부 통계 패키지 없이 선형 보간 분위수를 계산한다."""
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure(callable_, device: torch.device, warmup: int, repeats: int) -> dict:
    """warm-up 뒤 반복 실행의 median, p90, 평균 지연시간을 밀리초로 반환한다."""
    for _ in range(warmup):
        callable_()
    synchronize(device)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        synchronize(device)
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": statistics.median(durations),
        "p90_ms": percentile(durations, 0.90),
        "mean_ms": statistics.fmean(durations),
        "repeats": repeats,
    }


def main() -> None:
    """실제 입력 크기로 support calibration과 query batch별 encoder 시간을 측정한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--absence-trials", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    options = parser.parse_args()
    if options.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(options.device)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    model = build_calibration_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prompt_classes = tuple(int(value) for value in model.prompt_classes)
    labels = torch.tensor(prompt_classes, device=device).repeat_interleave(2)
    generator = torch.Generator(device=device).manual_seed(17017)

    def random_csi(count: int) -> torch.Tensor:
        """학습 cache와 동일한 [B,T,L,S,I/Q] 형태의 고정 난수 입력을 만든다."""
        return torch.randn(
            count, C.CACHE_FRAMES, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2,
            generator=generator, device=device,
        )

    support = random_csi(len(labels))
    support_mask = torch.ones(
        len(labels), C.CACHE_FRAMES, C.N_LINKS, dtype=torch.bool, device=device,
    )
    absence = random_csi(options.absence_trials)
    absence_mask = torch.ones(
        options.absence_trials, C.CACHE_FRAMES, C.N_LINKS,
        dtype=torch.bool, device=device,
    )

    @torch.inference_mode()
    def run_query(batch: int) -> dict[str, torch.Tensor]:
        """고정 support·absence와 새 query batch로 encoder 및 두 분류 head를 실행한다."""
        query = random_csi(batch)
        query_mask = torch.ones(
            batch, C.CACHE_FRAMES, C.N_LINKS, dtype=torch.bool, device=device,
        )
        return model(
            query, query_mask, support, support_mask, labels,
            absence, absence_mask,
        )

    @torch.inference_mode()
    def run_support_pass() -> dict[str, torch.Tensor]:
        """현장 calibration 때 필요한 16개 support embedding 계산을 실행한다."""
        return model(
            support, support_mask, support, support_mask, labels,
            absence, absence_mask,
        )

    result = {
        "checkpoint": str(options.checkpoint.resolve()),
        "architecture": checkpoint["model_config"].get("architecture"),
        "device": str(device),
        "torch_version": torch.__version__,
        "input": {
            "frames": C.CACHE_FRAMES,
            "links": C.N_LINKS,
            "live_subcarriers": C.N_LIVE_SUBCARRIERS,
            "support_trials": len(labels),
            "absence_trials": options.absence_trials,
        },
        "support_calibration_encoder": measure(
            run_support_pass, device, options.warmup, options.repeats,
        ),
        "query_batch_1_encoder_heads": measure(
            lambda: run_query(1), device, options.warmup, options.repeats,
        ),
        "query_batch_8_encoder_heads": measure(
            lambda: run_query(8), device, options.warmup, options.repeats,
        ),
        "scope": "CAL60 encoder and classification heads; excludes CSV parsing, CAL17 and CAL23 retrieval",
        "sealed_yja_used": False,
    }
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
