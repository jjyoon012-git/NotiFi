"""CAL32 전체 bundle의 현장 calibration, 분류, 3D 시뮬레이션 시간을 측정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from benchmark_cal20_runtime import measure  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.deployment import CAL20Deployment  # noqa: E402


def main() -> None:
    """실제 크기의 고정 난수 CSI로 전체 배포 경로의 반복 지연시간을 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    options = parser.parse_args()
    runtime = CAL20Deployment.load(str(options.bundle))
    device = torch.device(runtime.device)
    generator = torch.Generator(device=device).manual_seed(32017)

    def random_csi(count: int) -> torch.Tensor:
        """cache와 동일한 [B,T,L,S,I/Q] 입력을 생성한다."""
        return torch.randn(
            count, C.CACHE_FRAMES, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2,
            generator=generator, device=device,
        )

    labels = torch.tensor(
        runtime.support_contract["prompt_classes"], device=device,
    ).repeat_interleave(int(runtime.support_contract["shots_per_prompt"]))
    support = random_csi(len(labels))
    support_mask = torch.ones(
        len(labels), C.CACHE_FRAMES, C.N_LINKS, dtype=torch.bool, device=device,
    )
    absence_count = int(runtime.support_contract["absence_trials"])
    absence = random_csi(absence_count)
    absence_mask = torch.ones(
        absence_count, C.CACHE_FRAMES, C.N_LINKS,
        dtype=torch.bool, device=device,
    )
    query = random_csi(1)
    query_mask = torch.ones(
        1, C.CACHE_FRAMES, C.N_LINKS, dtype=torch.bool, device=device,
    )

    def calibrate():
        """16개 support와 12개 absence를 latent anchor로 변환한다."""
        return runtime.calibrate(
            support, support_mask, labels, absence, absence_mask,
        )

    calibration = calibrate()
    result = {
        "bundle": str(options.bundle.resolve()),
        "bundle_mb": options.bundle.stat().st_size / (1024.0 ** 2),
        "device": runtime.device,
        "calibration": measure(
            calibrate, device, options.warmup, options.repeats,
        ),
        "classification_only": measure(
            lambda: runtime.predict(
                query, query_mask, calibration, simulate_pose=False,
            ),
            device, options.warmup, options.repeats,
        ),
        "classification_and_pose": measure(
            lambda: runtime.predict(
                query, query_mask, calibration, simulate_pose=True,
            ),
            device, options.warmup, options.repeats,
        ),
        "input": {
            "frames": C.CACHE_FRAMES,
            "links": C.N_LINKS,
            "live_subcarriers": C.N_LIVE_SUBCARRIERS,
            "support_trials": len(labels),
            "absence_trials": absence_count,
            "pose_candidates": len(runtime.pose_library["pose"]),
        },
        "scope": "in-memory tensors; excludes raw CSV parsing and disk I/O",
        "sealed_yja_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
