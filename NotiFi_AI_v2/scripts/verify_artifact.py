"""배포용 단일 PT를 로드하고 calibration부터 3D 복원까지 연기 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_pose import contract as C
from notifi_pose.deployment import CAL44Deployment


def random_batch(count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """실제 입력 계약과 같은 모양의 유한 CSI와 유효 mask를 만든다."""
    csi = torch.randn(
        count,
        C.CACHE_FRAMES,
        C.N_LINKS,
        C.N_LIVE_SUBCARRIERS,
        2,
        generator=generator,
    )
    mask = torch.ones(count, C.CACHE_FRAMES, C.N_LINKS, dtype=torch.bool)
    return csi, mask


def main() -> None:
    """가짜 CSI로 공개 API의 입력 계약, 출력 shape, 유한 값을 확인한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()

    runtime = CAL44Deployment.load(str(options.artifact), device=options.device)
    generator = torch.Generator().manual_seed(20260814)
    support_labels = torch.tensor([
        class_id
        for class_id in C.CALIBRATION_PROMPT_CLASSES
        for _ in range(2)
    ])
    danger_labels = torch.tensor(C.DANGER_CALIBRATION_CLASSES)
    warning_labels = torch.tensor([9, 10, 11])
    support_csi, support_mask = random_batch(len(support_labels), generator)
    absence_csi, absence_mask = random_batch(12, generator)
    danger_csi, danger_mask = random_batch(len(danger_labels), generator)
    warning_csi, warning_mask = random_batch(len(warning_labels), generator)
    query_csi, query_mask = random_batch(2, generator)

    calibration = runtime.calibrate(
        support_csi,
        support_mask,
        support_labels,
        absence_csi,
        absence_mask,
        danger_csi,
        danger_mask,
        danger_labels,
        warning_csi,
        warning_mask,
        warning_labels,
    )
    output = runtime.predict(
        query_csi,
        query_mask,
        calibration,
        simulate_pose=True,
    )
    expected = {
        "action_probability": (2, C.N_CLASSES),
        "risk_probability": (2, C.N_RISK),
        "pose_rel": (2, C.CACHE_FRAMES, C.N_JOINTS, 3),
    }
    for key, shape in expected.items():
        value = output[key]
        if tuple(value.shape) != shape:
            raise RuntimeError(f"{key}: expected {shape}, got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{key}: non-finite output")
    if calibration.motion_mapping is None:
        raise RuntimeError("motion calibration mapping was not produced")
    print(json.dumps({
        "artifact": str(options.artifact),
        "device": options.device,
        "action_shape": list(output["action_probability"].shape),
        "risk_shape": list(output["risk_probability"].shape),
        "pose_shape": list(output["pose_rel"].shape),
        "motion_mapping_shape": list(calibration.motion_mapping.shape),
        "status": "ok",
    }, indent=2))


if __name__ == "__main__":
    main()
