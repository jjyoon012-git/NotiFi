"""calibration 계열 checkpoint의 architecture 표식을 안전하게 복원한다."""

from __future__ import annotations

from torch import nn

from .cal14 import CAL14InvariantCosine
from .cal20 import CAL20RelativeMotionDG


def build_calibration_model(config: dict) -> nn.Module:
    """checkpoint config를 복사한 뒤 명시된 모델 class만 생성한다."""
    kwargs = dict(config)
    architecture = kwargs.pop("architecture", "cal14_invariant_cosine")
    if architecture == "cal20_relative_motion_dg":
        return CAL20RelativeMotionDG(**kwargs)
    if architecture == "cal14_invariant_cosine":
        return CAL14InvariantCosine(**kwargs)
    raise ValueError(f"unsupported calibration architecture: {architecture}")
