"""CAL43 fixed-25% guarded phase calibration runtime."""

from __future__ import annotations

from torch import nn

from .cal42_kp10 import Cal42GuardedCalibrator


class Cal43GuardedCalibrator(Cal42GuardedCalibrator):
    """Post-hoc 25% phase candidate; requires a new sealed audit."""

    def __init__(
        self,
        energy_calibrator: nn.Module,
        phase_calibrator: nn.Module,
        *,
        allow_experimental: bool = False,
    ):
        super().__init__(
            energy_calibrator,
            phase_calibrator,
            phase_weight=0.25,
            allow_experimental=allow_experimental,
        )
