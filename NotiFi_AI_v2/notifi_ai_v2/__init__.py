"""NotiFi AI v2 research package."""

from .geometry import InstallationGeometry
from .model import MotionCalibratedEncoder, MotionEncoderConfig

__all__ = [
    "InstallationGeometry",
    "MotionCalibratedEncoder",
    "MotionEncoderConfig",
]
