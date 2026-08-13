"""Installation geometry used instead of learnable site identifiers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .constants import N_LINKS


@dataclass(frozen=True)
class InstallationGeometry:
    """TX positions relative to the north-facing RX coordinate system."""

    tx_vectors: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.tx_vectors) != N_LINKS:
            raise ValueError(f"expected {N_LINKS} TX vectors")
        for vector in self.tx_vectors:
            if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
                raise ValueError("each TX vector must contain three finite values")
            if sum(v * v for v in vector) <= 1e-12:
                raise ValueError("TX vectors must be non-zero")

    @classmethod
    def cardinal_default(cls) -> "InstallationGeometry":
        """Return TX1 south, TX2 west, and TX3 east of the RX."""

        return cls(
            tx_vectors=(
                (0.0, -1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            )
        )

    def normalized_tensor(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return unit link vectors in stable TX1, TX2, TX3 order."""

        vectors = torch.tensor(self.tx_vectors, device=device, dtype=dtype)
        return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
