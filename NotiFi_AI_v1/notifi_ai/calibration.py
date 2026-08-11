"""User and installation calibration without updating the frozen backbone."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .constants import N_LINKS, N_SUBCARRIERS, SCHEMA_VERSION
from .preprocessing import pad_or_trim


def _bounded_logit_bias(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    bound: float = 0.75,
) -> torch.Tensor:
    """Fit a strongly regularized support-only bias for a frozen classifier."""

    if len(logits) == 0:
        return torch.zeros(classes)
    logits = logits.detach().clone().float()
    labels = labels.detach().clone().long()
    bias = torch.zeros(classes, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=0.05)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        regularizer = 0.35 * bias.square().mean()
        loss = F.cross_entropy(logits.float() + bias, labels.long()) + regularizer
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            bias.clamp_(-bound, bound)
            bias.sub_(bias.mean())
    return bias.detach().clamp(-bound, bound)


@dataclass
class CalibrationProfile:
    """Calibration state created only from installation support trials."""

    device_id: str
    absence_baseline: np.ndarray
    baseline_link_valid: np.ndarray
    link_coverage: np.ndarray
    amplitude_mad: np.ndarray
    display_action_bias: np.ndarray = field(default_factory=lambda: np.zeros(17, np.float32))
    retrieval_action_bias: np.ndarray = field(default_factory=lambda: np.zeros(17, np.float32))
    display_risk_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    retrieval_risk_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    support_action_counts: np.ndarray = field(default_factory=lambda: np.zeros(17, np.int64))
    support_risk_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, np.int64))
    schema_version: int = SCHEMA_VERSION
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    mode: str = "absence_mean_and_support_logit_bias"

    @classmethod
    def fit_absence(
        cls,
        device_id: str,
        trials: Iterable[tuple[np.ndarray, np.ndarray]],
        minimum_coverage: float = 0.35,
    ) -> "CalibrationProfile":
        """Estimate per-link and per-subcarrier static CSI from empty-room trials."""

        prepared = [pad_or_trim(csi, mask) for csi, mask in trials]
        if not prepared:
            raise ValueError("at least one absence trial is required")
        baseline = np.zeros((N_LINKS, N_SUBCARRIERS, 2), np.float32)
        amplitude_mad = np.ones((N_LINKS, N_SUBCARRIERS), np.float32)
        coverage = np.zeros(N_LINKS, np.float32)
        valid_link = np.zeros(N_LINKS, bool)
        for link in range(N_LINKS):
            rows = []
            total_valid = 0
            total_frames = 0
            for csi, mask in prepared:
                link_mask = mask[:, link]
                total_valid += int(link_mask.sum())
                total_frames += len(link_mask)
                if link_mask.any():
                    rows.append(csi[link_mask, link])
            coverage[link] = total_valid / max(total_frames, 1)
            if not rows or coverage[link] < minimum_coverage:
                continue
            values = np.concatenate(rows, axis=0).astype(np.float32)
            # The frozen model was trained with empty-room mean subtraction.
            # Deployment must use the same statistic to avoid a train/serve skew.
            center = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
            baseline[link] = center
            amplitude_mad[link] = (
                np.median(np.abs(values[..., 0] - center[..., 0]), axis=0)
                .clip(1e-3)
                .astype(np.float32)
            )
            valid_link[link] = True
        if not valid_link.any():
            raise ValueError("absence calibration found no usable CSI link")
        return cls(
            device_id=device_id,
            absence_baseline=baseline,
            baseline_link_valid=valid_link,
            link_coverage=coverage,
            amplitude_mad=amplitude_mad,
        )

    def apply_csi(
        self, csi: np.ndarray, link_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subtract the installation baseline only where packets really exist."""

        values, mask = pad_or_trim(csi, link_mask)
        calibrated = values.copy()
        for link in range(N_LINKS):
            if not self.baseline_link_valid[link]:
                mask[:, link] = False
                calibrated[:, link] = 0.0
                continue
            calibrated[:, link] -= self.absence_baseline[link]
        calibrated *= mask[:, :, None, None]
        return calibrated.astype(np.float32), mask

    def fit_logit_biases(
        self,
        display_action_logits: torch.Tensor,
        retrieval_action_logits: torch.Tensor,
        display_risk_logits: torch.Tensor,
        retrieval_risk_logits: torch.Tensor,
        action_labels: torch.Tensor,
        risk_labels: torch.Tensor,
    ) -> None:
        """Fit bounded output offsets while leaving all model weights frozen."""

        if len(action_labels) == 0:
            return
        self.display_action_bias = _bounded_logit_bias(
            display_action_logits.cpu(), action_labels.cpu(), 17
        ).numpy().astype(np.float32)
        self.retrieval_action_bias = _bounded_logit_bias(
            retrieval_action_logits.cpu(), action_labels.cpu(), 17
        ).numpy().astype(np.float32)
        self.display_risk_bias = _bounded_logit_bias(
            display_risk_logits.cpu(), risk_labels.cpu(), 3
        ).numpy().astype(np.float32)
        self.retrieval_risk_bias = _bounded_logit_bias(
            retrieval_risk_logits.cpu(), risk_labels.cpu(), 3
        ).numpy().astype(np.float32)
        self.support_action_counts = np.bincount(
            action_labels.cpu().numpy(), minlength=17
        ).astype(np.int64)
        self.support_risk_counts = np.bincount(
            risk_labels.cpu().numpy(), minlength=3
        ).astype(np.int64)

    def apply_display_action(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + logits.new_tensor(self.display_action_bias)

    def apply_retrieval_action(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + logits.new_tensor(self.retrieval_action_bias)

    def apply_display_risk(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + logits.new_tensor(self.display_risk_bias)

    def apply_retrieval_risk(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + logits.new_tensor(self.retrieval_risk_bias)

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "device_id": self.device_id,
            "created_utc": self.created_utc,
            "mode": self.mode,
            "absence_baseline": torch.from_numpy(self.absence_baseline),
            "baseline_link_valid": torch.from_numpy(self.baseline_link_valid),
            "link_coverage": torch.from_numpy(self.link_coverage),
            "amplitude_mad": torch.from_numpy(self.amplitude_mad),
            "display_action_bias": torch.from_numpy(self.display_action_bias),
            "retrieval_action_bias": torch.from_numpy(self.retrieval_action_bias),
            "display_risk_bias": torch.from_numpy(self.display_risk_bias),
            "retrieval_risk_bias": torch.from_numpy(self.retrieval_risk_bias),
            "support_action_counts": torch.from_numpy(self.support_action_counts),
            "support_risk_counts": torch.from_numpy(self.support_risk_counts),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("unsupported calibration schema")
        array_keys = {
            "absence_baseline",
            "baseline_link_valid",
            "link_coverage",
            "amplitude_mad",
            "display_action_bias",
            "retrieval_action_bias",
            "display_risk_bias",
            "retrieval_risk_bias",
            "support_action_counts",
            "support_risk_counts",
        }
        values = {
            key: value.cpu().numpy() if key in array_keys else value
            for key, value in payload.items()
        }
        return cls(**values)

    def summary(self) -> dict:
        return {
            "device_id": self.device_id,
            "created_utc": self.created_utc,
            "mode": self.mode,
            "baseline_link_valid": self.baseline_link_valid.tolist(),
            "link_coverage": self.link_coverage.astype(float).tolist(),
            "support_action_counts": self.support_action_counts.tolist(),
            "support_risk_counts": self.support_risk_counts.tolist(),
        }
