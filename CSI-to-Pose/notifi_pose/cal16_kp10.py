"""Identity-preserving dynamic-spectrum calibration for CAL16-KP10."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .calibration_quality import SAFE_CALIBRATION_CLASSES


TARGET_CALIBRATION_SPLIT_SEED = 272


def trial_dynamic_spectrum(
    csi: torch.Tensor,
    link_mask: torch.Tensor,
    lags: tuple[int, ...] = (1, 3, 7),
) -> torch.Tensor:
    """Return log RMS CSI changes as ``[B, lag, link, subcarrier, IQ]``.

    CSI levels are intentionally absent.  This matches the KP4 kinetic
    encoder, which consumes temporal differences and a high-pass residual.
    """
    if csi.ndim == 4:
        csi = csi[None]
        link_mask = link_mask[None]
    if csi.ndim != 5 or link_mask.shape != csi.shape[:3]:
        raise ValueError("expected CSI [B,T,L,F,2] and mask [B,T,L]")
    spectra = []
    for lag in lags:
        delta = csi[:, lag:] - csi[:, :-lag]
        valid = link_mask[:, lag:] & link_mask[:, :-lag]
        weight = valid[..., None, None].to(csi.dtype)
        count = weight.sum(1).clamp_min(1.0)
        rms = torch.sqrt((delta.square() * weight).sum(1) / count + 1e-6)
        available = valid.any(1)[..., None, None]
        spectra.append(torch.where(available, rms.log(), torch.nan))
    return torch.stack(spectra, dim=1)


def _nanmedian(values: torch.Tensor, dim: int) -> torch.Tensor:
    result = torch.nanmedian(values, dim=dim).values
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def fit_site_balanced_reference(
    spectra: torch.Tensor,
    labels: torch.Tensor,
    site_ids: torch.Tensor,
    classes: tuple[int, ...] = SAFE_CALIBRATION_CLASSES,
) -> dict[str, torch.Tensor]:
    """Fit class spectra with equal weight per physical source site."""
    if len(spectra) != len(labels) or len(labels) != len(site_ids):
        raise ValueError("spectra, labels, and site_ids must align")
    unique_sites = site_ids.unique(sorted=True)
    references, availability = [], []
    for class_id in classes:
        site_values = []
        for site_id in unique_sites:
            selected = (labels == class_id) & (site_ids == site_id)
            if selected.any():
                site_values.append(_nanmedian(spectra[selected], dim=0))
        if site_values:
            references.append(_nanmedian(torch.stack(site_values), dim=0))
            availability.append(True)
        else:
            references.append(torch.zeros_like(spectra[0]))
            availability.append(False)
    return {
        "classes": torch.tensor(classes, dtype=torch.long),
        "log_spectrum": torch.stack(references),
        "available": torch.tensor(availability, dtype=torch.bool),
    }


def _smooth_frequency(log_gain: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 1:
        return log_gain
    if width % 2 != 1:
        raise ValueError("smoothing width must be odd")
    links, subcarriers, components = log_gain.shape
    values = log_gain.permute(0, 2, 1).reshape(links * components, 1, subcarriers)
    smoothed = F.avg_pool1d(
        values, width, stride=1, padding=width // 2,
        count_include_pad=False,
    )
    return smoothed.reshape(links, components, subcarriers).permute(0, 2, 1)


def fit_identity_spectrum_calibration(
    reference: dict[str, torch.Tensor],
    support_spectra: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    max_gain: float = 1.8,
    smoothing_width: int = 9,
    minimum_log_energy: float = -5.5,
) -> dict[str, torch.Tensor | float]:
    """Estimate one bounded gain curve per fixed TX/subcarrier/IQ channel."""
    if max_gain <= 1.0:
        raise ValueError("max_gain must exceed one")
    class_ratios = []
    before = []
    reference_classes = reference["classes"].tolist()
    for ref_index, class_id in enumerate(reference_classes):
        selected = support_labels == int(class_id)
        if not selected.any() or not bool(reference["available"][ref_index]):
            continue
        target = _nanmedian(support_spectra[selected], dim=0)
        source = reference["log_spectrum"][ref_index].to(target)
        reliable = (source > minimum_log_energy) & (target > minimum_log_energy)
        ratio = torch.where(reliable, source - target, torch.nan)
        class_ratios.append(ratio)
        before.append(torch.abs(ratio[torch.isfinite(ratio)]).median())
    if not class_ratios:
        raise ValueError("no calibration class overlaps the source reference")
    # First aggregate classes, then lags.  Each prompted action contributes
    # equally and no class with more repeats can dominate the calibration.
    log_gain_by_lag = torch.nanmedian(torch.stack(class_ratios), dim=0).values
    log_gain = torch.nanmedian(log_gain_by_lag, dim=0).values
    log_gain = torch.nan_to_num(log_gain, nan=0.0)
    log_gain = _smooth_frequency(log_gain, smoothing_width)
    limit = float(torch.log(torch.tensor(max_gain)))
    log_gain = log_gain.clamp(-limit, limit)
    gain = log_gain.exp()

    after_terms = []
    for ratio in class_ratios:
        residual = ratio - log_gain[None]
        finite = torch.isfinite(residual)
        if finite.any():
            after_terms.append(torch.abs(residual[finite]).median())
    before_value = float(torch.stack(before).median()) if before else float("inf")
    after_value = float(torch.stack(after_terms).median()) if after_terms else float("inf")
    boundary = (log_gain.abs() >= limit - 1e-6).float().mean()
    return {
        "gain": gain,
        "log_gain": log_gain,
        "discrepancy_before": before_value,
        "discrepancy_after": after_value,
        "relative_improvement": (
            (before_value - after_value) / max(before_value, 1e-8)
        ),
        "boundary_fraction": float(boundary),
        "overlapping_classes": float(len(class_ratios)),
    }


def apply_dynamic_spectrum_calibration(
    csi: torch.Tensor,
    link_mask: torch.Tensor,
    gain: torch.Tensor,
    strength: float,
    lowpass_window: int = 31,
) -> torch.Tensor:
    """Scale only the temporal residual, preserving levels and TX identity."""
    if lowpass_window % 2 != 1:
        raise ValueError("lowpass window must be odd")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    batch, frames, links, subcarriers, components = csi.shape
    flat = csi.permute(0, 2, 3, 4, 1).reshape(
        batch * links * subcarriers * components, 1, frames
    )
    low = F.avg_pool1d(
        flat, lowpass_window, stride=1, padding=lowpass_window // 2,
        count_include_pad=False,
    ).reshape(batch, links, subcarriers, components, frames).permute(0, 4, 1, 2, 3)
    effective = torch.lerp(torch.ones_like(gain), gain.to(csi), float(strength))
    output = low + (csi - low) * effective[None, None]
    return output * link_mask[..., None, None].to(output.dtype)


class IdentitySpectrumCalibratedKP4(nn.Module):
    """Run a frozen KP4 after fixed-order dynamic-spectrum calibration."""

    def __init__(self, base: nn.Module, gain: torch.Tensor, strength: float,
                 lowpass_window: int = 31):
        super().__init__()
        self.base = base
        self.register_buffer("gain", gain.float())
        self.strength = float(strength)
        self.lowpass_window = int(lowpass_window)
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor,
                coarse_pose: torch.Tensor) -> dict:
        calibrated = apply_dynamic_spectrum_calibration(
            csi, link_mask, self.gain, self.strength, self.lowpass_window
        )
        output = self.base(calibrated, link_mask, coarse_pose)
        return {**output, "calibrated_csi": calibrated}


@dataclass(frozen=True)
class SpectrumCalibrationDecision:
    status: str
    reason: str


def assess_spectrum_calibration(
    audit: dict[str, torch.Tensor | float],
    link_coverage: list[float],
    *,
    minimum_improvement: float = 0.05,
    maximum_boundary_fraction: float = 0.20,
    minimum_link_coverage: float = 0.90,
) -> SpectrumCalibrationDecision:
    """Reject weak or extrapolative calibration instead of guessing a pose."""
    if min(link_coverage, default=0.0) < minimum_link_coverage:
        return SpectrumCalibrationDecision("REJECT", "insufficient fixed-link coverage")
    if float(audit["boundary_fraction"]) > maximum_boundary_fraction:
        return SpectrumCalibrationDecision("REJECT", "too many gains hit safety bounds")
    if float(audit["relative_improvement"]) < minimum_improvement:
        return SpectrumCalibrationDecision("REJECT", "support spectrum did not align")
    return SpectrumCalibrationDecision("READY", "fixed-link dynamic spectrum aligned")


__all__ = [
    "IdentitySpectrumCalibratedKP4", "SpectrumCalibrationDecision",
    "apply_dynamic_spectrum_calibration", "assess_spectrum_calibration",
    "fit_identity_spectrum_calibration", "fit_site_balanced_reference",
    "trial_dynamic_spectrum", "TARGET_CALIBRATION_SPLIT_SEED",
]
