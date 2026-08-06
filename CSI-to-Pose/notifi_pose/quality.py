"""Reliability weighting for timestamped CSI-to-pose trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import contract as C


def protocol_audit_path(protocol: str) -> Path:
    """Prefer a protocol-scoped observability audit over the legacy report."""
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in protocol
    )
    scoped = C.WORK_ROOT / "reports" / f"motion_alignment_audit_{safe}.csv"
    if scoped.exists():
        return scoped
    return C.WORK_ROOT / "reports" / "motion_alignment_audit.csv"


def trial_quality_table(audit_path: Path | None = None) -> pd.DataFrame:
    """Return a trial-indexed reliability table without changing timestamps."""
    path = audit_path or C.WORK_ROOT / "reports" / "motion_alignment_audit.csv"
    audit = pd.read_csv(path).set_index("trial_id")
    rows = []
    for trial_id, item in audit.iterrows():
        speed = float(item.mean_gt_speed_mps)
        correlation = float(item.zero_lag_correlation)
        if speed < 0.08:
            observability = 1.0
        else:
            observability = float(np.clip((correlation + 0.20) / 0.70, 0.45, 1.0))
            status_factor = {
                "aligned_observable": 1.0,
                "lag_candidate": 0.85,
                "low_observability": 0.65,
            }.get(str(item.status), 0.75)
            observability *= status_factor
        rows.append({
            "trial_id": trial_id,
            "observability_weight": observability,
            "zero_lag_correlation": correlation,
            "alignment_status": str(item.status),
        })
    return pd.DataFrame(rows).set_index("trial_id")


def quality_scores(index: pd.DataFrame,
                   audit_path: Path | None = None) -> np.ndarray:
    """Combine timestamp, link, and motion evidence into [0.35, 1] scores."""
    audit = trial_quality_table(audit_path)
    scores = np.ones(len(index), dtype=np.float32)
    for position, item in enumerate(index.itertuples(index=False)):
        timestamp = 1.0 if str(item.time_method) == "timestamps" else 0.70
        links = {1: 0.70, 2: 0.85, 3: 1.0}.get(int(item.n_alive), 0.70)
        trial = audit.loc[item.trial_id] if item.trial_id in audit.index else None
        observability = (
            float(trial.observability_weight) if trial is not None else 0.75
        )
        scores[position] = np.clip(
            0.35 + 0.65 * timestamp * links * observability, 0.35, 1.0
        )
    return scores


class QualityWeightedDataset(Dataset):
    """Attach a reliability weight while preserving the wrapped sample contract."""

    def __init__(self, target: Dataset, audit_path: Path | None = None):
        self.target = target
        self.index = target.index
        self.rows = target.rows
        self.weights = quality_scores(self.index, audit_path)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict:
        sample = self.target[index]
        sample["quality_weight"] = torch.tensor(
            self.weights[index], dtype=torch.float32
        )
        return sample

    def sampler_weights(self) -> torch.Tensor:
        labels = self.index.class_id.to_numpy(dtype=np.int64)
        counts = np.bincount(labels, minlength=C.N_CLASSES)
        balance = 1.0 / np.sqrt(np.maximum(counts[labels], 1))
        weights = self.weights * balance
        weights = weights / max(float(weights.mean()), 1e-8)
        return torch.tensor(weights, dtype=torch.double)


def quality_summary(dataset: QualityWeightedDataset) -> dict:
    values = dataset.weights
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }
