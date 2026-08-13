"""Source-only M1 training and nested LOSO evaluation."""

from __future__ import annotations

import copy
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CsiPoseDataset
from .metrics import classification_metrics
from .targets import motion_targets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _symmetric_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_log = left.log_softmax(-1)
    right_log = right.log_softmax(-1)
    return 0.5 * (
        F.kl_div(left_log, right_log.exp(), reduction="batchmean")
        + F.kl_div(right_log, left_log.exp(), reduction="batchmean")
    )


def supervised_contrastive(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    """Pull same-action cross-site pairs together in the motion space."""

    embedding = F.normalize(embedding, dim=-1)
    logits = embedding @ embedding.transpose(0, 1) / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    allowed = ~identity
    logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
    denominator = torch.logsumexp(logits.masked_fill(~allowed, -1e4), dim=-1)
    log_probability = logits - denominator[:, None]
    count = positive.sum(dim=-1)
    valid = count > 0
    if not valid.any():
        return embedding.new_zeros(())
    return -(
        (log_probability * positive).sum(dim=-1)[valid]
        / count[valid].to(log_probability.dtype)
    ).mean()


def augment_amp_phase(
    csi: torch.Tensor,
    link_mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a label-preserving RF-style view of cached amp/phase CSI."""

    values = csi.clone()
    mask = link_mask.clone()
    batch, _, links, subcarriers, _ = values.shape
    gain = torch.exp(0.30 * torch.randn(
        batch, links, generator=generator, device=values.device
    ))
    values[..., 0] *= gain[:, None, :, None]
    phase_offset = torch.empty(
        batch, links, device=values.device
    ).uniform_(-torch.pi, torch.pi, generator=generator)
    values[..., 1] += phase_offset[:, None, :, None]

    frequency = torch.linspace(-1.0, 1.0, subcarriers, device=values.device)
    curvature = 0.18 * torch.randn(
        batch, links, generator=generator, device=values.device
    )
    ripple = curvature[:, :, None] * (
        frequency.square() - frequency.square().mean()
    )[None, None]
    values[..., 1] += ripple[:, None]
    spectral_tilt = 0.08 * torch.randn(
        batch, links, 2, generator=generator, device=values.device
    )
    amplitude_curve = torch.exp(
        spectral_tilt[:, :, 0, None] * frequency[None, None]
        + spectral_tilt[:, :, 1, None]
        * (frequency.square() - frequency.square().mean())[None, None]
    )
    values[..., 0] *= amplitude_curve[:, None]

    for sample in range(batch):
        if float(torch.rand((), generator=generator, device=values.device)) < 0.25:
            alive = torch.nonzero(
                mask[sample].any(dim=0), as_tuple=True
            )[0]
            if len(alive) > 1:
                choice = int(alive[torch.randint(
                    len(alive), (1,), generator=generator, device=values.device
                )])
                mask[sample, :, choice] = False
        if float(torch.rand((), generator=generator, device=values.device)) < 0.25:
            width = max(4, int(round(subcarriers * 0.10)))
            start = int(torch.randint(
                0, subcarriers - width + 1, (1,),
                generator=generator, device=values.device,
            ))
            link = int(torch.randint(
                links, (1,), generator=generator, device=values.device
            ))
            values[sample, :, link, start:start + width] = 0.0
    values *= mask[..., None, None].to(values.dtype)
    return values, mask


def training_loss(
    clean: dict[str, torch.Tensor],
    augmented: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    risk_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    action = F.cross_entropy(
        clean["action_logits"], batch["action_id"], label_smoothing=0.03
    )
    risk = F.cross_entropy(
        clean["risk_logits"], batch["risk_id"],
        weight=risk_weight, label_smoothing=0.02,
    )
    target = motion_targets(batch["pose_rel"], batch["root"], batch["valid"])
    frame_weight = batch["valid"].to(target.dtype)
    trial_weight = torch.where(
        batch["risk_id"] == 2,
        torch.tensor(1.6, device=target.device),
        torch.tensor(1.0, device=target.device),
    )
    motion_error = F.smooth_l1_loss(
        clean["motion"], target, reduction="none", beta=0.08
    ).mean(dim=-1)
    motion = (
        motion_error * frame_weight * trial_weight[:, None]
    ).sum() / (frame_weight * trial_weight[:, None]).sum().clamp_min(1.0)
    consistency = (
        _symmetric_kl(clean["action_logits"], augmented["action_logits"])
        + 0.7 * _symmetric_kl(clean["risk_logits"], augmented["risk_logits"])
        + 0.2 * (
            1.0 - F.cosine_similarity(
                clean["embedding"], augmented["embedding"], dim=-1
            ).mean()
        )
    )
    contrastive = supervised_contrastive(
        clean["embedding"], batch["action_id"]
    )
    total = action + 0.65 * risk + 0.55 * motion + 0.18 * consistency + 0.12 * contrastive
    return total, {
        "loss": float(total.detach()),
        "action_loss": float(action.detach()),
        "risk_loss": float(risk.detach()),
        "motion_loss": float(motion.detach()),
        "consistency_loss": float(consistency.detach()),
        "contrastive_loss": float(contrastive.detach()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    datasets: Iterable[CsiPoseDataset],
    device: str,
    batch_size: int = 16,
) -> tuple[dict[str, dict], dict]:
    """Report site-level metrics and their macro mean."""

    model.eval()
    site_metrics: dict[str, dict] = {}
    for dataset in datasets:
        actions, risks, action_logits, risk_logits = [], [], [], []
        motion_error, motion_weight = 0.0, 0.0
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            output = model(
                batch["csi"], batch["link_mask"], representation="amp_phase"
            )
            target = motion_targets(
                batch["pose_rel"], batch["root"], batch["valid"]
            )
            selected = batch["valid"][..., None].expand_as(target)
            motion_error += float(
                (output["motion"] - target).abs()[selected].sum()
            )
            motion_weight += int(selected.sum())
            action_logits.append(output["action_logits"].cpu())
            risk_logits.append(output["risk_logits"].cpu())
            actions.append(batch["action_id"].cpu())
            risks.append(batch["risk_id"].cpu())
        values = classification_metrics(
            torch.cat(action_logits), torch.cat(risk_logits),
            torch.cat(actions), torch.cat(risks),
        )
        values["motion_mae"] = motion_error / max(motion_weight, 1)
        site_metrics[dataset.records[0].site] = values
    numeric = (
        "action_accuracy", "action_macro_f1", "risk_accuracy",
        "risk_macro_f1", "danger_recall", "danger_action_accuracy",
        "safe_to_danger_rate", "motion_mae",
    )
    macro = {
        key: float(np.mean([site[key] for site in site_metrics.values()]))
        for key in numeric
    }
    macro["sites"] = len(site_metrics)
    macro["trials"] = sum(int(site["trials"]) for site in site_metrics.values())
    return site_metrics, macro


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
