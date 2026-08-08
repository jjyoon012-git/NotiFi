"""KP-v2를 target 비공개 source subject-LOSO 프로토콜로 학습한다."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal12 import cross_site_supervised_contrastive  # noqa: E402
from notifi_pose.cal13 import (  # noqa: E402
    pose_motion_descriptor,
    shift_robust_motion_loss,
)
from notifi_pose.kpv2 import (  # noqa: E402
    KPV2ActionPose,
    MotionDescriptorClassifier,
)
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from train_cal20_source_folds import (  # noqa: E402
    PoseStore,
    cal12_site_selection_score,
    nested_site_split,
    risk_consistency_loss,
)


SOURCE_SITES = {
    "ajh_E01", "ajh_E02", "ajh_E03",
    "mhw_E01", "mhw_E02", "mhw_E03", "lmh_E01",
}
ACTION_CLASSES = tuple(class_id for class_id in range(C.N_CLASSES) if class_id != 6)


def _subject(site: str) -> str:
    """site 문자열에서 사람 식별자만 분리한다."""
    return site.split("_")[0]


def _decorrelation_loss(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """행동 content와 사람·환경 style의 batch 상관을 줄인다."""
    content = (content - content.mean(0, keepdim=True)) / content.std(
        0, keepdim=True
    ).clamp_min(1e-4)
    style = (style - style.mean(0, keepdim=True)) / style.std(
        0, keepdim=True
    ).clamp_min(1e-4)
    cross = content.transpose(0, 1) @ style / max(len(content) - 1, 1)
    return cross.square().mean()


def _symmetric_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """두 분류 view가 과도하게 다른 posterior를 내지 않게 맞춘다."""
    left_log = left.log_softmax(-1)
    right_log = right.log_softmax(-1)
    return 0.5 * (
        F.kl_div(left_log, right_log.exp(), reduction="batchmean")
        + F.kl_div(right_log, left_log.exp(), reduction="batchmean")
    )


def _motion_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    risks: torch.Tensor,
) -> dict[str, float]:
    """pose descriptor 오차와 상관을 전체·danger로 요약한다."""
    frame_error = (predicted - target).abs().mean(-1)
    weight = valid.to(frame_error.dtype)
    overall = (frame_error * weight).sum() / weight.sum().clamp_min(1.0)
    danger = (risks == 2)[:, None] & valid
    danger_weight = danger.to(frame_error.dtype)
    danger_mae = (
        (frame_error * danger_weight).sum() / danger_weight.sum().clamp_min(1.0)
    )
    selected = valid[..., None].expand_as(predicted)
    if int(selected.sum()) > 1:
        pred_flat = predicted[selected]
        truth_flat = target[selected]
        pred_flat = pred_flat - pred_flat.mean()
        truth_flat = truth_flat - truth_flat.mean()
        correlation = (
            (pred_flat * truth_flat).mean()
            / (pred_flat.square().mean().sqrt()
               * truth_flat.square().mean().sqrt()).clamp_min(1e-8)
        )
    else:
        correlation = predicted.new_zeros(())
    return {
        "motion_descriptor_mae": float(overall),
        "danger_motion_descriptor_mae": float(danger_mae),
        "motion_descriptor_correlation": float(correlation),
    }


def _site_episode(
    site: str,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    device: str,
    seed: int,
) -> dict:
    """한 site의 support·absence와 행동별 독립 query pool을 준비한다."""
    rows = base.site_rows(selected_rows, sites, site)
    support = base.select_support(rows, index, seed)
    absence = base.select_absence(site, index, seed + 1)
    support_set = set(support.tolist())
    query = np.asarray(
        [row for row in rows if int(row) not in support_set], dtype=np.int64
    )
    labels = index.class_id.iloc[query].to_numpy(dtype=np.int64)
    pools = {
        class_id: query[labels == class_id]
        for class_id in ACTION_CLASSES
    }
    missing = [class_id for class_id, pool in pools.items() if not len(pool)]
    if missing:
        raise RuntimeError(f"{site} query pool misses classes {missing}")
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    return {
        "site": site,
        "support_csi": support_csi,
        "support_mask": support_mask,
        "support_labels": torch.tensor(
            index.class_id.iloc[support].to_numpy(), device=device
        ).long(),
        "absence_csi": absence_csi,
        "absence_mask": absence_mask,
        "query": query,
        "pools": pools,
    }


def _aligned_rows(
    left: dict,
    right: dict,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """서로 다른 사람의 같은 행동 label을 같은 순서로 뽑는다."""
    labels = rng.choice(
        ACTION_CLASSES, size=batch_size,
        replace=batch_size > len(ACTION_CLASSES),
    ).astype(np.int64)
    left_rows = np.asarray([
        rng.choice(left["pools"][int(class_id)]) for class_id in labels
    ], dtype=np.int64)
    right_rows = np.asarray([
        rng.choice(right["pools"][int(class_id)]) for class_id in labels
    ], dtype=np.int64)
    return left_rows, right_rows, labels


def _pretrain_pose_teacher(
    pose_store: PoseStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    train_sites: list[str],
    device: str,
    hidden: int,
    seed: int,
    epochs: int,
) -> MotionDescriptorClassifier:
    """현재 fold의 GT 움직임만으로 행동 의미 공간을 먼저 학습한다."""
    teacher = MotionDescriptorClassifier(
        hidden=hidden, bins=8, dropout=0.15, cosine_scale=10.0
    ).to(device)
    optimizer = torch.optim.AdamW(
        teacher.parameters(), lr=8e-4, weight_decay=2e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=8e-5
    )
    train_rows = selected_rows[np.isin(sites, train_sites)]
    risk_weight = torch.tensor((0.8, 1.0, 1.25), device=device)
    risk_weight /= risk_weight.mean()
    for epoch in range(epochs):
        teacher.train()
        for rows in base.balanced_batches(
            train_rows, index, batch_size=64, seed=seed + epoch * 17
        ):
            pose, valid = pose_store.get(rows, device)
            descriptor = pose_motion_descriptor(pose, valid)
            output = teacher(descriptor, valid)
            labels = torch.tensor(
                index.class_id.iloc[rows].to_numpy(),
                device=device, dtype=torch.long,
            )
            risks = torch.tensor(
                index.risk_id.iloc[rows].to_numpy(),
                device=device, dtype=torch.long,
            )
            loss = (
                F.cross_entropy(
                    output["action_logits"], labels, label_smoothing=0.03
                )
                + 0.60 * F.cross_entropy(
                    output["risk_logits"], risks, weight=risk_weight
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _cross_modal_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """같은 행동의 CSI와 GT 궤적을 모두 positive로 묶어 정렬한다."""
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    logits = student @ teacher.transpose(0, 1) / temperature
    positive = labels[:, None].eq(labels[None, :]).to(logits.dtype)
    positive /= positive.sum(-1, keepdim=True).clamp_min(1.0)
    forward = -(positive * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    backward = -(
        positive.transpose(0, 1)
        * F.log_softmax(logits.transpose(0, 1), dim=-1)
    ).sum(-1).mean()
    return 0.5 * (forward + backward)


def _augment_episode(
    episode: dict,
    query_csi: torch.Tensor,
    query_mask: torch.Tensor,
    rng: np.random.Generator,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """support와 query의 관계를 유지하며 source RF 변형을 적용한다."""
    tensors = [
        (episode["absence_csi"], episode["absence_mask"]),
        (episode["support_csi"], episode["support_mask"]),
        (query_csi, query_mask),
    ]
    if rng.random() < 0.70:
        tensors = base.augment_site(tensors, seed)
    if rng.random() < 0.25:
        tensors = base.reflect_east_west(tensors)
    if rng.random() < 0.40:
        tensors[2] = base.temporal_warp_trials(
            *tensors[2], seed=seed + 17, strength=0.20
        )
    (absence_csi, absence_mask), (support_csi, support_mask), (
        query_csi, query_mask
    ) = tensors
    return (
        query_csi, query_mask, support_csi, support_mask,
        absence_csi, absence_mask,
    )


@torch.no_grad()
def evaluate_site(
    model: KPV2ActionPose,
    store: base.RawStore,
    pose_store: PoseStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
    device: str,
    seed: int = 17017,
) -> dict:
    """고정 support만 이용해 미지 query의 분류와 motion 복원을 평가한다."""
    model.eval()
    episode = _site_episode(
        site, store, index, selected_rows, sites, device, seed
    )
    query = episode["query"]
    action_logits, risk_logits = [], []
    predicted_motion, target_motion, valid_rows = [], [], []
    for start in range(0, len(query), 24):
        rows = query[start:start + 24]
        query_csi, query_mask = store.get(rows, device)
        output = model(
            query_csi, query_mask,
            episode["support_csi"], episode["support_mask"],
            episode["support_labels"],
            episode["absence_csi"], episode["absence_mask"],
        )
        pose, valid = pose_store.get(rows, device)
        action_logits.append(output["action_logits"].cpu())
        risk_logits.append(output["risk_logits"].cpu())
        predicted_motion.append(output["pose_motion"].cpu())
        target_motion.append(pose_motion_descriptor(pose, valid).cpu())
        valid_rows.append(valid.cpu())
    labels = torch.tensor(index.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(index.risk_id.iloc[query].to_numpy()).long()
    metrics = classification_metrics(
        torch.cat(action_logits), torch.cat(risk_logits), labels, risks
    )
    metrics.update(_motion_errors(
        torch.cat(predicted_motion), torch.cat(target_motion),
        torch.cat(valid_rows), risks,
    ))
    metrics.update({
        "site": site,
        "support_trials": int(len(episode["support_labels"])),
        "absence_trials": int(len(episode["absence_csi"])),
    })
    return metrics


def _selection_score(metrics: dict) -> float:
    """행동 붕괴를 우선 차단하고 motion 복원도 함께 반영한다."""
    classification = cal12_site_selection_score(metrics)["score"]
    motion_quality = max(
        0.0, 1.0 - metrics["motion_descriptor_mae"]
    )
    correlation = max(0.0, metrics["motion_descriptor_correlation"])
    return float(classification + 0.15 * motion_quality + 0.10 * correlation)


def train_model(
    store: base.RawStore,
    pose_store: PoseStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    train_sites: list[str],
    validation_sites: list[str] | None,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int,
    hidden: int,
    width: int,
    steps_per_epoch: int | None = None,
    dropout: float = 0.20,
    domain_grl: float = 1.0,
    use_distance_features: bool = False,
    use_motion_classifier: bool = True,
    use_support_relative_energy: bool = True,
    use_learned_support_matcher: bool = False,
    use_explicit_support_energy: bool = True,
    use_pose_teacher: bool = False,
    teacher_epochs: int = 20,
) -> tuple[KPV2ActionPose, list[dict], dict | None]:
    """class-aligned cross-person episode로 KP-v2를 처음부터 학습한다."""
    model = KPV2ActionPose(
        hidden=hidden, width=width, domains=len(train_sites),
        dropout=dropout, domain_grl=domain_grl,
        progress_bins=16, frequency_bins=8,
        use_distance_features=use_distance_features,
        use_motion_classifier=use_motion_classifier,
        use_support_relative_energy=use_support_relative_energy,
        use_learned_support_matcher=use_learned_support_matcher,
        use_explicit_support_energy=use_explicit_support_energy,
    ).to(device)
    pose_teacher = (
        _pretrain_pose_teacher(
            pose_store, index, selected_rows, sites, train_sites,
            device, hidden, seed + 70_000, teacher_epochs,
        )
        if use_pose_teacher else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=3e-5
    )
    risk_weight = torch.tensor((0.8, 1.0, 1.25), device=device)
    risk_weight /= risk_weight.mean()
    history: list[dict] = []
    best_state = None
    best_record = None
    best_score = -float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 101)
        episodes = {
            site: _site_episode(
                site, store, index, selected_rows, sites, device,
                seed * 10_000 + epoch * 97 + number * 11,
            )
            for number, site in enumerate(train_sites)
        }
        by_subject: dict[str, list[str]] = {}
        for site in train_sites:
            by_subject.setdefault(_subject(site), []).append(site)
        subjects = sorted(by_subject)
        if len(subjects) < 2:
            raise RuntimeError("KP-v2 requires at least two source subjects")
        total_queries = sum(len(item["query"]) for item in episodes.values())
        steps = steps_per_epoch or max(24, total_queries // (2 * batch_size))
        components: list[dict[str, float]] = []

        for step in range(steps):
            left_subject, right_subject = rng.choice(
                subjects, size=2, replace=False
            )
            left_site = str(rng.choice(by_subject[str(left_subject)]))
            right_site = str(rng.choice(by_subject[str(right_subject)]))
            left_episode = episodes[left_site]
            right_episode = episodes[right_site]
            left_rows, right_rows, aligned_labels = _aligned_rows(
                left_episode, right_episode, batch_size, rng
            )
            outputs, all_rows, domain_ids = [], [], []
            for pair_number, (site, episode, rows) in enumerate((
                (left_site, left_episode, left_rows),
                (right_site, right_episode, right_rows),
            )):
                query_csi, query_mask = store.get(rows, device)
                tensors = _augment_episode(
                    episode, query_csi, query_mask, rng,
                    seed * 1_000_000 + epoch * 10_000 + step * 31 + pair_number,
                )
                output = model(
                    tensors[0], tensors[1], tensors[2], tensors[3],
                    episode["support_labels"], tensors[4], tensors[5],
                )
                outputs.append(output)
                all_rows.append(rows)
                domain_ids.append(torch.full(
                    (batch_size,), train_sites.index(site),
                    device=device, dtype=torch.long,
                ))
                if step == 0:
                    model.canonicalizer.update_source_scale(
                        output["amp_scale"], output["phase_scale"], momentum=0.98
                    )

            labels = torch.tensor(
                np.concatenate((aligned_labels, aligned_labels)),
                device=device, dtype=torch.long,
            )
            actual_labels = torch.tensor(
                index.class_id.iloc[np.concatenate(all_rows)].to_numpy(),
                device=device, dtype=torch.long,
            )
            if not torch.equal(labels, actual_labels):
                raise RuntimeError("class-aligned sampler changed label order")
            risks = torch.tensor(
                index.risk_id.iloc[np.concatenate(all_rows)].to_numpy(),
                device=device, dtype=torch.long,
            )
            domains = torch.cat(domain_ids)
            action = torch.cat([item["action_logits"] for item in outputs])
            risk = torch.cat([item["risk_logits"] for item in outputs])
            direct_risk = torch.cat([
                item["direct_risk_logits"] for item in outputs
            ])
            action_risk = torch.cat([
                item["action_risk_logits"] for item in outputs
            ])
            time_action = torch.cat([
                item["time_action_logits"] for item in outputs
            ])
            frequency_action = torch.cat([
                item["frequency_action_logits"] for item in outputs
            ])
            content_action = torch.cat([
                item["content_action_logits"] for item in outputs
            ])
            motion_action = torch.cat([
                item["motion_action_logits"] for item in outputs
            ])
            motion_risk = torch.cat([
                item["motion_risk_logits"] for item in outputs
            ])
            coarse = torch.cat([item["coarse_logits"] for item in outputs])
            start = torch.cat([item["start_logits"] for item in outputs])
            content = torch.cat([
                item["content_embedding"] for item in outputs
            ])
            style = torch.cat([item["style_embedding"] for item in outputs])
            domain_logits = torch.cat([
                item["domain_logits"] for item in outputs
            ])
            style_domain = torch.cat([
                item["style_domain_logits"] for item in outputs
            ])

            pose, pose_valid = pose_store.get(np.concatenate(all_rows), device)
            motion_target = pose_motion_descriptor(pose, pose_valid)
            motion_predicted = torch.cat([
                item["pose_motion"] for item in outputs
            ])
            motion_loss, motion_shift = shift_robust_motion_loss(
                motion_predicted, motion_target, pose_valid, max_shift=6
            )
            if pose_teacher is not None:
                with torch.no_grad():
                    teacher_output = pose_teacher(motion_target, pose_valid)
                student_embedding = torch.cat([
                    item["embedding"] for item in outputs
                ])
                cross_modal_loss = _cross_modal_distillation(
                    student_embedding, teacher_output["embedding"], labels
                )
                latent_distillation = (
                    1.0 - F.cosine_similarity(
                        student_embedding,
                        teacher_output["embedding"], dim=-1,
                    )
                ).mean()
                distillation_temperature = 2.0
                output_distillation = F.kl_div(
                    F.log_softmax(
                        action / distillation_temperature, dim=-1
                    ),
                    F.softmax(
                        teacher_output["action_logits"]
                        / distillation_temperature,
                        dim=-1,
                    ),
                    reduction="batchmean",
                ) * distillation_temperature ** 2
            else:
                cross_modal_loss = action.new_zeros(())
                latent_distillation = action.new_zeros(())
                output_distillation = action.new_zeros(())
            action_loss = F.cross_entropy(
                action, labels, label_smoothing=0.05
            )
            risk_loss = F.cross_entropy(risk, risks, weight=risk_weight)
            direct_loss = F.cross_entropy(
                direct_risk, risks, weight=risk_weight
            )
            view_loss = 0.5 * (
                F.cross_entropy(time_action, labels)
                + F.cross_entropy(frequency_action, labels)
            )
            grounded_classification = 0.5 * (
                F.cross_entropy(motion_action, labels)
                + F.cross_entropy(motion_risk, risks, weight=risk_weight)
            )
            content_classification = F.cross_entropy(
                content_action, labels, label_smoothing=0.05
            )
            hierarchy_loss = 0.5 * (
                F.cross_entropy(coarse, model.coarse_action_target[labels])
                + F.cross_entropy(start, model.start_posture_target[labels])
            )
            domain_loss = F.cross_entropy(domain_logits, domains)
            style_domain_loss = F.cross_entropy(style_domain, domains)
            supervised_contrastive = cross_site_supervised_contrastive(
                content, labels, domains
            )
            paired_alignment = 1.0 - F.cosine_similarity(
                content[:batch_size], content[batch_size:], dim=-1
            ).mean()
            mixing = torch.empty(
                batch_size, 1, device=device
            ).uniform_(0.25, 0.75)
            mixed_embedding = (
                mixing * outputs[0]["embedding"]
                + (1.0 - mixing) * outputs[1]["embedding"]
            )
            interpolation_loss = (
                F.cross_entropy(model.action_head(mixed_embedding), labels[:batch_size])
                + 0.50 * F.cross_entropy(
                    model.risk_head(mixed_embedding), risks[:batch_size],
                    weight=risk_weight,
                )
            )
            view_consistency = (
                _symmetric_kl(action, time_action)
                + _symmetric_kl(action, frequency_action)
                + _symmetric_kl(time_action, frequency_action)
            ) / 3.0
            decorrelation = _decorrelation_loss(content, style)
            risk_consistency = risk_consistency_loss(direct_risk, action_risk)
            loss = (
                action_loss
                + 0.75 * risk_loss
                + 0.20 * direct_loss
                + 0.25 * view_loss
                + 0.25 * grounded_classification
                + 0.15 * content_classification
                + 0.12 * hierarchy_loss
                + 0.15 * domain_loss
                + 0.06 * style_domain_loss
                + 0.18 * supervised_contrastive
                + 0.30 * paired_alignment
                + 0.25 * interpolation_loss
                + 0.08 * view_consistency
                + 0.08 * risk_consistency
                + 0.08 * decorrelation
                + 0.45 * motion_loss
                + 0.30 * cross_modal_loss
                + 0.12 * latent_distillation
                + 0.12 * output_distillation
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            components.append({
                "loss": float(loss.detach()),
                "action": float(action_loss.detach()),
                "risk": float(risk_loss.detach()),
                "view": float(view_loss.detach()),
                "grounded_classification": float(
                    grounded_classification.detach()
                ),
                "domain": float(domain_loss.detach()),
                "style_domain": float(style_domain_loss.detach()),
                "contrastive": float(supervised_contrastive.detach()),
                "paired": float(paired_alignment.detach()),
                "interpolation": float(interpolation_loss.detach()),
                "decorrelation": float(decorrelation.detach()),
                "motion": float(motion_loss.detach()),
                "motion_shift": float(motion_shift),
                "cross_modal": float(cross_modal_loss.detach()),
                "latent_distillation": float(latent_distillation.detach()),
                "output_distillation": float(output_distillation.detach()),
            })
        scheduler.step()

        record: dict = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "motion_fusion": float(
                torch.sigmoid(model.motion_fusion_logit).detach()
                if model.use_motion_classifier else 0.0
            ),
            "risk_fusion": float(torch.sigmoid(model.risk_fusion).detach()),
            "train": {
                key: float(np.mean([item[key] for item in components]))
                for key in components[0]
            },
        }
        if validation_sites:
            validation = {
                site: evaluate_site(
                    model, store, pose_store, index, selected_rows, sites,
                    site, device,
                )
                for site in validation_sites
            }
            scores = [_selection_score(value) for value in validation.values()]
            selection = 0.5 * float(np.mean(scores)) + 0.5 * float(np.min(scores))
            record.update({
                "validation": validation,
                "selection_score": selection,
                "mean_site_score": float(np.mean(scores)),
                "worst_site_score": float(np.min(scores)),
            })
            if selection > best_score + 1e-5:
                best_score = selection
                best_state = copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                })
                best_record = copy.deepcopy(record)
                stale = 0
            else:
                stale += 1
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation_sites and stale >= 8:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_record


def _aggregate_fold_metrics(fold_results: dict) -> dict[str, float]:
    """outer subject-LOSO site 지표를 macro 평균으로 합친다."""
    rows = [
        metrics
        for fold in fold_results.values()
        for metrics in fold["outer_test"].values()
    ]
    keys = (
        "action_accuracy", "action_macro_f1", "risk_accuracy",
        "risk_macro_f1", "danger_recall", "danger_action_accuracy",
        "motion_descriptor_mae", "danger_motion_descriptor_mae",
        "motion_descriptor_correlation",
    )
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    """세 source 사람 LOSO와 전체 source deployment 학습을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--domain-grl", type=float, default=1.0)
    parser.add_argument("--use-distance-features", action="store_true")
    parser.add_argument(
        "--disable-motion-classifier", action="store_true"
    )
    parser.add_argument("--pose-teacher", action="store_true")
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument(
        "--disable-support-relative-energy", action="store_true"
    )
    parser.add_argument("--learned-support-matcher", action="store_true")
    parser.add_argument(
        "--disable-explicit-support-energy", action="store_true"
    )
    parser.add_argument("--training-seed", type=int, default=22001)
    parser.add_argument(
        "--folds", nargs="+", choices=("ajh", "mhw", "lmh"),
        default=("ajh", "mhw", "lmh"),
    )
    parser.add_argument("--skip-deployment", action="store_true")
    options = parser.parse_args()
    options.run_dir.mkdir(parents=True, exist_ok=True)

    base.ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
    base.PROMPT_SHOTS = {class_id: 2 for class_id in MOTION_PROMPT_CLASSES}
    torch.manual_seed(options.training_seed)
    np.random.seed(options.training_seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected_index = index.iloc[selected_rows]
    if "yja" in set(selected_index.subject.astype(str)):
        raise RuntimeError("sealed yja entered KP-v2 source rows")
    sites = (
        selected_index.subject + "_" + selected_index.environment
    ).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected KP-v2 source sites: {all_sites}")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS)
            & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in all_sites
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    pose_store = PoseStore(WORK / "cache")

    fold_results = {}
    best_epochs = []
    for fold_number, held_out in enumerate(options.folds):
        train_sites, validation_sites, outer_sites = nested_site_split(
            all_sites, held_out
        )
        model, history, best = train_model(
            store, pose_store, index, selected_rows, sites,
            train_sites, validation_sites, options.epochs, device,
            options.training_seed + fold_number * 101,
            options.batch_size, options.hidden, options.width,
            options.steps_per_epoch,
            options.dropout, options.domain_grl,
            options.use_distance_features,
            not options.disable_motion_classifier,
            not options.disable_support_relative_energy,
            options.learned_support_matcher,
            not options.disable_explicit_support_energy,
            options.pose_teacher,
            options.teacher_epochs,
        )
        if best is None:
            raise RuntimeError(f"{held_out} fold produced no selected checkpoint")
        outer = {
            site: evaluate_site(
                model, store, pose_store, index, selected_rows, sites,
                site, device,
            )
            for site in outer_sites
        }
        best_epochs.append(int(best["epoch"]))
        fold_results[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": validation_sites,
            "outer_test_sites": outer_sites,
            "best": best,
            "outer_test": outer,
            "history": history,
        }
        torch.save({
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "held_out_subject": held_out,
            "protocol_version": "kpv2_nested_source_v1",
            "best": best,
            "outer_test": outer,
            "outer_holdout_used_for_selection": False,
            "target_subject_used": False,
            "sealed_yja_used": False,
        }, options.run_dir / f"selection_{held_out}.pt")

    result = {
        "run": "KP-v2-ACTION-CONTENT",
        "device": device,
        "training_seed": options.training_seed,
        "training_config": {
            "epochs": options.epochs,
            "batch_size": options.batch_size,
            "hidden": options.hidden,
            "width": options.width,
            "steps_per_epoch": options.steps_per_epoch,
            "dropout": options.dropout,
            "domain_grl": options.domain_grl,
            "use_distance_features": options.use_distance_features,
            "use_motion_classifier": not options.disable_motion_classifier,
            "use_support_relative_energy": (
                not options.disable_support_relative_energy
            ),
            "use_learned_support_matcher": options.learned_support_matcher,
            "use_explicit_support_energy": (
                not options.disable_explicit_support_energy
            ),
            "use_pose_teacher": options.pose_teacher,
            "teacher_epochs": options.teacher_epochs,
        },
        "fold_results": fold_results,
        "fold_best_epochs": best_epochs,
        "outer_macro": _aggregate_fold_metrics(fold_results),
        "selection_protocol": "nested_source_subject_loso_v2",
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }

    if not options.skip_deployment and set(options.folds) == {"ajh", "mhw", "lmh"}:
        locked_epochs = max(1, int(round(float(np.median(best_epochs)))))
        deployment, deployment_history, _ = train_model(
            store, pose_store, index, selected_rows, sites,
            all_sites, None, locked_epochs, device,
            options.training_seed + 1000, options.batch_size,
            options.hidden, options.width, options.steps_per_epoch,
            options.dropout, options.domain_grl,
            options.use_distance_features,
            not options.disable_motion_classifier,
            not options.disable_support_relative_energy,
            options.learned_support_matcher,
            not options.disable_explicit_support_energy,
            options.pose_teacher,
            options.teacher_epochs,
        )
        source_metrics = {
            site: evaluate_site(
                deployment, store, pose_store, index, selected_rows, sites,
                site, device,
            )
            for site in all_sites
        }
        torch.save({
            "model": deployment.state_dict(),
            "model_config": deployment.model_config(),
            "locked_epochs": locked_epochs,
            "source_sites": all_sites,
            "support_contract": {
                "absence_trials": 2,
                "prompt_classes": list(MOTION_PROMPT_CLASSES),
                "shots_per_prompt": 2,
                "target_pose_gt_used": False,
            },
            "selection_protocol": "nested_source_subject_loso_v2",
            "outer_holdout_used_for_selection": False,
            "target_subject_used": False,
            "sealed_yja_used": False,
        }, options.run_dir / "deployment_model.pt")
        result.update({
            "locked_epochs": locked_epochs,
            "source_metrics": source_metrics,
            "deployment_history": deployment_history,
        })

    (options.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "outer_macro": result["outer_macro"],
        "fold_best_epochs": best_epochs,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
