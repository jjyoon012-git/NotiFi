"""CAL20-RELATIVE-MOTION-DG를 yja 없이 source subject-LOSO로 학습한다."""

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
from notifi_pose.cal20 import CAL20RelativeMotionDG  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.pose_semantic_teacher import (  # noqa: E402
    PoseSemanticTeacher,
    cross_modal_supervised_contrastive,
    paired_pose_distillation_loss,
)
from notifi_pose.metrics import classification_metrics  # noqa: E402


SOURCE_SITES = {
    "ajh_E01", "ajh_E02", "ajh_E03",
    "mhw_E01", "mhw_E02", "mhw_E03", "lmh_E01",
}
ACTION_CLASSES = tuple(
    class_id for class_id in range(C.N_CLASSES) if class_id != 6
)


class PoseStore:
    """source GT를 학습 보조교사로만 읽고 평가·추론 입력에서는 분리한다."""

    def __init__(self, cache: Path):
        self.pose = np.load(cache / "pose_rel.npy", mmap_mode="r")
        self.valid = np.load(cache / "valid.npy", mmap_mode="r")

    def get(
        self, rows: np.ndarray, device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """선택한 source row의 자세와 유효 mask만 장치로 복사한다."""
        pose = torch.from_numpy(np.asarray(self.pose[rows]).copy()).to(device)
        valid = torch.from_numpy(np.asarray(self.valid[rows]).copy()).to(device)
        return pose, valid


def pretrain_pose_semantic_teacher(
    pose_store: PoseStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    train_sites: list[str],
    device: str,
    hidden: int,
    seed: int,
    epochs: int,
) -> PoseSemanticTeacher:
    """Train a fold-local GT teacher and then restore the student RNG stream."""
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
    try:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        teacher = PoseSemanticTeacher(
            hidden=hidden, bins=8, dropout=0.15
        ).to(device)
        optimizer = torch.optim.AdamW(
            teacher.parameters(), lr=8e-4, weight_decay=2e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=8e-5
        )
        rows = selected_rows[np.isin(sites, train_sites)]
        risk_weight = torch.tensor((0.8, 1.0, 1.25), device=device)
        risk_weight /= risk_weight.mean()
        for epoch in range(epochs):
            teacher.train()
            for batch in base.balanced_batches(
                rows, index, batch_size=64, seed=seed + epoch * 17
            ):
                pose, valid = pose_store.get(batch, device)
                descriptor = pose_motion_descriptor(pose, valid)
                output = teacher(descriptor, valid)
                labels = torch.tensor(
                    index.class_id.iloc[batch].to_numpy(),
                    dtype=torch.long, device=device,
                )
                risks = torch.tensor(
                    index.risk_id.iloc[batch].to_numpy(),
                    dtype=torch.long, device=device,
                )
                loss = (
                    F.cross_entropy(
                        output["action_logits"], labels,
                        label_smoothing=0.03,
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
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def nested_site_split(
    all_sites: list[str], held_out_subject: str,
) -> tuple[list[str], list[str], list[str]]:
    """외부 평가 사람을 숨긴 채 남은 사람의 E03만 epoch 선택에 사용한다."""
    outer_test = [
        site for site in all_sites if site.startswith(f"{held_out_subject}_")
    ]
    candidates = [site for site in all_sites if site not in outer_test]
    inner_validation = [site for site in candidates if site.endswith("_E03")]
    if not inner_validation:
        # E03가 없는 구성에서도 다중 환경 사람의 마지막 site 하나만 고정한다.
        multi_site_subjects = sorted({
            site.split("_")[0] for site in candidates
            if sum(
                other.startswith(f"{site.split('_')[0]}_")
                for other in candidates
            ) > 1
        })
        if not multi_site_subjects:
            raise RuntimeError("nested validation requires a multi-site source subject")
        subject = multi_site_subjects[0]
        inner_validation = [
            sorted(
                site for site in candidates if site.startswith(f"{subject}_")
            )[-1]
        ]
    inner_train = [site for site in candidates if site not in inner_validation]
    if len(inner_train) < 2:
        raise RuntimeError(f"not enough nested training sites: {inner_train}")
    return inner_train, inner_validation, outer_test


def site_batches(
    rows: np.ndarray,
    index: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """한 site에서 action 균형 query batch를 만든다."""
    return base.balanced_batches(rows, index, batch_size, seed)


def risk_consistency_loss(
    direct_logits: torch.Tensor,
    action_risk_logits: torch.Tensor,
) -> torch.Tensor:
    """독립 위험 head와 17행동 합산 위험도가 모순되지 않게 맞춘다."""
    direct = direct_logits.log_softmax(-1)
    action = action_risk_logits.log_softmax(-1)
    middle = torch.logsumexp(
        torch.stack((direct, action), dim=0), dim=0
    ) - torch.log(torch.tensor(2.0, device=direct.device))
    return 0.5 * (
        F.kl_div(middle, direct.exp(), reduction="batchmean")
        + F.kl_div(middle, action.exp(), reduction="batchmean")
    )


def counterfactual_view_consistency(
    clean: dict[str, torch.Tensor],
    augmented: dict[str, torch.Tensor],
) -> torch.Tensor:
    """같은 trial의 원본·환경변환 view가 같은 행동 의미를 내도록 묶는다."""
    terms = []
    for key in ("action_logits", "risk_logits"):
        left = clean[key].log_softmax(-1)
        right = augmented[key].log_softmax(-1)
        middle = torch.logsumexp(
            torch.stack((left, right), dim=0), dim=0
        ) - torch.log(torch.tensor(2.0, device=left.device))
        terms.append(0.5 * (
            F.kl_div(middle, left.exp(), reduction="batchmean")
            + F.kl_div(middle, right.exp(), reduction="batchmean")
        ))
    embedding = 1.0 - F.cosine_similarity(
        clean["embedding"], augmented["embedding"], dim=-1
    ).mean()
    return terms[0] + 0.70 * terms[1] + 0.25 * embedding


def teacher_student_view_consistency(
    teacher: dict[str, torch.Tensor],
    student: dict[str, torch.Tensor],
) -> torch.Tensor:
    """clean EMA teacher의 행동 의미를 변형된 student view에 단방향 전이한다."""
    action = F.kl_div(
        student["action_logits"].log_softmax(-1),
        teacher["action_logits"].softmax(-1),
        reduction="batchmean",
    )
    risk = F.kl_div(
        student["risk_logits"].log_softmax(-1),
        teacher["risk_logits"].softmax(-1),
        reduction="batchmean",
    )
    embedding = 1.0 - F.cosine_similarity(
        student["embedding"], teacher["embedding"], dim=-1
    ).mean()
    return action + 0.70 * risk + 0.25 * embedding


@torch.no_grad()
def update_ema_teacher(
    teacher: nn.Module, student: nn.Module, decay: float,
) -> None:
    """student의 이동평균을 teacher에 복사해 급격한 self-target 변화를 막는다."""
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for key, teacher_value in teacher_state.items():
        student_value = student_state[key].to(teacher_value)
        if teacher_value.is_floating_point():
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


@torch.no_grad()
def evaluate_site(
    model: CAL20RelativeMotionDG,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
    device: str,
    seed: int,
    absence_trials: int = 2,
) -> dict:
    """고정 support/absence만 사용해 한 미관측 source site를 평가한다."""
    model.eval()
    rows = base.site_rows(selected_rows, sites, site)
    support = base.select_support(rows, index, seed)
    absence = base.select_absence(
        site, index, seed + 1, trials=absence_trials
    )
    support_set = set(support.tolist())
    query = np.asarray([row for row in rows if row not in support_set], dtype=np.int64)
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    support_labels = torch.tensor(
        index.class_id.iloc[support].to_numpy(), device=device
    )
    action_logits, risk_logits, reliability, link_weights = [], [], [], []
    for start in range(0, len(query), 24):
        batch = query[start:start + 24]
        query_csi, query_mask = store.get(batch, device)
        output = model(
            query_csi, query_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        action_logits.append(output["action_logits"].cpu())
        risk_logits.append(output["risk_logits"].cpu())
        reliability.append(output["motion_reliability"].cpu())
        link_weights.append(output["query_link_weight"].cpu())
    labels = torch.tensor(index.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(index.risk_id.iloc[query].to_numpy()).long()
    metrics = classification_metrics(
        torch.cat(action_logits), torch.cat(risk_logits), labels, risks
    )
    metrics.update({
        "site": site,
        "support_trials": int(len(support)),
        "absence_trials": int(len(absence)),
        "motion_reliability": torch.stack(reliability).mean(0).tolist(),
        "link_weight_mean": torch.cat(link_weights).mean((0, 1)).tolist(),
    })
    return metrics


def group_dro_objective(
    losses: torch.Tensor,
    group_indices: torch.Tensor,
    group_weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Upweight source sites with persistently high task loss."""
    if losses.ndim != 1 or group_indices.shape != losses.shape:
        raise ValueError("group losses and indices must be one-dimensional")
    if eta <= 0.0:
        return losses.mean()
    with torch.no_grad():
        for loss, group in zip(losses, group_indices):
            group_weights[group] *= torch.exp(
                float(eta) * loss.detach().clamp(max=20.0)
            )
        group_weights /= group_weights.sum().clamp_min(1e-8)
    local = group_weights[group_indices]
    local = local / local.sum().clamp_min(1e-8)
    return (local * losses).sum()


def build_episode(
    site: str,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    device: str,
    seed: int,
    batch_size: int,
) -> dict:
    """한 epoch 동안 재사용할 site별 calibration support와 query batch를 준비한다."""
    rows = base.site_rows(selected_rows, sites, site)
    support = base.select_support(rows, index, seed)
    absence = base.select_absence(site, index, seed + 1)
    support_set = set(support.tolist())
    query = np.asarray([row for row in rows if row not in support_set], dtype=np.int64)
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    return {
        "site": site,
        "support_csi": support_csi,
        "support_mask": support_mask,
        "support_labels": torch.tensor(
            index.class_id.iloc[support].to_numpy(), device=device
        ),
        "absence_csi": absence_csi,
        "absence_mask": absence_mask,
        "query": query,
        "pools": {
            class_id: query[
                index.class_id.iloc[query].to_numpy() == class_id
            ]
            for class_id in ACTION_CLASSES
        },
        "batches": site_batches(query, index, batch_size, seed + 2),
    }


def train_model(
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    train_sites: list[str],
    validation_sites: list[str] | None,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int,
    relative_support: bool,
    use_doppler: bool,
    phase_strength: float,
    pose_store: PoseStore | None = None,
    lambda_motion_grounding: float = 0.0,
    fixed_swa: bool = False,
    swa_start: int = 8,
    cross_site_style_probability: float = 0.0,
    cross_subject_pairing: bool = False,
    motion_phase_bins: int = 0,
    reflection_probability: float = 0.0,
    temporal_warp_probability: float = 0.0,
    temporal_warp_strength: float = 0.25,
    initial_state: dict[str, torch.Tensor] | None = None,
    learned_support_matcher: bool = False,
    hierarchical_action_heads: bool = False,
    support_film: bool = False,
    subcarrier_mask_probability: float = 0.0,
    subcarrier_mask_fraction: float = 0.12,
    pose_semantic_distillation: bool = False,
    pose_teacher_epochs: int = 20,
    lambda_pose_semantic: float = 0.20,
    class_aligned_cross_subject: bool = False,
    adaptive_subcarrier_groups: int = 0,
    latent_support_normalization: bool = False,
    group_dro_eta: float = 0.0,
    compositional_action_fusion: bool = False,
    environment_consistency_weight: float = 0.0,
    motion_salience_attention: bool = False,
    ema_environment_consistency_weight: float = 0.0,
    ema_teacher_decay: float = 0.995,
    paired_pose_distillation: bool = False,
    link_dropout_probability: float = 0.0,
    domain_grl: float = 1.0,
) -> tuple[CAL20RelativeMotionDG, list[dict], dict | None]:
    """두 site씩 묶은 episode로 행동 불변성과 calibration을 동시에 학습한다."""
    model_kwargs = {
        "hidden": 64, "width": 192, "domains": len(train_sites),
        "dropout": 0.10, "domain_grl": domain_grl,
        "relative_support": relative_support,
        "use_doppler": use_doppler,
        "phase_strength": phase_strength,
        "motion_phase_bins": motion_phase_bins,
        "learned_support_matcher": learned_support_matcher,
        "hierarchical_action_heads": hierarchical_action_heads,
        "support_film": support_film,
        "adaptive_subcarrier_groups": adaptive_subcarrier_groups,
        "latent_support_normalization": latent_support_normalization,
        "compositional_action_fusion": compositional_action_fusion,
        "motion_salience_attention": motion_salience_attention,
    }
    model = CAL20RelativeMotionDG(**model_kwargs).to(device)
    if initial_state is not None:
        incompatible = model.load_state_dict(initial_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("motion_progress.")
            and not key.startswith("motion_matcher.")
            and not key.startswith("state_matcher.")
            and not key.startswith("coarse_head.")
            and not key.startswith("start_head.")
            and not key.startswith("support_film_head.")
            and not key.startswith("motion_encoder.branch.group_")
            and not key.startswith("motion_salience_projection.")
            and key not in ("motion_matcher_gate", "state_matcher_gate")
            and key != "support_film_gate"
            and key != "latent_support_gate"
            and key != "composition_gate"
            and key != "motion_salience_gate"
            and key != "motion_phase_gate"
        ]
        if unexpected or invalid_missing:
            raise RuntimeError(
                f"invalid source initialization: missing={invalid_missing}, "
                f"unexpected={unexpected}"
            )
    if pose_semantic_distillation and pose_store is None:
        raise ValueError("pose semantic distillation requires source pose GT")
    ema_teacher = None
    if ema_environment_consistency_weight > 0.0:
        ema_teacher = copy.deepcopy(model).eval()
        for parameter in ema_teacher.parameters():
            parameter.requires_grad_(False)
    view_consistency_weight = (
        ema_environment_consistency_weight
        if ema_teacher is not None else environment_consistency_weight
    )
    pose_teacher = (
        pretrain_pose_semantic_teacher(
            pose_store, index, selected_rows, sites, train_sites,
            device, model.hidden, seed + 70_001, pose_teacher_epochs,
        )
        if pose_semantic_distillation else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-3)
    risk_weight = torch.tensor((0.8, 1.0, 1.25), device=device)
    risk_weight /= risk_weight.mean()
    group_weights = torch.full(
        (len(train_sites),), 1.0 / len(train_sites), device=device
    )
    history: list[dict] = []
    best_score = -float("inf")
    best_state = None
    stale = 0
    patience = 8
    swa_state = None
    swa_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        episodes = {
            site: build_episode(
                site, store, index, selected_rows, sites, device,
                seed * 10_000 + epoch * 101 + number * 7,
                batch_size,
            )
            for number, site in enumerate(train_sites)
        }
        schedule = []
        for site, episode in episodes.items():
            schedule.extend((site, number) for number in range(len(episode["batches"])))
        rng.shuffle(schedule)
        losses = []
        diagnostics = []
        for step in range(0, len(schedule), 2):
            chosen = schedule[step:step + 2]
            same_pair_group = len(chosen) == 2 and (
                chosen[0][0].split("_")[0] == chosen[1][0].split("_")[0]
                if cross_subject_pairing
                else chosen[0][0] == chosen[1][0]
            )
            if len(chosen) == 1 or same_pair_group:
                if cross_subject_pairing:
                    # 다른 사람끼리 묶어 사람 지문 대신 공통 행동을 대조한다.
                    subject = chosen[0][0].split("_")[0]
                    alternatives = [
                        site for site in train_sites
                        if site.split("_")[0] != subject
                    ]
                    if not alternatives:
                        alternatives = [
                            site for site in train_sites if site != chosen[0][0]
                        ]
                else:
                    alternatives = [
                        site for site in train_sites if site != chosen[0][0]
                    ]
                other = str(rng.choice(alternatives))
                replacement = (
                    other, int(rng.integers(len(episodes[other]["batches"])))
                )
                if len(chosen) == 1:
                    chosen.append(replacement)
                else:
                    chosen[1] = replacement
            outputs = []
            clean_outputs = []
            all_labels = []
            all_risks = []
            all_domains = []
            all_rows = []
            aligned_rows = None
            if class_aligned_cross_subject and len(chosen) == 2:
                aligned_labels = rng.choice(
                    ACTION_CLASSES,
                    size=batch_size,
                    replace=batch_size > len(ACTION_CLASSES),
                )
                aligned_rows = [
                    np.asarray([
                        rng.choice(episodes[site]["pools"][int(class_id)])
                        for class_id in aligned_labels
                    ], dtype=np.int64)
                    for site, _ in chosen
                ]
            for chosen_index, (site, batch_number) in enumerate(chosen):
                episode = episodes[site]
                batch = (
                    aligned_rows[chosen_index]
                    if aligned_rows is not None
                    else episode["batches"][batch_number]
                )
                query_csi, query_mask = store.get(batch, device)
                clean_absence_csi, clean_absence_mask = (
                    episode["absence_csi"], episode["absence_mask"]
                )
                clean_support_csi, clean_support_mask = (
                    episode["support_csi"], episode["support_mask"]
                )
                clean_query_csi, clean_query_mask = query_csi, query_mask
                if (
                    len(chosen) == 2
                    and cross_site_style_probability > 0.0
                    and rng.random() < cross_site_style_probability
                ):
                    donor_site = chosen[1 - chosen_index][0]
                    donor = episodes[donor_site]
                    style_strength = float(rng.uniform(0.50, 1.0))
                    (absence_csi, absence_mask), (
                        support_csi, support_mask
                    ), (query_csi, query_mask) = base.transfer_site_style(
                        [
                            (episode["absence_csi"], episode["absence_mask"]),
                            (episode["support_csi"], episode["support_mask"]),
                            (query_csi, query_mask),
                        ],
                        (donor["absence_csi"], donor["absence_mask"]),
                        style_strength,
                    )
                else:
                    absence_csi, absence_mask = (
                        episode["absence_csi"], episode["absence_mask"]
                    )
                    support_csi, support_mask = (
                        episode["support_csi"], episode["support_mask"]
                    )
                if rng.random() < 0.75:
                    (absence_csi, absence_mask), (support_csi, support_mask), (
                        query_csi, query_mask
                    ) = base.augment_site(
                        [
                            (absence_csi, absence_mask),
                            (support_csi, support_mask),
                            (query_csi, query_mask),
                        ],
                        seed=seed * 1_000_000 + epoch * 10_000 + step,
                    )
                if rng.random() < reflection_probability:
                    (absence_csi, absence_mask), (support_csi, support_mask), (
                        query_csi, query_mask
                    ) = base.reflect_east_west([
                        (absence_csi, absence_mask),
                        (support_csi, support_mask),
                        (query_csi, query_mask),
                    ])
                if rng.random() < link_dropout_probability:
                    (absence_csi, absence_mask), (
                        support_csi, support_mask
                    ), (query_csi, query_mask) = base.drop_episode_link(
                        [
                            (absence_csi, absence_mask),
                            (support_csi, support_mask),
                            (query_csi, query_mask),
                        ],
                        seed=seed * 1_000_000 + epoch * 10_000 + step + 53,
                    )
                if rng.random() < temporal_warp_probability:
                    query_csi, query_mask = base.temporal_warp_trials(
                        query_csi, query_mask,
                        seed=seed * 1_000_000 + epoch * 10_000 + step + 37,
                        strength=temporal_warp_strength,
                    )
                if (
                    subcarrier_mask_probability > 0.0
                    and rng.random() < subcarrier_mask_probability
                ):
                    (absence_csi, absence_mask), (
                        support_csi, support_mask
                    ), (query_csi, query_mask) = base.mask_subcarrier_band(
                        [
                            (absence_csi, absence_mask),
                            (support_csi, support_mask),
                            (query_csi, query_mask),
                        ],
                        seed=seed * 1_000_000 + epoch * 10_000 + step + 71,
                        fraction=subcarrier_mask_fraction,
                    )
                output = model(
                    query_csi, query_mask,
                    support_csi, support_mask, episode["support_labels"],
                    absence_csi, absence_mask,
                )
                if ema_teacher is not None:
                    with torch.no_grad():
                        clean_outputs.append(ema_teacher(
                            clean_query_csi, clean_query_mask,
                            clean_support_csi, clean_support_mask,
                            episode["support_labels"],
                            clean_absence_csi, clean_absence_mask,
                        ))
                elif environment_consistency_weight > 0.0:
                    clean_outputs.append(model(
                        clean_query_csi, clean_query_mask,
                        clean_support_csi, clean_support_mask,
                        episode["support_labels"],
                        clean_absence_csi, clean_absence_mask,
                    ))
                domain = train_sites.index(site)
                outputs.append(output)
                all_labels.append(torch.tensor(
                    index.class_id.iloc[batch].to_numpy(), device=device
                ))
                all_risks.append(torch.tensor(
                    index.risk_id.iloc[batch].to_numpy(), device=device
                ))
                all_domains.append(torch.full(
                    (len(batch),), domain, dtype=torch.long, device=device
                ))
                all_rows.append(batch)
                if step == 0:
                    model.canonicalizer.update_source_scale(
                        output["amp_scale"], output["phase_scale"], momentum=0.98
                    )

            action_logits = torch.cat([item["action_logits"] for item in outputs])
            risk_logits = torch.cat([item["risk_logits"] for item in outputs])
            direct_risk = torch.cat([
                item["direct_risk_logits"] for item in outputs
            ])
            action_risk = torch.cat([
                item["action_risk_logits"] for item in outputs
            ])
            embedding = torch.cat([item["embedding"] for item in outputs])
            coarse_logits = torch.cat([
                item["coarse_logits"] for item in outputs
            ])
            start_logits = torch.cat([
                item["start_logits"] for item in outputs
            ])
            domain_logits = torch.cat([item["domain_logits"] for item in outputs])
            base_action_logits = torch.cat([
                item["base_action_logits"] for item in outputs
            ])
            base_risk_logits = torch.cat([
                item["base_risk_logits"] for item in outputs
            ])
            action_residual = torch.cat([
                item["action_residual"] for item in outputs
            ])
            risk_residual = torch.cat([
                item["risk_residual"] for item in outputs
            ])
            adapter_gate = torch.cat([
                item["adapter_gate"] for item in outputs
            ])
            latent_support_gate = torch.cat([
                item["latent_support_gate"] for item in outputs
            ])
            composition_strength = torch.cat([
                item["composition_strength"] for item in outputs
            ])
            motion_salience_strength = torch.cat([
                item["motion_salience_strength"] for item in outputs
            ])
            labels = torch.cat(all_labels)
            risks = torch.cat(all_risks)
            domains = torch.cat(all_domains)

            if pose_store is not None:
                pose, pose_valid = pose_store.get(
                    np.concatenate(all_rows), device
                )
                motion_target = pose_motion_descriptor(pose, pose_valid)
                predicted_motion = torch.cat([
                    item["pose_motion"] for item in outputs
                ])
                motion_grounding, selected_shift = shift_robust_motion_loss(
                    predicted_motion, motion_target, pose_valid, max_shift=6
                )
            else:
                motion_grounding = embedding.sum() * 0.0
                selected_shift = embedding.new_zeros(())

            if pose_teacher is not None:
                with torch.no_grad():
                    teacher_output = pose_teacher(motion_target, pose_valid)
                pose_semantic = (
                    paired_pose_distillation_loss(
                        embedding, action_logits, risk_logits, teacher_output
                    )
                    if paired_pose_distillation
                    else cross_modal_supervised_contrastive(
                        embedding, teacher_output["embedding"], labels
                    )
                )
            else:
                pose_semantic = embedding.new_zeros(())

            environment_consistency = (
                torch.stack([
                    (
                        teacher_student_view_consistency(clean, augmented)
                        if ema_teacher is not None
                        else counterfactual_view_consistency(clean, augmented)
                    )
                    for clean, augmented in zip(clean_outputs, outputs)
                ]).mean()
                if clean_outputs else embedding.new_zeros(())
            )

            action_loss = F.cross_entropy(action_logits, labels)
            hierarchy_loss = (
                0.5 * (
                    F.cross_entropy(
                        coarse_logits, model.coarse_action_target[labels]
                    )
                    + F.cross_entropy(
                        start_logits, model.start_posture_target[labels]
                    )
                )
                if model.hierarchical_action_heads
                else embedding.new_zeros(())
            )
            risk_loss = F.cross_entropy(risk_logits, risks, weight=risk_weight)
            direct_risk_loss = F.cross_entropy(
                direct_risk, risks, weight=risk_weight
            )
            domain_loss = F.cross_entropy(domain_logits, domains)
            contrastive = cross_site_supervised_contrastive(
                embedding, labels, domains
            )
            consistency = risk_consistency_loss(direct_risk, action_risk)
            base_action_loss = F.cross_entropy(base_action_logits, labels)
            base_risk_loss = F.cross_entropy(
                base_risk_logits, risks, weight=risk_weight
            )
            calibrated_per_item = (
                F.cross_entropy(action_logits, labels, reduction="none")
                + 0.70 * F.cross_entropy(
                    risk_logits, risks, weight=risk_weight, reduction="none"
                )
            )
            site_task_losses = torch.stack([
                F.cross_entropy(output["action_logits"], current_labels)
                + 0.70 * F.cross_entropy(
                    output["risk_logits"], current_risks,
                    weight=risk_weight,
                )
                for output, current_labels, current_risks in zip(
                    outputs, all_labels, all_risks
                )
            ])
            primary_task_loss = (
                group_dro_objective(
                    site_task_losses,
                    torch.tensor([
                        train_sites.index(site) for site, _ in chosen
                    ], dtype=torch.long, device=device),
                    group_weights,
                    group_dro_eta,
                )
                if group_dro_eta > 0.0
                else action_loss + 0.70 * risk_loss
            )
            base_per_item = (
                F.cross_entropy(base_action_logits, labels, reduction="none")
                + 0.70 * F.cross_entropy(
                    base_risk_logits, risks, weight=risk_weight, reduction="none"
                )
            )
            no_harm = F.relu(calibrated_per_item - base_per_item).mean()
            residual_penalty = (
                action_residual.square().mean() + risk_residual.square().mean()
            )
            link_entropy = torch.stack([
                -(item["query_link_weight"].clamp_min(1e-7).log()
                  * item["query_link_weight"]).sum(-1).mean()
                for item in outputs
            ]).mean()
            loss = (
                primary_task_loss
                + 0.20 * direct_risk_loss
                + 0.12 * domain_loss
                + 0.15 * contrastive
                + 0.08 * consistency
                + 0.30 * (base_action_loss + 0.70 * base_risk_loss)
                + 0.20 * no_harm
                + 0.005 * residual_penalty
                + 0.005 * adapter_gate.mean()
                + 0.002 * latent_support_gate.abs().mean()
                + 0.001 * composition_strength.abs().mean()
                + 0.001 * motion_salience_strength.mean()
                - 0.01 * link_entropy
                + lambda_motion_grounding * motion_grounding
                + 0.15 * hierarchy_loss
                + lambda_pose_semantic * pose_semantic
                + view_consistency_weight * environment_consistency
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if ema_teacher is not None:
                update_ema_teacher(ema_teacher, model, ema_teacher_decay)
            losses.append(float(loss.detach()))
            diagnostics.append({
                "action": float(action_loss.detach()),
                "risk": float(risk_loss.detach()),
                "domain": float(domain_loss.detach()),
                "contrastive": float(contrastive.detach()),
                "motion_grounding": float(motion_grounding.detach()),
                "motion_shift": float(selected_shift),
                "hierarchy": float(hierarchy_loss.detach()),
                "pose_semantic": float(pose_semantic.detach()),
                "environment_consistency": float(
                    environment_consistency.detach()
                ),
            })

        record: dict = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_components": {
                key: float(np.mean([item[key] for item in diagnostics]))
                for key in diagnostics[0]
            },
            "risk_fusion": float(torch.sigmoid(model.risk_fusion).detach()),
            "source_amp_scale": model.canonicalizer.source_amp_scale.tolist(),
            "source_phase_scale": model.canonicalizer.source_phase_scale.tolist(),
        }
        if hasattr(model, "hierarchy_logit"):
            record["hierarchy_strength"] = float(
                torch.sigmoid(model.hierarchy_logit).detach()
            )
        if model.motion_phase_gate is not None:
            record["motion_phase_strength"] = float(
                torch.sigmoid(model.motion_phase_gate).detach()
            )
        if model.compositional_action_fusion:
            record["composition_strength"] = float(
                (0.5 * torch.tanh(model.composition_gate)).detach()
            )
        if model.motion_salience_projection is not None:
            record["motion_salience_strength"] = float(
                torch.sigmoid(model.motion_salience_gate).detach()
            )
        if model.support_film_head is not None:
            record["support_film_strength"] = float(
                torch.sigmoid(model.support_film_gate).detach()
            )
        if model.latent_support_normalization:
            record["latent_support_strength"] = float(
                (0.5 * torch.tanh(model.latent_support_gate)).detach()
            )
        if group_dro_eta > 0.0:
            record["group_dro_weights"] = {
                site: float(group_weights[number].detach())
                for number, site in enumerate(train_sites)
            }
        if fixed_swa and epoch >= swa_start:
            current_state = model.state_dict()
            if swa_state is None:
                swa_state = {
                    key: value.detach().cpu().clone()
                    for key, value in current_state.items()
                }
            else:
                for key, value in current_state.items():
                    if value.is_floating_point():
                        swa_state[key].lerp_(
                            value.detach().cpu(), 1.0 / float(swa_count + 1)
                        )
                    else:
                        swa_state[key].copy_(value.detach().cpu())
            swa_count += 1
        if validation_sites:
            validation = {
                site: evaluate_site(
                    model, store, index, selected_rows, sites,
                    site, device, seed=17017,
                )
                for site in validation_sites
            }
            site_diagnostics = {
                site: cal12_site_selection_score(metrics)
                for site, metrics in validation.items()
            }
            scores = [item["score"] for item in site_diagnostics.values()]
            mean_score = float(np.mean(scores))
            worst_score = float(np.min(scores))
            selection_score = 0.5 * mean_score + 0.5 * worst_score
            record.update({
                "validation": validation,
                "validation_diagnostics": site_diagnostics,
                "mean_site_score": mean_score,
                "worst_site_score": worst_score,
                "selection_score": selection_score,
            })
            if selection_score > best_score + 1e-5:
                best_score = selection_score
                best_state = copy.deepcopy({
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                })
                stale = 0
            else:
                stale += 1
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation_sites and not fixed_swa and stale >= patience:
            break

    if fixed_swa:
        if swa_state is None:
            raise RuntimeError("SWA did not receive any epoch")
        model.load_state_dict(swa_state)
        if validation_sites:
            validation = {
                site: evaluate_site(
                    model, store, index, selected_rows, sites,
                    site, device, seed=17017,
                )
                for site in validation_sites
            }
            diagnostics = {
                site: cal12_site_selection_score(metrics)
                for site, metrics in validation.items()
            }
            scores = [item["score"] for item in diagnostics.values()]
            best = {
                "epoch": epochs,
                "selection": "fixed_swa_no_validation_selection",
                "swa_start": swa_start,
                "swa_snapshots": swa_count,
                "validation": validation,
                "validation_diagnostics": diagnostics,
                "mean_site_score": float(np.mean(scores)),
                "worst_site_score": float(np.min(scores)),
                "selection_score": float(
                    0.5 * np.mean(scores) + 0.5 * np.min(scores)
                ),
            }
        else:
            best = None
    elif best_state is not None:
        model.load_state_dict(best_state)
        best = max(history, key=lambda item: item["selection_score"])
    else:
        best = None
    return model, history, best


def cal12_site_selection_score(metrics: dict) -> dict[str, float]:
    """행동이 무너지면 danger 점수만으로 checkpoint가 선택되지 않게 계산한다."""
    safe_specificity = 1.0 - (
        metrics["safe_to_danger"] / max(metrics["safe_total"], 1)
    )
    danger_recall = metrics["danger_recall"]
    danger_balance = (
        2.0 * danger_recall * safe_specificity
        / max(danger_recall + safe_specificity, 1e-8)
    )
    action_utility = (
        0.70 * metrics["action_macro_f1"]
        + 0.30 * metrics["action_accuracy"]
    )
    risk_utility = (
        0.50 * metrics["risk_macro_f1"] + 0.50 * danger_balance
    )
    score = (
        3.0 * action_utility
        + np.sqrt(max(action_utility * risk_utility, 0.0))
        + 0.25 * metrics["danger_action_accuracy"]
    )
    return {
        "score": float(score),
        "safe_specificity": float(safe_specificity),
        "danger_balance": float(danger_balance),
        "action_utility": float(action_utility),
        "risk_utility": float(risk_utility),
    }


def load_clean_source_state(path: Path) -> dict[str, torch.Tensor]:
    """봉인 target이나 outer 선택에 오염되지 않은 source checkpoint만 읽는다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_false = (
        "outer_holdout_used_for_selection",
        "target_subject_used",
        "sealed_yja_used",
    )
    contaminated = [
        key for key in required_false if checkpoint.get(key) is not False
    ]
    if contaminated:
        raise RuntimeError(f"source initialization is not clean: {contaminated}")
    if "model" not in checkpoint:
        raise RuntimeError(f"source initialization has no model: {path}")
    return checkpoint["model"]


def experiment_name(options: argparse.Namespace) -> str:
    """활성화된 행동 불변성 조합을 재현 가능한 실험 이름으로 변환한다."""
    reflected = options.reflection_probability > 0.0
    warped = options.temporal_warp_probability > 0.0
    learned_matcher = bool(getattr(options, "learned_support_matcher", False))
    hierarchical = bool(getattr(options, "hierarchical_action_heads", False))
    support_film = bool(getattr(options, "support_film", False))
    pose_semantic = bool(
        getattr(options, "pose_semantic_distillation", False)
    )
    aligned = bool(
        getattr(options, "class_aligned_cross_subject", False)
    )
    grouped = int(
        getattr(options, "adaptive_subcarrier_groups", 0)
    ) >= 2
    latent_normalization = bool(
        getattr(options, "latent_support_normalization", False)
    )
    group_dro = float(getattr(options, "group_dro_eta", 0.0)) > 0.0
    compositional = bool(
        getattr(options, "compositional_action_fusion", False)
    )
    environment_consistency = float(
        getattr(options, "environment_consistency_weight", 0.0)
    ) > 0.0
    motion_salience = bool(
        getattr(options, "motion_salience_attention", False)
    )
    ema_consistency = float(
        getattr(options, "ema_environment_consistency_weight", 0.0)
    ) > 0.0
    paired_pose = bool(
        getattr(options, "paired_pose_distillation", False)
    )
    link_dropout = float(
        getattr(options, "link_dropout_probability", 0.0)
    ) > 0.0
    subcarrier_masked = float(
        getattr(options, "subcarrier_mask_probability", 0.0)
    ) > 0.0
    if subcarrier_masked and reflected and warped:
        return "KP-v2-A7-CAL60-SUBCARRIER-MASK-DG"
    if paired_pose and reflected and warped:
        return "KP-v2-A22-CAL60-PAIRED-POSE-DISTILLATION"
    if link_dropout and reflected and warped:
        return "KP-v2-A23-CAL60-LINK-DROPOUT"
    if pose_semantic and reflected and warped:
        return "KP-v2-A8-CAL60-POSE-SEMANTIC-DISTILLATION"
    if aligned and reflected and warped:
        return "KP-v2-A9-CAL60-CLASS-ALIGNED-CROSS-SUBJECT"
    if grouped and reflected and warped:
        return "KP-v2-A10-CAL60-ADAPTIVE-SUBCARRIER-GROUPS"
    if latent_normalization and reflected and warped:
        return "KP-v2-A13-CAL60-LATENT-SUPPORT-NORMALIZATION"
    if group_dro and reflected and warped:
        return "KP-v2-A14-CAL60-GROUP-DRO"
    if environment_consistency and reflected and warped:
        return "KP-v2-A18-CAL60-ENVIRONMENT-CONSISTENCY"
    if motion_salience and reflected and warped:
        return "KP-v2-A19-CAL60-MOTION-SALIENCE"
    if ema_consistency and reflected and warped:
        return "KP-v2-A20-CAL60-EMA-ENVIRONMENT-DISTILLATION"
    if compositional and reflected and warped:
        return "KP-v2-A16-CAL60-COMPOSITIONAL-ACTION"
    if support_film and reflected and warped:
        return "KP-v2-A6-CAL60-SUPPORT-FILM"
    if hierarchical and learned_matcher and reflected and warped:
        return "KP-v2-A6-CAL60-MATCHER-HIERARCHY"
    if hierarchical and reflected and warped:
        return "KP-v2-A5-CAL60-ACTION-HIERARCHY"
    if learned_matcher and reflected and warped:
        return "KP-v2-A4-CAL60-LEARNED-SUPPORT-MATCHER"
    if learned_matcher:
        return "KP-v2-CAL20-LEARNED-SUPPORT-MATCHER"
    if reflected and warped:
        return "CAL60-PHYSICAL-INVARIANCE-DG"
    if warped:
        return "CAL58-MONOTONIC-TIME-WARP-DG"
    if reflected:
        return "CAL53-EAST-WEST-REFLECTION-DG"
    if options.motion_phase_bins >= 2:
        return "CAL34-MOTION-PHASE-DG"
    if options.cross_site_style_probability > 0.0:
        return "CAL33-CROSS-SITE-STYLE"
    return "CAL20-RELATIVE-MOTION-DG"


def validate_training_options(options: argparse.Namespace) -> None:
    """증강 확률과 진행률 설정의 잘못된 조합을 학습 시작 전에 차단한다."""
    for name in (
        "cross_site_style_probability",
        "reflection_probability",
        "temporal_warp_probability",
        "link_dropout_probability",
    ):
        value = float(getattr(options, name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if options.motion_phase_bins < 0 or options.motion_phase_bins == 1:
        raise ValueError("motion_phase_bins must be 0 or at least 2")
    if options.temporal_warp_strength < 0.0:
        raise ValueError("temporal_warp_strength cannot be negative")
    if (
        bool(getattr(options, "compositional_action_fusion", False))
        and not bool(getattr(options, "hierarchical_action_heads", False))
    ):
        raise ValueError(
            "compositional action fusion requires hierarchical action heads"
        )
    if (
        bool(getattr(options, "paired_pose_distillation", False))
        and not bool(getattr(options, "pose_semantic_distillation", False))
    ):
        raise ValueError(
            "paired pose distillation requires pose semantic distillation"
        )
    subcarrier_mask_probability = float(
        getattr(options, "subcarrier_mask_probability", 0.0)
    )
    subcarrier_mask_fraction = float(
        getattr(options, "subcarrier_mask_fraction", 0.12)
    )
    if not 0.0 <= subcarrier_mask_probability <= 1.0:
        raise ValueError("subcarrier_mask_probability must be between 0 and 1")
    if not 0.0 < subcarrier_mask_fraction < 1.0:
        raise ValueError("subcarrier_mask_fraction must be between 0 and 1")
    if getattr(options, "pose_teacher_epochs", 20) < 1:
        raise ValueError("pose_teacher_epochs must be positive")
    if getattr(options, "lambda_pose_semantic", 0.20) < 0.0:
        raise ValueError("lambda_pose_semantic cannot be negative")
    if (
        getattr(options, "class_aligned_cross_subject", False)
        and not getattr(options, "cross_subject_pairing", False)
    ):
        raise ValueError(
            "class_aligned_cross_subject requires cross_subject_pairing"
        )
    adaptive_groups = int(getattr(options, "adaptive_subcarrier_groups", 0))
    if adaptive_groups not in (0,) and adaptive_groups < 2:
        raise ValueError("adaptive_subcarrier_groups must be 0 or at least 2")
    if getattr(options, "group_dro_eta", 0.0) < 0.0:
        raise ValueError("group_dro_eta cannot be negative")
    if getattr(options, "environment_consistency_weight", 0.0) < 0.0:
        raise ValueError("environment_consistency_weight cannot be negative")
    if getattr(options, "ema_environment_consistency_weight", 0.0) < 0.0:
        raise ValueError(
            "ema_environment_consistency_weight cannot be negative"
        )
    decay = float(getattr(options, "ema_teacher_decay", 0.995))
    if not 0.0 <= decay < 1.0:
        raise ValueError("ema_teacher_decay must be in [0, 1)")
    if getattr(options, "domain_grl", 1.0) < 0.0:
        raise ValueError("domain_grl cannot be negative")


def main() -> None:
    """source subject-LOSO와 전체 source deployment 학습을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--training-seed", type=int, default=12012,
        help="fold 학습·augmentation·deployment 재현에 사용할 기준 seed",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-relative-support", action="store_true")
    parser.add_argument("--use-doppler", action="store_true")
    parser.add_argument("--phase-strength", type=float, default=1.0)
    parser.add_argument("--motion-grounding", action="store_true")
    parser.add_argument("--lambda-motion-grounding", type=float, default=0.30)
    parser.add_argument("--fixed-swa", action="store_true")
    parser.add_argument("--swa-start", type=int, default=8)
    parser.add_argument(
        "--cross-site-style-probability", type=float, default=0.0,
        help="다른 source absence의 정적 반사 기준선으로 episode를 옮길 확률",
    )
    parser.add_argument(
        "--cross-subject-pairing", action="store_true",
        help="paired episode를 서로 다른 source 사람으로 강제한다",
    )
    parser.add_argument(
        "--motion-phase-bins", type=int, default=0,
        help="실제 시간과 누적 움직임 진행률을 요약할 bin 개수",
    )
    parser.add_argument(
        "--reflection-probability", type=float, default=0.0,
        help="동·서 TX를 episode 전체에서 반사할 학습 확률",
    )
    parser.add_argument(
        "--temporal-warp-probability", type=float, default=0.0,
        help="query 수행 속도를 단조 재표집할 학습 확률",
    )
    parser.add_argument(
        "--temporal-warp-strength", type=float, default=0.25,
        help="시간 warp 지수의 log 범위",
    )
    parser.add_argument(
        "--initialize-from-run", type=Path,
        help="target-clean source checkpoint에서 새 adapter를 이어 학습한다",
    )
    parser.add_argument("--learned-support-matcher", action="store_true")
    parser.add_argument("--hierarchical-action-heads", action="store_true")
    parser.add_argument("--compositional-action-fusion", action="store_true")
    parser.add_argument("--support-film", action="store_true")
    parser.add_argument("--subcarrier-mask-probability", type=float, default=0.0)
    parser.add_argument("--subcarrier-mask-fraction", type=float, default=0.12)
    parser.add_argument("--pose-semantic-distillation", action="store_true")
    parser.add_argument("--pose-teacher-epochs", type=int, default=20)
    parser.add_argument("--lambda-pose-semantic", type=float, default=0.20)
    parser.add_argument("--class-aligned-cross-subject", action="store_true")
    parser.add_argument("--adaptive-subcarrier-groups", type=int, default=0)
    parser.add_argument("--latent-support-normalization", action="store_true")
    parser.add_argument("--group-dro-eta", type=float, default=0.0)
    parser.add_argument(
        "--environment-consistency-weight", type=float, default=0.0,
    )
    parser.add_argument("--motion-salience-attention", action="store_true")
    parser.add_argument(
        "--ema-environment-consistency-weight", type=float, default=0.0,
    )
    parser.add_argument("--ema-teacher-decay", type=float, default=0.995)
    parser.add_argument("--paired-pose-distillation", action="store_true")
    parser.add_argument("--link-dropout-probability", type=float, default=0.0)
    parser.add_argument(
        "--domain-grl", type=float, default=1.0,
        help="site domain adversary가 encoder에 전달하는 gradient reversal 강도",
    )
    options = parser.parse_args()
    validate_training_options(options)
    run = options.run_dir
    run.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("sealed yja must not appear in CAL20 source rows")
    sites = (selected_index.subject + "_" + selected_index.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected CAL20 source sites: {all_sites}")

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
    pose_store = PoseStore(WORK / "cache") if options.motion_grounding else None

    fold_results = {}
    best_epochs = []
    for fold_number, held_out in enumerate(("ajh", "mhw", "lmh")):
        train_sites, validation_sites, outer_test_sites = nested_site_split(
            all_sites, held_out
        )
        checkpoint_path = run / f"selection_{held_out}.pt"
        if options.skip_existing and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("protocol_version") != "nested_source_v1":
                raise RuntimeError(
                    f"legacy checkpoint cannot be resumed safely: {checkpoint_path}"
                )
            best = checkpoint["best"]
            best_epochs.append(int(best["epoch"]))
            fold_results[held_out] = {
                "train_sites": train_sites,
                "inner_validation_sites": validation_sites,
                "outer_test_sites": outer_test_sites,
                "best": best,
                "outer_test": checkpoint["outer_test"],
                "history": [],
                "resumed_from_checkpoint": True,
            }
            continue
        model, history, best = train_model(
            store, index, selected_rows, sites,
            train_sites, validation_sites,
            options.epochs, device, options.training_seed + fold_number * 101,
            options.batch_size,
            not options.disable_relative_support,
            options.use_doppler,
            options.phase_strength,
            pose_store,
            options.lambda_motion_grounding,
            options.fixed_swa,
            options.swa_start,
            options.cross_site_style_probability,
            options.cross_subject_pairing,
            options.motion_phase_bins,
            options.reflection_probability,
            options.temporal_warp_probability,
            options.temporal_warp_strength,
            (
                load_clean_source_state(
                    options.initialize_from_run / f"selection_{held_out}.pt"
                )
                if options.initialize_from_run is not None else None
            ),
            options.learned_support_matcher,
            options.hierarchical_action_heads,
            options.support_film,
            options.subcarrier_mask_probability,
            options.subcarrier_mask_fraction,
            options.pose_semantic_distillation,
            options.pose_teacher_epochs,
            options.lambda_pose_semantic,
            options.class_aligned_cross_subject,
            options.adaptive_subcarrier_groups,
            options.latent_support_normalization,
            options.group_dro_eta,
            options.compositional_action_fusion,
            options.environment_consistency_weight,
            options.motion_salience_attention,
            options.ema_environment_consistency_weight,
            options.ema_teacher_decay,
            options.paired_pose_distillation,
            options.link_dropout_probability,
            options.domain_grl,
        )
        if best is None:
            raise RuntimeError(f"{held_out} fold did not produce a checkpoint")
        outer_test = {
            site: evaluate_site(
                model, store, index, selected_rows, sites,
                site, device, seed=17017,
            )
            for site in outer_test_sites
        }
        best_epochs.append(int(best["epoch"]))
        fold_results[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": validation_sites,
            "outer_test_sites": outer_test_sites,
            "best": best,
            "outer_test": outer_test,
            "history": history,
        }
        torch.save({
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "held_out_subject": held_out,
            "best": best,
            "outer_test": outer_test,
            "protocol_version": "nested_source_v1",
            "outer_holdout_used_for_selection": False,
            "inner_validation_used_for_selection": not options.fixed_swa,
            "target_subject_used": False,
            "sealed_yja_used": False,
        }, checkpoint_path)

    locked_epochs = max(1, int(round(float(np.median(best_epochs)))))
    deployment, deployment_history, _ = train_model(
        store, index, selected_rows, sites,
        all_sites, None, locked_epochs, device,
        options.training_seed + 1000, options.batch_size,
        not options.disable_relative_support,
        options.use_doppler,
        options.phase_strength,
        pose_store,
        options.lambda_motion_grounding,
        options.fixed_swa,
        options.swa_start,
        options.cross_site_style_probability,
        options.cross_subject_pairing,
        options.motion_phase_bins,
        options.reflection_probability,
        options.temporal_warp_probability,
        options.temporal_warp_strength,
        (
            load_clean_source_state(
                options.initialize_from_run / "deployment_model.pt"
            )
            if options.initialize_from_run is not None else None
        ),
        options.learned_support_matcher,
        options.hierarchical_action_heads,
        options.support_film,
        options.subcarrier_mask_probability,
        options.subcarrier_mask_fraction,
        options.pose_semantic_distillation,
        options.pose_teacher_epochs,
        options.lambda_pose_semantic,
        options.class_aligned_cross_subject,
        options.adaptive_subcarrier_groups,
        options.latent_support_normalization,
        options.group_dro_eta,
        options.compositional_action_fusion,
        options.environment_consistency_weight,
        options.motion_salience_attention,
        options.ema_environment_consistency_weight,
        options.ema_teacher_decay,
        options.paired_pose_distillation,
        options.link_dropout_probability,
        options.domain_grl,
    )
    source_metrics = {
        site: evaluate_site(
            deployment, store, index, selected_rows, sites,
            site, device, seed=17017,
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
            "source_pose_gt_training_only": options.motion_grounding,
        },
        "selection_protocol": (
            "fixed_source_swa_v1" if options.fixed_swa
            else "nested_source_subject_loso_v1"
        ),
        "outer_holdout_used_for_selection": False,
        "inner_validation_used_for_selection": not options.fixed_swa,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }, run / "deployment_model.pt")
    result = {
        "run": experiment_name(options),
        "cross_site_style_probability": options.cross_site_style_probability,
        "cross_subject_pairing": options.cross_subject_pairing,
        "motion_phase_bins": options.motion_phase_bins,
        "reflection_probability": options.reflection_probability,
        "temporal_warp_probability": options.temporal_warp_probability,
        "temporal_warp_strength": options.temporal_warp_strength,
        "learned_support_matcher": options.learned_support_matcher,
        "hierarchical_action_heads": options.hierarchical_action_heads,
        "support_film": options.support_film,
        "subcarrier_mask_probability": options.subcarrier_mask_probability,
        "subcarrier_mask_fraction": options.subcarrier_mask_fraction,
        "pose_semantic_distillation": options.pose_semantic_distillation,
        "pose_teacher_epochs": options.pose_teacher_epochs,
        "lambda_pose_semantic": options.lambda_pose_semantic,
        "class_aligned_cross_subject": options.class_aligned_cross_subject,
        "adaptive_subcarrier_groups": options.adaptive_subcarrier_groups,
        "latent_support_normalization": options.latent_support_normalization,
        "group_dro_eta": options.group_dro_eta,
        "compositional_action_fusion": options.compositional_action_fusion,
        "environment_consistency_weight": options.environment_consistency_weight,
        "motion_salience_attention": options.motion_salience_attention,
        "ema_environment_consistency_weight": (
            options.ema_environment_consistency_weight
        ),
        "ema_teacher_decay": options.ema_teacher_decay,
        "paired_pose_distillation": options.paired_pose_distillation,
        "link_dropout_probability": options.link_dropout_probability,
        "domain_grl": options.domain_grl,
        "initialized_from_run": (
            str(options.initialize_from_run.resolve())
            if options.initialize_from_run is not None else None
        ),
        "device": device,
        "training_seed": options.training_seed,
        "fold_results": fold_results,
        "fold_best_epochs": best_epochs,
        "locked_epochs": locked_epochs,
        "source_metrics": source_metrics,
        "deployment_history": deployment_history,
        "selection_protocol": (
            "fixed_source_swa_v1" if options.fixed_swa
            else "nested_source_subject_loso_v1"
        ),
        "outer_holdout_used_for_selection": False,
        "inner_validation_used_for_selection": not options.fixed_swa,
        "source_pose_gt_training_only": options.motion_grounding,
        "query_pose_gt_used": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }
    (run / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "fold_best_epochs": best_epochs,
        "locked_epochs": locked_epochs,
        "fold_best": {
            subject: value["best"] for subject, value in fold_results.items()
        },
        "source_metrics": source_metrics,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
