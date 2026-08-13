"""CAL20: 절대 환경 state를 차단하고 support-relative motion만 분류하는 모델."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from . import contract as C
from .cal12 import (
    CLASS_RANGES,
    DYNAMIC_PROMPT_CLASSES,
    PhysicsSupportCanonicalizer,
    SubcarrierBranch,
    _GradientReverse,
)
from .cal13 import MOTION_DESCRIPTOR_DIM
from .cal14 import CosineClassifier
from .doppler_pose import DopplerFilterBank
from .meta_calibration import MOTION_PROMPT_CLASSES, masked_ordered_summary


COARSE_ACTION_TARGET = (
    0, 1, 1, 1, 2, 2, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4,
)
START_POSTURE_TARGET = (
    0, 1, 2, 3, 3, 1, 1, 2, 1, 0, 0, 3, 1, 0, 3, 3, 2,
)


class LearnedSupportMatcher(nn.Module):
    """현장 support와 query 사이에서 행동에 유효한 차이를 학습한다."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, 1),
        )

    def forward(
        self, query: torch.Tensor, prototype: torch.Tensor,
    ) -> torch.Tensor:
        """모든 query-prototype 쌍의 학습된 관계 점수를 반환한다."""
        query = F.normalize(query, dim=-1)[:, None]
        prototype = F.normalize(prototype, dim=-1)[None]
        query = query.expand(-1, prototype.shape[1], -1)
        prototype = prototype.expand(len(query), -1, -1)
        return self.score(torch.cat((
            query, prototype, (query - prototype).abs(), query * prototype,
        ), dim=-1)).squeeze(-1)


class MotionShapeEncoder(nn.Module):
    """trial별 크기와 평균을 제거한 motion shape를 고정 링크 방향으로 통합한다."""

    def __init__(
        self,
        hidden: int,
        dropout: float,
        use_doppler: bool,
        adaptive_subcarrier_groups: int = 0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.branch = SubcarrierBranch(
            6, hidden, dropout,
            adaptive_groups=adaptive_subcarrier_groups,
        )
        self.doppler = (
            DopplerFilterBank(hidden, hidden, dropout) if use_doppler else None
        )
        self.register_buffer(
            "link_geometry", torch.tensor(C.LINK_GEOMETRY, dtype=torch.float32)
        )
        self.geometry = nn.Sequential(
            nn.Linear(self.link_geometry.shape[1], hidden), nn.Tanh()
        )
        self.link_gate = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 7 + C.N_LINKS),
            nn.Linear(hidden * 7 + C.N_LINKS, hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
        )
        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden, hidden, kernel_size=5, padding=2 * dilation,
                    dilation=dilation, groups=hidden,
                ),
                nn.Conv1d(hidden, hidden * 2, kernel_size=1),
                nn.GLU(dim=1),
                nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
                nn.GELU(), nn.Dropout(dropout),
            )
            for dilation in (1, 2, 4, 8)
        ])
        self.output_norm = nn.LayerNorm(hidden)

    def forward(
        self, motion: torch.Tensor, link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """환경별 feature 평균·분산을 제거하고 상대 움직임 세기는 별도로 반환한다."""
        feature = self.branch(motion)
        if self.doppler is not None:
            feature = feature + self.doppler(feature)
        weight = link_mask.to(feature.dtype)[..., None]
        count = weight.sum(1, keepdim=True).clamp_min(1.0)
        mean = (feature * weight).sum(1, keepdim=True) / count
        variance = ((feature - mean).square() * weight).sum(1, keepdim=True) / count
        standard = variance.clamp_min(1e-4).sqrt()
        energy = (standard * weight).sum((1, 3)) / weight.sum(
            (1, 3)
        ).clamp_min(1.0)
        feature = ((feature - mean) / standard) * weight
        feature = feature + self.geometry(self.link_geometry).to(feature)[
            None, None
        ]
        score = self.link_gate(feature).squeeze(-1).masked_fill(~link_mask, -1e4)
        link_weight = torch.softmax(score, dim=-1)
        link_weight = torch.where(
            link_mask.any(-1, keepdim=True), link_weight,
            torch.zeros_like(link_weight),
        )
        weighted = (feature * link_weight[..., None]).sum(-2)
        masked_links = feature * weight
        differences = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            valid = (link_mask[..., left] & link_mask[..., right]).to(feature.dtype)
            differences.append(
                (feature[..., left, :] - feature[..., right, :]) * valid[..., None]
            )
        fused = self.fusion(torch.cat((
            weighted, masked_links.flatten(-2),
            torch.cat(differences, dim=-1), link_mask.to(feature.dtype),
        ), dim=-1))
        frame_mask = link_mask.any(-1)
        values = fused.transpose(1, 2)
        for block in self.temporal:
            values = values + block(values)
            values = values * frame_mask[:, None].to(values.dtype)
        output = self.output_norm(values.transpose(1, 2))
        output = output * frame_mask[..., None].to(output.dtype)
        return output, frame_mask, link_weight, energy


class MotionProgressEncoder(nn.Module):
    """동작 속도와 무관한 진행률 bin과 실제 시간 bin을 함께 인코딩한다."""

    def __init__(self, hidden: int, bins: int, dropout: float):
        super().__init__()
        if bins < 2:
            raise ValueError("motion progress requires at least two bins")
        self.hidden = int(hidden)
        self.bins = int(bins)
        self.register_buffer("centers", torch.linspace(0.0, 1.0, bins))
        dimension = hidden * bins * 2
        self.projection = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )

    def _soft_pool(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        coordinate: torch.Tensor,
    ) -> torch.Tensor:
        """연속 좌표 주변의 프레임을 Gaussian 가중 평균해 고정 개수 token으로 만든다."""
        bandwidth = 0.75 / max(self.bins - 1, 1)
        distance = coordinate[..., None] - self.centers.to(coordinate)[None, None]
        weight = torch.exp(-0.5 * (distance / bandwidth).square())
        weight = weight * frame_mask.to(weight.dtype)[..., None]
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("btp,bth->bph", weight, features)

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """시간 위치와 누적 움직임 진행률을 결합해 순서 보존 trial 표현을 만든다."""
        if features.ndim != 3 or frame_mask.shape != features.shape[:2]:
            raise ValueError("expected features [B,T,H] and frame mask [B,T]")
        valid_pair = frame_mask[:, 1:] & frame_mask[:, :-1]
        speed = torch.linalg.vector_norm(
            features[:, 1:] - features[:, :-1], dim=-1
        ) * valid_pair.to(features.dtype)
        speed = F.pad(speed, (1, 0))
        activity = speed + 1e-4 * frame_mask.to(features.dtype)
        progress = activity.cumsum(1)
        progress = progress / progress[:, -1:].clamp_min(1e-6)

        ordinal = frame_mask.to(features.dtype).cumsum(1) - 1.0
        timeline = ordinal / (frame_mask.sum(1, keepdim=True) - 1).clamp_min(1)
        clock_tokens = self._soft_pool(features, frame_mask, timeline)
        progress_tokens = self._soft_pool(features, frame_mask, progress)
        return self.projection(
            torch.cat((clock_tokens, progress_tokens), dim=-1).flatten(1)
        )


class RelativeStateEncoder(nn.Module):
    """절대 state는 head에 주지 않고 support와 비교할 trial embedding만 만든다."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.branch = SubcarrierBranch(3, hidden, dropout)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden * C.N_LINKS + C.N_LINKS),
            nn.Linear(hidden * C.N_LINKS + C.N_LINKS, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden),
        )

    def forward(self, state: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        """링크별 시간 평균을 보존하되 최종 출력은 support 거리 계산에만 사용한다."""
        feature = self.branch(state)
        weight = link_mask.to(feature.dtype)[..., None]
        mean = (feature * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        available = link_mask.any(1).to(feature.dtype)
        return self.projection(torch.cat((mean.flatten(1), available), dim=-1))


class CAL20RelativeMotionDG(nn.Module):
    """motion shape와 사용자 기본동작의 상대 좌표만으로 17행동·3위험을 예측한다."""

    def __init__(
        self,
        hidden: int = 64,
        width: int = 192,
        domains: int = 7,
        dropout: float = 0.10,
        domain_grl: float = 1.0,
        prompt_classes: tuple[int, ...] = MOTION_PROMPT_CLASSES,
        relative_support: bool = True,
        use_doppler: bool = True,
        phase_strength: float = 0.25,
        cosine_scale: float = 10.0,
        motion_phase_bins: int = 0,
        learned_support_matcher: bool = False,
        hierarchical_action_heads: bool = False,
        support_film: bool = False,
        adaptive_subcarrier_groups: int = 0,
        latent_support_normalization: bool = False,
        compositional_action_fusion: bool = False,
        motion_salience_attention: bool = False,
    ):
        super().__init__()
        if not relative_support:
            raise ValueError("CAL20 requires deployment support")
        self.hidden = int(hidden)
        self.width = int(width)
        self.domains = int(domains)
        self.dropout = float(dropout)
        self.domain_grl = float(domain_grl)
        self.prompt_classes = tuple(int(value) for value in prompt_classes)
        self.relative_support = True
        self.use_doppler = bool(use_doppler)
        self.phase_strength = float(phase_strength)
        self.cosine_scale = float(cosine_scale)
        self.motion_phase_bins = int(motion_phase_bins)
        self.learned_support_matcher = bool(learned_support_matcher)
        self.hierarchical_action_heads = bool(hierarchical_action_heads)
        self.support_film = bool(support_film)
        self.adaptive_subcarrier_groups = int(adaptive_subcarrier_groups)
        self.latent_support_normalization = bool(latent_support_normalization)
        self.compositional_action_fusion = bool(compositional_action_fusion)
        self.motion_salience_attention = bool(motion_salience_attention)
        if self.compositional_action_fusion and not self.hierarchical_action_heads:
            raise ValueError(
                "compositional action fusion requires hierarchical action heads"
            )
        self.canonicalizer = PhysicsSupportCanonicalizer(
            phase_strength=phase_strength
        )
        self.motion_encoder = MotionShapeEncoder(
            hidden, dropout, use_doppler,
            adaptive_subcarrier_groups=self.adaptive_subcarrier_groups,
        )
        if self.latent_support_normalization:
            self.latent_support_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer(
                "latent_support_gate", torch.zeros(()), persistent=False
            )
        self.motion_progress = (
            MotionProgressEncoder(hidden, motion_phase_bins, dropout)
            if motion_phase_bins >= 2 else None
        )
        self.motion_phase_gate = (
            nn.Parameter(torch.tensor(-2.0))
            if self.motion_progress is not None else None
        )
        self.state_encoder = RelativeStateEncoder(hidden, dropout)
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(hidden * 5), nn.Linear(hidden * 5, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, hidden), nn.LayerNorm(hidden),
        )
        relative_dim = (
            hidden + len(self.prompt_classes) * 4 + C.N_LINKS + C.N_LINKS * 2
        )
        if self.learned_support_matcher:
            optional_rng = torch.random.get_rng_state()
            try:
                self.motion_matcher = LearnedSupportMatcher(hidden, dropout)
                self.state_matcher = LearnedSupportMatcher(hidden, dropout)
                self.motion_matcher_gate = nn.Parameter(torch.tensor(-2.0))
                self.state_matcher_gate = nn.Parameter(torch.tensor(-2.0))
            finally:
                torch.random.set_rng_state(optional_rng)
        else:
            self.motion_matcher = None
            self.state_matcher = None
            self.register_buffer(
                "motion_matcher_gate", torch.tensor(-2.0), persistent=False
            )
            self.register_buffer(
                "state_matcher_gate", torch.tensor(-2.0), persistent=False
            )
        self.embedding = nn.Sequential(
            nn.LayerNorm(relative_dim), nn.Linear(relative_dim, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, hidden), nn.LayerNorm(hidden),
        )
        if self.motion_salience_attention:
            optional_rng = torch.random.get_rng_state()
            try:
                self.motion_salience_projection = nn.Sequential(
                    nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                nn.init.zeros_(self.motion_salience_projection[-1].weight)
                nn.init.zeros_(self.motion_salience_projection[-1].bias)
                self.motion_salience_gate = nn.Parameter(torch.tensor(-2.0))
            finally:
                torch.random.set_rng_state(optional_rng)
        else:
            self.motion_salience_projection = None
            self.register_buffer(
                "motion_salience_gate", torch.tensor(-2.0), persistent=False
            )
        if self.support_film:
            optional_rng = torch.random.get_rng_state()
            try:
                self.support_film_head = nn.Sequential(
                    nn.LayerNorm(hidden * 4),
                    nn.Linear(hidden * 4, width), nn.GELU(),
                    nn.Linear(width, hidden * 2),
                )
                nn.init.zeros_(self.support_film_head[-1].weight)
                nn.init.zeros_(self.support_film_head[-1].bias)
                self.support_film_gate = nn.Parameter(torch.tensor(-2.0))
            finally:
                torch.random.set_rng_state(optional_rng)
        else:
            self.support_film_head = None
            self.register_buffer(
                "support_film_gate", torch.tensor(-2.0), persistent=False
            )
        self.cosine_action = CosineClassifier(hidden, C.N_CLASSES, cosine_scale)
        self.cosine_risk = CosineClassifier(hidden, C.N_RISK, cosine_scale)
        if self.hierarchical_action_heads:
            optional_rng = torch.random.get_rng_state()
            try:
                self.coarse_head = nn.Linear(hidden, 5)
                self.start_head = nn.Linear(hidden, 4)
            finally:
                torch.random.set_rng_state(optional_rng)
        else:
            self.coarse_head = None
            self.start_head = None
        if self.compositional_action_fusion:
            self.composition_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer(
                "composition_gate", torch.zeros(()), persistent=False
            )
        self.register_buffer(
            "coarse_action_target", torch.tensor(COARSE_ACTION_TARGET),
            persistent=False,
        )
        self.register_buffer(
            "start_posture_target", torch.tensor(START_POSTURE_TARGET),
            persistent=False,
        )
        self.risk_fusion = nn.Parameter(torch.tensor(0.0))
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, width // 2), nn.GELU(),
            nn.Linear(width // 2, domains),
        )
        self.pose_motion_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, MOTION_DESCRIPTOR_DIM),
        )

    def model_config(self) -> dict:
        """배포 checkpoint가 support-relative 구조를 동일하게 복원하도록 기록한다."""
        return {
            "hidden": self.hidden,
            "width": self.width,
            "domains": self.domains,
            "dropout": self.dropout,
            "domain_grl": self.domain_grl,
            "prompt_classes": self.prompt_classes,
            "relative_support": True,
            "use_doppler": self.use_doppler,
            "phase_strength": self.phase_strength,
            "cosine_scale": self.cosine_scale,
            "motion_phase_bins": self.motion_phase_bins,
            "learned_support_matcher": self.learned_support_matcher,
            "hierarchical_action_heads": self.hierarchical_action_heads,
            "support_film": self.support_film,
            "adaptive_subcarrier_groups": self.adaptive_subcarrier_groups,
            "latent_support_normalization": self.latent_support_normalization,
            "compositional_action_fusion": self.compositional_action_fusion,
            "motion_salience_attention": self.motion_salience_attention,
            "architecture": "cal20_relative_motion_dg",
        }

    @staticmethod
    def action_to_risk(action_logits: torch.Tensor) -> torch.Tensor:
        """17행동 확률 질량을 safe/warning/danger 세 구간으로 합친다."""
        return torch.stack([
            torch.logsumexp(action_logits[:, start:stop], dim=-1)
            for start, stop in CLASS_RANGES
        ], dim=-1)

    @staticmethod
    def _prototypes(
        embedding: torch.Tensor,
        labels: torch.Tensor,
        classes: tuple[int, ...],
    ) -> torch.Tensor:
        """배포 계약의 각 기본동작 support 중심을 고정된 순서로 만든다."""
        return torch.stack([
            embedding[labels == class_id].mean(0) for class_id in classes
        ])

    def forward(
        self,
        query_csi: torch.Tensor,
        query_link_mask: torch.Tensor,
        support_csi: torch.Tensor,
        support_link_mask: torch.Tensor,
        support_labels: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_link_mask: torch.Tensor,
        calibration_strength: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """절대 반사 대신 target support에 대한 상대 motion/state로 query를 분류한다."""
        prepared = self.canonicalizer.prepare(
            query_csi, query_link_mask,
            support_csi, support_link_mask, support_labels,
            absence_csi, absence_link_mask, calibration_strength,
        )
        support_count = len(support_csi)
        features, frame_mask, link_weight, energy = self.motion_encoder(
            torch.cat((prepared["support_motion"], prepared["query_motion"])),
            torch.cat((prepared["support_mask"], prepared["query_mask"])),
        )
        state_embedding = self.state_encoder(
            torch.cat((prepared["support_state"], prepared["query_state"])),
            torch.cat((prepared["support_mask"], prepared["query_mask"])),
        )
        latent_gate = features.new_zeros(())
        if self.latent_support_normalization:
            # Target support로 encoder 분포를 먼저 맞춘 뒤 temporal summary를 만든다.
            # 이 위치가 뒤로 밀리면 분류 경로에는 정규화가 전혀 반영되지 않는다.
            support_weight = frame_mask[:support_count].to(
                features.dtype
            )[..., None]
            support_count_valid = support_weight.sum((0, 1)).clamp_min(1.0)
            support_mean = (
                features[:support_count] * support_weight
            ).sum((0, 1)) / support_count_valid
            support_variance = (
                (features[:support_count] - support_mean).square()
                * support_weight
            ).sum((0, 1)) / support_count_valid
            normalized_features = (
                (features - support_mean[None, None])
                / support_variance.clamp_min(1e-4).sqrt()[None, None]
            )

            support_state_mean = state_embedding[:support_count].mean(0)
            support_state_scale = state_embedding[:support_count].std(
                0, unbiased=False
            ).clamp_min(1e-2)
            normalized_state = (
                (state_embedding - support_state_mean[None])
                / support_state_scale[None]
            )

            latent_gate = 0.5 * torch.tanh(self.latent_support_gate)
            features = features + latent_gate * (
                normalized_features - features
            )
            features *= frame_mask[..., None].to(features.dtype)
            state_embedding = state_embedding + latent_gate * (
                normalized_state - state_embedding
            )
        summaries = masked_ordered_summary(features, frame_mask)
        motion_embedding = self.motion_projection(torch.cat(summaries, dim=-1))
        salience_strength = features.new_zeros(())
        motion_salience = frame_mask.to(features.dtype)
        if self.motion_salience_projection is not None:
            all_motion = torch.cat((
                prepared["support_motion"], prepared["query_motion"]
            ))
            # residual 이외의 1/3-frame 차분·가속도·phase 차분만 motion으로 본다.
            per_link = all_motion[..., 1:].square().mean((-1, -2)).sqrt()
            salience_link_weight = frame_mask.new_ones(
                per_link.shape, dtype=features.dtype
            ) * torch.cat((
                prepared["support_mask"], prepared["query_mask"]
            )).to(features.dtype)
            salience_energy = (
                per_link * salience_link_weight
            ).sum(-1) / salience_link_weight.sum(
                -1
            ).clamp_min(1.0)
            salience_energy = F.avg_pool1d(
                salience_energy[:, None], kernel_size=5, stride=1, padding=2
            ).squeeze(1)
            masked_energy = salience_energy.masked_fill(~frame_mask, torch.nan)
            center = torch.nanmedian(masked_energy, dim=1).values[:, None]
            deviation = torch.nanmedian(
                (masked_energy - center).abs(), dim=1
            ).values[:, None].clamp_min(1e-4)
            score = (
                (salience_energy - center) / deviation
            ).clamp(-4.0, 8.0)
            score = score.masked_fill(~frame_mask, -1e4)
            motion_salience = torch.softmax(score, dim=1)
            motion_salience = torch.where(
                frame_mask.any(1, keepdim=True), motion_salience,
                torch.zeros_like(motion_salience),
            )
            salient_summary = torch.einsum(
                "bt,bth->bh", motion_salience, features
            )
            salience_strength = torch.sigmoid(self.motion_salience_gate)
            motion_embedding = (
                motion_embedding
                + salience_strength
                * self.motion_salience_projection(salient_summary)
            )
        phase_embedding = (
            self.motion_progress(features, frame_mask)
            if self.motion_progress is not None else None
        )
        if phase_embedding is not None:
            motion_embedding = F.layer_norm(
                motion_embedding
                + torch.sigmoid(self.motion_phase_gate) * phase_embedding,
                (self.hidden,),
            )
        support_motion = motion_embedding[:support_count]
        query_motion = motion_embedding[support_count:]
        support_state = state_embedding[:support_count]
        query_state = state_embedding[support_count:]
        support_energy = energy[:support_count]
        query_energy = energy[support_count:]
        motion_prototypes = self._prototypes(
            support_motion, support_labels, self.prompt_classes
        )
        state_prototypes = self._prototypes(
            support_state, support_labels, self.prompt_classes
        )
        motion_similarity = F.normalize(query_motion, dim=-1) @ F.normalize(
            motion_prototypes, dim=-1
        ).transpose(0, 1)
        state_similarity = F.normalize(query_state, dim=-1) @ F.normalize(
            state_prototypes, dim=-1
        ).transpose(0, 1)
        motion_distance = torch.cdist(
            query_motion, motion_prototypes
        ) / self.hidden ** 0.5
        state_distance = torch.cdist(
            query_state, state_prototypes
        ) / self.hidden ** 0.5
        dynamic_keep = torch.zeros_like(support_labels, dtype=torch.bool)
        for class_id in DYNAMIC_PROMPT_CLASSES:
            dynamic_keep |= support_labels == class_id
        reference_energy = support_energy[dynamic_keep].median(0).values.clamp_min(
            1e-4
        )
        relative_energy = torch.log(
            (query_energy / reference_energy[None]).clamp(0.10, 10.0)
        )
        quality = torch.cat((
            prepared["motion_reliability"], prepared["static_reliability"]
        ))[None].expand(len(query_csi), -1)
        learned_motion_similarity = motion_similarity.new_zeros(
            motion_similarity.shape
        )
        learned_state_similarity = state_similarity.new_zeros(
            state_similarity.shape
        )
        if self.motion_matcher is not None and self.state_matcher is not None:
            learned_motion_similarity = self.motion_matcher(
                query_motion, motion_prototypes
            )
            learned_state_similarity = self.state_matcher(
                query_state, state_prototypes
            )
            motion_similarity = (
                motion_similarity
                + torch.sigmoid(self.motion_matcher_gate)
                * learned_motion_similarity
            )
            state_similarity = (
                state_similarity
                + torch.sigmoid(self.state_matcher_gate)
                * learned_state_similarity
            )
        relative_parts = [
            query_motion, motion_similarity, motion_distance,
            state_similarity, state_distance, relative_energy, quality,
        ]
        embedding = self.embedding(torch.cat(relative_parts, dim=-1))
        base_embedding = embedding
        film_gate = embedding.new_zeros(())
        if self.support_film_head is not None:
            support_context = torch.cat((
                motion_prototypes.mean(0),
                motion_prototypes.std(0, unbiased=False),
                state_prototypes.mean(0),
                state_prototypes.std(0, unbiased=False),
            ), dim=-1)
            film_scale, film_shift = self.support_film_head(
                support_context[None]
            ).chunk(2, dim=-1)
            film_gate = torch.sigmoid(self.support_film_gate)
            embedding = (
                embedding
                + film_gate * (
                    embedding * torch.tanh(film_scale) + film_shift
                )
            )
        coarse_logits = (
            self.coarse_head(embedding)
            if self.coarse_head is not None
            else embedding.new_zeros(len(embedding), 5)
        )
        start_logits = (
            self.start_head(embedding)
            if self.start_head is not None
            else embedding.new_zeros(len(embedding), 4)
        )
        composition_strength = embedding.new_zeros(())
        composition_logits = embedding.new_zeros(len(embedding), C.N_CLASSES)
        if self.compositional_action_fusion:
            composition_logits = (
                coarse_logits.log_softmax(-1)[:, self.coarse_action_target]
                + start_logits.log_softmax(-1)[:, self.start_posture_target]
            )
            composition_logits = composition_logits - composition_logits.mean(
                -1, keepdim=True
            )
            composition_strength = 0.5 * torch.tanh(self.composition_gate)
        action_logits = (
            self.cosine_action(embedding)
            + composition_strength * composition_logits
        )
        direct_risk = self.cosine_risk(embedding)
        action_risk = self.action_to_risk(action_logits)
        fusion = torch.sigmoid(self.risk_fusion)
        risk_logits = (1.0 - fusion) * direct_risk + fusion * action_risk
        base_coarse_logits = (
            self.coarse_head(base_embedding)
            if self.coarse_head is not None
            else base_embedding.new_zeros(len(base_embedding), 5)
        )
        base_start_logits = (
            self.start_head(base_embedding)
            if self.start_head is not None
            else base_embedding.new_zeros(len(base_embedding), 4)
        )
        base_composition_logits = base_embedding.new_zeros(
            len(base_embedding), C.N_CLASSES
        )
        if self.compositional_action_fusion:
            base_composition_logits = (
                base_coarse_logits.log_softmax(-1)[:, self.coarse_action_target]
                + base_start_logits.log_softmax(-1)[:, self.start_posture_target]
            )
            base_composition_logits = (
                base_composition_logits
                - base_composition_logits.mean(-1, keepdim=True)
            )
        base_action_logits = (
            self.cosine_action(base_embedding)
            + composition_strength * base_composition_logits
        )
        base_direct_risk = self.cosine_risk(base_embedding)
        base_action_risk = self.action_to_risk(base_action_logits)
        base_risk_logits = (
            (1.0 - fusion) * base_direct_risk + fusion * base_action_risk
        )
        invariant = _GradientReverse.apply(embedding, self.domain_grl)
        query_features = features[support_count:]
        query_frame_mask = frame_mask[support_count:]
        return {
            "action_logits": action_logits,
            "risk_logits": risk_logits,
            "direct_risk_logits": direct_risk,
            "action_risk_logits": action_risk,
            "base_action_logits": base_action_logits,
            "base_risk_logits": base_risk_logits,
            "base_direct_risk_logits": base_direct_risk,
            "action_residual": action_logits - base_action_logits,
            "risk_residual": direct_risk - base_direct_risk,
            "adapter_gate": film_gate.expand(len(action_logits)),
            "embedding": embedding,
            "domain_logits": self.domain_head(invariant),
            "support_film_gate": film_gate.expand(len(embedding)),
            "latent_support_gate": latent_gate.expand(len(embedding)),
            "composition_strength": composition_strength.expand(len(embedding)),
            "motion_salience_strength": salience_strength.expand(len(embedding)),
            "motion_salience": motion_salience[support_count:],
            "composition_logits": composition_logits,
            "coarse_logits": coarse_logits,
            "start_logits": start_logits,
            "query_features": query_features,
            "query_frame_mask": query_frame_mask,
            "motion_phase_embedding": (
                phase_embedding[support_count:]
                if phase_embedding is not None else query_motion.new_zeros(
                    len(query_motion), 0
                )
            ),
            "query_link_weight": link_weight[support_count:],
            "pose_motion": self.pose_motion_head(query_features),
            "prompt_similarity": torch.cat((
                motion_similarity, state_similarity
            ), dim=-1),
            "prompt_distance": torch.cat((
                motion_distance, state_distance
            ), dim=-1),
            "learned_prompt_similarity": torch.cat((
                learned_motion_similarity, learned_state_similarity,
            ), dim=-1),
            **prepared,
        }
