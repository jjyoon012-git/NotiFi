"""KP v2: 사람·환경 style과 시간순서 행동 content를 분리하는 CSI encoder."""

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
from .cal20 import RelativeStateEncoder
from .meta_calibration import MOTION_PROMPT_CLASSES


COARSE_ACTION_TARGET = (
    0, 1, 1, 1, 2, 2, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4,
)
START_POSTURE_TARGET = (
    0, 1, 2, 3, 3, 1, 1, 2, 1, 0, 0, 3, 1, 0, 3, 3, 2,
)


class ChannelLayerNorm(nn.Module):
    """TCN의 각 프레임을 채널축으로만 정규화해 padding 통계 누출을 막는다."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """[B,C,T]를 [B,T,C]로 바꾸어 시간축과 독립적으로 정규화한다."""
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class LearnedPrototypeMatcher(nn.Module):
    """고정 거리 대신 query와 현장 support의 행동 관계를 학습한다."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, 1),
        )

    def forward(
        self, query: torch.Tensor, prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """각 query와 기본 행동 prototype 쌍에 학습된 유사도를 반환한다."""
        query = F.normalize(query, dim=-1)[:, None]
        prototypes = F.normalize(prototypes, dim=-1)[None]
        query = query.expand(-1, prototypes.shape[1], -1)
        prototypes = prototypes.expand(len(query), -1, -1)
        pair = torch.cat((
            query, prototypes, (query - prototypes).abs(),
            query * prototypes,
        ), dim=-1)
        return self.score(pair).squeeze(-1)


class ScaleFreeMotionEncoder(nn.Module):
    """움직임 크기 style을 떼고 시간·진행률·주파수 content를 함께 인코딩한다."""

    def __init__(
        self,
        hidden: int = 64,
        progress_bins: int = 16,
        frequency_bins: int = 16,
        dropout: float = 0.10,
        use_support_relative_energy: bool = False,
    ):
        super().__init__()
        if progress_bins < 4 or frequency_bins < 4:
            raise ValueError("KP v2 needs at least four progress/frequency bins")
        self.hidden = int(hidden)
        self.progress_bins = int(progress_bins)
        self.frequency_bins = int(frequency_bins)
        self.use_support_relative_energy = bool(use_support_relative_energy)
        self.branch = SubcarrierBranch(6, hidden, dropout)
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
        self.link_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 7 + C.N_LINKS),
            nn.Linear(hidden * 7 + C.N_LINKS, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
        )
        self.local_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden, hidden, kernel_size=5, padding=2 * dilation,
                    dilation=dilation, groups=hidden,
                ),
                nn.Conv1d(hidden, hidden * 2, kernel_size=1),
                nn.GLU(dim=1),
                ChannelLayerNorm(hidden),
                nn.GELU(), nn.Dropout(dropout),
            )
            for dilation in (1, 2, 4, 8)
        ])
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 3,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.token_context = nn.TransformerEncoder(layer, num_layers=2)
        self.token_type = nn.Parameter(torch.zeros(2, progress_bins, hidden))
        nn.init.trunc_normal_(self.token_type, std=0.02)
        self.time_projection = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.frequency_projection = nn.Sequential(
            nn.LayerNorm(hidden * frequency_bins),
            nn.Linear(hidden * frequency_bins, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        style_dimension = C.N_LINKS * 6 * 2 + C.N_LINKS
        self.style_projection = nn.Sequential(
            nn.LayerNorm(style_dimension),
            nn.Linear(style_dimension, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        if self.use_support_relative_energy:
            energy_dimension = C.N_LINKS * 6 * 3
            self.frame_energy_projection = nn.Sequential(
                nn.LayerNorm(C.N_LINKS * 6 + C.N_LINKS),
                nn.Linear(C.N_LINKS * 6 + C.N_LINKS, hidden), nn.GELU(),
            )
            self.energy_projection = nn.Sequential(
                nn.LayerNorm(energy_dimension),
                nn.Linear(energy_dimension, hidden * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
            )
        else:
            self.frame_energy_projection = None
            self.energy_projection = None
        content_inputs = 3 if self.use_support_relative_energy else 2
        self.content_fusion = nn.Sequential(
            nn.LayerNorm(hidden * content_inputs),
            nn.Linear(hidden * content_inputs, hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.register_buffer("centers", torch.linspace(0.0, 1.0, progress_bins))

    @staticmethod
    def _masked_standardize(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """trial·링크·subcarrier·채널별 시간 크기를 style로 빼고 unit motion을 만든다."""
        weight = mask.to(values.dtype)[..., None, None]
        count = weight.sum(1, keepdim=True).clamp_min(1.0)
        mean = (values * weight).sum(1, keepdim=True) / count
        variance = ((values - mean).square() * weight).sum(1, keepdim=True) / count
        scale = variance.clamp_min(1e-4).sqrt()
        normalized = ((values - mean) / scale).clamp(-6.0, 6.0) * weight
        return normalized, mean.squeeze(1), scale.squeeze(1)

    def _soft_pool(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
        coordinate: torch.Tensor,
    ) -> torch.Tensor:
        """연속 시간 좌표를 Gaussian 가중치로 고정 개수 token에 모은다."""
        bandwidth = 0.75 / max(self.progress_bins - 1, 1)
        distance = coordinate[..., None] - self.centers.to(coordinate)[None, None]
        weight = torch.exp(-0.5 * (distance / bandwidth).square())
        weight *= frame_mask.to(weight.dtype)[..., None]
        weight /= weight.sum(1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("btp,bth->bph", weight, features)

    def forward(
        self,
        motion: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """scale-free frame·progress·frequency content와 분리한 style을 반환한다."""
        if motion.ndim != 5 or link_mask.shape != motion.shape[:3]:
            raise ValueError("expected motion [B,T,L,S,6] and mask [B,T,L]")
        normalized, mean, scale = self._masked_standardize(motion, link_mask)
        relative_energy = torch.log1p(
            motion.square().mean(-2).clamp_max(144.0).sqrt()
        )
        relative_energy *= link_mask[..., None].to(relative_energy.dtype)
        energy_weight = link_mask.to(motion.dtype)[..., None]
        link_energy = (
            motion.square().mean(-2) * energy_weight
        ).sum((1, 3)) / (
            energy_weight.sum(1).squeeze(-1) * motion.shape[-1]
        ).clamp_min(1.0)
        link_energy = link_energy.clamp_min(1e-8).sqrt()
        energy_embedding = None
        if self.energy_projection is not None:
            energy_weight = link_mask.to(relative_energy.dtype)[..., None]
            energy_count = energy_weight.sum(1).clamp_min(1.0)
            energy_mean = (
                relative_energy * energy_weight
            ).sum(1) / energy_count
            energy_variance = (
                (relative_energy - energy_mean[:, None]).square()
                * energy_weight
            ).sum(1) / energy_count
            energy_maximum = relative_energy.masked_fill(
                ~link_mask[..., None], -torch.inf
            ).amax(1)
            energy_maximum = torch.where(
                torch.isfinite(energy_maximum),
                energy_maximum, torch.zeros_like(energy_maximum),
            )
            energy_embedding = self.energy_projection(torch.cat((
                energy_mean.flatten(1),
                energy_variance.sqrt().flatten(1),
                energy_maximum.flatten(1),
            ), dim=-1))
        links = self.branch(normalized)
        weight = link_mask.to(links.dtype)[..., None]
        count = weight.sum(1, keepdim=True).clamp_min(1.0)
        temporal_mean = (links * weight).sum(1, keepdim=True) / count
        temporal_variance = (
            (links - temporal_mean).square() * weight
        ).sum(1, keepdim=True) / count
        links = (
            (links - temporal_mean)
            / temporal_variance.clamp_min(1e-4).sqrt()
        ) * weight
        links = links + self.geometry(self.link_geometry).to(links)[None, None]
        score = self.link_gate(links).squeeze(-1).masked_fill(~link_mask, -1e4)
        link_weight = torch.softmax(score, dim=-1)
        link_weight = torch.where(
            link_mask.any(-1, keepdim=True), link_weight,
            torch.zeros_like(link_weight),
        )
        weighted = (links * link_weight[..., None]).sum(-2)
        differences = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            valid = (link_mask[..., left] & link_mask[..., right]).to(links.dtype)
            differences.append(
                (links[..., left, :] - links[..., right, :]) * valid[..., None]
            )
        fused = self.link_fusion(torch.cat((
            weighted,
            (links * weight).flatten(-2),
            torch.cat(differences, dim=-1),
            link_mask.to(links.dtype),
        ), dim=-1))
        if self.frame_energy_projection is not None:
            fused = fused + self.frame_energy_projection(torch.cat((
                relative_energy.flatten(-2),
                link_mask.to(relative_energy.dtype),
            ), dim=-1))
        frame_mask = link_mask.any(-1)
        fused *= frame_mask[..., None].to(fused.dtype)
        sequence = fused.transpose(1, 2)
        for block in self.local_blocks:
            sequence = sequence + block(sequence)
            sequence *= frame_mask[:, None].to(sequence.dtype)
        frame = self.output_norm(sequence.transpose(1, 2))
        frame *= frame_mask[..., None].to(frame.dtype)

        valid_pair = link_mask[:, 1:] & link_mask[:, :-1]
        activity_source = (
            motion if self.use_support_relative_energy else normalized
        )
        activity = activity_source[:, 1:, ..., 1:].square().mean(
            (2, 3, 4)
        ).sqrt()
        activity *= valid_pair.any(-1).to(activity.dtype)
        activity = F.pad(activity, (1, 0))
        activity = activity + 1e-4 * frame_mask.to(activity.dtype)
        progress = activity.cumsum(1)
        progress /= progress[:, -1:].clamp_min(1e-6)
        ordinal = frame_mask.to(frame.dtype).cumsum(1) - 1.0
        clock = ordinal / (frame_mask.sum(1, keepdim=True) - 1).clamp_min(1)
        progress_token = self._soft_pool(frame, frame_mask, progress)
        clock_token = self._soft_pool(frame, frame_mask, clock)
        tokens = torch.cat((
            clock_token + self.token_type[0][None],
            progress_token + self.token_type[1][None],
        ), dim=1)
        contextual = self.token_context(tokens)
        half = self.progress_bins
        time_embedding = self.time_projection(torch.cat((
            contextual[:, :half].mean(1), contextual[:, :half].amax(1),
            contextual[:, half:].mean(1), contextual[:, half:].amax(1),
        ), dim=-1))

        # 진행률로 고정 길이 resampling한 뒤 FFT를 적용해야 뒤쪽 padding 길이가
        # 달라져도 동일한 동작의 주파수 좌표가 변하지 않는다.
        spectrum = torch.fft.rfft(progress_token, dim=1).abs()[
            :, 1:self.frequency_bins + 1
        ]
        if spectrum.shape[1] < self.frequency_bins:
            spectrum = F.pad(
                spectrum, (0, 0, 0, self.frequency_bins - spectrum.shape[1])
            )
        spectrum /= spectrum.sum(1, keepdim=True).clamp_min(1e-6)
        frequency_embedding = self.frequency_projection(spectrum.flatten(1))
        content_parts = [time_embedding, frequency_embedding]
        if energy_embedding is not None:
            content_parts.append(energy_embedding)
        content = self.content_fusion(torch.cat(content_parts, dim=-1))

        style_mean = mean.mean(-2)
        style_scale = scale.mean(-2).clamp_min(1e-5).log()
        coverage = link_mask.to(motion.dtype).mean(1)
        style = self.style_projection(torch.cat((
            style_mean.flatten(1), style_scale.flatten(1), coverage,
        ), dim=-1))
        return {
            "frame": frame,
            "frame_mask": frame_mask,
            "link_weight": link_weight,
            "content": content,
            "time_content": time_embedding,
            "frequency_content": frequency_embedding,
            "energy_content": (
                energy_embedding
                if energy_embedding is not None
                else torch.zeros_like(content)
            ),
            "link_energy": link_energy,
            "style": style,
            "tokens": contextual,
        }


class MotionDescriptorClassifier(nn.Module):
    """CSI가 복원한 신체 움직임 궤적 자체에서 행동과 위험을 다시 판정한다."""

    def __init__(
        self, hidden: int, bins: int = 8, dropout: float = 0.10,
        cosine_scale: float = 10.0,
    ):
        super().__init__()
        self.bins = int(bins)
        self.register_buffer("centers", torch.linspace(0.0, 1.0, bins))
        dimension = MOTION_DESCRIPTOR_DIM * (bins + 3)
        self.projection = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, hidden * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.action = CosineClassifier(hidden, C.N_CLASSES, cosine_scale)
        self.risk = CosineClassifier(hidden, C.N_RISK, cosine_scale)

    def forward(
        self, descriptor: torch.Tensor, valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """통계와 시간순서 bin을 결합해 motion-grounded 분류 logit을 만든다."""
        weight = valid.to(descriptor.dtype)[..., None]
        count = weight.sum(1).clamp_min(1.0)
        mean = (descriptor * weight).sum(1) / count
        variance = (
            (descriptor - mean[:, None]).square() * weight
        ).sum(1) / count
        maximum = descriptor.masked_fill(~valid[..., None], -torch.inf).amax(1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        ordinal = valid.to(descriptor.dtype).cumsum(1) - 1.0
        coordinate = ordinal / (valid.sum(1, keepdim=True) - 1).clamp_min(1)
        bandwidth = 0.75 / max(self.bins - 1, 1)
        distance = coordinate[..., None] - self.centers.to(coordinate)[None, None]
        bin_weight = torch.exp(-0.5 * (distance / bandwidth).square())
        bin_weight *= valid.to(bin_weight.dtype)[..., None]
        bin_weight /= bin_weight.sum(1, keepdim=True).clamp_min(1e-6)
        ordered = torch.einsum("btp,btd->bpd", bin_weight, descriptor)
        embedding = self.projection(torch.cat((
            mean, variance.sqrt(), maximum, ordered.flatten(1),
        ), dim=-1))
        return {
            "embedding": embedding,
            "action_logits": self.action(embedding),
            "risk_logits": self.risk(embedding),
        }


class KPV2ActionPose(nn.Module):
    """행동 content를 주 분류·pose head에, RF style을 domain 진단에만 사용하는 모델."""

    def __init__(
        self,
        hidden: int = 64,
        width: int = 192,
        domains: int = 7,
        dropout: float = 0.10,
        domain_grl: float = 0.20,
        prompt_classes: tuple[int, ...] = MOTION_PROMPT_CLASSES,
        phase_strength: float = 1.0,
        progress_bins: int = 16,
        frequency_bins: int = 16,
        cosine_scale: float = 10.0,
        use_distance_features: bool = True,
        use_motion_classifier: bool = False,
        use_support_relative_energy: bool = False,
        use_learned_support_matcher: bool = False,
        use_explicit_support_energy: bool = False,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.width = int(width)
        self.domains = int(domains)
        self.dropout = float(dropout)
        self.domain_grl = float(domain_grl)
        self.prompt_classes = tuple(int(value) for value in prompt_classes)
        self.phase_strength = float(phase_strength)
        self.progress_bins = int(progress_bins)
        self.frequency_bins = int(frequency_bins)
        self.cosine_scale = float(cosine_scale)
        self.use_distance_features = bool(use_distance_features)
        self.use_motion_classifier = bool(use_motion_classifier)
        self.use_support_relative_energy = bool(use_support_relative_energy)
        self.use_learned_support_matcher = bool(use_learned_support_matcher)
        self.use_explicit_support_energy = bool(use_explicit_support_energy)
        self.relative_support = True
        self.use_doppler = False
        self.motion_phase_bins = 0
        self.motion_phase_gate = None
        self.canonicalizer = PhysicsSupportCanonicalizer(
            phase_strength=phase_strength
        )
        self.motion_encoder = ScaleFreeMotionEncoder(
            hidden, progress_bins, frequency_bins, dropout,
            use_support_relative_energy=use_support_relative_energy,
        )
        self.state_encoder = RelativeStateEncoder(hidden, dropout)
        relative_channels = 4 if self.use_distance_features else 2
        if self.use_learned_support_matcher:
            relative_channels += 2
            self.content_matcher = LearnedPrototypeMatcher(hidden, dropout)
            self.state_matcher = LearnedPrototypeMatcher(hidden, dropout)
        else:
            self.content_matcher = None
            self.state_matcher = None
        relative_dimension = hidden + len(self.prompt_classes) * relative_channels
        if self.use_explicit_support_energy:
            relative_dimension += C.N_LINKS
        self.embedding = nn.Sequential(
            nn.LayerNorm(relative_dimension),
            nn.Linear(relative_dimension, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, hidden),
            nn.LayerNorm(hidden),
        )
        self.action_head = CosineClassifier(hidden, C.N_CLASSES, cosine_scale)
        self.risk_head = CosineClassifier(hidden, C.N_RISK, cosine_scale)
        self.time_action_head = CosineClassifier(
            hidden, C.N_CLASSES, cosine_scale
        )
        self.frequency_action_head = CosineClassifier(
            hidden, C.N_CLASSES, cosine_scale
        )
        self.coarse_head = nn.Linear(hidden, 5)
        self.start_head = nn.Linear(hidden, 4)
        self.risk_fusion = nn.Parameter(torch.tensor(0.0))
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, width // 2), nn.GELU(),
            nn.Linear(width // 2, domains),
        )
        self.style_domain_head = nn.Sequential(
            nn.Linear(hidden, width // 2), nn.GELU(),
            nn.Linear(width // 2, domains),
        )
        self.pose_motion_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, MOTION_DESCRIPTOR_DIM),
        )
        self.motion_classifier = (
            MotionDescriptorClassifier(
                hidden, bins=8, dropout=dropout, cosine_scale=cosine_scale
            )
            if self.use_motion_classifier else None
        )
        if self.use_motion_classifier:
            self.motion_fusion_logit = nn.Parameter(torch.tensor(-0.85))
        else:
            self.register_buffer(
                "motion_fusion_logit", torch.tensor(-0.85), persistent=False
            )
        self.register_buffer(
            "coarse_action_target", torch.tensor(COARSE_ACTION_TARGET)
        )
        self.register_buffer(
            "start_posture_target", torch.tensor(START_POSTURE_TARGET)
        )

    def model_config(self) -> dict:
        """checkpoint가 v2 content/style 구조를 정확히 복원하도록 설정을 기록한다."""
        return {
            "architecture": "kpv2_action_pose",
            "hidden": self.hidden,
            "width": self.width,
            "domains": self.domains,
            "dropout": self.dropout,
            "domain_grl": self.domain_grl,
            "prompt_classes": self.prompt_classes,
            "phase_strength": self.phase_strength,
            "progress_bins": self.progress_bins,
            "frequency_bins": self.frequency_bins,
            "cosine_scale": self.cosine_scale,
            "use_distance_features": self.use_distance_features,
            "use_motion_classifier": self.use_motion_classifier,
            "use_support_relative_energy": self.use_support_relative_energy,
            "use_learned_support_matcher": self.use_learned_support_matcher,
            "use_explicit_support_energy": self.use_explicit_support_energy,
        }

    @staticmethod
    def action_to_risk(action_logits: torch.Tensor) -> torch.Tensor:
        """17-way 행동 확률 질량을 safe/warning/danger로 합친다."""
        return torch.stack([
            torch.logsumexp(action_logits[:, start:stop], dim=-1)
            for start, stop in CLASS_RANGES
        ], dim=-1)

    @staticmethod
    def _prototypes(
        values: torch.Tensor,
        labels: torch.Tensor,
        classes: tuple[int, ...],
    ) -> torch.Tensor:
        """현장 기본동작의 content 또는 state 중심을 고정 순서로 만든다."""
        return torch.stack([
            values[labels == class_id].mean(0) for class_id in classes
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
        """target support로 보정한 뒤 query의 행동·위험·pose motion을 예측한다."""
        prepared = self.canonicalizer.prepare(
            query_csi, query_link_mask,
            support_csi, support_link_mask, support_labels,
            absence_csi, absence_link_mask, calibration_strength,
        )
        support_count = len(support_csi)
        motion = self.motion_encoder(
            torch.cat((prepared["support_motion"], prepared["query_motion"])),
            torch.cat((prepared["support_mask"], prepared["query_mask"])),
        )
        state = self.state_encoder(
            torch.cat((prepared["support_state"], prepared["query_state"])),
            torch.cat((prepared["support_mask"], prepared["query_mask"])),
        )
        support_content = motion["content"][:support_count]
        query_content = motion["content"][support_count:]
        support_state = state[:support_count]
        query_state = state[support_count:]
        support_link_energy = motion["link_energy"][:support_count]
        query_link_energy = motion["link_energy"][support_count:]
        content_prototype = self._prototypes(
            support_content, support_labels, self.prompt_classes
        )
        state_prototype = self._prototypes(
            support_state, support_labels, self.prompt_classes
        )
        content_similarity = F.normalize(query_content, dim=-1) @ F.normalize(
            content_prototype, dim=-1
        ).transpose(0, 1)
        state_similarity = F.normalize(query_state, dim=-1) @ F.normalize(
            state_prototype, dim=-1
        ).transpose(0, 1)
        content_distance = torch.cdist(
            query_content, content_prototype
        ) / self.hidden ** 0.5
        state_distance = torch.cdist(
            query_state, state_prototype
        ) / self.hidden ** 0.5
        relative_features = [content_similarity, state_similarity]
        if self.use_distance_features:
            relative_features.extend((content_distance, state_distance))
        learned_content_similarity = content_similarity.new_zeros(
            content_similarity.shape
        )
        learned_state_similarity = state_similarity.new_zeros(
            state_similarity.shape
        )
        if self.content_matcher is not None and self.state_matcher is not None:
            learned_content_similarity = self.content_matcher(
                query_content, content_prototype
            )
            learned_state_similarity = self.state_matcher(
                query_state, state_prototype
            )
            relative_features.extend((
                learned_content_similarity, learned_state_similarity,
            ))
        support_relative_energy = query_link_energy.new_zeros(
            len(query_link_energy), C.N_LINKS
        )
        if self.use_explicit_support_energy:
            dynamic_keep = torch.zeros_like(
                support_labels, dtype=torch.bool
            )
            for class_id in DYNAMIC_PROMPT_CLASSES:
                dynamic_keep |= support_labels == class_id
            reference_energy = support_link_energy[
                dynamic_keep
            ].median(0).values.clamp_min(1e-4)
            support_relative_energy = torch.log(
                (query_link_energy / reference_energy[None]).clamp(0.10, 10.0)
            )
            relative_features.append(support_relative_energy)
        embedding = self.embedding(torch.cat((
            query_content, *relative_features,
        ), dim=-1))
        content_action = self.action_head(embedding)
        content_risk = self.risk_head(embedding)
        query_frame = motion["frame"][support_count:]
        query_frame_mask = motion["frame_mask"][support_count:]
        pose_motion = self.pose_motion_head(query_frame)
        if self.motion_classifier is not None:
            # 분류 gradient로 descriptor를 임의 코드로 바꾸지 못하게 한다.
            # pose-motion head는 실제 GVHMR motion grounding으로만 학습된다.
            motion_classification = self.motion_classifier(
                pose_motion.detach(), query_frame_mask
            )
            motion_fusion = torch.sigmoid(self.motion_fusion_logit)
            action_logits = (
                (1.0 - motion_fusion) * content_action
                + motion_fusion * motion_classification["action_logits"]
            )
            direct_risk = (
                (1.0 - motion_fusion) * content_risk
                + motion_fusion * motion_classification["risk_logits"]
            )
        else:
            motion_classification = {
                "embedding": torch.zeros_like(embedding),
                "action_logits": content_action,
                "risk_logits": content_risk,
            }
            action_logits = content_action
            direct_risk = content_risk
            motion_fusion = action_logits.new_zeros(())
        action_risk = self.action_to_risk(action_logits)
        fusion = torch.sigmoid(self.risk_fusion)
        risk_logits = (1.0 - fusion) * direct_risk + fusion * action_risk
        zeros_action = torch.zeros_like(action_logits)
        zeros_risk = torch.zeros_like(direct_risk)
        return {
            "action_logits": action_logits,
            "content_action_logits": content_action,
            "motion_action_logits": motion_classification["action_logits"],
            "risk_logits": risk_logits,
            "direct_risk_logits": direct_risk,
            "content_risk_logits": content_risk,
            "motion_risk_logits": motion_classification["risk_logits"],
            "action_risk_logits": action_risk,
            "base_action_logits": action_logits,
            "base_risk_logits": risk_logits,
            "base_direct_risk_logits": direct_risk,
            "action_residual": zeros_action,
            "risk_residual": zeros_risk,
            "adapter_gate": action_logits.new_zeros(len(action_logits)),
            "embedding": embedding,
            "content_embedding": query_content,
            "style_embedding": motion["style"][support_count:],
            "motion_embedding": motion_classification["embedding"],
            "motion_fusion": motion_fusion.expand(len(query_csi)),
            "time_action_logits": self.time_action_head(
                motion["time_content"][support_count:]
            ),
            "frequency_action_logits": self.frequency_action_head(
                motion["frequency_content"][support_count:]
            ),
            "coarse_logits": self.coarse_head(embedding),
            "start_logits": self.start_head(embedding),
            "domain_logits": self.domain_head(
                _GradientReverse.apply(query_content, self.domain_grl)
            ),
            "style_domain_logits": self.style_domain_head(
                motion["style"][support_count:]
            ),
            "query_features": query_frame,
            "query_frame_mask": query_frame_mask,
            "query_link_weight": motion["link_weight"][support_count:],
            "progress_tokens": motion["tokens"][support_count:],
            "pose_motion": pose_motion,
            "prompt_similarity": torch.cat((
                content_similarity, state_similarity,
            ), dim=-1),
            "prompt_distance": torch.cat((
                content_distance, state_distance,
            ), dim=-1),
            "learned_prompt_similarity": torch.cat((
                learned_content_similarity, learned_state_similarity,
            ), dim=-1),
            "support_relative_energy": support_relative_energy,
            **prepared,
        }
