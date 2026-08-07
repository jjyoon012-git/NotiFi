"""CAL12: 정적 반사와 동적 행동을 분리하는 안전한 CSI calibration 모델."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C
from .meta_calibration import (
    MOTION_PROMPT_CLASSES,
    SupportBaselineCanonicalizer,
    masked_ordered_summary,
)
from .doppler_pose import DopplerFilterBank


STATIC_PROMPT_CLASSES = (1, 2, 3)
DYNAMIC_PROMPT_CLASSES = (0, 4, 5, 7, 8)
CLASS_RANGES = ((0, 9), (9, 12), (12, 17))


class _GradientReverse(torch.autograd.Function):
    """행동 embedding에서 source site 정보를 제거하도록 gradient 방향을 뒤집는다."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def circular_delta(values: torch.Tensor, lag: int) -> torch.Tensor:
    """위상 wrap 경계를 가로질러도 튀지 않는 시간 차분을 계산한다."""
    output = torch.zeros_like(values)
    difference = values[:, lag:] - values[:, :-lag]
    output[:, lag:] = torch.atan2(torch.sin(difference), torch.cos(difference))
    return output / torch.pi


def masked_time_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """링크별 유효 frame만 사용해 시간 평균을 계산한다."""
    weight = mask.to(values.dtype)[..., None]
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def masked_link_energy(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """trial마다 링크별 절대 운동 에너지를 계산한다."""
    weight = mask.to(values.dtype)[..., None]
    count = weight.expand_as(values).sum((1, 3)).clamp_min(1.0)
    return (values.abs() * weight).sum((1, 3)) / count


class PhysicsSupportCanonicalizer(nn.Module):
    """absence와 기본 동작 support로 정적 반사와 링크 감도를 따로 보정한다."""

    def __init__(
        self,
        minimum_motion_scale: float = 0.05,
        maximum_scale_ratio: float = 2.0,
        phase_strength: float = 1.0,
    ):
        super().__init__()
        self.minimum_motion_scale = float(minimum_motion_scale)
        self.maximum_scale_ratio = float(maximum_scale_ratio)
        self.phase_strength = float(phase_strength)
        self.baseline = SupportBaselineCanonicalizer()
        self.register_buffer("source_amp_scale", torch.ones(C.N_LINKS))
        self.register_buffer("source_phase_scale", torch.ones(C.N_LINKS))
        self.register_buffer("source_initialized", torch.zeros(1, dtype=torch.bool))

    def support_motion_scale(
        self,
        support: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """안전한 전환 동작에서 링크별 진폭·위상 운동 감도와 신뢰도를 추정한다."""
        keep = torch.zeros_like(support_labels, dtype=torch.bool)
        for class_id in DYNAMIC_PROMPT_CLASSES:
            keep |= support_labels == class_id
        if not bool(keep.any()):
            one = support.new_ones(C.N_LINKS)
            return one, one, support.new_zeros(C.N_LINKS)

        values = support[keep]
        mask = support_mask[keep]
        delta_mask = mask[:, 1:] & mask[:, :-1]
        amp_delta = values[:, 1:, ..., 0] - values[:, :-1, ..., 0]
        phase_delta = circular_delta(values[..., 1] * torch.pi, 1)[:, 1:]
        amp_trial = masked_link_energy(amp_delta, delta_mask)
        phase_trial = masked_link_energy(phase_delta, delta_mask)
        amp_scale = amp_trial.median(0).values.clamp_min(self.minimum_motion_scale)
        phase_scale = phase_trial.median(0).values.clamp_min(
            self.minimum_motion_scale
        )

        amp_spread = (amp_trial - amp_scale).abs().median(0).values / amp_scale
        phase_spread = (
            (phase_trial - phase_scale).abs().median(0).values / phase_scale
        )
        coverage = mask.to(values.dtype).mean((0, 1))
        reliability = (
            coverage * torch.exp(-0.5 * (amp_spread + phase_spread))
        ).clamp(0.0, 1.0)
        return amp_scale, phase_scale, reliability

    def support_static_anchor(
        self,
        support: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """서기·앉기·눕기 support로 체형과 점유 상태의 중심·척도를 추정한다."""
        prototypes = []
        consistencies = []
        for class_id in STATIC_PROMPT_CLASSES:
            keep = support_labels == class_id
            if not bool(keep.any()):
                continue
            trial = masked_time_mean(support[keep, ..., 0], support_mask[keep])
            prototypes.append(trial.mean(0))
            consistencies.append(trial.std(0, unbiased=False).mean(-1))
        if not prototypes:
            center = support.new_zeros(C.N_LINKS, C.N_LIVE_SUBCARRIERS)
            scale = support.new_ones(C.N_LINKS, C.N_LIVE_SUBCARRIERS)
            return center, scale, support.new_zeros(C.N_LINKS)
        stacked = torch.stack(prototypes)
        center = stacked.mean(0)
        scale = stacked.std(0, unbiased=False).clamp_min(0.25)
        consistency = torch.stack(consistencies).mean(0)
        reliability = torch.exp(-consistency).clamp(0.0, 1.0)
        return center, scale, reliability

    @torch.no_grad()
    def update_source_scale(
        self,
        amp_scale: torch.Tensor,
        phase_scale: torch.Tensor,
        momentum: float = 0.98,
    ) -> None:
        """학습 source support의 평균 링크 감도를 안전 fallback 기준으로 누적한다."""
        if not bool(self.source_initialized):
            self.source_amp_scale.copy_(amp_scale.detach())
            self.source_phase_scale.copy_(phase_scale.detach())
            self.source_initialized.fill_(True)
            return
        self.source_amp_scale.lerp_(amp_scale.detach(), 1.0 - momentum)
        self.source_phase_scale.lerp_(phase_scale.detach(), 1.0 - momentum)

    def _safe_scale(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        reliability: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """비정상 support scale을 source 범위로 제한하고 신뢰도만큼만 적용한다."""
        source = source.to(target).clamp_min(self.minimum_motion_scale)
        ratio = (target / source).clamp(
            1.0 / self.maximum_scale_ratio, self.maximum_scale_ratio
        )
        amount = reliability * strength
        return source * torch.exp(amount * torch.log(ratio))

    def prepare(
        self,
        query_csi: torch.Tensor,
        query_mask: torch.Tensor,
        support_csi: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_mask: torch.Tensor,
        calibration_strength: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        """query/support를 state 3채널과 motion 6채널로 변환한다."""
        combined = torch.cat((support_csi, query_csi), dim=0)
        combined_mask = torch.cat((support_mask, query_mask), dim=0)
        canonical, combined_mask, baseline = self.baseline(
            combined, combined_mask, absence_csi, absence_mask
        )
        support_count = len(support_csi)
        support = canonical[:support_count]
        query = canonical[support_count:]
        canonical_support_mask = combined_mask[:support_count]
        canonical_query_mask = combined_mask[support_count:]
        strength = torch.as_tensor(
            calibration_strength, dtype=query.dtype, device=query.device
        ).clamp(0.0, 1.0)

        amp_scale, phase_scale, motion_reliability = self.support_motion_scale(
            support, canonical_support_mask, support_labels
        )
        source_amp = self.source_amp_scale if bool(self.source_initialized) else amp_scale
        source_phase = (
            self.source_phase_scale if bool(self.source_initialized) else phase_scale
        )
        safe_amp = self._safe_scale(
            amp_scale, source_amp, motion_reliability, strength
        )
        safe_phase = self._safe_scale(
            phase_scale, source_phase, motion_reliability, strength
        )
        static_center, static_scale, static_reliability = self.support_static_anchor(
            support, canonical_support_mask, support_labels
        )
        static_amount = static_reliability[:, None] * strength

        def transform(
            values: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            amplitude = values[..., 0]
            phase = values[..., 1] * torch.pi
            state_amplitude = amplitude + static_amount[None, None] * (
                (amplitude - static_center[None, None])
                / static_scale[None, None]
                - amplitude
            )
            state = torch.stack((
                state_amplitude.clamp(-8.0, 8.0),
                torch.sin(phase), torch.cos(phase),
            ), dim=-1)

            temporal_center = masked_time_mean(amplitude, mask)
            residual = amplitude - temporal_center[:, None]
            delta1 = torch.zeros_like(amplitude)
            delta3 = torch.zeros_like(amplitude)
            delta1[:, 1:] = amplitude[:, 1:] - amplitude[:, :-1]
            delta3[:, 3:] = amplitude[:, 3:] - amplitude[:, :-3]
            acceleration = torch.zeros_like(amplitude)
            acceleration[:, 2:] = delta1[:, 2:] - delta1[:, 1:-1]
            phase1 = circular_delta(phase, 1)
            phase3 = circular_delta(phase, 3)
            amp_denominator = safe_amp[None, None, :, None]
            phase_denominator = safe_phase[None, None, :, None]
            motion = torch.stack((
                residual / amp_denominator,
                delta1 / amp_denominator,
                delta3 / amp_denominator,
                acceleration / amp_denominator,
                self.phase_strength * phase1 / phase_denominator,
                self.phase_strength * phase3 / phase_denominator,
            ), dim=-1).clamp(-12.0, 12.0)
            weight = mask[..., None, None].to(values.dtype)
            return state * weight, motion * weight

        support_state, support_motion = transform(
            support, canonical_support_mask
        )
        query_state, query_motion = transform(query, canonical_query_mask)
        return {
            "query_state": query_state,
            "query_motion": query_motion,
            "query_mask": canonical_query_mask,
            "support_state": support_state,
            "support_motion": support_motion,
            "support_mask": canonical_support_mask,
            "amp_scale": amp_scale,
            "phase_scale": phase_scale,
            "safe_amp_scale": safe_amp,
            "safe_phase_scale": safe_phase,
            "motion_reliability": motion_reliability,
            "static_reliability": static_reliability,
            "baseline_valid_links": baseline["valid_links"],
        }


class SubcarrierBranch(nn.Module):
    """한 frame의 subcarrier 패턴을 링크별 작은 latent로 압축한다."""

    def __init__(self, channels: int, hidden: int, dropout: float):
        super().__init__()
        spectral = max(24, hidden // 2)
        groups = 8 if spectral % 8 == 0 else 1
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Conv1d(channels, spectral, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(groups, spectral),
            nn.GELU(),
            nn.Conv1d(spectral, spectral, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, spectral),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(spectral * 2, hidden), nn.LayerNorm(hidden)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """[B,T,L,S,C]를 [B,T,L,H]로 변환한다."""
        batch, frames, links, subcarriers, channels = values.shape
        flat = values.reshape(
            batch * frames * links, subcarriers, channels
        ).transpose(1, 2)
        pooled = []
        for chunk in torch.split(flat, 4096):
            encoded = self.network(chunk)
            pooled.append(torch.cat((encoded.mean(-1), encoded.amax(-1)), dim=-1))
        return self.projection(torch.cat(pooled)).reshape(
            batch, frames, links, self.hidden
        )


class CrossDomainMixStyle(nn.Module):
    """초기 feature 통계를 섞어 source에 없던 RF style을 학습 중 합성한다."""

    def __init__(self, probability: float = 0.5, alpha: float = 0.3):
        super().__init__()
        self.probability = float(probability)
        self.alpha = float(alpha)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """시간축 평균·표준편차만 섞고 정규화된 동작 내용은 보존한다."""
        if not self.training or len(values) < 2:
            return values
        if float(torch.rand((), device=values.device)) >= self.probability:
            return values
        weight = mask.to(values.dtype)[..., None]
        count = weight.sum(1, keepdim=True).clamp_min(1.0)
        mean = (values * weight).sum(1, keepdim=True) / count
        variance = ((values - mean).square() * weight).sum(1, keepdim=True) / count
        standard = variance.clamp_min(1e-6).sqrt()
        normalized = (values - mean) / standard
        permutation = torch.randperm(len(values), device=values.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        mixing = beta.sample((len(values), 1, 1, 1)).to(values.device, values.dtype)
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_standard = mixing * standard + (1.0 - mixing) * standard[permutation]
        return (normalized * mixed_standard + mixed_mean) * weight


class DualPathCSIEncoder(nn.Module):
    """정적 자세와 동적 움직임을 별도 경로로 인코딩한 뒤 링크 방향을 결합한다."""

    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.10,
        use_doppler: bool = False,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.state_branch = SubcarrierBranch(3, hidden, dropout)
        self.motion_branch = SubcarrierBranch(6, hidden, dropout)
        self.use_doppler = bool(use_doppler)
        self.doppler = (
            DopplerFilterBank(hidden, hidden, dropout)
            if self.use_doppler else None
        )
        self.mix_style = CrossDomainMixStyle(probability=0.5, alpha=0.3)
        self.register_buffer(
            "link_geometry", torch.tensor(C.LINK_GEOMETRY, dtype=torch.float32)
        )
        self.geometry = nn.Sequential(
            nn.Linear(self.link_geometry.shape[1], hidden), nn.Tanh()
        )
        self.link_projection = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU()
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
        blocks = []
        for dilation in (1, 2, 4, 8):
            blocks.append(nn.Sequential(
                nn.Conv1d(
                    hidden, hidden, kernel_size=5, padding=2 * dilation,
                    dilation=dilation, groups=hidden,
                ),
                nn.Conv1d(hidden, hidden * 2, kernel_size=1),
                nn.GLU(dim=1),
                nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
                nn.GELU(), nn.Dropout(dropout),
            ))
        self.temporal = nn.ModuleList(blocks)
        self.output_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        state: torch.Tensor,
        motion: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """두 경로와 세 링크의 방향 차이를 multi-scale 시계열로 인코딩한다."""
        state_feature = self.state_branch(state)
        motion_feature = self.motion_branch(motion)
        if self.doppler is not None:
            motion_feature = motion_feature + self.doppler(motion_feature)
        motion_feature = self.mix_style(motion_feature, link_mask)
        links = self.link_projection(torch.cat((state_feature, motion_feature), dim=-1))
        links = links + self.geometry(self.link_geometry).to(links)[None, None]
        score = self.link_gate(links).squeeze(-1).masked_fill(~link_mask, -1e4)
        link_weight = torch.softmax(score, dim=-1)
        link_weight = torch.where(
            link_mask.any(-1, keepdim=True), link_weight,
            torch.zeros_like(link_weight),
        )
        weighted = (links * link_weight[..., None]).sum(-2)
        masked_links = links * link_mask[..., None].to(links.dtype)
        differences = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            valid = (link_mask[..., left] & link_mask[..., right]).to(links.dtype)
            differences.append(
                (links[..., left, :] - links[..., right, :]) * valid[..., None]
            )
        fused = self.fusion(torch.cat((
            weighted,
            masked_links.flatten(-2),
            torch.cat(differences, dim=-1),
            link_mask.to(links.dtype),
        ), dim=-1))
        frame_mask = link_mask.any(-1)
        values = fused.transpose(1, 2)
        for block in self.temporal:
            values = values + block(values)
            values = values * frame_mask[:, None].to(values.dtype)
        output = self.output_norm(values.transpose(1, 2))
        output *= frame_mask[..., None].to(output.dtype)
        return output, frame_mask, link_weight


class CAL12PhysicsDG(nn.Module):
    """물리 canonicalization과 domain-generalized 분류 head를 결합한다."""

    def __init__(
        self,
        hidden: int = 64,
        width: int = 192,
        domains: int = 7,
        dropout: float = 0.10,
        domain_grl: float = 0.20,
        prompt_classes: Iterable[int] = MOTION_PROMPT_CLASSES,
        relative_support: bool = True,
        use_doppler: bool = False,
        phase_strength: float = 1.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.width = int(width)
        self.domains = int(domains)
        self.dropout = float(dropout)
        self.domain_grl = float(domain_grl)
        self.prompt_classes = tuple(int(value) for value in prompt_classes)
        self.relative_support = bool(relative_support)
        self.use_doppler = bool(use_doppler)
        self.phase_strength = float(phase_strength)
        self.canonicalizer = PhysicsSupportCanonicalizer(
            phase_strength=phase_strength
        )
        self.encoder = DualPathCSIEncoder(
            hidden=hidden, dropout=dropout, use_doppler=use_doppler
        )
        summary_dim = hidden * 5
        self.embedding = nn.Sequential(
            nn.LayerNorm(summary_dim), nn.Linear(summary_dim, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, hidden), nn.LayerNorm(hidden),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, C.N_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, width // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width // 2, C.N_RISK),
        )
        relative_dim = hidden + len(self.prompt_classes) * 2 + C.N_LINKS * 2
        self.relative_context = nn.Sequential(
            nn.LayerNorm(relative_dim), nn.Linear(relative_dim, width), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_residual = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, C.N_CLASSES)
        )
        self.risk_residual = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, C.N_RISK)
        )
        self.calibration_gate = nn.Sequential(
            nn.Linear(width, width // 4), nn.GELU(), nn.Linear(width // 4, 1)
        )
        nn.init.zeros_(self.action_residual[-1].weight)
        nn.init.zeros_(self.action_residual[-1].bias)
        nn.init.zeros_(self.risk_residual[-1].weight)
        nn.init.zeros_(self.risk_residual[-1].bias)
        nn.init.zeros_(self.calibration_gate[-1].weight)
        nn.init.constant_(self.calibration_gate[-1].bias, -1.5)
        self.risk_fusion = nn.Parameter(torch.tensor(0.0))
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, width // 2), nn.GELU(),
            nn.Linear(width // 2, domains),
        )

    def model_config(self) -> dict:
        """checkpoint에서 동일한 CAL12 구조를 복원할 설정을 반환한다."""
        return {
            "hidden": self.hidden,
            "width": self.width,
            "domains": self.domains,
            "dropout": self.dropout,
            "domain_grl": self.domain_grl,
            "prompt_classes": self.prompt_classes,
            "relative_support": self.relative_support,
            "use_doppler": self.use_doppler,
            "phase_strength": self.phase_strength,
        }

    @staticmethod
    def action_to_risk(action_logits: torch.Tensor) -> torch.Tensor:
        """17행동 확률을 safe/warning/danger 세 그룹 logit으로 합친다."""
        return torch.stack([
            torch.logsumexp(action_logits[:, start:stop], dim=-1)
            for start, stop in CLASS_RANGES
        ], dim=-1)

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
        """현장 support로 query를 보정하고 17행동·3위험을 함께 예측한다."""
        prepared = self.canonicalizer.prepare(
            query_csi, query_link_mask,
            support_csi, support_link_mask, support_labels,
            absence_csi, absence_link_mask, calibration_strength,
        )
        if self.relative_support:
            support_count = len(support_csi)
            combined_feature, combined_frame_mask, combined_link_weight = self.encoder(
                torch.cat((prepared["support_state"], prepared["query_state"])),
                torch.cat((prepared["support_motion"], prepared["query_motion"])),
                torch.cat((prepared["support_mask"], prepared["query_mask"])),
            )
            support_feature = combined_feature[:support_count]
            support_frame_mask = combined_frame_mask[:support_count]
            query_feature = combined_feature[support_count:]
            query_frame_mask = combined_frame_mask[support_count:]
            query_link_weight = combined_link_weight[support_count:]
            support_summaries = masked_ordered_summary(
                support_feature, support_frame_mask
            )
            support_embedding = self.embedding(torch.cat(support_summaries, dim=-1))
            prompt_prototypes = torch.stack([
                support_embedding[support_labels == class_id].mean(0)
                for class_id in self.prompt_classes
            ])
        else:
            query_feature, query_frame_mask, query_link_weight = self.encoder(
                prepared["query_state"], prepared["query_motion"],
                prepared["query_mask"],
            )
            prompt_prototypes = query_feature.new_zeros(
                len(self.prompt_classes), self.hidden
            )
        query_summaries = masked_ordered_summary(query_feature, query_frame_mask)
        embedding = self.embedding(torch.cat(query_summaries, dim=-1))
        normalized_query = F.normalize(embedding, dim=-1)
        normalized_prompt = F.normalize(prompt_prototypes, dim=-1)
        prompt_similarity = normalized_query @ normalized_prompt.transpose(0, 1)
        prompt_distance = torch.cdist(
            embedding, prompt_prototypes, p=2
        ) / self.hidden ** 0.5
        quality = torch.cat((
            prepared["motion_reliability"], prepared["static_reliability"]
        ))
        quality = quality[None].expand(len(query_csi), -1)
        relative_context = self.relative_context(torch.cat((
            embedding, prompt_similarity, prompt_distance, quality
        ), dim=-1))
        raw_gate = self.calibration_gate(relative_context).squeeze(-1)
        support_quality = quality.mean(-1)
        strength = torch.as_tensor(
            calibration_strength, dtype=embedding.dtype, device=embedding.device
        ).clamp(0.0, 1.0)
        gate = (
            0.50 * torch.sigmoid(raw_gate) * support_quality * strength
            if self.relative_support else torch.zeros_like(raw_gate)
        )

        base_action_logits = self.action_head(embedding)
        base_direct_risk = self.risk_head(embedding)
        action_residual = self.action_residual(relative_context)
        direct_risk_residual = self.risk_residual(relative_context)
        action_logits = base_action_logits + gate[:, None] * action_residual
        direct_risk = base_direct_risk + gate[:, None] * direct_risk_residual
        action_risk = self.action_to_risk(action_logits)
        base_action_risk = self.action_to_risk(base_action_logits)
        fusion = torch.sigmoid(self.risk_fusion)
        risk_logits = (1.0 - fusion) * direct_risk + fusion * action_risk
        base_risk_logits = (
            (1.0 - fusion) * base_direct_risk + fusion * base_action_risk
        )
        invariant = _GradientReverse.apply(embedding, self.domain_grl)
        return {
            "action_logits": action_logits,
            "risk_logits": risk_logits,
            "direct_risk_logits": direct_risk,
            "action_risk_logits": action_risk,
            "base_action_logits": base_action_logits,
            "base_risk_logits": base_risk_logits,
            "base_direct_risk_logits": base_direct_risk,
            "action_residual": action_residual,
            "risk_residual": direct_risk_residual,
            "prompt_similarity": prompt_similarity,
            "prompt_distance": prompt_distance,
            "prompt_prototypes": prompt_prototypes,
            "embedding": embedding,
            "domain_logits": self.domain_head(invariant),
            "query_features": query_feature,
            "query_frame_mask": query_frame_mask,
            "query_link_weight": query_link_weight,
            "adapter_gate": gate,
            **prepared,
        }


def cross_site_supervised_contrastive(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """다른 site의 같은 행동만 positive로 사용해 사람·환경 공통 동작을 묶는다."""
    normalized = F.normalize(embedding, dim=-1)
    logits = normalized @ normalized.transpose(0, 1) / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (
        (labels[:, None] == labels[None])
        & (domains[:, None] != domains[None])
        & ~identity
    )
    valid = positive.any(1)
    if not bool(valid.any()):
        return embedding.sum() * 0.0
    logits = logits - logits.max(1, keepdim=True).values.detach()
    denominator = torch.logsumexp(logits.masked_fill(identity, -1e4), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
    return -(numerator[valid] - denominator[valid]).mean()
