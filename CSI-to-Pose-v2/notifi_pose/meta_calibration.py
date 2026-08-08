"""기본 동작 CSI로 신규 사용자·환경에 적응하는 support-conditioned 보정기."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C


PROMPT_CLASSES = (0, 1, 2, 3)
MOTION_PROMPT_CLASSES = C.CALIBRATION_PROMPT_CLASSES


class SupportBaselineCanonicalizer(nn.Module):
    """빈방 support만으로 링크별 진폭과 위상의 설치 지문을 제거한다."""

    def __init__(self, minimum_amplitude_scale: float = 0.05):
        super().__init__()
        self.minimum_amplitude_scale = float(minimum_amplitude_scale)

    def estimate(
        self,
        absence_csi: torch.Tensor,
        absence_link_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """두 개 이상의 빈방 trial에서 배포 환경의 기준 진폭과 위상을 계산한다."""
        if absence_csi.ndim != 5 or absence_csi.shape[-1] != 2:
            raise ValueError("expected absence CSI [A,T,L,S,2]")
        if absence_link_mask.shape != absence_csi.shape[:3]:
            raise ValueError("absence link mask shape does not match CSI")
        weight = absence_link_mask.to(absence_csi.dtype)[..., None]
        count = weight.sum((0, 1)).clamp_min(1.0)
        amplitude = torch.log1p(absence_csi[..., 0].clamp_min(0.0))
        amplitude_mean = (amplitude * weight).sum((0, 1)) / count
        amplitude_variance = (
            (amplitude - amplitude_mean[None, None]).square() * weight
        ).sum((0, 1)) / count
        amplitude_scale = amplitude_variance.clamp_min(
            self.minimum_amplitude_scale ** 2
        ).sqrt()

        phase = absence_csi[..., 1]
        sine = (torch.sin(phase) * weight).sum((0, 1)) / count
        cosine = (torch.cos(phase) * weight).sum((0, 1)) / count
        phase_mean = torch.atan2(sine, cosine)
        valid_links = absence_link_mask.any(1).any(0)
        return {
            "amplitude_mean": amplitude_mean,
            "amplitude_scale": amplitude_scale,
            "phase_mean": phase_mean,
            "valid_links": valid_links,
        }

    @staticmethod
    def apply(
        csi: torch.Tensor,
        link_mask: torch.Tensor,
        baseline: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """빈방 기준선을 실제 CSI에 적용하고 사용할 수 없는 링크를 끈다."""
        if link_mask.shape != csi.shape[:3]:
            raise ValueError("link mask shape does not match CSI")
        amplitude = (
            torch.log1p(csi[..., 0].clamp_min(0.0))
            - baseline["amplitude_mean"][None, None]
        ) / baseline["amplitude_scale"][None, None]
        phase_delta = csi[..., 1] - baseline["phase_mean"][None, None]
        phase = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)) / torch.pi
        mask = link_mask & baseline["valid_links"][None, None]
        canonical = torch.stack((amplitude, phase), dim=-1)
        canonical = canonical * mask[..., None, None].to(canonical.dtype)
        return canonical, mask

    def forward(
        self,
        csi: torch.Tensor,
        link_mask: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """현재 환경의 빈방 통계로 CSI를 canonical 공간에 투영한다."""
        baseline = self.estimate(absence_csi, absence_link_mask)
        canonical, mask = self.apply(csi, link_mask, baseline)
        return canonical, mask, baseline


class LinkAwareCSIEncoder(nn.Module):
    """Raw CSI에서 TX별 공간 통계와 시간 변화를 분리해 인코딩한다."""

    def __init__(
        self,
        hidden: int = 64,
        temporal_layers: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.temporal_layers = int(temporal_layers)
        self.dropout = float(dropout)
        spectral = max(32, hidden // 2)
        groups = 8 if spectral % 8 == 0 else 1
        self.spectral = nn.Sequential(
            nn.Conv1d(6, spectral, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(groups, spectral),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(spectral, spectral, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, spectral),
            nn.GELU(),
        )
        self.spectral_projection = nn.Sequential(
            nn.Linear(spectral * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.link_embedding = nn.Parameter(torch.zeros(C.N_LINKS, hidden))
        nn.init.normal_(self.link_embedding, std=0.02)
        self.register_buffer(
            "link_geometry", torch.tensor(C.LINK_GEOMETRY, dtype=torch.float32)
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(self.link_geometry.shape[1], hidden), nn.Tanh()
        )
        self.link_gate = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.directional_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 6 + C.N_LINKS),
            nn.Linear(hidden * 6 + C.N_LINKS, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        blocks = []
        for layer in range(temporal_layers):
            dilation = 2 ** layer
            blocks.append(nn.Sequential(
                nn.Conv1d(
                    hidden, hidden, kernel_size=5,
                    padding=2 * dilation, dilation=dilation,
                    groups=hidden,
                ),
                nn.Conv1d(hidden, hidden * 2, kernel_size=1),
                nn.GLU(dim=1),
                nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.temporal = nn.ModuleList(blocks)
        self.output_norm = nn.LayerNorm(hidden)

    def encode_subcarriers(self, csi: torch.Tensor) -> torch.Tensor:
        """원 CSI와 단기 차분을 함께 보며 subcarrier 국소 패턴을 보존한다."""
        if csi.ndim != 5 or csi.shape[-1] != 2:
            raise ValueError("expected CSI [B,T,L,S,2]")
        delta1 = torch.zeros_like(csi)
        delta3 = torch.zeros_like(csi)
        delta1[:, 1:] = csi[:, 1:] - csi[:, :-1]
        delta3[:, 3:] = csi[:, 3:] - csi[:, :-3]
        values = torch.cat((csi, delta1, delta3), dim=-1)
        batch, frames, links, subcarriers = values.shape[:4]
        flat = values.reshape(batch * frames * links, subcarriers, 6).transpose(1, 2)
        pooled = []
        for chunk in torch.split(flat, 4096):
            encoded = self.spectral(chunk)
            pooled.append(torch.cat((encoded.mean(-1), encoded.amax(-1)), dim=-1))
        return self.spectral_projection(torch.cat(pooled)).reshape(
            batch, frames, links, self.hidden
        )

    def forward(
        self,
        csi: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TX별 token을 신뢰도 가중 결합하고 다중 시간 범위로 인코딩한다."""
        if link_mask.shape != csi.shape[:3]:
            raise ValueError("link mask shape does not match CSI")
        links = self.encode_subcarriers(csi)
        geometry = self.geometry_projection(self.link_geometry).to(links.dtype)
        links = links + self.link_embedding[None, None] + geometry[None, None]
        score = self.link_gate(links).squeeze(-1).masked_fill(~link_mask, -1e4)
        weight = torch.softmax(score, dim=-1)
        weight = torch.where(
            link_mask.any(-1, keepdim=True), weight, torch.zeros_like(weight)
        )
        masked_links = links * link_mask[..., None].to(links.dtype)
        differences = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            valid = (link_mask[..., left] & link_mask[..., right]).to(links.dtype)
            differences.append(
                (links[..., left, :] - links[..., right, :]) * valid[..., None]
            )
        directional = torch.cat((
            masked_links.flatten(-2), torch.cat(differences, dim=-1),
            link_mask.to(links.dtype),
        ), dim=-1)
        fused = self.directional_fusion(directional)
        frame_mask = link_mask.any(-1)
        values = fused.transpose(1, 2)
        for block in self.temporal:
            values = values + block(values)
            values = values * frame_mask[:, None].to(values.dtype)
        output = self.output_norm(values.transpose(1, 2))
        output = output * frame_mask[..., None].to(output.dtype)
        return output, frame_mask, weight


class RawSupportConditionedModel(nn.Module):
    """KP4 feature 없이 raw CSI와 현장 support만으로 행동·위험 특징을 만든다."""

    def __init__(
        self,
        hidden: int = 64,
        token_dim: int = 96,
        width: int = 192,
        domains: int = 6,
        dropout: float = 0.10,
        prompt_classes: Iterable[int] = PROMPT_CLASSES,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.token_dim = int(token_dim)
        self.width = int(width)
        self.domains = int(domains)
        self.dropout = float(dropout)
        self.prompt_classes = tuple(int(value) for value in prompt_classes)
        self.canonicalizer = SupportBaselineCanonicalizer()
        self.encoder = LinkAwareCSIEncoder(
            hidden=hidden, temporal_layers=4, dropout=dropout
        )
        self.calibrator = SupportConditionedCalibrator(
            feature_dim=hidden,
            token_dim=token_dim,
            width=width,
            heads=4,
            layers=2,
            domains=domains,
            domain_grl=0.20,
            max_log_gain=0.25,
            max_bias=0.50,
            prompt_classes=self.prompt_classes,
            ordered_summary=True,
        )

    def model_config(self) -> dict:
        """배포 checkpoint에서 raw calibration 모델을 재구성할 설정을 반환한다."""
        return {
            "hidden": self.hidden,
            "token_dim": self.token_dim,
            "width": self.width,
            "domains": self.domains,
            "dropout": self.dropout,
            "prompt_classes": self.prompt_classes,
        }

    def forward(
        self,
        query_csi: torch.Tensor,
        query_link_mask: torch.Tensor,
        support_csi: torch.Tensor,
        support_link_mask: torch.Tensor,
        support_labels: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_link_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """같은 현장의 support와 query를 공유 encoder로 처리해 예측한다."""
        combined_csi = torch.cat((support_csi, query_csi), dim=0)
        combined_mask = torch.cat((support_link_mask, query_link_mask), dim=0)
        combined_csi, combined_mask, baseline = self.canonicalizer(
            combined_csi, combined_mask, absence_csi, absence_link_mask
        )
        encoded, frame_mask, link_weight = self.encoder(
            combined_csi, combined_mask
        )
        support_count = len(support_csi)
        output = self.calibrator(
            encoded[support_count:], frame_mask[support_count:],
            encoded[:support_count], frame_mask[:support_count],
            support_labels,
        )
        output["query_link_weight"] = link_weight[support_count:]
        output["support_link_weight"] = link_weight[:support_count]
        output["baseline_valid_links"] = baseline["valid_links"]
        return output


class _GradientReverse(torch.autograd.Function):
    """행동 특징에서 사이트 정보를 제거하기 위해 domain gradient를 반전한다."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def masked_moments(
    features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """유효 프레임의 평균·표준편차·시간 변화량을 계산한다."""
    if features.ndim != 3 or frame_mask.shape != features.shape[:2]:
        raise ValueError("expected features [N,T,D] and frame mask [N,T]")
    weight = frame_mask.to(features.dtype)[..., None]
    count = weight.sum(1).clamp_min(1.0)
    mean = (features * weight).sum(1) / count
    variance = ((features - mean[:, None]).square() * weight).sum(1) / count
    standard_deviation = variance.clamp_min(1e-8).sqrt()
    delta_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
    delta_weight = delta_mask.to(features.dtype)[..., None]
    delta = (features[:, 1:] - features[:, :-1]).abs()
    motion = (delta * delta_weight).sum(1) / delta_weight.sum(1).clamp_min(1.0)
    return mean, standard_deviation, motion


def masked_ordered_summary(
    features: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """평균 통계에 peak와 시간 방향성을 더해 동작 순서를 보존한다."""
    mean, standard_deviation, motion = masked_moments(features, frame_mask)
    centered = features - mean[:, None]
    peak = centered.abs().masked_fill(~frame_mask[..., None], 0.0).amax(1)
    timeline = torch.linspace(
        -1.0, 1.0, features.shape[1], device=features.device,
        dtype=features.dtype,
    )
    time_weight = frame_mask.to(features.dtype) * timeline[None]
    denominator = time_weight.square().sum(1).clamp_min(1.0)
    trend = (centered * time_weight[..., None]).sum(1) / denominator[:, None]
    return mean, standard_deviation, motion, peak, trend


class SupportConditionedCalibrator(nn.Module):
    """환경 기준선 제거 후 기본 동작별 token으로 query 특징을 보정한다."""

    def __init__(
        self,
        feature_dim: int = 128,
        token_dim: int = 128,
        width: int = 256,
        heads: int = 4,
        layers: int = 2,
        domains: int = 7,
        domain_grl: float = 0.15,
        max_log_gain: float = 0.30,
        max_bias: float = 0.75,
        prompt_classes: Iterable[int] = PROMPT_CLASSES,
        ordered_summary: bool = False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.width = int(width)
        self.heads = int(heads)
        self.layers = int(layers)
        self.domains = int(domains)
        self.domain_grl = float(domain_grl)
        self.max_log_gain = float(max_log_gain)
        self.max_bias = float(max_bias)
        self.prompt_classes = tuple(int(value) for value in prompt_classes)
        self.ordered_summary = bool(ordered_summary)
        if not self.prompt_classes:
            raise ValueError("at least one calibration prompt is required")

        self.register_buffer(
            "reference_prompt", torch.zeros(len(self.prompt_classes), feature_dim)
        )
        self.register_buffer("reference_initialized", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("reference_center", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.prompt_embedding = nn.Embedding(C.N_CLASSES, token_dim)
        summary_width = 5 if self.ordered_summary else 3
        self.support_input = nn.Sequential(
            nn.LayerNorm(feature_dim * summary_width),
            nn.Linear(feature_dim * summary_width, width),
            nn.GELU(),
            nn.Linear(width, token_dim),
        )
        support_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=width * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.support_context = nn.TransformerEncoder(
            support_layer, num_layers=layers, norm=nn.LayerNorm(token_dim)
        )
        self.query_input = nn.Sequential(
            nn.LayerNorm(feature_dim * summary_width),
            nn.Linear(feature_dim * summary_width, width),
            nn.GELU(),
            nn.Linear(width, token_dim),
        )
        self.query_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=0.10, batch_first=True
        )
        context_dim = token_dim * 3
        self.adapter = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, width),
            nn.GELU(),
            nn.Linear(width, feature_dim * 2 + 1),
        )
        self.calibrated_pool = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, token_dim), nn.GELU()
        )
        prediction_dim = context_dim + token_dim
        self.action_head = nn.Sequential(
            nn.LayerNorm(prediction_dim),
            nn.Linear(prediction_dim, width),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(width, C.N_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(prediction_dim),
            nn.Linear(prediction_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, C.N_RISK),
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, domains),
        )

    def set_reference(
        self,
        prompt_prototypes: torch.Tensor,
        center: torch.Tensor,
        feature_scale: torch.Tensor,
    ) -> None:
        """source prompt prototype과 특징 척도를 checkpoint buffer에 고정한다."""
        if prompt_prototypes.shape != self.reference_prompt.shape:
            raise ValueError("prompt prototype shape mismatch")
        if center.shape != self.reference_center.shape:
            raise ValueError("reference center shape mismatch")
        if feature_scale.shape != self.feature_scale.shape:
            raise ValueError("feature scale shape mismatch")
        self.reference_prompt.copy_(prompt_prototypes.detach())
        self.reference_center.copy_(center.detach())
        self.feature_scale.copy_(feature_scale.detach().clamp_min(1e-3))
        self.reference_initialized.fill_(True)

    @torch.no_grad()
    def update_reference(
        self,
        prompt_prototypes: torch.Tensor,
        momentum: float = 0.995,
    ) -> None:
        """학습 source 사이트의 기본동작 prototype을 EMA 기준점으로 누적한다."""
        if prompt_prototypes.shape != self.reference_prompt.shape:
            raise ValueError("prompt prototype shape mismatch")
        if not bool(self.reference_initialized):
            self.reference_prompt.copy_(prompt_prototypes.detach())
            self.reference_initialized.fill_(True)
        else:
            self.reference_prompt.lerp_(prompt_prototypes.detach(), 1.0 - momentum)
        self.reference_center.copy_(self.reference_prompt.mean(0))
        self.feature_scale.copy_(
            self.reference_prompt.std(0, unbiased=False).clamp_min(0.10)
        )

    def model_config(self) -> dict:
        """checkpoint에서 동일한 구조를 다시 만들기 위한 설정을 반환한다."""
        return {
            "feature_dim": self.feature_dim,
            "token_dim": self.token_dim,
            "width": self.width,
            "heads": self.heads,
            "layers": self.layers,
            "domains": self.domains,
            "domain_grl": self.domain_grl,
            "max_log_gain": self.max_log_gain,
            "max_bias": self.max_bias,
            "prompt_classes": self.prompt_classes,
            "ordered_summary": self.ordered_summary,
        }

    def summarize(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """모델 설정에 맞는 trial 시계열 요약을 계산한다."""
        if self.ordered_summary:
            return masked_ordered_summary(features, frame_mask)
        return masked_moments(features, frame_mask)

    def encode_support(
        self,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """동작별 support trial을 순서와 무관한 네 개의 calibration token으로 만든다."""
        summaries = self.summarize(support_features, support_mask)
        tokens = []
        for prompt_index, class_id in enumerate(self.prompt_classes):
            keep = support_labels == class_id
            if not bool(keep.any()):
                raise ValueError(f"missing calibration prompt class {class_id}")
            components = [summary[keep].mean(0) for summary in summaries]
            components[0] = components[0] - self.reference_prompt[prompt_index]
            descriptor = torch.cat(components)
            token = self.support_input(descriptor)
            token = token + self.prompt_embedding.weight[class_id]
            tokens.append(token)
        return self.support_context(torch.stack(tokens)[None]).squeeze(0)

    def prompt_means(
        self,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """입력 순서와 무관하게 네 기본동작의 latent 중심을 계산한다."""
        mean = masked_moments(support_features, support_mask)[0]
        return torch.stack([
            mean[support_labels == class_id].mean(0)
            for class_id in self.prompt_classes
        ])

    def align_from_prompts(
        self,
        query_features: torch.Tensor,
        target_prompts: torch.Tensor,
    ) -> torch.Tensor:
        """네 기본동작 anchor의 중심과 척도로 target latent를 source 공간에 정렬한다."""
        if not bool(self.reference_initialized):
            return query_features
        target_center = target_prompts.mean(0)
        target_scale = target_prompts.std(0, unbiased=False).clamp_min(0.10)
        source_center = self.reference_prompt.mean(0)
        source_scale = self.reference_prompt.std(0, unbiased=False).clamp_min(0.10)
        gain = (source_scale / target_scale).clamp(0.67, 1.50)
        aligned = source_center[None, None] + (
            query_features - target_center[None, None]
        ) * gain[None, None]
        return query_features + 0.50 * (aligned - query_features)

    def forward(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """support token을 이용해 query를 보정하고 행동·위험도를 함께 예측한다."""
        support_tokens = self.encode_support(
            support_features, support_mask, support_labels
        )
        target_prompts = self.prompt_means(
            support_features, support_mask, support_labels
        )
        anchor_aligned = self.align_from_prompts(query_features, target_prompts)
        summaries = list(self.summarize(anchor_aligned, query_mask))
        summaries[0] = summaries[0] - self.reference_center
        query_token = self.query_input(torch.cat(summaries, dim=-1))
        expanded_support = support_tokens[None].expand(len(query_features), -1, -1)
        attended, attention = self.query_attention(
            query_token[:, None], expanded_support, expanded_support,
            need_weights=True,
        )
        global_support = expanded_support.mean(1)
        context = torch.cat((
            query_token, attended.squeeze(1), global_support
        ), dim=-1)
        parameters = self.adapter(context)
        log_gain, normalized_bias, raw_gate = torch.split(
            parameters, (self.feature_dim, self.feature_dim, 1), dim=-1
        )
        gain = torch.exp(self.max_log_gain * torch.tanh(log_gain))
        bias = (
            self.max_bias * torch.tanh(normalized_bias)
            * self.feature_scale[None]
        )
        gate = torch.sigmoid(raw_gate)
        centered = anchor_aligned - self.reference_center[None, None]
        transformed = self.reference_center[None, None] + centered * gain[:, None]
        transformed = transformed + bias[:, None]
        calibrated = anchor_aligned + gate[:, None] * (transformed - anchor_aligned)
        calibrated = torch.where(
            query_mask[..., None], calibrated, anchor_aligned
        )
        calibrated_mean, _, _ = masked_moments(calibrated, query_mask)
        prediction = torch.cat((context, self.calibrated_pool(calibrated_mean)), dim=-1)
        invariant = _GradientReverse.apply(calibrated_mean, self.domain_grl)
        return {
            "calibrated_features": calibrated,
            "action_logits": self.action_head(prediction),
            "risk_logits": self.risk_head(prediction),
            "domain_logits": self.domain_head(invariant),
            "support_attention": attention.squeeze(1),
            "adapter_gate": gate.squeeze(-1),
            "calibrated_mean": calibrated_mean,
            "support_prompt_means": target_prompts,
        }


def prototype_alignment_loss(
    calibrated_mean: torch.Tensor,
    class_id: torch.Tensor,
    class_prototypes: torch.Tensor,
    feature_scale: torch.Tensor,
) -> torch.Tensor:
    """보정 특징을 동일 행동의 source 공통 prototype에 정렬한다."""
    target = class_prototypes.index_select(0, class_id)
    normalized = (calibrated_mean - target) / feature_scale.clamp_min(1e-3)
    return F.smooth_l1_loss(normalized, torch.zeros_like(normalized), beta=0.5)
