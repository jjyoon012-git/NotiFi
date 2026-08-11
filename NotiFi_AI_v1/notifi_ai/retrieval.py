"""Minimal train-bank motion retrieval runtime used by NotiFi AI v1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import ACTION_LABELS, N_JOINTS, RISK_LABELS, TARGET_FPS


N_CLASSES = len(ACTION_LABELS)
N_RISK = len(RISK_LABELS)
CACHE_FRAMES = 304
UP_AXIS = 1
JOINT_GROUPS = {
    "head": (12, 15),
    "torso": (0, 3, 6, 9, 13, 14),
    "left_arm": (16, 18, 20),
    "right_arm": (17, 19, 21),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}


class LocalTemporalBlock(nn.Module):
    """Apply a residual depthwise temporal convolution."""

    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            5,
            padding=2 * dilation,
            dilation=dilation,
            groups=hidden,
        )
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(values).transpose(1, 2)
        hidden = self.pointwise(F.gelu(self.depthwise(hidden))).transpose(1, 2)
        return values + self.dropout(hidden)


def masked_temporal_bins(
    features: torch.Tensor,
    mask: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average variable-length frame features into stable temporal bins."""

    weight = mask[:, None].to(features.dtype)
    numerator = F.adaptive_avg_pool1d(features.transpose(1, 2) * weight, bins)
    denominator = F.adaptive_avg_pool1d(weight, bins)
    pooled = numerator / denominator.clamp_min(1e-5)
    return pooled.transpose(1, 2), denominator.squeeze(1) > 0.0


class TemporalMotionSelector(nn.Module):
    """Predict motion embeddings and semantic logits from CSI features."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        width: int = 192,
        bins: int = 38,
        layers: int = 2,
        heads: int = 6,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.bins = int(bins)
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.position = nn.Parameter(torch.zeros(1, bins, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            block,
            num_layers=layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, embedding_dim),
        )
        self.action_head = nn.Linear(width * 2, N_CLASSES)
        self.risk_head = nn.Linear(width * 2, N_RISK)

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> dict:
        values, valid = masked_temporal_bins(features, frame_mask, self.bins)
        values = self.input(values) + self.position
        values = self.temporal(values, src_key_padding_mask=~valid)
        score = self.attention(values).squeeze(-1).masked_fill(~valid, -1e4)
        attention = torch.softmax(score, dim=-1)
        attended = (values * attention[..., None]).sum(1)
        weight = valid[..., None].to(values.dtype)
        mean = (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled = torch.cat((attended, mean), dim=-1)
        return {
            "motion_embedding": self.embedding_head(pooled),
            "action_logits": self.action_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "temporal_attention": attention,
            "pooled_features": pooled,
        }


class CandidateMotionReranker(nn.Module):
    """Score train-bank motion candidates against one CSI observation."""

    def __init__(
        self,
        query_dim: int,
        embedding_dim: int,
        class_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.class_embedding = nn.Embedding(N_CLASSES, class_dim)
        input_dim = query_dim + embedding_dim * 4 + class_dim + N_RISK + 2
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        query_embedding: torch.Tensor,
        candidate_embedding: torch.Tensor,
        candidate_class: torch.Tensor,
        risk_probability: torch.Tensor,
        retrieval_score: torch.Tensor,
        action_log_probability: torch.Tensor,
    ) -> torch.Tensor:
        candidates = candidate_embedding.shape[1]
        query = query_features[:, None].expand(-1, candidates, -1)
        embedding = query_embedding[:, None].expand(-1, candidates, -1)
        risk = risk_probability[:, None].expand(-1, candidates, -1)
        values = torch.cat(
            (
                query,
                embedding,
                candidate_embedding,
                (embedding - candidate_embedding).abs(),
                embedding * candidate_embedding,
                self.class_embedding(candidate_class),
                risk,
                retrieval_score[..., None],
                action_log_probability[..., None],
            ),
            dim=-1,
        )
        return self.score(values).squeeze(-1)


class ProfileCandidateRanker(nn.Module):
    """Rank motions using pose and anatomical motion-profile evidence."""

    def __init__(
        self,
        feature_dim: int = 8,
        class_dim: int = 16,
        hidden: int = 64,
        dropout: float = 0.10,
        context_dim: int = 0,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.class_embedding = nn.Embedding(N_CLASSES, class_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim + class_dim + self.context_dim),
            nn.Linear(feature_dim + class_dim + self.context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        class_id: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = [features, self.class_embedding(class_id)]
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError(f"Expected context dimension {self.context_dim}")
            values.append(context)
        return self.score(torch.cat(values, dim=-1)).squeeze(-1)


class MotionProfileHead(nn.Module):
    """Predict framewise whole-body speed from frozen CSI features."""

    def __init__(self, input_dim: int, width: int = 128, dropout: float = 0.10):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.speed = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
            nn.Softplus(),
        )
        self.motion = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> dict:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        return {
            "speed": self.speed(values).squeeze(-1) * frame_mask,
            "motion_logits": self.motion(values).squeeze(-1),
            "profile_features": values,
        }


class PartMotionProfileHead(nn.Module):
    """Predict framewise speed for six anatomical body regions."""

    def __init__(
        self,
        input_dim: int,
        parts: int = 6,
        width: int = 160,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.parts = int(parts)
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            LocalTemporalBlock(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        )
        self.part_speed = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, self.parts),
            nn.Softplus(),
        )
        self.part_motion = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, self.parts)
        )

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> dict:
        values = self.input(features)
        weight = frame_mask[..., None].to(values.dtype)
        for block in self.temporal:
            values = block(values) * weight
        return {
            "part_speed": self.part_speed(values) * weight,
            "part_motion_logits": self.part_motion(values),
            "profile_features": values,
        }


def profile_features(cache: dict, indices: torch.Tensor | None = None) -> torch.Tensor:
    """Build the motion-profile input used during training."""

    def take(value: torch.Tensor) -> torch.Tensor:
        return value if indices is None else value.index_select(0, indices)

    return torch.cat(
        (
            take(cache["features"]).float(),
            take(cache["motion_activity"]).float()[..., None],
            torch.softmax(take(cache["phase_logits"]).float(), dim=-1),
        ),
        dim=-1,
    )


@torch.no_grad()
def predict_selector(
    model: TemporalMotionSelector,
    cache: dict,
    batch_size: int,
    device: str,
) -> dict:
    """Run a temporal selector in bounded batches."""

    model.eval()
    embeddings, actions, risks, pooled = [], [], [], []
    for start in range(0, len(cache["features"]), batch_size):
        stop = min(start + batch_size, len(cache["features"]))
        output = model(
            cache["features"][start:stop].to(device).float(),
            cache["frame_mask"][start:stop].to(device),
        )
        embeddings.append(output["motion_embedding"].float().cpu())
        actions.append(output["action_logits"].float().cpu())
        risks.append(output["risk_logits"].float().cpu())
        pooled.append(output["pooled_features"].float().cpu())
    return {
        "embedding": torch.cat(embeddings),
        "action_logits": torch.cat(actions),
        "risk_logits": torch.cat(risks),
        "pooled_features": torch.cat(pooled),
    }


def canonicalize(pose: torch.Tensor, valid: torch.Tensor, frames: int) -> torch.Tensor:
    """Resample one valid trajectory to normalized action phase."""

    selected = pose[valid]
    if len(selected) == 0:
        return pose.new_zeros(frames, N_JOINTS, 3)
    if len(selected) == 1:
        return selected.expand(frames, -1, -1).clone()
    result = F.interpolate(
        selected.flatten(1).T[None], size=frames, mode="linear", align_corners=True
    )[0].T
    return result.reshape(frames, N_JOINTS, 3)


def render(canonical: torch.Tensor, valid: torch.Tensor, frames: int) -> torch.Tensor:
    """Render a normalized trajectory on the observed valid frame span."""

    output = canonical.new_zeros(frames, N_JOINTS, 3)
    positions = torch.nonzero(valid, as_tuple=False).flatten()
    if len(positions) == 0:
        return output
    if len(positions) == 1:
        output[positions] = canonical[0]
        return output
    values = F.interpolate(
        canonical.flatten(1).T[None],
        size=len(positions),
        mode="linear",
        align_corners=True,
    )[0].T.reshape(len(positions), N_JOINTS, 3)
    output[positions] = values
    return output


def model_inputs(
    pool: dict,
    selector_output: dict,
    checkpoint: dict,
    risk_probability: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Assemble candidate-reranker tensors."""

    candidate_indices = pool["indices"].index_select(0, indices)
    return (
        selector_output["pooled_features"].index_select(0, indices),
        selector_output["embedding"].index_select(0, indices),
        checkpoint["train_embedding"][candidate_indices],
        checkpoint["train_class"][candidate_indices],
        risk_probability.index_select(0, indices),
        pool["retrieval_score"].index_select(0, indices),
        pool["action_log_probability"].index_select(0, indices),
    )


@torch.no_grad()
def predict_profile(model, cache: dict, valid: torch.Tensor, device: str) -> torch.Tensor:
    """Predict whole-body speed profiles."""

    values = []
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(
            model(
                profile_features(cache, indices).to(device),
                valid.index_select(0, indices).to(device),
            )["speed"].float().cpu()
        )
    return torch.cat(values)


@torch.no_grad()
def predict_part_profile(
    model,
    cache: dict,
    valid: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Predict six body-region speed profiles."""

    values = []
    model.eval()
    for start in range(0, len(valid), 64):
        indices = torch.arange(start, min(start + 64, len(valid)))
        values.append(
            model(
                profile_features(cache, indices).to(device),
                valid.index_select(0, indices).to(device),
            )["part_speed"].float().cpu()
        )
    return torch.cat(values)


def candidate_speed_profiles(
    train_bank: torch.Tensor,
    pool: dict,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Measure framewise speed for candidate trajectories."""

    profiles = []
    for item, valid in enumerate(target_valid):
        poses = torch.stack(
            [render(train_bank[int(index)], valid, CACHE_FRAMES) for index in pool["indices"][item]]
        )
        speed = torch.zeros(poses.shape[:2])
        speed[:, 1:] = torch.linalg.vector_norm(
            poses[:, 1:] - poses[:, :-1], dim=-1
        ).mean(-1) * TARGET_FPS
        profiles.append(speed)
    return torch.stack(profiles)


def candidate_part_speed_profiles(
    train_bank: torch.Tensor,
    pool: dict,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Measure framewise speed for six body regions of each candidate."""

    profiles = []
    for item, valid in enumerate(target_valid):
        poses = torch.stack(
            [render(train_bank[int(index)], valid, CACHE_FRAMES) for index in pool["indices"][item]]
        )
        velocity = torch.zeros_like(poses)
        velocity[:, 1:] = (poses[:, 1:] - poses[:, :-1]) * TARGET_FPS
        parts = [
            torch.linalg.vector_norm(velocity[:, :, list(joints)], dim=-1).mean(-1)
            for joints in JOINT_GROUPS.values()
        ]
        profiles.append(torch.stack(parts, dim=-1))
    return torch.stack(profiles)


def standardize(values: torch.Tensor) -> torch.Tensor:
    """Normalize the candidate axis independently for each query."""

    return (values - values.mean(-1, keepdim=True)) / values.std(
        -1, keepdim=True
    ).clamp_min(1e-5)


def profile_distance(
    predicted: torch.Tensor,
    candidates: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Compare scalar speed profiles by shape, scale, and peak timing."""

    mask = valid[:, None]
    predicted = F.avg_pool1d(predicted[:, None], 9, stride=1, padding=4)[:, 0]
    candidates = F.avg_pool1d(
        candidates.flatten(0, 1)[:, None], 9, stride=1, padding=4
    )[:, 0].reshape_as(candidates)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    query_mean = (predicted[:, None] * mask).sum(-1, keepdim=True) / count
    candidate_mean = (candidates * mask).sum(-1, keepdim=True) / count
    query_centered = (predicted[:, None] - query_mean) * mask
    candidate_centered = (candidates - candidate_mean) * mask
    correlation = (query_centered * candidate_centered).sum(-1) / (
        torch.linalg.vector_norm(query_centered, dim=-1)
        * torch.linalg.vector_norm(candidate_centered, dim=-1)
    ).clamp_min(1e-6)
    log_error = (
        torch.log1p(candidates) - torch.log1p(predicted[:, None])
    ).abs()
    log_error = (log_error * mask).sum(-1) / count.squeeze(-1)
    query_peak = predicted.masked_fill(~valid, -1).argmax(-1)
    candidate_peak = candidates.masked_fill(~mask, -1).argmax(-1)
    peak = (candidate_peak - query_peak[:, None]).abs() / count.squeeze(-1)
    return (1.0 - correlation) + 0.50 * log_error + 0.20 * peak


def part_profile_distance(
    predicted: torch.Tensor,
    candidates: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Compare body-region speed profiles independently."""

    batch, choices, frames, parts = candidates.shape
    predicted = F.avg_pool1d(
        predicted.permute(0, 2, 1).reshape(batch * parts, 1, frames),
        9,
        stride=1,
        padding=4,
    ).reshape(batch, parts, frames).permute(0, 2, 1)
    candidates = F.avg_pool1d(
        candidates.permute(0, 1, 3, 2).reshape(batch * choices * parts, 1, frames),
        9,
        stride=1,
        padding=4,
    ).reshape(batch, choices, parts, frames).permute(0, 1, 3, 2)
    mask = valid[:, None, :, None]
    count = mask.sum(2).clamp_min(1)
    query_mean = (predicted[:, None] * mask).sum(2) / count
    candidate_mean = (candidates * mask).sum(2) / count
    query_centered = (predicted[:, None] - query_mean[:, :, None]) * mask
    candidate_centered = (candidates - candidate_mean[:, :, None]) * mask
    correlation = (query_centered * candidate_centered).sum(2) / (
        torch.linalg.vector_norm(query_centered, dim=2)
        * torch.linalg.vector_norm(candidate_centered, dim=2)
    ).clamp_min(1e-6)
    log_error = (
        torch.log1p(candidates) - torch.log1p(predicted[:, None])
    ).abs()
    log_error = (log_error * mask).sum(2) / count
    query_peak = predicted.masked_fill(~valid[..., None], -1).argmax(1)
    candidate_peak = candidates.masked_fill(~mask, -1).argmax(2)
    peak = (candidate_peak - query_peak[:, None]).abs() / count
    return (1.0 - correlation) + 0.45 * log_error + 0.15 * peak


def weighted_part_distance(values: torch.Tensor) -> torch.Tensor:
    """Combine anatomical profile errors with limb-heavy weights."""

    weights = values.new_tensor((0.8, 0.6, 1.7, 1.7, 1.7, 1.7))
    weights = weights / weights.sum()
    return standardize((values * weights).sum(-1))


def bank_profiles(train_bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute scalar and body-region speed profiles for the bank."""

    velocity = torch.zeros_like(train_bank)
    velocity[:, 1:] = (train_bank[:, 1:] - train_bank[:, :-1]) * TARGET_FPS
    scalar = torch.linalg.vector_norm(velocity, dim=-1).mean(-1)
    parts = [
        torch.linalg.vector_norm(velocity[:, :, list(joints)], dim=-1).mean(-1)
        for joints in JOINT_GROUPS.values()
    ]
    return scalar, torch.stack(parts, dim=-1)


def query_motion_context(
    data: dict,
    item: int,
    valid: torch.Tensor,
    action_probability: torch.Tensor,
) -> torch.Tensor:
    """Summarize the CSI-predicted dynamics of one query."""

    scalar = data["predicted_scalar_profile"][item][valid]
    part = data["predicted_part_profile"][item][valid]
    if len(scalar) == 0:
        scalar = data["predicted_scalar_profile"].new_zeros(1)
        part = data["predicted_part_profile"].new_zeros(1, len(JOINT_GROUPS))
    peak = scalar.argmax().to(scalar.dtype) / max(len(scalar) - 1, 1)
    scalar_stats = torch.stack(
        (scalar.mean(), scalar.std(unbiased=False), scalar.amax(), peak)
    )
    return torch.cat(
        (
            data["risk_probability"][item],
            action_probability.reshape(1),
            scalar_stats,
            part.mean(0),
            part.amax(0),
        )
    )


def retrieval_features(
    data: dict,
    action_top_k: int = 3,
    action_temperature: float = 1.0,
) -> list[list[dict]]:
    """Build action-grouped train-bank features without query labels."""

    train_class = data["checkpoint"]["train_class"]
    scalar_bank, part_bank = bank_profiles(data["train_bank"])
    probability = torch.softmax(
        data["fused_action"] / float(action_temperature), dim=-1
    )
    top_classes = probability.topk(action_top_k, dim=-1).indices
    result = []
    for item, valid in enumerate(data["inference_valid"]):
        groups = []
        for class_id in top_classes[item].tolist():
            indices = torch.nonzero(train_class == class_id, as_tuple=False).flatten()
            if len(indices) == 0:
                continue
            members = data["train_bank"].index_select(0, indices)
            pose = torch.linalg.vector_norm(
                members - data["baseline_bank"][item][None], dim=-1
            ).mean((1, 2))[None]
            scalar = profile_distance(
                data["predicted_scalar_profile"][item : item + 1],
                scalar_bank.index_select(0, indices)[None],
                valid[None],
            )
            part = part_profile_distance(
                data["predicted_part_profile"][item : item + 1],
                part_bank.index_select(0, indices)[None],
                valid[None],
            )
            part_values = (part - part.mean(1, keepdim=True)) / part.std(
                1, keepdim=True
            ).clamp_min(1e-5)
            groups.append(
                {
                    "class_id": class_id,
                    "indices": indices,
                    "probability": probability[item, class_id],
                    "pose": standardize(pose)[0],
                    "scalar": standardize(scalar)[0],
                    "part": weighted_part_distance(part)[0],
                    "part_values": part_values[0],
                    "context": query_motion_context(
                        data, item, valid, probability[item, class_id]
                    ),
                }
            )
        result.append(groups)
    return result


def group_candidate_values(group: dict) -> torch.Tensor:
    """Concatenate ranker evidence for one action group."""

    return torch.cat(
        (group["pose"][:, None], group["scalar"][:, None], group["part_values"]),
        dim=-1,
    )


@torch.no_grad()
def render_ranked_action(
    model: ProfileCandidateRanker,
    data: dict,
    features: list[list[dict]],
    device: str,
    inner_top_k: int = 2,
    inner_temperature: float = 1.0,
) -> torch.Tensor:
    """Rank within predicted actions and render a weighted motion prior."""

    model.eval()
    motions = []
    for groups, valid in zip(features, data["inference_valid"]):
        candidates, probabilities = [], []
        for group in groups:
            values = group_candidate_values(group).to(device)
            class_id = torch.full(
                (len(values),),
                int(group["class_id"]),
                dtype=torch.long,
                device=device,
            )
            score = model(values, class_id, None).cpu()
            count = min(int(inner_top_k), len(score))
            local = score.topk(count).indices
            bank_indices = group["indices"].index_select(0, local)
            inner_weight = torch.softmax(
                score.index_select(0, local) / float(inner_temperature), dim=0
            )
            candidate = (
                data["train_bank"].index_select(0, bank_indices)
                * inner_weight[:, None, None, None]
            ).sum(0)
            candidates.append(candidate)
            probabilities.append(group["probability"])
        probability = torch.stack(probabilities)
        probability = probability / probability.sum().clamp_min(1e-6)
        canonical = (
            torch.stack(candidates) * probability[:, None, None, None]
        ).sum(0)
        motions.append(render(canonical, valid, CACHE_FRAMES))
    return torch.stack(motions)


def candidate_semantic_scores(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Score candidate agreement with CSI-predicted action and risk."""

    indices = data["pool"]["indices"]
    candidate_class = data["checkpoint"]["train_class"][indices]
    candidate_risk = torch.where(
        candidate_class < 9,
        0,
        torch.where(candidate_class < 12, 1, 2),
    )
    risk_log = data["risk_probability"].clamp_min(1e-6).log()
    risk_log = risk_log.gather(1, candidate_risk)
    return standardize(data["pool"]["action_log_probability"]), standardize(risk_log)


def render_mixture(
    data: dict,
    adjusted: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Blend the highest-scoring train-bank trajectories."""

    top = adjusted.topk(top_k, dim=-1).indices
    probability = torch.softmax(adjusted / temperature, dim=-1)
    weight = probability.gather(1, top)
    weight = weight / weight.sum(1, keepdim=True)
    motions = []
    for item, valid in enumerate(data["inference_valid"]):
        bank_indices = data["pool"]["indices"][item].gather(0, top[item])
        canonical = (
            data["train_bank"].index_select(0, bank_indices)
            * weight[item, :, None, None, None]
        ).sum(0)
        motions.append(render(canonical, valid, CACHE_FRAMES))
    return torch.stack(motions)


def smooth_profile(values: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    """Smooth one-dimensional motion energy."""

    return F.avg_pool1d(
        values[:, None], kernel, stride=1, padding=kernel // 2
    )[:, 0]


def pose_speed(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute mean joint speed for each frame."""

    values = pose.new_zeros(valid.shape)
    values[:, 1:] = torch.linalg.vector_norm(
        pose[:, 1:] - pose[:, :-1], dim=-1
    ).mean(-1) * TARGET_FPS
    return values * valid


def monotonic_energy_warp(
    pose: torch.Tensor,
    query_activity: torch.Tensor,
    valid: torch.Tensor,
    warp_strength: float,
    floor_fraction: float,
) -> torch.Tensor:
    """Retime a candidate using cumulative CSI motion energy."""

    candidate_activity = pose_speed(pose, valid)
    output = torch.zeros_like(pose)
    for item, mask in enumerate(valid):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        length = len(positions)
        if length < 2:
            output[item, positions] = pose[item, positions]
            continue
        query = smooth_profile(query_activity[item, positions][None])[0].clamp_min(0)
        candidate = smooth_profile(
            candidate_activity[item, positions][None]
        )[0].clamp_min(0)
        query_floor = floor_fraction * query.mean().clamp_min(0.02)
        candidate_floor = floor_fraction * candidate.mean().clamp_min(0.02)
        query_cdf = torch.cumsum(query + query_floor, dim=0)
        candidate_cdf = torch.cumsum(candidate + candidate_floor, dim=0)
        query_cdf = query_cdf / query_cdf[-1].clamp_min(1e-6)
        candidate_cdf = candidate_cdf / candidate_cdf[-1].clamp_min(1e-6)
        upper = torch.searchsorted(candidate_cdf, query_cdf).clamp(0, length - 1)
        lower = (upper - 1).clamp_min(0)
        fraction = (query_cdf - candidate_cdf[lower]) / (
            candidate_cdf[upper] - candidate_cdf[lower]
        ).clamp_min(1e-6)
        mapped = lower.float() + fraction * (upper - lower).float()
        identity = torch.arange(length, device=pose.device, dtype=pose.dtype)
        mapped = (1.0 - warp_strength) * identity + warp_strength * mapped
        source_lower = mapped.floor().long().clamp(0, length - 1)
        source_upper = (source_lower + 1).clamp_max(length - 1)
        alpha = (mapped - source_lower.float())[:, None, None]
        source = pose[item, positions]
        output[item, positions] = (
            (1.0 - alpha) * source[source_lower] + alpha * source[source_upper]
        )
    return output


def adaptive_strength(risk_probability: torch.Tensor, config: dict) -> torch.Tensor:
    """Blend more or less retrieval prior from predicted risk uncertainty."""

    entropy = -(
        risk_probability.clamp_min(1e-6)
        * risk_probability.clamp_min(1e-6).log()
    ).sum(-1) / torch.log(risk_probability.new_tensor(3.0))
    return (
        config["base"]
        + config["danger_delta"] * risk_probability[:, 2]
        + config["uncertainty_delta"] * entropy
    ).clamp(0.40, 0.95)


def predict_locked(data: dict, adaptive_config: dict) -> torch.Tensor:
    """Render the frozen, validation-selected retrieval core."""

    action_score, risk_score = candidate_semantic_scores(data)
    adjusted = (
        data["logits"]
        - 0.20 * data["scalar_distance"]
        - 0.10 * weighted_part_distance(data["part_distance"])
        + 0.35 * action_score
        + 0.50 * risk_score
    )
    candidate = render_mixture(data, adjusted, 0.50, 5)
    activity = 0.50 * data["predicted_scalar_profile"] + 0.50 * data[
        "predicted_part_profile"
    ][..., 2:].mean(-1)
    warped = monotonic_energy_warp(
        candidate, activity, data["inference_valid"], 0.50, 0.30
    )
    strength = adaptive_strength(data["risk_probability"], adaptive_config)
    return (
        (1.0 - strength)[:, None, None, None] * data["baseline"]
        + strength[:, None, None, None] * warped
    )


def smooth_valid_delta(
    delta: torch.Tensor,
    valid: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Low-pass a trajectory correction only on observed frames."""

    result = torch.zeros_like(delta)
    for item, mask in enumerate(valid):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(positions) == 0:
            continue
        values = delta[item, positions].flatten(1).T[None]
        left = (window - 1) // 2
        values = F.pad(values, (left, left), mode="replicate")
        values = F.avg_pool1d(values, window, stride=1)
        result[item, positions] = values[0].T.reshape(len(positions), N_JOINTS, 3)
    return result


__all__ = [
    "CandidateMotionReranker",
    "MotionProfileHead",
    "PartMotionProfileHead",
    "ProfileCandidateRanker",
    "TemporalMotionSelector",
    "candidate_part_speed_profiles",
    "candidate_speed_profiles",
    "canonicalize",
    "model_inputs",
    "monotonic_energy_warp",
    "part_profile_distance",
    "predict_locked",
    "predict_part_profile",
    "predict_profile",
    "predict_selector",
    "profile_distance",
    "render_ranked_action",
    "retrieval_features",
    "smooth_valid_delta",
    "standardize",
]
