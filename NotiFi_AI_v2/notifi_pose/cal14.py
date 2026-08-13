"""CAL14: unseen feature 크기 이동에 강한 cosine 분류 head."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from . import contract as C
from .cal13 import CAL13MotionGrounded


class CosineClassifier(nn.Module):
    """feature와 class 중심의 방향만 비교해 domain별 logit 크기 이동을 제거한다."""

    def __init__(self, dimension: int, classes: int, initial_scale: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        nn.init.normal_(self.weight, std=dimension ** -0.5)
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """정규화된 feature-class cosine에 제한된 학습 온도를 곱한다."""
        scale = self.log_scale.exp().clamp(1.0, 30.0)
        return scale * F.linear(
            F.normalize(values, dim=-1), F.normalize(self.weight, dim=-1)
        )


class CAL14InvariantCosine(CAL13MotionGrounded):
    """CAL13 encoder에 bias 없는 cosine action/risk 중심을 적용한다."""

    def __init__(self, *args, cosine_scale: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosine_scale = float(cosine_scale)
        self.cosine_action = CosineClassifier(
            self.hidden, C.N_CLASSES, self.cosine_scale
        )
        self.cosine_risk = CosineClassifier(
            self.hidden, C.N_RISK, self.cosine_scale
        )

    def model_config(self) -> dict:
        """cosine 온도를 포함한 재현 설정을 저장한다."""
        config = super().model_config()
        config["cosine_scale"] = self.cosine_scale
        return config

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """CAL13 시간 특징을 정규화된 action/risk class 중심과 비교한다."""
        output = super().forward(*args, **kwargs)
        embedding = output["embedding"]
        unconditioned = self.cosine_action(embedding)
        base_direct_risk = self.cosine_risk(embedding)
        gate = output["adapter_gate"][:, None]
        adapted_action = unconditioned + gate * output["action_residual"]
        direct_risk = base_direct_risk + gate * output["risk_residual"]
        hierarchy = output["hierarchy_strength"]
        action_logits = adapted_action + hierarchy * self.expand_risk_prior(
            direct_risk.log_softmax(-1)
        )
        action_risk = self.action_to_risk(action_logits)
        fusion = torch.sigmoid(self.risk_fusion)
        risk_logits = (1.0 - fusion) * direct_risk + fusion * action_risk
        output.update({
            "unconditioned_action_logits": unconditioned,
            "base_action_logits": unconditioned,
            "action_logits": action_logits,
            "direct_risk_logits": direct_risk,
            "base_direct_risk_logits": base_direct_risk,
            "action_risk_logits": action_risk,
            "risk_logits": risk_logits,
            "base_risk_logits": (
                (1.0 - fusion) * base_direct_risk
                + fusion * self.action_to_risk(unconditioned)
            ),
        })
        return output
