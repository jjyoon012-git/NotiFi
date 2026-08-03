"""모델 구조 — 3링크 CSI -> GVHMR 22관절.

    [B, 304, 3, 114, 2]
      -> PerLinkNorm      링크별 정규화 (캘리브레이션이 갈아끼우는 지점)
      -> LinkEncoder      228 -> hidden, 3링크 가중치 공유 + 링크 임베딩
      -> MaskedFusion     3링크 -> 1개, 죽은 링크 제외
      -> TemporalEncoder  TinyTCN(수용영역 4.2초) 또는 TinyMamba
      -> heads            pose / root / class / risk

TinyTCN·MambaBlock·TinyMamba 는 legacy `scripts/train_csi_to_pose.py` 627~748행에서
이식했다(출력 차원과 입력 규약만 조정). 나머지는 3링크 구조라 새로 작성했다.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import contract as C


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


# ------------------------------------------------------------------ ① 정규화


class PerLinkNorm(nn.Module):
    """링크별·subcarrier별 정규화. **캘리브레이션의 핵심 지점.**

    실측 문제: 같은 TX1 이라도 사이트마다 평균 진폭이 4.5~71.1 로 15.9배 차이난다.
    그대로 넣으면 모델이 자세가 아니라 "이건 어느 집인가"를 학습한다.

    통계를 데이터 로더가 아니라 **모델 buffer** 로 들고 있는 이유:
      - 체크포인트에 함께 저장되어 "이 모델은 이 통계로 정규화한다"가 명시된다
      - 새 현장에서는 fit() 을 한 번 더 부르는 것으로 끝난다(재학습 불필요)
      - 학습과 배포가 **같은 함수**를 쓴다. 다르면 재현 불가능한 버그가 된다
    """

    def __init__(self, n_links: int = C.N_LINKS, n_sc: int = C.N_LIVE_SUBCARRIERS):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_links, n_sc, 2))
        self.register_buffer("sigma", torch.ones(n_links, n_sc, 2))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, csi: torch.Tensor, link_mask: torch.Tensor, eps: float = 1e-3) -> None:
        """csi [.., T, L, S, 2] 와 link_mask [.., T, L] 로 통계를 추정한다.

        배포에서는 '빈 방 10초' CSI 를 그대로 넣으면 된다. GT 도 라벨도 필요 없다.
        학습에서는 train 셋의 통계로 초기화한다.
        """
        x = csi.reshape(-1, csi.shape[-3], csi.shape[-2], csi.shape[-1]).float()
        m = link_mask.reshape(-1, link_mask.shape[-1]).float()          # [N, L]
        w = m[:, :, None, None]                                          # [N, L, 1, 1]
        cnt = w.sum(0).clamp(min=1.0)
        mu = (x * w).sum(0) / cnt
        var = (((x - mu) ** 2) * w).sum(0) / cnt
        self.mu.copy_(mu)
        self.sigma.copy_(var.clamp(min=0).sqrt().clamp(min=eps))
        self.fitted.fill_(True)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        x = (csi - self.mu) / self.sigma
        return x * link_mask[..., None, None].to(x.dtype)


# ------------------------------------------------------------------ ② 링크 인코더


class LinkEncoder(nn.Module):
    """한 프레임 한 링크의 228개 숫자(114 subcarrier x I/Q)를 hidden 차원으로 압축.

    **3링크가 가중치를 공유한다.** 구현상으로는 링크 축을 배치 축에 접어 넣어
    한 번에 통과시킨다. 공유하는 이유:
      - 셋 다 같은 종류의 WiFi CSI 다. 다른 함수를 배울 이유가 없다
      - 학습 데이터가 3배가 된다(링크마다 따로면 각자 1/3 만 본다)
      - 링크가 죽은 trial 에서도 살아있는 링크가 같은 인코더를 계속 훈련시킨다
      - 파라미터가 1/3

    대신 **링크 임베딩**을 더해 어느 TX 인지를 남긴다. 배치 규약상 TX1(정면 80cm),
    TX2(측벽 150cm, 자세 구분), TX3(구석 30cm, 낙상 착지)의 역할이 다르기 때문이다.
    """

    def __init__(self, n_sc: int = C.N_LIVE_SUBCARRIERS, hidden: int = 96,
                 n_links: int = C.N_LINKS, dropout: float = 0.1,
                 film: bool = False):
        """
        Args:
            film: 링크별 **곱셈** 변조를 쓸 것인가. link_emb 의 덧셈은 평행이동뿐이라
                  "TX1 에서는 이 채널이 중요하고 TX3 에서는 아니다"를 만들 수 없다.
                  FiLM 은 GELU **앞**에서 채널별 배율을 링크마다 다르게 준다.
                  비선형이 반응하는 지점 자체가 링크마다 달라진다.
                  gamma=1, beta=0 으로 초기화해서 처음엔 film=False 와 완전히 같다.
        """
        super().__init__()
        in_dim = n_sc * 2
        mid = hidden * 2
        self.fc1 = nn.Linear(in_dim, mid)
        self.norm1 = nn.LayerNorm(mid)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mid, hidden)
        self.film = film
        if film:
            # LayerNorm 뒤에 두어야 정규화가 변조를 되돌리지 않는다
            self.gamma = nn.Parameter(torch.ones(n_links, mid))
            self.beta = nn.Parameter(torch.zeros(n_links, mid))
        self.link_emb = nn.Parameter(torch.zeros(n_links, hidden))
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, L, S, 2] -> [B, T, L, hidden]"""
        B, T, L = x.shape[:3]
        h = self.fc1(x.reshape(B * T * L, -1)).reshape(B, T, L, -1)   # [B,T,L,mid]
        h = self.norm1(h)
        if self.film:
            h = h * self.gamma + self.beta          # [L,mid] 가 B,T 로 브로드캐스트
        h = self.drop(F.gelu(h))
        h = self.fc2(h)                                                # [B,T,L,hidden]
        return self.out_norm(h + self.link_emb)


# ------------------------------------------------------------------ ③ 융합


class MaskedFusion(nn.Module):
    """3링크를 1개로 합친다. 죽은 링크는 **평균에 0으로 섞지 않고 아예 제외**한다.

    단순 평균이면 (0 + 0 + 진짜) / 3 이 되어 신호가 1/3 로 희석된다.
    우리 개발셋에 1~2링크만 살아있는 trial 이 244개 있으므로 실제로 문제가 된다.

    게이트 점수는 학습으로 배운다. 상황에 따라 유용한 링크가 다르기 때문이다
    (바닥에 누우면 TX3, 서 있으면 TX2 가 잘 본다).
    """

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, h: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        """h [B, T, L, D], link_mask [B, T, L] -> [B, T, D]"""
        score = self.gate(h).squeeze(-1)                       # [B, T, L]
        score = score.masked_fill(~link_mask, float("-inf"))
        # 전 링크가 죽은 프레임은 softmax 가 NaN 이 되므로 균등 가중으로 대체한 뒤
        # 출력 자체를 0 으로 만든다(입력이 없으니 특징도 없어야 한다).
        any_alive = link_mask.any(dim=-1, keepdim=True)         # [B, T, 1]
        score = torch.where(any_alive, score, torch.zeros_like(score))
        w = torch.softmax(score, dim=-1).unsqueeze(-1)          # [B, T, L, 1]
        out = (h * w).sum(dim=-2)
        return out * any_alive.to(out.dtype)


class ConcatFusion(nn.Module):
    """3링크를 **이어붙여** 합친다. 링크 정체성을 보존한다.

    MaskedFusion(가중평균)의 한계: 평균은 링크 간 **차이**를 표현할 수 없다.
    위치 추정은 삼각측량이고, 삼각측량은 'TX1 경로는 짧아지는데 TX2 는 길어진다'
    처럼 링크를 **비교**해야 한다. 평균을 내면 부호가 반대인 신호가 상쇄된다.

    실측(수평 이동 30cm 이상 1,381 trial, 전반 2초 대비 후반 2초 진폭 변화율):
        개별 링크 ~ 변위 상관   최대 0.49  (같은 사이트에서 링크마다 부호가 반대)
        3링크 평균 ~ 변위 상관       0.04  <- 상쇄되어 사라진다

    죽은 링크는 0 으로 채우고 **마스크를 특징으로 함께** 넣는다. 그래야 다음 층이
    '이 링크는 없음'과 '값이 0'을 구분할 수 있다. 이게 없으면 MaskedFusion 이
    해결하던 신호 희석 문제가 그대로 돌아온다.
    """

    def __init__(self, hidden: int = 96, n_links: int = C.N_LINKS,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden * n_links + n_links, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        """h [B, T, L, D], link_mask [B, T, L] -> [B, T, D]"""
        B, T, L, D = h.shape
        m = link_mask.to(h.dtype)
        x = (h * m[..., None]).reshape(B, T, L * D)
        out = self.proj(torch.cat([x, m], dim=-1))
        any_alive = link_mask.any(dim=-1, keepdim=True).to(out.dtype)
        return self.out_norm(out) * any_alive


FUSIONS = {"gate": MaskedFusion, "concat": ConcatFusion}


# ------------------------------------------------------------------ ④ 시간 인코더
#  아래 두 클래스는 legacy scripts/train_csi_to_pose.py 627~748행에서 이식했다.
#  입출력을 (B, T, D) 규약으로 통일한 것 외에는 구조가 같다.


class TinyTCN(nn.Module):
    """dilated Conv1d 스택.

    수용영역 = 1 + (k-1)·Σdilations. k=5, (1,2,4,8,16) -> 125프레임 = 4.2초.
    낙상 추론('버스트 후 정적 = 쓰러짐')은 버스트와 현재를 한 시야에 담아야 하므로
    수용영역이 1초(=(1,2,4))면 부족하다.
    """

    def __init__(self, in_dim: int, hidden: int = 96,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.1):
        super().__init__()
        k = 5
        layers = []
        ch = in_dim
        for d in dilations:
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=k, padding=(k - 1) // 2 * d, dilation=d),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.receptive_field = 1 + (k - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, hidden]"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MambaBlock(nn.Module):
    """Selective SSM (Mamba) 블록 — 순수 PyTorch 구현 (mamba-minimal 계열).

    공식 mamba-ssm 은 CUDA 커널 필수 + Windows 빌드 불가라 시간축 for-loop 스캔을 쓴다.
    구성은 C-MambaPose 와 동일: d_state=16, d_conv=4, expand=2.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xc = self.conv1d(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xc = F.silu(xc)
        y = self._ssm(xc)
        return self.out_proj(y * F.silu(z))

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.A_log)
        d_in, N = A.shape
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        h = x.new_zeros(x.shape[0], d_in, N)
        ys = []
        for t in range(x.shape[1]):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
        return torch.stack(ys, dim=1) + x * self.D


class TinyMamba(nn.Module):
    """TCN 의 몸통을 Mamba 로 교체한 대응 모델. 입출력 규약 동일.

    TCN 은 대칭 padding 으로 비인과(양방향 시야)이므로 공정 비교를 위해 Mamba 도
    양방향으로 쓴다: 각 층에서 정방향 + 역방향(시간 뒤집기)의 합을 residual 로 더한다.
    """

    def __init__(self, in_dim: int, hidden: int = 96, n_blocks: int = 2,
                 d_state: int = 16, bidir: bool = True):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])
        self.fwd = nn.ModuleList([MambaBlock(hidden, d_state) for _ in range(n_blocks)])
        self.bwd = nn.ModuleList([MambaBlock(hidden, d_state)
                                  for _ in range(n_blocks)]) if bidir else None
        self.out_norm = nn.LayerNorm(hidden)
        self.receptive_field = -1        # 이론상 제한 없음

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for i, blk in enumerate(self.fwd):
            hn = self.norms[i](h)
            upd = blk(hn)
            if self.bwd is not None:
                upd = upd + torch.flip(self.bwd[i](torch.flip(hn, [1])), [1])
            h = h + upd
        return self.out_norm(h)


# ------------------------------------------------------------------ structure-preserving backbone


class SubcarrierResidual(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, groups=channels),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SubcarrierConvEncoder(nn.Module):
    """Preserve local frequency structure instead of flattening 114 subcarriers."""

    def __init__(self, hidden: int, n_links: int = C.N_LINKS, dropout: float = 0.1):
        super().__init__()
        width = max(32, hidden // 2)
        groups = 8 if width % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv1d(2, width, 7, padding=3),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            SubcarrierResidual(width, dropout),
            SubcarrierResidual(width, dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(width * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.link_embedding = nn.Parameter(torch.zeros(n_links, hidden))
        nn.init.normal_(self.link_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, links, subcarriers = x.shape[:4]
        h = x.reshape(b * t * links, subcarriers, 2).transpose(1, 2)
        h = self.stem(h)
        h = torch.cat((h.mean(-1), h.amax(-1)), dim=-1)
        h = self.proj(h).reshape(b, t, links, -1)
        return h + self.link_embedding


class LinkAttentionFusion(nn.Module):
    """Fuse links with a learned query while retaining TX identity and missing-link masks."""

    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, links: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        b, t, n_links, hidden = links.shape
        flat = links.reshape(b * t, n_links, hidden)
        query = self.query.expand(b * t, -1, -1)
        tokens = torch.cat((query, flat), dim=1)
        query_valid = torch.ones(b * t, 1, dtype=torch.bool, device=links.device)
        valid = torch.cat((query_valid, link_mask.reshape(b * t, n_links)), dim=1)
        attended, _ = self.attn(
            tokens, tokens, tokens, key_padding_mask=~valid, need_weights=False
        )
        fused = self.norm1(tokens[:, 0] + attended[:, 0])
        fused = self.norm2(fused + self.ffn(fused)).reshape(b, t, hidden)
        any_link = link_mask.any(-1, keepdim=True).to(fused.dtype)
        return fused * any_link


def _sinusoidal_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding


class LocalTemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(
            hidden, hidden, 5, padding=2 * dilation, dilation=dilation, groups=hidden
        )
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = self.pointwise(F.gelu(self.depthwise(h))).transpose(1, 2)
        return x + self.dropout(h)


class TemporalTransformer(nn.Module):
    """LayerNorm local dynamics plus global bidirectional temporal context."""

    def __init__(self, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.local = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout) for dilation in (1, 2, 4)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        for block in self.local:
            x = block(x)
        x = x + _sinusoidal_position(x.shape[1], x.shape[2], x.device).to(x.dtype)
        safe_mask = frame_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        x = self.transformer(x, src_key_padding_mask=~safe_mask)
        return x * frame_mask[..., None].to(x.dtype)


def _skeleton_adjacency() -> torch.Tensor:
    adjacency = torch.eye(C.N_JOINTS, dtype=torch.float32)
    for parent, child in C.SKELETON_EDGES:
        adjacency[parent, child] = 1.0
        adjacency[child, parent] = 1.0
    degree = adjacency.sum(-1).clamp(min=1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class JointGraphBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.register_buffer("adjacency", _skeleton_adjacency())
        self.norm1 = nn.LayerNorm(hidden)
        self.self_proj = nn.Linear(hidden, hidden)
        self.graph_proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        h = self.norm1(joints)
        neighbors = torch.einsum("ij,btjd->btid", self.adjacency, h)
        joints = joints + self.self_proj(h) + self.graph_proj(neighbors)
        return joints + self.ffn(self.norm2(joints))


class KinematicGraphDecoder(nn.Module):
    """Decode parent-relative bones, then compose the SMPL-22 kinematic tree."""

    def __init__(self, hidden: int, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.joint_queries = nn.Parameter(torch.zeros(C.N_JOINTS, hidden))
        self.blocks = nn.ModuleList(JointGraphBlock(hidden, dropout) for _ in range(blocks))
        self.bone_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        nn.init.normal_(self.joint_queries, std=0.02)

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        joints = temporal[:, :, None, :] + self.joint_queries[None, None, :, :]
        for block in self.blocks:
            joints = block(joints)
        local = self.bone_head(joints)
        pose = torch.zeros_like(local)
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                pose[:, :, child] = pose[:, :, parent] + local[:, :, child]
        return pose


class HybridGraphDecoder(nn.Module):
    """Blend direct joint coordinates with a kinematic-tree reconstruction.

    The tree branch supplies a skeletal prior, while the direct branch prevents
    parent errors from accumulating all the way to wrists and feet.
    """

    def __init__(self, hidden: int, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.joint_queries = nn.Parameter(torch.zeros(C.N_JOINTS, hidden))
        self.blocks = nn.ModuleList(JointGraphBlock(hidden, dropout) for _ in range(blocks))
        self.direct_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.bone_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        # Start mostly direct; the optimizer can increase the tree contribution.
        self.tree_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.joint_queries, std=0.02)

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        joints = temporal[:, :, None, :] + self.joint_queries[None, None, :, :]
        for block in self.blocks:
            joints = block(joints)

        direct = self.direct_head(joints)
        direct = direct - direct[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]

        local = self.bone_head(joints)
        tree = torch.zeros_like(local)
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent >= 0:
                tree[:, :, child] = tree[:, :, parent] + local[:, :, child]
        mix = torch.sigmoid(self.tree_logit)
        return direct * (1.0 - mix) + tree * mix


class PoseTemporalRefiner(nn.Module):
    """Learn a bounded temporal residual over an already plausible pose.

    The output layer is zero-initialized, making this module an exact identity
    when a robust GraphFormer checkpoint is first warm-started.
    """

    def __init__(self, hidden: int, dropout: float = 0.1,
                 max_delta: float = 0.15,
                 joint_scale: tuple[float, ...] | list[float] | None = None):
        super().__init__()
        self.max_delta = max_delta
        scale = torch.ones(C.N_JOINTS, dtype=torch.float32)
        if joint_scale is not None:
            if len(joint_scale) != C.N_JOINTS:
                raise ValueError(
                    f"refiner joint scale needs {C.N_JOINTS} values, got {len(joint_scale)}"
                )
            scale = torch.tensor(joint_scale, dtype=torch.float32)
        self.register_buffer("joint_scale", scale, persistent=False)
        self.pose_projection = nn.Linear(C.N_JOINTS * 3, hidden)
        self.blocks = nn.ModuleList(
            LocalTemporalBlock(hidden, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, C.N_JOINTS * 3)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, coarse: torch.Tensor, temporal: torch.Tensor,
                frame_mask: torch.Tensor) -> torch.Tensor:
        batch, frames = coarse.shape[:2]
        features = self.pose_projection(coarse.reshape(batch, frames, -1)) + temporal
        for block in self.blocks:
            features = block(features)
        delta = self.max_delta * torch.tanh(self.head(features))
        delta = delta.reshape(batch, frames, C.N_JOINTS, 3)
        delta = delta * self.joint_scale[None, None, :, None]
        delta = delta * frame_mask[:, :, None, None].to(delta.dtype)
        refined = coarse + delta
        return refined - refined[:, :, C.ROOT_JOINT:C.ROOT_JOINT + 1]


class GraphPoseNet(nn.Module):
    """Structure-preserving CSI encoder and factorized pose/root decoder."""

    def __init__(self, hidden: int = 128, n_blocks: int = 3, dropout: float = 0.1,
                 heads: int = 4, graph_blocks: int = 2, decoder: str = "tree",
                 robust_heads: bool = False, domain_grl: float = 0.2,
                 temporal_refiner: bool = False,
                 refiner_joint_scale: tuple[float, ...] | list[float] | None = None,
                 **_: object):
        super().__init__()
        self.hidden = hidden
        self.robust_heads = robust_heads
        self.temporal_refiner = temporal_refiner
        self.domain_grl = domain_grl
        self.norm = PerLinkNorm()
        self.encoder = SubcarrierConvEncoder(hidden, dropout=dropout)
        self.fusion = LinkAttentionFusion(hidden, heads=heads, dropout=dropout)
        self.temporal = TemporalTransformer(hidden, n_blocks, heads, dropout)
        if decoder == "tree":
            self.pose_decoder = KinematicGraphDecoder(hidden, graph_blocks, dropout)
        elif decoder == "hybrid":
            self.pose_decoder = HybridGraphDecoder(hidden, graph_blocks, dropout)
        else:
            raise ValueError(f"unknown graph decoder {decoder!r}")
        self.decoder_kind = decoder
        if temporal_refiner:
            self.pose_refiner = PoseTemporalRefiner(
                hidden, dropout, joint_scale=refiner_joint_scale
            )
        self.root_decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.motion_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.class_head = nn.Linear(hidden, C.N_CLASSES)
        self.risk_head = nn.Linear(hidden, C.N_RISK)
        if robust_heads:
            self.phase_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
            self.contact_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))
            self.embedding_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
            )
            self.domain_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 9)
            )

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        x = self.norm(csi, link_mask)
        links = self.encoder(x)
        fused = self.fusion(links, link_mask)
        frame_mask = link_mask.any(-1)
        temporal = self.temporal(fused, frame_mask)
        motion = self.motion_head(temporal).squeeze(-1)

        scores = motion.masked_fill(~frame_mask, -1e4)
        attention = torch.softmax(scores, dim=1)
        attended = (temporal * attention[..., None]).sum(1)
        mean = (temporal * frame_mask[..., None]).sum(1)
        mean = mean / frame_mask.sum(1, keepdim=True).clamp(min=1)
        pooled = 0.5 * (attended + mean)

        coarse_pose = self.pose_decoder(temporal)
        pose = (
            self.pose_refiner(coarse_pose, temporal, frame_mask)
            if self.temporal_refiner else coarse_pose
        )
        output = {
            "pose_rel": pose,
            "root": self.root_decoder(temporal),
            "motion": motion,
            "class_logits": self.class_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "temporal_features": temporal,
        }
        if self.temporal_refiner:
            output["pose_coarse"] = coarse_pose
        if self.robust_heads:
            embedding = F.normalize(self.embedding_head(pooled), dim=-1)
            reversed_embedding = _GradientReverse.apply(embedding, self.domain_grl)
            output.update({
                "phase_logits": self.phase_head(temporal),
                "contact_logits": self.contact_head(temporal),
                "embedding": embedding,
                "domain_logits": self.domain_head(reversed_embedding),
            })
        return output

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def describe(self) -> str:
        layers = len(self.temporal.transformer.layers)
        return (
            f"GraphPoseNet(hidden={self.hidden}, temporal_layers={layers}, "
            f"decoder={self.decoder_kind}, joint_graph_blocks={len(self.pose_decoder.blocks)}, "
            f"robust_heads={self.robust_heads}, temporal_refiner={self.temporal_refiner}) "
            f"params={self.n_params():,}"
        )


# ------------------------------------------------------------------ ⑤ 전체


class PoseNet(nn.Module):
    """CSI [B,T,3,114,2] -> pose/root/class/risk.

    출력:
        pose_rel [B, T, 22, 3]   골반 기준 상대 좌표 (주 타깃)
        root     [B, T, 3]       골반 절대 위치 (주 타깃)
        class    [B, 17]         행동 17종 — pose 학습 보조. risk(3종)로 감독하면
                                 서있기/앉기/눕기가 모두 safe 로 묶여 자세 분리
                                 신호가 되지 못한다
        risk     [B, 3]          safe/warning/danger — 배포에서 실제로 쓰는 출력
    """

    def __init__(self, arch: str = "tcn", hidden: int = 96,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 n_blocks: int = 2, dropout: float = 0.1,
                 fusion: str = "gate", film: bool = False):
        super().__init__()
        self.arch = arch
        self.hidden = hidden
        self.fusion_kind = fusion
        self.film = film
        self.norm = PerLinkNorm()
        self.encoder = LinkEncoder(hidden=hidden, dropout=dropout, film=film)
        if fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {fusion!r} (가능: {sorted(FUSIONS)})")
        self.fusion = (ConcatFusion(hidden=hidden, dropout=dropout) if fusion == "concat"
                       else MaskedFusion(hidden=hidden))
        if arch == "mamba":
            self.temporal = TinyMamba(hidden, hidden, n_blocks=n_blocks)
        elif arch == "tcn":
            self.temporal = TinyTCN(hidden, hidden, dilations=dilations, dropout=dropout)
        else:
            raise ValueError(f"unknown arch {arch!r}")

        self.pose_head = nn.Linear(hidden, C.N_JOINTS * 3)
        self.root_head = nn.Linear(hidden, 3)
        self.class_head = nn.Linear(hidden, C.N_CLASSES)
        self.risk_head = nn.Linear(hidden, C.N_RISK)

    def forward(self, csi: torch.Tensor, link_mask: torch.Tensor) -> dict:
        B, T = csi.shape[:2]
        x = self.norm(csi, link_mask)
        h = self.encoder(x)                       # [B, T, L, D]
        h = self.fusion(h, link_mask)             # [B, T, D]
        h = self.temporal(h)                      # [B, T, D]

        pose = self.pose_head(h).reshape(B, T, C.N_JOINTS, 3)
        root = self.root_head(h)

        # 분류는 10초 전체에 하나. 입력이 있는 프레임만 평균낸다.
        frame_mask = link_mask.any(dim=-1, keepdim=True).to(h.dtype)   # [B, T, 1]
        pooled = (h * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1.0)

        return {
            "pose_rel": pose,
            "root": root,
            "class_logits": self.class_head(pooled),
            "risk_logits": self.risk_head(pooled),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        rf = getattr(self.temporal, "receptive_field", -1)
        rf_s = "무제한" if rf < 0 else f"{rf}프레임 = {rf / C.TARGET_FPS:.1f}초"
        return (f"PoseNet(arch={self.arch}, hidden={self.hidden}, "
                f"fusion={self.fusion_kind}{', film' if self.film else ''}) "
                f"params={self.n_params():,}  수용영역={rf_s}")


def build_model(arch: str = "tcn", **kw) -> PoseNet | GraphPoseNet:
    if arch == "v3":
        from .v3 import V3PoseNet
        return V3PoseNet(**kw)
    if arch == "graphformer":
        return GraphPoseNet(**kw)
    if arch == "robust_graphformer":
        return GraphPoseNet(robust_heads=True, **kw)
    if arch == "impact_graphformer":
        return GraphPoseNet(robust_heads=True, temporal_refiner=True, **kw)
    if arch == "latent_flow":
        from .latent_flow import LatentFlowPoseNet
        return LatentFlowPoseNet(**kw)
    return PoseNet(arch=arch, **kw)
