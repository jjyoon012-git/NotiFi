"""학습용 Dataset — 캐시에서 배치를 꺼낸다.

샘플 = trial 하나 = 10초 = 최대 304프레임. 배포에서 10초를 모아 한 번에 예측하므로
학습도 같은 단위로 한다(학습과 배포의 입력 규약이 다르면 재현 불가능한 버그가 된다).

한 샘플이 내놓는 것:
    csi        [304, 3, 114, 2] float32   입력
    link_mask  [304, 3]         bool      이 시각 이 링크가 유효한가
    pose_rel   [304, 22, 3]     float32   골반 기준 상대 좌표 (주 타깃)
    root       [304, 3]         float32   골반 절대 위치 (주 타깃)
    valid      [304]            bool      이 프레임의 GT 를 채점해도 되는가
    class_id   ()               int64     행동 17종 (보조 타깃)
    risk_id    ()               int64     safe/warning/danger (배포 출력)

link_mask 는 두 가지를 AND 로 합친 것이다.
  (1) 프레임 단위: 그 시각 주변에 실측 패킷이 있었는가 (캐시에 저장됨)
  (2) trial 단위: 그 링크가 애초에 살아있는가 (linkqc 판정)
(2)를 빼먹으면 yja E01 처럼 '잡음만 들어오는 링크'를 진짜 신호로 학습한다.

짧은 trial(198프레임 등)은 캐시에서 뒤가 0으로 채워져 있고 link_mask/valid 도 False 다.
따로 처리할 필요 없이 마스크가 알아서 배제한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import contract as C
from .. import linkqc
from . import cache as cache_mod


def load_experiments() -> dict:
    path = C.SPLIT_DIR / "experiments.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 -> python -m notifi_pose.tools.build_splits")
    return json.loads(path.read_text(encoding="utf-8"))


def select_rows(index: pd.DataFrame, spec: dict) -> np.ndarray:
    """experiments.json 의 한 항목({subjects, roles})으로 캐시 행 번호를 고른다."""
    m = index.subject.isin(spec["subjects"]) & index.role.isin(spec["roles"])
    m &= index.split_group == "dev"
    m &= index.cache_ok
    return np.flatnonzero(m.to_numpy())


def select_sealed_rows(index: pd.DataFrame, spec: dict) -> np.ndarray:
    m = ((index.subject == spec["subject"])
         & (index.environment == spec["environment"])
         & (index.split_group == "sealed") & index.cache_ok)
    if spec.get("task"):
        m &= index.task == spec["task"]
    return np.flatnonzero(m.to_numpy())


class SiteBaseline:
    """사이트별 빈방 기준선 제거 (Phase 2). build_site_baseline.py 가 만든 표를 쓴다.

    모드:
        "none"   적용 안 함
        "sub"    x - mu0             빈방 프로파일을 뺀다. 사이트 지문이 사라진다
        "sub_z"  (x - mu0 - rm)/rs   뺀 뒤 남은 잔차까지 정규화
                                     분석상 판별력 최고(0.939 vs sub 0.788 vs 미적용 0.668)

    **마스크 처리가 핵심이다.** 두 가지를 반드시 지킨다.

      1) 마스크가 False 인 자리(패킷 없음 / 죽은 링크)는 값이 0 이다. 여기서 mu0 를 빼면
         -mu0 가 되는데 이는 **그 사이트의 빈방 프로파일 = 사이트 지문 그 자체**다.
         지문을 지우려다 없는 자리에 지문을 새겨 넣는 셈이 된다. 유효한 자리만 빼고
         나머지는 0 을 유지한다.

      2) 빈방 녹음 때 죽어 있던 링크는 mu0 가 잡음이라 빼면 안 된다. 해당 링크는
         calibration만 건너뛰고 원 CSI를 유지한다. linkqc와 빈방 판정이 어긋날 때
         링크를 꺼버리면 일부 trial의 입력 전체가 사라질 수 있기 때문이다.

    대안으로 'trial 자체 시간평균'을 기준선으로 쓰는 것은 안 된다 — 정지 자세 trial 의
    시간평균은 '정적 성분 + 서 있는 사람'이라, 빼면 자세 정보 자체가 사라진다.
    """

    MODES = ("none", "sub", "sub_z")

    def __init__(self, mode: str = "none", path=None):
        if mode not in self.MODES:
            raise ValueError(f"baseline mode 는 {self.MODES} 중 하나: {mode!r}")
        self.mode = mode
        self.table: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if mode == "none":
            return
        path = path or (C.CACHE_DIR / "site_baseline.npz")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 없음 -> python -m notifi_pose.tools.build_site_baseline")
        z = np.load(path, allow_pickle=False)
        if str(z["preproc_version"]) != C.PREPROC_VERSION:
            raise RuntimeError(
                f"기준선 전처리 버전 불일치: {z['preproc_version']} != {C.PREPROC_VERSION}"
                f"  -> build_site_baseline 을 다시 실행하라")
        for k in [str(s) for s in z["sites"]]:
            mu0 = z[f"mu0__{k}"].astype(np.float32)
            valid = z[f"valid__{k}"].astype(bool)
            if mode == "sub":
                self.table[k] = (mu0, np.ones_like(mu0), valid)
            else:
                self.table[k] = (mu0 + z[f"rm__{k}"], z[f"rs__{k}"], valid)

    def apply(self, csi: np.ndarray, link_mask: np.ndarray,
              site: str) -> tuple[np.ndarray, np.ndarray]:
        """csi [T,L,S,2], link_mask [T,L] -> 기준선 제거된 (csi, link_mask)."""
        if self.mode == "none":
            return csi, link_mask
        entry = self.table.get(site)
        if entry is None:                       # 기준선이 없는 사이트는 손대지 않는다
            return csi, link_mask
        mu, sd, valid = entry
        calibrated = (csi - mu) / sd
        out = np.where(valid[None, :, None, None], calibrated, csi)
        return out * link_mask[:, :, None, None], link_mask

    def __repr__(self):
        return f"SiteBaseline(mode={self.mode}, sites={len(self.table)})"


@dataclass
class DropoutConfig:
    """link dropout — 링크 고장 상황을 학습 중에 미리 겪게 한다.

    실제 데이터에 1~2링크만 살아있는 trial 이 351개 있고, 배포 현장에서도 보드가
    죽을 수 있다. 이걸 학습하지 않으면 3링크가 다 있을 때만 동작하는 모델이 된다.
    캘리브레이션 Stage 0 에서 죽은 링크를 감지해 끄더라도 모델이 버텨야 한다.
    """
    p: float = 0.25          # 이 확률로 dropout 을 시도한다
    max_drop: int = 2        # 최대 몇 개까지 끌 것인가 (전부 끄지는 않는다)
    rf_augment: bool = False
    gain_std: float = 0.15
    phase_std: float = 0.25
    phase_slope_std: float = 0.08
    subcarrier_mask_p: float = 0.20
    temporal_jitter: int = 2


class PoseDataset(Dataset):
    def __init__(
        self,
        rows: np.ndarray,
        cache: cache_mod.Cache,
        link_ok: np.ndarray,
        train: bool = False,
        dropout: DropoutConfig | None = None,
        seed: int = 0,
        baseline: "SiteBaseline | None" = None,
    ):
        """
        Args:
            rows: 캐시 행 번호
            link_ok: [N_cache, 3] bool — trial 별 링크 사용 가능 여부(linkqc)
            train: True 면 link dropout 을 적용한다
            baseline: 사이트별 빈방 기준선 제거기 (Phase 2). None 이면 적용 안 함
        """
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cache = cache
        self.link_ok = link_ok
        self.train = train
        self.dropout = dropout or DropoutConfig()
        self.seed = seed
        self.epoch = 0
        self.baseline = baseline or SiteBaseline("none")
        self.index = cache.index.iloc[self.rows].reset_index(drop=True)
        self._site = (cache.index.subject + "_" + cache.index.environment).to_numpy()
        subject_order = {
            name: index for index, name in enumerate(("ajh", "lmh", "mhw", "yja"))
        }
        source_sites = {
            f"{subject}_{environment}": index
            for index, (subject, environment) in enumerate(
                (s, e)
                for s in ("ajh", "lmh", "mhw")
                for e in ("E01", "E02", "E03")
            )
        }
        self._subject_id = (
            cache.index.subject.map(subject_order).fillna(-1).to_numpy(dtype=np.int64)
        )
        self._domain_id = np.asarray(
            [source_sites.get(site, -1) for site in self._site], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _drop_links(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """살아있는 링크 중 일부를 끈다. 최소 1개는 남긴다."""
        alive = np.flatnonzero(mask.any(axis=0))
        if len(alive) <= 1 or rng.random() >= self.dropout.p:
            return mask
        k = int(rng.integers(1, min(self.dropout.max_drop, len(alive) - 1) + 1))
        drop = rng.choice(alive, size=k, replace=False)
        mask = mask.copy()
        mask[:, drop] = False
        return mask

    @staticmethod
    def _shift(values: np.ndarray, amount: int, fill=0) -> np.ndarray:
        if amount == 0:
            return values
        shifted = np.roll(values, amount, axis=0)
        if amount > 0:
            shifted[:amount] = fill
        else:
            shifted[amount:] = fill
        return shifted

    def _augment_rf(self, csi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply hardware-style complex gain and frequency response changes."""
        config = self.dropout
        links = csi.shape[1]
        scale = np.exp(
            rng.normal(0.0, config.gain_std, size=links)
        ).astype(np.float32)
        offset = rng.normal(0.0, config.phase_std, size=links).astype(np.float32)
        slope = rng.normal(0.0, config.phase_slope_std, size=links).astype(np.float32)
        frequency = np.linspace(-1.0, 1.0, csi.shape[2], dtype=np.float32)
        angle = offset[:, None] + slope[:, None] * frequency[None]
        cosine, sine = np.cos(angle), np.sin(angle)
        real, imag = csi[..., 0].copy(), csi[..., 1].copy()
        csi[..., 0] = scale[None, :, None] * (
            real * cosine[None] - imag * sine[None]
        )
        csi[..., 1] = scale[None, :, None] * (
            real * sine[None] + imag * cosine[None]
        )

        if rng.random() < config.subcarrier_mask_p:
            width = int(rng.integers(4, min(17, csi.shape[2] + 1)))
            start = int(rng.integers(0, csi.shape[2] - width + 1))
            link = int(rng.integers(0, links))
            csi[:, link, start:start + width] = 0.0
        return csi

    def __getitem__(self, i: int) -> dict:
        row = int(self.rows[i])
        a = self.cache.arrays
        meta = self.cache.index.iloc[row]

        csi = np.array(a["csi_iq"][row], dtype=np.float32)       # [304,3,114,2]
        mask = np.array(a["link_mask"][row], dtype=bool)          # [304,3]
        pose = np.array(a["pose_rel"][row], dtype=np.float32)
        root = np.array(a["root"][row], dtype=np.float32)
        valid = np.array(a["valid"][row], dtype=bool)
        # (2) trial 단위 링크 판정을 AND 로 합친다
        mask = mask & self.link_ok[row][None, :]

        # (3) 사이트별 빈방 기준선 제거. 마스크가 True 인 자리만 빼고, 기준선을
        #     신뢰할 수 없는 링크는 마스크에서 빠진다(SiteBaseline 주석 참조).
        csi, mask = self.baseline.apply(csi, mask, self._site[row])

        if self.train:
            rng = np.random.default_rng(
                self.seed * 1_000_003 + self.epoch * 10_007 + row
            )
            if self.dropout.rf_augment:
                csi = self._augment_rf(csi, rng)
                jitter = int(rng.integers(
                    -self.dropout.temporal_jitter,
                    self.dropout.temporal_jitter + 1,
                ))
                csi = self._shift(csi, jitter, 0.0)
                mask = self._shift(mask, jitter, False)
                pose = self._shift(pose, jitter, 0.0)
                root = self._shift(root, jitter, 0.0)
                valid = self._shift(valid, jitter, False)
            mask = self._drop_links(mask, rng)

        # 꺼진 링크의 값은 0 으로 — 모델이 마스크를 무시해도 신호가 새지 않게
        csi = csi * mask[:, :, None, None].astype(np.float32)

        return {
            "csi": torch.from_numpy(csi),
            "link_mask": torch.from_numpy(mask),
            "pose_rel": torch.from_numpy(pose),
            "root": torch.from_numpy(root),
            "valid": torch.from_numpy(valid),
            "class_id": torch.tensor(int(meta.class_id), dtype=torch.long),
            "risk_id": torch.tensor(int(meta.risk_id), dtype=torch.long),
            "subject_id": torch.tensor(self._subject_id[row], dtype=torch.long),
            "domain_id": torch.tensor(self._domain_id[row], dtype=torch.long),
            "row": torch.tensor(row, dtype=torch.long),
        }

    def set_epoch(self, epoch: int) -> None:
        """Keep link dropout reproducible while changing it every epoch."""
        self.epoch = int(epoch)

    # ---------------- 통계 ----------------

    def class_counts(self) -> np.ndarray:
        c = np.zeros(C.N_CLASSES, dtype=np.int64)
        vc = self.index.class_id.value_counts()
        c[vc.index.to_numpy()] = vc.to_numpy()
        return c

    def risk_counts(self) -> np.ndarray:
        c = np.zeros(C.N_RISK, dtype=np.int64)
        vc = self.index.risk_id.value_counts()
        c[vc.index.to_numpy()] = vc.to_numpy()
        return c

    def describe(self) -> str:
        d = self.index
        return (f"n={len(d)}  subjects={sorted(set(d.subject))}  "
                f"envs={sorted(set(d.environment))}  "
                f"classes={d.class_id.nunique()}/{C.N_CLASSES}  "
                f"링크수 {d.n_alive.value_counts().sort_index().to_dict()}  "
                f"baseline={self.baseline.mode}")


# ------------------------------------------------------------------ 팩토리


def _link_ok_matrix(cache: cache_mod.Cache) -> np.ndarray:
    """캐시 행 순서에 맞춘 [N, 3] 링크 사용 가능 여부."""
    usable = linkqc.link_mask_per_trial()
    aligned = usable.reindex(cache.index.trial_id)
    # 판정 정보가 없으면(있을 수 없지만) 보수적으로 전부 사용 가능 처리
    return aligned.fillna(True).to_numpy(dtype=bool)


def build_datasets(
    exp: str = "single_split",
    fold: str | None = None,
    dropout: DropoutConfig | None = None,
    seed: int = 0,
    baseline: str = "none",
) -> dict[str, PoseDataset]:
    """experiments.json 의 정의대로 train/val/test 데이터셋을 만든다.

    Args:
        exp: "single_split" | "yja_holdout" | "loso" | "sealed"
        fold: exp="loso" 일 때 "test_ajh" 등
        baseline: 빈방 기준선 모드 "none" | "sub" | "sub_z" (SiteBaseline 참조)
    """
    base = SiteBaseline(baseline)
    cache = cache_mod.open_cache()
    link_ok = _link_ok_matrix(cache)
    experiments = load_experiments()

    if exp == "sealed":
        name = fold or "yja_E02"
        spec = experiments["sealed"][name]
        rows = select_sealed_rows(cache.index, spec)
        return {"test": PoseDataset(rows, cache, link_ok, train=False, seed=seed,
                                    baseline=base)}

    if exp == "yja_holdout":
        cfg = experiments["yja_holdout"]
        out = {}
        for split in ("train", "val"):
            rows = select_rows(cache.index, cfg[split])
            out[split] = PoseDataset(
                rows, cache, link_ok, train=(split == "train"),
                dropout=dropout, seed=seed, baseline=base,
            )
        test_rows = select_sealed_rows(cache.index, cfg["test"])
        out["test"] = PoseDataset(
            test_rows, cache, link_ok, train=False, seed=seed, baseline=base,
        )
        return out

    if exp == "loso":
        if not fold:
            raise ValueError("exp='loso' 에는 fold 가 필요하다 (예: test_ajh)")
        cfg = experiments["loso"][fold]
    elif exp == "single_split":
        cfg = experiments["single_split"]
    else:
        raise ValueError(f"unknown exp {exp!r}")

    out = {}
    for split in ("train", "val", "test"):
        rows = select_rows(cache.index, cfg[split])
        out[split] = PoseDataset(rows, cache, link_ok,
                                 train=(split == "train"),
                                 dropout=dropout, seed=seed, baseline=base)
    return out


def absence_rows(cache: cache_mod.Cache, subject: str, environment: str) -> np.ndarray:
    """PerLinkNorm 보정용 — 그 사이트의 사람 없는(absence) trial 행 번호.

    배포의 '빈 방 10초 녹음'에 해당하는 데이터다. GT 가 없어 학습에는 못 쓰지만
    링크 통계를 잡는 데는 이게 정답이다.
    """
    ix = cache.index
    m = ((ix.subject == subject) & (ix.environment == environment)
         & (ix.task == C.TASK_CLS) & ix.cache_ok)
    return np.flatnonzero(m.to_numpy())
