"""학습 루프.

구조는 legacy `scripts/train_csi_to_pose.py` 792~910행에서 가져왔다
(AdamW, early stopping, 최고 체크포인트 저장, 학습 곡선 기록). 달라진 부분:
  - 입력이 [B,T,3,114,2] + link_mask 로 4차원 + 마스크
  - 손실이 pose/root/bone/class/risk 다섯 항
  - PerLinkNorm 통계를 학습 전에 train 셋으로 한 번 맞춘다
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from . import contract as C
from . import losses as L
from . import nets


@dataclass
class TrainConfig:
    arch: str = "tcn"
    hidden: int = 96
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_blocks: int = 2
    heads: int = 4
    graph_blocks: int = 2
    decoder: str = "tree"
    dropout: float = 0.1

    #: 3링크를 합치는 방식.
    #:   "gate"   softmax 가중평균 (기본). 링크 간 차이가 상쇄된다
    #:   "concat" 이어붙이기. 링크 정체성 보존 -> 삼각측량 가능
    fusion: str = "gate"

    #: LinkEncoder 에서 링크별 곱셈 변조(FiLM)를 쓸 것인가.
    #: False 면 링크 구분이 상수 덧셈(link_emb)뿐이라 세 링크가 같은 비선형 특징을 쓴다.
    film: bool = False

    epochs: int = 400          # 상한. 실제 종료는 early stopping(patience)이 결정한다
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    num_workers: int = 0
    seed: int = 7

    lambda_root: float = 1.0
    lambda_bone: float = 0.1
    lambda_cls: float = 0.2
    lambda_risk: float = 0.1
    lambda_velocity: float = 0.0
    lambda_motion: float = 0.0
    lambda_acceleration: float = 0.0
    lambda_jerk: float = 0.0
    lambda_impact: float = 0.0
    lambda_coarse: float = 0.0
    lambda_displacement: float = 0.0
    lambda_flow: float = 0.0
    lambda_contact: float = 0.0
    lambda_phase: float = 0.0
    lambda_foot_slide: float = 0.0
    lambda_floor: float = 0.0
    lambda_domain: float = 0.0
    lambda_supcon: float = 0.0
    lambda_latent: float = 0.0
    motion_weight: float = 0.0
    group_dro_eta: float = 0.0
    balanced_batches: bool = False
    rf_augment: bool = False
    frequency_tokens: int = 12
    geometry_path: str | None = None
    motion_prior_path: str | None = None
    init_checkpoint: str | None = None
    backbone_lr_scale: float = 1.0
    refiner_warmup_epochs: int = 0
    refiner_joint_scale: tuple[float, ...] | None = None
    flow_steps: int = 4
    flow_noise: float = 0.25
    domain_grl: float = 0.2
    weight_average_start: int = 10

    #: 사이트별 빈방 기준선 제거 (Phase 2). "none" | "sub" | "sub_z"
    #: dataset.SiteBaseline 참조. PerLinkNorm(전역 정규화)은 이 뒤의 2차 정규화로 유지된다.
    baseline: str = "none"

    #: PerLinkNorm 통계를 무엇으로 맞출 것인가.
    #:   "train"   train 셋 전체 (기본)
    #:   "absence" 사람 없는 trial 만 — 배포 절차와 동일한 조건. 캘리브레이션 실험용
    norm_source: str = "train"
    norm_batches: int = 20


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CrossDomainBatchSampler(Sampler[list[int]]):
    """Build class-balanced batches with cross-domain positive pairs."""

    def __init__(self, dataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.groups: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        domains = dataset._domain_id[dataset.rows]
        labels = dataset.index.class_id.to_numpy(dtype=np.int64)
        for local_index, (label, domain) in enumerate(zip(labels, domains)):
            if domain >= 0:
                self.groups[int(label)][int(domain)].append(local_index)
        self.pairable = [
            label for label, domain_rows in self.groups.items()
            if len(domain_rows) >= 2
        ]
        self.all_indices = np.arange(len(dataset), dtype=np.int64)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1009)
        self.epoch += 1
        pairs = max(1, self.batch_size // 2)
        for _ in range(len(self)):
            batch: list[int] = []
            labels = rng.choice(self.pairable, size=pairs, replace=True)
            for label in labels:
                domains = list(self.groups[int(label)])
                selected = rng.choice(domains, size=2, replace=False)
                for domain in selected:
                    batch.append(int(rng.choice(self.groups[int(label)][int(domain)])))
            while len(batch) < self.batch_size:
                batch.append(int(rng.choice(self.all_indices)))
            yield batch[:self.batch_size]


class GroupDRO:
    def __init__(self, n_groups: int, eta: float, device):
        self.eta = eta
        self.weights = torch.ones(n_groups, device=device)

    def __call__(self, losses: torch.Tensor, domains: torch.Tensor,
                 risks: torch.Tensor) -> torch.Tensor:
        valid = domains >= 0
        if not valid.any() or self.eta <= 0:
            return losses.mean()
        group_ids = domains[valid] * C.N_RISK + risks[valid]
        unique = torch.unique(group_ids)
        means = []
        for group in unique:
            means.append(losses[valid][group_ids == group].mean())
        means = torch.stack(means)
        with torch.no_grad():
            self.weights[unique] *= torch.exp(self.eta * means.detach())
        selected = self.weights[unique]
        selected = selected / selected.sum().clamp_min(1e-8)
        return (selected * means).sum()


@torch.no_grad()
def fit_bone_lengths(model, dataset) -> None:
    if not hasattr(model, "set_bone_lengths"):
        return
    arrays = dataset.cache.arrays
    totals = np.zeros(C.N_JOINTS, dtype=np.float64)
    counts = np.zeros(C.N_JOINTS, dtype=np.float64)
    for start in range(0, len(dataset.rows), 64):
        rows = dataset.rows[start:start + 64]
        pose = np.asarray(arrays["pose_rel"][rows], dtype=np.float32)
        valid = np.asarray(arrays["valid"][rows], dtype=bool)
        for child, parent in enumerate(C.JOINT_PARENTS):
            if parent < 0:
                continue
            length = np.linalg.norm(pose[:, :, child] - pose[:, :, parent], axis=-1)
            totals[child] += length[valid].sum()
            counts[child] += valid.sum()
    lengths = totals / np.maximum(counts, 1.0)
    lengths[C.ROOT_JOINT] = 0.0
    model.set_bone_lengths(torch.tensor(lengths, dtype=torch.float32, device=next(model.parameters()).device))


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def fit_norm(model: nets.PoseNet | nets.GraphPoseNet, loader: DataLoader, device,
             max_batches: int = 20) -> None:
    """PerLinkNorm 통계 추정. 배포에서 '빈 방 10초'로 하는 것과 같은 연산이다."""
    csis, masks = [], []
    for i, b in enumerate(loader):
        csis.append(b["csi"])
        masks.append(b["link_mask"])
        if i + 1 >= max_batches:
            break
    csi = torch.cat(csis).to(device)
    mask = torch.cat(masks).to(device)
    model.norm.fit(csi, mask)


@torch.no_grad()
def evaluate(model: nets.PoseNet, loader: DataLoader, loss_fn: L.PoseLoss,
             device) -> dict:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    impact_sum = 0.0
    impact_count = 0
    cls_ok = risk_ok = cls_n = 0
    for b in loader:
        b = _to_device(b, device)
        out = model(b["csi"], b["link_mask"])
        if (hasattr(model, "encode_pose_target")
                and bool(getattr(model, "motion_prior_loaded", torch.zeros(1)).item())):
            out["target_motion_latent"] = model.encode_pose_target(
                b["pose_rel"], b["valid"]
            )
        _, parts = loss_fn(out, b)
        parts = {key: value for key, value in parts.items() if not key.startswith("_")}
        bs = b["csi"].shape[0]
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        agg["mpjpe"] = agg.get("mpjpe", 0.0) + L.mpjpe(
            out["pose_rel"], b["pose_rel"], b["valid"]) * bs
        agg["root_err"] = agg.get("root_err", 0.0) + L.root_error(
            out["root"], b["root"], b["valid"]) * bs
        agg["distal_mpjpe"] = agg.get("distal_mpjpe", 0.0) + L.distal_mpjpe(
            out["pose_rel"], b["pose_rel"], b["valid"]
        ) * bs
        impact_mask = L.impact_window(
            b["pose_rel"], b["root"], b["valid"], b["risk_id"]
        )
        if impact_mask.any():
            predicted = out["pose_rel"] + out["root"][:, :, None]
            target = b["pose_rel"] + b["root"][:, :, None]
            distance = torch.linalg.vector_norm(predicted - target, dim=-1)
            impact_sum += float(
                (distance * impact_mask[..., None]).sum().detach()
            )
            impact_count += int(impact_mask.sum()) * C.N_JOINTS
        cls_ok += int((out["class_logits"].argmax(-1) == b["class_id"]).sum())
        risk_ok += int((out["risk_logits"].argmax(-1) == b["risk_id"]).sum())
        cls_n += bs
        n += bs
    res = {k: v / max(n, 1) for k, v in agg.items()}
    res["class_acc"] = cls_ok / max(cls_n, 1)
    res["risk_acc"] = risk_ok / max(cls_n, 1)
    res["impact_mpjpe"] = impact_sum / max(impact_count, 1)
    return res


def _selection_score(metrics: dict) -> float:
    """Favor accurate impact/distal reconstruction without accepting static poses."""
    return (
        metrics["mpjpe"]
        + 0.15 * metrics.get("impact_mpjpe", 0.0)
        + 0.10 * metrics.get("distal_mpjpe", 0.0)
        + 0.15 * metrics["root_err"]
        + 0.02 * metrics.get("velocity", 0.0)
        + 0.002 * metrics.get("acceleration", 0.0)
        + 0.0005 * metrics.get("jerk", 0.0)
        + 0.03 * metrics.get("displacement", 0.0)
    )


def train(datasets: dict, cfg: TrainConfig, out_dir: Path,
          device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = {}
    for split, dataset in datasets.items():
        if split == "train" and cfg.balanced_batches:
            sampler = CrossDomainBatchSampler(dataset, cfg.batch_size, cfg.seed)
            loaders[split] = DataLoader(
                dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
                pin_memory=(device == "cuda"),
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=(split == "train"),
                num_workers=cfg.num_workers, pin_memory=(device == "cuda"),
                drop_last=False,
            )

    model_kwargs = {
        "hidden": cfg.hidden,
        "n_blocks": cfg.n_blocks,
        "dropout": cfg.dropout,
    }
    if cfg.arch in {
        "graphformer", "robust_graphformer", "impact_graphformer", "latent_flow"
    }:
        model_kwargs.update(
            heads=cfg.heads, graph_blocks=cfg.graph_blocks, decoder=cfg.decoder,
            domain_grl=cfg.domain_grl,
        )
        if cfg.arch == "impact_graphformer":
            model_kwargs["refiner_joint_scale"] = cfg.refiner_joint_scale
        elif cfg.arch == "latent_flow":
            model_kwargs.update(
                flow_steps=cfg.flow_steps, flow_noise=cfg.flow_noise
            )
    elif cfg.arch == "v3":
        model_kwargs.update(
            heads=cfg.heads, graph_blocks=cfg.graph_blocks,
            frequency_tokens=cfg.frequency_tokens,
            geometry_path=cfg.geometry_path, domain_grl=cfg.domain_grl,
        )
    else:
        model_kwargs.update(
            dilations=tuple(cfg.dilations), fusion=cfg.fusion, film=cfg.film
        )
    model = nets.build_model(cfg.arch, **model_kwargs).to(device)
    print(f"[train] {model.describe()}  device={device}")

    if cfg.init_checkpoint:
        init_path = Path(cfg.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"initial checkpoint not found: {init_path}")
        initial = torch.load(init_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        if hasattr(model, "allows_warm_start_missing"):
            invalid_missing = [
                key for key in missing if not model.allows_warm_start_missing(key)
            ]
        else:
            invalid_missing = [
                key for key in missing if not key.startswith("pose_refiner.")
            ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "incompatible initial checkpoint: "
                f"missing={invalid_missing}, unexpected={list(unexpected)}"
            )
        print(
            f"[train] warm-started {init_path} "
            f"(new parameters: {len(missing)})"
        )

    if cfg.motion_prior_path and hasattr(model, "load_motion_prior"):
        prior_path = Path(cfg.motion_prior_path)
        if not prior_path.exists():
            raise FileNotFoundError(f"motion prior not found: {prior_path}")
        prior = torch.load(prior_path, map_location=device, weights_only=False)
        model.load_motion_prior(prior)
        print(f"[train] loaded motion prior: {prior_path}")
    fit_bone_lengths(model, datasets["train"])

    augment_state = getattr(datasets["train"].dropout, "rf_augment", False)
    datasets["train"].dropout.rf_augment = False
    norm_loader = DataLoader(
        datasets["train"], batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device == "cuda"),
    )
    if cfg.init_checkpoint:
        print("[train] preserving PerLinkNorm statistics from the initial checkpoint")
    else:
        fit_norm(model, norm_loader, device, cfg.norm_batches)
    datasets["train"].dropout.rf_augment = augment_state
    print(f"[train] PerLinkNorm 통계 추정 완료 "
          f"(mu 범위 {float(model.norm.mu.min()):.2f}~{float(model.norm.mu.max()):.2f}, "
          f"sigma 중앙 {float(model.norm.sigma.median()):.2f})")

    loss_fn = L.PoseLoss(
        class_counts=datasets["train"].class_counts(),
        risk_counts=datasets["train"].risk_counts(),
        lambda_root=cfg.lambda_root, lambda_bone=cfg.lambda_bone,
        lambda_cls=cfg.lambda_cls, lambda_risk=cfg.lambda_risk,
        lambda_velocity=cfg.lambda_velocity, lambda_motion=cfg.lambda_motion,
        lambda_acceleration=cfg.lambda_acceleration,
        lambda_jerk=cfg.lambda_jerk, lambda_impact=cfg.lambda_impact,
        lambda_coarse=cfg.lambda_coarse,
        lambda_displacement=cfg.lambda_displacement,
        lambda_flow=cfg.lambda_flow,
        lambda_contact=cfg.lambda_contact, lambda_phase=cfg.lambda_phase,
        lambda_foot_slide=cfg.lambda_foot_slide, lambda_floor=cfg.lambda_floor,
        lambda_domain=cfg.lambda_domain, lambda_supcon=cfg.lambda_supcon,
        lambda_latent=cfg.lambda_latent,
        motion_weight=cfg.motion_weight,
        device=device,
    ).to(device)
    group_dro = GroupDRO(9 * C.N_RISK, cfg.group_dro_eta, device)

    adaptation_prefixes = getattr(
        model, "adaptation_parameter_prefixes", ("pose_refiner.",)
    )
    permanently_frozen = {
        id(parameter) for parameter in model.parameters()
        if not parameter.requires_grad
    }
    refiner_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(adaptation_prefixes)
    ]
    refiner_ids = {id(parameter) for parameter in refiner_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in refiner_ids
    ]
    parameter_groups = [
        {"params": backbone_parameters, "lr": cfg.lr * cfg.backbone_lr_scale}
    ]
    if refiner_parameters:
        parameter_groups.append({"params": refiner_parameters, "lr": cfg.lr})
    opt = torch.optim.AdamW(parameter_groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    averaged = None
    if cfg.weight_average_start > 0:
        # PerLinkNorm contains boolean fitted flags, which cannot be arithmetically
        # averaged. With use_buffers=False PyTorch copies buffers from the live model.
        averaged = torch.optim.swa_utils.AveragedModel(model, use_buffers=False)

    history = []
    best = math.inf
    best_path = out_dir / "best_model.pt"
    patience = 0
    t0 = time.perf_counter()

    if cfg.init_checkpoint:
        initial_val = evaluate(model, loaders["val"], loss_fn, device)
        best = _selection_score(initial_val)
        torch.save(
            {
                "model": model.state_dict(), "cfg": asdict(cfg), "epoch": 0,
                "val_mpjpe": initial_val["mpjpe"],
                "validation_selection_score": best,
                "warm_start_baseline": True,
                "preproc_version": C.PREPROC_VERSION,
                "joint_names": list(C.JOINT_NAMES),
            },
            best_path,
        )
        print(
            f"[train] epoch 0 baseline | MPJPE {initial_val['mpjpe']*100:.2f}cm "
            f"| impact {initial_val['impact_mpjpe']*100:.2f}cm "
            f"| select {best:.4f}"
        )

    for epoch in range(1, cfg.epochs + 1):
        refiner_only = bool(
            cfg.init_checkpoint and refiner_parameters
            and epoch <= cfg.refiner_warmup_epochs
        )
        for name, parameter in model.named_parameters():
            if id(parameter) in permanently_frozen:
                parameter.requires_grad_(False)
            else:
                parameter.requires_grad_(
                    not refiner_only or name.startswith(adaptation_prefixes)
                )
        if hasattr(datasets["train"], "set_epoch"):
            datasets["train"].set_epoch(epoch)
        model.train()
        tr = {}
        n = 0
        for b in loaders["train"]:
            b = _to_device(b, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out = model(b["csi"], b["link_mask"])
                if hasattr(model, "flow_matching_per_sample"):
                    out["flow_per_sample"] = model.flow_matching_per_sample(out, b)
                if (hasattr(model, "encode_pose_target")
                        and bool(getattr(model, "motion_prior_loaded", torch.zeros(1)).item())):
                    out["target_motion_latent"] = model.encode_pose_target(
                        b["pose_rel"], b["valid"]
                    )
                loss, parts = loss_fn(out, b)
                per_sample = parts.pop("_per_sample_total")
                supcon_total = parts.pop("_supcon_total")
                if cfg.group_dro_eta > 0:
                    loss = group_dro(
                        per_sample, b["domain_id"], b["risk_id"]
                    ) + supcon_total
                    parts["total"] = float(loss.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            bs = b["csi"].shape[0]
            for k, v in parts.items():
                tr[k] = tr.get(k, 0.0) + v * bs
            n += bs
        tr = {f"train_{k}": v / max(n, 1) for k, v in tr.items()}
        va = {f"val_{k}": v for k, v in evaluate(model, loaders["val"], loss_fn, device).items()}
        sched.step()
        if averaged is not None and epoch >= cfg.weight_average_start:
            averaged.update_parameters(model)

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **tr, **va}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        # Position remains primary; root and dynamics break ties between models that
        # predict a plausible but nearly static mean pose.
        score = _selection_score({key.removeprefix("val_"): value
                                  for key, value in va.items()})
        if score < best:
            best, patience = score, 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "epoch": epoch, "val_mpjpe": va["val_mpjpe"],
                        "validation_selection_score": score,
                        "preproc_version": C.PREPROC_VERSION,
                        "joint_names": list(C.JOINT_NAMES)}, best_path)
        else:
            patience += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"  ep {epoch:3d} | train {tr['train_total']:.4f} "
                  f"| val {va['val_total']:.4f} | MPJPE {va['val_mpjpe']*100:.2f}cm "
                  f"| root {va['val_root_err']*100:.2f}cm "
                  f"| impact {va['val_impact_mpjpe']*100:.2f}cm "
                  f"distal {va['val_distal_mpjpe']*100:.2f}cm "
                  f"| cls {va['val_class_acc']:.3f} risk {va['val_risk_acc']:.3f} "
                  f"| select {score:.4f} best {best:.4f} | pat {patience}/{cfg.patience}")
        if patience >= cfg.patience:
            print(f"  early stop at epoch {epoch} (best selection score {best:.4f})")
            break

    if averaged is not None and int(averaged.n_averaged.item()) > 0:
        averaged.module.eval()
        avg_val = evaluate(averaged.module, loaders["val"], loss_fn, device)
        avg_score = _selection_score(avg_val)
        averaged_path = out_dir / "averaged_model.pt"
        torch.save(
            {
                "model": averaged.module.state_dict(), "cfg": asdict(cfg),
                "epoch": history[-1]["epoch"], "val_mpjpe": avg_val["mpjpe"],
                "validation_selection_score": avg_score,
                "weight_averaged": True,
                "n_averaged": int(averaged.n_averaged.item()),
                "preproc_version": C.PREPROC_VERSION,
                "joint_names": list(C.JOINT_NAMES),
            },
            averaged_path,
        )
        print(
            f"  weight average ({int(averaged.n_averaged.item())} epochs) | "
            f"MPJPE {avg_val['mpjpe']*100:.2f}cm | select {avg_score:.4f}"
        )
        if avg_score < best:
            best = avg_score
            torch.save(
                torch.load(averaged_path, map_location="cpu", weights_only=False),
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    result = {"best_epoch": ckpt["epoch"], "best_val_mpjpe": ckpt["val_mpjpe"],
              "best_validation_selection_score": ckpt["validation_selection_score"],
              "weight_averaged": bool(ckpt.get("weight_averaged", False)),
              "n_averaged": int(ckpt.get("n_averaged", 0)),
              "train_seconds": round(time.perf_counter() - t0, 1)}
    for split in ("val", "test"):
        if split in loaders:
            result[split] = evaluate(model, loaders[split], loss_fn, device)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"\n[train] best epoch {result['best_epoch']} "
          f"({result['train_seconds']:.0f}s)")
    for split in ("val", "test"):
        if split in result:
            r = result[split]
            print(f"  {split:5s} MPJPE {r['mpjpe']*100:.2f}cm  root {r['root_err']*100:.2f}cm  "
                  f"cls {r['class_acc']:.3f}  risk {r['risk_acc']:.3f}")
    return result
