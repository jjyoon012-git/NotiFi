"""Train KP5-MPR-S: a CSI-to-motion embedding retrieval selector."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..motion_retrieval import TemporalMotionSelector
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_motion_retrieval_oracle import (
    _canonicalize,
    _load_pose_arrays,
    _metric_batch,
    _render,
)
from .diagnose_observability import pose_only, report_path
from .evaluate_motion_retrieval_pose import _load_model
from .train_kinetic_pose import CoarsePoseStore, pose_selection_score


def motion_descriptor(bank: torch.Tensor, bins: int = 38,
                      mode: str = "position_velocity") -> torch.Tensor:
    """Compact phase descriptor with an optional dynamics-only target space."""
    joint_weight = bank.new_ones(C.N_JOINTS)
    distal = (
        C.JOINT_GROUPS["head"]
        + C.JOINT_GROUPS["left_arm"][-1:]
        + C.JOINT_GROUPS["right_arm"][-1:]
        + C.JOINT_GROUPS["left_leg"][-2:]
        + C.JOINT_GROUPS["right_leg"][-2:]
    )
    joint_weight[list(sorted(set(distal)))] = 1.5
    weighted = bank * joint_weight[None, None, :, None]
    position = F.adaptive_avg_pool1d(
        weighted.flatten(2).transpose(1, 2), bins
    ).transpose(1, 2)
    velocity = position[:, 1:] - position[:, :-1]
    if mode == "dynamic":
        displacement = position - position[:, :1]
        velocity = torch.cat((torch.zeros_like(position[:, :1]), velocity), dim=1)
        acceleration = torch.zeros_like(velocity)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        return torch.cat((
            0.50 * displacement.flatten(1),
            velocity.flatten(1),
            0.50 * acceleration.flatten(1),
        ), dim=-1)
    if mode != "position_velocity":
        raise ValueError(f"Unsupported motion descriptor mode: {mode}")
    return torch.cat((position.flatten(1), 0.35 * velocity.flatten(1)), dim=-1)


def fit_motion_pca(descriptor: torch.Tensor, dimension: int,
                   device: str) -> dict[str, torch.Tensor]:
    mean = descriptor.mean(0)
    centered = descriptor - mean
    q = min(int(dimension), min(centered.shape) - 1)
    _, _, components = torch.pca_lowrank(
        centered.to(device), q=q, center=False, niter=5
    )
    components = components.cpu()
    score = centered @ components
    scale = score.std(0).clamp_min(1e-4)
    return {"mean": mean, "components": components, "scale": scale}


def project_motion(descriptor: torch.Tensor,
                   pca: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        (descriptor - pca["mean"]) @ pca["components"]
    ) / pca["scale"]


@torch.no_grad()
def extract_features(model, dataset: QualityWeightedDataset,
                     coarse_store: CoarsePoseStore, path: Path,
                     device: str, batch_size: int, protocol: str) -> dict:
    expected = torch.from_numpy(dataset.target.rows).long()
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("protocol") != protocol:
            raise RuntimeError(f"feature cache protocol mismatch: {path}")
        required = {
            "features", "frame_mask", "baseline_pose",
            "base_action_logits", "base_risk_logits", "contact_logits",
            "phase_logits", "motion_activity", "rows",
        }
        if required.issubset(cached) and torch.equal(cached["rows"], expected):
            return cached
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    values: dict[str, list[torch.Tensor]] = {
        "features": [], "frame_mask": [], "baseline_pose": [],
        "base_action_logits": [], "base_risk_logits": [], "rows": [],
        "contact_logits": [], "phase_logits": [], "motion_activity": [],
    }
    for batch in loader:
        output = model(
            batch["csi"].to(device), batch["link_mask"].to(device),
            coarse_pose=coarse_store.lookup(batch["row"], device),
        )
        values["features"].append(
            output["conditioned_features"].detach().cpu().half()
        )
        values["frame_mask"].append(batch["link_mask"].any(-1).cpu())
        values["baseline_pose"].append(output["pose_rel"].detach().cpu().half())
        values["base_action_logits"].append(
            output["action_logits"].detach().cpu().half()
        )
        values["base_risk_logits"].append(
            output["risk_logits"].detach().cpu().half()
        )
        values["contact_logits"].append(
            output["contact_logits"].detach().cpu().half()
        )
        values["phase_logits"].append(
            output["phase_logits"].detach().cpu().half()
        )
        values["motion_activity"].append(
            output["motion_activity"].detach().cpu().half()
        )
        values["rows"].append(batch["row"].long())
    result = {key: torch.cat(items) for key, items in values.items()}
    result.update({"protocol": protocol, "source": "frozen_KP4_DCC"})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def _class_weights(labels: torch.Tensor, classes: int,
                   device: str) -> torch.Tensor:
    count = torch.bincount(labels, minlength=classes).float()
    weight = count.sum() / count.clamp_min(1.0)
    return (weight / weight.mean()).to(device)


def _augment_features(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not features.requires_grad:
        features = features.clone()
    features = features + 0.015 * torch.randn_like(features)
    if features.shape[1] > 4:
        temporal_drop = torch.rand(
            features.shape[:2], device=features.device
        ) < 0.035
        features = features.masked_fill(
            (temporal_drop & mask)[..., None], 0.0
        )
    return features


def train_epoch(model: TemporalMotionSelector, loader: DataLoader,
                optimizer, scaler, scheduler, device: str,
                class_weight: torch.Tensor,
                risk_weight: torch.Tensor) -> dict:
    model.train()
    totals: dict[str, list[float]] = {}
    for features, mask, target, class_id, risk_id in loader:
        features = features.to(device).float()
        mask = mask.to(device)
        target = target.to(device).float()
        class_id = class_id.to(device)
        risk_id = risk_id.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            output = model(_augment_features(features, mask), mask)
            predicted = output["motion_embedding"]
            regression = F.smooth_l1_loss(predicted, target, beta=0.5)
            cosine = (1.0 - F.cosine_similarity(predicted, target)).mean()
            contrastive_logits = (
                F.normalize(predicted, dim=-1)
                @ F.normalize(target, dim=-1).T
            ) / 0.08
            contrastive = F.cross_entropy(
                contrastive_logits,
                torch.arange(len(target), device=device),
            )
            action = F.cross_entropy(
                output["action_logits"], class_id,
                weight=class_weight, label_smoothing=0.04,
            )
            risk = F.cross_entropy(
                output["risk_logits"], risk_id,
                weight=risk_weight, label_smoothing=0.03,
            )
            loss = (
                regression + 0.20 * cosine + 0.10 * contrastive
                + 0.25 * action + 0.15 * risk
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        current = {
            "total": loss, "regression": regression, "cosine": cosine,
            "contrastive": contrastive, "action": action, "risk": risk,
        }
        for key, value in current.items():
            totals.setdefault(key, []).append(float(value.detach()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def predict_selector(model: TemporalMotionSelector, cache: dict,
                     batch_size: int, device: str) -> dict:
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


def _select_indices(latent_distance: torch.Tensor,
                    pose_distance: torch.Tensor,
                    bank_class: torch.Tensor,
                    action_logits: torch.Tensor,
                    mode: str) -> torch.Tensor:
    result = []
    for item in range(len(latent_distance)):
        latent = latent_distance[item]
        pose = pose_distance[item]
        if "fused_top2" in mode:
            classes = action_logits[item].topk(2).indices
            keep = (bank_class == classes[0]) | (bank_class == classes[1])
        elif "base_top2" in mode:
            classes = action_logits[item].topk(2).indices
            keep = (bank_class == classes[0]) | (bank_class == classes[1])
        else:
            keep = torch.ones_like(bank_class, dtype=torch.bool)
        if mode.startswith("baseline"):
            score = pose
        elif mode.startswith("learned"):
            score = latent
        else:
            alpha = float(mode.rsplit("_", 1)[-1]) / 100.0
            latent_scale = torch.quantile(latent[keep], 0.50).clamp_min(1e-5)
            pose_scale = torch.quantile(pose[keep], 0.50).clamp_min(1e-5)
            score = alpha * latent / latent_scale + (1.0 - alpha) * pose / pose_scale
        score = score.masked_fill(~keep, float("inf"))
        result.append(score.argmin())
    return torch.stack(result)


def evaluate_retrieval(model: TemporalMotionSelector, cache: dict,
                       target_pose: torch.Tensor, target_valid: torch.Tensor,
                       target_class: torch.Tensor, target_risk: torch.Tensor,
                       train_bank: torch.Tensor, train_embedding: torch.Tensor,
                       train_class: torch.Tensor, pca: dict,
                       batch_size: int, device: str,
                       descriptor_mode: str = "position_velocity") -> dict:
    output = predict_selector(model, cache, batch_size, device)
    baseline = cache["baseline_pose"].float()
    baseline_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(baseline, target_valid)
    ])
    baseline_embedding = project_motion(
        motion_descriptor(baseline_bank, mode=descriptor_mode), pca
    )
    latent_distance = torch.cdist(output["embedding"], train_embedding)
    pose_distance = torch.cdist(baseline_embedding, train_embedding)
    fused_action = output["action_logits"] + cache["base_action_logits"].float()
    modes = (
        "baseline_base_top2", "learned_global", "learned_base_top2",
        "learned_fused_top2", "hybrid_base_top2_25",
        "hybrid_base_top2_50", "hybrid_base_top2_75",
    )
    metrics = {
        "locked_kp2_dh": _metric_batch(
            baseline, target_pose, target_valid, target_risk
        )
    }
    selected_rows = {}
    for mode in modes:
        action = (
            fused_action if "fused_top2" in mode
            else cache["base_action_logits"].float()
        )
        indices = _select_indices(
            latent_distance, pose_distance, train_class, action, mode
        )
        selected_rows[mode] = indices
        candidate = torch.stack([
            _render(train_bank[int(index)], valid, C.CACHE_FRAMES)
            for index, valid in zip(indices, target_valid)
        ])
        for strength in (0.25, 0.50, 0.75):
            name = f"{mode}_blend_{int(strength * 100):03d}"
            metrics[name] = _metric_batch(
                (1.0 - strength) * baseline + strength * candidate,
                target_pose, target_valid, target_risk,
            )
    scores = {name: pose_selection_score(value) for name, value in metrics.items()}
    best_name = min(scores, key=scores.get)
    return {
        "selection": {"name": best_name, "score": scores[best_name]},
        "scores": scores,
        "metrics": metrics,
        "embedding_mae": float(
            (output["embedding"] - project_motion(
                motion_descriptor(torch.stack([
                    _canonicalize(pose, valid, C.CACHE_FRAMES)
                    for pose, valid in zip(target_pose, target_valid)
                ]), mode=descriptor_mode), pca
            )).abs().mean()
        ),
        "selector_action_accuracy": float(
            (output["action_logits"].argmax(-1) == target_class).float().mean()
        ),
        "fused_action_accuracy": float(
            (fused_action.argmax(-1) == target_class).float().mean()
        ),
        "selector_risk_accuracy": float(
            (output["risk_logits"].argmax(-1) == target_risk).float().mean()
        ),
        "selected_rows": selected_rows.get(
            best_name.rsplit("_blend_", 1)[0],
            torch.full((len(target_pose),), -1, dtype=torch.long),
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--pca-dim", type=int, default=96)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--danger-weight", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--descriptor-mode", choices=("position_velocity", "dynamic"),
        default="position_velocity",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "kp4_dcc_staged_seed17"
        / "deployment_model.pt",
    )
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp5_mpr_selector_seed17",
    )
    parser.add_argument(
        "--feature-root", type=Path, default=None,
        help="Reuse protocol-locked frozen feature caches from another run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = build_datasets(exp=args.exp, baseline="sub", seed=args.seed)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_pose, train_valid, train_class, train_risk = _load_pose_arrays(train)
    val_pose, val_valid, val_class, val_risk = _load_pose_arrays(validation)
    train_bank = torch.stack([
        _canonicalize(pose, valid, C.CACHE_FRAMES)
        for pose, valid in zip(train_pose, train_valid)
    ])
    descriptor = motion_descriptor(train_bank, mode=args.descriptor_mode)
    pca = fit_motion_pca(descriptor, args.pca_dim, device)
    train_embedding = project_motion(descriptor, pca)

    frozen, source_checkpoint = _load_model(args.checkpoint, device)
    raw_cache = torch.load(args.coarse_cache, map_location="cpu", weights_only=False)
    coarse_store = CoarsePoseStore(raw_cache["rows"], raw_cache["pose"])
    feature_root = args.feature_root or args.run_dir
    train_cache = extract_features(
        frozen, train, coarse_store, feature_root / "train_features.pt",
        device, args.batch_size, args.exp,
    )
    val_cache = extract_features(
        frozen, validation, coarse_store, feature_root / "val_features.pt",
        device, args.batch_size, args.exp,
    )
    del frozen
    if device == "cuda":
        torch.cuda.empty_cache()

    model = TemporalMotionSelector(
        input_dim=train_cache["features"].shape[-1],
        embedding_dim=train_embedding.shape[-1], width=args.width,
        layers=args.layers, heads=args.heads,
    ).to(device)
    weights = train.sampler_weights()
    weights *= torch.where(
        train_risk == 2, torch.tensor(args.danger_weight, dtype=torch.double),
        torch.tensor(1.0, dtype=torch.double),
    )
    sampler = WeightedRandomSampler(
        weights, len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    tensor_data = TensorDataset(
        train_cache["features"], train_cache["frame_mask"],
        train_embedding, train_class, train_risk,
    )
    loader = DataLoader(
        tensor_data, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=device == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    class_weight = _class_weights(train_class, C.N_CLASSES, device)
    risk_weight = _class_weights(train_risk, C.N_RISK, device)
    risk_weight[2] *= 1.5

    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        trained = train_epoch(
            model, loader, optimizer, scaler, scheduler, device,
            class_weight, risk_weight,
        )
        validation_result = evaluate_retrieval(
            model, val_cache, val_pose, val_valid, val_class, val_risk,
            train_bank, train_embedding, train_class, pca,
            args.batch_size * 2, device, args.descriptor_mode,
        )
        row = {
            "epoch": epoch, "train": trained,
            "validation_selection": validation_result["selection"],
            "embedding_mae": validation_result["embedding_mae"],
            "selector_action_accuracy": validation_result["selector_action_accuracy"],
            "fused_action_accuracy": validation_result["fused_action_accuracy"],
            "selector_risk_accuracy": validation_result["selector_risk_accuracy"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = validation_result["selection"]["score"]
        if best is None or score < best["score"] - 1e-5:
            best = {
                "epoch": epoch, "score": score,
                "state": copy.deepcopy(model.state_dict()),
                "validation": validation_result,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best is not None
    model.load_state_dict(best["state"])
    selected_rows = best["validation"].pop("selected_rows")
    result = {
        "run": "KP5-MPR-S-EXP01",
        "model_family": "NotiFi-KP5",
        "candidate_version": "KP5-MPR-S",
        "status": "validation_selected_test_untouched",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "config": vars(args) | {
            "checkpoint": report_path(args.checkpoint),
            "coarse_cache": report_path(args.coarse_cache),
            "run_dir": report_path(args.run_dir),
            "feature_root": report_path(feature_root),
        },
        "architecture": {
            "frozen_features": "KP4-DCC conditioned 304-frame CSI features",
            "selector": "38-bin temporal Transformer",
            "motion_space": f"train-only {train_embedding.shape[-1]}D PCA",
            "motion_descriptor": args.descriptor_mode,
            "candidate_bank": "train-only normalized-phase GT trajectories",
            "selection_modes": "learned, locked-pose, action top-2, hybrid",
        },
        "selection": {
            "epoch": best["epoch"], "score": best["score"],
            "name": best["validation"]["selection"]["name"],
        },
        "validation": best["validation"],
        "history": history,
        "source_run": source_checkpoint.get("run"),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": result["run"], "protocol": args.exp,
        "model": best["state"],
        "model_config": {
            "input_dim": train_cache["features"].shape[-1],
            "embedding_dim": train_embedding.shape[-1],
            "width": args.width, "layers": args.layers, "heads": args.heads,
        },
        "pca": pca | {"descriptor_mode": args.descriptor_mode},
        "train_embedding": train_embedding,
        "train_bank": train_bank.half(), "train_class": train_class,
        "train_rows": torch.from_numpy(train.target.rows).long(),
        "selected_validation_rows": selected_rows,
        "selection": result["selection"],
        "validation": result["validation"],
        "source": {"checkpoint": report_path(args.checkpoint)},
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
