"""Train KP2-A: Doppler CSI encoding plus trial-level motion correspondence."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .. import contract as C
from ..doppler_pose import DopplerPoseResidual, correspondence_loss
from ..quality import quality_summary
from ..trainer import set_seed
from .audit_kinetic_pose import SignalCounterfactualDataset, delta
from .diagnose_observability import report_path
from .train_kinetic_pose import (
    build_components,
    evaluate_strengths,
    kinetic_pose_loss,
    load_or_create_coarse_store,
    make_loaders,
    pose_selection_score,
)


class CorrespondenceBatchSampler(Sampler[list[int]]):
    """Build weighted batches containing same-class/site trial negatives."""

    def __init__(self, index, weights: torch.Tensor, batch_size: int, seed: int):
        if batch_size < 2:
            raise ValueError("correspondence batches need at least two samples")
        self.index = index.reset_index(drop=True)
        self.weights = weights.double().cpu()
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        columns = ["subject", "environment", "class_id"]
        self.groups = {
            int(position): np.asarray(positions, dtype=np.int64)
            for positions in self.index.groupby(columns, sort=False).indices.values()
            for position in positions
        }

    def __len__(self) -> int:
        return math.ceil(len(self.index) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 10_007)
        rng = np.random.default_rng(self.seed + self.epoch * 10_007)
        self.epoch += 1
        pairs_per_batch = self.batch_size // 2
        for _ in range(len(self)):
            anchors = torch.multinomial(
                self.weights, pairs_per_batch, replacement=True,
                generator=generator,
            ).tolist()
            batch = []
            for anchor in anchors:
                candidates = self.groups[int(anchor)]
                alternatives = candidates[candidates != anchor]
                partner = (
                    int(rng.choice(alternatives)) if len(alternatives)
                    else int(anchor)
                )
                batch.extend((int(anchor), partner))
            if self.batch_size % 2:
                extra = int(torch.multinomial(
                    self.weights, 1, replacement=True, generator=generator
                ))
                batch.append(extra)
            yield batch


def use_correspondence_batches(loaders: dict, train, args, device: str) -> None:
    weights = train.sampler_weights()
    danger = torch.tensor(
        train.index.risk_id.to_numpy(dtype=np.int64) == 2, dtype=torch.bool
    )
    weights = weights * torch.where(
        danger, torch.tensor(args.danger_weight, dtype=weights.dtype),
        torch.tensor(1.0, dtype=weights.dtype),
    )
    sampler = CorrespondenceBatchSampler(
        train.index, weights, args.batch_size, args.seed
    )
    loaders["train"] = DataLoader(
        train, batch_sampler=sampler, num_workers=0,
        pin_memory=device == "cuda",
    )


def train_epoch(model: DopplerPoseResidual, loader: DataLoader,
                optimizer: torch.optim.Optimizer, scaler, device: str,
                args, coarse_store) -> dict:
    model.train()
    model.set_residual_strength(1.0)
    totals: dict[str, list[float]] = {}
    for step, batch in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            output = model(
                batch["csi"], batch["link_mask"],
                coarse_pose=coarse_store.lookup(batch["row"].cpu(), device),
            )
            pose_loss, parts = kinetic_pose_loss(output, batch, args)
            target_embedding = model.encode_target_motion(
                batch["pose_rel"], batch["valid"].bool()
            )
            cross_modal, cross_parts = correspondence_loss(
                output["csi_motion_embedding"], target_embedding,
                batch["row"], batch["class_id"], batch["domain_id"],
                temperature=args.contrastive_temperature,
                same_class_bias=args.same_class_negative_bias,
                same_domain_bias=args.same_domain_negative_bias,
            )
            loss = pose_loss + args.lambda_correspondence * cross_modal
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        parts["total_with_correspondence"] = float(loss.detach())
        parts.update(cross_parts)
        for key, value in parts.items():
            if math.isfinite(value):
                totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def counterfactual_audit(model: DopplerPoseResidual, test,
                         coarse_store, strength: float,
                         batch_size: int, seed: int, device: str) -> dict:
    metrics = {}
    for mode in ("clean", "matched_shuffle", "temporal_reverse", "temporal_mean"):
        loader = DataLoader(
            SignalCounterfactualDataset(test, mode, seed),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        metrics[mode] = evaluate_strengths(
            model, loader, [strength], device, coarse_store
        )[strength]
    return {
        "clean": metrics["clean"],
        "counterfactuals": {
            mode: {
                "metrics": metrics[mode],
                "delta_from_clean": delta(metrics["clean"], metrics[mode]),
            }
            for mode in ("matched_shuffle", "temporal_reverse", "temporal_mean")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_sub_single_clean_finetune" / "best_model.pt",
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=C.PROJECT_ROOT / "docs" / "results" / "v13s_pruned_pose_root_ensemble.json",
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=C.WORK_ROOT / "runs" / "p2_v12w_robust_classification_ensemble" / "validation.json",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--danger-weight", type=float, default=3.0)
    parser.add_argument("--danger-frame-boost", type=float, default=0.50)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--distal-joint-weight", type=float, default=2.5)
    parser.add_argument("--lambda-distal", type=float, default=0.40)
    parser.add_argument("--lambda-velocity", type=float, default=0.25)
    parser.add_argument("--lambda-aux-velocity", type=float, default=0.30)
    parser.add_argument("--lambda-acceleration", type=float, default=0.01)
    parser.add_argument("--lambda-bone-length", type=float, default=0.10)
    parser.add_argument("--lambda-bone-direction", type=float, default=0.04)
    parser.add_argument("--lambda-static", type=float, default=0.08)
    parser.add_argument("--lambda-endpoint", type=float, default=0.30)
    parser.add_argument("--lambda-correspondence", type=float, default=0.08)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--same-class-negative-bias", type=float, default=0.35)
    parser.add_argument("--same-domain-negative-bias", type=float, default=0.15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument(
        "--condition-on-coarse", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--activity-floor", type=float, default=0.0)
    parser.add_argument("--experiment-name", default="KP2-A-EXP01")
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument(
        "--coarse-cache", type=Path,
        default=C.WORK_ROOT / "runs" / "kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=C.WORK_ROOT / "runs" / "kp2a_exp01_doppler_correspondence",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, loaders = make_loaders(args, device)
    train, validation, test = datasets
    use_correspondence_batches(loaders, train, args, device)
    baseline, normalizer, baseline_config = build_components(args, device)
    coarse_store = load_or_create_coarse_store(
        baseline, datasets, args.coarse_cache, device, args.batch_size, args.exp
    )
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    model = DopplerPoseResidual(
        None, normalizer, hidden=args.hidden,
        temporal_layers=args.temporal_layers, heads=args.heads,
        dropout=args.dropout, max_delta=args.max_delta,
        condition_on_coarse=args.condition_on_coarse,
        activity_floor=args.activity_floor,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    best = {"score": math.inf, "epoch": 0, "strength": 0.0, "state": None}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, scaler, device, args, coarse_store
        )
        candidates = evaluate_strengths(
            model, loaders["val"], list(args.strengths), device, coarse_store
        )
        scores = {
            strength: pose_selection_score(metrics)
            for strength, metrics in candidates.items()
        }
        selected_strength = min(scores, key=scores.get)
        selected_score = scores[selected_strength]
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation_score": selected_score,
            "validation_strength": selected_strength,
            "validation": candidates[selected_strength],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if selected_score < best["score"] - 1e-5:
            best = {
                "score": selected_score,
                "epoch": epoch,
                "strength": selected_strength,
                "state": copy.deepcopy(model.trainable_state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best["state"] is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_trainable_state_dict(best["state"])
    strength = float(best["strength"])
    validation_metrics = evaluate_strengths(
        model, loaders["val"], [0.0, strength], device, coarse_store
    )
    test_metrics = evaluate_strengths(
        model, loaders["test"], [0.0, strength], device, coarse_store
    )
    counterfactual = counterfactual_audit(
        model, test, coarse_store, strength,
        args.batch_size * 2, args.seed + 54, device,
    )
    result = {
        "run": args.experiment_name,
        "model_family": "NotiFi-KP2",
        "candidate_version": "KP2-A",
        "promotion_status": "experimental",
        "protocol": args.exp,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "objective": "pelvis_relative_pose_and_trial_correspondence",
        "device": device,
        "seed": args.seed,
        "dataset": {
            "train": train.target.describe(),
            "validation": validation.target.describe(),
            "test": test.target.describe(),
            "quality": quality_summary(train),
        },
        "architecture": {
            "dynamic_inputs": ["delta_1", "delta_3", "delta_7", "high_pass_15"],
            "doppler_windows_frames": [17, 33, 65],
            "doppler_cycles_per_window": [1.0, 2.0],
            "doppler_inputs": ["amplitude_delta_1", "phase_delta_1",
                                "amplitude_delta_3", "phase_delta_3"],
            "ordered_link_pairs": [[0, 1], [0, 2], [1, 2]],
            "trial_correspondence": "multi_positive_hard_negative_infonce",
            "static_csi_visible_to_residual": False,
            "baseline_frozen": True,
            "root_trained": False,
            "hidden": args.hidden,
            "embedding_dim": args.embedding_dim,
            "temporal_layers": args.temporal_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "max_delta_m": args.max_delta,
            "condition_on_coarse": args.condition_on_coarse,
            "activity_floor": args.activity_floor,
        },
        "loss": {
            "lambda_correspondence": args.lambda_correspondence,
            "contrastive_temperature": args.contrastive_temperature,
            "same_class_negative_bias": args.same_class_negative_bias,
            "same_domain_negative_bias": args.same_domain_negative_bias,
        },
        "baseline_configuration": baseline_config,
        "coarse_pose_cache": report_path(args.coarse_cache),
        "selection": {
            "epoch": int(best["epoch"]),
            "score": float(best["score"]),
            "residual_strength": strength,
        },
        "validation": {
            "v13s_strength_0": validation_metrics[0.0],
            "kp2a_selected": validation_metrics[strength],
        },
        "test": {
            "v13s_strength_0": test_metrics[0.0],
            "kp2a_selected": test_metrics[strength],
        },
        "counterfactual": counterfactual,
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "run": args.experiment_name,
        "model_family": result["model_family"],
        "protocol": args.exp,
        "trainable_model": best["state"],
        "residual_strength": strength,
        "architecture": result["architecture"],
        "source": {
            "p2_checkpoint": report_path(args.p2_checkpoint),
            "root_calibration": report_path(args.root_calibration),
            "classification_calibration": report_path(args.classification_calibration),
        },
        "selection": result["selection"],
        "validation": result["validation"],
        "test": result["test"],
    }, args.run_dir / "best_model.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
