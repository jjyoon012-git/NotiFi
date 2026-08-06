"""Train and audit CAL23 dynamic-only calibration without target-query leakage."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .. import contract as C
from ..cal16_kp10 import TARGET_CALIBRATION_SPLIT_SEED
from ..cal23_kp10 import DynamicMotionClassifier
from ..dataio.dataset import DropoutConfig, build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_cal16_identity_spectrum_kp10 import _cache_for_model, _evaluate_pair
from .evaluate_cal4_linkmap_kp10 import load_coarse
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import add_paths, configure_work_root, slice_cache, split_support_query
from .train_cal2_kp10 import SiteDiscriminator
from .train_cal3_kp10 import load_heads
from .train_calibration_aware_v14 import subset_dataset
from .train_dynamic_motion import classification_metrics
from .train_motion_retrieval_selector import predict_selector


class PairedViews(Dataset):
    def __init__(self, clean, augmented):
        if not np.array_equal(clean.rows, augmented.rows):
            raise RuntimeError("clean and augmented dataset rows differ")
        self.clean = clean
        self.augmented = augmented

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        return {"clean": self.clean[index], "augmented": self.augmented[index]}

    def set_epoch(self, epoch: int) -> None:
        self.clean.target.set_epoch(epoch)
        self.augmented.target.set_epoch(epoch)


def cross_site_contrastive(features: torch.Tensor, labels: torch.Tensor,
                           domains: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features, dim=-1)
    similarity = normalized @ normalized.T
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = (
        (labels[:, None] == labels[None])
        & (domains[:, None] != domains[None]) & ~eye
    )
    negative = (labels[:, None] != labels[None]) & ~eye
    zero = similarity.sum() * 0.0
    pull = (1.0 - similarity[positive]).mean() if positive.any() else zero
    push = F.relu(similarity[negative] - 0.20).mean() if negative.any() else zero
    return pull + 0.20 * push


def _move(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def danger_score(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 2] - torch.logsumexp(logits[:, :2], dim=-1)


def safe_location_scale(scores: torch.Tensor, risk: torch.Tensor) -> tuple[float, float]:
    values = scores[risk == 0]
    median = float(values.median())
    mad = float((values - median).abs().median().clamp_min(0.05))
    return median, mad


def standardize_score(scores: torch.Tensor,
                      statistics: tuple[float, float]) -> torch.Tensor:
    return (scores - statistics[0]) / statistics[1]


def choose_safe_threshold(scores: torch.Tensor, risk: torch.Tensor,
                          maximum_safe_fpr: float = 0.10) -> dict:
    safe = scores[risk == 0]
    danger = scores[risk == 2]
    candidates = torch.unique(scores).sort().values
    candidates = torch.cat((candidates, candidates[-1:] + 1e-4))
    rows = []
    for threshold in candidates:
        false_positive = float((safe >= threshold).float().mean())
        recall = float((danger >= threshold).float().mean())
        if false_positive <= maximum_safe_fpr:
            rows.append((recall, -false_positive, float(threshold)))
    recall, negative_fpr, threshold = max(rows) if rows else (0.0, -1.0, math.inf)
    return {
        "threshold": threshold, "danger_recall": recall,
        "safe_fpr": -negative_fpr,
    }


def threshold_risk(logits: torch.Tensor, standardized_score: torch.Tensor,
                   threshold: float) -> torch.Tensor:
    prediction = logits[:, :2].argmax(-1)
    return torch.where(
        standardized_score >= threshold,
        torch.full_like(prediction, 2), prediction,
    )


def metrics_with_threshold(action_logits: torch.Tensor, risk_logits: torch.Tensor,
                           standardized_score: torch.Tensor, threshold: float,
                           action: torch.Tensor, risk: torch.Tensor) -> dict:
    adjusted = risk_logits.new_full(risk_logits.shape, -20.0)
    prediction = threshold_risk(risk_logits, standardized_score, threshold)
    adjusted.scatter_(1, prediction[:, None], 20.0)
    return classification_metrics(action_logits, adjusted, action, risk)


def action_risk_consistency(action_logits: torch.Tensor,
                            risk_logits: torch.Tensor,
                            weight: float) -> torch.Tensor:
    class_risk = action_logits.new_tensor(
        [0] * 9 + [1] * 3 + [2] * 5, dtype=torch.long
    )
    evidence = F.log_softmax(risk_logits, dim=-1).index_select(1, class_risk)
    return action_logits + float(weight) * evidence


def select_hierarchy(action_logits: torch.Tensor, risk_logits: torch.Tensor,
                     action: torch.Tensor, risk: torch.Tensor,
                     weights: tuple[float, ...]) -> dict:
    candidates = []
    for weight in weights:
        metrics = classification_metrics(
            action_risk_consistency(action_logits, risk_logits, weight),
            risk_logits, action, risk,
        )
        candidates.append({
            "weight": float(weight),
            "action_accuracy": metrics["action_accuracy"],
            "action_macro_f1": metrics["action_macro_f1"],
        })
    selected = max(candidates, key=lambda value: (
        value["action_accuracy"], value["action_macro_f1"], -value["weight"]
    ))
    return {"selected": selected, "candidates": candidates}


def conformal_safe_threshold(scores: torch.Tensor,
                             maximum_fpr: float = 0.10) -> float:
    """Return an empirical threshold allowing at most floor(alpha*n) alarms."""
    allowed = int(math.floor(maximum_fpr * len(scores)))
    descending = scores.sort(descending=True).values
    if allowed >= len(descending):
        return -math.inf
    return float(descending[allowed] + 1e-5)


@torch.no_grad()
def predict(model: DynamicMotionClassifier, dataset, batch_size: int,
            device: str) -> dict:
    model.eval()
    output = {key: [] for key in (
        "action_logits", "risk_logits", "embedding", "class_id", "risk_id", "row"
    )}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        result = model(batch["csi"].to(device), batch["link_mask"].to(device))
        output["action_logits"].append(result["action_logits"].float().cpu())
        output["risk_logits"].append(result["risk_logits"].float().cpu())
        output["embedding"].append(result["embedding"].float().cpu())
        for key in ("class_id", "risk_id", "row"):
            output[key].append(batch[key].long().cpu())
    return {key: torch.cat(value) for key, value in output.items()}


def train_epoch(args, model, discriminator, loader, optimizer, scaler,
                domain_lookup: dict[int, int], epoch: int, device: str) -> dict:
    model.train()
    discriminator.train()
    loader.dataset.set_epoch(epoch)
    totals: dict[str, list[float]] = {}
    reversal = min(1.0, epoch / max(args.domain_warmup_epochs, 1))
    for paired in loader:
        clean = _move(paired["clean"], device)
        augmented = _move(paired["augmented"], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            first = model(clean["csi"], clean["link_mask"])
            second = model(augmented["csi"], augmented["link_mask"])
            labels = clean["class_id"].long()
            risk = clean["risk_id"].long()
            action = 0.5 * (
                F.cross_entropy(first["action_logits"], labels, label_smoothing=0.04)
                + F.cross_entropy(second["action_logits"], labels, label_smoothing=0.04)
            )
            risk_each = 0.5 * (
                F.cross_entropy(first["risk_logits"], risk, reduction="none")
                + F.cross_entropy(second["risk_logits"], risk, reduction="none")
            )
            risk_loss = (risk_each * torch.where(risk == 2, 2.5, 1.0)).mean()
            consistency = F.kl_div(
                F.log_softmax(second["action_logits"], -1),
                F.softmax(first["action_logits"].detach(), -1), reduction="batchmean",
            )
            domains = torch.tensor([
                domain_lookup[int(value)] for value in clean["domain_id"].tolist()
            ], dtype=torch.long, device=device)
            domain = F.cross_entropy(discriminator(first["embedding"], reversal), domains)
            invariant = cross_site_contrastive(
                first["embedding"], labels, domains
            )
            loss = (
                action + args.risk_weight * risk_loss
                + args.consistency_weight * consistency
                + args.domain_weight * domain
                + args.invariant_weight * invariant
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(discriminator.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        for key, value in {
            "loss": loss, "action": action, "risk": risk_loss,
            "consistency": consistency, "domain": domain,
            "invariant": invariant,
        }.items():
            totals.setdefault(key, []).append(float(value.detach()))
    return {key: float(np.mean(value)) for key, value in totals.items()}


def selection_score(metrics: dict) -> float:
    return (
        1.0 - metrics["action_accuracy"]
        + 0.45 * (1.0 - metrics["risk_macro_f1"])
        + 0.70 * (1.0 - metrics["danger_recall"])
    )


def calibrated_cache(identity: dict, motion_logits: torch.Tensor,
                     weight: float, temperature: float) -> dict:
    output = dict(identity)
    output["base_action_logits"] = (
        identity["base_action_logits"].float()
        + float(weight) * motion_logits.float() / float(temperature)
    ).half()
    output["source"] = "fixed-geometry KP10 + CAL23 dynamic action evidence"
    return output


def select_fusion(args, validation_cache: dict, motion: dict,
                  classifier, selector, device: str) -> dict:
    action = motion["class_id"]
    risk = motion["risk_id"]
    candidates = []
    for temperature in args.temperatures:
        for weight in args.fusion_weights:
            current = calibrated_cache(
                validation_cache, motion["action_logits"], weight, temperature
            )
            metrics = classification_with_preserved_risk(
                args, current, validation_cache, action, risk,
                classifier, selector, device,
            )
            candidates.append({
                "temperature": float(temperature), "weight": float(weight),
                "action_accuracy": metrics["action_accuracy"],
                "action_macro_f1": metrics["action_macro_f1"],
            })
    selected = max(candidates, key=lambda value: (
        value["action_accuracy"], value["action_macro_f1"], -value["weight"]
    ))
    return {"selected": selected, "candidates": candidates}


def classification_with_preserved_risk(args, action_cache: dict, risk_cache: dict,
                                       action: torch.Tensor, risk: torch.Tensor,
                                       classifier, selector, device: str) -> dict:
    classified = predict_selector(classifier, action_cache, 64, device)
    selected = predict_selector(selector, risk_cache, 64, device)
    action_logits = (
        1.50 * action_cache["base_action_logits"].float()
        + 0.75 * classified["action_logits"]
    )
    risk_logits = risk_cache["base_risk_logits"].float() + selected["risk_logits"]
    return classification_metrics(action_logits, risk_logits, action, risk)


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--fold", default=None)
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--risk-weight", type=float, default=0.65)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--domain-weight", type=float, default=0.04)
    parser.add_argument("--invariant-weight", type=float, default=0.0)
    parser.add_argument("--domain-warmup-epochs", type=int, default=5)
    parser.add_argument("--target-reserve-per-class", type=int, default=4)
    parser.add_argument("--fusion-weights", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--temperatures", type=float, nargs="+", default=(0.75, 1.0, 1.5, 2.0))
    parser.add_argument("--hierarchy-weights", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--shape-only", action="store_true")
    parser.add_argument(
        "--feature-mode", choices=("energy", "physical_phase"), default="energy"
    )
    parser.add_argument(
        "--kp4-checkpoint", type=Path,
        default=default_work / "runs/kp4_dcc_staged_seed17/deployment_model.pt",
    )
    parser.add_argument(
        "--source-coarse", type=Path,
        default=default_work / "runs/kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--yja-coarse", type=Path,
        default=default_work / "runs/cal2_kp10_seed223_danger_gate/yja_e02_v13s_coarse.pt",
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal23_dynamic_meta_kp10"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clean_sets = build_datasets(
        exp=args.exp, fold=args.fold, baseline=args.baseline, seed=args.seed,
        dropout=DropoutConfig(p=0.0, rf_augment=False),
    )
    augmented_sets = build_datasets(
        exp=args.exp, fold=args.fold, baseline=args.baseline, seed=args.seed,
        dropout=DropoutConfig(
            p=0.30, max_drop=1, rf_augment=True, gain_std=0.45,
            phase_std=0.65, phase_slope_std=0.20,
            subcarrier_mask_p=0.35, temporal_jitter=2,
        ),
    )
    audit = protocol_audit_path(args.exp)
    clean_train = QualityWeightedDataset(clean_sets["train"], audit)
    augmented_train = QualityWeightedDataset(augmented_sets["train"], audit)
    validation = QualityWeightedDataset(clean_sets["val"], audit)
    paired = PairedViews(clean_train, augmented_train)
    labels = torch.tensor(clean_train.index.class_id.to_numpy(dtype=np.int64))
    counts = torch.bincount(labels, minlength=C.N_CLASSES).float()
    weights = 1.0 / torch.sqrt(counts[labels].clamp_min(1.0))
    loader = DataLoader(
        paired, batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights.double(), len(weights), replacement=True),
        num_workers=0, pin_memory=True,
    )
    model_config = {
        "shape_only": bool(args.shape_only),
        "feature_mode": args.feature_mode,
    }
    if args.resume_checkpoint is not None:
        resume_header = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        model_config = resume_header.get("model_config", model_config)
    model = DynamicMotionClassifier(**model_config).to(device)
    domain_ids = sorted({
        int(clean_train[index]["domain_id"])
        for index in range(len(clean_train))
    })
    domain_lookup = {value: index for index, value in enumerate(domain_ids)}
    discriminator = SiteDiscriminator(128, len(domain_ids)).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    if args.resume_checkpoint is not None:
        resumed = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        best_state = resumed["model_state_dict"]
        model.load_state_dict(best_state)
        best_epoch = int(resumed.get("best_epoch", -1))
        history = []
    else:
        best_state = copy.deepcopy(model.state_dict())
        best_score = math.inf
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            training = train_epoch(
                args, model, discriminator, loader, optimizer, scaler,
                domain_lookup, epoch, device
            )
            validation_prediction = predict(model, validation, args.batch_size, device)
            validation_metrics = classification_metrics(
                validation_prediction["action_logits"], validation_prediction["risk_logits"],
                validation_prediction["class_id"], validation_prediction["risk_id"],
            )
            score = selection_score(validation_metrics)
            history.append({
                "epoch": epoch, "train": training, "validation": validation_metrics,
                "selection_score": score,
            })
            print(json.dumps({
                "epoch": epoch, "loss": training["loss"], "score": score,
                "action": validation_metrics["action_accuracy"],
                "risk": validation_metrics["risk_accuracy"],
                "danger_recall": validation_metrics["danger_recall"],
            }), flush=True)
            if score < best_score - 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)

    train_prediction = predict(model, clean_train, args.batch_size, device)
    validation_prediction = predict(model, validation, args.batch_size, device)
    validation_metrics = classification_metrics(
        validation_prediction["action_logits"], validation_prediction["risk_logits"],
        validation_prediction["class_id"], validation_prediction["risk_id"],
    )
    hierarchy = select_hierarchy(
        validation_prediction["action_logits"], validation_prediction["risk_logits"],
        validation_prediction["class_id"], validation_prediction["risk_id"],
        tuple(args.hierarchy_weights),
    )
    source_stats = safe_location_scale(
        danger_score(train_prediction["risk_logits"]), train_prediction["risk_id"]
    )
    source_threshold = choose_safe_threshold(
        standardize_score(danger_score(validation_prediction["risk_logits"]), source_stats),
        validation_prediction["risk_id"],
    )

    if args.classification_only:
        checkpoint = {
            "run": "CAL23-DYNAMIC-META-CAL-KP10",
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "source_safe_statistics": source_stats,
            "source_danger_threshold": source_threshold,
            "hierarchy": hierarchy,
            "training_protocol": {"exp": args.exp, "fold": args.fold},
            "model_config": model_config,
        }
        result = {
            "run": "CAL23-DYNAMIC-META-CAL-KP10-LOSO",
            "status": "source_only_audit_checkpoint",
            "protocol": {"exp": args.exp, "fold": args.fold},
            "best_epoch": best_epoch,
            "history": history,
            "source_validation": validation_metrics,
            "source_danger_threshold": source_threshold,
            "source_action_risk_hierarchy": hierarchy,
            "target_data_used_for_training_or_selection": False,
        }
        torch.save(checkpoint, args.run_dir / "calibration_candidate.pt")
        (args.run_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    validation_pose = QualityWeightedDataset(pose_only(clean_sets["val"]), audit)
    validation_pose_prediction = predict(
        model, validation_pose, args.batch_size, device
    )
    validation_cache = torch.load(
        args.selector_checkpoint.parent / "val_features.pt",
        map_location="cpu", weights_only=False,
    )
    if len(validation_pose_prediction["action_logits"]) != len(
        validation_cache["base_action_logits"]
    ):
        raise RuntimeError("CAL23 source pose-validation cache order changed")
    classifier, selector = load_heads(args, device)
    fusion = select_fusion(
        args, validation_cache, validation_pose_prediction,
        classifier, selector, device
    )

    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline=args.baseline, seed=args.seed
    )["test"]
    pool, query = split_support_query(
        sealed.index, ("yja_E02",), args.target_reserve_per_class,
        TARGET_CALIBRATION_SPLIT_SEED,
    )
    support_positions = pool["yja_E02"]
    support = QualityWeightedDataset(subset_dataset(sealed, support_positions), None)
    target = QualityWeightedDataset(subset_dataset(sealed, query), None)
    support_prediction = predict(model, support, args.batch_size, device)
    query_prediction = predict(model, target, args.batch_size, device)
    hierarchy_weight = hierarchy["selected"]["weight"]
    support_action_logits = action_risk_consistency(
        support_prediction["action_logits"], support_prediction["risk_logits"],
        hierarchy_weight,
    )
    query_action_logits = action_risk_consistency(
        query_prediction["action_logits"], query_prediction["risk_logits"],
        hierarchy_weight,
    )
    support_metrics = classification_metrics(
        support_action_logits, support_prediction["risk_logits"],
        support_prediction["class_id"], support_prediction["risk_id"],
    )
    target_stats = safe_location_scale(
        danger_score(support_prediction["risk_logits"]), support_prediction["risk_id"]
    )
    support_score = standardize_score(
        danger_score(support_prediction["risk_logits"]), target_stats
    )
    query_score = standardize_score(
        danger_score(query_prediction["risk_logits"]), target_stats
    )
    target_conformal_threshold = conformal_safe_threshold(support_score, 0.10)
    deployed_threshold = max(
        source_threshold["threshold"], target_conformal_threshold
    )
    support_false_danger = int((support_score >= deployed_threshold).sum())
    direct_query_metrics = classification_metrics(
        query_action_logits, query_prediction["risk_logits"],
        query_prediction["class_id"], query_prediction["risk_id"],
    )
    calibrated_query_metrics = metrics_with_threshold(
        query_action_logits, query_prediction["risk_logits"],
        query_score, deployed_threshold,
        query_prediction["class_id"], query_prediction["risk_id"],
    )
    action_ready = bool(
        support_metrics["action_accuracy"] >= 0.25
        and validation_metrics["action_accuracy"] >= 0.60
    )
    degraded = bool(
        not action_ready
        and support_metrics["action_accuracy"] >= 0.125
        and validation_metrics["action_accuracy"] >= 0.40
    )

    base_model, _ = _load_model(args.kp4_checkpoint, device)
    coarse = load_coarse(args.yja_coarse)
    identity = _cache_for_model(
        base_model, target, coarse, "yja_E02", device, "CAL23 fixed yja query"
    )
    selected = fusion["selected"]
    fused = calibrated_cache(
        identity, query_action_logits,
        selected["weight"], selected["temperature"],
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[query].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, query[pose_local]), None
    )
    local = torch.from_numpy(pose_local).long()
    base_pose, candidate_pose = _evaluate_pair(
        args, pose_target, slice_cache(identity, local), slice_cache(fused, local),
        args.run_dir / "yja_fixed_query", device,
    )

    result = {
        "run": "CAL23-DYNAMIC-META-CAL-KP10",
        "status": "PARTIAL" if action_ready else "DEGRADED" if degraded else "REJECT",
        "contract": {
            "physical_link_order": ["TX1_South", "TX2_West", "TX3_East"],
            "link_permutation": False,
            "static_level_discarded": True,
            "source_only_training_and_selection": True,
            "target_support_uses_known_safe_action_only": True,
            "target_query_used_for_training_or_selection": False,
            "fixed_target_calibration_split_seed": TARGET_CALIBRATION_SPLIT_SEED,
        },
        "best_epoch": best_epoch,
        "history": history,
        "source_validation": validation_metrics,
        "source_danger_threshold": source_threshold,
        "source_action_risk_hierarchy": hierarchy,
        "source_fusion_selection": fusion,
        "target_calibration": {
            "accepted_for_normal_inference": False,
            "accepted_for_action_pose_inference": action_ready,
            "risk_ready": False,
            "risk_reason": "safe_only_calibration_cannot_validate_danger_direction",
            "support_trials": len(support),
            "support_action_accuracy": support_metrics["action_accuracy"],
            "support_false_danger": support_false_danger,
            "source_locked_threshold": source_threshold["threshold"],
            "target_conformal_threshold": target_conformal_threshold,
            "deployed_threshold": deployed_threshold,
            "safe_score_statistics": {"median": target_stats[0], "mad": target_stats[1]},
        },
        "yja_e02": {
            "query_trials": len(query), "danger_trials": int((query_prediction["risk_id"] == 2).sum()),
            "dynamic_direct": direct_query_metrics,
            "dynamic_safe_calibrated_risk": calibrated_query_metrics,
            "kp10_base": base_pose,
            "kp10_plus_dynamic_action": candidate_pose,
        },
    }
    torch.save({
        "run": result["run"], "deployable": False,
        "action_pose_deployable": action_ready,
        "model_state_dict": best_state, "source_safe_statistics": source_stats,
        "source_danger_threshold": source_threshold, "fusion": selected,
        "best_epoch": best_epoch, "hierarchy": hierarchy,
        "model_config": model_config,
        "support_rows": sealed.rows[support_positions].tolist(),
    }, args.run_dir / "calibration_candidate.pt")
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "run": result["run"], "status": result["status"],
        "source_validation": validation_metrics,
        "target_calibration": result["target_calibration"],
        "yja_e02": result["yja_e02"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
