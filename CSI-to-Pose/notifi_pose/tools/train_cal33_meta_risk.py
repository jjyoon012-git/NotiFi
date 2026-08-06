"""Train CAL33 risk calibration through source-site unseen episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .. import contract as C
from ..cal23_kp10 import DynamicMotionClassifier
from ..cal33_kp10 import MetaRiskHead, build_safe_context, meta_risk_features
from ..calibration_quality import SAFE_CALIBRATION_CLASSES
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset
from ..trainer import set_seed
from .train_cal1_kp10 import add_paths, configure_work_root
from .train_cal23_dynamic_meta_kp10 import predict
from .train_dynamic_motion import classification_metrics


def _sites(index) -> np.ndarray:
    return (index.subject.astype(str) + "_" + index.environment.astype(str)).to_numpy()


def _episode_features(prediction: dict, index, support_per_class: int,
                      seed: int, support_classes,
                      allowed_subjects=None,
                      include_all_query: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    generator = np.random.default_rng(seed)
    site_values = _sites(index)
    features, targets = [], []
    for site in sorted(set(site_values.tolist())):
        site_positions = np.flatnonzero(site_values == site)
        subject = str(index.iloc[int(site_positions[0])].subject)
        if allowed_subjects and subject not in allowed_subjects:
            continue
        support = []
        complete = True
        for class_id in support_classes:
            choices = site_positions[
                prediction["class_id"][site_positions].numpy() == class_id
            ]
            if len(choices) < support_per_class:
                complete = False
                break
            support.extend(generator.choice(
                choices, support_per_class, replace=False
            ).tolist())
        if not complete:
            continue
        support_index = torch.as_tensor(support).long()
        context = build_safe_context(
            prediction["embedding"].index_select(0, support_index),
            prediction["risk_logits"].index_select(0, support_index),
            prediction["class_id"].index_select(0, support_index),
            support_classes,
        )
        query = np.setdiff1d(site_positions, np.asarray(support), assume_unique=False)
        if not include_all_query:
            query = site_positions
        query_index = torch.as_tensor(query).long()
        features.append(meta_risk_features(
            prediction["embedding"].index_select(0, query_index),
            prediction["risk_logits"].index_select(0, query_index),
            context,
        ))
        targets.append(prediction["risk_id"].index_select(0, query_index))
    if not features:
        raise RuntimeError("no source site has a complete CAL33 safe prompt")
    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def _evaluate(model: MetaRiskHead, features: torch.Tensor,
              target: torch.Tensor, device: str) -> dict:
    model.eval()
    logits = model(features.to(device)).cpu()
    dummy_action = torch.zeros(len(target), C.N_CLASSES)
    metrics = classification_metrics(
        dummy_action, logits, torch.zeros_like(target), target
    )
    safe = target == 0
    prediction = logits.argmax(-1)
    metrics["safe_fpr"] = float((prediction[safe] == 2).float().mean())
    return {
        key: value for key, value in metrics.items()
        if key.startswith("risk_") or key.startswith("danger_")
        or key in {"safe_to_danger", "safe_total", "safe_fpr", "trials"}
    }


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="loso")
    parser.add_argument("--fold", default="test_ajh")
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=433)
    parser.add_argument("--support-per-class", type=int, default=4)
    parser.add_argument(
        "--prompt-mode", choices=("safe", "safe_warning"), default="safe"
    )
    parser.add_argument("--train-subjects", nargs="+", default=None)
    parser.add_argument("--validation-subjects", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--cal23-checkpoint", type=Path, required=True)
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal33_meta_risk"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    support_classes = tuple(SAFE_CALIBRATION_CLASSES)
    if args.prompt_mode == "safe_warning":
        support_classes += (9, 10, 11)
    input_features = 2 * len(support_classes) + 6

    checkpoint = torch.load(args.cal23_checkpoint, map_location="cpu", weights_only=False)
    encoder = DynamicMotionClassifier(**checkpoint.get("model_config", {})).to(device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    datasets = build_datasets(
        exp=args.exp, fold=args.fold, baseline=args.baseline, seed=args.seed
    )
    train_prediction = predict(
        encoder, QualityWeightedDataset(datasets["train"], None),
        args.batch_size, device,
    )
    validation_prediction = predict(
        encoder, QualityWeightedDataset(datasets["val"], None),
        args.batch_size, device,
    )
    validation_features, validation_target = _episode_features(
        validation_prediction, datasets["val"].index,
        min(2, args.support_per_class), args.seed + 1000, support_classes,
        args.validation_subjects,
    )

    model = MetaRiskHead(input_features=input_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    class_weight = torch.tensor((1.0, 1.35, 2.5), device=device)
    history, best, best_state, stale = [], None, None, 0
    for epoch in range(1, args.epochs + 1):
        train_features, train_target = _episode_features(
            train_prediction, datasets["train"].index,
            args.support_per_class, args.seed + epoch, support_classes,
            args.train_subjects,
        )
        loader = DataLoader(
            TensorDataset(train_features, train_target),
            batch_size=args.batch_size, shuffle=True,
        )
        model.train()
        losses = []
        for feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature.to(device))
            loss = F.cross_entropy(logits, target.to(device), weight=class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = _evaluate(model, validation_features, validation_target, device)
        score = (
            metrics["risk_macro_f1"] + 0.8 * metrics["danger_recall"]
            - 0.4 * metrics["safe_fpr"]
        )
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "score": score, "validation": metrics}
        history.append(row)
        if best is None or score > best["score"]:
            best, best_state, stale = row, {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row))
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    result = {
        "run": (
            "CAL34-SAFE-WARNING-META-RISK-KP10"
            if args.prompt_mode == "safe_warning"
            else "CAL33-EPISODIC-META-RISK-KP10"
        ),
        "protocol": {"exp": args.exp, "fold": args.fold},
        "best_epoch": best["epoch"],
        "source_validation": _evaluate(
            model, validation_features, validation_target, device
        ),
        "target_data_used_for_training_or_selection": False,
        "support_per_safe_class_during_meta_training": args.support_per_class,
        "prompt_mode": args.prompt_mode,
        "support_classes": support_classes,
        "meta_train_subjects": args.train_subjects,
        "meta_validation_subjects": args.validation_subjects,
        "history": history,
    }
    torch.save({
        "run": result["run"],
        "model_state_dict": best_state,
        "model_config": {"input_features": input_features},
        "cal23_checkpoint": str(args.cal23_checkpoint),
        "result": result,
    }, args.run_dir / "meta_risk_candidate.pt")
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["source_validation"], indent=2))


if __name__ == "__main__":
    main()
