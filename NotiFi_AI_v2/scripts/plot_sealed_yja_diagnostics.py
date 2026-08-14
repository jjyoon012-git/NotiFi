"""Render normalized confusion matrices and one-vs-rest ROC curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notifi_pose.contract import ACTION_NAMES, RISK_NAMES  # noqa: E402


def confusion_percent(
    target: np.ndarray, predicted: np.ndarray, classes: int,
) -> np.ndarray:
    """Return a row-normalized confusion matrix in percent."""
    matrix = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(matrix, (target, predicted), 1.0)
    return 100.0 * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)


def save_confusion(
    matrix: np.ndarray,
    labels: tuple[str, ...],
    title: str,
    output: Path,
    size: tuple[float, float],
) -> None:
    """Save a percentage confusion matrix without displaying raw counts."""
    figure, axis = plt.subplots(figsize=size, constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=100.0)
    axis.set_title(title, fontsize=17, pad=14, fontweight="bold")
    axis.set_xlabel("Predicted class", fontsize=12)
    axis.set_ylabel("True class", fontsize=12)
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(labelsize=9 if len(labels) > 3 else 11)
    threshold = 52.0
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            if value < 0.05:
                continue
            axis.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=6.5 if len(labels) > 3 else 10,
                color="white" if value < threshold else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Share of each true class (%)", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def interpolate_macro_roc(
    target: np.ndarray, probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[int, dict | None]]:
    """Compute per-class, macro, and micro one-vs-rest ROC statistics."""
    classes = probability.shape[1]
    one_hot = np.eye(classes, dtype=np.int64)[target]
    per_class: dict[int, dict | None] = {}
    grid = np.linspace(0.0, 1.0, 1001)
    interpolated = []
    for class_id in range(classes):
        positives = int(one_hot[:, class_id].sum())
        if positives == 0 or positives == len(one_hot):
            per_class[class_id] = None
            continue
        false_positive, true_positive, _ = roc_curve(
            one_hot[:, class_id], probability[:, class_id]
        )
        class_auc = auc(false_positive, true_positive)
        per_class[class_id] = {
            "false_positive": false_positive,
            "true_positive": true_positive,
            "auc": float(class_auc),
        }
        interpolated.append(np.interp(grid, false_positive, true_positive))
    macro_true_positive = np.mean(interpolated, axis=0)
    macro_true_positive[0] = 0.0
    macro_true_positive[-1] = 1.0
    macro_auc = float(auc(grid, macro_true_positive))
    micro_false_positive, micro_true_positive, _ = roc_curve(
        one_hot.ravel(), probability.ravel()
    )
    micro_auc = float(auc(micro_false_positive, micro_true_positive))
    return (
        grid,
        macro_true_positive,
        macro_auc,
        micro_auc,
        per_class,
    )


def save_roc(
    target: np.ndarray,
    probability: np.ndarray,
    labels: tuple[str, ...],
    title: str,
    output: Path,
) -> dict:
    """Save one-vs-rest ROC curves and return serializable AUC values."""
    grid, macro_tpr, macro_auc, micro_auc, per_class = interpolate_macro_roc(
        target, probability
    )
    one_hot = np.eye(len(labels), dtype=np.int64)[target]
    micro_fpr, micro_tpr, _ = roc_curve(one_hot.ravel(), probability.ravel())
    figure, axis = plt.subplots(figsize=(11.5, 8.0), constrained_layout=True)
    palette = plt.get_cmap("tab20")
    for class_id, label in enumerate(labels):
        values = per_class[class_id]
        if values is None:
            continue
        axis.plot(
            values["false_positive"],
            values["true_positive"],
            linewidth=1.2,
            alpha=0.70,
            color=palette(class_id % 20),
            label=f"{label} ({values['auc']:.3f})",
        )
    axis.plot(
        grid,
        macro_tpr,
        color="#111827",
        linewidth=3.0,
        label=f"macro average ({macro_auc:.3f})",
    )
    axis.plot(
        micro_fpr,
        micro_tpr,
        color="#dc2626",
        linestyle="--",
        linewidth=2.5,
        label=f"micro average ({micro_auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1.5)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("False positive rate", fontsize=12)
    axis.set_ylabel("True positive rate", fontsize=12)
    axis.set_title(title, fontsize=17, pad=14, fontweight="bold")
    axis.grid(alpha=0.18)
    axis.legend(
        loc="lower right",
        fontsize=8 if len(labels) > 3 else 10,
        ncol=2 if len(labels) > 3 else 1,
        frameon=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    return {
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "class_auc": {
            labels[class_id]: (values["auc"] if values is not None else None)
            for class_id, values in per_class.items()
        },
    }


def save_pose_error_cdf(
    errors: np.ndarray, risks: np.ndarray, output: Path,
) -> None:
    """Save risk-stratified empirical pose-error CDFs using percentages."""
    figure, axis = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    colors = ("#2563eb", "#d97706", "#dc2626")
    for risk_id, (label, color) in enumerate(zip(RISK_NAMES, colors)):
        values = np.sort(errors[risks == risk_id])
        cumulative = 100.0 * np.arange(1, len(values) + 1) / len(values)
        axis.step(
            values,
            cumulative,
            where="post",
            linewidth=2.4,
            color=color,
            label=f"{label} (median {np.median(values):.1f} cm)",
        )
    axis.set_xlabel("Per-trial pose MPJPE (cm)", fontsize=12)
    axis.set_ylabel("Trials at or below error (%)", fontsize=12)
    axis.set_title(
        "Sealed yja/E02: pose error distribution by risk",
        fontsize=17,
        pad=14,
        fontweight="bold",
    )
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.20)
    axis.legend(loc="lower right", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    """Create all sealed yja/E02 diagnostic figures and a JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    options = parser.parse_args()

    payload = np.load(options.input, allow_pickle=False)
    true_action = payload["true_action"]
    true_risk = payload["true_risk"]
    predicted_action = payload["predicted_action"]
    predicted_risk = payload["predicted_risk"]
    action_confusion = confusion_percent(
        true_action, predicted_action, len(ACTION_NAMES)
    )
    risk_confusion = confusion_percent(true_risk, predicted_risk, len(RISK_NAMES))

    action_display_labels = tuple(
        "absence (support only)" if label == "absence" else label
        for label in ACTION_NAMES
    )
    save_confusion(
        action_confusion,
        action_display_labels,
        "Sealed yja/E02: 17-action confusion matrix (row-normalized)",
        options.output_dir / "action_confusion_matrix_percent.png",
        (18.0, 15.5),
    )
    save_confusion(
        risk_confusion,
        RISK_NAMES,
        "Sealed yja/E02: 3-risk confusion matrix (row-normalized)",
        options.output_dir / "risk_confusion_matrix_percent.png",
        (8.5, 7.2),
    )
    action_roc = save_roc(
        true_action,
        payload["action_probability"],
        ACTION_NAMES,
        "Sealed yja/E02: action one-vs-rest ROC (16 query classes)",
        options.output_dir / "action_roc_auc.png",
    )
    risk_roc = save_roc(
        true_risk,
        payload["risk_probability"],
        RISK_NAMES,
        "Sealed yja/E02: 3-risk one-vs-rest ROC",
        options.output_dir / "risk_roc_auc.png",
    )
    save_pose_error_cdf(
        payload["pose_error_cm"],
        true_risk,
        options.output_dir / "pose_error_cdf_percent.png",
    )
    summary = {
        "protocol": "sealed yja/E02; 239 query trials; support excluded",
        "confusion_normalization": "each true-class row sums to 100 percent",
        "artifact_sha256": str(payload["artifact_sha256"]),
        "support_seed": int(payload["support_seed"]),
        "query_trials": int(len(true_action)),
        "action_confusion_percent": action_confusion.tolist(),
        "risk_confusion_percent": risk_confusion.tolist(),
        "action_roc_auc": action_roc,
        "risk_roc_auc": risk_roc,
        "pose_error_cm": {
            "mean": float(payload["pose_error_cm"].mean()),
            "median": float(np.median(payload["pose_error_cm"])),
            "p90": float(np.percentile(payload["pose_error_cm"], 90)),
        },
    }
    options.summary_output.parent.mkdir(parents=True, exist_ok=True)
    options.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
