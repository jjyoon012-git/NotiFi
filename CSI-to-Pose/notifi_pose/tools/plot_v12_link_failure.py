"""Plot V12 versus V12G on deterministic drop-one-link validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("docs/results/v12_input_robustness.json"),
    )
    parser.add_argument(
        "--guard", type=Path,
        default=Path("docs/results/v12_link_failure_guard_robustness.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/v12_link_failure_comparison.png"),
    )
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))[
        "results"
    ]["drop_one_link"]
    guard = json.loads(args.guard.read_text(encoding="utf-8"))[
        "results"
    ]["drop_one_link"]
    error_keys = (
        ("MPJPE", "mpjpe_m"),
        ("Root", "root_error_m"),
        ("Danger", "danger_mpjpe_m"),
        ("Endpoint", "danger_endpoint_mpjpe_m"),
    )
    class_keys = (
        ("17-class acc.", "class_accuracy"),
        ("Risk acc.", "risk_accuracy"),
        ("Danger recall", "danger_recall"),
    )

    plt.rcParams.update({"font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = ("#697386", "#16856b")
    width = 0.36

    labels = [label for label, _ in error_keys]
    x = np.arange(len(labels))
    base_values = [100 * baseline["trajectory"][key] for _, key in error_keys]
    guard_values = [100 * guard["trajectory"][key] for _, key in error_keys]
    axes[0].bar(x - width / 2, base_values, width, label="V12", color=colors[0])
    axes[0].bar(x + width / 2, guard_values, width, label="V12RG", color=colors[1])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Error (cm), lower is better")
    axes[0].set_title("Pose and trajectory under one-link loss")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    labels = [label for label, _ in class_keys]
    x = np.arange(len(labels))
    base_values = [100 * baseline[key] for _, key in class_keys]
    guard_values = [100 * guard[key] for _, key in class_keys]
    axes[1].bar(x - width / 2, base_values, width, label="V12", color=colors[0])
    axes[1].bar(x + width / 2, guard_values, width, label="V12RG", color=colors[1])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(55, 90)
    axes[1].set_ylabel("Score (%), higher is better")
    axes[1].set_title(
        "Classification (safe-to-danger: "
        f"{baseline['safe_to_danger']} to {guard['safe_to_danger']})"
    )
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("Deterministic validation stress test; test split unopened")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
