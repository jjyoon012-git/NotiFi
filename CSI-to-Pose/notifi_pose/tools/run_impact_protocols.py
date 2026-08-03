"""Fine-tune impact-aware GraphFormer on Protocol A and fixed LOSO folds."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .. import contract as C


PROTOCOLS = (
    ("yja_holdout", None, "yja_e02"),
    ("loso", "test_ajh", "loso_ajh"),
    ("loso", "test_lmh", "loso_lmh"),
    ("loso", "test_mhw", "loso_mhw"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "yja_e02", "loso", "loso_ajh", "loso_lmh", "loso_mhw"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--rerun", action="store_true",
        help="run even when a completed result.json already exists",
    )
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for exp, fold, name in PROTOCOLS:
        if args.only not in {"all", name}:
            if args.only != "loso" or not name.startswith("loso_"):
                continue
        source = C.WORK_ROOT / "runs" / f"robust_gf_{name}" / "best_model.pt"
        destination = C.WORK_ROOT / "runs" / f"impact_gf_{name}"
        if not source.exists():
            raise FileNotFoundError(f"robust checkpoint not found: {source}")
        completed = (destination / "result.json").exists() and not args.rerun
        if completed:
            print(f"[skip training] completed {destination}", flush=True)
        else:
            command = [
                sys.executable, "-m", "notifi_pose.tools.train",
                "--exp", exp, "--arch", "impact_graphformer", "--decoder", "hybrid",
                "--init-checkpoint", str(source), "--tag", f"impact_gf_{name}",
                "--epochs", str(args.epochs), "--patience", str(args.patience),
                "--batch-size", str(args.batch_size), "--hidden", "128",
                "--temporal-layers", "3", "--heads", "4", "--graph-blocks", "2",
                "--lr", "0.0001", "--baseline", "sub", "--link-dropout", "0.15",
                "--backbone-lr-scale", "0.10", "--refiner-warmup-epochs", "2",
                "--rf-augment", "--balanced-batches", "--group-dro-eta", "0.01",
                "--lambda-root", "1.0", "--lambda-bone", "0.1",
                "--lambda-cls", "0.03", "--lambda-risk", "0.03",
                "--lambda-velocity", "0.10", "--lambda-motion", "0.05",
                "--lambda-acceleration", "0.005", "--lambda-jerk", "0.001",
                "--lambda-impact", "0.50", "--lambda-coarse", "0.20",
                "--lambda-contact", "0.02", "--lambda-phase", "0.03",
                "--lambda-foot-slide", "0.01", "--lambda-floor", "0.02",
                "--lambda-domain", "0.02", "--lambda-supcon", "0.02",
                "--motion-weight", "3.0", "--domain-grl", "0.2",
                "--weight-average-start", "8",
            ]
            if fold:
                command.extend(("--fold", fold))
            print("\n$ " + " ".join(command), flush=True)
            subprocess.run(command, check=True, env=environment)

        calibrated = destination / "calibrated_model.pt"
        if not calibrated.exists() or args.rerun:
            command = [
                sys.executable, "-m", "notifi_pose.tools.calibrate_refiner",
                str(destination / "best_model.pt"), "--exp", exp,
            ]
            if fold:
                command.extend(("--fold", fold))
            print("\n$ " + " ".join(command), flush=True)
            subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
