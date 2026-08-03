"""Train the coherent-motion refiner while keeping the robust backbone frozen."""

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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for exp, fold, name in PROTOCOLS:
        if args.only not in {"all", name}:
            if args.only != "loso" or not name.startswith("loso_"):
                continue
        source = C.WORK_ROOT / "runs" / f"robust_gf_{name}" / "best_model.pt"
        destination = C.WORK_ROOT / "runs" / f"coherent_gf_{name}"
        if (destination / "result.json").exists() and not args.rerun:
            print(f"[skip] completed {destination}", flush=True)
            continue
        command = [
            sys.executable, "-m", "notifi_pose.tools.train",
            "--exp", exp, "--arch", "impact_graphformer", "--decoder", "hybrid",
            "--init-checkpoint", str(source), "--tag", f"coherent_gf_{name}",
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--batch-size", str(args.batch_size), "--hidden", "128",
            "--temporal-layers", "3", "--heads", "4", "--graph-blocks", "2",
            "--lr", "0.0001", "--baseline", "sub", "--link-dropout", "0.10",
            "--backbone-lr-scale", "0.0", "--refiner-warmup-epochs", str(args.epochs),
            "--rf-augment", "--balanced-batches", "--group-dro-eta", "0.01",
            "--lambda-root", "0.0", "--lambda-bone", "0.10",
            "--lambda-cls", "0.0", "--lambda-risk", "0.0",
            "--lambda-velocity", "0.05", "--lambda-acceleration", "0.002",
            "--lambda-jerk", "0.0005", "--lambda-impact", "0.25",
            "--lambda-displacement", "0.25", "--lambda-foot-slide", "0.01",
            "--lambda-floor", "0.01", "--motion-weight", "3.0",
            "--weight-average-start", "0",
        ]
        if fold:
            command.extend(("--fold", fold))
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
