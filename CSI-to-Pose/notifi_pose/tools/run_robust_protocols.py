"""Run the final robust GraphFormer on Protocol A and fixed LOSO folds."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    selected = []
    for protocol in PROTOCOLS:
        name = protocol[2]
        if args.only == "all" or args.only == name:
            selected.append(protocol)
        elif args.only == "loso" and name.startswith("loso_"):
            selected.append(protocol)

    for exp, fold, name in selected:
        command = [
            sys.executable, "-m", "notifi_pose.tools.train",
            "--exp", exp, "--arch", "robust_graphformer", "--decoder", "hybrid",
            "--tag", f"robust_gf_{name}", "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--batch-size", str(args.batch_size),
            "--hidden", "128", "--temporal-layers", "3", "--heads", "4",
            "--graph-blocks", "2", "--lr", "0.0005", "--baseline", "sub",
            "--link-dropout", "0.25", "--rf-augment", "--balanced-batches",
            "--group-dro-eta", "0.01", "--lambda-root", "1.0",
            "--lambda-bone", "0.1", "--lambda-cls", "0.05",
            "--lambda-risk", "0.05", "--lambda-velocity", "0.10",
            "--lambda-motion", "0.05", "--lambda-acceleration", "0.005",
            "--lambda-contact", "0.02", "--lambda-phase", "0.03",
            "--lambda-foot-slide", "0.01", "--lambda-floor", "0.02",
            "--lambda-domain", "0.03", "--lambda-supcon", "0.03",
            "--motion-weight", "3.0", "--domain-grl", "0.2",
            "--weight-average-start", "15",
        ]
        if fold:
            command.extend(("--fold", fold))
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
