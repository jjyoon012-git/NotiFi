"""Run leakage-free motion-prior and V3 training for the agreed protocols.

The four runs are Protocol A (yja E02 holdout) and the three fixed work_v2
LOSO folds. Each motion prior sees only that run's CSI-training subjects.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .. import contract as C


PROTOCOLS = (
    ("yja_holdout", None, "yja_e02"),
    ("loso", "test_ajh", "loso_ajh"),
    ("loso", "test_lmh", "loso_lmh"),
    ("loso", "test_mhw", "loso_mhw"),
)


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "yja_e02", "loso_ajh", "loso_lmh", "loso_mhw"), default="all")
    parser.add_argument("--prior-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=96)
    args = parser.parse_args()

    selected = [p for p in PROTOCOLS if args.only in ("all", p[2])]
    for exp, fold, name in selected:
        prior_tag = f"v3_prior_{name}"
        run_tag = f"v3_robust_{name}"
        prior_path = C.WORK_ROOT / "priors" / prior_tag / "motion_prior.pt"

        prior = [
            sys.executable, "-m", "notifi_pose.tools.pretrain_motion_prior",
            "--exp", exp, "--hidden", str(args.hidden), "--layers", "2",
            "--heads", "4", "--graph-blocks", "2",
            "--epochs", str(args.prior_epochs), "--batch-size", "32",
            "--tag", prior_tag,
        ]
        if fold:
            prior.extend(("--fold", fold))
        run(prior)

        training = [
            sys.executable, "-m", "notifi_pose.tools.train",
            "--exp", exp, "--arch", "v3", "--tag", run_tag,
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--batch-size", str(args.batch_size), "--hidden", str(args.hidden),
            "--temporal-layers", "2", "--heads", "4", "--graph-blocks", "2",
            "--frequency-tokens", "8", "--lr", "0.0005",
            "--link-dropout", "0.15", "--rf-augment", "--balanced-batches",
            "--group-dro-eta", "0.01", "--lambda-root", "1.0",
            "--lambda-bone", "0.1", "--lambda-cls", "0.05",
            "--lambda-risk", "0.05", "--lambda-velocity", "0.10",
            "--lambda-motion", "0.05", "--lambda-acceleration", "0.005",
            "--lambda-contact", "0.02", "--lambda-phase", "0.03",
            "--lambda-foot-slide", "0.01", "--lambda-floor", "0.02",
            "--lambda-domain", "0.03", "--lambda-supcon", "0.03",
            "--lambda-latent", "0.20", "--motion-weight", "3.0",
            "--domain-grl", "0.2", "--motion-prior", str(prior_path),
            "--weight-average-start", "10",
        ]
        if fold:
            training.extend(("--fold", fold))
        run(training)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
