"""Pretrain leakage-safe motion priors and train latent-flow pose models."""

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


def selected_protocols(choice: str):
    for protocol in PROTOCOLS:
        name = protocol[2]
        if choice in {"all", name} or (choice == "loso" and name.startswith("loso_")):
            yield protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "yja_e02", "loso", "loso_ajh", "loso_lmh", "loso_mhw"),
        default="all",
    )
    parser.add_argument("--prior-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for exp, fold, name in selected_protocols(args.only):
        prior_dir = C.WORK_ROOT / "priors" / f"motion_prior_{name}"
        prior = prior_dir / "motion_prior.pt"
        if not prior.exists() or args.rerun:
            command = [
                sys.executable, "-m", "notifi_pose.tools.pretrain_motion_prior",
                "--exp", exp, "--tag", f"motion_prior_{name}",
                "--epochs", str(args.prior_epochs), "--hidden", "128",
                "--batch-size", "32", "--lr", "0.0005",
            ]
            if fold:
                command.extend(("--fold", fold))
            print("\n$ " + " ".join(command), flush=True)
            subprocess.run(command, check=True, env=environment)

        source = C.WORK_ROOT / "runs" / f"robust_gf_{name}" / "best_model.pt"
        destination = C.WORK_ROOT / "runs" / f"latent_flow_{name}"
        if (destination / "result.json").exists() and not args.rerun:
            print(f"[skip] completed {destination}", flush=True)
            continue
        command = [
            sys.executable, "-m", "notifi_pose.tools.train",
            "--exp", exp, "--arch", "latent_flow", "--decoder", "hybrid",
            "--init-checkpoint", str(source), "--motion-prior", str(prior),
            "--tag", f"latent_flow_{name}", "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--batch-size", str(args.batch_size),
            "--hidden", "128", "--temporal-layers", "3", "--heads", "4",
            "--graph-blocks", "2", "--flow-steps", "2", "--flow-noise", "0.25",
            "--lr", "0.0002", "--backbone-lr-scale", "0.05",
            "--refiner-warmup-epochs", "3", "--baseline", "sub",
            "--link-dropout", "0.15", "--rf-augment", "--balanced-batches",
            "--group-dro-eta", "0.01", "--lambda-root", "1.0",
            "--lambda-bone", "0.10", "--lambda-cls", "0.02",
            "--lambda-risk", "0.02", "--lambda-velocity", "0.05",
            "--lambda-motion", "0.02", "--lambda-acceleration", "0.002",
            "--lambda-jerk", "0.0005", "--lambda-impact", "0.25",
            "--lambda-displacement", "0.15", "--lambda-flow", "1.0",
            "--lambda-latent", "0.20", "--lambda-foot-slide", "0.01",
            "--lambda-floor", "0.01", "--lambda-domain", "0.01",
            "--lambda-supcon", "0.01", "--motion-weight", "3.0",
            "--weight-average-start", "8",
        ]
        if fold:
            command.extend(("--fold", fold))
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
