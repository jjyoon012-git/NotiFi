"""Example that packages already-preprocessed calibration tensors."""

from pathlib import Path

import numpy as np


def main() -> None:
    source = Path("calibration_arrays")
    absence_csi = np.load(source / "absence_csi.npy")
    absence_mask = np.load(source / "absence_mask.npy")
    support_csi = np.load(source / "support_csi.npy")
    support_mask = np.load(source / "support_mask.npy")
    support_action = np.load(source / "support_action.npy")
    support_risk = np.load(source / "support_risk.npy")
    np.savez_compressed(
        "calibration.npz",
        absence_csi=absence_csi,
        absence_mask=absence_mask,
        support_csi=support_csi,
        support_mask=support_mask,
        support_action=support_action,
        support_risk=support_risk,
    )


if __name__ == "__main__":
    main()

