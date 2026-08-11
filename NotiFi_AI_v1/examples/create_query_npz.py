"""Package one already-preprocessed CSI tensor for the HTTP API."""

from pathlib import Path

import numpy as np


def main() -> None:
    source = Path("query_arrays")
    csi = np.load(source / "csi.npy").astype(np.float32)
    link_mask = np.load(source / "link_mask.npy").astype(bool)
    np.savez_compressed("query.npz", csi=csi, link_mask=link_mask)


if __name__ == "__main__":
    main()
