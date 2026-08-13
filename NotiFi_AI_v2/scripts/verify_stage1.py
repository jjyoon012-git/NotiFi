"""Run a lightweight shape and invariance check without dataset access."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notifi_ai_v2.model import MotionCalibratedEncoder


def main() -> None:
    torch.manual_seed(7)
    csi = torch.randn(2, 64, 3, 114, 2)
    mask = torch.ones(2, 64, 3, dtype=torch.bool)
    mask[1, :, 2] = False
    model = MotionCalibratedEncoder().eval()
    with torch.no_grad():
        output = model(csi, mask)
    print(json.dumps({
        "action_logits": list(output["action_logits"].shape),
        "risk_logits": list(output["risk_logits"].shape),
        "motion": list(output["motion"].shape),
        "finite": all(torch.isfinite(value).all().item() for value in output.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
