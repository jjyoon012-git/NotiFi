"""검증된 CAL44 bundle에 v2 support/motion ridge 설정을 넣어 단일 PT로 내보낸다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    """artifact와 검증 보고서의 재현용 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """기존 검증 가중치를 복사하고 새 calibration 설정만 추가한 bundle을 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ridge-config", type=Path, required=True)
    parser.add_argument("--classification-result", type=Path, required=True)
    parser.add_argument("--pose-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    bundle = torch.load(options.base, map_location="cpu", weights_only=False)
    for key in ("sealed_yja_used", "target_subject_used", "query_labels_or_pose_gt_used"):
        if bundle.get(key) is not False:
            raise RuntimeError(f"unclean base artifact flag: {key}")
    ridge = json.loads(options.ridge_config.read_text(encoding="utf-8"))
    global_configs = list(ridge["fold_configs"].values())
    if any(config != global_configs[0] for config in global_configs[1:]):
        raise ValueError("deployment ridge config must be global, not fold-specific")

    bundle["bundle_version"] = "notifi_ai_v2_full_support_ridge_v2"
    bundle["support_ridge_config"] = dict(global_configs[0])
    bundle["warning_support_contract"] = dict(ridge["warning_support_contract"])
    bundle["risk_from_action"] = bool(ridge["risk_from_action"])
    bundle["risk_profiles"] = {
        "conservative": {"danger_bias": 0.0},
        "safety": {"danger_bias": 0.0},
    }
    bundle["motion_ridge_config"] = {
        "classes": [0, 1, 2, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16],
        "regularization": 100.0,
        "mixture": 0.5,
        "gate": "risk_sqrt",
    }
    bundle["deployment_action_profile_mode"] = "average"
    bundle["v2_provenance"] = {
        "base_artifact": str(options.base),
        "base_artifact_sha256": sha256(options.base),
        "ridge_config": str(options.ridge_config),
        "ridge_config_sha256": sha256(options.ridge_config),
        "classification_result": str(options.classification_result),
        "classification_result_sha256": sha256(options.classification_result),
        "pose_result": str(options.pose_result),
        "pose_result_sha256": sha256(options.pose_result),
        "sealed_yja_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, options.output)
    print(json.dumps({
        "output": str(options.output),
        "bytes": options.output.stat().st_size,
        "sha256": sha256(options.output),
        "bundle_version": bundle["bundle_version"],
    }, indent=2))


if __name__ == "__main__":
    main()
