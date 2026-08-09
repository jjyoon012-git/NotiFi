"""기존 CAL40 전체 bundle에 낙상 calibration 계약을 넣어 단일 CAL44로 만든다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from notifi_pose import contract as C


def main() -> None:
    """외부 checkpoint를 참조하지 않는 CAL44 단일 배포 파일을 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    bundle = torch.load(options.input, map_location="cpu", weights_only=True)
    if bundle.get("bundle_version") != "cal40_fixed_deep_action_safety_risk_v1":
        raise RuntimeError("input must be the locked CAL40 full deployment bundle")
    if any(bundle.get(key) is not False for key in (
        "sealed_yja_used", "target_subject_used", "query_labels_or_pose_gt_used",
    )):
        raise RuntimeError("input bundle is not sealed-target clean")
    bundle["bundle_version"] = "cal44_fall_support_single_bundle_v1"
    bundle["danger_support_contract"] = {
        "classes": list(C.DANGER_CALIBRATION_CLASSES),
        "shots_per_class": 1,
        "query_overlap_allowed": False,
        "pose_gt_required": False,
        "video_required": False,
    }
    # 낙상 support는 우선 danger 5종 내부 분포만 보정한다. 위험 여부 logit은
    # source nested-LOSO 검증 전까지 건드리지 않아 false alarm 증가를 막는다.
    bundle["danger_support_config"] = {
        "temperature": 0.10,
        "subtype_weight": 0.50,
        "action_margin_gain": 0.0,
        "action_bias": 0.0,
        "risk_margin_gain": 0.0,
        "risk_bias": 0.0,
    }
    bundle["provenance"] = dict(bundle.get("provenance", {}))
    bundle["provenance"]["upgraded_from_bundle_version"] = (
        "cal40_fixed_deep_action_safety_risk_v1"
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, options.output)
    print(options.output)


if __name__ == "__main__":
    main()
