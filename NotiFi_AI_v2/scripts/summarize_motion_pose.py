"""고정 motion-ridge pose 결과를 5-seed 평균과 표준편차로 집계한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


METRICS = (
    "pose_cm",
    "distal_cm",
    "pa_pose_cm",
    "danger_pose_cm",
    "danger_distal_cm",
)


def main() -> None:
    """seed별 pose JSON을 읽고 matched baseline과 motion-ridge를 집계한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8-sig")) for path in options.inputs]
    if len({run["support_seed"] for run in runs}) != len(runs):
        raise ValueError("support seeds must be unique")
    aggregate = {}
    for side in ("baseline", "outer"):
        aggregate[side] = {}
        for metric in METRICS:
            values = [float(run["aggregate"][side][metric]) for run in runs]
            aggregate[side][metric] = {
                "mean": mean(values),
                "std": pstdev(values),
            }
    result = {
        "run": "NOTIFI-AI-V2-MOTION-RIDGE-POSE-FIXED-5SEED",
        "protocol": (
            "source nested-LOSO; fixed reg=100, mixture=0.5, risk_sqrt gate; "
            "basic and danger support labels only"
        ),
        "aggregate": aggregate,
        "seeds": [
            {
                "support_seed": int(run["support_seed"]),
                "baseline": run["aggregate"]["baseline"],
                "selected": run["aggregate"]["outer"],
            }
            for run in runs
        ],
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
        "calibration_pose_gt_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
