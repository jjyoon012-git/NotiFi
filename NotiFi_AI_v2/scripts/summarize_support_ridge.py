"""고정 support-ridge 5-seed 결과를 논문 표에 쓸 compact JSON으로 집계한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


METRICS = (
    "action_accuracy",
    "action_macro_f1",
    "risk_accuracy",
    "risk_macro_f1",
    "danger_recall",
    "danger_action_accuracy",
    "safe_to_danger_rate",
    "worst_site_action",
)


def distribution(values: list[float]) -> dict[str, float]:
    """반복 seed 값의 평균과 모집단 표준편차를 반환한다."""
    return {"mean": mean(values), "std": pstdev(values)}


def main() -> None:
    """seed별 원본 JSON에서 baseline과 고정 ridge의 핵심 지표만 집계한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="NOTIFI-AI-V2-SUPPORT-RIDGE-FIXED-5SEED",
    )
    parser.add_argument(
        "--configuration-source",
        default="configs/cal44_support_ridge_fixed.json",
    )
    parser.add_argument(
        "--protocol",
        default=(
            "source nested-LOSO; fixed config selected by source-inner mode; "
            "danger support excluded from query"
        ),
    )
    options = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8-sig")) for path in options.inputs]
    if len({run["support_seed"] for run in runs}) != len(runs):
        raise ValueError("support seeds must be unique")
    aggregate = {
        side: {
            metric: distribution([float(run[side][metric]) for run in runs])
            for metric in METRICS
        }
        for side in ("baseline", "selected")
    }
    result = {
        "run": options.run_name,
        "protocol": options.protocol,
        "configuration_source": options.configuration_source,
        "seeds": [
            {
                "support_seed": int(run["support_seed"]),
                "baseline": run["baseline"],
                "selected": run["selected"],
            }
            for run in runs
        ],
        "aggregate": aggregate,
        "outer_holdout_used_for_selection": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_or_pose_gt_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
