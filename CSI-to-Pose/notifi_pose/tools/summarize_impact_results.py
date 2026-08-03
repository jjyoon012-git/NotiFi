"""Consolidate robust-versus-calibrated impact results into report artifacts."""

from __future__ import annotations

import json

import pandas as pd

from .. import contract as C


METRICS = (
    "mpjpe_m", "dynamic_mpjpe_m", "distal_mpjpe_m", "head_mpjpe_m",
    "impact_mpjpe_m", "root_error_m", "pose_speed_ratio", "root_speed_ratio",
)

CASES = {
    "yja_E02": (
        "robust_gf_yja_e02/eval_yja_E02_smooth5/summary.json",
        "impact_gf_yja_e02/eval_calibrated_yja_E02_smooth5/summary.json",
        "impact_gf_yja_e02/calibrated_model.pt",
    ),
    "loso_ajh": (
        "robust_gf_loso_ajh/eval_loso_test_ajh_test_smooth5/summary.json",
        "impact_gf_loso_ajh/eval_calibrated_test_ajh_smooth5/summary.json",
        "impact_gf_loso_ajh/calibrated_model.pt",
    ),
    "loso_lmh": (
        "robust_gf_loso_lmh/eval_loso_test_lmh_test_smooth5/summary.json",
        "impact_gf_loso_lmh/eval_calibrated_test_lmh_smooth5/summary.json",
        "impact_gf_loso_lmh/calibrated_model.pt",
    ),
    "loso_mhw": (
        "robust_gf_loso_mhw/eval_loso_test_mhw_test_smooth5/summary.json",
        "impact_gf_loso_mhw/eval_calibrated_test_mhw_smooth5/summary.json",
        "impact_gf_loso_mhw/calibrated_model.pt",
    ),
}


def read_overall(path):
    return json.loads(path.read_text(encoding="utf-8"))["overall"]


def main() -> int:
    runs = C.WORK_ROOT / "runs"
    records = []
    details = {}
    for protocol, (baseline_rel, final_rel, checkpoint_rel) in CASES.items():
        baseline = read_overall(runs / baseline_rel)
        final = read_overall(runs / final_rel)
        comparison = {}
        row = {"protocol": protocol, "checkpoint": str(runs / checkpoint_rel)}
        for metric in METRICS:
            comparison[metric] = {
                "baseline": baseline[metric],
                "final": final[metric],
                "delta": final[metric] - baseline[metric],
            }
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"final_{metric}"] = final[metric]
            row[f"delta_{metric}"] = final[metric] - baseline[metric]
        details[protocol] = comparison
        records.append(row)

    frame = pd.DataFrame(records)
    loso = frame[frame.protocol.str.startswith("loso_")]
    loso_mean = {
        metric: {
            "baseline": float(loso[f"baseline_{metric}"].mean()),
            "final": float(loso[f"final_{metric}"].mean()),
            "delta": float(loso[f"delta_{metric}"].mean()),
        }
        for metric in METRICS
    }
    coherent = read_overall(
        runs / "coherent_gf_yja_e02/eval_yja_E02_smooth5/summary.json"
    )
    coherent_rejection = {
        "protocol": "yja_E02",
        "reason": "five-frame displacement loss did not improve coherent motion",
        "baseline_pose_speed_ratio": details["yja_E02"]["pose_speed_ratio"]["baseline"],
        "coherent_pose_speed_ratio": coherent["pose_speed_ratio"],
        "baseline_mpjpe_m": details["yja_E02"]["mpjpe_m"]["baseline"],
        "coherent_mpjpe_m": coherent["mpjpe_m"],
    }

    output = C.WORK_ROOT / "reports"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation": "CSI-only, pose trials, five-frame smoothing",
        "selection": "checkpoint and joint residual scales selected on validation only",
        "protocols": details,
        "loso_mean": loso_mean,
        "rejected_experiment": coherent_rejection,
    }
    (output / "impact_calibrated_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frame.to_csv(output / "impact_calibrated_results.csv", index=False)

    lines = [
        "# Impact-aware calibrated GraphFormer results",
        "",
        "All checkpoint and residual-scale choices use validation data only. "
        "Test sets are used once for the table below.",
        "",
        "| Protocol | MPJPE (cm) | Dynamic (cm) | Distal (cm) | Impact (cm) | Head (cm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        def pair(metric):
            return f"{row[f'baseline_{metric}']*100:.2f} -> {row[f'final_{metric}']*100:.2f}"
        lines.append(
            f"| {row['protocol']} | {pair('mpjpe_m')} | {pair('dynamic_mpjpe_m')} "
            f"| {pair('distal_mpjpe_m')} | {pair('impact_mpjpe_m')} "
            f"| {pair('head_mpjpe_m')} |"
        )
    lines.extend([
        "",
        "## LOSO mean",
        "",
        f"- MPJPE: {loso_mean['mpjpe_m']['baseline']*100:.2f} -> "
        f"{loso_mean['mpjpe_m']['final']*100:.2f} cm",
        f"- Dynamic MPJPE: {loso_mean['dynamic_mpjpe_m']['baseline']*100:.2f} -> "
        f"{loso_mean['dynamic_mpjpe_m']['final']*100:.2f} cm",
        f"- Distal MPJPE: {loso_mean['distal_mpjpe_m']['baseline']*100:.2f} -> "
        f"{loso_mean['distal_mpjpe_m']['final']*100:.2f} cm",
        f"- Impact MPJPE: {loso_mean['impact_mpjpe_m']['baseline']*100:.2f} -> "
        f"{loso_mean['impact_mpjpe_m']['final']*100:.2f} cm",
        "",
        "## Rejected experiment",
        "",
        "The coherent-displacement variant was rejected: on yja E02 its smoothed "
        f"pose-speed ratio changed from {coherent_rejection['baseline_pose_speed_ratio']:.3f} "
        f"to {coherent_rejection['coherent_pose_speed_ratio']:.3f} while MPJPE changed from "
        f"{coherent_rejection['baseline_mpjpe_m']*100:.2f} to "
        f"{coherent_rejection['coherent_mpjpe_m']*100:.2f} cm.",
        "",
        "The calibrated impact model is a conservative spatial improvement. It does "
        "not solve the remaining low-frequency motion-amplitude collapse.",
    ])
    (output / "impact_calibrated_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["loso_mean"], indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
