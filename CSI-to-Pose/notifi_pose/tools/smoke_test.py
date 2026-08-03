"""파서 · 시간정렬 · GT 타깃 통합 점검.

계획서(S2/S3/S4)의 완료 조건을 실제 데이터로 확인한다.

실행:
    python -m notifi_pose.tools.smoke_test                 # subject/env 조합당 2 trial
    python -m notifi_pose.tools.smoke_test --n 5
    python -m notifi_pose.tools.smoke_test --fall-check 20 # 낙상 정렬 검증만 크게

산출:
    work_v2/reports/smoke_test.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio import align, csi_parser, index as idx, targets


def check_trial(row) -> dict:
    """trial 하나를 끝까지 로드하고 계약을 검사한다."""
    t_start = time.perf_counter()

    tgt = targets.load_pose_target(row.gt_pose_path)
    ft = align.frame_times(row.csi_path, tgt.n_frames, row.time_method)
    csi = csi_parser.load_csi_trial(row.csi_path, grid_times=ft)

    elapsed = time.perf_counter() - t_start

    # --- 계약 검사 ---
    problems = []
    if csi.iq.shape != (tgt.n_frames, C.N_LINKS, C.N_LIVE_SUBCARRIERS, 2):
        problems.append(f"iq shape {csi.iq.shape}")
    if csi.link_mask.shape != (tgt.n_frames, C.N_LINKS):
        problems.append(f"link_mask shape {csi.link_mask.shape}")
    if tgt.pose_rel.shape != (tgt.n_frames, C.N_JOINTS, 3):
        problems.append(f"pose_rel shape {tgt.pose_rel.shape}")
    if not np.isfinite(csi.iq).all():
        problems.append("iq has non-finite")
    if not np.allclose(tgt.pose_rel[:, C.ROOT_JOINT], 0.0, atol=1e-5):
        problems.append("pose_rel root != 0")
    if not np.isfinite(ft).all() or (np.diff(ft) <= 0).any():
        problems.append("frame_times not strictly increasing")

    # 뼈 길이 불변성 — 관절 순서 계약이 맞는지에 대한 강한 검사
    bl = targets.bone_lengths(tgt.absolute())
    bone_rel_std = float((bl.std(axis=0) / np.maximum(bl.mean(axis=0), 1e-6)).max())
    if bone_rel_std > 1e-3:
        problems.append(f"bone length varies (rel_std={bone_rel_std:.4f})")

    # 바닥 정렬 — 최저 관절이 y≈0 근처여야 한다
    min_y = float(tgt.absolute()[:, :, C.UP_AXIS].min())

    return {
        "trial_id": row.trial_id,
        "subject": row.subject,
        "environment": row.environment,
        "scenario_id": row.scenario_id,
        "time_method": row.time_method,
        "n_frames": tgt.n_frames,
        "load_sec": round(elapsed, 3),
        "bad_row_ratio": round(csi.meta["bad_row_ratio"], 5),
        "link_hz": {k: round(v["hz"], 1) for k, v in csi.meta["links"].items()},
        "link_coverage": {k: round(v, 3) for k, v in csi.meta["link_coverage"].items()},
        "any_link_coverage": round(csi.meta["any_link_coverage"], 3),
        "all_link_coverage": round(csi.meta["all_link_coverage"], 3),
        "bone_rel_std": round(bone_rel_std, 6),
        "floor_min_y": round(min_y, 3),
        "has_smpl": tgt.meta["has_smpl"],
        "problems": problems,
    }


def check_alignment(rows) -> dict:
    """낙상 trial 에서 GT 관절 속도 피크와 CSI 모션 에너지 피크의 시차.

    두 전역 최대 피크가 같은 신체 사건이라는 보장은 없으므로 진단값으로만 기록한다.
    시간 정렬의 계약 검사는 기록 timestamp의 단조성, 길이, 시작/끝 보존으로 수행한다.
    """
    diffs = []
    for row in rows:
        tgt = targets.load_pose_target(row.gt_pose_path)
        ft = align.frame_times(row.csi_path, tgt.n_frames, row.time_method)
        csi = csi_parser.load_csi_trial(row.csi_path, grid_times=ft)

        t_gt = align.joint_velocity_peak(ft, tgt.absolute())
        t_csi = align.motion_energy_peak(csi.times, csi.iq, csi.link_mask)
        if np.isfinite(t_gt) and np.isfinite(t_csi):
            diffs.append({
                "trial_id": row.trial_id,
                "subject": row.subject,
                "time_method": row.time_method,
                "gt_peak_s": round(float(t_gt), 3),
                "csi_peak_s": round(float(t_csi), 3),
                "abs_diff_ms": round(abs(float(t_gt - t_csi)) * 1000.0, 1),
            })

    if not diffs:
        return {"n": 0}
    d = pd.DataFrame(diffs)
    by_method = {
        m: round(float(g["abs_diff_ms"].median()), 1)
        for m, g in d.groupby("time_method")
    }
    return {
        "n": len(d),
        "median_abs_diff_ms": round(float(d["abs_diff_ms"].median()), 1),
        "p90_abs_diff_ms": round(float(d["abs_diff_ms"].quantile(0.9)), 1),
        "median_by_time_method": by_method,
        "worst": d.nlargest(5, "abs_diff_ms").to_dict("records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2,
                    help="subject x environment 조합당 표본 trial 수")
    ap.add_argument("--fall-check", type=int, default=20,
                    help="정렬 검증에 쓸 낙상(danger) trial 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = idx.pose_only(idx.load_all_index())
    print(f"[smoke] pose trials: {len(df):,}")

    sample = (df.groupby(["subject", "environment"], group_keys=False)
                .apply(lambda g: g.sample(min(args.n, len(g)), random_state=args.seed),
                       include_groups=True))
    print(f"[smoke] loading {len(sample)} trials ...")

    results = [check_trial(r) for r in sample.itertuples()]
    bad = [r for r in results if r["problems"]]

    print()
    print(f"{'trial_id':<22} {'method':<14} {'link_hz':<26} {'cover(any/all)':<16} {'bone':<9} floor_y")
    for r in results:
        hz = "/".join(str(int(v)) for v in r["link_hz"].values())
        cov = f"{r['any_link_coverage']:.2f}/{r['all_link_coverage']:.2f}"
        print(f"{r['trial_id']:<22} {r['time_method']:<14} {hz:<26} {cov:<16} "
              f"{r['bone_rel_std']:<9.6f} {r['floor_min_y']:+.3f}")

    print()
    load_times = [r["load_sec"] for r in results]
    print(f"[smoke] trial 로딩 시간: 중앙 {np.median(load_times):.2f}s  "
          f"최대 {max(load_times):.2f}s  -> 3,155 trial 추정 "
          f"{np.median(load_times) * 3155 / 60:.0f}분 (단일 프로세스)")

    danger = df[df["risk"] == "danger"]
    fall_rows = list(danger.sample(min(args.fall_check, len(danger)),
                                   random_state=args.seed).itertuples())
    print(f"[smoke] 정렬 검증: danger trial {len(fall_rows)}개 ...")
    align_report = check_alignment(fall_rows)
    print(f"  GT 속도 피크 vs CSI 에너지 피크 시차: "
          f"중앙 {align_report.get('median_abs_diff_ms')} ms, "
          f"p90 {align_report.get('p90_abs_diff_ms')} ms")
    print(f"  time_method 별 중앙: {align_report.get('median_by_time_method')}")

    report = {
        "preproc_version": C.PREPROC_VERSION,
        "n_checked": len(results),
        "n_with_problems": len(bad),
        "problems": [r for r in bad],
        "trials": results,
        "alignment": align_report,
    }
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rp = C.REPORT_DIR / "smoke_test.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[smoke] wrote {rp}")

    if bad:
        print(f"[smoke] FAILED - {len(bad)} trials with contract violations:")
        for r in bad:
            print(f"    - {r['trial_id']}: {r['problems']}")
        return 1
    print(f"[smoke] OK - {len(results)} trials passed contract checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
