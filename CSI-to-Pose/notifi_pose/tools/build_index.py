"""전체 trial 인덱스 생성 + 데이터셋 전수 검증.

실행:
    python -m notifi_pose.tools.build_index
    python -m notifi_pose.tools.build_index --no-verify-files   # 파일 존재 확인 생략(빠름)

산출:
    work_v2/index/all_index.csv      3,299행
    work_v2/reports/dataset_verify.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio import index as idx


def _check(report: dict, name: str, ok: bool, detail) -> bool:
    report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    return ok


def verify(df: pd.DataFrame, verify_files: bool = True) -> dict:
    """인덱스 무결성 검증. 반환값의 `passed` 가 최종 판정."""
    report: dict = {
        "dataset_root": str(C.DATASET_ROOT),
        "preproc_version": C.PREPROC_VERSION,
        "n_rows": int(len(df)),
        "checks": [],
    }

    # --- 개수 ---
    n_pose = int((df["task"] == C.TASK_POSE).sum())
    n_cls = int((df["task"] == C.TASK_CLS).sum())
    _check(report, "trial_count_total", len(df) == C.EXPECTED_TRIALS_TOTAL,
           {"got": len(df), "expected": C.EXPECTED_TRIALS_TOTAL})
    _check(report, "trial_count_pose", n_pose == C.EXPECTED_TRIALS_POSE,
           {"got": n_pose, "expected": C.EXPECTED_TRIALS_POSE})
    _check(report, "trial_count_classification", n_cls == C.EXPECTED_TRIALS_CLS,
           {"got": n_cls, "expected": C.EXPECTED_TRIALS_CLS})

    # --- 중복 ---
    dup = df["trial_id"].duplicated().sum()
    _check(report, "trial_id_unique", dup == 0, {"duplicates": int(dup)})

    # --- 축 값 ---
    _check(report, "subjects", set(df["subject"]) == set(C.SUBJECTS),
           {"got": sorted(set(df["subject"]))})
    _check(report, "environments", set(df["environment"]) == set(C.ENVIRONMENTS),
           {"got": sorted(set(df["environment"]))})
    _check(report, "class_count", df["class_id"].nunique() == C.N_CLASSES,
           {"got": int(df["class_id"].nunique()), "expected": C.N_CLASSES})

    # --- gt_schema ---
    schemas = set(df["gt_schema"].dropna())
    _check(report, "gt_schema_values",
           schemas <= {C.GT_SCHEMA_JOINTS, C.GT_SCHEMA_SMPL}, {"got": sorted(schemas)})
    # classification_only 는 GT 가 없는 것이 정상
    bad_gt = df[(df["task"] == C.TASK_CLS) & df["gt_pose"].notna()]
    _check(report, "classification_has_no_gt", len(bad_gt) == 0,
           {"violations": bad_gt["trial_id"].tolist()[:10]})
    missing_gt = df[(df["task"] == C.TASK_POSE) & df["gt_pose"].isna()]
    _check(report, "pose_has_gt", len(missing_gt) == 0,
           {"violations": missing_gt["trial_id"].tolist()[:10]})

    # --- fold 누수 ---
    for fold in C.LOSO_FOLDS:
        col = f"fold_{fold}"
        held_out = fold.replace("test_", "")
        assigned = df[col].notna().sum()
        _check(report, f"{fold}_covers_all", assigned == len(df),
               {"assigned": int(assigned), "total": len(df)})
        # test split 은 held-out subject 만, train/val 에는 없어야 한다
        test_subj = set(df.loc[df[col] == "test", "subject"])
        dev_subj = set(df.loc[df[col].isin(["train", "val"]), "subject"])
        _check(report, f"{fold}_subject_isolation",
               test_subj == {held_out} and held_out not in dev_subj,
               {"test_subjects": sorted(test_subj), "dev_subjects": sorted(dev_subj)})
        # 각 split 에 전 환경/전 클래스가 있어야 한다
        for split in C.SPLITS:
            sub = df[df[col] == split]
            _check(report, f"{fold}_{split}_all_environments",
                   set(sub["environment"]) == set(C.ENVIRONMENTS),
                   {"got": sorted(set(sub["environment"]))})
            _check(report, f"{fold}_{split}_all_classes",
                   sub["class_id"].nunique() == C.N_CLASSES,
                   {"got": int(sub["class_id"].nunique())})

    # --- 시간 정렬 정보 ---
    tm = Counter(df["time_method"].fillna("MISSING"))
    _check(report, "time_method_known",
           set(tm) <= {C.TIME_METHOD_TIMESTAMPS, C.TIME_METHOD_UNIFORM},
           dict(tm))
    unusable = df[df["timestamp_usable"] != True]  # noqa: E712 — NaN 도 잡아야 함
    _check(report, "timestamp_usable", len(unusable) <= 1,
           {"n_unusable": int(len(unusable)),
            "trial_ids": unusable["trial_id"].tolist()[:10]})

    # --- 파일 존재 ---
    if verify_files:
        missing = {"csi": [], "gt_pose": [], "video_timestamps": []}
        for row in df.itertuples():
            csi = row.csi_path
            if csi is None or not csi.exists():
                missing["csi"].append(row.trial_id)
            else:
                vts = csi.parent / "video_timestamps.csv"
                if not vts.exists():
                    missing["video_timestamps"].append(row.trial_id)
            if row.task == C.TASK_POSE:
                gt = row.gt_pose_path
                if gt is None or not gt.exists():
                    missing["gt_pose"].append(row.trial_id)
        for key, ids in missing.items():
            _check(report, f"file_exists_{key}", len(ids) == 0,
                   {"n_missing": len(ids), "trial_ids": ids[:10]})

    report["passed"] = all(c["ok"] for c in report["checks"])
    report["n_failed"] = sum(1 for c in report["checks"] if not c["ok"])
    return report


def summarize(df: pd.DataFrame) -> dict:
    """사람이 읽는 요약."""
    return {
        "by_subject": df.groupby("subject").size().to_dict(),
        "by_environment": df.groupby("environment").size().to_dict(),
        "by_task": df.groupby("task").size().to_dict(),
        "by_gt_schema": df["gt_schema"].fillna("NONE").value_counts().to_dict(),
        "by_time_method": df["time_method"].fillna("MISSING").value_counts().to_dict(),
        "by_risk": df.groupby("risk").size().to_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-fold", default="test_ajh", choices=C.LOSO_FOLDS,
                    help="union 기준 fold (어느 것을 써도 동일 universe)")
    ap.add_argument("--no-verify-files", action="store_true",
                    help="파일 존재 확인 생략 (3,299 trial × 3파일 stat 이라 수십 초 걸림)")
    args = ap.parse_args()

    print(f"[build_index] dataset_root = {C.DATASET_ROOT}")
    df = idx.build_all_index(reference_fold=args.reference_fold)
    print(f"[build_index] {len(df):,} trials")

    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out = idx.all_index_path()
    # 절대경로 컬럼은 머신 종속이라 저장하지 않는다 (로딩 시 재계산).
    df.drop(columns=[c for c in df.columns if c.endswith("_path")]).to_csv(
        out, index=False, encoding="utf-8")
    print(f"[build_index] wrote {out}")

    print("[build_index] verifying ...")
    report = verify(df, verify_files=not args.no_verify_files)
    report["summary"] = summarize(df)

    rp = C.REPORT_DIR / "dataset_verify.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                             default=lambda o: int(o) if isinstance(o, np.integer) else str(o)),
                  encoding="utf-8")
    print(f"[build_index] wrote {rp}")

    print()
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print()
    if report["passed"]:
        print(f"[build_index] OK - {len(report['checks'])} checks passed")
        return 0
    print(f"[build_index] FAILED - {report['n_failed']}/{len(report['checks'])} checks failed:")
    for c in report["checks"]:
        if not c["ok"]:
            print(f"    - {c['name']}: {c['detail']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
