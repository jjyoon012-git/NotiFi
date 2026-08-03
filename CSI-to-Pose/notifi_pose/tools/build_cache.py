"""캐시 생성 — csi.csv + gt_pose.npz 를 한 번만 파싱해서 memmap 배열로 굽는다.

대상: 개발셋(dev_index.csv) + 봉인셋(sealed_index.csv). 봉인셋도 캐시는 만들되
학습 코드가 실수로 집어가지 못하도록 인덱스에 `split_group` 으로 구분해 둔다.

실행:
    python -m notifi_pose.tools.build_cache
    python -m notifi_pose.tools.build_cache --workers 8
    python -m notifi_pose.tools.build_cache --rebuild     # 전처리 버전 바뀐 뒤
    python -m notifi_pose.tools.build_cache --limit 50    # 빠른 점검
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from .. import contract as C
from ..dataio import align, cache, csi_parser, targets


def _one(args) -> tuple[int, dict | None, str]:
    """trial 하나를 파싱해 배열로 만든다. 워커에서 실행."""
    row, trial_id, csi_rel, gt_rel, time_method, task = args
    csi_path = C.DATASET_ROOT / csi_rel
    try:
        if task == C.TASK_POSE and isinstance(gt_rel, str):
            tgt = targets.load_pose_target(C.DATASET_ROOT / gt_rel)
            n = min(tgt.n_frames, C.CACHE_FRAMES)
            ft = align.frame_times(csi_path, tgt.n_frames, time_method)[:n]
            pose_rel = tgt.pose_rel[:n]
            root = tgt.root[:n]
            valid = tgt.valid[:n]
        else:
            # absence(classification_only): GT 가 없다. CSI 만 캐시하고 valid=False.
            #
            # 프레임 수를 타임스탬프 행 수로 잡으면 안 된다 — lmh 는 그 로그에 행이
            # 빠져 있어서 10초짜리가 3.3초로 잘린다(실제로 99프레임까지 내려갔다).
            # 맞출 GT 프레임이 없으므로 그냥 규정 길이(10초=300프레임)의 균일 격자를
            # 쓰고, 시작점만 영상 타임스탬프의 첫 값에 맞춘다.
            n = C.NOMINAL_TRIAL_FRAMES
            ts = align.video_timestamps_path(csi_path)
            t0 = 0.0
            if ts.exists():
                e = pd.read_csv(ts, usecols=["pc_elapsed_s"])["pc_elapsed_s"]
                if len(e):
                    t0 = float(e.min())
            ft = (t0 + np.arange(n, dtype=np.float64) / C.TARGET_FPS).astype(np.float32)
            pose_rel = np.zeros((n, C.N_JOINTS, 3), dtype=np.float32)
            root = np.zeros((n, 3), dtype=np.float32)
            valid = np.zeros(n, dtype=bool)

        csi = csi_parser.load_csi_trial(csi_path, grid_times=ft, trim_guard=True)
        return row, {
            "csi_iq": csi.iq.astype(np.float16),
            "link_mask": csi.link_mask,
            "pose_rel": pose_rel.astype(np.float32),
            "root": root.astype(np.float32),
            "valid": valid,
            "n_frames": int(n),
            "bad_row_ratio": float(csi.meta["bad_row_ratio"]),
            "any_link_coverage": float(csi.meta["any_link_coverage"]),
            "all_link_coverage": float(csi.meta["all_link_coverage"]),
        }, ""
    except Exception as exc:  # noqa: BLE001
        return row, None, f"{type(exc).__name__}: {exc}"


KEEP_COLS = ["trial_id", "subject", "environment", "task", "class_id", "scenario_id",
             "detail_label", "risk", "risk_id", "n_alive", "time_method",
             "split_group", "role", "n_frames", "bad_row_ratio",
             "any_link_coverage", "all_link_coverage", "cache_ok"]


def reindex(df: pd.DataFrame) -> int:
    """cache_index.csv 만 split 파일 기준으로 다시 쓴다. 배열은 건드리지 않는다.

    role 만 바뀌었을 때(사이트 제외 등) 1.14GB 를 다시 구울 이유가 없다. 다만
    배열은 행 번호로 접근하므로 **trial 순서가 완전히 같을 때만** 안전하다.
    다르면 즉시 실패시킨다 — 조용히 넘어가면 라벨이 어긋난 채로 학습된다.
    """
    old = pd.read_csv(cache.index_path())
    if len(old) != len(df) or (old.trial_id.to_numpy() != df.trial_id.to_numpy()).any():
        print("[build_cache] --reindex 불가: trial 순서/집합이 달라졌다.")
        print(f"  기존 {len(old)} vs 새 {len(df)}")
        diff = set(old.trial_id) ^ set(df.trial_id)
        if diff:
            print(f"  집합 차이 {len(diff)}개 예: {sorted(diff)[:5]}")
        print("  -> python -m notifi_pose.tools.build_cache --rebuild")
        return 1

    merged = df.copy()
    for c in ("n_frames", "bad_row_ratio", "any_link_coverage",
              "all_link_coverage", "cache_ok"):
        merged[c] = old[c].to_numpy()          # 파싱 결과는 그대로 유지
    merged[KEEP_COLS].to_csv(cache.index_path(), index=False, encoding="utf-8")

    changed = int((old.role.to_numpy() != merged.role.to_numpy()).sum())
    print(f"[build_cache] --reindex 완료: {len(merged)} trial, role 변경 {changed}건")
    print(merged[merged.cache_ok].groupby(["split_group", "role"]).size().to_string())
    print(f"\n  wrote {cache.index_path()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(mp.cpu_count(), 8))
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만 (점검용)")
    ap.add_argument("--rebuild", action="store_true", help="기존 캐시를 무시하고 새로 만든다")
    ap.add_argument("--reindex", action="store_true",
                    help="배열은 두고 cache_index.csv 만 갱신 (role 변경용)")
    args = ap.parse_args()

    dev = pd.read_csv(C.SPLIT_DIR / "dev_index.csv").assign(split_group="dev")
    sealed = pd.read_csv(C.SPLIT_DIR / "sealed_index.csv").assign(
        split_group="sealed", role="sealed")
    df = pd.concat([dev, sealed], ignore_index=True)
    if args.limit:
        df = df.head(args.limit)
    n = len(df)

    if args.reindex:
        return reindex(df)

    if cache.meta_path().exists() and not args.rebuild:
        old = cache.read_meta()
        if old.get("preproc_version") == C.PREPROC_VERSION and old.get("n_trials") == n:
            print(f"[build_cache] 이미 최신 캐시가 있다 "
                  f"(version={old['preproc_version']}, n={old['n_trials']}).")
            print("  다시 만들려면 --rebuild")
            return 0

    gb = cache.estimated_bytes(n) / 1e9
    print(f"[build_cache] {n:,} trials  workers={args.workers}")
    print(f"  프레임 {C.CACHE_FRAMES}, 링크 {C.N_LINKS}, subcarrier {C.N_LIVE_SUBCARRIERS}")
    print(f"  예상 용량 {gb:.2f} GB -> {C.CACHE_DIR}")

    arrays = cache.create(n)
    rows = []
    errors = []
    t0 = time.perf_counter()

    jobs = [(i, r.trial_id, r.csi, r.gt_pose, r.time_method, r.task)
            for i, r in enumerate(df.itertuples())]
    with mp.Pool(args.workers) as pool:
        for done, (row, res, err) in enumerate(
                pool.imap_unordered(_one, jobs, chunksize=4), 1):
            if res is None:
                errors.append((df.trial_id.iloc[row], err))
                rows.append({"row": row, "n_frames": 0, "bad_row_ratio": np.nan,
                             "any_link_coverage": 0.0, "all_link_coverage": 0.0,
                             "cache_ok": False})
            else:
                k = res["n_frames"]
                for name in cache.ARRAY_SPEC:
                    arrays[name][row, :k] = res[name]
                    arrays[name][row, k:] = 0
                rows.append({"row": row, "n_frames": k,
                             "bad_row_ratio": res["bad_row_ratio"],
                             "any_link_coverage": res["any_link_coverage"],
                             "all_link_coverage": res["all_link_coverage"],
                             "cache_ok": True})
            if done % 250 == 0 or done == n:
                el = time.perf_counter() - t0
                print(f"  {done}/{n}  ({el:.0f}s, 남은 예상 {el/done*(n-done):.0f}s)")

    for a in arrays.values():
        a.flush()

    stat = pd.DataFrame(rows).set_index("row").sort_index()
    out = df.reset_index(drop=True).join(stat)
    out[KEEP_COLS].to_csv(cache.index_path(), index=False, encoding="utf-8")
    cache.write_meta(n, n_errors=len(errors), source_splits=["dev_index", "sealed_index"])

    el = time.perf_counter() - t0
    print(f"\n[build_cache] 완료 {el:.0f}s ({el/max(n,1)*1000:.0f} ms/trial)")
    print(f"  성공 {int(out.cache_ok.sum()):,} / 실패 {len(errors)}")
    if errors:
        for t, e in errors[:10]:
            print(f"    - {t}: {e}")

    print("\n=== 캐시 요약 ===")
    ok = out[out.cache_ok]
    print(f"  프레임 수: {ok.n_frames.min()}~{ok.n_frames.max()} (중앙 {int(ok.n_frames.median())})")
    print(f"  파손 행 비율: 중앙 {ok.bad_row_ratio.median():.5f}, 최대 {ok.bad_row_ratio.max():.5f}")
    print(f"  링크 커버리지: any 중앙 {ok.any_link_coverage.median():.3f}, "
          f"all 중앙 {ok.all_link_coverage.median():.3f}")
    print(ok.groupby(["split_group", "role"]).size().to_string())
    print(f"\n  wrote {cache.index_path()}")
    print(f"  wrote {cache.meta_path()}")
    for name in cache.ARRAY_SPEC:
        p = cache.array_path(name)
        print(f"  wrote {p.name}  {p.stat().st_size/1e9:.3f} GB")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
