"""
build_cache.py

manifest_full.csv의 모든 CSI 파일을 미리 파싱해서 .npz 캐시로 저장한다.
다음 학습부터 로딩 시간이 대폭 줄어든다.

실행:
    python build_cache.py               # 3명 전체
    python build_cache.py --subject mhw # 특정 subject만
    python build_cache.py --workers 4   # 병렬 코어 수 지정 (기본: CPU 수)
"""

import argparse
import ast
import csv
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "split_manifest" / "manifest_full.csv"


def parse_csi_array(raw_line):
    start = raw_line.rfind("[")
    end   = raw_line.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        arr = np.array(ast.literal_eval(raw_line[start:end + 1]), dtype=np.float32)
    except Exception:
        return None
    if len(arr) < 2:
        return None
    if len(arr) % 2 == 1:
        arr = arr[:-1]
    real = arr[0::2]
    imag = arr[1::2]
    return np.sqrt(real * real + imag * imag)


def build_v2_one(csi_path_str):
    """_v2.npz(148d = 균일재샘플 amp132 + 변동스펙트럼16) 생성. 기존 캐시는 그대로.
    로직은 train_csi_to_pose의 parse_csi_csv/build_v2_features를 직접 호출 (복제 없음)."""
    import train_csi_to_pose as T
    csi_path = Path(csi_path_str)
    v2_path = csi_path.with_name(csi_path.stem + "_v2.npz")
    if v2_path.exists():
        return "skip", v2_path.name
    try:
        raw_t, raw_amp = T.parse_csi_csv(csi_path)
        times2, feats2 = T.build_v2_features(raw_t, raw_amp)
        np.savez(v2_path, times=times2, features=feats2)
        return "done", v2_path.name
    except Exception as e:
        return "error", f"{csi_path.name}: {e}"


def build_v3_one(csi_path_str):
    """_v3.npz(404d = v2 148 + 새니타이즈 위상 cos/sin 256) 생성. 기존 캐시는 그대로.
    로직은 train_csi_to_pose의 parse_csi_csv_iq/build_v3_features를 직접 호출 (복제 없음)."""
    import train_csi_to_pose as T
    csi_path = Path(csi_path_str)
    v3_path = csi_path.with_name(csi_path.stem + "_v3.npz")
    if v3_path.exists():
        return "skip", v3_path.name
    try:
        raw_t, raw_amp, raw_cos, raw_sin = T.parse_csi_csv_iq(csi_path)
        times3, feats3 = T.build_v3_features(raw_t, raw_amp, raw_cos, raw_sin)
        np.savez(v3_path, times=times3, features=feats3)
        return "done", v3_path.name
    except Exception as e:
        return "error", f"{csi_path.name}: {e}"


def build_one(csi_path_str):
    csi_path   = Path(csi_path_str)
    cache_path = csi_path.with_suffix(".npz")

    if cache_path.exists():
        return "skip", csi_path.name

    try:
        df    = pd.read_csv(csi_path)
        rel_t = (pd.to_numeric(df["pc_time_ms"], errors="coerce")
                 - df["pc_time_ms"].iloc[0]) / 1000.0

        features, times = [], []
        for _, row in df.iterrows():
            amp = parse_csi_array(row["raw_line"])
            if amp is None:
                continue
            amp = amp.astype(np.float32)
            amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
            # per-packet 스케일 정규화: AGC/Tx power/거리 게인(전체 배율)을 제거하고
            # subcarrier 프로파일의 "모양"(방향)만 남긴다. → cross-domain 전이용.
            # train_csi_to_pose.load_csi_features 와 반드시 동일하게 유지할 것.
            amp = amp / (np.linalg.norm(amp) + 1e-6)
            feat = np.concatenate([
                amp,
                np.array([np.nanmean(amp), np.nanstd(amp),
                          np.nanmax(amp),  np.nanmin(amp)],
                         dtype=np.float32),
            ])
            features.append(feat)
            times.append(float(rel_t.loc[row.name]))

        np.savez(cache_path,
                 times=np.asarray(times,    dtype=np.float32),
                 features=np.asarray(features, dtype=np.float32))
        return "done", csi_path.name

    except Exception as e:
        return "error", f"{csi_path.name}: {e}"


def collect_csi_paths(subject_filter=None):
    paths = []
    seen  = set()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["valid"] != "True":
                continue
            if row["csi_exists"] != "True":
                continue
            if subject_filter and row["subject"] not in subject_filter:
                continue
            p = str(ROOT / row["csi_path"])
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", nargs="+", default=None,
                        help="캐시 생성할 subject (기본: 전체). 예: --subject mhw lmh")
    parser.add_argument("--workers", type=int,
                        default=min(mp.cpu_count(), 8),
                        help="병렬 코어 수 (기본: CPU 수, 최대 8)")
    parser.add_argument("--v2", action="store_true",
                        help="변동 스펙트럼 포함 _v2.npz(148d) 생성 (기존 캐시는 그대로 두고 추가)")
    parser.add_argument("--v3", action="store_true",
                        help="위상 포함 _v3.npz(404d = v2 148 + cos/sin 256) 생성 (기존 캐시 무손상)")
    args = parser.parse_args()

    paths = collect_csi_paths(set(args.subject) if args.subject else None)
    total = len(paths)
    worker_fn = build_v3_one if args.v3 else (build_v2_one if args.v2 else build_one)
    print(f"CSI 파일 {total}개 발견 (workers={args.workers}, v2={args.v2}, v3={args.v3})")

    already = sum(1 for p in paths if Path(p).with_suffix(".npz").exists())
    print(f"  이미 캐시 있음: {already}개 → 새로 생성: {total - already}개\n")

    done = skip = error = 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (status, name) in enumerate(pool.imap_unordered(worker_fn, paths), 1):
            if status == "done":
                done += 1
                tag = "✓"
            elif status == "skip":
                skip += 1
                tag = "-"
            else:
                error += 1
                tag = "✗"
            print(f"[{i:3d}/{total}] {tag} {name}")

    print(f"\n완료: 생성 {done}개 | 스킵 {skip}개 | 에러 {error}개")

    # 용량 계산
    npz_sizes = []
    for p in paths:
        np_path = Path(p).with_suffix(".npz")
        if np_path.exists():
            npz_sizes.append(np_path.stat().st_size)
    if npz_sizes:
        total_mb = sum(npz_sizes) / 1024 / 1024
        avg_kb   = (sum(npz_sizes) / len(npz_sizes)) / 1024
        print(f"캐시 총 용량: {total_mb:.1f} MB  (평균 {avg_kb:.0f} KB/파일)")


if __name__ == "__main__":
    main()
