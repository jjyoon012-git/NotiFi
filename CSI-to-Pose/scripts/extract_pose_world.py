"""
extract_pose_world.py  (standalone, 기존 데이터 무손상)

기존 pose_gt/(image 좌표)는 전혀 건드리지 않고, MediaPipe pose_world_landmarks(미터·골반원점)로
proxy13 / full33 GT를 **별도 폴더** pose_gt_world/ 에 새로 만든다.

  입력 : csi_to_pose_{subject}/videos/**/{PREFIX}_{label}_{trial}.avi|.mp4  (+ *_frame_timestamps.csv)
  출력 : csi_to_pose_{subject}/pose_gt_world/proxy_13/{label}/{PREFIX}/{PREFIX}_{label}_{trial}_proxy13.csv
         csi_to_pose_{subject}/pose_gt_world/full_33_landmarks/{label}/{PREFIX}/..._full33.csv

- 컬럼 포맷은 기존 extract_pose_features.py의 헬퍼를 그대로 재사용 → 학습 코드가 그대로 읽을 수 있음
- 오버레이(mp4) 생성 안 함  (용량·시간 절약)
- 이미 만들어진 출력은 건너뜀(resumable) → --overwrite 로 강제 재생성

실행:
    python scripts/extract_pose_world.py                 # 전체 (mhw, lmh, ajh)
    python scripts/extract_pose_world.py --subjects mhw lmh
    python scripts/extract_pose_world.py --limit 2       # 시간 측정용: 앞 2개만
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import extract_pose_features as E   # build_proxy / build_derived / flatten_landmarks / MP_INDEX / landmark_to_dict

SUBJECTS = {"mhw": "mhw", "lmh": "lmh", "ajh": "AJH"}   # 폴더키 → 파일 prefix(=subject 컬럼)
EXCLUDE = set()                                          # 제외 없음 (ajh post_fall 영상 복구됨)


def discover(subject_key, prefix):
    vroot = ROOT / f"csi_to_pose_{subject_key}" / "videos"
    vids = []
    for ext in ("*.avi", "*.mp4"):
        vids += list(vroot.rglob(ext))
    out = []
    for v in vids:
        if "_overlay" in v.stem:
            continue
        parts = v.stem.split("_")
        if len(parts) < 3:
            continue
        subj, trial, label = parts[0], parts[-1], "_".join(parts[1:-1])
        if subj != prefix:
            continue
        if (subject_key, label) in EXCLUDE:
            continue
        out.append((v, prefix, label, trial))
    out.sort(key=lambda x: (x[2], x[3]))
    return out


def extract_one(video_path, prefix, label, trial, subject_key, overwrite=False):
    base = f"{prefix}_{label}_{trial}"
    out_root = ROOT / f"csi_to_pose_{subject_key}" / "pose_gt_world"
    full_out = out_root / "full_33_landmarks" / label / prefix / f"{base}_full33.csv"
    proxy_out = out_root / "proxy_13" / label / prefix / f"{base}_proxy13.csv"
    if proxy_out.exists() and full_out.exists() and not overwrite:
        return "skip"
    proxy_out.parent.mkdir(parents=True, exist_ok=True)
    full_out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "fail(open)"
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    frame_ts = E.load_frame_timestamps(video_path)

    full_rows, proxy_rows = [], []
    prev_proxy = None
    prev_t = None
    with mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t_sec = float(frame_ts[fi]) if (frame_ts is not None and fi < len(frame_ts)) else fi / fps
            meta = {"frame_idx": fi, "timestamp_sec": t_sec, "subject": prefix,
                    "label": label, "trial": trial}
            # ★ 유일한 차이: pose_landmarks → pose_world_landmarks (미터·골반원점)
            if res.pose_world_landmarks:
                full = {name: E.landmark_to_dict(res.pose_world_landmarks.landmark[idx])
                        for name, idx in E.MP_INDEX.items()}
                proxy = E.build_proxy(full)
                sr = (1.0 / (t_sec - prev_t)) if (prev_t is not None and t_sec - prev_t > 1e-6) else fps
                derived = E.build_derived(proxy, prev_proxy, sr)
                prev_proxy, prev_t = proxy, t_sec
                fr = dict(meta, pose_detected=1); fr.update(E.flatten_landmarks("", full))
                pr = dict(meta, pose_detected=1); pr.update(E.flatten_landmarks("", proxy)); pr.update(derived)
                full_rows.append(fr); proxy_rows.append(pr)
            else:
                full_rows.append(dict(meta, pose_detected=0))
                proxy_rows.append(dict(meta, pose_detected=0))
            fi += 1
    cap.release()
    pd.DataFrame(full_rows).to_csv(full_out, index=False)
    pd.DataFrame(proxy_rows).to_csv(proxy_out, index=False)
    det = sum(r.get("pose_detected", 0) for r in proxy_rows)
    return f"ok {det}/{len(proxy_rows)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS.keys()))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="시간 측정용: 전체에서 앞 N개만 처리")
    args = ap.parse_args()

    tasks = []
    for sk in args.subjects:
        tasks += [(sk, *t) for t in discover(sk, SUBJECTS[sk])]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[extract_pose_world] 대상 {len(tasks)}개  subjects={args.subjects}", flush=True)

    t0 = time.time()
    done = skip = fail = 0
    for i, (sk, v, prefix, label, trial) in enumerate(tasks, 1):
        r = extract_one(v, prefix, label, trial, sk, overwrite=args.overwrite)
        if r.startswith("ok"):
            done += 1
        elif r == "skip":
            skip += 1
        else:
            fail += 1
        el = time.time() - t0
        rate = el / max(done + fail, 1)
        eta = rate * (len(tasks) - i) / 60.0
        print(f"[{i}/{len(tasks)}] {sk}/{label}/{trial}  {r}  "
              f"({el:.0f}s 경과, ~{eta:.0f}분 남음)", flush=True)
    print(f"\n[DONE] ok={done} skip={skip} fail={fail}  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
