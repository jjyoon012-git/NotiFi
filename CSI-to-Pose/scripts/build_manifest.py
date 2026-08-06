"""
build_manifest.py

모든 subject/label/trial을 스캔해서 manifest_full.csv를 생성한다.

ambient 블록별 고정 분할:
  quiet  t001-t007 : t001-t005=train, t006=val, t007=test
  aircon t008-t013 : t008-t011=train, t012=val, t013=test
  tv     t014-t019 : t014-t017=train, t018=val, t019=test
  music  t020-t025 : t020-t023=train, t024=val, t025=test

LOSO는 manifest_full.csv에서 subject 컬럼으로 필터해서 사용.
  예) test_subject=ajh → valid=True 행 중 subject==ajh → test, 나머지 → train/val(split_mode1 그대로)

실행:
  python build_manifest.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent          # NotiFi-CSI-to-Pose/
OUT_DIR = ROOT / "split_manifest"
OUT_DIR.mkdir(exist_ok=True)

# ── subject 설정 ───────────────────────────────────────────────
SUBJECTS = {
    "ajh": "AJH",
    "lmh": "lmh",
    "mhw": "mhw",
}

# ── label → CSI 하위 경로 매핑 ────────────────────────────────
LABEL_CSI_PATH = {
    "standing_still":     "safe/posture/standing_still",
    "sitting_still":      "safe/posture/sitting_still",
    "lying_still":        "safe/posture/lying_still",
    "walking":            "safe/motion/walking",
    "sit_to_stand":       "safe/transition/sit_to_stand",
    "stand_to_sit":       "safe/transition/stand_to_sit",
    "lie_to_stand":       "safe/transition/lie_to_stand",
    "stand_to_lie_normal":"safe/transition/stand_to_lie_normal",
    "bed_exit_failed":    "warning/bed_exit/bed_exit_failed",
    "unstable_walking":   "warning/gait/unstable_walking",
    "post_fall_inactive": "danger/fall/post_fall_inactive",
}

# ── ambient 블록 정의 ─────────────────────────────────────────
AMBIENT_BLOCKS = [
    ("quiet",  list(range(1,  8))),   # t001-t007
    ("aircon", list(range(8,  14))),  # t008-t013
    ("tv",     list(range(14, 20))),  # t014-t019
    ("music",  list(range(20, 26))),  # t020-t025
]

# ── ambient 블록 내 split 규칙 ────────────────────────────────
# quiet(7개): 5 train / 1 val / 1 test
# 나머지(6개): 4 train / 1 val / 1 test
def get_split_within_block(nums: list[int]) -> dict[int, str]:
    n = len(nums)
    result = {}
    for i, num in enumerate(nums):
        if i < n - 2:
            result[num] = "train"
        elif i == n - 2:
            result[num] = "val"
        else:
            result[num] = "test"
    return result

TRIAL_SPLIT: dict[int, str] = {}
TRIAL_AMBIENT: dict[int, str] = {}
for ambient, nums in AMBIENT_BLOCKS:
    TRIAL_SPLIT.update(get_split_within_block(nums))
    for n in nums:
        TRIAL_AMBIENT[n] = ambient

# ── 현재 무효 데이터 목록 ─────────────────────────────────────
# ajh post_fall_inactive 재수집 완료 (2026-07-02) → 비워둠
INVALID_SET: set[tuple[str, str]] = {}


def build_manifest(pose_dir="pose_gt", out_name="manifest_full.csv"):
    """
    pose_dir : pose GT 폴더명. 기본 "pose_gt"(image 좌표).
               "pose_gt_world" 로 주면 world 좌표 GT를 가리키는 manifest 생성.
    out_name : 출력 파일명. world용은 "manifest_full_world.csv" 권장.
    CSI 입력 경로/split/valid 등은 pose_dir과 무관하게 동일하게 채워진다.
    """
    rows = []

    for subject, prefix in SUBJECTS.items():
        subj_root = ROOT / f"csi_to_pose_{subject}"

        for label, csi_subpath in LABEL_CSI_PATH.items():
            csi_label_dir  = subj_root / "csi" / csi_subpath / prefix
            pg13_label_dir = subj_root / pose_dir / "proxy_13" / label / prefix
            pgf_label_dir  = subj_root / pose_dir / "full_33_landmarks" / label / prefix

            is_invalid = (subject, label) in INVALID_SET

            for trial_num in range(1, 26):
                trial = f"t{trial_num:03d}"
                ambient  = TRIAL_AMBIENT[trial_num]
                split_m1 = TRIAL_SPLIT[trial_num]
                valid    = not is_invalid

                csi_file  = csi_label_dir  / f"{prefix}_{label}_{trial}.csv"
                pg13_file = pg13_label_dir / f"{prefix}_{label}_{trial}_proxy13.csv"
                pgf_file  = pgf_label_dir  / f"{prefix}_{label}_{trial}_full33.csv"

                rows.append({
                    "subject":       subject,
                    "prefix":        prefix,
                    "label":         label,
                    "trial":         trial,
                    "trial_num":     trial_num,
                    "ambient":       ambient,
                    "split_mode1":   split_m1,
                    "valid":         valid,
                    "csi_exists":    csi_file.exists(),
                    "proxy13_exists": pg13_file.exists(),
                    "full33_exists": pgf_file.exists(),
                    "csi_path":      csi_file.relative_to(ROOT).as_posix(),
                    "proxy13_path":  pg13_file.relative_to(ROOT).as_posix(),
                    "full33_path":   pgf_file.relative_to(ROOT).as_posix(),
                })

    out_path = OUT_DIR / out_name
    fieldnames = [
        "subject", "prefix", "label", "trial", "trial_num",
        "ambient", "split_mode1", "valid",
        "csi_exists", "proxy13_exists", "full33_exists",
        "csi_path", "proxy13_path", "full33_path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest 저장: {out_path}  ({len(rows)}행)")
    _print_summary(rows)


def _print_summary(rows):
    valid_rows = [r for r in rows if r["valid"]]
    invalid_rows = [r for r in rows if not r["valid"]]

    print(f"\n[전체]  총={len(rows)}  valid={len(valid_rows)}  invalid={len(invalid_rows)}")

    print("\n[Mode1 split - valid 행 기준]")
    for split in ("train", "val", "test"):
        cnt = sum(1 for r in valid_rows if r["split_mode1"] == split)
        print(f"  {split:5s}: {cnt}행")

    print("\n[ambient 분포 - valid 행]")
    for ambient, _ in AMBIENT_BLOCKS:
        cnt = sum(1 for r in valid_rows if r["ambient"] == ambient)
        print(f"  {ambient:6s}: {cnt}행")

    print("\n[invalid 목록]")
    seen = set()
    for r in invalid_rows:
        key = (r["subject"], r["label"])
        if key not in seen:
            seen.add(key)
            n = sum(1 for x in invalid_rows if x["subject"]==r["subject"] and x["label"]==r["label"])
            print(f"  {r['subject']} / {r['label']} → {n}개 trial (valid=False)")

    print("\n[LOSO 구성 가이드]")
    subjects = list(SUBJECTS.keys())
    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]
        train_cnt = sum(1 for r in valid_rows if r["subject"] in train_subjs)
        test_cnt  = sum(1 for r in valid_rows if r["subject"] == test_subj)
        print(f"  test={test_subj}: train({'+'.join(train_subjs)})={train_cnt}  test={test_cnt}")

    missing = [r for r in valid_rows if not r["csi_exists"] or not r["proxy13_exists"]]
    if missing:
        print(f"\n[경고] 파일 누락 {len(missing)}건 (csi 또는 proxy13 없음)")
        for r in missing[:10]:
            print(f"  {r['csi_path']}")
    else:
        print("\n[OK] valid 행 전체 파일 존재 확인")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_dir", default="pose_gt",
                    help="pose GT 폴더명 (world 학습: pose_gt_world)")
    ap.add_argument("--out", default="manifest_full.csv",
                    help="출력 파일명 (world용: manifest_full_world.csv)")
    args = ap.parse_args()
    build_manifest(pose_dir=args.pose_dir, out_name=args.out)
