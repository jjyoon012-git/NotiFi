import argparse
import sys
from pathlib import Path

from extract_pose_features import extract

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "csi_to_pose"
VIDEO_ROOT = EXPERIMENT_ROOT / "videos"
POSE_ROOT = EXPERIMENT_ROOT / "pose_gt"


def find_videos(subject=None, label=None):
    videos = []
    for mp4 in list(VIDEO_ROOT.glob("**/*.mp4")) + list(VIDEO_ROOT.glob("**/*.avi")):
        if "_overlay" in mp4.name:
            continue
        parts = mp4.stem.split("_")
        if len(parts) < 3:
            continue
        # {subject}_{label...}_{trial} — trial은 마지막, subject는 첫번째
        trial = parts[-1]
        subj = parts[0]
        lbl = "_".join(parts[1:-1])

        if subject and subj != subject:
            continue
        if label and lbl != label:
            continue

        videos.append((mp4, subj, lbl, trial))

    videos.sort(key=lambda x: (x[2], x[1], x[3]))
    return videos


def already_done(subject, label, trial):
    proxy = POSE_ROOT / "proxy_13" / label / subject / f"{subject}_{label}_{trial}_proxy13.csv"
    return proxy.exists()


def main():
    parser = argparse.ArgumentParser(
        description="영상 폴더 전체를 순서대로 pose 추출합니다."
    )
    parser.add_argument("--subject", default=None, help="특정 subject만 처리 (기본: 전체)")
    parser.add_argument("--label", default=None, help="특정 label만 처리 (기본: 전체)")
    parser.add_argument("--no_overlay", action="store_true", help="overlay 영상 생성 생략")
    parser.add_argument("--overwrite", action="store_true", help="이미 처리된 trial도 재처리")
    args = parser.parse_args()

    videos = find_videos(subject=args.subject, label=args.label)

    if not videos:
        print("[WARN] 처리할 영상이 없습니다. subject/label 확인하세요.")
        sys.exit(0)

    print(f"[INFO] 처리 대상: {len(videos)}개\n")

    success, skipped, failed = 0, 0, 0

    for idx, (mp4, subj, lbl, trial) in enumerate(videos, 1):
        tag = f"{subj}_{lbl}_{trial}"
        print(f"[{idx}/{len(videos)}] {tag}")

        if not args.overwrite and already_done(subj, lbl, trial):
            print(f"  → 이미 처리됨, 건너뜀 (--overwrite로 재처리 가능)")
            skipped += 1
            continue

        try:
            extract(
                video_path=mp4,
                label=lbl,
                subject=subj,
                trial=trial,
                make_overlay=not args.no_overlay,
            )
            print(f"  → 완료")
            success += 1
        except Exception as e:
            print(f"  → 실패: {e}")
            failed += 1

    print(f"\n[DONE] 완료={success} / 건너뜀={skipped} / 실패={failed}")


if __name__ == "__main__":
    main()
