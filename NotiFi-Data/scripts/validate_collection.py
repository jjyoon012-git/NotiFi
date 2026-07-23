from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.labels import LABEL_SPECS
from notifi_collection.files import verify_trial_checksums


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate collected NotiFi v2.0 trials.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "collection_data" / "v2",
    )
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    meta_files = sorted(args.root.rglob("*_meta.json"))
    problems: list[dict[str, str]] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for meta_path in meta_files:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append({"trial": str(meta_path), "problem": f"invalid_json:{exc}"})
            continue
        uid = data.get("source_trial_uid", meta_path.stem)
        files = data.get("files", {})
        for key in ("csi", "video", "video_timestamps"):
            path = meta_path.parent / str(files.get(key, ""))
            if not path.is_file() or path.stat().st_size == 0:
                problems.append({"trial": uid, "problem": f"missing_or_empty:{key}"})
        for checksum_problem in verify_trial_checksums(meta_path.parent):
            problems.append({"trial": uid, "problem": checksum_problem})
        qc = data.get("automatic_qc", {})
        link_counts = qc.get("link_counts", {})
        for tx_id in ("TX1", "TX2", "TX3"):
            if int(link_counts.get(tx_id, 0)) == 0:
                problems.append({"trial": uid, "problem": f"zero_frames:{tx_id}"})
        if data.get("valid_for_reconstruction"):
            pose_qc = meta_path.parent / f"{uid}_pose_qc.json"
            if pose_qc.exists():
                pose = json.loads(pose_qc.read_text(encoding="utf-8"))
                if float(pose.get("pose_valid_ratio", 0)) < 0.95:
                    problems.append({"trial": uid, "problem": "pose_valid_ratio_below_0.95"})
            else:
                problems.append({"trial": uid, "problem": "pose_qc_pending"})
        counts[
            (
                str(data.get("subject_id")),
                str(data.get("environment_id")),
                str(data.get("scenario_label")),
            )
        ] += 1

    report = {
        "root": str(args.root.resolve()),
        "trial_count": len(meta_files),
        "problem_count": len(problems),
        "problems": problems,
    }
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"[VALIDATE] trials={len(meta_files)} problems={len(problems)}")
    for item in problems[:50]:
        print(f"[ISSUE] {item['trial']}: {item['problem']}")
    if len(problems) > 50:
        print(f"... and {len(problems) - 50} more")

    target_map = {spec.label: spec.repeats_per_environment for spec in LABEL_SPECS}
    for (subject, environment, label), count in sorted(counts.items()):
        target = target_map.get(label, 0)
        if target and count > target:
            print(f"[WARN] excess trials: {subject}/{environment}/{label}={count}>{target}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
