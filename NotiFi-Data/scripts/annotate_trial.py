from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.files import write_trial_checksums


def update_manifest(manifest_path: Path, source_uid: str, values: dict[str, object]) -> None:
    if not manifest_path.exists():
        return
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"Manifest has no header: {manifest_path}")
    found = False
    for row in rows:
        if row.get("source_trial_uid") == source_uid:
            for key, value in values.items():
                if key in row:
                    row[key] = value
            found = True
    if not found:
        raise ValueError(f"Trial not found in manifest: {source_uid}")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add phase timestamps and manual QC to a trial.")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--actual-onset", type=float)
    parser.add_argument("--action-end", type=float)
    parser.add_argument("--impact", type=float)
    parser.add_argument("--manual-qc", choices=("ACCEPT", "REVIEW", "REJECT"), required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    data = json.loads(args.metadata.read_text(encoding="utf-8"))
    duration = float(data["duration_s"])
    is_static = data.get("planned_cue_s") is None
    if is_static:
        actual_onset = 0.0 if args.actual_onset is None else args.actual_onset
        action_end = duration if args.action_end is None else args.action_end
    else:
        if args.actual_onset is None or args.action_end is None:
            raise ValueError("Dynamic trials require --actual-onset and --action-end")
        actual_onset = args.actual_onset
        action_end = args.action_end
    if data.get("risk_label") == "DANGER" and args.impact is None:
        raise ValueError("DANGER trials require --impact")
    timestamps = [actual_onset, action_end]
    if args.impact is not None:
        timestamps.append(args.impact)
    if any(value < 0 or value > duration for value in timestamps):
        raise ValueError(f"All timestamps must be within 0-{duration}s")
    if actual_onset > action_end:
        raise ValueError("actual onset cannot be after action end")
    if args.impact is not None and not actual_onset <= args.impact <= action_end:
        raise ValueError("impact must be between actual onset and action end")

    data.update(
        {
            "actual_onset_s": actual_onset,
            "impact_s": args.impact,
            "action_end_s": action_end,
            "annotation_status": "COMPLETE",
            "manual_qc": args.manual_qc,
            "manual_qc_note": args.note,
        }
    )
    args.metadata.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = args.manifest
    if manifest_path is None:
        candidate = args.metadata
        while candidate.parent != candidate:
            possible = candidate / "manifests" / "trials.csv"
            if possible.exists():
                manifest_path = possible
                break
            candidate = candidate.parent
    if manifest_path is not None:
        update_manifest(
            manifest_path,
            data["source_trial_uid"],
            {
                "actual_onset_s": actual_onset,
                "impact_s": "" if args.impact is None else args.impact,
                "action_end_s": action_end,
                "manual_qc": args.manual_qc,
            },
        )
    write_trial_checksums(args.metadata.parent)
    print(f"[OK] annotated: {data['source_trial_uid']} -> {args.manual_qc}")


if __name__ == "__main__":
    main()
