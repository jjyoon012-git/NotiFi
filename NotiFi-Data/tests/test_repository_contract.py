import json
from pathlib import Path

from scripts.collect_dataset import (
    available_trial_numbers,
    usable_manifest_rows,
)


ROOT = Path(__file__).resolve().parent.parent


def test_exported_config_matches_collection_plan() -> None:
    labels = json.loads((ROOT / "config" / "labels_v2.json").read_text(encoding="utf-8"))
    assert len(labels) == 17
    assert sum(item["repeats_per_environment"] for item in labels) == 275
    assert {
        item["label"] for item in labels
    } == {
        "walking",
        "standing_still",
        "sitting_still",
        "lying_still",
        "lie_to_stand",
        "stand_to_lie_normal",
        "absence",
        "sit_to_stand",
        "stand_to_sit",
        "unstable_walking",
        "stumble_recover",
        "bed_exit_failed",
        "fall_from_standing",
        "fall_while_walking",
        "fall_collapse",
        "bed_fall",
        "chair_fall",
    }


def test_collection_cli_has_no_legacy_ambient_switch() -> None:
    source = (ROOT / "scripts" / "collect_dataset.py").read_text(encoding="utf-8")
    assert "--ambient" not in source
    assert "LABEL_MAP" not in source
    assert "csi_plot_path" in source
    assert "save_csi_visualization" in source
    assert "play_error_alarm" in source
    assert "ensure_no_mobile_camera" in source
    assert "open_laptop_camera" in source


def test_team_guides_exist_and_have_all_commands() -> None:
    for subject in ("ajh", "lmh", "mhw", "yja"):
        text = (ROOT / f"{subject}.md").read_text(encoding="utf-8")
        assert f"--subject {subject}" in text
        assert text.count("python scripts/collect_dataset.py --label") == 17
        assert "python scripts/check_camera_source.py" in text
        assert "환경별 목표: `275`" in text
        assert "개인 전체 목표: `825`" in text


def test_camera_guard_is_present() -> None:
    guard = (ROOT / "notifi_collection" / "camera_guard.py").read_text(encoding="utf-8")
    preview = (ROOT / "scripts" / "preview_camera.py").read_text(encoding="utf-8")
    check_script = (ROOT / "scripts" / "check_camera_source.py").read_text(encoding="utf-8")

    assert "continuity camera" in guard.lower()
    assert "iphone" in guard.lower()
    assert "ensure_no_mobile_camera" in preview
    assert "ensure_no_mobile_camera" in check_script


def test_deleted_trial_artifacts_do_not_block_trial_number_reuse(tmp_path) -> None:
    existing_csi = tmp_path / "trial3_csi.csv"
    existing_video = tmp_path / "trial3_video.mp4"
    existing_meta = tmp_path / "trial3_meta.json"
    for path in (existing_csi, existing_video, existing_meta):
        path.write_text("ok", encoding="utf-8")

    rows = [
        {
            "trial_number": "1",
            "csi_path": "deleted_t001_csi.csv",
            "video_path": "deleted_t001_video.mp4",
            "metadata_path": "deleted_t001_meta.json",
        },
        {
            "trial_number": "3",
            "csi_path": existing_csi.name,
            "video_path": existing_video.name,
            "metadata_path": existing_meta.name,
        },
    ]

    usable_rows = usable_manifest_rows(tmp_path, rows)
    assert [row["trial_number"] for row in usable_rows] == ["3"]
    assert available_trial_numbers([3], repeat=2, target=4, allow_extra=False) == [1, 2]
