import json
from pathlib import Path


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


def test_team_guides_exist_and_have_all_commands() -> None:
    for subject in ("ajh", "lmh", "mhw", "yja"):
        text = (ROOT / f"{subject}.md").read_text(encoding="utf-8")
        assert f"--subject {subject}" in text
        assert text.count("python scripts/collect_dataset.py --label") == 17
        assert "환경별 목표: `275`" in text
        assert "개인 전체 목표: `825`" in text
