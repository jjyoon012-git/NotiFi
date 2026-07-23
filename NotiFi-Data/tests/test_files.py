from notifi_collection.files import verify_trial_checksums, write_trial_checksums


def test_checksum_round_trip(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    write_trial_checksums(tmp_path)
    assert verify_trial_checksums(tmp_path) == []
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    assert verify_trial_checksums(tmp_path) == ["checksum_mismatch:a.txt"]
