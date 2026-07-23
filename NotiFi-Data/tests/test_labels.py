from collections import Counter

from notifi_collection.labels import (
    LABEL_SPECS,
    cue_for_trial,
    total_repeats,
    variant_for_trial,
)


def test_v2_label_counts() -> None:
    counts = Counter(spec.risk for spec in LABEL_SPECS)
    repeats = Counter()
    for spec in LABEL_SPECS:
        repeats[spec.risk] += spec.repeats_per_environment
    assert counts == {"SAFE": 9, "WARNING": 3, "DANGER": 5}
    assert repeats == {"SAFE": 150, "WARNING": 75, "DANGER": 50}
    assert total_repeats() == 275


def test_cue_cycle_and_static_behavior() -> None:
    walking = next(spec for spec in LABEL_SPECS if spec.label == "walking")
    standing = next(spec for spec in LABEL_SPECS if spec.label == "standing_still")
    assert [cue_for_trial(walking, number) for number in range(1, 7)] == [
        2.5,
        3.0,
        3.5,
        2.5,
        3.0,
        3.5,
    ]
    assert cue_for_trial(standing, 1) is None


def test_variants_are_balanced() -> None:
    danger = next(spec for spec in LABEL_SPECS if spec.label == "fall_from_standing")
    values = [variant_for_trial(danger, number) for number in range(1, 11)]
    assert values.count("fall_left") == 5
    assert values.count("fall_right") == 5

