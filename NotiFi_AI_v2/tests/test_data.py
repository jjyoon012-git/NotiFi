import unittest

from notifi_ai_v2.data import (
    CacheRecord,
    PROMPT_CLASSES,
    nested_source_split,
    reserve_support,
)


def record(trial_id, subject, environment, action_id=0):
    return CacheRecord(
        row=0,
        trial_id=trial_id,
        subject=subject,
        environment=environment,
        task="pose_and_action",
        action_id=action_id,
        risk_id=0,
        role="train",
        cache_ok=True,
        time_method="timestamps",
    )


class DataProtocolTests(unittest.TestCase):
    def test_support_reservation_is_disjoint_and_reproducible(self):
        rows = []
        for action_id in PROMPT_CLASSES:
            for shot in range(3):
                rows.append(record(
                    f"ajh_E01_{action_id}_{shot}", "ajh", "E01", action_id
                ))
        first_support, first_query = reserve_support(rows, seed=17017)
        second_support, second_query = reserve_support(rows, seed=17017)
        self.assertEqual(
            [row.trial_id for row in first_support],
            [row.trial_id for row in second_support],
        )
        self.assertEqual(len(first_support), 2 * len(PROMPT_CLASSES))
        self.assertEqual(len(first_query), len(PROMPT_CLASSES))
        self.assertFalse(
            {row.trial_id for row in first_support}
            & {row.trial_id for row in second_query}
        )

    def test_nested_split_never_trains_on_held_subject(self):
        rows = [
            record("a1", "ajh", "E01"),
            record("a2", "ajh", "E02"),
            record("a3", "ajh", "E03"),
            record("m1", "mhw", "E01"),
            record("m2", "mhw", "E02"),
            record("m3", "mhw", "E03"),
            record("l1", "lmh", "E01"),
        ]
        train, validation, outer = nested_source_split(rows, "ajh")
        self.assertEqual(outer, ["ajh_E01", "ajh_E02", "ajh_E03"])
        self.assertTrue(all(not site.startswith("ajh_") for site in train))
        self.assertTrue(all(not site.startswith("ajh_") for site in validation))


if __name__ == "__main__":
    unittest.main()
