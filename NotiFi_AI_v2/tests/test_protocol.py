import unittest

from notifi_ai_v2.protocol import (
    ProtocolError,
    SplitRole,
    TrialRecord,
    assert_selection_is_source_only,
    validate_protocol,
)


class ProtocolTests(unittest.TestCase):
    def test_source_and_sealed_roles_are_valid_when_disjoint(self):
        rows = [
            TrialRecord("ajh_train", "ajh", "E01", SplitRole.TRAIN),
            TrialRecord("yja_support", "yja", "E02", SplitRole.SEALED_SUPPORT),
            TrialRecord("yja_query", "yja", "E02", SplitRole.SEALED_QUERY),
        ]
        self.assertEqual(validate_protocol(rows), rows)

    def test_yja_cannot_enter_training_or_validation(self):
        with self.assertRaisesRegex(ProtocolError, "cannot enter source development"):
            validate_protocol([
                TrialRecord("leak", "yja", "E02", SplitRole.VALIDATION)
            ])

    def test_lmh_bad_gt_sites_are_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "excluded GT site"):
            validate_protocol([
                TrialRecord("bad_gt", "lmh", "E03", SplitRole.TRAIN)
            ])

    def test_sealed_metrics_cannot_select_a_model(self):
        with self.assertRaisesRegex(ProtocolError, "cannot select a model"):
            assert_selection_is_source_only([
                TrialRecord("support", "yja", "E02", SplitRole.SEALED_SUPPORT)
            ])


if __name__ == "__main__":
    unittest.main()
