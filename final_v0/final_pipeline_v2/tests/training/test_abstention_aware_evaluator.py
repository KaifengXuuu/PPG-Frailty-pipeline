"""Participant-level abstention-aware metric contracts."""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.training import (
    evaluate_predictions,
    evaluate_predictions_with_abstentions,
)


class AbstentionAwareEvaluatorTests(unittest.TestCase):
    def test_no_abstention_matches_existing_conditional_metrics(self) -> None:
        labels = np.asarray([0, 1, 2, 0])
        probabilities = np.asarray(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.10, 0.70, 0.20],
                [0.70, 0.20, 0.10],
            ]
        )
        expected = evaluate_predictions(labels, probabilities, class_order=(0, 1, 2))
        result = evaluate_predictions_with_abstentions(
            labels,
            probabilities,
            np.asarray([], dtype=np.int64),
            class_order=(0, 1, 2),
        )

        self.assertEqual(result.conditional_metrics, expected)
        self.assertAlmostEqual(result.balanced_accuracy, expected.balanced_accuracy)
        self.assertAlmostEqual(result.macro_f1, expected.macro_f1)
        self.assertEqual(result.abstention_counts_by_class, ((0, 0), (1, 0), (2, 0)))
        self.assertEqual((result.n_total, result.n_retained, result.n_abstained), (4, 4, 0))
        self.assertEqual(result.probability_metrics_scope, "retained_only")
        self.assertEqual(
            result.retained_multiclass_log_loss,
            expected.multiclass_log_loss,
        )
        self.assertEqual(result.retained_multiclass_brier, expected.multiclass_brier)
        self.assertEqual(
            result.retained_expected_calibration_error,
            expected.expected_calibration_error,
        )

    def test_abstaining_every_participant_of_one_class_adds_false_negatives(self) -> None:
        result = evaluate_predictions_with_abstentions(
            np.asarray([0, 1]),
            np.asarray([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]),
            np.asarray([2, 2]),
            class_order=(0, 1, 2),
        )

        self.assertIsNotNone(result.conditional_metrics)
        assert result.conditional_metrics is not None
        self.assertEqual(result.conditional_metrics.balanced_accuracy, 1.0)
        self.assertAlmostEqual(result.balanced_accuracy, 2.0 / 3.0)
        self.assertAlmostEqual(result.macro_precision, 2.0 / 3.0)
        self.assertAlmostEqual(result.macro_recall, 2.0 / 3.0)
        self.assertAlmostEqual(result.macro_f1, 2.0 / 3.0)
        class_two = result.per_class[2]
        self.assertEqual(class_two.support, 2)
        self.assertEqual(class_two.retained_support, 0)
        self.assertEqual(class_two.abstention_count, 2)
        self.assertEqual(class_two.false_negative, 2)
        self.assertEqual(class_two.false_positive, 0)
        self.assertEqual((class_two.precision, class_two.recall, class_two.f1), (0.0, 0.0, 0.0))
        self.assertEqual(result.abstention_counts_by_class, ((0, 0), (1, 0), (2, 2)))
        self.assertEqual(result.coverage_rate, 0.5)

    def test_all_abstained_has_no_conditional_or_probability_metrics(self) -> None:
        result = evaluate_predictions_with_abstentions(
            np.asarray([], dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.asarray([0, 1, 2, 2]),
            class_order=(0, 1, 2),
        )

        self.assertIsNone(result.conditional_metrics)
        self.assertEqual(result.abstention_counts_by_class, ((0, 1), (1, 1), (2, 2)))
        self.assertEqual((result.n_total, result.n_retained, result.n_abstained), (4, 0, 4))
        self.assertEqual(result.coverage_rate, 0.0)
        self.assertEqual(
            (
                result.balanced_accuracy,
                result.macro_precision,
                result.macro_recall,
                result.macro_f1,
            ),
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertIsNone(result.retained_multiclass_log_loss)
        self.assertIsNone(result.retained_multiclass_brier)
        self.assertIsNone(result.retained_expected_calibration_error)
        for item in result.per_class:
            self.assertEqual(item.false_negative, item.support)
            self.assertEqual((item.precision, item.recall, item.f1), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
