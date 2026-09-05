from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ppg_frailty.reporting.classification_diagnostics import (
    classification_per_class_metric_rows,
    normalize_classification_rows,
)


class ClassificationPerClassMetricTests(unittest.TestCase):
    def test_multiclass_metrics_are_dynamic_and_one_vs_rest(self) -> None:
        probabilities = (
            (0.8, 0.1, 0.1),
            (0.2, 0.7, 0.1),
            (0.1, 0.8, 0.1),
            (0.1, 0.2, 0.7),
            (0.1, 0.1, 0.8),
            (0.7, 0.2, 0.1),
        )
        labels = (0, 0, 1, 1, 2, 2)
        normalized = normalize_classification_rows(
            [
                {
                    "case_id": "three-class-model",
                    "participant_id": f"p{index}",
                    "label": label,
                    "class_order": (0, 1, 2),
                    "probabilities": probability,
                }
                for index, (label, probability) in enumerate(
                    zip(labels, probabilities, strict=True)
                )
            ]
        )

        rows = classification_per_class_metric_rows(
            normalized,
            class_names={0: "A", 1: "B", 2: "C"},
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["class_name"] for row in rows}, {"A", "B", "C"})
        # AUC consumes the normalized persisted vectors exactly, so the
        # expected values use that same report-facing representation.
        matrix = np.asarray(
            [row["probabilities"] for row in normalized], dtype=np.float64
        )
        label_array = np.asarray(labels, dtype=np.int64)
        for row in rows:
            class_label = int(row["class_label"])
            self.assertEqual(row["true_positive"], 1)
            self.assertEqual(row["false_positive"], 1)
            self.assertEqual(row["true_negative"], 3)
            self.assertEqual(row["false_negative"], 1)
            self.assertEqual(row["support"], 2)
            self.assertEqual(row["predicted_support"], 2)
            self.assertEqual(row["observation_count"], 6)
            self.assertAlmostEqual(float(row["precision"]), 0.5)
            self.assertAlmostEqual(float(row["sensitivity"]), 0.5)
            self.assertEqual(row["recall"], row["sensitivity"])
            self.assertAlmostEqual(float(row["specificity"]), 0.75)
            self.assertAlmostEqual(float(row["balanced_accuracy_ovr"]), 0.625)
            self.assertAlmostEqual(float(row["f1"]), 0.5)
            binary = label_array == class_label
            self.assertAlmostEqual(
                float(row["roc_auc_ovr"]),
                float(roc_auc_score(binary, matrix[:, class_label])),
            )
            self.assertAlmostEqual(
                float(row["pr_auc_ovr"]),
                float(average_precision_score(binary, matrix[:, class_label])),
            )
            self.assertEqual(row["probability_metric_applicability"], "available")
            self.assertEqual(
                row["metric_scope"],
                "one_vs_rest_equal_weight_conditional_on_retention",
            )

    def test_binary_metrics_preserve_frozen_threshold_predictions(self) -> None:
        normalized = normalize_classification_rows(
            [
                {
                    "participant_id": f"p{index}",
                    "file_id": f"f{index}",
                    "activity_label": label,
                    "p_active": score,
                    "threshold": 0.8,
                }
                for index, (label, score) in enumerate(
                    ((0, 0.1), (0, 0.7), (1, 0.75), (1, 0.9))
                )
            ],
            classifier_id="motion-model",
            evaluation_id="frozen-target",
            aggregation_level="file",
            label_field="activity_label",
        )
        self.assertEqual(
            [row["predicted_label"] for row in normalized],
            [0, 0, 0, 1],
        )

        rows = classification_per_class_metric_rows(
            normalized,
            class_names={0: "static", 1: "motion"},
        )
        by_class = {int(row["class_label"]): row for row in rows}

        self.assertEqual(
            {
                key: by_class[1][key]
                for key in (
                    "true_positive",
                    "false_positive",
                    "true_negative",
                    "false_negative",
                )
            },
            {
                "true_positive": 1,
                "false_positive": 0,
                "true_negative": 2,
                "false_negative": 1,
            },
        )
        self.assertEqual(
            {
                key: by_class[0][key]
                for key in (
                    "true_positive",
                    "false_positive",
                    "true_negative",
                    "false_negative",
                )
            },
            {
                "true_positive": 2,
                "false_positive": 1,
                "true_negative": 1,
                "false_negative": 0,
            },
        )
        self.assertAlmostEqual(float(by_class[1]["precision"]), 1.0)
        self.assertAlmostEqual(float(by_class[1]["recall"]), 0.5)
        self.assertAlmostEqual(float(by_class[1]["specificity"]), 1.0)
        self.assertAlmostEqual(float(by_class[1]["balanced_accuracy_ovr"]), 0.75)
        self.assertEqual(
            by_class[1]["prediction_rule_source"],
            "normalized_predicted_label_preserves_frozen_threshold",
        )

    def test_groups_remain_separate_and_invalid_group_fails_closed(self) -> None:
        normalized = normalize_classification_rows(
            [
                {
                    "case_id": "model-a",
                    "evaluation_id": "outer",
                    "level": "participant",
                    "participant_id": "p0",
                    "label": 0,
                    "class_order": (0, 1),
                    "probabilities": (0.8, 0.2),
                },
                {
                    "case_id": "model-a",
                    "evaluation_id": "outer",
                    "level": "participant",
                    "participant_id": "p1",
                    "label": 1,
                    "class_order": (0, 1),
                    "probabilities": (0.2, 0.8),
                },
                {
                    "case_id": "model-b",
                    "evaluation_id": "external",
                    "level": "file",
                    "participant_id": "p2",
                    "label": 4,
                    "class_order": (4, 8),
                    "probabilities": (0.6, 0.4),
                },
                {
                    "case_id": "model-b",
                    "evaluation_id": "external",
                    "level": "file",
                    "participant_id": "p3",
                    "label": 8,
                    "class_order": (4, 8),
                    "probabilities": (0.4, 0.6),
                },
            ]
        )
        rows = classification_per_class_metric_rows(normalized)
        self.assertEqual(
            {
                (
                    row["classifier_id"],
                    row["evaluation_id"],
                    row["aggregation_level"],
                )
                for row in rows
            },
            {
                ("model-a", "outer", "participant"),
                ("model-b", "external", "file"),
            },
        )
        self.assertEqual(
            {
                row["class_label"]
                for row in rows
                if row["classifier_id"] == "model-b"
            },
            {4, 8},
        )

        malformed = [dict(row) for row in normalized[:2]]
        malformed[1]["class_order"] = (1, 0)
        with self.assertRaisesRegex(ValueError, "class_order differs"):
            classification_per_class_metric_rows(malformed)

    def test_fully_abstained_classifier_retains_each_class_as_explicit_na(self) -> None:
        normalized = normalize_classification_rows(
            [
                {
                    "case_id": "abstained-route",
                    "participant_id": "p0",
                    "label": 0,
                    "class_order": (0, 1, 2),
                    "probabilities": (0.8, 0.1, 0.1),
                    "retained": False,
                },
                {
                    "case_id": "abstained-route",
                    "participant_id": "p1",
                    "label": 1,
                    "class_order": (0, 1, 2),
                    "probabilities": (0.1, 0.8, 0.1),
                    "retained": False,
                },
            ]
        )
        rows = classification_per_class_metric_rows(normalized)
        self.assertEqual(len(rows), 3)
        self.assertTrue(
            all(
                row["result_applicability"]
                == "N/A_no_retained_classification_observations"
                and row["input_observation_count"] == 2
                and row["retained_observation_count"] == 0
                and row["excluded_observation_count"] == 2
                and row["f1"] is None
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
