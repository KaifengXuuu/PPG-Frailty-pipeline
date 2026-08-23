from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ppg_frailty.reporting.analyze import _paired_participant_inference
from ppg_frailty.reporting.collect import CollectedStudy, _config_metrics


class ReportingInferenceTests(unittest.TestCase):
    def test_config_metrics_projects_both_bootstrap_ci95_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config_metrics_v2.json").write_text(
                json.dumps(
                    {
                        "status": "passed_trusted_metrics_rebuilt_from_typed_oof",
                        "config_metrics": {
                            "participant_mean_balanced_accuracy": 0.72,
                            "participant_mean_macro_f1": 0.70,
                        },
                        "bootstrap_results": [
                            {
                                "metric": "balanced_accuracy",
                                "estimate": 0.72,
                                "ci95_lower": 0.61,
                                "ci95_upper": 0.81,
                                "n_resamples": 10000,
                                "seed": 42,
                                "n_participants": 29,
                                "n_repeats": 5,
                                "interval_method": "percentile_two_sided_95",
                                "cluster_unit": "participant_with_all_repeats",
                            },
                            {
                                "metric": "macro_f1",
                                "estimate": 0.70,
                                "ci95_lower": 0.59,
                                "ci95_upper": 0.80,
                                "n_resamples": 10000,
                                "seed": 42,
                                "n_participants": 29,
                                "n_repeats": 5,
                                "interval_method": "percentile_two_sided_95",
                                "cluster_unit": "participant_with_all_repeats",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            projected = _config_metrics("case", root)

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(
            projected["participant_cluster_balanced_accuracy_ci95_low"],
            0.61,
        )
        self.assertEqual(
            projected["participant_cluster_balanced_accuracy_ci95_high"],
            0.81,
        )
        self.assertEqual(projected["participant_cluster_macro_f1_ci95_low"], 0.59)
        self.assertEqual(projected["participant_cluster_macro_f1_ci95_high"], 0.80)

    def test_declared_reference_uses_participant_cluster_permutation_and_holm(self) -> None:
        labels = (0, 0, 1, 1, 2, 2)
        reference_probability = {
            0: (0.60, 0.25, 0.15),
            1: (0.45, 0.40, 0.15),
            2: (0.20, 0.55, 0.25),
        }
        better_probability = {
            0: (0.90, 0.05, 0.05),
            1: (0.05, 0.90, 0.05),
            2: (0.05, 0.05, 0.90),
        }

        def rows(case_id: str, *, better: bool = False):
            values = []
            for repeat in range(2):
                for index, label in enumerate(labels):
                    values.append(
                        {
                            "case_id": case_id,
                            "participant_id": f"P{index + 1:02d}",
                            "repeat": repeat,
                            "fold": index % 2,
                            "split_seed": 42 + repeat,
                            "label": label,
                            "probabilities": (
                                better_probability[label]
                                if better
                                else reference_probability[label]
                            ),
                            "class_order": (0, 1, 2),
                            "retained": True,
                            "level": "participant",
                        }
                    )
            return values

        policy = {
            "cluster_unit": "participant_with_all_five_repeat_oof_predictions",
            "paired_permutation_replicates": 31,
            "seed": 42,
            "paired_exchange_unit": "participant",
            "multiplicity_correction": "holm_within_comparison_family",
            "affects_automatic_selection": False,
        }
        case_ids = ("reference", "same", "better")
        collected = CollectedStudy(
            root=Path("."),
            plan={"study": {"study_id": "inference_fixture"}},
            manifest={
                "reference_case_id": "reference",
                "cases": [{"case_id": value} for value in case_ids],
            },
            case_records=(),
            varied_parameters=(),
            controlled_parameters=(),
            cell_rows=(),
            history_rows=(),
            file_oof_rows=(),
            subject_oof_rows=(),
            role_oof_rows=(),
            quality_rows=(),
            trusted_config_metrics=(),
            limitations=(),
            resolved_aggregation_configs=tuple(
                {
                    "case_id": value,
                    "evaluation_statistics": policy,
                    "aggregation": {},
                }
                for value in case_ids
            ),
        )
        output, limitations = _paired_participant_inference(
            collected,
            oof_by_case={
                "reference": rows("reference"),
                "same": rows("same"),
                "better": rows("better", better=True),
            },
            case_ids=case_ids,
        )

        self.assertEqual(limitations, [])
        self.assertEqual(len(output), 4)
        self.assertEqual(
            {row["metric"] for row in output},
            {"balanced_accuracy", "macro_f1"},
        )
        self.assertTrue(all(row["n_resamples"] == 31 for row in output))
        self.assertTrue(all(row["participant_count"] == 6 for row in output))
        self.assertTrue(all(row["repeat_count"] == 2 for row in output))
        self.assertTrue(all(row["holm_family_size"] == 2 for row in output))
        self.assertTrue(
            all(0.0 <= row["raw_two_sided_p_value"] <= 1.0 for row in output)
        )
        identical = [row for row in output if row["candidate_case_id"] == "same"]
        self.assertTrue(all(row["raw_two_sided_p_value"] == 1.0 for row in identical))


if __name__ == "__main__":
    unittest.main()
