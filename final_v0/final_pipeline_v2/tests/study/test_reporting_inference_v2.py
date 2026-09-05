from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ppg_frailty.reporting.analyze import _paired_participant_inference
from ppg_frailty.reporting.collect import CollectedStudy, _config_metrics
from ppg_frailty.reporting.conclusions import (
    holm_adjust_paired_inference_rows,
    paired_inference_against_reference,
    paired_repeat_deltas_against_reference,
)


class ReportingInferenceTests(unittest.TestCase):
    @staticmethod
    def _binary_rows(case_id: str, *, repeats: tuple[int, ...] = (0, 1)):
        rows = []
        for repeat in repeats:
            for index, (label, score) in enumerate(
                ((0, 0.10), (0, 0.35), (1, 0.65), (1, 0.90))
            ):
                rows.append(
                    {
                        "classifier_id": case_id,
                        "participant_id": f"P{index + 1:02d}",
                        "repeat": repeat,
                        "fold": index % 2,
                        "split_seed": 42 + repeat,
                        "true_label": label,
                        "class_order": (0, 1),
                        "probabilities": (1.0 - score, score),
                        "retained": True,
                    }
                )
        return rows

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
        self.assertEqual(len(output), 6)
        self.assertEqual(
            {row["metric"] for row in output},
            {"balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"},
        )
        permutation_rows = [
            row for row in output if row["metric"] != "macro_roc_auc_ovr"
        ]
        roc_rows = [
            row for row in output if row["metric"] == "macro_roc_auc_ovr"
        ]
        self.assertTrue(all(row["n_resamples"] == 31 for row in permutation_rows))
        self.assertTrue(all(row["n_resamples"] is None for row in roc_rows))
        self.assertTrue(all(row["bootstrap_resamples"] == 31 for row in output))
        self.assertTrue(all(row["participant_count"] == 6 for row in output))
        self.assertTrue(all(row["repeat_count"] == 2 for row in output))
        self.assertTrue(
            all(row["holm_family_size"] == 2 for row in permutation_rows)
        )
        self.assertTrue(all(row["holm_family_size"] is None for row in roc_rows))
        self.assertTrue(
            all(
                0.0 <= row["raw_two_sided_p_value"] <= 1.0
                for row in permutation_rows
            )
        )
        identical = [row for row in output if row["candidate_case_id"] == "same"]
        self.assertTrue(
            all(
                row["raw_two_sided_p_value"] == 1.0
                for row in identical
                if row["metric"] != "macro_roc_auc_ovr"
            )
        )
        self.assertTrue(
            all(
                row["participant_cluster_delta_ci95_low"] == 0.0
                and row["participant_cluster_delta_ci95_high"] == 0.0
                for row in identical
            )
        )

    def test_binary_pair_and_each_declared_repeat_are_supported(self) -> None:
        predictions = [
            *self._binary_rows("reference"),
            *self._binary_rows("candidate"),
        ]
        inference = paired_inference_against_reference(
            predictions,
            reference_case_id="reference",
            candidate_case_ids=("candidate", "missing"),
            expected_repeats=(0, 1),
            comparison_family="binary_family",
            inference_role="declared_binary_test",
            n_resamples=31,
            bootstrap_resamples=40,
            seed=7,
        )
        self.assertEqual(len(inference), 6)
        available = [
            row for row in inference if row["candidate_case_id"] == "candidate"
        ]
        unavailable = [
            row for row in inference if row["candidate_case_id"] == "missing"
        ]
        self.assertEqual(
            {row["metric"] for row in available},
            {"balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"},
        )
        self.assertTrue(
            all(
                row["comparison_contract_status"] == "matched_complete_roster"
                for row in available
            )
        )
        self.assertTrue(
            all(
                row["comparison_contract_status"]
                == "N/A_candidate_has_no_valid_participant_oof"
                for row in unavailable
            )
        )

        repeat_rows = paired_repeat_deltas_against_reference(
            predictions,
            reference_case_id="reference",
            candidate_case_ids=("candidate", "missing"),
            expected_repeats=(0, 1),
            comparison_family="binary_family",
            comparison_role="binary_model_comparison",
        )
        self.assertEqual(len(repeat_rows), 4)
        self.assertEqual(
            {
                row["repeat"]
                for row in repeat_rows
                if row["candidate_case_id"] == "candidate"
            },
            {0, 1},
        )
        self.assertTrue(
            all(
                row["comparison_contract_status"] == "matched_complete_roster"
                for row in repeat_rows
                if row["candidate_case_id"] == "candidate"
            )
        )
        missing_repeat_rows = [
            row
            for row in repeat_rows
            if row["candidate_case_id"] == "missing"
        ]
        self.assertEqual(len(missing_repeat_rows), 2)
        self.assertTrue(
            all(
                row["balanced_accuracy_delta"] is None
                and row["macro_f1_delta"] is None
                and row["macro_roc_auc_ovr_delta"] is None
                for row in missing_repeat_rows
            )
        )

    def test_missing_declared_repeat_is_explicit_and_never_nan(self) -> None:
        predictions = [
            *self._binary_rows("reference", repeats=(0,)),
            *self._binary_rows("candidate", repeats=(0,)),
        ]
        rows = paired_repeat_deltas_against_reference(
            predictions,
            reference_case_id="reference",
            candidate_case_ids=("candidate",),
            expected_repeats=(0, 1),
            comparison_family="repeat_audit",
            comparison_role="declared_ablation",
        )
        self.assertEqual(len(rows), 2)
        by_repeat = {row["repeat"]: row for row in rows}
        self.assertEqual(
            by_repeat[0]["comparison_contract_status"],
            "matched_complete_roster",
        )
        self.assertEqual(
            by_repeat[1]["comparison_contract_status"],
            "N/A_declared_repeat_missing_from_matched_oof",
        )
        self.assertIsNone(by_repeat[1]["balanced_accuracy_delta"])
        self.assertIsNone(by_repeat[1]["macro_f1_delta"])
        self.assertIsNone(by_repeat[1]["macro_roc_auc_ovr_delta"])
        json.dumps(rows, allow_nan=False)

    def test_frozen_registry_roster_mismatch_invalidates_all_pair_metrics(self) -> None:
        predictions = [
            *self._binary_rows("reference"),
            *self._binary_rows("candidate"),
        ]
        # P01 is deliberately assigned to the wrong fold in the authoritative
        # registry.  Agreement between candidate and reference OOF is therefore
        # insufficient to authorize either inference or descriptive deltas.
        mismatched_membership = {
            (f"P{index + 1:02d}", repeat): (
                (1 if index == 0 else index % 2),
                42 + repeat,
            )
            for repeat in (0, 1)
            for index in range(4)
        }
        inference = paired_inference_against_reference(
            predictions,
            reference_case_id="reference",
            candidate_case_ids=("candidate",),
            expected_repeats=(0, 1),
            expected_membership=mismatched_membership,
            comparison_family="frozen_registry_contract",
            inference_role="declared_binary_test",
            n_resamples=31,
            bootstrap_resamples=40,
            seed=7,
        )
        repeat_rows = paired_repeat_deltas_against_reference(
            predictions,
            reference_case_id="reference",
            candidate_case_ids=("candidate",),
            expected_repeats=(0, 1),
            expected_membership=mismatched_membership,
            comparison_family="frozen_registry_contract",
            comparison_role="declared_binary_test",
        )
        self.assertEqual(len(inference), 3)
        self.assertEqual(len(repeat_rows), 2)
        self.assertTrue(
            all(
                row["comparison_contract_status"]
                == "N/A_reference_frozen_split_registry_roster_mismatch"
                and row["candidate_minus_reference"] is None
                and row["participant_cluster_delta_ci95_low"] is None
                and row["participant_cluster_delta_ci95_high"] is None
                for row in inference
            )
        )
        self.assertTrue(
            all(
                row["comparison_contract_status"]
                == "N/A_reference_frozen_split_registry_roster_mismatch"
                and row["balanced_accuracy_delta"] is None
                and row["macro_f1_delta"] is None
                and row["macro_roc_auc_ovr_delta"] is None
                for row in repeat_rows
            )
        )

    def test_holm_is_reapplied_across_all_pairs_in_one_family(self) -> None:
        rows = []
        for index in range(8):
            for metric in ("balanced_accuracy", "macro_f1"):
                rows.append(
                    {
                        "comparison_family": "eight_matched_models",
                        "comparison_id": f"pair_{index}",
                        "metric": metric,
                        "raw_two_sided_p_value": 0.01 + index * 0.005,
                    }
                )
            rows.append(
                {
                    "comparison_family": "eight_matched_models",
                    "comparison_id": f"pair_{index}",
                    "metric": "macro_roc_auc_ovr",
                    "raw_two_sided_p_value": None,
                }
            )
        adjusted = holm_adjust_paired_inference_rows(rows)
        self.assertTrue(
            all(
                row["holm_family_size"] == 8
                for row in adjusted
                if row["metric"] in {"balanced_accuracy", "macro_f1"}
            )
        )
        self.assertTrue(
            all(
                row["holm_family_size"] is None
                for row in adjusted
                if row["metric"] == "macro_roc_auc_ovr"
            )
        )


if __name__ == "__main__":
    unittest.main()
