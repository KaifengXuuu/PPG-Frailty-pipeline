"""Contracts for reusable model/module-owned reporter profiles."""

from __future__ import annotations

import unittest

from ppg_frailty.module_registry import (
    ALL_MODULES,
    MODEL_MODULES,
    MODULE_REPORTER_BINDINGS,
    module_reporter_binding,
)
from ppg_frailty.reporting.conclusions import (
    _markdown_table as _conclusion_markdown_table,
    classification_comparison_rows,
    classification_comparison_table_views,
    classification_conclusion_rows,
    paired_inference_against_reference,
)
from ppg_frailty.reporting.profiles import (
    REPORTER_PROFILES,
    REPORTER_PROFILE_VIEW_SCHEMAS,
    annotate_component_row,
    markdown_reporter_profile_tables,
    reporter_methods_markdown,
    reporter_profile_rows,
    required_figure_modules,
)


class ReporterProfilesAndConclusionsTests(unittest.TestCase):
    def test_result_interpretation_rejects_wide_human_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            _conclusion_markdown_table(
                [{f"column_{index}": index for index in range(9)}]
            )

    def test_human_reporter_profile_views_are_complete_and_narrow(self) -> None:
        component = annotate_component_row(
            {
                "component_role": "classifier",
                "module_id": "CompactCNN1D",
                "execution_state": "executed",
            }
        )
        rows = reporter_profile_rows([component])
        lossless_fields = set(rows[0])
        view_fields = {
            field
            for _title, schema in REPORTER_PROFILE_VIEW_SCHEMAS
            for field, _label in schema
        }
        self.assertEqual(view_fields, lossless_fields)
        self.assertTrue(
            all(len(schema) <= 8 for _title, schema in REPORTER_PROFILE_VIEW_SCHEMAS)
        )

        rendered = markdown_reporter_profile_tables(rows)
        for title, _schema in REPORTER_PROFILE_VIEW_SCHEMAS:
            self.assertIn(f"### {title}", rendered)
        markdown_headers = [line for line in rendered.splitlines() if line.startswith("| ")]
        self.assertTrue(markdown_headers)
        self.assertTrue(all(line.count("|") - 1 <= 8 for line in markdown_headers))

    def test_registry_owns_a_valid_binding_for_every_registered_module(self) -> None:
        registered = {(row.family, row.module_id) for row in ALL_MODULES}
        self.assertEqual(set(MODULE_REPORTER_BINDINGS), registered)
        for binding in MODULE_REPORTER_BINDINGS.values():
            self.assertIn(binding.reporter_extension_id, REPORTER_PROFILES)
            self.assertIn(binding.binding_kind, {"extension", "audit_only"})
            self.assertTrue(binding.algorithm_summary)
            self.assertTrue(binding.references)

    def test_every_registered_model_has_an_explicit_model_extension(self) -> None:
        for model in MODEL_MODULES:
            binding = module_reporter_binding(model.module_id, family="model")
            self.assertEqual(binding["reporter_binding_kind"], "extension")
            self.assertNotEqual(
                binding["reporter_extension_id"], "audit_provenance_v1"
            )
            self.assertIn(binding["reporter_extension_id"], REPORTER_PROFILES)

    def test_model_machine_id_uses_factory_canonicalization(self) -> None:
        canonical = module_reporter_binding("InceptionTimeFull", family="model")
        machine = module_reporter_binding("inception_full", family="model")
        self.assertEqual(machine, canonical)

    def test_unknown_active_component_fails_closed_even_with_profile_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown active component"):
            annotate_component_row(
                {
                    "component_role": "classifier",
                    "module_id": "unregistered_classifier",
                    "execution_state": "enabled",
                    "reporter_profile_id": "multiclass_participant_oof_v1",
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown active component_role"):
            annotate_component_row(
                {
                    "component_role": "new_unregistered_role",
                    "module_id": "anything",
                    "execution_state": "enabled",
                }
            )

    def test_disabled_unknown_component_is_explicitly_not_applicable(self) -> None:
        component = annotate_component_row(
            {
                "component_role": "classifier",
                "module_id": "unregistered_disabled_comparator",
                "execution_state": "disabled_control",
            }
        )
        self.assertEqual(component["reporter_profile_id"], "audit_provenance_v1")
        self.assertEqual(component["model_reporter_extension_id"], "not_applicable")
        self.assertEqual(
            component["reporter_binding_kind"], "inactive_not_applicable"
        )

    def test_stage5_denoiser_profile_has_correct_applicability_language(self) -> None:
        component = annotate_component_row(
            {
                "component_role": "denoiser",
                "module_id": "pca_bss",
                "execution_state": "executed",
                # Historical name remains a valid rebuild input.
                "reporter_profile_id": "denoiser_ecg_ppg_endpoint_v1",
            }
        )
        self.assertEqual(
            component["reporter_profile_id"], "stage5_ecg_ppg_denoiser_v1"
        )
        profile = REPORTER_PROFILES["stage5_ecg_ppg_denoiser_v1"]
        profile_text = " ".join(
            (
                profile.algorithm_summary,
                *profile.statistical_methods,
                *profile.limitations,
            )
        )
        self.assertNotIn("Q_rate", profile_text)
        self.assertIn("between-subject dispersion", profile_text)
        self.assertIn("not training-repeat variability", profile_text)
        self.assertIn("participant-paired Monte-Carlo sign-flip", profile_text)
        self.assertIn("P values are exploratory", profile_text)
        self.assertIn("Holm correction", profile_text)

    def test_classifier_composes_endpoint_and_model_extension(self) -> None:
        component = annotate_component_row(
            {
                "component_role": "classifier_tuning_candidate",
                "module_id": "InceptionTimeFull",
                "algorithm_kernel_description": "",
            }
        )
        self.assertEqual(
            component["reporter_profile_id"], "multiclass_participant_oof_v1"
        )
        self.assertEqual(
            component["model_reporter_extension_id"],
            "inceptiontime_single_network_model_v1",
        )
        self.assertIn("single-network", component["algorithm_kernel_description"])
        self.assertIn("10.1007/s10618-020-00710-y", component["algorithm_references"])
        profiles = reporter_profile_rows([component])
        self.assertEqual(
            {row["profile_id"] for row in profiles},
            {
                "multiclass_participant_oof_v1",
                "inceptiontime_single_network_model_v1",
            },
        )
        figures = required_figure_modules([component])
        self.assertIn("classification_roc_auc_curves", figures)
        self.assertIn("classification_prediction_tsne", figures)
        methods = reporter_methods_markdown([component])
        self.assertIn("Student-t 95% CI", methods)
        self.assertIn("count and seed are configurable (default 10,000)", methods)
        self.assertIn("count and seed are configurable (default 100,000)", methods)
        self.assertIn("implementation version and NumPy", methods)

    def test_legacy_peak_contract_is_not_relabelled_current(self) -> None:
        component = annotate_component_row(
            {
                "component_role": "peak_detector",
                "module_id": "aboy_project_v1",
                "reporter_profile_id": "beat_detector_legacy_persisted_v1",
            }
        )
        self.assertEqual(
            component["reporter_profile_id"],
            "beat_detector_legacy_persisted_v1",
        )
        methods = reporter_methods_markdown([component])
        self.assertIn("not back-applied", methods)
        self.assertIn("Aboy et al. (2005)", methods)

    def test_exploratory_paired_inference_never_rewrites_selection(self) -> None:
        predictions = []
        for case_id in ("selected", "challenger"):
            for repeat in range(2):
                for participant_index, label in enumerate((0, 0, 1, 1, 2, 2)):
                    probabilities = [0.05, 0.05, 0.05]
                    probabilities[label] = 0.90
                    predictions.append(
                        {
                            "classifier_id": case_id,
                            "participant_id": f"p{participant_index}",
                            "repeat": repeat,
                            "fold": participant_index % 2,
                            "split_seed": 42 + 10_000 * repeat,
                            "true_label": label,
                            "class_order": [0, 1, 2],
                            "probabilities": probabilities,
                        }
                    )
        inference = paired_inference_against_reference(
            predictions,
            reference_case_id="selected",
            comparison_family="post_selection_audit",
            inference_role="exploratory_post_selection",
            n_resamples=100,
            bootstrap_resamples=100,
        )
        self.assertEqual(len(inference), 3)
        permutation_rows = [
            row for row in inference if row["metric"] in {"balanced_accuracy", "macro_f1"}
        ]
        roc_rows = [row for row in inference if row["metric"] == "macro_roc_auc_ovr"]
        self.assertTrue(
            all(row["holm_adjusted_p_value"] == 1.0 for row in permutation_rows)
        )
        self.assertEqual(len(roc_rows), 1)
        self.assertIsNone(roc_rows[0]["holm_adjusted_p_value"])
        self.assertEqual(
            roc_rows[0]["p_value_applicability"],
            "N/A_no_registered_roc_auc_permutation_test",
        )
        self.assertTrue(all(row["automatic_selection"] is False for row in inference))
        self.assertTrue(
            all(row["permutation_implementation_version"] for row in permutation_rows)
        )
        self.assertTrue(
            all(row["permutation_rng_contract"] for row in permutation_rows)
        )

        comparison = classification_comparison_rows(
            [
                {
                    "case_id": "challenger",
                    "status": "passed",
                    "complete_for_requested_execution": True,
                    "repeat_count": 5,
                    "fold_cell_count": 25,
                    "participant_mean_balanced_accuracy": 0.61,
                    "repeat_balanced_accuracy_sample_sd": 0.02,
                    "participant_mean_macro_f1": 0.60,
                    "repeat_macro_f1_sample_sd": 0.02,
                    "participant_mean_macro_roc_auc_ovr": 0.75,
                    "repeat_macro_roc_auc_ovr_sample_sd": 0.02,
                    "participant_cluster_balanced_accuracy_ci95_low": 0.50,
                    "participant_cluster_balanced_accuracy_ci95_high": 0.70,
                    "participant_cluster_macro_f1_ci95_low": 0.49,
                    "participant_cluster_macro_f1_ci95_high": 0.69,
                },
                {
                    "case_id": "selected",
                    "status": "passed",
                    "complete_for_requested_execution": True,
                    "repeat_count": 5,
                    "fold_cell_count": 25,
                    "participant_mean_balanced_accuracy": 0.60,
                    "repeat_balanced_accuracy_sample_sd": 0.02,
                    "participant_mean_macro_f1": 0.59,
                    "repeat_macro_f1_sample_sd": 0.02,
                    "participant_mean_macro_roc_auc_ovr": 0.74,
                    "repeat_macro_roc_auc_ovr_sample_sd": 0.02,
                    "participant_cluster_balanced_accuracy_ci95_low": 0.49,
                    "participant_cluster_balanced_accuracy_ci95_high": 0.69,
                    "participant_cluster_macro_f1_ci95_low": 0.48,
                    "participant_cluster_macro_f1_ci95_high": 0.68,
                },
            ],
            paired_inference=inference,
        )
        conclusions = classification_conclusion_rows(
            comparison,
            selected_case_id="selected",
            selection_basis="persisted protocol ranking",
            study_role="tuning",
            planned_case_count=2,
        )
        selection = next(row for row in conclusions if row["angle"] == "selection")
        self.assertEqual(selection["confidence"], "low_metric_disagreement")
        self.assertEqual(
            selection["selection_effect"],
            "retain_persisted_choice_without_rewriting_history",
        )

    def test_single_repeat_preserves_means_and_declared_inference(self) -> None:
        inference = [
            {
                "reference_case_id": "base",
                "candidate_case_id": "candidate",
                "metric": "balanced_accuracy",
                "candidate_minus_reference": 0.01,
                "raw_two_sided_p_value": 0.8,
                "holm_adjusted_p_value": 0.8,
                "reject_null_after_holm": False,
                "inference_role": "declared_reference_confirmatory",
            }
        ]
        comparison = classification_comparison_rows(
            [
                {
                    "case_id": "candidate",
                    "status": "passed",
                    "complete_for_requested_execution": True,
                    "repeat_count": 1,
                    "fold_cell_count": 5,
                    "participant_mean_balanced_accuracy": 0.61,
                    "repeat_balanced_accuracy_sample_sd": None,
                    "participant_mean_macro_f1": 0.60,
                    "repeat_macro_f1_sample_sd": None,
                    "participant_mean_macro_roc_auc_ovr": 0.75,
                    "repeat_macro_roc_auc_ovr_sample_sd": None,
                },
                {
                    "case_id": "base",
                    "status": "passed",
                    "complete_for_requested_execution": True,
                    "repeat_count": 1,
                    "fold_cell_count": 5,
                    "participant_mean_balanced_accuracy": 0.60,
                    "repeat_balanced_accuracy_sample_sd": None,
                    "participant_mean_macro_f1": 0.59,
                    "repeat_macro_f1_sample_sd": None,
                    "participant_mean_macro_roc_auc_ovr": 0.74,
                    "repeat_macro_roc_auc_ovr_sample_sd": None,
                },
            ],
            paired_inference=inference,
        )
        self.assertEqual(
            comparison[0]["balanced_accuracy_mean_sd_percent"],
            "61.0 (SD N/A; n=1 repeat)",
        )
        conclusions = classification_conclusion_rows(
            comparison,
            selected_case_id=None,
            selection_basis="manual review",
            study_role="ablation",
            planned_case_count=2,
            inference_reference_case_ids=("base",),
        )
        paired = next(row for row in conclusions if row["angle"] == "paired_inference")
        self.assertIn("1 candidate contrasts", paired["finding"])
        self.assertEqual(paired["confidence"], "confirmatory")

    def test_comparison_table_rejects_ambiguous_inference_families(self) -> None:
        case_summary = [
            {
                "case_id": "candidate",
                "participant_mean_balanced_accuracy": 0.60,
                "participant_mean_macro_f1": 0.59,
            }
        ]
        paired = [
            {
                "candidate_case_id": "candidate",
                "reference_case_id": "reference_a",
                "comparison_family": "family_a",
                "metric": "balanced_accuracy",
            },
            {
                "candidate_case_id": "candidate",
                "reference_case_id": "reference_b",
                "comparison_family": "family_b",
                "metric": "balanced_accuracy",
            },
        ]
        with self.assertRaisesRegex(ValueError, "multiple inference families"):
            classification_comparison_rows(
                case_summary,
                paired_inference=paired,
            )

    def test_comparison_presentation_is_split_into_narrow_lossless_views(self) -> None:
        paired = [
            {
                "candidate_case_id": "candidate",
                "reference_case_id": "reference",
                "metric": "balanced_accuracy",
                "candidate_minus_reference": 0.01,
                "participant_cluster_delta_ci95_low": -0.01,
                "participant_cluster_delta_ci95_high": 0.03,
                "holm_adjusted_p_value": 0.4,
                "reject_null_after_holm": False,
                "p_value_applicability": "available",
            }
        ]
        comparison = classification_comparison_rows(
            [
                {
                    "case_id": "candidate",
                    "status": "passed",
                    "complete_for_requested_execution": True,
                    "repeat_count": 5,
                    "fold_cell_count": 25,
                    "participant_mean_balanced_accuracy": 0.61,
                    "repeat_balanced_accuracy_sample_sd": 0.02,
                    "repeat_balanced_accuracy_ci95_low": 0.58,
                    "repeat_balanced_accuracy_ci95_high": 0.64,
                    "participant_cluster_balanced_accuracy_ci95_low": 0.50,
                    "participant_cluster_balanced_accuracy_ci95_high": 0.70,
                    "participant_mean_macro_f1": 0.60,
                    "repeat_macro_f1_sample_sd": 0.03,
                    "participant_mean_macro_roc_auc_ovr": 0.75,
                    "repeat_macro_roc_auc_ovr_sample_sd": 0.04,
                    "participant_mean_macro_pr_auc_ovr": 0.70,
                    "repeat_macro_pr_auc_ovr_sample_sd": 0.05,
                    "worst_fold_balanced_accuracy": 0.55,
                    "worst_class_recall": 0.50,
                    "worst_class_f1": 0.49,
                    "expected_calibration_error": 0.10,
                }
            ],
            paired_inference=paired,
        )
        compatibility_copy = [dict(row) for row in comparison]
        views = classification_comparison_table_views(
            comparison,
            paired_inference=paired,
        )
        self.assertEqual(
            set(views),
            {
                "ranking_performance",
                "uncertainty_ci",
                "paired_inference",
                "robustness",
            },
        )
        self.assertTrue(
            all(len(row) <= 8 for rows in views.values() for row in rows)
        )
        self.assertEqual(
            next(iter(views["ranking_performance"][0])), "case_id"
        )
        self.assertEqual(next(iter(views["uncertainty_ci"][0])), "case_id")
        self.assertEqual(
            next(iter(views["paired_inference"][0])), "candidate_case_id"
        )
        self.assertEqual(next(iter(views["robustness"][0])), "case_id")
        self.assertEqual(
            views["ranking_performance"][0][
                "balanced_accuracy_mean_sd_percent"
            ],
            "61.0 ± 2.0",
        )
        ba_uncertainty = next(
            row
            for row in views["uncertainty_ci"]
            if row["metric"] == "balanced_accuracy"
        )
        self.assertEqual(ba_uncertainty["repeat_t_ci95_percent"], "[58.0, 64.0]")
        self.assertEqual(
            ba_uncertainty["participant_cluster_ci95_percent"],
            "[50.0, 70.0]",
        )
        self.assertEqual(
            views["paired_inference"][0]["holm_adjusted_p_value"], 0.4
        )
        self.assertEqual(comparison, compatibility_copy)


if __name__ == "__main__":
    unittest.main()
