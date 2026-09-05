from __future__ import annotations

from pathlib import Path
import json
import re
import tempfile
import unittest

import numpy as np

from ppg_frailty.reporting.historical import (
    _absolute_historical_cluster_ci_rows,
    _archived_per_class_tables,
    _early_exploratory_tests,
    _factor_signal_pair_tables,
    _historical_pairwise_display_tables,
    _holm_adjust,
    _md_display_tables as _historical_md_display_tables,
    _md_table as _historical_md_table,
    _metric_summary,
    _prepare_output_directory,
    _report_path,
)
from ppg_frailty.reporting.historical_suite import (
    _SUITE_INDEX_COLUMNS,
    _derive_design_factors,
    _historical_display_tables,
    _html_display_tables as _suite_html_display_tables,
    _html_table as _suite_html_table,
    _leaderboard_display,
    _md_display_tables as _suite_md_display_tables,
    _md_table as _suite_md_table,
    _plot_specs,
)


def _synthetic_overfit_runs():
    import pandas as pd

    rows = []
    seeds = (42, 10042, 20042, 30042, 40042)
    for config_id, offset in (("base", 0.0), ("candidate", 0.1)):
        for repeat, seed in enumerate(seeds, start=1):
            row = {
                "overfit_config_id": config_id,
                "overfit_config_name": f"name_{config_id}",
                "model": "inceptiontime",
                "resolved_model": "inception_time",
                "repeat": repeat,
                "seed": seed,
                "subject_balanced_accuracy": 0.50 + offset + repeat / 1000.0,
                "subject_macro_f1": 0.45 + offset + repeat / 1000.0,
            }
            for class_index, prefix in enumerate(
                ("pre_frail", "robust_non_frail", "young")
            ):
                row[f"{prefix}_precision"] = 0.50 + offset + class_index / 100.0
                row[f"{prefix}_recall"] = 0.55 + offset + class_index / 100.0
                row[f"{prefix}_f1"] = 0.52 + offset + class_index / 100.0
                row[f"{prefix}_support"] = (9, 12, 8)[class_index]
            rows.append(row)
    return pd.DataFrame(rows)


class HistoricalReportingV2Tests(unittest.TestCase):
    def test_historical_human_tables_are_shared_semantic_views_at_most_eight_columns(
        self,
    ) -> None:
        import pandas as pd

        summary = pd.DataFrame(
            [
                {
                    "config_id": "cfg",
                    "model_id": "InceptionTime",
                    "descriptive_rank": 1,
                    "subject_balanced_accuracy_mean": 0.60,
                    "subject_balanced_accuracy_sample_sd": 0.02,
                    "subject_balanced_accuracy_repeat_t_ci95_low": 0.57,
                    "subject_balanced_accuracy_repeat_t_ci95_high": 0.63,
                    "subject_macro_f1_mean": 0.58,
                    "subject_macro_f1_sample_sd": 0.03,
                    "subject_macro_f1_repeat_t_ci95_low": 0.54,
                    "subject_macro_f1_repeat_t_ci95_high": 0.62,
                }
            ]
        )
        leaderboard = _leaderboard_display(
            summary,
            config_column="config_id",
            model_column="model_id",
        ).to_dict(orient="records")
        inventory = [
            {
                "parameter": "cnn_dropout",
                "parameter_group": "optimization_and_regularization",
                "source_study": "fixture",
                "unique_value_count": 1,
                "observed_values": "0.5",
                "parameter_role": "fixed",
                "comparison_interpretation": "descriptive",
            }
        ]
        tables = _historical_display_tables(leaderboard, inventory)
        self.assertTrue(all(len(columns) <= 8 for _title, _rows, columns in tables))
        self.assertEqual(tables[0][2][0], "config_or_model")
        self.assertEqual(tables[1][2][0], "config_or_model")
        self.assertEqual(tables[2][2][0], "parameter")
        self.assertEqual(_SUITE_INDEX_COLUMNS[0], "Report")
        self.assertLessEqual(len(_SUITE_INDEX_COLUMNS), 8)

        markdown = _suite_md_display_tables(tables)
        markdown_headers = [
            line
            for line, following in zip(markdown.splitlines(), markdown.splitlines()[1:])
            if line.startswith("| ") and following.startswith("| ---")
        ]
        self.assertTrue(markdown_headers)
        self.assertTrue(
            all(len(header.split("|")[1:-1]) <= 8 for header in markdown_headers)
        )
        html = _suite_html_display_tables(tables)
        html_headers = re.findall(r"<thead><tr>(.*?)</tr></thead>", html)
        self.assertEqual(len(html_headers), len(tables))
        self.assertTrue(all(header.count("<th>") <= 8 for header in html_headers))
        for title, _rows, _columns in tables:
            self.assertIn(title, markdown)
            self.assertIn(title, html)

        wide_columns = tuple(f"column_{index}" for index in range(9))
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            _suite_md_table(({},), wide_columns)
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            _suite_html_table(({},), wide_columns)

    def test_historical_pairwise_markdown_splits_lossless_rows_into_narrow_views(
        self,
    ) -> None:
        row = {
            "comparison_family": "historical_matched_three_model_all_pairs",
            "comparison_id": "candidate_vs_reference",
            "comparison_role": "exploratory_legacy_model_comparison_not_ablation",
            "reference_case_id": "reference",
            "candidate_case_id": "candidate",
            "repeat": 0,
            "split_seed": 42,
            "comparison_contract_status": "matched_aggregate_repeat_seed",
            "difference_direction": "candidate_minus_reference",
            "reference_balanced_accuracy": 0.5,
            "candidate_balanced_accuracy": 0.6,
            "balanced_accuracy_delta": 0.1,
            "reference_macro_f1": 0.45,
            "candidate_macro_f1": 0.55,
            "macro_f1_delta": 0.1,
            "reference_macro_roc_auc_ovr": None,
            "candidate_macro_roc_auc_ovr": None,
            "macro_roc_auc_ovr_delta": None,
            "macro_roc_auc_ovr_applicability": (
                "N/A_probability_level_participant_oof_not_archived"
            ),
            "automatic_selection": False,
        }
        tables = _historical_pairwise_display_tables((row,))
        self.assertTrue(all(len(columns) <= 8 for _title, _rows, columns in tables))
        self.assertTrue(
            all(columns[0] == "candidate_case_id" for _title, _rows, columns in tables)
        )
        markdown = _historical_md_display_tables(tables)
        self.assertIn("N/A", markdown)
        self.assertNotIn("| None |", markdown)
        self.assertIn("macro_roc_auc_ovr_applicability", row)

        wide_columns = tuple(f"column_{index}" for index in range(9))
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            _historical_md_table(({},), wide_columns)

    def test_subject_confusion_recovers_specificity_and_one_vs_rest_ba(self) -> None:
        runs = _synthetic_overfit_runs()
        runs["pre_frail_recall"] = 3 / 9
        runs["robust_non_frail_recall"] = 8 / 12
        runs["young_recall"] = 5 / 8
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            report = source / "reports" / "shared_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps(
                    {"subject_confusion_matrix": [[3, 2, 4], [1, 8, 3], [0, 3, 5]]}
                ),
                encoding="utf-8",
            )
            runs["report_path"] = str(report)
            repeat_rows, summary = _archived_per_class_tables(
                runs,
                source_study="archive",
                source=source,
            )
        first = repeat_rows.loc[
            repeat_rows["case_id"].eq("base")
            & repeat_rows["split_seed"].eq(42)
            & repeat_rows["class_label"].eq("pre_frail")
        ].iloc[0]
        self.assertEqual(first["true_positive"], 3)
        self.assertEqual(first["false_negative"], 6)
        self.assertEqual(first["false_positive"], 1)
        self.assertEqual(first["true_negative"], 19)
        self.assertAlmostEqual(first["specificity"], 19 / 20)
        self.assertAlmostEqual(first["balanced_accuracy"], 0.5 * (3 / 9 + 19 / 20))
        self.assertTrue(summary["specificity_mean"].notna().all())
        self.assertTrue(summary["balanced_accuracy_mean"].notna().all())
        self.assertTrue(summary["roc_auc"].isna().all())

    def test_generalization_plot_contract_uses_joint_design_factors(self) -> None:
        import pandas as pd

        rows = []
        for model in ("inceptiontime", "small_inceptiontime"):
            for weight_decay, smoothing in ((0.005, 0.2), (0.01, 0.3)):
                for sampler, quota in (("none", "all"), ("subject_balanced", "16")):
                    rows.append(
                        {
                            "model": model,
                            "resolved_model": model,
                            "cnn_epochs": 10,
                            "cnn_weight_decay": weight_decay,
                            "cnn_dropout": 0.5,
                            "cnn_label_smoothing": smoothing,
                            "sqi_mode": "none",
                            "aggregation": "mean_prob",
                            "window_sampler": sampler,
                            "windows_per_subject_per_epoch": quota,
                            "train_overlap_pct": 0.0,
                            "stage1_regularization_factor": "generalization_grid",
                        }
                    )
        runs = _derive_design_factors(pd.DataFrame(rows))
        parameters = [parameter for parameter, _label, _subset in _plot_specs(runs)]
        self.assertIn("regularization_bundle", parameters)
        self.assertIn("sampling_policy", parameters)
        self.assertIn("quality_route_bundle", runs)
        self.assertNotIn("cnn_weight_decay", parameters)
        self.assertNotIn("cnn_label_smoothing", parameters)
        self.assertNotIn("window_sampler", parameters)
        self.assertNotIn("windows_per_subject_per_epoch", parameters)

    def test_overfit_per_class_tables_cover_every_config_repeat_and_class(self) -> None:
        runs = _synthetic_overfit_runs()
        runs.loc[
            runs["overfit_config_id"].eq("candidate") & runs["seed"].eq(40042),
            "young_f1",
        ] = np.nan
        repeat_rows, summary = _archived_per_class_tables(
            runs,
            source_study="archive",
        )
        self.assertEqual(len(repeat_rows), 2 * 5 * 3)
        self.assertEqual(len(summary), 2 * 3)
        missing = repeat_rows.loc[
            repeat_rows["case_id"].eq("candidate")
            & repeat_rows["class_label"].eq("young")
            & repeat_rows["split_seed"].eq(40042)
        ].iloc[0]
        self.assertTrue(np.isnan(missing["f1"]))
        self.assertEqual(
            missing["f1_applicability"],
            "N/A_archived_field_missing:young_f1",
        )
        candidate_young = summary.loc[
            summary["case_id"].eq("candidate") & summary["class_label"].eq("young")
        ].iloc[0]
        self.assertEqual(candidate_young["f1_n_repeats_available"], 4)
        self.assertEqual(
            candidate_young["f1_applicability"],
            "partial_archive_one_or_more_repeat_fields_missing",
        )
        self.assertTrue(summary["roc_auc"].isna().all())
        self.assertTrue(
            summary["roc_auc_applicability"]
            .str.startswith("N/A_probability_level")
            .all()
        )

    def test_factor_signals_export_five_repeat_deltas_and_three_na_cluster_cis(
        self,
    ) -> None:
        import pandas as pd

        runs = _synthetic_overfit_runs()
        effects = pd.DataFrame(
            [
                {
                    "epoch": 10,
                    "factor": "dropout",
                    "baseline_config_id": "base",
                    "best_observed_config_id": "candidate",
                }
            ]
        )
        repeat_rows, inference = _factor_signal_pair_tables(
            runs,
            effects,
            source_study="archive",
        )
        self.assertEqual(len(repeat_rows), 5)
        self.assertTrue(np.allclose(repeat_rows["balanced_accuracy_delta"], 0.1))
        self.assertTrue(np.allclose(repeat_rows["macro_f1_delta"], 0.1))
        self.assertTrue(repeat_rows["macro_roc_auc_ovr_delta"].isna().all())
        self.assertEqual(len(inference), 3)
        self.assertEqual(
            set(inference["metric"]),
            {"balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"},
        )
        self.assertTrue(inference["participant_cluster_delta_ci95_low"].isna().all())
        self.assertTrue(
            inference["participant_cluster_ci_unavailability_reason"]
            .str.contains("participant-level OOF")
            .all()
        )
        roc = inference.loc[inference["metric"].eq("macro_roc_auc_ovr")].iloc[0]
        self.assertTrue(np.isnan(roc["candidate_minus_reference"]))
        self.assertIn("probability_level", roc["point_delta_source"])

    def test_absolute_cluster_ci_is_explicit_for_each_classifier_metric(self) -> None:
        import pandas as pd

        summary = pd.DataFrame(
            [
                {
                    "overfit_config_id": "cfg",
                    "resolved_model": "inception_time",
                    "subject_balanced_accuracy_mean": 0.61,
                    "subject_macro_f1_mean": 0.59,
                }
            ]
        )
        rows = _absolute_historical_cluster_ci_rows(
            summary,
            source_study="archive",
            classifier_id_column="overfit_config_id",
            model_column="resolved_model",
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            set(rows["metric"]),
            {"balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"},
        )
        self.assertTrue(rows["participant_cluster_ci95_low"].isna().all())
        self.assertTrue(
            rows["participant_cluster_ci_applicability"]
            .eq("N/A_participant_level_oof_rows_not_archived")
            .all()
        )
        roc = rows.loc[rows["metric"].eq("macro_roc_auc_ovr")].iloc[0]
        self.assertTrue(np.isnan(roc["point_estimate"]))

    def test_repeat_summary_uses_sample_sd_and_student_t_interval(self) -> None:
        result = _metric_summary((0.5, 0.6, 0.7, 0.8, 0.9))
        self.assertEqual(result["n_repeats"], 5)
        self.assertAlmostEqual(result["mean"], 0.7)
        self.assertAlmostEqual(
            result["sample_sd"], np.std([0.5, 0.6, 0.7, 0.8, 0.9], ddof=1)
        )
        self.assertLess(result["repeat_t_ci95_low"], result["mean"])
        self.assertGreater(result["repeat_t_ci95_high"], result["mean"])
        self.assertEqual(result["repeat_t_ci95_method"], "two_sided_student_t_0.95")

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = _holm_adjust((0.01, 0.04, 0.03))
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_repeat_sign_flip_is_explicitly_non_v2_inference(self) -> None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        rows = []
        values = {
            "A": (0.5, 0.5, 0.5, 0.5, 0.5),
            "B": (0.6, 0.6, 0.6, 0.6, 0.6),
            "C": (0.7, 0.7, 0.7, 0.7, 0.7),
        }
        for model, scores in values.items():
            for index, score in enumerate(scores):
                rows.append(
                    {
                        "model_display": model,
                        "seed": 42 + index,
                        "subject_balanced_accuracy": score,
                        "subject_macro_f1": score,
                    }
                )
        result = _early_exploratory_tests(pd.DataFrame(rows))
        self.assertEqual(len(result), 6)
        self.assertTrue((result["exact_sign_patterns"] == 32).all())
        self.assertTrue((result["formal_v2_inference"] == False).all())  # noqa: E712
        self.assertTrue(
            np.isfinite(result["holm_adjusted_p_within_metric_three_pairs"]).all()
        )

    def test_output_directory_refuses_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fresh"
            self.assertEqual(_prepare_output_directory(target), target.resolve())
            (target / "stale.csv").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                _prepare_output_directory(target)

    def test_report_path_is_scoped_to_the_declared_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            archived = source / "reports" / "case" / "report.json"
            archived.parent.mkdir(parents=True)
            archived.write_text("{}\n", encoding="utf-8")
            external = root / "other" / "reports" / "case" / "report.json"
            external.parent.mkdir(parents=True)
            external.write_text('{"wrong": true}\n', encoding="utf-8")
            self.assertEqual(_report_path(source, external), archived.resolve())


if __name__ == "__main__":
    unittest.main()
