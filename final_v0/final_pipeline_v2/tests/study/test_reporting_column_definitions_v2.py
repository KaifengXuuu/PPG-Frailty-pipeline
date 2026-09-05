from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ppg_frailty.reporting.tabular import (
    ReportTable,
    column_definition,
    column_definition_rows,
    html_column_definitions_block,
    markdown_column_definitions_block,
    table_column_definition_rows,
    table_column_definition_rows_from_csv_directory,
    write_csv,
    write_table_column_definitions,
)


class ReportingColumnDefinitionTests(unittest.TestCase):
    def test_common_classifier_fields_have_explicit_formulas_and_safe_fallbacks(
        self,
    ) -> None:
        fields = (
            "case_id",
            "balanced_accuracy",
            "BA_legacy_aggregation",
            "macro_f1",
            "macro_roc_auc_ovr",
            "participant_cluster_balanced_accuracy_ci95",
            "participant_cluster_delta_ci95_low",
            "ba_paired_delta_cluster_ci95_low",
            "macro_roc_auc_ovr_candidate_minus_reference_participant_cluster_ci95",
            "macro_f1_delta",
            "balanced_accuracy_p_value",
            "repeat_ci95_method",
            "participant_mean_macro_f1",
            "repeat_macro_f1_sample_sd",
            "coverage_rate",
            "retained_participant_count",
            "unregistered_payload",
        )

        rows = column_definition_rows(fields, table_name="classifier_results")

        self.assertEqual(len(rows), len(fields))
        by_name = {row["column_name"]: row for row in rows}
        self.assertIn("sum_c", by_name["balanced_accuracy"]["formula"])
        self.assertIn("sum_c", by_name["BA_legacy_aggregation"]["formula"])
        self.assertIn("sum_c F1_c", by_name["macro_f1"]["formula"])
        self.assertIn("sum_c ROC-AUC_c", by_name["macro_roc_auc_ovr"]["formula"])
        self.assertIn(
            "resamples participant IDs with replacement",
            by_name["participant_cluster_balanced_accuracy_ci95"]["formula"],
        )
        self.assertIn(
            "metric_candidate,b - metric_reference,b",
            by_name["participant_cluster_delta_ci95_low"]["formula"],
        )
        self.assertIn(
            "metric_candidate,b - metric_reference,b",
            by_name["ba_paired_delta_cluster_ci95_low"]["formula"],
        )
        self.assertIn(
            "metric_candidate,b - metric_reference,b",
            by_name[
                "macro_roc_auc_ovr_candidate_minus_reference_participant_cluster_ci95"
            ]["formula"],
        )
        self.assertIn("candidate - metric_reference", by_name["macro_f1_delta"]["formula"])
        self.assertIn("Pr_H0", by_name["balanced_accuracy_p_value"]["formula"])
        self.assertTrue(by_name["repeat_ci95_method"]["formula"].startswith("N/A"))
        self.assertIn("(1/n) * sum_i", by_name["participant_mean_macro_f1"]["formula"])
        self.assertIn("n - 1", by_name["repeat_macro_f1_sample_sd"]["formula"])
        self.assertEqual(
            by_name["coverage_rate"]["formula"],
            "coverage = n_retained / n_total",
        )
        self.assertIn("sum_i 1", by_name["retained_participant_count"]["formula"])
        self.assertTrue(by_name["case_id"]["formula"].startswith("N/A"))
        self.assertTrue(
            by_name["unregistered_payload"]["formula"].startswith("N/A")
        )
        self.assertTrue(all(row["definition"] and row["formula"] for row in rows))

    def test_mean_sd_composite_documents_both_source_fields_and_scale(self) -> None:
        item = column_definition(
            (
                "participant_mean_balanced_accuracy",
                "repeat_balanced_accuracy_sample_sd",
                True,
            ),
            display_label="BA mean +/- SD (%)",
        )

        self.assertEqual(
            item.source_fields,
            (
                "participant_mean_balanced_accuracy",
                "repeat_balanced_accuracy_sample_sd",
            ),
        )
        self.assertEqual(item.display_label, "BA mean +/- SD (%)")
        self.assertIn("100 * participant_mean_balanced_accuracy", item.formula)
        self.assertIn("n - 1", item.formula)

    def test_stage5_compact_columns_document_endpoint_scale_and_p_test(self) -> None:
        mean_sd = column_definition("participant_macro_mean_sd")
        interval = column_definition("participant_bootstrap_ci95")
        denoiser_p = column_definition("holm_p_vs_identity")
        detector_p = column_definition("holm_p_vs_frailty29_trained")
        configurable_p = column_definition("holm_p_vs_reference")

        self.assertIn("s=100", mean_sd.formula)
        self.assertIn("participant IDs are sampled with replacement", interval.formula)
        self.assertIn("B+1", denoiser_p.formula)
        self.assertIn("Holm", denoiser_p.definition)
        self.assertIn("identical target roster", detector_p.definition)
        self.assertIn("configurable reference", configurable_p.definition)

        rmse = column_definition("RMSE ± SD (ms)")
        beat_f1 = column_definition("F1 ± SD (%)")
        rmse_p = column_definition("RMSE P versus identity")
        self.assertIn("between-participant sample SD", rmse.definition)
        self.assertIn("Participant-macro", beat_f1.definition)
        self.assertIn("Holm-adjusted", rmse_p.definition)

    def test_markdown_and_html_blocks_document_every_displayed_column(self) -> None:
        columns = (
            "classifier_id",
            ("balanced_accuracy_mean", "balanced_accuracy_sample_sd", True),
            "macro_roc_auc_ovr",
        )
        labels = ("Classifier", "BA mean +/- SD (%)", "Macro ROC-AUC")

        markdown = markdown_column_definitions_block(
            columns,
            display_labels=labels,
        )
        html = html_column_definitions_block(columns, display_labels=labels)

        self.assertIn("<summary>Column definitions and formulas</summary>", markdown)
        self.assertIn("**Classifier**", markdown)
        self.assertIn("BA mean +/- SD (%)", markdown)
        self.assertEqual(markdown.count("Formula:"), len(columns))
        self.assertIn('class="column-definitions"', html)
        self.assertIn("<strong>Macro ROC-AUC</strong>", html)
        self.assertEqual(html.count("<em>Formula:</em>"), len(columns))

    def test_report_table_catalog_uses_actual_compact_output_columns(self) -> None:
        rows = table_column_definition_rows(
            (
                ReportTable(
                    name="metric_distribution_summary",
                    rows=(
                        {
                            "case_id": "case-a",
                            "metric": "balanced_accuracy",
                            "mean": 0.7,
                            "sample_sd": 0.1,
                            "ci95_low": 0.5,
                            "ci95_high": 0.9,
                        },
                    ),
                    description="Repeat metric distribution",
                    compact=True,
                ),
                ReportTable(
                    name="table_column_definitions",
                    rows=({"column_name": "must_not_recurse"},),
                ),
            )
        )

        fields = {row["column_name"] for row in rows}
        self.assertEqual(fields, {"case_id", "metric", "mean_sd", "ci95"})
        self.assertEqual({row["table_name"] for row in rows}, {"metric_distribution_summary"})
        self.assertTrue(all(row["table_description"] for row in rows))

    def test_csv_catalog_writer_excludes_itself_and_nested_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tables = Path(temporary) / "tables"
            tables.mkdir()
            write_csv(
                tables / "classifier_results.csv",
                (
                    {
                        "classifier_id": "cnn",
                        "balanced_accuracy": 0.75,
                        "macro_f1": 0.70,
                    },
                ),
            )
            nested = tables / "nested"
            nested.mkdir()
            write_csv(nested / "ignored.csv", ({"secret_column": 1},))
            write_csv(
                tables / "table_column_definitions.csv",
                ({"column_name": "stale_self_row"},),
            )

            prewrite_rows = table_column_definition_rows_from_csv_directory(tables)
            self.assertEqual(
                {row["column_name"] for row in prewrite_rows},
                {"classifier_id", "balanced_accuracy", "macro_f1"},
            )
            csv_path, json_path, markdown_path = write_table_column_definitions(
                tables,
                csv_directory=tables,
            )

            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 3)
            self.assertNotIn(
                "table_column_definitions",
                {row["table_name"] for row in payload},
            )
            self.assertNotIn("secret_column", {row["column_name"] for row in payload})
            with csv_path.open(encoding="utf-8", newline="") as stream:
                persisted = list(csv.DictReader(stream))
            self.assertEqual(len(persisted), len(payload))
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("## `classifier_results`", markdown)
            self.assertIn("excludes itself", markdown)

            # Rebuilding remains stable: the newly written catalog is still
            # excluded and cannot recursively document its own columns.
            _, second_json, _ = write_table_column_definitions(
                tables,
                csv_directory=tables,
            )
            second_payload = json.loads(second_json.read_text(encoding="utf-8"))
            self.assertEqual(second_payload, payload)

    def test_definition_input_validation_rejects_misaligned_or_invalid_columns(self) -> None:
        with self.assertRaises(ValueError):
            column_definition_rows(("case_id",), display_labels=("Case", "Extra"))
        with self.assertRaises(ValueError):
            column_definition("")
        with self.assertRaises(TypeError):
            column_definition(("mean", "sample_sd", "yes"))  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_table_column_definitions(Path(temporary))
            with self.assertRaises(ValueError):
                write_table_column_definitions(
                    Path(temporary),
                    tables=(),
                    csv_directory=Path(temporary),
                )


if __name__ == "__main__":
    unittest.main()
