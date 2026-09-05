"""Fail-closed reporting contracts for interrupted ordinary/hyper studies."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import yaml

import frailty_3class_sweep_v2
import hyperparameter_studies_v2
from ppg_frailty.reporting.incomplete import (
    _declared_comparison_pairs,
    _html_table,
    _markdown_table,
    generate_incomplete_study_report,
)


def _event(event: str, **values: object) -> dict[str, object]:
    return {
        "event": event,
        "current": 0,
        "total": 0,
        "case_id": None,
        "repeat": None,
        "fold": None,
        "message": "",
        "timestamp_utc": "2026-08-23T00:00:00.000+00:00",
        **values,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _workbook_sheet_names(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
    return re.findall(r'<sheet name="([^"]+)"', workbook)


class IncompleteStudyReporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _ordinary_study(self) -> Path:
        root = self.root / "ordinary_interrupted"
        root.mkdir()
        plan = {
            "schema_version": "ppg_frailty.study_plan.v2",
            "study": {
                "study_id": "ordinary_interrupted",
                "kind": "catalog_sweep",
                "decision_role": "ablation",
                "reference_case_id": "control",
            },
            "cases": [
                {
                    "case_id": "control",
                    "model_id": "LogisticRegressionL2",
                    "representation_mode": "feature_vector",
                },
                {
                    "case_id": "route_b",
                    "model_id": "CompactCNN1D",
                    "representation_mode": "raw",
                },
            ],
            "execution": {"repeats": [0, 1], "folds": [0, 1]},
        }
        (root / "study_plan.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False), encoding="utf-8"
        )
        events = [_event("study_started", total=2)]
        for repeat in (0, 1):
            for fold in (0, 1):
                events.extend(
                    (
                        _event(
                            "cell_start",
                            case_id="control",
                            repeat=repeat,
                            fold=fold,
                            total=4,
                        ),
                        _event(
                            "cell_complete",
                            case_id="control",
                            repeat=repeat,
                            fold=fold,
                            total=4,
                        ),
                    )
                )
        events.append(_event("case_finished", case_id="control", message="passed"))
        events.extend(
            (
                _event("cell_start", case_id="route_b", repeat=0, fold=0, total=4),
                _event(
                    "cell_complete",
                    case_id="route_b",
                    repeat=0,
                    fold=0,
                    total=4,
                    message="failed_closed: synthetic routing error",
                ),
                _event("cell_start", case_id="route_b", repeat=0, fold=1, total=4),
            )
        )
        _write_jsonl(root / "progress_events.jsonl", events)
        case_dir = root / "raw" / "control"
        case_dir.mkdir(parents=True)
        (case_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": "test.pipeline.v2",
                    "config_id": "control",
                    "manifest": {
                        "source_dataset_id": "fixture_dataset",
                        "path": "inputs/fixture_manifest.csv",
                        "expected_participant_count": 2,
                        "expected_record_count": 4,
                        "channel_order": [
                            "RED",
                            "IR",
                            "AX",
                            "AY",
                            "AZ",
                            "GX",
                            "GY",
                            "GZ",
                        ],
                    },
                    "splits": {"registry_id": "fixture_participant_grouped_2fold"},
                    "representation_mode": "feature_vector",
                    "model": {
                        "model_id": "LogisticRegressionL2",
                        "architecture_parameters": {
                            "estimator": "sklearn.linear_model.LogisticRegression",
                            "penalty": "l2",
                            "audit_blob": "x" * 140_000,
                        },
                    },
                    "training": {},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (case_dir / "case_result.json").write_text(
            json.dumps(
                {
                    "case_id": "control",
                    "status": "passed",
                    "result": {
                        "cell_results": [
                            {
                                "repeat_index": repeat,
                                "fold_index": fold,
                                "status": "passed",
                                "metrics": {"balanced_accuracy": 0.99},
                            }
                            for repeat in (0, 1)
                            for fold in (0, 1)
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        tables = root / "tables"
        tables.mkdir()
        (tables / "controlled_parameters.csv").write_text(
            "parameter,value\ntraining.epochs,10\n", encoding="utf-8"
        )
        (tables / "varied_parameters.csv").write_text(
            "parameter,control,route_b\nrepresentation_mode,feature_vector,raw\n",
            encoding="utf-8",
        )
        return root

    def test_explicit_report_table_fields_are_a_strict_projection(self) -> None:
        rows = ({"kept": None, "extra": "must not leak"},)

        markdown = _markdown_table(rows, fields=("kept",))
        self.assertIn("| kept |", markdown)
        self.assertIn("| N/A |", markdown)
        self.assertNotIn("extra", markdown)
        self.assertNotIn("must not leak", markdown)

        html = _html_table(rows, fields=("kept",))
        self.assertIn("<th>kept</th>", html)
        self.assertIn("<td>N/A</td>", html)
        self.assertNotIn("extra", html)
        self.assertNotIn("must not leak", html)

        wide_fields = tuple(f"field_{index}" for index in range(9))
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            _markdown_table(({},), fields=wide_fields)
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            _html_table(({},), fields=wide_fields)

    def _hyper_study(self) -> Path:
        root = self.root / "hyper_interrupted"
        root.mkdir()
        plan = {
            "schema_version": "ppg_frailty.hyperparameter_study_plan.v1",
            "study": {
                "study_id": "hyper_interrupted",
                "study_type": "dependent_regularization_grid",
            },
            "base": {
                "catalog_entry": "inception_full",
                "output_group": "raw",
                "common_overrides": {
                    "signal.dl_resampling.enabled": True,
                    "signal.dl_resampling.target_fs_hz": 64.0,
                    "windows.raw_dl.length_s": 5.0,
                    "windows.raw_dl.hop_s": 2.5,
                    "training.optimizer": "adamw",
                    "aggregation.balance_line": "line_b_equal_role_families",
                },
            },
            "candidates": [
                {
                    "case_id": "r1",
                    "overrides": {"training.weight_decay": 0.001},
                },
                {
                    "case_id": "r2",
                    "overrides": {"training.weight_decay": 0.005},
                },
            ],
            "execution": {"jobs": 1, "device": "cuda"},
            "resource": {"repeats": [0, 1], "folds": [0, 1]},
        }
        (root / "study_plan.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False), encoding="utf-8"
        )
        nested = root / "phases" / "full_cv" / "nested"
        _write_jsonl(
            nested / "progress_events.jsonl",
            [
                _event("study_started", total=2),
                _event("cell_start", case_id="r1", repeat=0, fold=0, total=4),
                _event("cell_complete", case_id="r1", repeat=0, fold=0, total=4),
                _event("cell_start", case_id="r1", repeat=0, fold=1, total=4),
            ],
        )
        tables = root / "tables"
        tables.mkdir()
        (tables / "screening_audit.csv").write_text(
            "phase,status\nfull_cv,interrupted\n", encoding="utf-8"
        )
        return root

    def test_execution_only_report_is_fail_closed_and_complete(self) -> None:
        root = self._ordinary_study()
        result = generate_incomplete_study_report(root)

        self.assertFalse((root / "study_manifest.json").exists())
        self.assertEqual(result.status, "incomplete_failed_and_interrupted")
        summary = json.loads(
            (root / "tables/execution_completeness.json").read_text(encoding="utf-8")
        )[0]
        self.assertEqual(summary["planned_case_count"], 2)
        self.assertEqual(summary["complete_case_count"], 1)
        self.assertEqual(summary["planned_cell_count"], 8)
        self.assertEqual(summary["passed_cell_count"], 4)
        self.assertEqual(summary["failed_closed_cell_count"], 1)
        self.assertEqual(summary["started_without_terminal_event_cell_count"], 1)
        self.assertEqual(summary["not_started_cell_count"], 2)
        self.assertEqual(summary["declared_classifier_count"], 2)
        self.assertEqual(summary["declared_pairwise_comparison_count"], 1)
        for field in (
            "formal_result_available",
            "ranking_eligible",
            "inference_eligible",
            "selection_eligible",
        ):
            self.assertFalse(summary[field])

        incomplete = json.loads(
            (root / "tables/incomplete_cases.json").read_text(encoding="utf-8")
        )
        self.assertEqual([row["case_id"] for row in incomplete], ["route_b"])
        failures = json.loads(
            (root / "tables/failure_events.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {row["classification"] for row in failures},
            {
                "failed_closed_cell",
                "started_without_terminal_event",
                "study_interruption",
            },
        )
        self.assertIn(
            "No formal model/module comparison",
            result.interpretation_markdown.read_text(),
        )
        self.assertIn("never synthesizes", result.methods_markdown.read_text())
        # Metric names can appear in planned reporter requirements, but the
        # persisted case-result score must never be promoted into this report.
        self.assertNotIn("0.99", result.summary_markdown.read_text())

        per_class = json.loads(
            (root / "tables/classifier_per_class_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(per_class), 6)
        self.assertEqual(
            {
                (row["classifier_id"], row["class_label"], row["class_name"])
                for row in per_class
            },
            {
                (classifier_id, class_label, class_name)
                for classifier_id in ("control", "route_b")
                for class_label, class_name in (
                    (0, "Pre-Frail"),
                    (1, "Robust/Non-Frail"),
                    (2, "Young"),
                )
            },
        )
        for row in per_class:
            for field in (
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
                "balanced_accuracy_ovr",
                "f1",
                "roc_auc_ovr",
            ):
                self.assertIsNone(row[field])
            self.assertEqual(
                row["result_applicability"],
                "N/A_incomplete_study_no_formal_classifier_result",
            )

        repeat_deltas = json.loads(
            (root / "tables/pairwise_repeat_metric_deltas.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(repeat_deltas), 2)
        self.assertEqual({row["repeat"] for row in repeat_deltas}, {0, 1})
        self.assertEqual(
            {
                (row["reference_case_id"], row["candidate_case_id"])
                for row in repeat_deltas
            },
            {("control", "route_b")},
        )
        for row in repeat_deltas:
            for metric in (
                "balanced_accuracy",
                "macro_f1",
                "macro_roc_auc_ovr",
            ):
                self.assertIsNone(row[f"reference_{metric}"])
                self.assertIsNone(row[f"candidate_{metric}"])
                self.assertIsNone(row[f"{metric}_delta"])
            self.assertIn("no_formal_root_manifest", row["unavailable_reason"])

        inference = json.loads(
            (root / "tables/paired_participant_inference.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(inference), 3)
        self.assertEqual(
            {row["metric"] for row in inference},
            {"balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"},
        )
        for row in inference:
            self.assertIsNone(row["candidate_minus_reference"])
            self.assertIsNone(row["participant_cluster_delta_ci95_low"])
            self.assertIsNone(row["participant_cluster_delta_ci95_high"])
            self.assertIn("no_formal_root_manifest", row["unavailable_reason"])

        definitions = json.loads(
            (root / "tables/table_column_definitions.json").read_text(encoding="utf-8")
        )
        for table_name in (
            "classifier_per_class_results",
            "pairwise_repeat_metric_deltas",
            "paired_participant_inference",
        ):
            selected = [row for row in definitions if row["table_name"] == table_name]
            self.assertTrue(selected)
            self.assertTrue(all(row["definition"] for row in selected))
            self.assertTrue(all(row["formula"] for row in selected))
        summary_markdown = result.summary_markdown.read_text(encoding="utf-8")
        self.assertIn("Classifier per-class results (explicit N/A)", summary_markdown)
        self.assertIn("Pairwise per-repeat metric differences", summary_markdown)
        for heading in (
            "Study identity and terminal state",
            "Formal-evidence eligibility",
            "Declared case and reporter inventory",
            "Declared comparisons and progress evidence",
            "Fold-cell execution counts",
            "Participation, state, and reporter binding",
            "Input data and fixed parameters",
            "Algorithm kernel and literature",
            "Execution-evidence interpretation",
            "Profile identity and planned components",
            "Required outputs",
            "Methods, limitations, and provenance",
            "Declared class roster and applicability",
            "Per-class confusion counts",
            "Per-class observation coverage",
            "Per-class thresholded metrics",
            "Per-class probability metrics",
            "Declared comparison contracts",
            "Repeat and matched-roster audit",
            "Repeat metric differences",
            "Repeat metric applicability and interpretation",
            "Inference contracts and applicability",
            "Effect estimates and participant-cluster intervals",
            "Inference sample support and exchange unit",
            "Bootstrap contract",
            "Bootstrap method and applicability",
            "Paired P values and Holm adjustment",
            "Multiplicity and selection audit",
            "Permutation and interpretation audit",
            "Case status and terminal evidence",
            "Case fold-cell counts",
            "Case formal-evidence eligibility",
            "Failure event identities",
            "Failure sources and messages",
        ):
            self.assertIn(f"### {heading}", summary_markdown)
        markdown_headers = [
            line
            for line, following in zip(
                summary_markdown.splitlines(),
                summary_markdown.splitlines()[1:],
            )
            if line.startswith("| ") and following.startswith("|---")
        ]
        self.assertTrue(markdown_headers)
        self.assertTrue(
            all(len(header.split("|")[1:-1]) <= 8 for header in markdown_headers)
        )
        pairwise_block = summary_markdown.split(
            "## Pairwise per-repeat metric differences (explicit N/A)", 1
        )[1].split("## Incomplete cases", 1)[0]
        pairwise_headers = [
            line
            for line, following in zip(
                pairwise_block.splitlines(), pairwise_block.splitlines()[1:]
            )
            if line.startswith("| ") and following.startswith("|---")
        ]
        self.assertTrue(pairwise_headers)
        self.assertTrue(
            all(
                header.startswith("| candidate_case_id |")
                for header in pairwise_headers
            )
        )
        self.assertNotIn(
            "prediction_rule_source | result_applicability", summary_markdown
        )
        summary_html = (root / "STUDY_SUMMARY.html").read_text(encoding="utf-8")
        html_header_rows = re.findall(r"<thead><tr>(.*?)</tr></thead>", summary_html)
        self.assertTrue(html_header_rows)
        self.assertTrue(all(header.count("<th>") <= 8 for header in html_header_rows))
        self.assertIn(
            "participant IDs with replacement", result.methods_markdown.read_text()
        )

        # Human tables are projections only: the CSV/JSON audit remains lossless.
        self.assertIn("prediction_rule_source", per_class[0])
        self.assertIn("matched_roster_sha256", repeat_deltas[0])
        self.assertIn("interpretation", inference[0])

        components = json.loads(
            (root / "tables/test_components.json").read_text(encoding="utf-8")
        )
        control_model = next(
            row
            for row in components
            if row["participating_cases"] == "control"
            and row["component_role"] == "classifier"
        )
        self.assertEqual(control_model["module_id"], "LogisticRegressionL2")
        self.assertEqual(control_model["execution_state"], "complete")
        self.assertEqual(control_model["configured_state"], "enabled")
        self.assertEqual(
            json.loads(control_model["input_data"])["dataset_id"], "fixture_dataset"
        )
        self.assertEqual(
            json.loads(control_model["fixed_parameters"])["model_id"],
            "LogisticRegressionL2",
        )
        self.assertEqual(
            len(
                json.loads(control_model["fixed_parameters"])[
                    "architecture_parameters"
                ]["audit_blob"]
            ),
            140_000,
        )
        route_model = next(
            row
            for row in components
            if row["participating_cases"] == "route_b"
            and row["component_role"] == "classifier"
        )
        self.assertEqual(route_model["module_id"], "CompactCNN1D")
        self.assertEqual(route_model["execution_state"], "partial")
        self.assertEqual(json.loads(route_model["input_data"])["availability"], "N/A")
        self.assertNotIn("executed", {row["execution_state"] for row in components})

        profiles = json.loads(
            (root / "tables/reporter_profiles.json").read_text(encoding="utf-8")
        )
        profile_ids = {row["profile_id"] for row in profiles}
        self.assertIn("logistic_l2_model_v1", profile_ids)
        self.assertIn("compactcnn_model_v1", profile_ids)
        methods_text = result.methods_markdown.read_text(encoding="utf-8")
        self.assertIn("Hastie, Tibshirani & Friedman", methods_text)
        self.assertIn("Execution states: `complete`", methods_text)
        self.assertIn(
            "route_b:classifier:CompactCNN1D — Project CompactCNN1D",
            methods_text,
        )
        component_markdown = (root / "TEST_COMPONENTS.md").read_text(encoding="utf-8")
        component_headers = [
            line
            for line, following in zip(
                component_markdown.splitlines(),
                component_markdown.splitlines()[1:],
            )
            if line.startswith("| ") and following.startswith("|---")
        ]
        self.assertTrue(component_headers)
        self.assertTrue(
            all(len(header.split("|")[1:-1]) <= 8 for header in component_headers)
        )
        self.assertTrue(
            all(header.startswith("| module_id |") for header in component_headers)
        )
        component_table = component_markdown[component_markdown.index("| ") :]
        self.assertIn(component_table.strip(), result.summary_markdown.read_text())

        csv_stems = {path.stem for path in (root / "tables").glob("*.csv")}
        sheet_names = _workbook_sheet_names(root / "tables/report_tables.xlsx")
        self.assertEqual(set(sheet_names), csv_stems)
        self.assertEqual(len(sheet_names), len(csv_stems))

        output_index = json.loads(result.outputs_index.read_text(encoding="utf-8"))
        self.assertEqual(
            output_index["inventory_scope"],
            "incomplete_report_inputs_and_outputs_only",
        )
        indexed_paths = {row["path"] for row in output_index["artifacts"]}
        self.assertIn("TEST_COMPONENTS.md", indexed_paths)
        self.assertIn("tables/test_components.csv", indexed_paths)
        self.assertIn("tables/test_components.json", indexed_paths)
        self.assertIn("tables/reporter_profiles.csv", indexed_paths)
        self.assertIn("tables/reporter_profiles.json", indexed_paths)
        self.assertIn("tables/classifier_per_class_results.csv", indexed_paths)
        self.assertIn("tables/classifier_per_class_results.json", indexed_paths)
        self.assertIn("tables/pairwise_repeat_metric_deltas.csv", indexed_paths)
        self.assertIn("tables/pairwise_repeat_metric_deltas.json", indexed_paths)
        self.assertIn("tables/paired_participant_inference.csv", indexed_paths)
        self.assertIn("tables/paired_participant_inference.json", indexed_paths)
        self.assertIn("tables/controlled_parameters.csv", indexed_paths)
        self.assertIn("tables/varied_parameters.csv", indexed_paths)
        self.assertIn("raw/control/resolved_config.yaml", indexed_paths)
        for row in output_index["artifacts"]:
            if row["path"] == "outputs_index.json":
                self.assertIsNone(row["sha256"])
                continue
            path = root / row["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"]
            )

    def test_ordinary_and_hyper_report_commands_dispatch_without_manifest(self) -> None:
        ordinary = self._ordinary_study()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(
                frailty_3class_sweep_v2.main(["report", "--study-dir", str(ordinary)]),
                0,
            )
        self.assertTrue((ordinary / "STUDY_SUMMARY.md").is_file())
        self.assertFalse((ordinary / "study_manifest.json").exists())

        hyper = self._hyper_study()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                hyperparameter_studies_v2.main(["report", "--study-dir", str(hyper)]),
                0,
            )
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["status"], "incomplete_report_regenerated")
        summary = json.loads(
            (hyper / "tables/execution_completeness.json").read_text(encoding="utf-8")
        )[0]
        self.assertEqual(summary["planned_case_count"], 2)
        self.assertEqual(summary["planned_cell_count"], 8)
        self.assertEqual(summary["passed_cell_count"], 1)
        self.assertEqual(summary["started_without_terminal_event_cell_count"], 1)
        self.assertEqual(summary["not_started_cell_count"], 6)
        self.assertEqual(summary["declared_classifier_count"], 2)
        self.assertEqual(summary["declared_pairwise_comparison_count"], 0)
        self.assertFalse((hyper / "study_manifest.json").exists())

        components = json.loads(
            (hyper / "tables/test_components.json").read_text(encoding="utf-8")
        )
        model_rows = [
            row for row in components if row["component_role"] == "classifier"
        ]
        self.assertEqual(
            {row["module_id"] for row in model_rows}, {"InceptionTimeFull"}
        )
        self.assertEqual(
            {row["execution_state"] for row in model_rows}, {"partial", "not_started"}
        )
        self.assertTrue(
            all(
                json.loads(row["input_data"])["availability"] == "N/A"
                for row in model_rows
            )
        )
        self.assertEqual(
            {
                json.loads(row["fixed_parameters"])["effective_declared_overrides"][
                    "training.weight_decay"
                ]
                for row in model_rows
            },
            {0.001, 0.005},
        )
        profiles = json.loads(
            (hyper / "tables/reporter_profiles.json").read_text(encoding="utf-8")
        )
        inception_profile = next(
            row
            for row in profiles
            if row["profile_id"] == "inceptiontime_single_network_model_v1"
        )
        self.assertEqual(
            set(inception_profile["execution_states"]), {"partial", "not_started"}
        )
        per_class = json.loads(
            (hyper / "tables/classifier_per_class_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(per_class), 6)
        self.assertEqual(
            {row["classifier_id"] for row in per_class},
            {"r1", "r2"},
        )
        self.assertEqual(
            json.loads(
                (hyper / "tables/pairwise_repeat_metric_deltas.json").read_text(
                    encoding="utf-8"
                )
            ),
            [],
        )
        self.assertEqual(
            json.loads(
                (hyper / "tables/paired_participant_inference.json").read_text(
                    encoding="utf-8"
                )
            ),
            [],
        )
        self.assertIn(
            "reference_balanced_accuracy",
            (hyper / "tables/pairwise_repeat_metric_deltas.csv")
            .read_text(encoding="utf-8")
            .splitlines()[0],
        )
        self.assertIn(
            "participant_cluster_delta_ci95_low",
            (hyper / "tables/paired_participant_inference.csv")
            .read_text(encoding="utf-8")
            .splitlines()[0],
        )
        methods = (hyper / "REPORT_METHODS.md").read_text(encoding="utf-8")
        self.assertIn("Fawaz et al. (2020)", methods)
        csv_stems = {path.stem for path in (hyper / "tables").glob("*.csv")}
        sheet_names = _workbook_sheet_names(hyper / "tables/report_tables.xlsx")
        self.assertEqual(set(sheet_names), csv_stems)
        self.assertEqual(len(sheet_names), len(csv_stems))
        indexed_paths = {
            row["path"]
            for row in json.loads(
                (hyper / "outputs_index.json").read_text(encoding="utf-8")
            )["artifacts"]
        }
        self.assertIn("tables/screening_audit.csv", indexed_paths)
        self.assertIn("tables/test_components.csv", indexed_paths)
        self.assertIn("tables/reporter_profiles.csv", indexed_paths)

    def test_reporter_refuses_to_replace_formal_manifest_path(self) -> None:
        root = self._ordinary_study()
        (root / "study_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "formal reporter"):
            generate_incomplete_study_report(root)

    def test_unmaterialized_centered_star_uses_within_model_declared_pairs(
        self,
    ) -> None:
        plan = {
            "study": {
                "study_id": "centered_fixture",
                "decision_role": "ablation",
                # This ordinary reference must not create forbidden cross-model
                # pairs when the more specific centered-star contract exists.
                "reference_case_id": "compact_b0",
            },
            "cases": [
                {"case_id": "compact_b0"},
                {"case_id": "inception_b0"},
                {"case_id": "compact_b1"},
                {"case_id": "inception_b1"},
            ],
            "legacy_bridge": {
                "design": "centered_star_v1",
                "profiles": [
                    {
                        "case_id": "compact_B0",
                        "catalog_case_id": "compact_b0",
                        "model_id": "CompactCNN1D",
                        "profile_id": "B0",
                    },
                    {
                        "case_id": "inception_B0",
                        "catalog_case_id": "inception_b0",
                        "model_id": "InceptionTimeFull",
                        "profile_id": "B0",
                    },
                    {
                        "case_id": "compact_B1",
                        "catalog_case_id": "compact_b1",
                        "model_id": "CompactCNN1D",
                        "profile_id": "B1",
                    },
                    {
                        "case_id": "inception_B1",
                        "catalog_case_id": "inception_b1",
                        "model_id": "InceptionTimeFull",
                        "profile_id": "B1",
                    },
                ],
            },
        }
        pairs = _declared_comparison_pairs(
            plan,
            ("compact_b0", "inception_b0", "compact_b1", "inception_b1"),
        )
        self.assertEqual(
            {(row["reference_case_id"], row["candidate_case_id"]) for row in pairs},
            {
                ("compact_b0", "compact_b1"),
                ("inception_b0", "inception_b1"),
            },
        )

    def test_formal_manifest_keeps_existing_runner_dispatch(self) -> None:
        root = self._ordinary_study()
        (root / "study_manifest.json").write_text("{}", encoding="utf-8")
        with patch.object(
            frailty_3class_sweep_v2,
            "generate_study_report",
            return_value=SimpleNamespace(summary_markdown=root / "formal.md"),
        ) as ordinary_report, contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(
                frailty_3class_sweep_v2.main(["report", "--study-dir", str(root)]),
                0,
            )
        ordinary_report.assert_called_once_with(root.resolve())

        with patch.object(
            hyperparameter_studies_v2,
            "regenerate_hyperparameter_report",
            return_value={"status": "formal_regenerated"},
        ) as hyper_report, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                hyperparameter_studies_v2.main(["report", "--study-dir", str(root)]),
                0,
            )
        hyper_report.assert_called_once_with(root.resolve())


if __name__ == "__main__":
    unittest.main()
