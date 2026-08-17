"""Focused expansion, fake-execution, resume, progress, and report tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from ppg_frailty.reporting import generate_study_report
from ppg_frailty.reporting.analyze import analyze_study
from ppg_frailty.reporting.collect import CollectedStudy
from ppg_frailty.study import (
    NullProgressSink,
    ProgressEvent,
    StudyRunner,
    parse_study_plan,
    validate_canonical_expansion,
)


def fake_executor(case, config_path, case_directory, plan, progress_sink):
    del config_path, case_directory, progress_sink
    value = float(case.changed_values.get("training.learning_rate", 0.001))
    score = 0.62 + value * 10.0
    cells: list[dict[str, Any]] = []
    for repeat in plan.execution.repeats:
        for fold in plan.execution.folds:
            current = score + repeat * 0.002 - fold * 0.001
            cells.append(
                {
                    "status": "passed",
                    "repeat_index": repeat,
                    "fold_index": fold,
                    "split_seed": [42, 10042, 20042, 30042, 40042][repeat],
                    "training_seed": [42, 10042, 20042, 30042, 40042][repeat],
                    "metrics": {
                        "balanced_accuracy": current,
                        "macro_f1": current - 0.02,
                        "expected_calibration_error": 0.08,
                        "coverage_rate": 0.95,
                        "confusion_matrix": [[3, 1, 0], [1, 3, 0], [0, 1, 3]],
                        "class_order": [0, 1, 2],
                    },
                    "training_history": [
                        {"epoch": 1, "loss": 1.0, "val_loss": 1.1},
                        {"epoch": 2, "loss": 0.8, "val_loss": 0.9},
                    ],
                }
            )
    return {"status": "passed", "cell_results": cells}


def failed_closed_executor(case, config_path, case_directory, plan, progress_sink):
    del config_path, case_directory, plan, progress_sink
    return {
        "schema_version": "ppg_frailty.experiment_result.v2",
        "status": "failed_closed",
        "config_id": case.config["config_id"],
        "cell_results": [],
        "failure_reasons": ["synthetic_protocol_failure"],
    }


class StudyProductTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.yaml"
        self.base.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "test.pipeline.v2",
                    "config_id": "fake_reference",
                    "model": {
                        "model_id": "LogisticRegressionL2",
                        "ensemble_size": 1,
                    },
                    "training": {
                        "learning_rate": 0.001,
                        "weight_decay": 0.0001,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def plan(self, *, ensemble: bool = False):
        if ensemble:
            payload = yaml.safe_load(self.base.read_text(encoding="utf-8"))
            payload["model"]["ensemble_size"] = 5
            self.base.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
        return parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "learning_rate_ablation",
                    "kind": "ablation",
                    "purpose": "Synthetic orchestration test only.",
                    "flow_position": "Product-layer unit test.",
                    "decision_role": "ablation",
                },
                "base_config": str(self.base),
                "axes": [
                    {
                        "path": "training.learning_rate",
                        "values": [0.0003, 0.001],
                        "reference": 0.001,
                    }
                ],
                "execution": {
                    "repeats": [0, 1],
                    "folds": [0, 1],
                    "jobs": 2,
                },
                "output": {"root": str(self.root / "outputs")},
                "report": {
                    "top_k": 10,
                    "write_html": True,
                    "write_static_figures": True,
                    "calibration_bins": 5,
                },
            }
        )

    def grid_plan(self):
        return parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "synthetic_two_axis_grid",
                    "kind": "grid",
                    "purpose": "Synthetic descriptive grid test only.",
                    "flow_position": "Product-layer unit test.",
                    "decision_role": "screening",
                },
                "base_config": str(self.base),
                "axes": [
                    {
                        "path": "training.learning_rate",
                        "values": [0.0003, 0.001],
                        "reference": 0.001,
                    },
                    {
                        "path": "training.weight_decay",
                        "values": [0.0001, 0.001],
                        "reference": 0.0001,
                    },
                ],
                "execution": {
                    "repeats": [0, 1],
                    "folds": [0, 1],
                    "jobs": 2,
                },
                "output": {"root": str(self.root / "outputs")},
                "report": {"write_static_figures": True},
            }
        )

    def test_reference_equal_to_base_is_valid_and_controls_are_explicit(self) -> None:
        runner = StudyRunner(pipeline_root=self.root, executor=fake_executor)
        expansion = runner.expand(self.plan())
        self.assertEqual(len(expansion.cases), 2)
        self.assertIsNotNone(expansion.reference_case_id)
        varied = {row["parameter_path"] for row in expansion.varied_parameters}
        self.assertEqual(varied, {"training.learning_rate"})
        controlled = {
            row["parameter_path"]: row["value"]
            for row in expansion.controlled_parameters
        }
        self.assertEqual(controlled["model.model_id"], "LogisticRegressionL2")

    def test_canonical_dry_run_validation_rejects_invalid_override(self) -> None:
        pipeline_root = Path(__file__).resolve().parents[2]
        canonical = pipeline_root / "configs" / "reference_static_role_aware_v2.yaml"
        plan = parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "invalid_fixed_epoch_dry_run",
                    "kind": "ablation",
                    "purpose": "Dry-run contract regression.",
                    "flow_position": "No execution.",
                    "decision_role": "ablation",
                },
                "base_config": str(canonical),
                "axes": [
                    {
                        "path": "training.fixed_epochs",
                        "values": [10, "bad"],
                        "reference": 10,
                    }
                ],
                "execution": {"repeats": [0], "folds": [0], "jobs": 1},
            }
        )
        expansion = StudyRunner(pipeline_root=pipeline_root).expand(plan)
        with self.assertRaisesRegex(ValueError, "not a valid canonical"):
            validate_canonical_expansion(expansion)

    def test_fixed_epoch_axis_updates_its_canonical_profile(self) -> None:
        pipeline_root = Path(__file__).resolve().parents[2]
        canonical = pipeline_root / "configs" / "reference_static_role_aware_v2.yaml"
        plan = parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "fixed_epoch_dry_run",
                    "kind": "ablation",
                    "purpose": "Dry-run contract regression.",
                    "flow_position": "No execution.",
                    "decision_role": "ablation",
                },
                "base_config": str(canonical),
                "axes": [
                    {
                        "path": "training.fixed_epochs",
                        "values": [7, 10, 15],
                        "reference": 10,
                    }
                ],
                "execution": {"repeats": [0], "folds": [0], "jobs": 1},
            }
        )
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=pipeline_root).expand(plan)
        )
        self.assertEqual(
            {row["parameter_path"] for row in expansion.varied_parameters},
            {"training.fixed_epochs"},
        )
        self.assertEqual(
            {
                (
                    case.config["training"]["fixed_epochs"],
                    case.config["training"]["epoch_profile"],
                )
                for case in expansion.cases
            },
            {(7, "ablation_7"), (10, "default_10"), (15, "ablation_15")},
        )

    def test_fake_parallel_run_report_and_resume(self) -> None:
        plan = self.plan()
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=fake_executor,
            progress_sink=NullProgressSink(),
        )
        result = runner.run(plan)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.effective_jobs, 2)
        self.assertEqual(
            (
                result.planned_cell_count,
                result.reported_cell_count,
                result.passed_cell_count,
                result.failed_cell_count,
                result.not_run_cell_count,
            ),
            (8, 8, 8, 0, 0),
        )
        self.assertIn("ablation_training-learning-rate", result.output_directory.name)
        self.assertTrue((result.output_directory / "progress_events.jsonl").is_file())
        report = generate_study_report(result.output_directory)
        self.assertTrue(report.summary_markdown.is_file())
        self.assertTrue(report.output_index.is_file())
        summary = report.summary_markdown.read_text(encoding="utf-8")
        self.assertIn("Predictive ranking", summary)
        self.assertIn("Deployment measurements", summary)
        self.assertIn("Macro-F1 LCB95", summary)
        index = json.loads(report.output_index.read_text(encoding="utf-8"))
        paths = {row["path"] for row in index["artifacts"]}
        self.assertEqual(
            index["inventory_scope"],
            "all_regular_files_below_study_directory",
        )
        self.assertIn("study_plan.yaml", paths)
        self.assertIn("study_manifest.json", paths)
        self.assertIn("study_run_result.json", paths)
        self.assertIn("progress_events.jsonl", paths)
        self.assertTrue(any(path.startswith("resolved_configs/") for path in paths))
        self.assertTrue(any(path.startswith("cases/") for path in paths))
        self.assertIn("tables/predictive_leaderboard.csv", paths)
        self.assertIn("tables/metric_distribution_summary.csv", paths)
        self.assertIn("tables/worst_class_f1_stability.csv", paths)
        self.assertIn("tables/incomplete_cases.csv", paths)
        self.assertIn("tables/confusion_counts.csv", paths)
        self.assertIn("tables/confusion_row_normalized.csv", paths)
        self.assertEqual(
            len(
                [
                    path
                    for path in paths
                    if path.startswith("tables/top_confusion_matrices/")
                    and path.endswith(".csv")
                ]
            ),
            4,
        )
        self.assertIn("figures/plot_status.json", paths)
        hashed = [
            row
            for row in index["artifacts"]
            if row["path"] != "outputs_index.json"
        ]
        self.assertTrue(hashed)
        self.assertTrue(all(len(str(row["sha256"])) == 64 for row in hashed))
        plot_status = json.loads(
            (result.output_directory / "figures" / "plot_status.json").read_text(
                encoding="utf-8"
            )
        )
        statuses = {row["figure"]: row["status"] for row in plot_status}
        self.assertIn("parameter_effects", statuses)
        self.assertEqual(statuses["parameter_interaction"], "N/A")
        self.assertEqual(statuses["worst_class_f1_stability"], "generated")
        self.assertEqual(
            statuses["confusion_matrices_row_normalized"],
            "generated",
        )
        self.assertEqual(statuses["top_learning_curves"], "generated")
        distributions = json.loads(
            (
                result.output_directory
                / "tables"
                / "metric_distribution_summary.json"
            ).read_text(encoding="utf-8")
        )
        ba_rows = [
            row for row in distributions if row["metric"] == "balanced_accuracy"
        ]
        self.assertEqual(len(ba_rows), 2)
        self.assertTrue(all(row["n"] == 2 for row in ba_rows))
        self.assertTrue(all(row["ci95_high"] is not None for row in ba_rows))
        self.assertTrue(all(row["minimum"] <= row["maximum"] for row in ba_rows))
        per_class = json.loads(
            (
                result.output_directory / "tables" / "per_class_metrics.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(per_class)
        self.assertEqual(
            {row["metric_source"] for row in per_class},
            {"summed_cell_confusion_fallback"},
        )
        self.assertEqual(
            json.loads(
                (
                    result.output_directory / "tables" / "incomplete_cases.json"
                ).read_text(encoding="utf-8")
            ),
            [],
        )
        normalized = json.loads(
            (
                result.output_directory
                / "tables"
                / "confusion_row_normalized.json"
            ).read_text(encoding="utf-8")
        )
        row_sums: dict[tuple[str, int], float] = {}
        for row in normalized:
            key = (str(row["case_id"]), int(row["true_class"]))
            row_sums[key] = row_sums.get(key, 0.0) + float(row["row_fraction"])
        self.assertTrue(row_sums)
        self.assertTrue(all(abs(value - 1.0) < 1e-12 for value in row_sums.values()))

        figures = result.output_directory / "figures"
        (figures / "leaderboard.NA.txt").write_text(
            "stale marker\n",
            encoding="utf-8",
        )
        (figures / "parameter_interaction.png").write_bytes(b"stale-png")
        generate_study_report(result.output_directory)
        self.assertTrue((figures / "leaderboard.png").is_file())
        self.assertFalse((figures / "leaderboard.NA.txt").exists())
        self.assertTrue((figures / "parameter_interaction.NA.txt").is_file())
        self.assertFalse((figures / "parameter_interaction.png").exists())
        self.assertFalse(any(path.name.startswith(".") for path in figures.iterdir()))
        resumed = runner.run(plan, resume_directory=result.output_directory)
        self.assertEqual(resumed.resumed_case_count, 2)
        self.assertEqual(resumed.passed_case_count, 2)

    def test_outer_cv_ensemble_is_fail_closed(self) -> None:
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=fake_executor,
            progress_sink=NullProgressSink(),
        )
        with self.assertRaisesRegex(RuntimeError, "repeat-by-member seed matrix"):
            runner.run(self.plan(ensemble=True))

    def test_two_axis_grid_writes_descriptive_interaction_view(self) -> None:
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=fake_executor,
            progress_sink=NullProgressSink(),
        )
        result = runner.run(self.grid_plan())
        generate_study_report(result.output_directory)
        statuses = {
            row["figure"]: row
            for row in json.loads(
                (
                    result.output_directory / "figures" / "plot_status.json"
                ).read_text(encoding="utf-8")
            )
        }
        interaction = statuses["parameter_interaction"]
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.assertEqual(interaction["status"], "N/A")
            self.assertIn("matplotlib unavailable", interaction["reason"])
        else:
            self.assertEqual(interaction["status"], "generated")

    def test_returned_experiment_failed_closed_is_not_counted_as_passed(self) -> None:
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=failed_closed_executor,
            progress_sink=NullProgressSink(),
        )
        result = runner.run(self.plan())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.passed_case_count, 0)
        self.assertEqual(result.failed_case_count, 2)
        self.assertEqual(result.not_run_case_count, 0)
        self.assertEqual(result.planned_cell_count, 8)
        self.assertEqual(result.reported_cell_count, 0)
        self.assertEqual(result.not_run_cell_count, 8)
        first = result.case_records[0]
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["result"]["status"], "failed_closed")
        self.assertEqual(first["error_type"], "CanonicalExperimentFailedClosed")

    def test_incomplete_case_is_reported_and_excluded_from_ranking(self) -> None:
        bundle = CollectedStudy(
            root=self.root,
            plan={
                "execution": {"repeats": [0, 1], "folds": [0, 1]},
                "report": {},
            },
            manifest={
                "cases": [{"case_id": "incomplete", "is_reference": True}],
                "reference_case_id": "incomplete",
            },
            case_records=({"case_id": "incomplete", "status": "passed"},),
            varied_parameters=(),
            controlled_parameters=(),
            cell_rows=(
                {
                    "case_id": "incomplete",
                    "status": "passed",
                    "repeat": 0,
                    "fold": 0,
                    "balanced_accuracy": 0.75,
                    "macro_f1": 0.70,
                },
            ),
            history_rows=(),
            subject_oof_rows=(),
            role_oof_rows=(),
            quality_rows=(),
            trusted_config_metrics=(),
            limitations=(),
        )
        analysis = analyze_study(bundle)
        self.assertEqual(analysis.predictive_leaderboard, ())
        self.assertEqual(len(analysis.incomplete_cases), 1)
        row = analysis.incomplete_cases[0]
        self.assertEqual(row["case_id"], "incomplete")
        self.assertIn("repeat_metric_count=1/2", row["incompleteness_reasons"])
        self.assertIn("passed_fold_cell_count=1/4", row["incompleteness_reasons"])

    def test_failed_case_resume_uses_a_new_nonoverwriting_attempt(self) -> None:
        calls: dict[str, int] = {}

        def fail_then_pass(case, config_path, attempt_directory, plan, progress_sink):
            del config_path, plan, progress_sink
            calls[case.case_id] = calls.get(case.case_id, 0) + 1
            experiment = attempt_directory / "experiment"
            experiment.mkdir()
            (experiment / "marker.txt").write_text(
                f"attempt={calls[case.case_id]}\n",
                encoding="utf-8",
            )
            return {
                "status": (
                    "failed_closed" if calls[case.case_id] == 1 else "passed"
                ),
                "output_dir": str(experiment),
                "cell_results": [],
            }

        plan = self.plan()
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=fail_then_pass,
            progress_sink=NullProgressSink(),
        )
        first = runner.run(plan)
        self.assertEqual(first.status, "failed")
        resumed = runner.run(plan, resume_directory=first.output_directory)
        self.assertEqual(resumed.status, "passed")
        for case in resumed.case_records:
            case_root = first.output_directory / "cases" / str(case["case_id"])
            self.assertEqual(case["attempt"], 2)
            self.assertEqual(
                case["artifact_root"],
                "attempts/attempt_002/experiment",
            )
            self.assertEqual(
                (
                    case_root
                    / "attempts"
                    / "attempt_001"
                    / "experiment"
                    / "marker.txt"
                ).read_text(encoding="utf-8"),
                "attempt=1\n",
            )
            self.assertEqual(
                (
                    case_root
                    / "attempts"
                    / "attempt_002"
                    / "experiment"
                    / "marker.txt"
                ).read_text(encoding="utf-8"),
                "attempt=2\n",
            )

    def test_canonical_progress_payload_is_normalized(self) -> None:
        event = ProgressEvent.from_value(
            {
                "stage": "cell_complete",
                "current_cell": 7,
                "total_cells": 25,
                "repeat_index": 1,
                "fold_index": 2,
            }
        )
        self.assertEqual(event.event, "cell_complete")
        self.assertEqual((event.current, event.total), (7, 25))
        self.assertEqual((event.repeat, event.fold), (1, 2))

    def test_route_role_quality_tables_keep_coverage_and_failures_separate(self) -> None:
        common = {
            "case_id": "case_001",
            "participant_id": "P01",
            "components": {
                "predictor_availability": {
                    "predictor_count": 10,
                    "available_predictor_count": 8,
                    "unavailable_predictor_count": 2,
                    "unavailable_feature_names": ["a", "b"],
                },
                "non_predictor_features": {
                    "sqi.q_rate": {"value": 0.8, "valid": True},
                },
            },
        }
        bundle = CollectedStudy(
            root=self.root,
            plan={"report": {}},
            manifest={
                "cases": [{"case_id": "case_001", "is_reference": True}],
                "reference_case_id": "case_001",
            },
            case_records=({"case_id": "case_001", "status": "passed"},),
            varied_parameters=(),
            controlled_parameters=(),
            cell_rows=(),
            history_rows=(),
            subject_oof_rows=(),
            role_oof_rows=(
                {
                    "case_id": "case_001",
                    "role": "B",
                    "retained": True,
                },
            ),
            quality_rows=(
                {
                    **common,
                    "role": "B",
                    "retained": True,
                    "route_status": "full_direct",
                    "signal_route": "direct",
                    "route_artifact": {
                        "state": "full_direct",
                        "source_signal": "x_filter",
                    },
                },
                {
                    **common,
                    "role": "R1",
                    "retained": False,
                    "route_status": "rejected_after_reduction",
                    "signal_route": "artifact_reduced",
                    "route_artifact": {
                        "state": "rejected_after_reduction",
                        "source_signal": "x_ar",
                        "reducer_status": "failed",
                    },
                },
            ),
            trusted_config_metrics=(),
            limitations=(),
        )
        analysis = analyze_study(bundle)
        by_role = {row["role"]: row for row in analysis.route_role_coverage}
        self.assertEqual(by_role["B"]["retained_coverage"], 1.0)
        self.assertEqual(by_role["B"]["unavailable_predictor_rate"], 0.2)
        self.assertEqual(by_role["B"]["role_oof_prediction_count"], 1)
        self.assertEqual(by_role["R1"]["retained_coverage"], 0.0)
        self.assertEqual(by_role["R1"]["reducer_failure_count"], 1)
        distributions = {
            (row["role"], row["component"]): row
            for row in analysis.quality_distributions
        }
        self.assertEqual(distributions[("B", "sqi.q_rate")]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
