"""Focused expansion, fake-execution, resume, progress, and report tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import yaml

from ppg_frailty.reporting import generate_study_report
from ppg_frailty.reporting.analyze import (
    _cell_repeat_rows,
    _denoiser_hr_tables,
    analyze_study,
)
from ppg_frailty.reporting.collect import (
    CollectedStudy,
    _cell_rows,
    _oof_rows,
    _quality_rows,
)
from ppg_frailty.experiment import _artifact_index_cell_summary
from ppg_frailty.training.aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    aggregate_hierarchy,
)
from ppg_frailty.training.oof import OofPredictionRow
from ppg_frailty.study import (
    ExecutionSpec,
    NullProgressSink,
    ProgressEvent,
    ResolvedCase,
    StudyRunner,
    default_experiment_executor,
    parse_study_plan,
    validate_canonical_expansion,
)
from ppg_frailty.study.runner import (
    _compact_experiment_result,
    _process_default_case,
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

    def test_primary_ranking_recomputes_complete_roster_abstention_metrics(self) -> None:
        labels = (0, 0, 1, 1, 2, 2)
        folds = (0, 1, 0, 1, 0, 1)

        def probability(prediction: int) -> list[float]:
            values = [0.0, 0.0, 0.0]
            values[prediction] = 1.0
            return values

        case_rows: list[dict[str, Any]] = []
        predictions = {
            "complete": (0, 0, 1, 0, 2, 0),
            "selective": (0, None, 1, None, 2, None),
            "all_abstain": (None, None, None, None, None, None),
        }
        for case_id, case_predictions in predictions.items():
            for index, (label, fold, predicted) in enumerate(
                zip(labels, folds, case_predictions)
            ):
                retained = predicted is not None
                case_rows.append(
                    {
                        "case_id": case_id,
                        "participant_id": f"P{index}",
                        "label": label,
                        "repeat": 0,
                        "fold": fold,
                        "retained": retained,
                        "probabilities": (
                            probability(int(predicted)) if retained else []
                        ),
                        "class_order": [0, 1, 2] if retained else [],
                    }
                )
        cells = tuple(
            {
                "case_id": case_id,
                "status": "passed",
                "repeat": 0,
                "fold": fold,
                # These deliberately misleading fold means must not override
                # complete participant-OOF recomputation.
                "balanced_accuracy": 0.99,
                "macro_f1": 0.99,
                "abstention_aware_balanced_accuracy": 0.99,
                "abstention_aware_macro_f1": 0.99,
            }
            for case_id in predictions
            for fold in (0, 1)
        )
        bundle = CollectedStudy(
            root=self.root,
            plan={
                "execution": {"repeats": [0], "folds": [0, 1]},
                "report": {"top_k": 3},
            },
            manifest={
                "cases": [
                    {"case_id": case_id, "is_reference": case_id == "complete"}
                    for case_id in predictions
                ],
                "reference_case_id": "complete",
            },
            case_records=tuple(
                {"case_id": case_id, "status": "passed"}
                for case_id in predictions
            ),
            varied_parameters=(),
            controlled_parameters=(),
            cell_rows=cells,
            history_rows=(),
            file_oof_rows=(),
            subject_oof_rows=tuple(case_rows),
            role_oof_rows=(),
            quality_rows=(
                {
                    "case_id": "selective",
                    "route_artifact": {
                        "motion_provenance": {
                            "enabled": True,
                            "valid_outer_oof_claim": False,
                            "frailty29_evaluation_relation": (
                                "in_sample_for_frailty29"
                            ),
                        }
                    },
                },
            ),
            trusted_config_metrics=(),
            limitations=(),
        )

        analysis = analyze_study(bundle)

        self.assertEqual(
            [row["case_id"] for row in analysis.predictive_leaderboard],
            ["complete", "selective", "all_abstain"],
        )
        summaries = {row["case_id"]: row for row in analysis.case_summary}
        self.assertAlmostEqual(
            summaries["complete"][
                "participant_mean_abstention_aware_balanced_accuracy"
            ],
            2.0 / 3.0,
        )
        self.assertEqual(
            summaries["selective"]["participant_mean_balanced_accuracy"], 1.0
        )
        self.assertEqual(
            summaries["selective"][
                "participant_mean_abstention_aware_balanced_accuracy"
            ],
            0.5,
        )
        self.assertEqual(
            summaries["all_abstain"][
                "participant_mean_abstention_aware_balanced_accuracy"
            ],
            0.0,
        )
        self.assertIsNone(
            summaries["all_abstain"]["participant_mean_balanced_accuracy"]
        )
        self.assertTrue(
            summaries["all_abstain"]["complete_for_requested_execution"]
        )
        self.assertFalse(
            summaries["selective"][
                "auxiliary_motion_evidence_valid_outer_oof"
            ]
        )
        self.assertEqual(
            summaries["selective"]["ranking_interpretation"],
            "comparison_only_in_sample_auxiliary",
        )
        self.assertEqual(
            summaries["selective"]["frailty_classification_evaluation_scope"],
            "outer_heldout_participant_oof",
        )
        self.assertIsNone(
            summaries["complete"][
                "auxiliary_motion_evidence_valid_outer_oof"
            ]
        )
        self.assertEqual(
            summaries["complete"]["ranking_interpretation"],
            "not_applicable_no_auxiliary_motion_evidence",
        )

    def test_abstention_metrics_are_projected_beside_conditional_metrics(self) -> None:
        rows = _cell_rows(
            "case_001",
            {
                "cell_results": [
                    {
                        "status": "passed",
                        "repeat_index": 0,
                        "fold_index": fold,
                        "metrics": {
                            "balanced_accuracy": 0.8 + fold * 0.1,
                            "macro_f1": 0.7 + fold * 0.1,
                            "abstention_aware_balanced_accuracy": 0.6 + fold * 0.1,
                            "abstention_aware_macro_precision": 0.65,
                            "abstention_aware_macro_recall": 0.6 + fold * 0.1,
                            "abstention_aware_macro_f1": 0.62 + fold * 0.1,
                            "abstention_count": fold + 1,
                            "abstention_counts_by_class": [[0, fold], [2, 1]],
                            "abstention_probability_metrics_scope": "retained_only",
                        },
                    }
                    for fold in (0, 1)
                ]
            },
            self.root,
        )
        repeats, folds = _cell_repeat_rows(rows)
        self.assertEqual(len(folds), 2)
        self.assertAlmostEqual(float(repeats[0]["balanced_accuracy"]), 0.85)
        self.assertAlmostEqual(
            float(repeats[0]["abstention_aware_balanced_accuracy"]),
            0.65,
        )
        self.assertEqual(repeats[0]["abstention_count"], 3)
        self.assertEqual(
            repeats[0]["abstention_counts_by_class"],
            [[0, 1], [2, 2]],
        )

    def test_quality_collection_merges_authoritative_route_artifacts(self) -> None:
        cell = self.root / "case" / "repeat_00_fold_01"
        cell.mkdir(parents=True)
        (cell / "quality_diagnostics.json").write_text(
            json.dumps(
                {
                    "schema_version": "ppg_frailty.quality_diagnostics.v2",
                    "quality_mode": "route",
                    "rows": [
                        {
                            "record_id": "record_a",
                            "role": "B",
                            "components": {
                                "non_predictor_features": {
                                    "sqi.q_rate": {"value": 0.8, "valid": True}
                                }
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (cell / "route_artifacts.json").write_text(
            json.dumps(
                {
                    "schema_version": "ppg_frailty.route_artifacts.v2",
                    "repeat_index": 0,
                    "fold_index": 1,
                    "rows": [
                        {
                            "record_id": "record_a",
                            "role": "B",
                            "route_artifact": {
                                "quality_tier": "excellent",
                                "motion_state": "low_motion",
                            },
                        },
                        {
                            "record_id": "record_b",
                            "role": "R1",
                            "route_artifact": {
                                "quality_tier": "unfit",
                                "motion_state": "high_motion",
                                "abstained": True,
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        rows = _quality_rows("case_001", self.root / "case")
        self.assertEqual(len(rows), 2)
        by_record = {row["record_id"]: row for row in rows}
        self.assertEqual(by_record["record_a"]["repeat"], 0)
        self.assertEqual(by_record["record_a"]["fold"], 1)
        self.assertEqual(by_record["record_a"]["quality_mode"], "route")
        self.assertIn("components", by_record["record_a"])
        self.assertEqual(
            by_record["record_a"]["route_artifact"]["quality_tier"],
            "excellent",
        )
        self.assertEqual(
            by_record["record_b"]["route_artifact"]["motion_state"],
            "high_motion",
        )
        self.assertIn("route_artifacts_artifact", by_record["record_b"])

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

    def test_study_operational_measurement_flag_reaches_canonical_runners(self) -> None:
        case = ResolvedCase(
            case_id="case_001",
            config={"config_id": "fixture"},
            changed_values={},
            config_sha256="a" * 64,
            is_reference=True,
        )
        output = self.root / "executor"
        output.mkdir()
        passed = {
            "status": "passed",
            "scientific_scope": "fixture",
            "config_id": "fixture",
            "config_hash": "a" * 64,
            "cell_results": [],
        }

        full_plan = replace(
            self.plan(),
            execution=ExecutionSpec(measure_operational_costs=True),
        )
        with patch(
            "ppg_frailty.experiment.run_full_experiment",
            return_value=passed,
        ) as full:
            default_experiment_executor(
                case,
                self.base,
                output / "full",
                full_plan,
                NullProgressSink(),
            )
        self.assertTrue(full.call_args.kwargs["measure_operational_costs"])

        cell_plan = replace(
            self.plan(),
            execution=ExecutionSpec(
                repeats=(0,),
                folds=(0,),
                measure_operational_costs=True,
            ),
        )
        with patch(
            "ppg_frailty.experiment.run_outer_cell",
            return_value=passed,
        ) as cell:
            default_experiment_executor(
                case,
                self.base,
                output / "cell",
                cell_plan,
                NullProgressSink(),
            )
        self.assertTrue(cell.call_args.kwargs["measure_operational_costs"])

    def test_study_execution_rejects_unknown_or_nonboolean_controls(self) -> None:
        payload = self.plan().to_dict()
        payload["execution"]["unknown_control"] = True
        with self.assertRaisesRegex(ValueError, "execution key mismatch"):
            parse_study_plan(payload)

        payload = self.plan().to_dict()
        payload["execution"]["measure_operational_costs"] = "false"
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            parse_study_plan(payload)

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
        with self.assertRaisesRegex(ValueError, "fixed_epochs must be an integer"):
            StudyRunner(pipeline_root=pipeline_root).expand(plan)

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
                        "values": [7, 10, 15, 37],
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
            {
                (7, "ablation_7"),
                (10, "default_10"),
                (15, "ablation_15"),
                (37, "configured_37"),
            },
        )

    def test_canonical_study_axis_can_target_a_materialized_default(self) -> None:
        pipeline_root = Path(__file__).resolve().parents[2]
        canonical = pipeline_root / "configs" / "reference_static_role_aware_v2.yaml"
        plan = parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "batch_size_ablation",
                    "kind": "ablation",
                    "purpose": "Exercise one runtime default as a normal axis.",
                    "flow_position": "No execution.",
                    "decision_role": "ablation",
                },
                "base_config": str(canonical),
                "axes": [
                    {
                        "path": "training.batch_size",
                        "values": [64, 32],
                        "reference": 64,
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
            {"training.batch_size"},
        )
        self.assertEqual(
            {case.config["training"]["batch_size"] for case in expansion.cases},
            {32, 64},
        )

    def test_optimizer_axis_materializes_each_module_own_defaults(self) -> None:
        pipeline_root = Path(__file__).resolve().parents[2]
        canonical = pipeline_root / "configs" / "reference_static_role_aware_v2.yaml"
        plan = parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "optimizer_module_ablation",
                    "kind": "ablation",
                    "purpose": "Switch executable optimizer modules.",
                    "flow_position": "No execution.",
                    "decision_role": "ablation",
                },
                "base_config": str(canonical),
                "axes": [
                    {
                        "path": "training.optimizer",
                        "values": ["adam", "sgd"],
                        "reference": "adam",
                    }
                ],
                "execution": {"repeats": [0], "folds": [0], "jobs": 1},
            }
        )
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=pipeline_root).expand(plan)
        )
        by_optimizer = {
            case.config["training"]["optimizer"]: case.config["training"][
                "optimizer_parameters"
            ]
            for case in expansion.cases
        }
        self.assertEqual(set(by_optimizer["adam"]), {"betas", "eps", "amsgrad", "maximize"})
        self.assertEqual(
            set(by_optimizer["sgd"]),
            {"momentum", "dampening", "nesterov", "maximize"},
        )
        self.assertEqual(
            {row["parameter_path"] for row in expansion.varied_parameters},
            {"training.optimizer"},
        )

    def test_aggregation_axis_derives_line_specific_hierarchy(self) -> None:
        pipeline_root = Path(__file__).resolve().parents[2]
        canonical = pipeline_root / "configs" / "reference_static_role_aware_v2.yaml"
        plan = parse_study_plan(
            {
                "schema_version": "ppg_frailty.study_plan.v2",
                "study": {
                    "study_id": "aggregation_module_ablation",
                    "kind": "ablation",
                    "purpose": "Switch the reporting aggregation module.",
                    "flow_position": "No execution.",
                    "decision_role": "ablation",
                },
                "base_config": str(canonical),
                "axes": [
                    {
                        "path": "aggregation.balance_line",
                        "values": [
                            "line_b_equal_role_families",
                            "line_a_equal_files",
                        ],
                        "reference": "line_b_equal_role_families",
                    }
                ],
                "execution": {"repeats": [0], "folds": [0], "jobs": 1},
            }
        )
        expansion = validate_canonical_expansion(
            StudyRunner(pipeline_root=pipeline_root).expand(plan)
        )
        by_line = {
            case.config["aggregation"]["balance_line"]: case.config["aggregation"]
            for case in expansion.cases
        }
        self.assertEqual(
            by_line["line_a_equal_files"]["hierarchy"],
            ["window", "file", "participant"],
        )
        self.assertEqual(
            by_line["line_b_equal_role_families"]["hierarchy"],
            ["window", "file", "role", "participant"],
        )
        self.assertEqual(
            {row["parameter_path"] for row in expansion.varied_parameters},
            {"aggregation.balance_line"},
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
        self.assertIn("Seed and data-split reproducibility", summary)
        self.assertIn("report-only evidence", summary)
        self.assertIn("Deployment measurements", summary)
        self.assertIn("Macro-F1 LCB95", summary)
        self.assertIn(
            "Aggregation sensitivity from the same file-level OOF",
            summary,
        )
        self.assertIn("not selection evidence", summary)
        self.assertIn("N/A — no rows were available.", summary)
        self.assertEqual(
            json.loads(
                (
                    result.output_directory
                    / "tables"
                    / "aggregation_line_comparison.json"
                ).read_text(encoding="utf-8")
            ),
            [],
        )
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
        self.assertIn("tables/test_components.csv", paths)
        self.assertIn("TEST_COMPONENTS.md", paths)
        self.assertIn("tables/worst_class_f1_stability.csv", paths)
        self.assertIn("tables/incomplete_cases.csv", paths)
        self.assertIn("tables/confusion_counts.csv", paths)
        self.assertIn("tables/confusion_row_normalized.csv", paths)
        self.assertIn("tables/aggregation_line_comparison.csv", paths)
        self.assertIn("tables/aggregation_line_repeat_metrics.csv", paths)
        self.assertIn("tables/aggregation_line_per_class_metrics.csv", paths)
        self.assertIn("tables/reproducibility_summary.csv", paths)
        self.assertIn("tables/reproducibility_cases.csv", paths)
        self.assertIn("tables/reproducibility_cells.csv", paths)
        self.assertIn("tables/reproducibility_splits.csv", paths)
        self.assertIn("tables/reproducibility_issues.csv", paths)
        reproducibility = json.loads(
            (
                result.output_directory
                / "tables"
                / "reproducibility_summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(reproducibility), 1)
        self.assertIn(
            reproducibility[0]["audit_status"],
            {"PASS", "FAIL", "NOT_VERIFIABLE"},
        )
        self.assertFalse(reproducibility[0]["training_or_report_gate"])
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

    def test_grouped_cases_are_self_contained_reportable_and_resumable(self) -> None:
        plan = self.plan()
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=fake_executor,
            progress_sink=NullProgressSink(),
        )
        expansion = runner.expand(plan)
        grouped_cases = tuple(
            replace(
                case,
                output_group=group,
                catalog_entry=f"{group}_model",
                screen_profile_id="screen_01",
                rationale="Synthetic grouped-output path test.",
            )
            for case, group in zip(expansion.cases, ("raw", "fusion"))
        )
        grouped_expansion = replace(expansion, cases=grouped_cases)
        runner.expand = lambda _: grouped_expansion  # type: ignore[method-assign]

        result = runner.run(plan)
        manifest = json.loads(
            (result.output_directory / "study_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        by_case = {row["case_id"]: row for row in manifest["cases"]}
        for case, group in zip(grouped_cases, ("raw", "fusion")):
            expected_root = result.output_directory / group / case.case_id
            self.assertTrue((expected_root / "resolved_config.yaml").is_file())
            self.assertTrue((expected_root / "case_result.json").is_file())
            self.assertTrue(
                (
                    expected_root
                    / "attempts"
                    / "attempt_001"
                    / "attempt_result.json"
                ).is_file()
            )
            self.assertEqual(by_case[case.case_id]["output_group"], group)
            self.assertEqual(
                by_case[case.case_id]["case_directory"],
                f"{group}/{case.case_id}",
            )
            self.assertEqual(
                by_case[case.case_id]["resolved_config_path"],
                f"{group}/{case.case_id}/resolved_config.yaml",
            )
        self.assertFalse((result.output_directory / "cases").exists())
        self.assertFalse((result.output_directory / "resolved_configs").exists())

        report = generate_study_report(result.output_directory)
        index = json.loads(report.output_index.read_text(encoding="utf-8"))
        indexed = {row["path"]: row for row in index["artifacts"]}
        indexed_paths = set(indexed)
        self.assertTrue(
            any(path.startswith("raw/") for path in indexed_paths)
        )
        self.assertTrue(
            any(path.startswith("fusion/") for path in indexed_paths)
        )
        self.assertTrue(
            all(
                indexed[path]["type"] == "case_artifact"
                for path in indexed_paths
                if path.startswith(("raw/", "fusion/"))
            )
        )

        config_mtimes = {
            case.case_id: (
                result.output_directory
                / group
                / case.case_id
                / "resolved_config.yaml"
            ).stat().st_mtime_ns
            for case, group in zip(grouped_cases, ("raw", "fusion"))
        }
        resumed = runner.run(plan, resume_directory=result.output_directory)
        self.assertEqual(resumed.resumed_case_count, 2)
        self.assertEqual(resumed.passed_case_count, 2)
        for case, group in zip(grouped_cases, ("raw", "fusion")):
            self.assertEqual(
                (
                    result.output_directory
                    / group
                    / case.case_id
                    / "resolved_config.yaml"
                ).stat().st_mtime_ns,
                config_mtimes[case.case_id],
            )
            attempts = (
                result.output_directory / group / case.case_id / "attempts"
            )
            self.assertEqual(
                [path.name for path in sorted(attempts.glob("attempt_*"))],
                ["attempt_001"],
            )

    def test_parallel_child_reconstruction_preserves_catalog_group_metadata(
        self,
    ) -> None:
        captured: dict[str, Any] = {}

        def capture_executor(
            case,
            config_path,
            attempt_directory,
            plan,
            progress_sink,
        ):
            del config_path, attempt_directory, plan, progress_sink
            captured.update(case.to_dict())
            return {"status": "passed", "cell_results": []}

        attempt = self.root / "parallel_attempt"
        attempt.mkdir()
        request = {
            "case_id": "raw_compact_screen_01",
            "config": {
                "config_id": "raw_compact_screen_01",
                "model": {"model_id": "CompactCNN1D"},
            },
            "changed_values": {"training.fixed_epochs": 10},
            "config_sha256": "a" * 64,
            "is_reference": False,
            "output_group": "raw",
            "catalog_entry": "compact_cnn",
            "screen_profile_id": "screen_01",
            "rationale": "Synthetic parallel reconstruction test.",
            "plan": self.plan().to_dict(),
            "config_path": str(self.base),
            "attempt_directory": str(attempt),
        }
        with patch(
            "ppg_frailty.study.runner.default_experiment_executor",
            side_effect=capture_executor,
        ):
            result = _process_default_case(request)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(captured["output_group"], "raw")
        self.assertEqual(captured["catalog_entry"], "compact_cnn")
        self.assertEqual(captured["screen_profile_id"], "screen_01")
        self.assertEqual(
            captured["rationale"],
            "Synthetic parallel reconstruction test.",
        )

    def test_outer_cv_ensemble_reaches_executor_after_seed_contract_is_frozen(self) -> None:
        runner = StudyRunner(
            pipeline_root=self.root,
            executor=fake_executor,
            progress_sink=NullProgressSink(),
        )
        result = runner.run(self.plan(ensemble=True))
        self.assertEqual(result.passed_case_count, 2)
        self.assertEqual(result.failed_case_count, 0)

    def test_study_executor_result_drops_large_details_after_persistence(self) -> None:
        compact = _compact_experiment_result(
            {
                "status": "passed",
                "scientific_scope": "selected_outer_cell",
                "config_id": "compact",
                "config_hash": "a" * 64,
                "repeat_indices": [0],
                "fold_indices": [0],
                "output_dir": str(self.root / "experiment"),
                "cell_results": [
                    {
                        "status": "passed",
                        "repeat_index": 0,
                        "fold_index": 0,
                        "metrics": {"balanced_accuracy": 0.5},
                        "quality_diagnostics": [{"blob": "x" * 100_000}],
                        "training_history": [{"epoch": 1}],
                    }
                ],
            }
        )
        self.assertEqual(compact["status"], "passed")
        self.assertEqual(
            compact["cell_results"][0]["metrics"]["balanced_accuracy"],
            0.5,
        )
        self.assertNotIn("quality_diagnostics", compact["cell_results"][0])
        self.assertNotIn("training_history", compact["cell_results"][0])
        self.assertLess(len(json.dumps(compact)), 10_000)

    def test_experiment_index_points_to_dedicated_large_artifacts(self) -> None:
        summary = _artifact_index_cell_summary(
            {
                "status": "passed",
                "metrics": {"macro_f1": 0.4},
                "quality_diagnostics": [{"record_id": "r1"}, {"record_id": "r2"}],
                "training_history": [{"epoch": 1}],
            }
        )
        self.assertNotIn("quality_diagnostics", summary)
        self.assertNotIn("training_history", summary)
        self.assertEqual(summary["quality_diagnostic_row_count"], 2)
        self.assertEqual(summary["training_history_row_count"], 1)
        self.assertEqual(
            summary["quality_diagnostics_artifact"],
            "quality_diagnostics.json",
        )

    def test_report_collector_combines_equal_depth_per_fold_oof_files(self) -> None:
        artifact_root = self.root / "experiment"
        paths: list[Path] = []
        for fold in (0, 1):
            target = (
                artifact_root
                / f"repeat_00_fold_{fold:02d}"
                / "oof_subject_predictions.parquet"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")
            paths.append(target)

        def fake_read(path: Path):
            return [{"participant_id": path.parent.name}]

        with patch(
            "ppg_frailty.training.read_oof_parquet",
            side_effect=fake_read,
        ):
            rows, limitation = _oof_rows(
                "case",
                artifact_root,
                filename="oof_subject_predictions.parquet",
            )
        self.assertIsNone(limitation)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["participant_id"] for row in rows},
            {"repeat_00_fold_00", "repeat_00_fold_01"},
        )

        root_aggregate = artifact_root / "oof_subject_predictions.parquet"
        root_aggregate.write_bytes(b"aggregate")
        with patch(
            "ppg_frailty.training.read_oof_parquet",
            side_effect=fake_read,
        ):
            aggregate_rows, limitation = _oof_rows(
                "case",
                artifact_root,
                filename="oof_subject_predictions.parquet",
            )
        self.assertIsNone(limitation)
        self.assertEqual(len(aggregate_rows), 1)
        self.assertEqual(
            aggregate_rows[0]["participant_id"],
            "experiment",
        )

    def test_file_oof_reports_line_a_sensitivity_without_changing_primary_rank(self) -> None:
        probabilities = {
            "P0": {
                "B": (0.98, 0.01, 0.01),
                "R": (0.20, 0.79, 0.01),
            },
            "P1": {
                "B": (0.05, 0.90, 0.05),
                "R": (0.05, 0.90, 0.05),
            },
            "P2": {
                "B": (0.05, 0.05, 0.90),
                "R": (0.05, 0.05, 0.90),
            },
        }
        file_rows: list[OofPredictionRow] = []
        for repeat, seed in ((0, 42), (1, 10042)):
            for participant_index, (participant_id, by_role) in enumerate(
                probabilities.items()
            ):
                current_by_role = by_role
                if repeat == 1 and participant_id == "P0":
                    current_by_role = {
                        "B": (0.01, 0.98, 0.01),
                        "R": (0.01, 0.98, 0.01),
                    }
                for role in ("B", "R1", "R2", "R3", "R4"):
                    role_family = role[0]
                    file_rows.append(
                        OofPredictionRow(
                            participant_id=participant_id,
                            file_id=f"{participant_id}_{role}",
                            role=role,
                            label=participant_index,
                            probabilities=current_by_role[role_family],
                            repeat=repeat,
                            fold=0,
                            split_seed=seed,
                            training_seed=seed,
                            config_hash="a" * 64,
                            manifest_hash="b" * 64,
                            fold_hash="c" * 64,
                            preprocessing_hash="d" * 64,
                            feature_hash="e" * 64,
                            model_hash="f" * 64,
                            representation_mode="raw",
                            signal_route="direct_x_filter",
                            quality_score=1.0,
                            retained=True,
                            level="file",
                            prediction_kind="single_model",
                            class_order=(0, 1, 2),
                            aggregation_rule=LINE_B_EQUAL_ROLE_FAMILIES,
                        )
                    )
            for role in ("B", "R1", "R2", "R3", "R4"):
                file_rows.append(
                    OofPredictionRow(
                        participant_id="P3",
                        file_id=f"P3_{role}",
                        role=role,
                        label=0,
                        probabilities=(),
                        repeat=repeat,
                        fold=0,
                        split_seed=seed,
                        training_seed=seed,
                        config_hash="a" * 64,
                        manifest_hash="b" * 64,
                        fold_hash="c" * 64,
                        preprocessing_hash="d" * 64,
                        feature_hash="e" * 64,
                        model_hash="f" * 64,
                        representation_mode="raw",
                        signal_route="direct_x_filter",
                        quality_score=1.0,
                        retained=False,
                        level="file",
                        prediction_kind="single_model",
                        class_order=(),
                        aggregation_rule=LINE_B_EQUAL_ROLE_FAMILIES,
                        rejection_reason="synthetic_all_files_dropped",
                    )
                )
        window_rows = tuple(
            replace(
                row,
                level="window",
                window_id=f"{row.file_id}::window_{window_index}",
            )
            for row in file_rows
            for window_index in range(2)
        )
        line_b_hierarchy = aggregate_hierarchy(
            file_rows,
            balance_line=LINE_B_EQUAL_ROLE_FAMILIES,
        )
        retained_subject_rows = line_b_hierarchy.participant_rows
        dropped_subject_rows = tuple(
            replace(
                next(
                    row
                    for row in file_rows
                    if row.repeat == repeat and row.participant_id == "P3"
                ),
                file_id=f"participant::P3",
                role="participant",
                level="participant",
            )
            for repeat in (0, 1)
        )
        subject_rows = (*retained_subject_rows, *dropped_subject_rows)
        bundle = CollectedStudy(
            root=self.root,
            plan={
                "execution": {"repeats": [0, 1], "folds": [0]},
                "report": {"calibration_bins": 5},
            },
            manifest={
                "cases": [{"case_id": "case_001", "is_reference": True}],
                "reference_case_id": "case_001",
            },
            case_records=({"case_id": "case_001", "status": "passed"},),
            varied_parameters=(),
            controlled_parameters=(),
            cell_rows=(
                {
                    "case_id": "case_001",
                    "status": "passed",
                    "repeat": 0,
                    "fold": 0,
                },
                {
                    "case_id": "case_001",
                    "status": "passed",
                    "repeat": 1,
                    "fold": 0,
                },
            ),
            history_rows=(),
            window_oof_rows=tuple(
                {"case_id": "case_001", **asdict(row)}
                for row in window_rows
            ),
            file_oof_rows=tuple(
                {"case_id": "case_001", **asdict(row)}
                for row in file_rows
            ),
            subject_oof_rows=tuple(
                {"case_id": "case_001", **asdict(row)}
                for row in subject_rows
            ),
            role_oof_rows=tuple(
                {"case_id": "case_001", **asdict(row)}
                for row in line_b_hierarchy.role_rows
            ),
            quality_rows=(),
            trusted_config_metrics=(),
            limitations=(),
            resolved_aggregation_configs=(
                {
                    "case_id": "case_001",
                    "resolved_config_path": "synthetic/resolved_config.yaml",
                    "aggregation": {
                        "balance_line": LINE_B_EQUAL_ROLE_FAMILIES,
                        "quality_weighting": False,
                        "quality_weight_source": "none",
                    },
                },
            ),
        )
        analysis = analyze_study(bundle)
        by_line = {
            str(row["balance_line"]): row
            for row in analysis.aggregation_line_comparison
        }
        self.assertEqual(set(by_line), {LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES})
        self.assertAlmostEqual(
            float(
                by_line[LINE_B_EQUAL_ROLE_FAMILIES][
                    "participant_mean_balanced_accuracy"
                ]
            ),
            5.0 / 6.0,
        )
        self.assertAlmostEqual(
            float(
                by_line[LINE_A_EQUAL_FILES][
                    "participant_mean_balanced_accuracy"
                ]
            ),
            2.0 / 3.0,
        )
        self.assertTrue(
            by_line[LINE_B_EQUAL_ROLE_FAMILIES]["primary_ranking_eligible"]
        )
        self.assertFalse(by_line[LINE_A_EQUAL_FILES]["primary_ranking_eligible"])
        self.assertEqual(
            by_line[LINE_A_EQUAL_FILES]["view_role"],
            "posthoc_aggregation_only",
        )
        self.assertAlmostEqual(
            float(
                by_line[LINE_A_EQUAL_FILES][
                    "line_a_minus_line_b_balanced_accuracy"
                ]
            ),
            -1.0 / 6.0,
        )
        self.assertAlmostEqual(
            float(by_line[LINE_B_EQUAL_ROLE_FAMILIES]["worst_class_f1"]),
            0.5,
        )
        self.assertEqual(
            by_line[LINE_B_EQUAL_ROLE_FAMILIES]["participant_oof_total_count"],
            8,
        )
        self.assertEqual(
            by_line[LINE_B_EQUAL_ROLE_FAMILIES]["dropped_participant_oof_count"],
            2,
        )
        self.assertEqual(
            by_line[LINE_B_EQUAL_ROLE_FAMILIES][
                "dropped_file_oof_prediction_count"
            ],
            10,
        )
        line_b_repeats = [
            row
            for row in analysis.aggregation_line_repeat_metrics
            if row["balance_line"] == LINE_B_EQUAL_ROLE_FAMILIES
        ]
        self.assertEqual(len(analysis.aggregation_line_repeat_metrics), 4)
        self.assertEqual(len(analysis.aggregation_line_per_class_metrics), 12)
        self.assertEqual(
            {
                row["aggregation_view"]
                for row in analysis.aggregation_view_comparison
            },
            {
                "window_balanced_to_participant",
                LINE_A_EQUAL_FILES,
                LINE_B_EQUAL_ROLE_FAMILIES,
            },
        )
        self.assertEqual(len(analysis.aggregation_view_confusion_matrices), 3)
        self.assertEqual(len(analysis.aggregation_view_per_class_metrics), 18)
        hierarchy_counts = {
            (row["aggregation_level"], row["group_label"]): row[
                "participant_count"
            ]
            for row in analysis.aggregation_hierarchy_coverage
            if row["repeat"] == 0
        }
        self.assertEqual(
            {hierarchy_counts[("window", role)] for role in ("B", "R1", "R2", "R3", "R4")},
            {3},
        )
        self.assertEqual(
            {hierarchy_counts[("role", role)] for role in ("B", "R")},
            {3},
        )
        self.assertAlmostEqual(
            float(
                by_line[LINE_B_EQUAL_ROLE_FAMILIES][
                    "expected_calibration_error"
                ]
            ),
            sum(float(row["expected_calibration_error"]) for row in line_b_repeats)
            / 2.0,
        )
        self.assertEqual(len(analysis.predictive_leaderboard), 1)

        report_bundle = replace(
            bundle,
            plan={
                **bundle.plan,
                "report": {
                    "calibration_bins": 5,
                    "write_static_figures": False,
                    "write_html": True,
                },
            },
        )
        report = generate_study_report(self.root, collected=report_bundle)
        reported_lines = json.loads(
            (
                self.root
                / "tables"
                / "aggregation_line_comparison.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(reported_lines), 2)
        self.assertEqual(
            {row["balance_line"] for row in reported_lines},
            {LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES},
        )
        reported_by_line = {
            str(row["balance_line"]): row
            for row in reported_lines
        }
        self.assertAlmostEqual(
            float(
                reported_by_line[LINE_B_EQUAL_ROLE_FAMILIES][
                    "participant_mean_balanced_accuracy"
                ]
            ),
            5.0 / 6.0,
        )
        self.assertAlmostEqual(
            float(
                reported_by_line[LINE_A_EQUAL_FILES][
                    "participant_mean_balanced_accuracy"
                ]
            ),
            2.0 / 3.0,
        )
        self.assertEqual(
            sum(
                bool(row["primary_ranking_eligible"])
                for row in reported_lines
            ),
            1,
        )
        markdown = report.summary_markdown.read_text(encoding="utf-8")
        self.assertIsNotNone(report.summary_html)
        html = report.summary_html.read_text(encoding="utf-8")
        for expected in (
            "Aggregation sensitivity from the same file-level OOF",
            LINE_A_EQUAL_FILES,
            LINE_B_EQUAL_ROLE_FAMILIES,
            "Mean BA",
        ):
            self.assertIn(expected, markdown)
            self.assertIn(expected, html)
        summary_payload = json.loads(
            (self.root / "study_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            len(
                summary_payload["analysis"][
                    "aggregation_line_comparison"
                ]
            ),
            2,
        )

        incomplete = analyze_study(
            replace(
                bundle,
                plan={
                    "execution": {"repeats": [0, 1, 2], "folds": [0]},
                    "report": {"calibration_bins": 5},
                },
            )
        )
        incomplete_by_line = {
            str(row["balance_line"]): row
            for row in incomplete.aggregation_line_comparison
        }
        self.assertFalse(
            incomplete_by_line[LINE_B_EQUAL_ROLE_FAMILIES][
                "primary_ranking_eligible"
            ]
        )

        unreadable = analyze_study(
            replace(
                bundle,
                oof_read_failures=(
                    {
                        "case_id": "case_001",
                        "oof_level": "file",
                        "error": "synthetic fold parquet failure",
                    },
                ),
            )
        )
        self.assertEqual(unreadable.aggregation_line_comparison, ())
        self.assertTrue(
            any(
                "OOF input was incomplete or unreadable" in note
                for note in unreadable.notes
            )
        )

        tampered_subject_rows = list(subject_rows)
        tampered_subject_rows[0] = replace(
            tampered_subject_rows[0],
            probabilities=(0.0, 1.0, 0.0),
        )
        tampered = replace(
            bundle,
            subject_oof_rows=tuple(
                {"case_id": "case_001", **asdict(row)}
                for row in tampered_subject_rows
            ),
        )
        rejected = analyze_study(tampered)
        self.assertEqual(rejected.aggregation_line_comparison, ())
        self.assertTrue(
            any(
                "source-line replay probability mismatch" in note
                for note in rejected.notes
            )
        )

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
            file_oof_rows=(),
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
            file_oof_rows=(),
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
                        "quality_tier": "excellent",
                        "motion_state": "low_motion",
                        "motion_record_probability": 0.2,
                        "motion_threshold": 0.5,
                        "motion_window_count": 4,
                        "motion_provenance": {
                            "enabled": True,
                            "evidence_sha256": "a" * 64,
                            "model_artifact_sha256": "b" * 64,
                            "training_scope": "frailty29_all_participants",
                            "frailty29_evaluation_relation": (
                                "in_sample_for_frailty29"
                            ),
                        },
                        "abstained": False,
                        "denoiser_attempted": False,
                        "direct_q_rate_state": "pass",
                        "direct_q_rate_score": 0.8,
                        "direct_q_rate_coverage": 0.9,
                        "direct_q_morph_state": "pass",
                        "direct_q_morph_score": 0.7,
                        "direct_q_morph_coverage": 0.85,
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
                        "quality_tier": "unfit",
                        "motion_state": "high_motion",
                        "motion_record_probability": 0.8,
                        "motion_threshold": 0.5,
                        "motion_window_count": 4,
                        "motion_provenance": {
                            "enabled": True,
                            "evidence_sha256": "a" * 64,
                            "model_artifact_sha256": "b" * 64,
                            "training_scope": "frailty29_all_participants",
                            "frailty29_evaluation_relation": (
                                "in_sample_for_frailty29"
                            ),
                        },
                        "abstained": True,
                        "abstention_reason": "denoiser_failed",
                        "denoiser_attempted": True,
                        "denoiser_id": "pca_bss",
                        "denoiser_status": "failed",
                        "direct_q_rate_state": "pass",
                        "direct_q_rate_score": 0.6,
                        "direct_q_rate_coverage": 0.7,
                        "direct_q_morph_state": "fail",
                        "direct_q_morph_score": 0.4,
                        "direct_q_morph_coverage": 0.6,
                        "post_q_rate_state": "fail",
                        "post_q_rate_score": 0.3,
                        "post_q_rate_coverage": 0.75,
                    },
                },
            ),
            trusted_config_metrics=(),
            limitations=(),
        )
        analysis = analyze_study(bundle)
        by_role = {row["role"]: row for row in analysis.route_role_coverage}
        self.assertEqual(by_role["B"]["retained_coverage"], 1.0)
        self.assertEqual(by_role["B"]["quality_tier"], "excellent")
        self.assertEqual(by_role["B"]["motion_state"], "low_motion")
        self.assertEqual(by_role["B"]["abstention_rate"], 0.0)
        self.assertEqual(by_role["B"]["mean_motion_record_probability"], 0.2)
        self.assertEqual(by_role["B"]["motion_evidence_sha256"], "a" * 64)
        self.assertEqual(by_role["B"]["direct_q_rate_states"], "pass")
        self.assertEqual(by_role["B"]["mean_direct_q_rate_coverage"], 0.9)
        self.assertEqual(
            by_role["B"]["motion_frailty29_relation"],
            "in_sample_for_frailty29",
        )
        self.assertEqual(by_role["B"]["unavailable_predictor_rate"], 0.2)
        self.assertEqual(by_role["B"]["role_oof_prediction_count"], 1)
        self.assertEqual(by_role["R1"]["retained_coverage"], 0.0)
        self.assertEqual(by_role["R1"]["abstention_rate"], 1.0)
        self.assertEqual(by_role["R1"]["denoiser_attempt_count"], 1)
        self.assertEqual(by_role["R1"]["denoiser_ids"], "pca_bss")
        self.assertEqual(by_role["R1"]["post_q_rate_states"], "fail")
        self.assertEqual(by_role["R1"]["mean_post_q_rate_coverage"], 0.75)
        self.assertEqual(by_role["R1"]["reducer_failure_count"], 1)
        distributions = {
            (row["role"], row["component"]): row
            for row in analysis.quality_distributions
        }
        self.assertEqual(distributions[("B", "sqi.q_rate")]["mean"], 0.8)

    def test_denoiser_hr_tables_preserve_pairs_and_participant_macro(self) -> None:
        rows = []
        for participant_id, direct_values, post_values in (
            ("P01", (70.0, 72.0), (75.0, 76.0)),
            ("P02", (80.0, 82.0), (81.0, 83.0)),
        ):
            for index, (direct_hr, post_hr) in enumerate(
                zip(direct_values, post_values, strict=True)
            ):
                rows.append(
                    {
                        "case_id": "raw_compact_cnn__sqi_motion_pca",
                        "repeat": 0,
                        "fold": index,
                        "outer_partition": "outer_oof",
                        "participant_id": participant_id,
                        "record_id": f"{participant_id}_R{index + 1}",
                        "role": f"R{index + 1}",
                        "retained": False,
                        "route_artifact": {
                            "denoiser_attempted": True,
                            "denoiser_id": "pca_bss",
                            "denoiser_status": "success",
                            "heart_rate_estimator": (
                                "60_over_median_valid_ppi_s"
                            ),
                            "direct_hr_bpm": direct_hr,
                            "post_denoise_hr_bpm": post_hr,
                            "direct_valid_ppi_count": 20,
                            "post_denoise_valid_ppi_count": 21,
                            "post_q_rate_state": "pass",
                        },
                    }
                )
        records, summary = _denoiser_hr_tables(
            SimpleNamespace(quality_rows=tuple(rows))
        )
        self.assertEqual(len(records), 4)
        overall = next(
            row
            for row in summary
            if row["role_scope"] == "ALL"
            and row["outer_partition"] == "outer_oof"
        )
        self.assertEqual(overall["paired_hr_record_count"], 4)
        self.assertEqual(overall["paired_participant_count"], 2)
        self.assertAlmostEqual(
            overall["participant_macro_direct_hr_bpm"], 76.0
        )
        self.assertAlmostEqual(
            overall["participant_macro_post_denoise_hr_bpm"], 78.75
        )
        self.assertAlmostEqual(
            overall["participant_macro_post_minus_direct_hr_bpm"], 2.75
        )


if __name__ == "__main__":
    unittest.main()
