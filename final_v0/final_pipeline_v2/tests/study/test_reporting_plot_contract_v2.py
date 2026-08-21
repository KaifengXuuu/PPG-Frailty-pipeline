"""Focused reporter contracts for categorical axes and learning curves."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ppg_frailty.reporting.collect import _history_rows
from ppg_frailty.reporting.plots import (
    _aggregation_view_metrics,
    _balanced_accuracy_learning_curves,
    _loss_history_metric_names,
    _save,
)
from ppg_frailty.reporting.report import _report_html


class ReportingPlotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as pyplot
        except ImportError as error:
            raise unittest.SkipTest(f"matplotlib unavailable: {error}") from error
        cls.pyplot = pyplot

    def tearDown(self) -> None:
        self.pyplot.close("all")

    def test_aggregation_case_ticks_are_shared_and_centered(self) -> None:
        views = (
            "window_balanced_to_participant",
            "line_a_equal_files",
            "line_b_equal_role_families",
        )
        rows = []
        for case_id in (
            "stage1__raw__raw__model_a",
            "stage2__raw__model_b",
        ):
            for view_index, view in enumerate(views):
                if case_id.startswith("stage2") and view == views[0]:
                    continue
                rows.append(
                    {
                        "case_id": case_id,
                        "aggregation_view": view,
                        "participant_mean_balanced_accuracy": (
                            0.5 + view_index * 0.01
                        ),
                        "participant_mean_macro_f1": 0.4 + view_index * 0.01,
                    }
                )
        figure = _aggregation_view_metrics(
            SimpleNamespace(aggregation_view_comparison=tuple(rows)),
            self.pyplot,
        )
        top, bottom = figure.axes
        self.assertTrue(top.get_shared_x_axes().joined(top, bottom))
        np.testing.assert_allclose(bottom.get_xticks(), [0.0, 1.0])
        self.assertEqual(
            [label.get_text() for label in bottom.get_xticklabels()],
            ["stage1\nraw\nraw\nmodel_a", "stage2\nraw\nmodel_b"],
        )
        self.assertEqual(
            {label.get_rotation() for label in bottom.get_xticklabels()},
            {25.0},
        )
        np.testing.assert_allclose(bottom.get_xlim(), [-0.5, 1.5])
        centres = [
            patch.get_x() + patch.get_width() / 2.0
            for container in bottom.containers
            for patch in container
        ]
        first_case = [value for value in centres if value < 0.5]
        second_case = [value for value in centres if value > 0.5]
        self.assertEqual(len(first_case), 3)
        self.assertEqual(len(second_case), 2)
        self.assertAlmostEqual(float(np.mean(first_case)), 0.0)
        self.assertAlmostEqual(float(np.mean(second_case)), 1.0)

    def test_ambiguous_outer_metric_writes_explicit_na_png(self) -> None:
        collected = SimpleNamespace(
            history_rows=(
                {
                    "case_id": "case_a",
                    "repeat": 0,
                    "fold": 0,
                    "epoch": 1,
                    "training_loss": 0.8,
                    "balanced_accuracy": 0.9,
                    "val_balanced_accuracy": 0.95,
                },
            )
        )
        self.assertEqual(
            _loss_history_metric_names(collected.history_rows),
            ("training_loss",),
        )
        with self.assertRaisesRegex(ValueError, "provenance-safe"):
            _balanced_accuracy_learning_curves(collected, self.pyplot)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = _save(
                root,
                "balanced_accuracy_learning_curves",
                lambda plot: _balanced_accuracy_learning_curves(
                    collected, plot
                ),
                self.pyplot,
                render_na_png=True,
            )
            self.assertEqual(result["status"], "N/A")
            self.assertIn(
                "outer held-out metrics are never converted",
                result["reason"],
            )
            self.assertTrue(
                (root / "balanced_accuracy_learning_curves.png").is_file()
            )
            self.assertFalse(
                (root / "balanced_accuracy_learning_curves.NA.txt").exists()
            )

    def test_explicit_inner_participant_history_is_plotted(self) -> None:
        collected = SimpleNamespace(
            history_rows=tuple(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "fold": 0,
                    "epoch": epoch,
                    "inner_participant_balanced_accuracy": score,
                    "inner_balanced_accuracy": 0.1,
                    "balanced_accuracy": 0.99,
                }
                for case_id in ("case_a", "case_b")
                for repeat in (0, 1)
                for epoch, score in ((1, 0.4), (2, 0.6))
            )
        )
        figure = _balanced_accuracy_learning_curves(collected, self.pyplot)
        self.assertEqual(len(figure.axes), 1)
        axis = figure.axes[0]
        self.assertEqual(
            axis.get_ylabel(), "inner_participant_balanced_accuracy"
        )
        np.testing.assert_allclose(axis.get_ylim(), [0.0, 1.0])
        self.assertEqual(
            {line.get_label() for line in axis.lines},
            {"case_a", "case_b"},
        )
        for line in axis.lines:
            np.testing.assert_allclose(line.get_xdata(), [1.0, 2.0])
            np.testing.assert_allclose(line.get_ydata(), [0.4, 0.6])

    def test_history_collection_projects_curve_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / "repeat_00_fold_00" / "training_history.json"
            history.parent.mkdir()
            history.write_text(
                json.dumps(
                    {
                        "schema_version": "ppg_frailty.training_history.v2",
                        "learning_curve_contract": {
                            "status": "outer_train_loss_only_fixed_epoch",
                            "training_data_scope": (
                                "outer_train_participants_only"
                            ),
                            "outer_heldout_used_for_epoch_selection_or_curve": False,
                            "validation_metric": (
                                "not_applicable_fixed_epoch_no_inner_validation"
                            ),
                        },
                        "rows": [
                            {
                                "repeat": 0,
                                "fold": 0,
                                "epoch": 1,
                                "training_loss": 0.8,
                                "learning_curve_outer_heldout_used": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows = _history_rows("case_a", {}, root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["learning_curve_status"],
            "outer_train_loss_only_fixed_epoch",
        )
        self.assertEqual(
            rows[0]["learning_curve_training_data_scope"],
            "outer_train_participants_only",
        )
        self.assertIs(rows[0]["learning_curve_outer_heldout_used"], False)

    def test_explicit_outer_heldout_flag_rejects_even_inner_named_metric(self) -> None:
        collected = SimpleNamespace(
            history_rows=(
                {
                    "case_id": "case_a",
                    "epoch": 1,
                    "inner_participant_balanced_accuracy": 0.9,
                    "learning_curve_outer_heldout_used": True,
                },
            )
        )
        with self.assertRaisesRegex(ValueError, "explicitly marks outer"):
            _balanced_accuracy_learning_curves(collected, self.pyplot)

    def test_mixed_safe_and_outer_heldout_ba_rejects_complete_figure(self) -> None:
        collected = SimpleNamespace(
            history_rows=(
                {
                    "case_id": "case_a",
                    "epoch": 1,
                    "inner_participant_balanced_accuracy": 0.4,
                    "learning_curve_outer_heldout_used": False,
                },
                {
                    "case_id": "case_a",
                    "epoch": 2,
                    "inner_participant_balanced_accuracy": 0.9,
                    "learning_curve_outer_heldout_used": True,
                },
                {
                    "case_id": "case_a",
                    "epoch": 2,
                    "training_loss": 0.2,
                    "learning_curve_outer_heldout_used": True,
                },
            )
        )
        with self.assertRaisesRegex(ValueError, "complete learning-curve"):
            _balanced_accuracy_learning_curves(collected, self.pyplot)

    def test_html_embeds_na_balanced_accuracy_png_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            collected = SimpleNamespace(
                root=Path(temporary),
                plan={"study": {"study_id": "report_test"}},
            )
            analysis = SimpleNamespace(
                notes=(),
                predictive_leaderboard=(
                    {
                        "case_id": "motion_case",
                        "participant_mean_balanced_accuracy": 0.8,
                        "participant_mean_macro_f1": 0.7,
                        "participant_mean_abstention_aware_balanced_accuracy": 0.6,
                        "participant_mean_abstention_aware_macro_precision": 0.65,
                        "participant_mean_abstention_aware_macro_recall": 0.6,
                        "participant_mean_abstention_aware_macro_f1": 0.62,
                        "abstention_count": 2,
                        "abstention_counts_by_class": [[0, 0], [1, 1], [2, 1]],
                    },
                ),
                aggregation_line_comparison=(),
                aggregation_view_comparison=(),
                aggregation_hierarchy_coverage=(),
                worst_class_f1_stability=(),
                incomplete_cases=(),
                deployment_table=(),
                route_role_coverage=(
                    {
                        "case_id": "motion_case",
                        "role": "B",
                        "quality_tier": "excellent",
                        "motion_state": "low_motion",
                        "motion_frailty29_relation": "in_sample_for_frailty29",
                        "motion_evidence_sha256": "a" * 64,
                        "abstention_rate": 0.0,
                        "direct_q_rate_states": "pass",
                        "mean_direct_q_rate_score": 0.8,
                        "mean_direct_q_rate_coverage": 0.9,
                    },
                ),
                quality_distributions=(),
            )
            reason = "no provenance-safe inner/train balanced-accuracy history"
            html = _report_html(
                collected,
                analysis,
                (
                    {
                        "figure": "balanced_accuracy_learning_curves",
                        "status": "N/A",
                        "path": (
                            "figures/balanced_accuracy_learning_curves.png"
                        ),
                        "reason": reason,
                    },
                ),
            )
        self.assertIn(
            "src='figures/balanced_accuracy_learning_curves.png'", html
        )
        self.assertIn("N/A:", html)
        self.assertIn(reason, html)
        self.assertIn("Frozen motion evidence used by each route", html)
        self.assertIn("SQI state, score, and coverage provenance", html)
        self.assertIn("in_sample_for_frailty29", html)
        self.assertIn("Conditional BA", html)
        self.assertIn("Abstention-aware Macro-F1", html)


if __name__ == "__main__":
    unittest.main()
