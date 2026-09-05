"""Synthetic, no-training contracts for the two Stage-3 bridge reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from ppg_frailty.reporting.analyze import (
    _aggregation_hierarchy_coverage,
    _legacy_bridge_report_tables,
    analyze_study,
)
from ppg_frailty.reporting.collect import CollectedStudy
from ppg_frailty.reporting.plots import (
    _legacy_bridge_execution_order_report,
    _legacy_bridge_numeric_ablation_report,
)
from ppg_frailty.reporting.report import generate_study_report
from ppg_frailty.training.aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
)
from ppg_frailty.training.oof import OofPredictionRow


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "configs"
    / "studies"
    / "static_line_b_staged_v2"
    / "stage3_alter.yaml"
)
WINDOW_VIEW = "window_balanced_to_participant"


def _plan() -> dict:
    payload = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - fixture clarity
        raise TypeError("stage3_alter.yaml root must be a mapping")
    return payload


def _synthetic_report_inputs(plan: dict):
    bridge = plan["legacy_bridge"]
    profiles = {
        row["case_id"]: row
        for row in bridge["profiles"]
        if row["model_id"] == "CompactCNN1D"
    }
    numeric_cases = [
        case_id
        for case_id in bridge["numeric_profile_order"]
        if case_id in profiles
    ]
    summaries = []
    views = []
    for level, case_id in enumerate(numeric_cases):
        catalog_case_id = profiles[case_id]["catalog_case_id"]
        summaries.append(
            {
                "case_id": catalog_case_id,
                "status": "passed",
                "complete_for_requested_execution": True,
            }
        )
        for view_index, view in enumerate(
            (WINDOW_VIEW, LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES)
        ):
            views.append(
                {
                    "case_id": catalog_case_id,
                    "aggregation_view": view,
                    "participant_mean_balanced_accuracy": (
                        0.50 + level * 0.02 + view_index * 0.005
                    ),
                    "participant_mean_macro_f1": (
                        0.45 + level * 0.015 + view_index * 0.005
                    ),
                    "worst_class_f1": (
                        0.35 + level * 0.01 + view_index * 0.005
                    ),
                    "metric_source": "synthetic_same_oof_report_reaggregation",
                    "evidence_role": "synthetic_report_test_only",
                }
            )
    return summaries, views


def _empty_bundle(root: Path, plan: dict) -> CollectedStudy:
    return CollectedStudy(
        root=root,
        plan=plan,
        manifest={"cases": [], "reference_case_id": None},
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
    )


class LegacyBridgeDualReportTests(unittest.TestCase):
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

    def _reports(self):
        plan = _plan()
        summaries, views = _synthetic_report_inputs(plan)
        numeric, execution, notes = _legacy_bridge_report_tables(
            SimpleNamespace(plan=plan),
            summaries,
            views,
        )
        return plan, numeric, execution, notes

    def test_runtime_catalog_ids_join_metrics_without_changing_display_order(self) -> None:
        plan, numeric, execution, _notes = self._reports()
        expected_display = [
            case_id
            for case_id in plan["legacy_bridge"]["numeric_profile_order"]
            if case_id.startswith("compact_cnn__")
        ]
        profile_by_display = {
            row["case_id"]: row for row in plan["legacy_bridge"]["profiles"]
        }
        self.assertEqual([row["case_id"] for row in numeric], expected_display)
        self.assertEqual(
            [row["display_case_id"] for row in numeric], expected_display
        )
        self.assertEqual(
            [row["catalog_case_id"] for row in numeric],
            [profile_by_display[value]["catalog_case_id"] for value in expected_display],
        )
        self.assertTrue(all(row["case_status"] == "passed" for row in numeric))
        self.assertTrue(
            all(row["BA_v2_aggregation"] is not None for row in numeric)
        )
        self.assertEqual(
            [row["display_case_id"] for row in execution],
            [
                case_id
                for case_id in plan["legacy_bridge"]["execution_order"]
                if case_id.startswith("compact_cnn__")
            ],
        )

    def test_numeric_report_has_baseline_plus_seven_adjacent_contrasts(self) -> None:
        _plan_value, numeric, _execution, notes = self._reports()
        self.assertEqual([row["profile"] for row in numeric], [f"L{i}" for i in range(8)])
        contrasts = [
            row
            for row in numeric
            if row["comparison_role"]
            == "predefined_adjacent_numeric_ablation"
        ]
        self.assertEqual(len(contrasts), 7)
        self.assertEqual(
            [row["numeric_comparison"] for row in contrasts],
            [f"L{i}->L{i + 1}" for i in range(7)],
        )
        for row in contrasts:
            self.assertAlmostEqual(row["delta_BA_legacy_aggregation"], 0.02)
            self.assertAlmostEqual(row["delta_BA_line_a_aggregation"], 0.02)
            self.assertAlmostEqual(row["delta_BA_v2_aggregation"], 0.02)
            self.assertTrue(row["contrast_metrics_available"])
        self.assertIn("including L7->L5", notes[0])

    def test_execution_report_has_absolute_metrics_and_no_delta_fields(self) -> None:
        _plan_value, numeric, execution, _notes = self._reports()
        self.assertEqual(
            [row["profile"] for row in execution],
            ["L7", "L5", "L6", "L4", "L3", "L2", "L1", "L0"],
        )
        self.assertEqual(execution[1]["execution_transition"], "L7->L5")
        self.assertTrue(
            all(not row["execution_transition_is_ablation"] for row in execution)
        )
        self.assertTrue(
            all(
                not any(key.startswith("delta_") for key in row)
                for row in execution
            )
        )
        numeric_by_case = {row["case_id"]: row for row in numeric}
        for row in execution:
            source = numeric_by_case[row["case_id"]]
            for metric in (
                "BA_legacy_aggregation",
                "BA_line_a_aggregation",
                "BA_v2_aggregation",
                "macroF1_legacy_aggregation",
                "macroF1_line_a_aggregation",
                "macroF1_v2_aggregation",
            ):
                self.assertEqual(row[metric], source[metric])

    def test_tampered_execution_order_fails_report_contract_closed(self) -> None:
        plan = _plan()
        plan["legacy_bridge"]["execution_order"][1:3] = reversed(
            plan["legacy_bridge"]["execution_order"][1:3]
        )
        summaries, views = _synthetic_report_inputs(_plan())
        numeric, execution, notes = _legacy_bridge_report_tables(
            SimpleNamespace(plan=plan),
            summaries,
            views,
        )
        self.assertEqual(numeric, [])
        self.assertEqual(execution, [])
        self.assertIn("frozen order contract is invalid", notes[0])

    def test_bridge_plots_preserve_the_two_distinct_orders(self) -> None:
        _plan_value, numeric, execution, _notes = self._reports()
        analysis = SimpleNamespace(
            legacy_bridge_numeric_ablation_report=tuple(numeric),
            legacy_bridge_execution_order_report=tuple(execution),
        )
        numeric_figure = _legacy_bridge_numeric_ablation_report(
            analysis, self.pyplot
        )
        execution_figure = _legacy_bridge_execution_order_report(
            analysis, self.pyplot
        )
        self.assertEqual(
            [label.get_text() for label in numeric_figure.axes[-1].get_xticklabels()],
            [f"L{i}→L{i + 1}" for i in range(7)],
        )
        self.assertEqual(
            [label.get_text() for label in execution_figure.axes[-1].get_xticklabels()],
            ["L7", "L5", "L6", "L4", "L3", "L2", "L1", "L0"],
        )
        self.assertIn(
            "not causal ablations",
            execution_figure._suptitle.get_text(),
        )

    def test_line_a_source_replays_line_b_role_coverage_from_same_file_oof(self) -> None:
        case_id = "compact_cnn__l0_legacy64_w15_fixed10"

        def prediction(participant: int, role: str, *, level: str) -> dict:
            row = OofPredictionRow(
                participant_id=f"P{participant:02d}",
                file_id=f"P{participant:02d}_{role}",
                role=role,
                label=participant % 3,
                probabilities=(0.6, 0.3, 0.1),
                repeat=0,
                fold=participant % 5,
                split_seed=42,
                training_seed=42,
                config_hash="config",
                manifest_hash="manifest",
                fold_hash="fold",
                preprocessing_hash="preprocessing",
                feature_hash="feature",
                model_hash="model",
                representation_mode="raw",
                signal_route="direct",
                quality_score=1.0,
                retained=True,
                level=level,
                window_id=(
                    f"P{participant:02d}_{role}::w0"
                    if level == "window"
                    else None
                ),
                class_order=(0, 1, 2),
                aggregation_rule=LINE_A_EQUAL_FILES,
            )
            return {"case_id": case_id, **asdict(row)}

        roles = ("B", "R1", "R2", "R3", "R4")
        file_rows = tuple(
            prediction(participant, role, level="file")
            for participant in range(29)
            for role in roles
        )
        window_rows = tuple(
            prediction(participant, role, level="window")
            for participant in range(29)
            for role in roles
        )
        collected = SimpleNamespace(
            window_oof_rows=window_rows,
            file_oof_rows=file_rows,
            role_oof_rows=(),
            subject_oof_rows=(),
            oof_read_failures=(),
            resolved_aggregation_configs=(
                {
                    "case_id": case_id,
                    "resolved_config_path": "synthetic/resolved_config.yaml",
                    "aggregation": {
                        "balance_line": LINE_A_EQUAL_FILES,
                        "quality_weighting": False,
                        "quality_weight_source": "none",
                    },
                },
            ),
            resolved_config_failures=(),
        )

        coverage = _aggregation_hierarchy_coverage(collected)
        role_coverage = {
            row["group_label"]: row
            for row in coverage
            if row["case_id"] == case_id
            and row["aggregation_level"] == "role"
        }
        self.assertEqual(set(role_coverage), {"B", "R", "ALL"})
        self.assertTrue(
            all(row["participant_count"] == 29 for row in role_coverage.values())
        )
        self.assertEqual(role_coverage["B"]["retained_oof_unit_count"], 29)
        self.assertEqual(role_coverage["R"]["retained_oof_unit_count"], 29)
        self.assertEqual(role_coverage["ALL"]["retained_oof_unit_count"], 58)
        window_coverage = {
            row["group_label"]: row
            for row in coverage
            if row["case_id"] == case_id
            and row["aggregation_level"] == "window"
        }
        self.assertEqual(set(window_coverage), {*roles, "ALL"})
        self.assertTrue(
            all(row["participant_count"] == 29 for row in window_coverage.values())
        )

    def test_full_report_writes_two_csv_json_tables_and_embeds_two_figures(self) -> None:
        plan, numeric, execution, notes = self._reports()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _empty_bundle(root, plan)
            analysis = replace(
                analyze_study(bundle),
                legacy_bridge_numeric_ablation_report=tuple(numeric),
                legacy_bridge_execution_order_report=tuple(execution),
                notes=tuple(notes),
            )
            with patch(
                "ppg_frailty.reporting.report.analyze_study",
                return_value=analysis,
            ):
                result = generate_study_report(root, collected=bundle)

            numeric_json = json.loads(
                (root / "tables/legacy_bridge_numeric_ablation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            execution_json = json.loads(
                (root / "tables/legacy_bridge_execution_order_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual([row["profile"] for row in numeric_json], [f"L{i}" for i in range(8)])
            self.assertEqual(
                [row["profile"] for row in execution_json],
                ["L7", "L5", "L6", "L4", "L3", "L2", "L1", "L0"],
            )
            self.assertTrue(
                all(
                    not any(key.startswith("delta_") for key in row)
                    for row in execution_json
                )
            )
            self.assertTrue(
                (root / "figures/legacy_bridge_numeric_ablation_report.png").is_file()
            )
            self.assertTrue(
                (root / "figures/legacy_bridge_execution_order_report.png").is_file()
            )
            markdown = result.summary_markdown.read_text(encoding="utf-8")
            html = result.summary_html.read_text(encoding="utf-8")
            for expected in (
                "bridge report A",
                "bridge report B",
                "L7→L5",
                "causal ablations",
                "figures/legacy_bridge_numeric_ablation_report.png",
                "figures/legacy_bridge_execution_order_report.png",
            ):
                self.assertIn(expected, markdown)
                self.assertIn(expected, html)
            inventory = json.loads(result.output_index.read_text(encoding="utf-8"))
            paths = {row["path"] for row in inventory["artifacts"]}
            self.assertIn("tables/legacy_bridge_numeric_ablation_report.csv", paths)
            self.assertIn("tables/legacy_bridge_execution_order_report.json", paths)


if __name__ == "__main__":
    unittest.main()
