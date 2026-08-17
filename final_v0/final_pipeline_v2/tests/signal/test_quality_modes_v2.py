"""V2 quality-mode policy tests; no SQI threshold experiment is run."""

from __future__ import annotations

import ast
import unittest
from unittest.mock import Mock
from pathlib import Path

import numpy as np

from ppg_frailty.contracts import SignalRoute
from ppg_frailty.peaks import CANONICAL_DETECTOR_ID
from ppg_frailty.quality import (
    QualityMode,
    QualityRoutingDisabledError,
    SqiConfig,
    evaluate_quality_diagnostics,
    quality_mode_from_config,
    run_quality_mode,
)


class QualityModeTests(unittest.TestCase):
    def test_default_off_never_calls_evaluator(self) -> None:
        evaluator = Mock(side_effect=AssertionError("SQI must not be computed"))
        outcome = run_quality_mode(object(), evaluator=evaluator)
        self.assertIs(outcome.mode, QualityMode.OFF)
        self.assertFalse(outcome.computed)
        self.assertIsNone(outcome.result)
        self.assertEqual(outcome.classification_action, "keep_unchanged")
        evaluator.assert_not_called()

    def test_diagnostics_only_computes_but_cannot_route(self) -> None:
        sentinel = object()
        evaluator = Mock(return_value=sentinel)
        outcome = run_quality_mode(
            "signal",
            mode="diagnostics_only",
            evaluator=evaluator,
            config="diagnostic-config",
        )
        self.assertIs(outcome.result, sentinel)
        self.assertFalse(outcome.affects_retention)
        self.assertFalse(outcome.affects_aggregation)
        self.assertFalse(outcome.affects_prediction)
        evaluator.assert_called_once_with("signal", config="diagnostic-config")

    def test_route_is_disabled_before_any_evaluation(self) -> None:
        evaluator = Mock(side_effect=AssertionError("disabled route must fail first"))
        with self.assertRaisesRegex(QualityRoutingDisabledError, "supervised routing"):
            run_quality_mode(object(), mode="route", evaluator=evaluator)
        evaluator.assert_not_called()

    def test_config_omission_means_off_and_unknown_mode_fails(self) -> None:
        self.assertIs(quality_mode_from_config({"quality": {}}), QualityMode.OFF)
        with self.assertRaises(ValueError):
            quality_mode_from_config({"quality": {"mode": "automatic"}})

    def test_diagnostics_have_raw_components_without_endpoint_policy(self) -> None:
        time = np.arange(0.0, 20.0, 1.0 / 400.0)
        values = np.column_stack(
            (
                np.sin(2.0 * np.pi * 1.2 * time),
                0.8 * np.sin(2.0 * np.pi * 1.2 * time + 0.03),
            )
        )
        diagnostics = evaluate_quality_diagnostics(
            values,
            route=SignalRoute.DIRECT,
            detector_id=CANONICAL_DETECTOR_ID,
            # Deliberately invalid endpoint thresholds prove they are not read.
            config=SqiConfig(q_rate_threshold=-10.0, q_morph_threshold=10.0),
        )
        diagnostics.validate()
        self.assertIn("rate.cardiac_concentration", diagnostics.components)
        self.assertIn("morph.template_correlation", diagnostics.components)
        self.assertFalse(diagnostics.aggregation_performed)
        self.assertFalse(diagnostics.weights_applied)
        self.assertFalse(diagnostics.endpoint_thresholds_applied)
        self.assertFalse(diagnostics.affects_classification)

    def test_sqi_source_has_unique_diagnostic_class_definitions(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "ppg_frailty"
            / "signal"
            / "sqi.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name
            in {"SqiDiagnosticConfig", "SqiDiagnosticComponent", "SqiDiagnostics"}
        ]
        self.assertEqual(
            names,
            ["SqiDiagnosticConfig", "SqiDiagnosticComponent", "SqiDiagnostics"],
        )


if __name__ == "__main__":
    unittest.main()
