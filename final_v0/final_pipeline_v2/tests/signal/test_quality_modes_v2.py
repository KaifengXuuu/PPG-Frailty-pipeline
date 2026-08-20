"""V2 quality-mode policy tests; no SQI threshold experiment is run."""

from __future__ import annotations

import ast
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from ppg_frailty.contracts import QualityResult, SignalRoute
from ppg_frailty.peaks import CANONICAL_DETECTOR_ID
from ppg_frailty.quality import (
    QualityMode,
    SqiConfig,
    evaluate_quality,
    evaluate_quality_diagnostics,
    quality_mode_from_config,
    run_quality_mode,
)


class QualityModeTests(unittest.TestCase):
    def test_sqi_forwards_configured_peak_thresholds(self) -> None:
        from ppg_frailty.peaks.resolver import detect_pulses as implementation

        time = np.arange(0.0, 20.0, 1.0 / 400.0)
        values = np.column_stack(
            (
                np.sin(2.0 * np.pi * 1.2 * time),
                0.8 * np.sin(2.0 * np.pi * 1.2 * time + 0.03),
            )
        )
        observed: dict[str, object] = {}

        def capture(*args: object, **kwargs: object) -> object:
            observed.update(kwargs)
            return implementation(*args, **kwargs)

        with patch("ppg_frailty.peaks.detect_pulses", side_effect=capture):
            evaluate_quality(
                values,
                route=SignalRoute.DIRECT,
                detector_id=CANONICAL_DETECTOR_ID,
                min_observation_sec=6.5,
                min_peaks=3,
                config=SqiConfig(),
            )
        self.assertEqual(observed["min_observation_sec"], 6.5)
        self.assertEqual(observed["min_peaks"], 3)

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

    def test_route_evaluates_and_declares_classification_effects(self) -> None:
        sentinel = object.__new__(QualityResult)
        values = object()
        evaluator = Mock(return_value=sentinel)
        outcome = run_quality_mode(
            values,
            mode="route",
            evaluator=evaluator,
            threshold=0.42,
        )
        evaluator.assert_called_once_with(values, threshold=0.42)
        self.assertIs(outcome.result, sentinel)
        self.assertEqual(outcome.classification_action, "apply_explicit_route_policy")
        self.assertTrue(outcome.affects_retention)
        self.assertTrue(outcome.affects_aggregation)
        self.assertTrue(outcome.affects_prediction)

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

    def test_partial_route_config_resolves_defaults_and_nested_parameters(self) -> None:
        config = SqiConfig.from_resolved(
            {
                "quality": {
                    "mode": "route",
                    "rate_threshold": None,
                    "morph_threshold": 0.73,
                    "calibrator": "deferred_supervised_design",
                    "calibrator_quantiles": [0.2, 0.8],
                    "rate_component_weights": {
                        "source_coverage": 0.7,
                    },
                    "component_normalization": {
                        "motion_rms_scale": 2.5,
                    },
                }
            }
        )
        self.assertEqual(config.q_rate_threshold, 0.50)
        self.assertEqual(config.q_morph_threshold, 0.73)
        self.assertEqual(config.calibrator, "outer_train_empirical_quantiles_v1")
        self.assertEqual(
            (config.calibrator_lower_quantile, config.calibrator_upper_quantile),
            (0.2, 0.8),
        )
        self.assertEqual(config.rate_component_weights["source_coverage"], 0.7)
        self.assertEqual(config.motion_rms_scale, 2.5)

    def test_configured_component_weights_are_executed(self) -> None:
        time = np.arange(0.0, 20.0, 1.0 / 400.0)
        values = np.column_stack(
            (
                np.sin(2.0 * np.pi * 1.2 * time),
                0.8 * np.sin(2.0 * np.pi * 1.2 * time + 0.03),
            )
        )
        weights = {name: 0.0 for name in SqiConfig().rate_component_weights}
        weights["source_coverage"] = 1.0
        config = SqiConfig(
            q_rate_threshold=0.99,
            rate_component_weights=weights,
        )
        result = evaluate_quality(
            values,
            route=SignalRoute.DIRECT,
            detector_id=CANONICAL_DETECTOR_ID,
            config=config,
        )
        self.assertAlmostEqual(result.q_rate.score or 0.0, 1.0)
        self.assertEqual(result.q_rate.threshold, 0.99)

    def test_nonfinite_or_invalid_route_parameters_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            SqiConfig(q_rate_threshold=float("nan")).validate()
        weights = dict(SqiConfig().rate_component_weights)
        weights["source_coverage"] = -0.1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SqiConfig(rate_component_weights=weights).validate()

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
