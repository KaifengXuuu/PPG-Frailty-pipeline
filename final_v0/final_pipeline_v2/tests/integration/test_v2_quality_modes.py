"""V2 quality-mode and decision-contract smoke tests / V2质量模式与决策合同测试。"""

from __future__ import annotations

from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ppg_frailty.config import load_config
from ppg_frailty.contracts import SignalRoute
from ppg_frailty.experiment import (
    _ExperimentProtocolError,
    _RuntimeRecord,
    _fit_quality_calibrator,
    _motion_recovery_decision,
    _quality_mode,
    _retain_without_quality_routing,
    _route_records,
)
from ppg_frailty.module_registry import resolve_peak_detector_config
from ppg_frailty.v2_contract import resolve_balance_line, validate_quality_mode
from ppg_frailty.signal.sqi import SqiDiagnosticComponent, SqiDiagnostics
from ppg_frailty.signal.views import CanonicalSignalViews
from ppg_frailty.quality.routing import run_quality_mode


ROOT = Path(__file__).resolve().parents[2]


class _Config:
    """Minimal explicit config facade / 最小显式配置接口。"""

    def __init__(
        self,
        mode: str,
        *,
        min_observation_sec: float = 8.0,
        min_peaks: int = 5,
    ) -> None:
        self.mode = mode
        self.min_observation_sec = min_observation_sec
        self.min_peaks = min_peaks

    def section(self, name: str) -> dict[str, object]:
        if name == "quality":
            return {"mode": self.mode}
        if name == "artifact":
            return {"reducer": "identity"}
        if name == "signal":
            return {
                "peak_detector": {
                    "detector_id": "aboy_project_v1",
                    "failure_action": "fail_closed_no_fallback",
                    "min_observation_sec": self.min_observation_sec,
                    "min_peaks": self.min_peaks,
                }
            }
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {"quality": self.section("quality")}


class _SqiDiagnosticConfig:
    """Diagnostics-only fixture facade / 仅诊断测试桩。"""

    @staticmethod
    def from_resolved(_: object) -> object:
        return SimpleNamespace(calibrator="unresolved")


class _RouteConfig:
    """Minimal real route config with no readiness authorization field."""

    def __init__(self) -> None:
        self.quality = {
            "mode": "route",
            "calibrator": "outer_train_empirical_quantiles_v1",
            "rate_threshold": 0.0,
            "morph_threshold": 0.0,
            "minimum_coverage": 0.5,
            "calibrator_quantiles": [0.2, 0.8],
        }

    def section(self, name: str) -> dict[str, object]:
        if name == "quality":
            return dict(self.quality)
        if name == "signal":
            return {
                "peak_detector": {
                    "detector_id": "aboy_project_v1",
                    "failure_action": "fail_closed_no_fallback",
                }
            }
        if name == "artifact":
            return {
                "reducer": "spectral_mask",
                "reducer_version": "spectral_mask_v1",
                "degraded_policy": "drop",
                "motion_detector_enabled": False,
            }
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {"quality": dict(self.quality)}


class V2QualityModeTest(unittest.TestCase):
    """Prove SQI off/diagnostics cannot route or drop / 证明两种模式不分流。"""

    def test_off_retains_without_evaluating_sqi(self) -> None:
        state = _RuntimeRecord(
            row=SimpleNamespace(record_id="P01_B"),
            views=SimpleNamespace(x_filter=SimpleNamespace(shape=(1,))),
        )
        with patch(
            "ppg_frailty.experiment._runtime_imports",
            return_value={
                "SignalRoute": SignalRoute,
                "run_quality_mode": run_quality_mode,
                "evaluate_quality_diagnostics": lambda *_a, **_k: None,
                "resolve_peak_detector_config": resolve_peak_detector_config,
            },
        ):
            provenance = _retain_without_quality_routing(
                [state], _Config("off"), diagnostics_only=False
            )
        self.assertTrue(state.retained)
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertEqual(state.route_status, "retained_direct_quality_off")
        self.assertEqual(provenance["classification_effect"], "none")

    def test_diagnostic_failure_never_drops_record(self) -> None:
        state = _RuntimeRecord(
            row=SimpleNamespace(record_id="P01_R1"),
            views=SimpleNamespace(x_filter=SimpleNamespace(shape=(1,))),
        )

        def fail_diagnostic(*_: object, **__: object) -> object:
            raise RuntimeError("diagnostic fixture failure")

        api = {
            "SignalRoute": SignalRoute,
            "SqiDiagnosticConfig": _SqiDiagnosticConfig,
            "evaluate_quality_diagnostics": fail_diagnostic,
            "run_quality_mode": run_quality_mode,
            "resolve_peak_detector_config": resolve_peak_detector_config,
        }
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            provenance = _retain_without_quality_routing(
                [state], _Config("diagnostics_only"), diagnostics_only=True
            )
        self.assertTrue(state.retained)
        self.assertIsNone(state.reason)
        self.assertIn("diagnostics_only_failed", state.diagnostic_reason or "")
        self.assertEqual(provenance["classification_effect"], "none")

    def test_diagnostics_archives_raw_values_without_decision_fields(self) -> None:
        state = _RuntimeRecord(
            row=SimpleNamespace(record_id="P01_R1"),
            views=SimpleNamespace(x_filter=SimpleNamespace(shape=(1,))),
        )
        observed = SqiDiagnostics(
            components={
                "snr_db": SqiDiagnosticComponent(12.5, True, "observed"),
            },
            coverage=1.0,
            route="direct",
            reasons=(),
        )
        observed_kwargs: dict[str, object] = {}

        def evaluate_fixture(*_args: object, **kwargs: object) -> SqiDiagnostics:
            observed_kwargs.update(kwargs)
            return observed

        api = {
            "SignalRoute": SignalRoute,
            "SqiDiagnosticConfig": _SqiDiagnosticConfig,
            "evaluate_quality_diagnostics": evaluate_fixture,
            "run_quality_mode": run_quality_mode,
            "resolve_peak_detector_config": resolve_peak_detector_config,
        }
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            provenance = _retain_without_quality_routing(
                [state],
                _Config(
                    "diagnostics_only",
                    min_observation_sec=6.5,
                    min_peaks=3,
                ),
                diagnostics_only=True,
            )
        self.assertTrue(state.retained)
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertEqual(
            state.diagnostic_components["components"]["snr_db"]["raw_value"],
            12.5,
        )
        for name in (
            "aggregation_performed",
            "weights_applied",
            "endpoint_thresholds_applied",
            "affects_classification",
        ):
            self.assertFalse(state.diagnostic_components[name])
        self.assertEqual(provenance["classification_effect"], "none")
        self.assertEqual(observed_kwargs["detector_id"], "aboy_project_v1")
        self.assertEqual(observed_kwargs["min_observation_sec"], 6.5)
        self.assertEqual(observed_kwargs["min_peaks"], 3)
        self.assertEqual(
            provenance["method"],
            "diagnostics_only_raw_components_no_weights_thresholds_or_fit",
        )

    def test_real_reference_config_runs_synthetic_diagnostics_without_training(
        self,
    ) -> None:
        config = load_config(
            ROOT / "configs" / "reference_static_role_aware_v2.yaml"
        )
        samples = 4000
        time = np.arange(samples, dtype=np.float64) / 400.0
        filtered = np.column_stack(
            (
                np.sin(2.0 * np.pi * 1.2 * time),
                0.8 * np.sin(2.0 * np.pi * 1.2 * time + 0.08),
            )
        )
        views = CanonicalSignalViews(
            x_native=filtered + np.array([1000.0, 1200.0]),
            x_filter=filtered,
            x_analysis_rate=filtered.copy(),
            imu_processed={
                "dynamic_magnitude": np.full(samples, 0.05, dtype=np.float64),
            },
            metadata={"record_id": "synthetic_diagnostics", "fs_hz": 400.0},
            source_valid_mask=np.ones_like(filtered, dtype=bool),
            repair_mask=np.zeros_like(filtered, dtype=bool),
            route=SignalRoute.DIRECT,
        )
        views.validate()
        state = _RuntimeRecord(
            row=SimpleNamespace(record_id="synthetic_diagnostics"),
            views=views,
        )

        provenance = _retain_without_quality_routing(
            [state],
            config,
            diagnostics_only=True,
        )

        self.assertTrue(state.retained)
        self.assertIs(state.route, SignalRoute.DIRECT)
        self.assertIsNone(state.diagnostic_reason)
        self.assertIsInstance(state.direct_quality, SqiDiagnostics)
        self.assertIn(
            "rate.cardiac_concentration",
            state.diagnostic_components["components"],
        )
        self.assertEqual(provenance["classification_effect"], "none")

    def test_route_is_selectable_and_retired_gate_input_is_not_an_api_slot(self) -> None:
        self.assertEqual(_quality_mode(_Config("off")), "off")
        self.assertEqual(_quality_mode(_Config("diagnostics_only")), "diagnostics_only")
        self.assertEqual(_quality_mode(_Config("route")), "route")
        with self.assertRaisesRegex(TypeError, "supervised_route_ready"):
            validate_quality_mode(
                "route",
                supervised_route_ready=False,
            )
        with self.assertRaisesRegex(TypeError, "supervised_route_ready"):
            validate_quality_mode("route", supervised_route_ready=True)
        with self.assertRaisesRegex(_ExperimentProtocolError, "unsupported_quality_mode"):
            _quality_mode(_Config("automatic"))

    def test_route_runtime_fits_train_only_and_routes_oof_without_training(self) -> None:
        fixed_config = _RouteConfig()
        fixed_config.quality["calibrator"] = "fixed_formula_thresholds_v1"
        fixed_sqi, fixed_calibrator = _fit_quality_calibrator(
            [],
            fixed_config,
            ("train_1",),
            ("oof_1",),
        )
        self.assertEqual(fixed_sqi.calibrator, "fixed_formula_thresholds_v1")
        self.assertIsNone(fixed_calibrator)

        samples = 4000
        time = np.arange(samples, dtype=np.float64) / 400.0
        filtered = np.column_stack(
            (
                np.sin(2.0 * np.pi * 1.2 * time),
                0.8 * np.sin(2.0 * np.pi * 1.2 * time + 0.08),
            )
        )

        def views(record_id: str) -> CanonicalSignalViews:
            result = CanonicalSignalViews(
                x_native=filtered + np.array([1000.0, 1200.0]),
                x_filter=filtered,
                x_analysis_rate=filtered.copy(),
                imu_processed={
                    "dynamic_magnitude": np.full(samples, 0.05, dtype=np.float64),
                },
                metadata={"record_id": record_id, "fs_hz": 400.0},
                source_valid_mask=np.ones_like(filtered, dtype=bool),
                repair_mask=np.zeros_like(filtered, dtype=bool),
                route=SignalRoute.DIRECT,
            )
            result.validate()
            return result

        states = [
            _RuntimeRecord(
                row=SimpleNamespace(
                    participant_id="train_1",
                    record_id="train_1_B",
                    role="B",
                ),
                views=views("train_1_B"),
            ),
            _RuntimeRecord(
                row=SimpleNamespace(
                    participant_id="oof_1",
                    record_id="oof_1_B",
                    role="B",
                ),
                views=views("oof_1_B"),
            ),
        ]
        config = _RouteConfig()
        sqi_config, calibrator = _fit_quality_calibrator(
            states,
            config,
            ("train_1",),
            ("oof_1",),
        )
        self.assertEqual(calibrator.fitted_on_participant_ids, ("train_1",))
        self.assertEqual(
            (
                sqi_config.calibrator_lower_quantile,
                sqi_config.calibrator_upper_quantile,
            ),
            (0.2, 0.8),
        )
        _route_records(
            states,
            config,
            SimpleNamespace(
                artifact={"runtime_reducer": "identity", "parameters": {}}
            ),
            sqi_config,
            calibrator,
        )
        self.assertTrue(all(state.retained for state in states))
        self.assertTrue(all(state.route is SignalRoute.DIRECT for state in states))
        self.assertTrue(all(state.artifact_name == "identity" for state in states))
        self.assertTrue(
            all(state.route_artifact["state"] == "full_direct" for state in states)
        )
        self.assertTrue(
            all(
                state.route_artifact["configured_reducer_not_executed"]
                == "spectral_mask"
                for state in states
            )
        )

    def test_motion_recovery_uses_signal_component_not_role_identity(self) -> None:
        failed_motion = SimpleNamespace(
            components={
                "rate.motion_energy_rms": SimpleNamespace(
                    state=SimpleNamespace(value="fail"),
                    raw_value=3.2,
                    normalized_value=0.4,
                )
            }
        )
        passed_motion = SimpleNamespace(
            components={
                "rate.motion_energy_rms": SimpleNamespace(
                    state=SimpleNamespace(value="pass"),
                    raw_value=0.1,
                    normalized_value=0.98,
                )
            }
        )
        recover, evidence = _motion_recovery_decision(
            failed_motion,
            detector_enabled=True,
            degraded_policy="denoise_then_extract_rate_features",
        )
        self.assertTrue(recover)
        self.assertTrue(evidence["motion_detected"])
        recover, _ = _motion_recovery_decision(
            passed_motion,
            detector_enabled=True,
            degraded_policy="denoise_then_extract_rate_features",
        )
        self.assertFalse(recover)
        recover, _ = _motion_recovery_decision(
            passed_motion,
            detector_enabled=False,
            degraded_policy="denoise_then_extract_rate_features",
        )
        self.assertTrue(recover)

    def test_training_and_reporting_balance_are_independent_modules(self) -> None:
        resolved = resolve_balance_line(
            "line_a_equal_files",
            training_balance="equal_role_families",
            aggregation="equal_files_no_role_layer",
        )
        self.assertEqual(resolved.line_id, "line_a_equal_files")
        self.assertEqual(resolved.training_balance, "equal_role_families")
        self.assertEqual(resolved.aggregation, "equal_files_no_role_layer")
        with self.assertRaisesRegex(ValueError, "unknown training balance"):
            resolve_balance_line(
                "line_b_equal_role_families",
                training_balance="invented",
                aggregation="equal_role_families",
            )


if __name__ == "__main__":
    unittest.main()
