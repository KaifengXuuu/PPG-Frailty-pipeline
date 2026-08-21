"""V2 quality-mode and decision-contract smoke tests / V2质量模式与决策合同测试。"""

from __future__ import annotations

from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ppg_frailty.config import load_config
from ppg_frailty.contracts import QualityState, SignalRoute
from ppg_frailty.experiment import (
    _ExperimentProtocolError,
    _RuntimeRecord,
    _abstention_aware_root_metrics,
    _evaluate_subjects,
    _direct_pulses_for_state,
    _fit_quality_calibrator,
    _fold_confusions_and_rosters_from_subject_rows,
    _participant_predictions_from_subject_rows,
    _persisted_route_artifact_row,
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


class _TierConfig:
    representation_mode = "feature_vector"

    def __init__(self, *, sqi: bool, motion: bool, denoiser: bool = False) -> None:
        self.quality = {"mode": "route" if sqi else "off"}
        self.artifact = {
            "reducer": "pca_bss" if denoiser else "identity",
            "degraded_policy": (
                "denoise_then_extract_rate_features" if denoiser else "drop"
            ),
            "motion_detector_enabled": motion,
            "denoiser_enabled": denoiser,
        }

    def section(self, name: str) -> dict[str, object]:
        if name == "quality":
            return dict(self.quality)
        if name == "artifact":
            return dict(self.artifact)
        if name == "signal":
            return {
                "peak_detector": {
                    "detector_id": "aboy_project_v1",
                    "failure_action": "fail_closed_no_fallback",
                    "min_observation_sec": 8.0,
                    "min_peaks": 5,
                }
            }
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "quality": dict(self.quality),
            "artifact": dict(self.artifact),
        }


class PersistedRouteArtifactTest(unittest.TestCase):
    def test_final_signal_route_is_present_without_quality_diagnostics(self) -> None:
        direct = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_B", role="B"
            ),
            route=SignalRoute.DIRECT,
            retained=True,
            route_status="retained_direct_quality_off",
        )
        dropped = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P02", record_id="P02_S1", role="S1"
            ),
            route=SignalRoute.DROPPED,
            retained=False,
            route_status="degraded_drop",
            reason="q_rate_fail",
        )

        direct_row = _persisted_route_artifact_row(direct)
        dropped_row = _persisted_route_artifact_row(dropped)

        self.assertEqual(direct_row["signal_route"], "direct_x_filter")
        self.assertTrue(direct_row["retained"])
        self.assertEqual(dropped_row["signal_route"], "dropped")
        self.assertFalse(dropped_row["retained"])


class V2QualityModeTest(unittest.TestCase):
    def test_all_abstained_outer_fold_emits_zero_aware_metrics(self) -> None:
        rows = tuple(
            SimpleNamespace(retained=False, label=label, probabilities=())
            for label in (0, 1, 2)
        )

        metrics = _evaluate_subjects(rows, total=3)

        self.assertEqual(metrics["status"], "unavailable_no_retained_participant")
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["confusion_matrix"], [[0, 0, 0]] * 3)
        self.assertEqual(metrics["abstention_aware_balanced_accuracy"], 0.0)
        self.assertEqual(metrics["abstention_aware_macro_f1"], 0.0)
        self.assertEqual(metrics["coverage_rate"], 0.0)
        self.assertEqual(metrics["abstention_count"], 3)

    def test_root_oof_roster_preserves_abstained_participant_label(self) -> None:
        common = {
            "level": "participant",
            "prediction_kind": "single_model",
            "repeat": 0,
            "fold": 0,
        }
        retained = SimpleNamespace(
            **common,
            participant_id="P0",
            label=0,
            retained=True,
            probabilities=(0.9, 0.05, 0.05),
            class_order=(0, 1, 2),
        )
        abstained = SimpleNamespace(
            **common,
            participant_id="P1",
            label=1,
            retained=False,
            probabilities=(),
            class_order=(),
        )

        rows = _participant_predictions_from_subject_rows((retained, abstained))
        matrices, rosters = _fold_confusions_and_rosters_from_subject_rows(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1].label, 1)
        self.assertFalse(rows[1].retained)
        self.assertEqual(sum(map(sum, matrices["r0f0"])), 1)
        self.assertEqual(rosters["r0f0"], ("P0", "P1"))

    def test_complete_root_metrics_accept_all_abstained_folds(self) -> None:
        rows = tuple(
            SimpleNamespace(
                participant_id=f"P{label}",
                label=label,
                repeat=repeat,
                retained=False,
                probabilities=(),
            )
            for repeat in range(5)
            for label in (0, 1, 2)
        )
        summaries = tuple(
            {
                "metrics": {
                    "balanced_accuracy": None,
                    "abstention_aware_balanced_accuracy": 0.0,
                }
            }
            for _ in range(25)
        )
        zero = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
        fold_confusions = {
            f"r{repeat}f{fold}": zero
            for repeat in range(5)
            for fold in range(5)
        }

        metrics = _abstention_aware_root_metrics(
            rows,
            summaries,
            config_id="case",
            registry_role="comparison",
            fold_confusions=fold_confusions,
            operational={
                "inference_cost": {"not_measured": None},
                "parameter_count": None,
                "eligible": False,
                "exclusion_reason": "not measured",
            },
        )

        self.assertIsNone(metrics["participant_mean_balanced_accuracy"])
        self.assertEqual(
            metrics["participant_mean_abstention_aware_balanced_accuracy"],
            0.0,
        )
        self.assertEqual(metrics["abstention_count"], 15)
        self.assertEqual(metrics["participant_mean_coverage_rate"], 0.0)

    def test_direct_peak_detection_is_reused(self) -> None:
        calls = 0

        def detect(*_args: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return {"RED": object(), "IR": object()}

        state = _RuntimeRecord(row=SimpleNamespace(), views=object())
        api = {"detect_pulses_per_wavelength": detect}
        detector = {
            "detector_id": "aboy_project_v1",
            "min_observation_sec": 8.0,
            "min_peaks": 5,
        }
        first = _direct_pulses_for_state(state, api, detector)
        second = _direct_pulses_for_state(state, api, detector)
        self.assertIs(first, second)
        self.assertEqual(calls, 1)

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

    def test_runtime_high_motion_overrides_passing_rate_and_morph_to_unfit(self) -> None:
        quality = SimpleNamespace(
            q_rate=SimpleNamespace(state=QualityState.PASS, score=0.9, coverage=1.0),
            q_morph=SimpleNamespace(state=QualityState.PASS, score=0.9, coverage=1.0),
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_B", role="B"
            ),
            views=SimpleNamespace(x_filter=np.zeros((4000, 2))),
        )
        api = __import__(
            "ppg_frailty.experiment", fromlist=["_runtime_imports"]
        )._runtime_imports()
        api.update(
            {
                "detect_pulses_per_wavelength": lambda *_a, **_k: {
                    "RED": object(), "IR": object()
                },
                "select_reference_wavelength": lambda _rows: "RED",
                "run_quality_mode": lambda *_a, **_k: SimpleNamespace(
                    result=quality
                ),
                "motion_recording_from_signal_views": lambda *_a, **_k: object(),
                "infer_reused_motion_recording": lambda *_a, **_k: SimpleNamespace(
                    motion_state="high_motion",
                    record_probability=0.9,
                    threshold=0.5,
                    window_count=4,
                    reason="high",
                ),
            }
        )
        motion_detector = SimpleNamespace(
            provenance={
                "execution": "inference_only_no_fit_no_recalibration",
                "training_scope": "frailty29_all_participants",
                "reuse_scope": "all29_reused",
                "frailty29_evaluation_relation": "in_sample_for_frailty29",
                "valid_outer_oof_claim": False,
                "evidence_path": "evidence.json",
                "evidence_sha256": "a" * 64,
                "model_artifact_sha256": "b" * 64,
                "ekf_config_sha256": "c" * 64,
                "frozen_bundle_threshold_sha256": "c" * 64,
                "threshold_source": "bundle_frozen",
                "runtime_device": "cpu",
                "window_probability_aggregation": "median",
            }
        )
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records(
                [state],
                _TierConfig(sqi=True, motion=True),
                SimpleNamespace(artifact={}),
                SimpleNamespace(calibrator="fixed_formula_thresholds_v1"),
                None,
                motion_detector=motion_detector,
            )
        self.assertFalse(state.retained)
        self.assertEqual(state.quality_tier, "unfit")
        self.assertEqual(state.route_status, "degraded_drop")
        self.assertTrue(state.route_artifact["abstained"])

    def test_runtime_sqi_off_low_motion_is_excellent(self) -> None:
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_S1", role="S1"
            ),
            views=SimpleNamespace(x_filter=np.zeros((4000, 2))),
        )
        api = __import__(
            "ppg_frailty.experiment", fromlist=["_runtime_imports"]
        )._runtime_imports()
        api.update(
            {
                "detect_pulses_per_wavelength": lambda *_a, **_k: {
                    "RED": object(), "IR": object()
                },
                "select_reference_wavelength": lambda _rows: "RED",
                "motion_recording_from_signal_views": lambda *_a, **_k: object(),
                "infer_reused_motion_recording": lambda *_a, **_k: SimpleNamespace(
                    motion_state="low_motion",
                    record_probability=0.1,
                    threshold=0.5,
                    window_count=4,
                    reason="low",
                ),
            }
        )
        motion_detector = SimpleNamespace(
            provenance={
                "execution": "inference_only_no_fit_no_recalibration",
                "training_scope": "frailty29_all_participants",
                "reuse_scope": "all29_reused",
                "frailty29_evaluation_relation": "in_sample_for_frailty29",
                "valid_outer_oof_claim": False,
                "evidence_path": "evidence.json",
                "evidence_sha256": "a" * 64,
                "model_artifact_sha256": "b" * 64,
                "ekf_config_sha256": "c" * 64,
                "frozen_bundle_threshold_sha256": "c" * 64,
                "threshold_source": "bundle_frozen",
                "runtime_device": "cpu",
                "window_probability_aggregation": "median",
            }
        )
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records(
                [state],
                _TierConfig(sqi=False, motion=True),
                SimpleNamespace(artifact={}),
                None,
                None,
                motion_detector=motion_detector,
            )
        self.assertTrue(state.retained)
        self.assertEqual(state.quality_tier, "excellent")
        self.assertEqual(state.route_status, "full_direct")

    def test_unfit_runs_exactly_one_configured_denoiser_then_q_rate(self) -> None:
        direct = SimpleNamespace(
            q_rate=SimpleNamespace(state=QualityState.PASS, score=0.9, coverage=1.0),
            q_morph=SimpleNamespace(state=QualityState.PASS, score=0.9, coverage=1.0),
        )
        post = SimpleNamespace(
            q_rate=SimpleNamespace(state=QualityState.PASS, score=0.8, coverage=1.0),
            q_morph=SimpleNamespace(
                state=QualityState.NOT_APPLICABLE, score=None, coverage=1.0
            ),
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_B", role="B"
            ),
            views=SimpleNamespace(x_filter=np.zeros((4000, 2))),
        )
        calls = {"quality": 0, "denoiser": 0}

        def quality_mode(*_args: object, **_kwargs: object) -> object:
            calls["quality"] += 1
            return SimpleNamespace(result=direct if calls["quality"] == 1 else post)

        reduced_views = SimpleNamespace(x_filter=np.zeros((4000, 2)))

        def denoise(*_args: object, **_kwargs: object) -> object:
            calls["denoiser"] += 1
            return SimpleNamespace(
                result=SimpleNamespace(
                    reducer_id="pca_bss",
                    reducer_version="pca_component_select_v2",
                    status="success",
                    reasons=(),
                ),
                views=reduced_views,
                route=SignalRoute.ARTIFACT_RATE_ONLY,
            )

        api = __import__(
            "ppg_frailty.experiment", fromlist=["_runtime_imports"]
        )._runtime_imports()
        api.update(
            {
                "detect_pulses_per_wavelength": lambda *_a, **_k: {
                    "RED": object(), "IR": object()
                },
                "select_reference_wavelength": lambda _rows: "RED",
                "run_quality_mode": quality_mode,
                "run_artifact_route": denoise,
                "motion_recording_from_signal_views": lambda *_a, **_k: object(),
                "infer_reused_motion_recording": lambda *_a, **_k: SimpleNamespace(
                    motion_state="high_motion",
                    record_probability=0.9,
                    threshold=0.5,
                    window_count=4,
                    reason="high",
                ),
            }
        )
        motion_detector = SimpleNamespace(
            provenance={
                "execution": "inference_only_no_fit_no_recalibration",
                "training_scope": "frailty29_all_participants",
                "reuse_scope": "all29_reused",
                "frailty29_evaluation_relation": "in_sample_for_frailty29",
                "valid_outer_oof_claim": False,
                "evidence_path": "evidence.json",
                "evidence_sha256": "a" * 64,
                "model_artifact_sha256": "b" * 64,
                "ekf_config_sha256": "c" * 64,
                "frozen_bundle_threshold_sha256": "c" * 64,
                "threshold_source": "bundle_frozen",
                "runtime_device": "cpu",
                "window_probability_aggregation": "median",
            }
        )
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records(
                [state],
                _TierConfig(sqi=True, motion=True, denoiser=True),
                SimpleNamespace(
                    artifact={"runtime_reducer": "pca_bss", "parameters": {}}
                ),
                SimpleNamespace(calibrator="fixed_formula_thresholds_v1"),
                None,
                motion_detector=motion_detector,
            )
        self.assertEqual(calls, {"quality": 2, "denoiser": 1})
        self.assertTrue(state.retained)
        self.assertEqual(state.quality_tier, "acceptable")
        self.assertEqual(state.route_status, "rate_only_processed")
        self.assertTrue(state.route_artifact["denoiser_attempted"])

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
