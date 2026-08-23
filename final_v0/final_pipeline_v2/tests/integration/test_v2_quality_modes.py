"""V2 quality-mode and decision-contract smoke tests / V2质量模式与决策合同测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ppg_frailty.config import load_config
from ppg_frailty.contracts import (
    PulseResult,
    QualityEndpoint,
    QualityResult,
    QualityState,
    SignalRoute,
)
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
    _route_records_window_level,
)
from ppg_frailty.module_registry import resolve_peak_detector_config
from ppg_frailty.v2_contract import resolve_balance_line, validate_quality_mode
from ppg_frailty.signal.sqi import SqiDiagnosticComponent, SqiDiagnostics
from ppg_frailty.signal.views import CanonicalSignalViews
from ppg_frailty.quality.routing import run_quality_mode
from ppg_frailty.quality.motion_bundle_adapter import (
    MotionWindowDecision,
    MotionWindowSeries,
)


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
        if name == "routing":
            return {
                "window_s": 8.0,
                "hop_s": 2.0,
                "fs_hz": 400.0,
                "source_grid": "canonical_acquisition_grid",
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

        direct_row = _persisted_route_artifact_row(
            direct,
            train_participant_ids=("P01",),
            oof_participant_ids=("P02",),
        )
        dropped_row = _persisted_route_artifact_row(
            dropped,
            train_participant_ids=("P01",),
            oof_participant_ids=("P02",),
        )

        self.assertEqual(direct_row["signal_route"], "direct_x_filter")
        self.assertEqual(direct_row["outer_partition"], "outer_train")
        self.assertTrue(direct_row["retained"])
        self.assertEqual(dropped_row["signal_route"], "dropped")
        self.assertEqual(dropped_row["outer_partition"], "outer_oof")
        self.assertFalse(dropped_row["retained"])


class V2QualityModeTest(unittest.TestCase):
    def test_window_router_runs_one_whole_record_reducer_and_preserves_direct_cells(self) -> None:
        samples = 4_000
        time = np.arange(samples, dtype=np.float64) / 400.0
        filtered = np.column_stack(
            (np.sin(2.0 * np.pi * time), np.cos(2.0 * np.pi * time))
        )
        direct_views = CanonicalSignalViews(
            x_native=filtered + 100.0,
            x_filter=filtered,
            x_analysis_rate=filtered.copy(),
            imu_processed={
                "dynamic_acc_mps2": np.column_stack(
                    (
                        np.full(samples, 0.1, dtype=np.float64),
                        np.zeros(samples, dtype=np.float64),
                        np.zeros(samples, dtype=np.float64),
                    )
                ),
                "gyro_rads": np.zeros((samples, 3), dtype=np.float64),
                "dynamic_magnitude": np.full(samples, 0.1, dtype=np.float64),
                "gyro_magnitude": np.zeros(samples, dtype=np.float64),
                "jerk_magnitude": np.zeros(samples, dtype=np.float64),
            },
            metadata={"record_id": "P01_B", "fs_hz": 400.0},
            source_valid_mask=np.ones_like(filtered, dtype=bool),
            repair_mask=np.zeros_like(filtered, dtype=bool),
            route=SignalRoute.DIRECT,
        )
        direct_views.validate()
        processed_views = replace(
            direct_views,
            x_ar=filtered * 0.9,
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            metadata={
                **direct_views.metadata,
                "non_identity_artifact_reduction": True,
                "rate_only": True,
                "q_morph_state": "not_applicable",
                "artifact_output_valid_mask": np.ones(samples, dtype=bool),
            },
        )
        processed_views.validate()

        def pulse(route: SignalRoute, run_id: str, wavelength: str) -> PulseResult:
            peak_times = np.arange(0.5, 10.0, 0.5)
            interval_count = peak_times.size - 1
            return PulseResult(
                peaks=np.rint(peak_times * 400).astype(np.int64),
                peak_timestamps_s=peak_times,
                accepted_peak_mask=np.ones(peak_times.size, dtype=bool),
                interval_start_peak_indices=np.arange(interval_count),
                interval_stop_peak_indices=np.arange(1, peak_times.size),
                ppi_s=np.full(interval_count, 0.5),
                valid_interval_mask=np.ones(interval_count, dtype=bool),
                adjacency_mask=np.ones(interval_count, dtype=bool),
                wavelength=wavelength,
                detector_version="fixture",
                confidence=np.ones(peak_times.size),
                source_route=route,
                detection_run_id=run_id,
                interval_run_ids=np.asarray([run_id] * interval_count),
                detector_id="aboy_project_v1",
                selected_polarity=1,
                block_hri_provenance_hash="a" * 64,
                interval_rejection_reasons=tuple("accepted" for _ in range(interval_count)),
                peak_ordinals=np.arange(peak_times.size),
                detector_score=1.0 if wavelength == "RED" else 0.9,
                detector_coverage=1.0,
            )

        direct_pulse = pulse(SignalRoute.DIRECT, "direct-global-red", "RED")
        direct_ir_pulse = pulse(SignalRoute.DIRECT, "direct-global-ir", "IR")
        processed_pulse = pulse(
            SignalRoute.ARTIFACT_RATE_ONLY, "processed-global-red", "RED"
        )
        processed_ir_pulse = pulse(
            SignalRoute.ARTIFACT_RATE_ONLY, "processed-global-ir", "IR"
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_B", role="B"
            ),
            views=direct_views,
            direct_pulses_per_wavelength={
                "RED": direct_pulse,
                "IR": direct_ir_pulse,
            },
        )

        def endpoint(
            state_value: QualityState, score: float | None
        ) -> QualityEndpoint:
            return QualityEndpoint(
                score=score,
                state=state_value,
                threshold=None if score is None else 0.5,
                components={},
                reasons=(),
                coverage=1.0,
            )

        direct_quality = QualityResult(
            q_rate=endpoint(QualityState.PASS, 0.9),
            q_morph=endpoint(QualityState.PASS, 0.9),
            state="pass",
            components={},
            reasons=(),
            coverage=1.0,
        )
        post_quality = QualityResult(
            q_rate=endpoint(QualityState.PASS, 0.8),
            q_morph=endpoint(QualityState.NOT_APPLICABLE, None),
            state="pass",
            components={},
            reasons=("rate_only",),
            coverage=1.0,
        )

        class Config:
            representation_mode = "raw"
            sha256 = "f" * 64

            def section(self, name: str) -> dict[str, object]:
                return {
                    "quality": {"mode": "route"},
                    "artifact": {
                        "motion_detector_enabled": True,
                        "denoiser_enabled": True,
                        "reducer": "pca_bss",
                    },
                    "routing": {
                        "window_s": 8.0,
                        "hop_s": 2.0,
                        "fs_hz": 400.0,
                        "source_grid": "canonical_acquisition_grid",
                    },
                    "signal": {
                        "peak_detector": {
                            "detector_id": "aboy_project_v1",
                            "failure_action": "fail_closed_no_fallback",
                            "min_observation_sec": 8.0,
                            "min_peaks": 5,
                        }
                    },
                }[name]

            def to_dict(self) -> dict[str, object]:
                return {
                    "quality": self.section("quality"),
                    "artifact": self.section("artifact"),
                }

        motion_series = MotionWindowSeries(
            decisions=(
                MotionWindowDecision(
                    "P01_B::routing_000000", 0, 3200, 1600,
                    0.1, 0.5, "low", "low",
                ),
                MotionWindowDecision(
                    "P01_B::routing_000001", 800, 4000, 2400,
                    0.9, 0.5, "high", "high",
                ),
            ),
            threshold=0.5,
            file_median_probability_diagnostic=0.5,
            reason="diagnostic_only",
        )
        motion_series.validate()
        motion_detector = SimpleNamespace(
            provenance={
                "execution": "inference_only_no_fit_no_recalibration",
                "training_scope": "matching_fold",
                "reuse_scope": "matching_outer_fold_reused",
                "frailty29_evaluation_relation": "held_out",
                "valid_outer_oof_claim": True,
                "evidence_path": "evidence.json",
                "evidence_sha256": "a" * 64,
                "model_artifact_sha256": "b" * 64,
                "model_input_schema_sha256": "c" * 64,
                "ekf_config_sha256": "d" * 64,
                "frozen_bundle_threshold_sha256": "e" * 64,
                "threshold_source": "bundle_frozen",
                "runtime_device": "cuda",
                "window_probability_aggregation": (
                    "native_windows_file_median_diagnostics_only"
                ),
            }
        )
        calls = {"reducer": 0, "processed_pulses": 0}

        def reduce_once(*_args: object, **_kwargs: object) -> object:
            calls["reducer"] += 1
            return SimpleNamespace(
                result=SimpleNamespace(
                    reducer_id="pca_bss",
                    reducer_version="pca_component_select_v2",
                    status="success",
                ),
                views=processed_views,
                route=SignalRoute.ARTIFACT_RATE_ONLY,
            )

        def processed_pulses(*_args: object, **_kwargs: object) -> object:
            calls["processed_pulses"] += 1
            return {"RED": processed_pulse, "IR": processed_ir_pulse}

        api = __import__(
            "ppg_frailty.experiment", fromlist=["_runtime_imports"]
        )._runtime_imports()
        api.update(
            {
                "evaluate_quality": lambda local, **_kwargs: (
                    post_quality
                    if local.route is SignalRoute.ARTIFACT_RATE_ONLY
                    else direct_quality
                ),
                "motion_recording_from_signal_views": lambda *_a, **_k: object(),
                "infer_reused_motion_windows": lambda *_a, **_k: motion_series,
                "run_artifact_route": reduce_once,
                "detect_pulses_per_wavelength": processed_pulses,
                "select_reference_wavelength": lambda _rows: "RED",
            }
        )
        report = SimpleNamespace(
            artifact={"runtime_reducer": "pca_bss", "parameters": {}},
            peak_detector={
                "detector_id": "aboy_project_v1",
                "failure_action": "fail_closed_no_fallback",
                "min_observation_sec": 8.0,
                "min_peaks": 5,
            },
            window_profiles={
                "engineering": {
                    "window_seconds": 10.0,
                    "hop_seconds": 2.0,
                    "end_alignment": "start",
                    "short_record_action": "reject",
                    "include_padded_tail": False,
                    "max_windows": None,
                    "cap_policy": "not_applicable",
                }
            },
        )
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records_window_level(
                [state],
                Config(),
                report,
                SimpleNamespace(
                    calibrator="fixed_formula_thresholds_v1",
                    to_dict=lambda: {"calibrator": "fixed_formula_thresholds_v1"},
                ),
                None,
                motion_detector=motion_detector,
            )

        self.assertEqual(calls, {"reducer": 1, "processed_pulses": 1})
        self.assertEqual(
            [(cell.final_tier, cell.source_view) for cell in state.routing_timeline.cells],
            [("excellent", "x_filter_400"), ("acceptable", "x_ar_400")],
        )
        self.assertFalse(state.route_artifact["canonical_hybrid_waveform_created"])
        self.assertEqual(state.route_artifact["denoiser_invocation_count"], 1)
        self.assertEqual(
            state.route_artifact["heart_rate_estimator"],
            "60_over_median_valid_ppi_s",
        )
        self.assertEqual(state.route_artifact["direct_valid_ppi_count"], 18)
        self.assertEqual(state.route_artifact["post_denoise_valid_ppi_count"], 18)
        self.assertAlmostEqual(state.route_artifact["direct_hr_bpm"], 120.0)
        self.assertAlmostEqual(state.route_artifact["post_denoise_hr_bpm"], 120.0)
        self.assertAlmostEqual(state.route_artifact["post_minus_direct_hr_bpm"], 0.0)

        from ppg_frailty.experiment import _extract_vector

        _extract_vector(state, report)
        self.assertIsNotNone(state.vector, state.reason)
        self.assertEqual(
            state.vector.provenance["routing_interval_source_routes"],
            [SignalRoute.DIRECT.value, SignalRoute.ARTIFACT_RATE_ONLY.value],
        )
        self.assertEqual(
            state.diagnostic_components["dual_optical_pairing"]["status"],
            "unavailable_mixed_or_excluded_routing_cells",
        )

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
        pulse_calls = {"count": 0}

        def pulses(*_args: object, **_kwargs: object) -> object:
            pulse_calls["count"] += 1
            ppi = (
                np.asarray([1.0, 0.8])
                if pulse_calls["count"] == 1
                else np.asarray([0.75, 0.75])
            )
            return {
                wavelength: SimpleNamespace(
                    ppi_s=ppi,
                    valid_interval_mask=np.ones(ppi.shape, dtype=bool),
                    peaks=np.arange(ppi.size + 1),
                    wavelength=wavelength,
                )
                for wavelength in ("RED", "IR")
            }

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
                "detect_pulses_per_wavelength": pulses,
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
        self.assertEqual(
            state.route_artifact["heart_rate_estimator"],
            "60_over_median_valid_ppi_s",
        )
        self.assertAlmostEqual(state.route_artifact["direct_hr_bpm"], 60.0 / 0.9)
        self.assertAlmostEqual(state.route_artifact["post_denoise_hr_bpm"], 80.0)
        self.assertAlmostEqual(
            state.route_artifact["post_minus_direct_hr_bpm"],
            80.0 - 60.0 / 0.9,
        )
        self.assertEqual(state.route_artifact["direct_valid_ppi_count"], 2)
        self.assertEqual(state.route_artifact["post_denoise_valid_ppi_count"], 2)

    def test_sqi_off_motion_denoiser_records_direct_and_post_hr(self) -> None:
        post = SimpleNamespace(
            q_rate=SimpleNamespace(
                state=QualityState.PASS, score=0.8, coverage=1.0
            ),
            q_morph=SimpleNamespace(
                state=QualityState.NOT_APPLICABLE, score=None, coverage=1.0
            ),
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_S1", role="S1"
            ),
            views=SimpleNamespace(x_filter=np.zeros((4000, 2))),
        )
        calls = {"quality": 0, "denoiser": 0, "pulses": 0}

        def pulses(*_args: object, **_kwargs: object) -> object:
            calls["pulses"] += 1
            ppi = np.asarray(
                [1.0, 1.0] if calls["pulses"] == 1 else [0.75, 0.75]
            )
            return {
                "RED": SimpleNamespace(
                    ppi_s=ppi,
                    valid_interval_mask=np.ones(ppi.shape, dtype=bool),
                    peaks=np.arange(3),
                    wavelength="RED",
                )
            }

        def quality_mode(*_args: object, **_kwargs: object) -> object:
            calls["quality"] += 1
            return SimpleNamespace(result=post)

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
                "detect_pulses_per_wavelength": pulses,
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
        config = _TierConfig(sqi=False, motion=True, denoiser=True)
        config.representation_mode = "raw"
        config.artifact["degraded_policy"] = "denoise_then_compare_rate_exclude"
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records(
                [state],
                config,
                SimpleNamespace(
                    artifact={"runtime_reducer": "pca_bss", "parameters": {}}
                ),
                SimpleNamespace(calibrator="fixed_formula_thresholds_v1"),
                None,
                motion_detector=motion_detector,
            )
        self.assertEqual(calls, {"quality": 1, "denoiser": 1, "pulses": 2})
        self.assertFalse(state.retained)
        self.assertEqual(state.route_status, "rate_only_diagnostic_excluded")
        self.assertAlmostEqual(state.route_artifact["direct_hr_bpm"], 60.0)
        self.assertAlmostEqual(state.route_artifact["post_denoise_hr_bpm"], 80.0)
        self.assertAlmostEqual(
            state.route_artifact["post_minus_direct_hr_bpm"], 20.0
        )

    def test_raw_denoiser_records_hr_then_excludes_rate_only_output(self) -> None:
        direct = SimpleNamespace(
            q_rate=SimpleNamespace(
                state=QualityState.FAIL, score=0.3, coverage=1.0
            ),
            q_morph=SimpleNamespace(
                state=QualityState.PASS, score=0.8, coverage=1.0
            ),
        )
        post = SimpleNamespace(
            q_rate=SimpleNamespace(
                state=QualityState.PASS, score=0.8, coverage=1.0
            ),
            q_morph=SimpleNamespace(
                state=QualityState.NOT_APPLICABLE, score=None, coverage=1.0
            ),
        )
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01", record_id="P01_R1", role="R1"
            ),
            views=SimpleNamespace(x_filter=np.zeros((4000, 2))),
        )
        quality_calls = {"count": 0}
        pulse_calls = {"count": 0}

        def quality_mode(*_args: object, **_kwargs: object) -> object:
            quality_calls["count"] += 1
            return SimpleNamespace(
                result=direct if quality_calls["count"] == 1 else post
            )

        def pulses(*_args: object, **_kwargs: object) -> object:
            pulse_calls["count"] += 1
            ppi = np.asarray(
                [1.0, 1.0]
                if pulse_calls["count"] == 1
                else [0.8, 0.8]
            )
            return {
                "RED": SimpleNamespace(
                    ppi_s=ppi,
                    valid_interval_mask=np.ones(ppi.shape, dtype=bool),
                    peaks=np.arange(3),
                    wavelength="RED",
                )
            }

        reduced_views = SimpleNamespace(x_filter=np.zeros((4000, 2)))
        api = __import__(
            "ppg_frailty.experiment", fromlist=["_runtime_imports"]
        )._runtime_imports()
        api.update(
            {
                "detect_pulses_per_wavelength": pulses,
                "select_reference_wavelength": lambda _rows: "RED",
                "run_quality_mode": quality_mode,
                "run_artifact_route": lambda *_a, **_k: SimpleNamespace(
                    result=SimpleNamespace(
                        reducer_id="pca_bss",
                        reducer_version="pca_component_select_v2",
                        status="success",
                        reasons=(),
                    ),
                    views=reduced_views,
                    route=SignalRoute.ARTIFACT_RATE_ONLY,
                ),
            }
        )
        config = _TierConfig(sqi=True, motion=False, denoiser=True)
        config.representation_mode = "raw"
        config.artifact["degraded_policy"] = (
            "denoise_then_compare_rate_exclude"
        )
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records(
                [state],
                config,
                SimpleNamespace(
                    artifact={"runtime_reducer": "pca_bss", "parameters": {}}
                ),
                SimpleNamespace(calibrator="fixed_formula_thresholds_v1"),
                None,
            )
        self.assertFalse(state.retained)
        self.assertEqual(state.quality_tier, "acceptable")
        self.assertEqual(state.route_status, "rate_only_diagnostic_excluded")
        self.assertEqual(
            state.route_artifact["abstention_reason"],
            "post_denoise_rate_only_diagnostic_not_classifier_input",
        )
        self.assertAlmostEqual(state.route_artifact["direct_hr_bpm"], 60.0)
        self.assertAlmostEqual(state.route_artifact["post_denoise_hr_bpm"], 75.0)

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
