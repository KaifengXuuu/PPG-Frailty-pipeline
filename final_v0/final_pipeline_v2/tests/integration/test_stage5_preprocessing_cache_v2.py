"""Lightweight Stage-5 preprocessing-cache and short-record regressions."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from ppg_frailty.contracts import SignalRoute
from ppg_frailty.data.preprocessing_cache import (
    PreprocessingCacheSession,
    _implementation_dependency_sha256,
)
from ppg_frailty.data.recording_cache import (
    NamedSourceDependency,
    OrderedModuleSpec,
    RecordingCacheError,
    RecordingCacheIdentity,
    RecordingCacheSourceError,
)
from ppg_frailty.data.windows import WindowPlan
from ppg_frailty.experiment import _RuntimeRecord, _route_records_window_level
from ppg_frailty.representations.motion import (
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_REFERENCE_PROFILE_ID,
    MotionWindowTensors,
)
from ppg_frailty.representations.raw import build_raw_windows
from ppg_frailty.signal.motion_imu import (
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
)
from ppg_frailty.signal.views import CanonicalSignalViews


def _source_row(repository_root: Path, *, samples: int) -> SimpleNamespace:
    payload = b"stage5-cache-source-fixture\n"
    source = repository_root / "fixture_record.csv"
    source.write_bytes(payload)
    return SimpleNamespace(
        record_id="P01_B",
        participant_id="P01",
        role="B",
        source_path=source.name,
        source_hash=hashlib.sha256(payload).hexdigest(),
        source_version="fixture.v1",
        fs=400.0,
        n_samples=samples,
        duration_s=samples / 400.0,
        channel_schema=("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"),
        channel_units={
            "RED": "a.u.",
            "IR": "a.u.",
            "AX": "g",
            "AY": "g",
            "AZ": "g",
            "GX": "deg/s",
            "GY": "deg/s",
            "GZ": "deg/s",
        },
        synchrony_status="sample_synchronous",
        reference_available=False,
        qc_status="pass",
        qc_reasons=(),
        manifest_version="internal_records_v2",
    )


def _views(record_id: str, *, samples: int) -> CanonicalSignalViews:
    time = np.arange(samples, dtype=np.float64) / 400.0
    ppg = np.column_stack(
        (
            np.sin(2.0 * np.pi * 1.2 * time),
            0.8 * np.sin(2.0 * np.pi * 1.2 * time + 0.1),
        )
    )
    dynamic = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 0.7 * time),
            0.1 * np.cos(2.0 * np.pi * 0.7 * time),
            0.05 * np.sin(2.0 * np.pi * 0.4 * time),
        )
    )
    gyro = 0.01 * np.column_stack(
        (
            np.sin(2.0 * np.pi * 0.3 * time),
            np.cos(2.0 * np.pi * 0.3 * time),
            np.sin(2.0 * np.pi * 0.2 * time),
        )
    )
    result = CanonicalSignalViews(
        x_native=ppg + np.asarray([1000.0, 1200.0]),
        x_filter=ppg,
        x_analysis_rate=ppg.copy(),
        imu_processed={
            "dynamic_acc_mps2": dynamic,
            "gyro_rads": gyro,
            "dynamic_magnitude": np.linalg.norm(dynamic, axis=1),
            "gyro_magnitude": np.linalg.norm(gyro, axis=1),
            "jerk_magnitude": np.zeros(samples, dtype=np.float64),
            "imu_valid_mask": np.ones(samples, dtype=bool),
        },
        metadata={"record_id": record_id, "fs_hz": 400.0},
        source_valid_mask=np.ones_like(ppg, dtype=bool),
        repair_mask=np.zeros_like(ppg, dtype=bool),
        route=SignalRoute.DIRECT,
    )
    result.validate()
    return result


class Stage5CacheIntegrationTests(unittest.TestCase):
    def test_cache_root_symlink_is_rejected_before_store_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline_root = Path(directory)
            studies = pipeline_root / "artifacts" / "studies"
            studies.mkdir(parents=True)
            outside = pipeline_root / "outside-cache"
            outside.mkdir()
            (studies / "cache").symlink_to(outside, target_is_directory=True)
            paths = SimpleNamespace(
                repository_root=pipeline_root,
                pipeline_root=pipeline_root,
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                PreprocessingCacheSession.from_mapping(
                    {"mode": "read_write"},
                    paths,
                )

    def test_named_implementation_dependency_manifest_changes_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("FIRST = 1\n", encoding="utf-8")
            second.write_text("SECOND = 2\n", encoding="utf-8")
            one_dependency = _implementation_dependency_sha256(
                {"first": first}
            )
            two_dependencies = _implementation_dependency_sha256(
                {"first": first, "second": second}
            )

            def identity(implementation_sha256: str) -> RecordingCacheIdentity:
                return RecordingCacheIdentity(
                    namespace="canonical_signal_views",
                    layer="dependency_fixture",
                    recording_id="P01_B",
                    source_dependencies=(
                        NamedSourceDependency(
                            name="source",
                            sha256="1" * 64,
                        ),
                    ),
                    module_chain=(
                        OrderedModuleSpec(
                            module_id="fixture",
                            module_version="v1",
                            implementation_sha256=implementation_sha256,
                            enabled=True,
                        ),
                    ),
                    producer_sha256=implementation_sha256,
                    output_schema={"type": "fixture"},
                )

            self.assertNotEqual(one_dependency, two_dependencies)
            self.assertNotEqual(
                identity(one_dependency).key,
                identity(two_dependencies).key,
            )
            first.write_text("FIRST = 100\n", encoding="utf-8")
            changed_dependency = _implementation_dependency_sha256(
                {"first": first}
            )
            self.assertNotEqual(one_dependency, changed_dependency)
            self.assertNotEqual(
                identity(one_dependency).key,
                identity(changed_dependency).key,
            )

    def test_canonical_view_dependency_manifest_includes_resampling_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline_root = Path(directory)
            row = _source_row(pipeline_root, samples=2400)
            paths = SimpleNamespace(
                repository_root=pipeline_root,
                pipeline_root=pipeline_root,
            )
            session = PreprocessingCacheSession.from_mapping(
                {"mode": "off"},
                paths,
            )
            observed_manifests: list[set[str]] = []

            def traced_hash(dependencies: dict[str, object]) -> str:
                observed_manifests.append(set(dependencies))
                return _implementation_dependency_sha256(dependencies)

            with patch(
                "ppg_frailty.data.preprocessing_cache."
                "_implementation_dependency_sha256",
                side_effect=traced_hash,
            ):
                session.canonical_views(
                    row,
                    maximum_samples=None,
                    signal_config={
                        "imu": {"gravity_method": "profile_a_lowpass_0p3hz"}
                    },
                    quality_preprocess_config={},
                    calibration=None,
                    builder=lambda: (
                        _views(row.record_id, samples=row.n_samples),
                        {},
                        {},
                    ),
                )

            canonical_names = set().union(*observed_manifests)
            self.assertIn("experiment_callsite", canonical_names)
            self.assertIn("signal_resample", canonical_names)
            self.assertIn("data_schema", canonical_names)

    def test_manifest_preprocessing_fields_change_canonical_view_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline_root = Path(directory)
            row = _source_row(pipeline_root, samples=2400)
            paths = SimpleNamespace(
                repository_root=pipeline_root,
                pipeline_root=pipeline_root,
            )
            policy = {
                "mode": "read_write",
                "root": "artifacts/studies/cache",
                "namespaces": ["canonical_signal_views"],
            }
            arguments = {
                "maximum_samples": None,
                "signal_config": {
                    "imu": {"gravity_method": "profile_a_lowpass_0p3hz"}
                },
                "quality_preprocess_config": {},
                "calibration": None,
            }
            cold = PreprocessingCacheSession.from_mapping(policy, paths)
            _, _, _, cold_key = cold.canonical_views(
                row,
                **arguments,
                builder=lambda: (
                    _views(row.record_id, samples=row.n_samples),
                    {"manifest_version": row.manifest_version},
                    {},
                ),
            )

            changed = SimpleNamespace(**vars(row))
            changed.duration_s = 999.0
            changed.manifest_version = "fixture_manifest_v2"
            builder_calls = 0

            def rebuilt() -> tuple[CanonicalSignalViews, dict[str, str], dict]:
                nonlocal builder_calls
                builder_calls += 1
                return (
                    _views(changed.record_id, samples=changed.n_samples),
                    {"manifest_version": changed.manifest_version},
                    {},
                )

            warm = PreprocessingCacheSession.from_mapping(policy, paths)
            _, qc, _, changed_key = warm.canonical_views(
                changed,
                **arguments,
                builder=rebuilt,
            )
            self.assertNotEqual(changed_key, cold_key)
            self.assertEqual(builder_calls, 1)
            self.assertEqual(qc["manifest_version"], "fixture_manifest_v2")

    def test_same_participant_b_calibration_is_reused_without_refit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline_root = Path(directory)
            row = _source_row(pipeline_root, samples=800)
            paths = SimpleNamespace(
                repository_root=pipeline_root,
                pipeline_root=pipeline_root,
            )
            policy = {
                "mode": "read_write",
                "root": "artifacts/studies/cache",
                "namespaces": ["imu_calibration"],
            }
            config = RollPitchEkfConfig(
                calibration_start_s=0.1,
                calibration_stop_s=1.5,
            )
            acceleration = np.zeros((row.n_samples, 3), dtype=np.float64)
            acceleration[:, 2] = 1.0
            gyroscope = np.zeros_like(acceleration)

            cold = PreprocessingCacheSession.from_mapping(policy, paths)
            observed_manifests: list[set[str]] = []

            def traced_hash(dependencies: dict[str, object]) -> str:
                observed_manifests.append(set(dependencies))
                return _implementation_dependency_sha256(dependencies)

            with patch(
                "ppg_frailty.data.preprocessing_cache."
                "_implementation_dependency_sha256",
                side_effect=traced_hash,
            ):
                cold_value = cold.calibration(
                    row,
                    asdict(config),
                    lambda: fit_motion_imu_calibration(
                        acceleration,
                        gyroscope,
                        participant_id=row.participant_id,
                        file_id=row.record_id,
                        source_role=row.role,
                        fs_hz=row.fs,
                        acceleration_unit="g",
                        gyroscope_unit="deg/s",
                        config=config,
                    ),
                )
            calibration_names = set().union(*observed_manifests)
            self.assertIn("contracts", calibration_names)
            self.assertIn("data_schema", calibration_names)
            self.assertIn("experiment_callsite", calibration_names)
            self.assertIn("signal_views", calibration_names)

            warm = PreprocessingCacheSession.from_mapping(policy, paths)

            def unexpected_builder() -> object:
                raise AssertionError("warm calibration hit must not refit EKF inputs")

            warm_value = warm.calibration(row, asdict(config), unexpected_builder)

            self.assertEqual(warm_value.artifact_sha256, cold_value.artifact_sha256)
            np.testing.assert_array_equal(
                warm_value.acceleration_bias_mps2,
                cold_value.acceleration_bias_mps2,
            )
            np.testing.assert_array_equal(
                warm_value.gyroscope_bias_rads,
                cold_value.gyroscope_bias_rads,
            )
            self.assertEqual(
                cold.audit_payload()["counts"],
                {"imu_calibration": {"written": 1}},
            )
            self.assertEqual(
                warm.audit_payload()["counts"],
                {"imu_calibration": {"hit": 1}},
            )
            self.assertTrue(
                warm.audit_payload()["calibration_source_role_cached"]
            )

    def test_cold_then_warm_views_and_raw_windows_are_value_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline_root = Path(directory)
            row = _source_row(pipeline_root, samples=4000)
            paths = SimpleNamespace(
                repository_root=pipeline_root,
                pipeline_root=pipeline_root,
            )
            policy = {
                "mode": "read_write",
                "root": "artifacts/studies/cache",
                "namespaces": [
                    "canonical_signal_views",
                    "motion_windows",
                    "raw_windows",
                ],
                "verify_source_sha256": True,
            }
            signal_config = {
                "imu": {"gravity_method": "profile_a_lowpass_0p3hz"},
                "filter": {"family": "butterworth_sos"},
            }
            quality_config = {
                "long_gap_max_samples": 8,
                "flatline_duration_s": 0.5,
            }
            plan = WindowPlan(
                source_record_id=row.record_id,
                window_seconds=5.0,
                hop_seconds=2.5,
                end_alignment="start",
                short_record_action="reject",
                include_padded_tail=False,
                max_windows=None,
                cap_policy="not_applicable",
                min_valid_fraction=1.0,
            )
            normalization = {
                "raw_ppg": "per_window_robust",
                "raw_imu": "none",
                "robust_iqr_divisor": 1.349,
                "iqr_fallback": "standard_deviation",
                "standard_ddof": 0,
                "mad_consistency_divisor": 0.6744897501960817,
                "scale_epsilon": 1e-8,
                "clip_after_scale": [-8.0, 8.0],
            }

            cold = PreprocessingCacheSession.from_mapping(policy, paths)
            cold_views, cold_qc, cold_profile, cold_view_key = cold.canonical_views(
                row,
                maximum_samples=None,
                signal_config=signal_config,
                quality_preprocess_config=quality_config,
                calibration=None,
                builder=lambda: (
                    _views(row.record_id, samples=row.n_samples),
                    {"status": "fixture_qc_pass"},
                    {"profile_id": "fixture_qc_v1"},
                ),
            )
            cold_raw, cold_raw_key = cold.raw_windows(
                row,
                upstream_views_key=cold_view_key,
                plan=plan,
                normalization=normalization,
                builder=lambda: build_raw_windows(
                    cold_views,
                    plan,
                    normalization=normalization,
                ),
            )
            recording = SimpleNamespace(
                record_id=row.record_id,
                participant_id=row.participant_id,
                role_or_activity=row.role,
                dataset_id="frailty29_fixture",
                fs_hz=400.0,
            )

            def build_motion() -> MotionWindowTensors:
                return MotionWindowTensors(
                    values=np.arange(
                        2 * 8 * 3200,
                        dtype=np.float32,
                    ).reshape(2, 8, 3200),
                    start_samples=np.asarray([0, 800], dtype=np.int64),
                    record_id=row.record_id,
                    participant_id=row.participant_id,
                    role_or_activity=row.role,
                    dataset_id=recording.dataset_id,
                    profile_id=MOTION_REFERENCE_PROFILE_ID,
                    channel_schema=MOTION_NETWORK_CHANNEL_SCHEMA,
                    schema_sha256=MOTION_NETWORK_SCHEMA_SHA256,
                )

            cold_motion, cold_motion_key = cold.motion_windows(
                row,
                upstream_views_key=cold_view_key,
                recording=recording,
                profile_id=MOTION_REFERENCE_PROFILE_ID,
                builder=build_motion,
            )

            warm = PreprocessingCacheSession.from_mapping(policy, paths)

            def unexpected_builder() -> object:
                raise AssertionError("warm cache hit must not execute its builder")

            warm_views, warm_qc, warm_profile, warm_view_key = warm.canonical_views(
                row,
                maximum_samples=None,
                signal_config=signal_config,
                quality_preprocess_config=quality_config,
                calibration=None,
                builder=unexpected_builder,
            )
            warm_raw, warm_raw_key = warm.raw_windows(
                row,
                upstream_views_key=warm_view_key,
                plan=plan,
                normalization=normalization,
                builder=unexpected_builder,
            )
            warm_motion, warm_motion_key = warm.motion_windows(
                row,
                upstream_views_key=warm_view_key,
                recording=recording,
                profile_id=MOTION_REFERENCE_PROFILE_ID,
                builder=unexpected_builder,
            )

            self.assertEqual(warm_view_key, cold_view_key)
            self.assertEqual(warm_raw_key, cold_raw_key)
            self.assertEqual(warm_motion_key, cold_motion_key)
            self.assertEqual(warm_qc, cold_qc)
            self.assertEqual(warm_profile, cold_profile)
            np.testing.assert_array_equal(warm_views.x_filter, cold_views.x_filter)
            np.testing.assert_array_equal(warm_raw.values, cold_raw.values)
            np.testing.assert_array_equal(warm_raw.valid_mask, cold_raw.valid_mask)
            np.testing.assert_array_equal(
                warm_raw.start_samples, cold_raw.start_samples
            )
            np.testing.assert_array_equal(warm_motion.values, cold_motion.values)
            np.testing.assert_array_equal(
                warm_motion.start_samples,
                cold_motion.start_samples,
            )
            self.assertIsNone(warm_raw.window_quality_scores)
            self.assertIsNone(warm_raw.window_aggregation_mask)
            self.assertEqual(
                cold.audit_payload()["counts"],
                {
                    "canonical_signal_views": {"written": 1},
                    "motion_windows": {"written": 1},
                    "raw_windows": {"written": 1},
                },
            )
            self.assertEqual(
                warm.audit_payload()["counts"],
                {
                    "canonical_signal_views": {"hit": 1},
                    "motion_windows": {"hit": 1},
                    "raw_windows": {"hit": 1},
                },
            )
            self.assertFalse(warm.audit_payload()["labels_cached"])
            self.assertFalse(
                warm.audit_payload()["supervision_target_labels_cached"]
            )
            self.assertFalse(
                warm.audit_payload()["calibration_source_role_cached"]
            )
            self.assertFalse(warm.audit_payload()["fold_local_artifacts_cached"])
            self.assertFalse(warm.audit_payload()["route_masks_cached"])

    def test_source_byte_drift_fails_before_a_warm_cache_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline_root = Path(directory)
            row = _source_row(pipeline_root, samples=2400)
            paths = SimpleNamespace(
                repository_root=pipeline_root,
                pipeline_root=pipeline_root,
            )
            session = PreprocessingCacheSession.from_mapping(
                {
                    "mode": "read_write",
                    "root": "artifacts/studies/cache",
                    "namespaces": ["canonical_signal_views"],
                },
                paths,
            )
            source = pipeline_root / row.source_path
            source.write_bytes(b"changed-after-manifest\n")
            with self.assertRaisesRegex(
                RecordingCacheSourceError, "source hash drift"
            ) as raised:
                session.canonical_views(
                    row,
                    maximum_samples=None,
                    signal_config={
                        "imu": {"gravity_method": "profile_a_lowpass_0p3hz"}
                    },
                    quality_preprocess_config={
                        "long_gap_max_samples": 8,
                        "flatline_duration_s": 0.5,
                    },
                    calibration=None,
                    builder=lambda: (
                        _views(row.record_id, samples=row.n_samples),
                        {},
                        {},
                    ),
                )
            self.assertIsInstance(raised.exception, RecordingCacheError)


class _ShortRouteConfig:
    representation_mode = "raw"
    sha256 = "f" * 64

    def section(self, name: str) -> dict[str, object]:
        return {
            "quality": {"mode": "route"},
            "artifact": {
                "motion_detector_enabled": False,
                "denoiser_enabled": False,
                "reducer": "identity",
            },
            "routing": {
                "window_s": 8.0,
                "hop_s": 2.0,
                "fs_hz": 400.0,
                "source_grid": "canonical_acquisition_grid",
            },
            "signal": {
                "peak_detector": {
                    "detector_id": "msptdfast_v2_3_python_port",
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


class Stage5ShortRecordRoutingTests(unittest.TestCase):
    def test_seven_second_record_is_excluded_without_peak_detection(self) -> None:
        state = _RuntimeRecord(
            row=SimpleNamespace(
                participant_id="P01",
                record_id="P01_W1_short",
                role="W1",
            ),
            views=_views("P01_W1_short", samples=7 * 400),
        )
        api = __import__(
            "ppg_frailty.experiment", fromlist=["_runtime_imports"]
        )._runtime_imports()
        api["detect_pulses_per_wavelength"] = lambda *_a, **_k: (
            self.fail("short records must be excluded before peak detection")
        )
        with patch("ppg_frailty.experiment._runtime_imports", return_value=api):
            _route_records_window_level(
                [state],
                _ShortRouteConfig(),
                SimpleNamespace(
                    artifact={"runtime_reducer": "identity", "parameters": {}}
                ),
                SimpleNamespace(
                    calibrator="fixed_formula_thresholds_v1",
                    to_dict=lambda: {
                        "calibrator": "fixed_formula_thresholds_v1"
                    },
                ),
                None,
                motion_detector=None,
            )

        self.assertFalse(state.retained)
        self.assertIs(state.route, SignalRoute.DROPPED)
        self.assertEqual(
            state.route_status,
            "dropped_no_complete_8_second_routing_window",
        )
        self.assertEqual(state.route_artifact["native_routing_window_count"], 0)
        self.assertEqual(
            state.route_artifact["short_record_action"],
            "exclude_record_no_padding",
        )
        self.assertEqual(len(state.routing_timeline.cells), 1)
        self.assertEqual(state.routing_timeline.cells[0].final_tier, "excluded")


if __name__ == "__main__":
    unittest.main()
