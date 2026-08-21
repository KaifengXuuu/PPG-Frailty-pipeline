"""Focused tests for frozen Stage5 Frailty29 motion-bundle reuse."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ppg_frailty.provenance import sha256_file, stable_payload_sha256
from ppg_frailty.quality.motion import (
    MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE,
    MOTION_DEPLOYMENT_THRESHOLD_SCHEMA,
    MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN,
    MOTION_INTERNAL_EVIDENCE_SCHEMA,
)
from ppg_frailty.quality.motion_adapters import (
    FormalMotionRuntime,
    MotionRecordingInput,
)
from ppg_frailty.quality.motion_bundle_adapter import (
    LoadedReusedMotionDetector,
    ReusedMotionDetectorConfig,
    infer_reused_motion_recording,
    load_reused_motion_detector,
    motion_recording_from_signal_views,
    resolve_reused_motion_detector_config,
)
from ppg_frailty.representations.motion import MOTION_NETWORK_SCHEMA_SHA256
from ppg_frailty.signal.motion_imu import (
    MotionImuResult,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
)
from ppg_frailty.signal.views import CanonicalSignalViews


def _imu(samples: int) -> MotionImuResult:
    acceleration = np.tile([0.0, 0.0, 1.0], (samples, 1))
    gyroscope = np.zeros((samples, 3), dtype=np.float64)
    config = RollPitchEkfConfig(
        calibration_start_s=0.2,
        calibration_stop_s=1.2,
    )
    calibration = fit_motion_imu_calibration(
        acceleration,
        gyroscope,
        participant_id="p1",
        file_id="p1_B",
        source_role="B",
        fs_hz=400.0,
        acceleration_unit="g",
        gyroscope_unit="deg/s",
        config=config,
    )
    return preprocess_motion_imu_calibrated_ekf(
        acceleration,
        gyroscope,
        fs_hz=400.0,
        acceleration_unit="g",
        gyroscope_unit="deg/s",
        participant_id="p1",
        calibration=calibration,
        config=config,
    )


def _recording(samples: int = 4000) -> MotionRecordingInput:
    time = np.arange(samples, dtype=np.float64) / 400.0
    return MotionRecordingInput(
        ppg_red_ir=np.column_stack(
            (np.sin(2.0 * np.pi * time), np.cos(2.0 * np.pi * time))
        ),
        motion_imu=_imu(samples),
        record_id="p1_B",
        participant_id="p1",
        role_or_activity="B",
        dataset_id="frailty29",
    )


def _loaded_detector(
    threshold: float = 0.5,
    *,
    ekf_config_sha256: str = "a" * 64,
) -> LoadedReusedMotionDetector:
    runtime = FormalMotionRuntime(
        model=object(),
        imu_transform=object(),
        device="cpu",
        batch_size=2,
    )
    return LoadedReusedMotionDetector(
        runtime=runtime,
        threshold=threshold,
        ekf_config_sha256=ekf_config_sha256,
        provenance={
            "reuse_scope": "all29_reused",
            "frailty29_evaluation_relation": "in_sample_for_frailty29",
        },
    )


class ReusedMotionConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_and_parameters_remain_configurable(self) -> None:
        default = resolve_reused_motion_detector_config()
        self.assertFalse(default.enabled)
        self.assertIsNone(default.evidence_path)
        self.assertIsNone(default.expected_evidence_sha256)
        self.assertEqual(default.device, "cuda")
        self.assertEqual(default.window_probability_aggregation, "median")
        self.assertEqual(default.threshold_source, "bundle_frozen")

        resolved = resolve_reused_motion_detector_config(
            {
                "enabled": True,
                "evidence_path": "evidence.json",
                "expected_evidence_sha256": "a" * 64,
                "device": "cpu",
                "batch_size": 7,
            }
        )
        self.assertEqual(resolved.evidence_path, Path("evidence.json"))
        self.assertEqual(resolved.batch_size, 7)
        with self.assertRaisesRegex(ValueError, "requires evidence_path"):
            resolve_reused_motion_detector_config({"enabled": True})
        with self.assertRaisesRegex(ValueError, "requires expected_evidence_sha256"):
            resolve_reused_motion_detector_config(
                {"enabled": True, "evidence_path": "evidence.json"}
            )
        with self.assertRaisesRegex(ValueError, "bundle_frozen"):
            resolve_reused_motion_detector_config(
                {
                    "enabled": True,
                    "evidence_path": "evidence.json",
                    "expected_evidence_sha256": "a" * 64,
                    "threshold_source": "explicit",
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown reused"):
            resolve_reused_motion_detector_config({"surprise": True})


class ReusedMotionBundleLoadTest(unittest.TestCase):
    def test_loader_binds_all29_model_oof_threshold_device_and_batch(self) -> None:
        roster = [f"p{index:02d}" for index in range(29)]
        threshold = {
            "schema_version": MOTION_DEPLOYMENT_THRESHOLD_SCHEMA,
            "score_origin": MOTION_DEPLOYMENT_THRESHOLD_SCORE_ORIGIN,
            "fit_scope": MOTION_DEPLOYMENT_THRESHOLD_FIT_SCOPE,
            "participant_ids": roster,
            "threshold": 0.51,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "formal_motion_model.pt"
            model_path.write_bytes(b"test model")
            evidence = {
                "schema_version": MOTION_INTERNAL_EVIDENCE_SCHEMA,
                "execution_status": "completed_formal_not_smoke",
                "scientific_scope": "frailty29_single_sgkf5_oof",
                "participant_count": 29,
                "model_id": "model_identity_from_bundle_not_adapter",
                "formal_source_evidence": {"ekf_config_sha256": "e" * 64},
                "final_model": {
                    "artifact_path": str(model_path),
                    "artifact_sha256": sha256_file(model_path),
                    "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
                    "training_participant_ids": roster,
                    "parameter_count": 11,
                    "inference_cost": {"device": "cuda"},
                },
                "final_threshold": threshold,
                "final_threshold_artifact_sha256": stable_payload_sha256(threshold),
            }
            evidence_path = root / "motion_internal_evidence.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            runtime = FormalMotionRuntime(
                model=object(),
                imu_transform=object(),
                device="cpu",
                batch_size=16,
            )
            config = ReusedMotionDetectorConfig(
                enabled=True,
                evidence_path=evidence_path,
                expected_evidence_sha256=sha256_file(evidence_path),
                device="cpu",
                batch_size=5,
            )
            with patch(
                "ppg_frailty.quality.motion_bundle_adapter.load_formal_motion_model",
                return_value=runtime,
            ) as loader:
                loaded = load_reused_motion_detector(config)

            self.assertEqual(loaded.threshold, 0.51)
            self.assertEqual(loaded.runtime.batch_size, 5)
            self.assertEqual(loaded.ekf_config_sha256, "e" * 64)
            self.assertEqual(loaded.provenance["ekf_config_sha256"], "e" * 64)
            self.assertEqual(
                loaded.provenance["frailty29_evaluation_relation"],
                "in_sample_for_frailty29",
            )
            self.assertFalse(loaded.provenance["valid_outer_oof_claim"])
            self.assertEqual(
                loaded.provenance["model_id"],
                "model_identity_from_bundle_not_adapter",
            )
            self.assertEqual(loader.call_args.kwargs["runtime_device"], "cpu")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_reused_motion_detector(
                    ReusedMotionDetectorConfig(
                        enabled=True,
                        evidence_path=evidence_path,
                        expected_evidence_sha256="b" * 64,
                        device="cpu",
                    )
                )

            missing_source_evidence = dict(evidence)
            missing_source_evidence.pop("formal_source_evidence")
            evidence_path.write_text(
                json.dumps(missing_source_evidence), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "formal_source_evidence"):
                load_reused_motion_detector(
                    ReusedMotionDetectorConfig(
                        enabled=True,
                        evidence_path=evidence_path,
                        expected_evidence_sha256=sha256_file(evidence_path),
                        device="cpu",
                    )
                )

    def test_loader_rejects_non29_training_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "motion_internal_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": MOTION_INTERNAL_EVIDENCE_SCHEMA,
                        "execution_status": "completed_formal_not_smoke",
                        "scientific_scope": "frailty29_single_sgkf5_oof",
                        "participant_count": 29,
                        "formal_source_evidence": {
                            "ekf_config_sha256": "e" * 64
                        },
                        "final_model": {
                            "training_participant_ids": ["p1"],
                            "model_input_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact 29-person roster"):
                load_reused_motion_detector(
                    ReusedMotionDetectorConfig(
                        enabled=True,
                        evidence_path=evidence_path,
                        expected_evidence_sha256=sha256_file(evidence_path),
                        device="cpu",
                    )
                )


class ReusedMotionInferenceTest(unittest.TestCase):
    def test_signal_views_adapter_uses_native_ppg_and_real_roll_pitch(self) -> None:
        samples = 4000
        motion = _imu(samples)
        time = np.arange(samples, dtype=np.float64) / 400.0
        native = np.column_stack(
            (np.sin(2.0 * np.pi * time), np.cos(2.0 * np.pi * time))
        )
        filtered = native * 0.25
        views = CanonicalSignalViews(
            x_native=native,
            x_filter=filtered,
            x_analysis_rate=filtered.copy(),
            imu_processed={
                "dynamic_acc_mps2": motion.values[:, :3],
                "gyro_rads": motion.values[:, 3:6],
                "dynamic_magnitude": motion.values[:, 6],
                "gyro_magnitude": motion.values[:, 7],
                "jerk_magnitude": motion.values[:, 8],
                "roll_rad": motion.roll_rad,
                "pitch_rad": motion.pitch_rad,
                "gravity_mps2": motion.gravity_mps2,
                "imu_valid_mask": motion.valid_mask,
            },
            metadata={
                "fs_hz": 400.0,
                "record_id": "p1_B",
                "gravity_method": motion.profile_id,
                "imu_diagnostics": motion.diagnostics,
            },
            source_valid_mask=np.ones_like(native, dtype=bool),
            repair_mask=np.zeros_like(native, dtype=bool),
        )
        detector = _loaded_detector(
            ekf_config_sha256=motion.diagnostics["ekf_config_sha256"]
        )
        recording = motion_recording_from_signal_views(
            views,
            detector=detector,
            record_id="p1_B",
            participant_id="p1",
            role="B",
        )
        np.testing.assert_array_equal(recording.ppg_red_ir, native)
        self.assertFalse(np.array_equal(recording.ppg_red_ir, filtered))
        np.testing.assert_array_equal(recording.motion_imu.roll_rad, motion.roll_rad)
        np.testing.assert_array_equal(recording.motion_imu.pitch_rad, motion.pitch_rad)

        repaired = CanonicalSignalViews(
            **{
                **views.__dict__,
                "repair_mask": np.ones_like(native, dtype=bool),
            }
        )
        with self.assertRaisesRegex(ValueError, "forbids gap-repaired"):
            motion_recording_from_signal_views(
                repaired,
                detector=detector,
                record_id="p1_B",
                participant_id="p1",
                role="B",
            )

        missing_diagnostics = dict(motion.diagnostics)
        missing_diagnostics.pop("ekf_config_sha256")
        missing_hash = CanonicalSignalViews(
            **{
                **views.__dict__,
                "metadata": {
                    **views.metadata,
                    "imu_diagnostics": missing_diagnostics,
                },
            }
        )
        with self.assertRaisesRegex(
            ValueError, "EKF configuration SHA-256 is missing"
        ):
            motion_recording_from_signal_views(
                missing_hash,
                detector=detector,
                record_id="p1_B",
                participant_id="p1",
                role="B",
            )

        mismatched_diagnostics = {
            **motion.diagnostics,
            "ekf_config_sha256": "f" * 64,
        }
        mismatched_hash = CanonicalSignalViews(
            **{
                **views.__dict__,
                "metadata": {
                    **views.metadata,
                    "imu_diagnostics": mismatched_diagnostics,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "EKF configuration mismatch"):
            motion_recording_from_signal_views(
                mismatched_hash,
                detector=detector,
                record_id="p1_B",
                participant_id="p1",
                role="B",
            )

    def test_record_median_and_threshold_equality_are_deterministic(self) -> None:
        detector = _loaded_detector(0.5)
        with patch(
            "ppg_frailty.quality.motion_bundle_adapter."
            "predict_formal_motion_probability",
            return_value=np.asarray([0.4, 0.6]),
        ):
            decision = infer_reused_motion_recording(detector, _recording())
        self.assertEqual(decision.window_count, 2)
        self.assertEqual(decision.record_probability, 0.5)
        self.assertEqual(decision.motion_state, "high_motion")
        self.assertIn("at_or_above", decision.reason)

    def test_missing_and_short_motion_fail_closed_as_unfit(self) -> None:
        detector = _loaded_detector()
        missing = infer_reused_motion_recording(detector, None)
        self.assertEqual(missing.motion_state, "unfit")
        self.assertEqual(missing.reason, "motion_signal_missing")

        short = MotionRecordingInput(
            ppg_red_ir=np.zeros((100, 2), dtype=np.float64),
            motion_imu=SimpleNamespace(values=np.zeros((100, 9))),
            record_id="short",
            participant_id="p1",
            role_or_activity="B",
            dataset_id="frailty29",
        )
        no_window = infer_reused_motion_recording(detector, short)
        self.assertEqual(no_window.motion_state, "unfit")
        self.assertEqual(no_window.reason, "no_complete_8_second_motion_window")
        self.assertEqual(no_window.window_count, 0)

        incomplete = MotionRecordingInput(
            ppg_red_ir=np.zeros((3200, 1), dtype=np.float64),
            motion_imu=SimpleNamespace(values=np.zeros((3200, 9))),
            record_id="incomplete",
            participant_id="p1",
            role_or_activity="B",
            dataset_id="frailty29",
        )
        incomplete_signal = infer_reused_motion_recording(detector, incomplete)
        self.assertEqual(incomplete_signal.motion_state, "unfit")
        self.assertEqual(incomplete_signal.reason, "motion_signal_missing")

    def test_unusable_model_probability_fails_closed(self) -> None:
        detector = _loaded_detector()
        with patch(
            "ppg_frailty.quality.motion_bundle_adapter."
            "predict_formal_motion_probability",
            return_value=np.asarray([np.nan, 0.2]),
        ):
            decision = infer_reused_motion_recording(detector, _recording())
        self.assertEqual(decision.motion_state, "unfit")
        self.assertEqual(decision.reason, "motion_probability_unavailable")


if __name__ == "__main__":
    unittest.main()
