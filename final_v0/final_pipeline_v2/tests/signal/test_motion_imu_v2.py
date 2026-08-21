"""Non-scientific unit tests for calibrated motion IMU preprocessing."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from ppg_frailty.signal.motion_imu import (
    CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
    MOTION_IMU_CALIBRATION_SCHEMA,
    MOTION_IMU_CHANNEL_SCHEMA,
    PROFILE_A_LPF_ID,
    PTT_STATIC_CALIBRATION_ROLE,
    RollPitchEkfConfig,
    _gravity_from_roll_pitch,
    _run_roll_pitch_ekf,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
    preprocess_motion_imu_lpf_ablation,
)
from ppg_frailty.signal import build_signal_views


class MotionImuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fs = 400.0
        self.samples = 1200
        self.acc_g = np.tile(np.array([0.0, 0.0, 1.0]), (self.samples, 1))
        self.gyro_dps = np.tile(np.array([0.5, -0.25, 0.1]), (self.samples, 1))
        self.config = RollPitchEkfConfig(
            calibration_start_s=0.2,
            calibration_stop_s=1.2,
        )
        self.calibration = fit_motion_imu_calibration(
            self.acc_g,
            self.gyro_dps,
            participant_id="p01",
            file_id="p01_B",
            source_role="B",
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            config=self.config,
        )

    def test_reference_converts_units_persists_covariance_and_removes_static_gravity(self) -> None:
        result = preprocess_motion_imu_calibrated_ekf(
            self.acc_g,
            self.gyro_dps,
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            participant_id="p01",
            calibration=self.calibration,
            config=self.config,
        )
        result.validate()
        self.assertEqual(result.profile_id, CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID)
        self.assertEqual(result.channel_schema, MOTION_IMU_CHANNEL_SCHEMA)
        self.assertEqual(result.values.shape, (self.samples, 9))
        self.assertLess(float(np.max(np.abs(result.values[:, :3]))), 1e-8)
        self.assertLess(float(np.max(np.abs(result.values[:, 3:6]))), 1e-8)
        self.assertEqual(
            result.diagnostics["process_covariance_diagonal_per_second"],
            list(self.config.process_covariance_diagonal_per_second),
        )
        self.assertEqual(
            result.diagnostics["observation_covariance_diagonal_rad2"],
            list(self.config.observation_covariance_diagonal_rad2),
        )
        self.assertFalse(result.diagnostics["silent_fallback"])
        self.assertEqual(result.diagnostics["gravity_mps2"], 9.81)
        self.assertEqual(
            result.diagnostics["unit_conversion"]["g_to_mps2_factor"], 9.81
        )
        self.assertEqual(result.diagnostics["sensor_filters"]["order"], 3)
        for name in (
            "source_acceleration_sha256",
            "source_gyroscope_sha256",
            "filtered_si_acceleration_sha256",
            "filtered_si_gyroscope_sha256",
            "ekf_config_sha256",
            "calibration_artifact_sha256",
            "output_values_sha256",
            "gravity_sha256",
            "roll_pitch_sha256",
            "lineage_sha256",
        ):
                self.assertEqual(len(result.diagnostics[name]), 64)

        tampered = result.values.copy()
        tampered[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "recomputable lineage"):
            replace(result, values=tampered).validate()
        tampered_config = dict(result.diagnostics)
        tampered_config["ekf_config"] = {
            **tampered_config["ekf_config"],
            "dynamic_observation_scale": 999.0,
        }
        with self.assertRaisesRegex(ValueError, "configuration lineage"):
            replace(result, diagnostics=tampered_config).validate()
        nonfinite_gravity = result.gravity_mps2.copy()
        nonfinite_gravity[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "fully finite"):
            replace(result, gravity_mps2=nonfinite_gravity).validate()

    def test_authoritative_observation_scale_is_one_sided(self) -> None:
        count = 8
        gyro = np.zeros((count, 3), dtype=np.float64)
        low_acc = np.tile(
            np.asarray([0.0, 0.0, 0.8 * self.config.gravity_mps2]),
            (count, 1),
        )
        _, _, _, low = _run_roll_pitch_ekf(
            low_acc,
            gyro,
            fs_hz=self.fs,
            initial_roll_rad=0.0,
            initial_pitch_rad=0.0,
            config=self.config,
        )
        self.assertEqual(low["observed_measurement_scale_min"], 1.0)
        self.assertEqual(low["observed_measurement_scale_max"], 1.0)
        high_acc = np.tile(
            np.asarray([0.0, 0.0, 1.2 * self.config.gravity_mps2]),
            (count, 1),
        )
        _, _, _, high = _run_roll_pitch_ekf(
            high_acc,
            gyro,
            fs_hz=self.fs,
            initial_roll_rad=0.0,
            initial_pitch_rad=0.0,
            config=self.config,
        )
        self.assertAlmostEqual(high["observed_measurement_scale_min"], 1.6)
        self.assertAlmostEqual(high["observed_measurement_scale_max"], 1.6)
        self.assertEqual(
            high["observation_scale_equation"],
            "1+alpha_R*max(0,norm_acc-g)/g",
        )

    def test_gravity_rotation_is_rx_then_ry_and_yaw_free(self) -> None:
        roll = np.asarray([0.3])
        pitch = np.asarray([-0.2])
        observed = _gravity_from_roll_pitch(
            roll,
            pitch,
            gravity_mps2=self.config.gravity_mps2,
        )[0]
        rx = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll[0]), -np.sin(roll[0])],
                [0.0, np.sin(roll[0]), np.cos(roll[0])],
            ]
        )
        ry = np.asarray(
            [
                [np.cos(pitch[0]), 0.0, np.sin(pitch[0])],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch[0]), 0.0, np.cos(pitch[0])],
            ]
        )
        expected = (rx @ ry).T @ np.asarray([0.0, 0.0, 9.81])
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)

    def test_filter_parameters_are_configurable_and_stale_calibration_is_rejected(self) -> None:
        self.assertEqual(self.config.sensor_filter_order, 3)
        for field, value in (
            ("accelerometer_lowpass_hz", 30.0),
            ("gyroscope_lowpass_hz", 50.0),
            ("gravity_lowpass_hz", 0.5),
            ("gravity_filter_order", 3),
            ("sensor_filter_order", 4),
        ):
            with self.subTest(config_field=field):
                replace(self.config, **{field: value}).validate()
        for field, value in (
            ("accelerometer_lowpass_hz", 0.0),
            ("gyroscope_lowpass_hz", 200.0),
            ("gravity_lowpass_hz", float("nan")),
            ("gravity_filter_order", 0),
            ("sensor_filter_order", True),
        ):
            with self.subTest(invalid_config_field=field):
                with self.assertRaises(ValueError):
                    replace(self.config, **{field: value}).validate()

        configured = replace(
            self.config,
            accelerometer_lowpass_hz=30.0,
            gyroscope_lowpass_hz=50.0,
            sensor_filter_order=4,
            gravity_lowpass_hz=0.5,
            gravity_filter_order=3,
        )
        configured_calibration = fit_motion_imu_calibration(
            self.acc_g,
            self.gyro_dps,
            participant_id="p01",
            file_id="p01_B_configured",
            source_role="B",
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            config=configured,
        )
        configured_result = preprocess_motion_imu_calibrated_ekf(
            self.acc_g,
            self.gyro_dps,
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            participant_id="p01",
            calibration=configured_calibration,
            config=configured,
        )
        self.assertEqual(configured_result.diagnostics["sensor_filters"]["order"], 4)
        self.assertEqual(
            configured_result.diagnostics["sensor_filters"]["acceleration_lowpass_hz"],
            30.0,
        )
        self.assertEqual(self.calibration.schema_version, MOTION_IMU_CALIBRATION_SCHEMA)
        with self.assertRaisesRegex(ValueError, "calibration schema drift"):
            replace(
                self.calibration,
                schema_version="ppg_frailty.motion_imu_calibration.v2",
            ).validate()

    def test_public_signal_facade_uses_explicit_calibration_without_fallback(self) -> None:
        time = np.arange(self.samples, dtype=np.float64) / self.fs
        ppg = np.column_stack(
            (
                1000.0 + 20.0 * np.sin(2.0 * np.pi * 1.2 * time),
                1200.0 + 15.0 * np.sin(2.0 * np.pi * 1.2 * time + 0.1),
            )
        )
        config = {
            "signal": {
                "internal_fs_hz": 400.0,
                "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
                "ppg_native_unit": "raw_counts",
                "accelerometer_input_unit": "g",
                "gyroscope_input_unit": "deg/s",
                "ppg_filter": {
                    "family": "butterworth_sos",
                    "order": 3,
                    "low_hz": 0.2,
                    "high_hz": 8.0,
                    "phase": "zero_phase",
                    "short_signal_policy": "reject",
                    "notch_enabled": False,
                },
                "analysis_view": {
                    "direct_source": "x_filter_0p2_to_8hz",
                    "non_identity_source": "aligned_x_ar",
                    "non_identity_semantics": "rate_only",
                    "additional_filter": "none",
                },
                "peak_detector": {
                    "detector_id": "aboy_project_v1",
                    "failure_action": "fail_closed_no_fallback",
                },
                "gap_repair": {
                    "method": "linear_inside_only",
                    "max_gap_samples": 100,
                    "edge_extrapolation": False,
                    "all_missing_channel_action": "reject_record",
                },
                "imu": {
                    "gravity_method": "calibrated_roll_pitch_ekf",
                    "initialization": "same_participant_static_calibration",
                    "comparison_method": "profile_a_lowpass_0p3hz",
                    "sensor_lowpass_acc_hz": 20.0,
                    "sensor_lowpass_gyro_hz": 40.0,
                    "sensor_filter_order": 3,
                    "gravity_lowpass_hz": 0.3,
                    "gravity_filter_order": 4,
                    "calibration_start_s": 0.2,
                    "calibration_stop_s": 1.2,
                    "process_covariance_diagonal_per_second": [5.0, 5.0, 0.05, 0.05, 0.05],
                    "observation_covariance_diagonal_rad2": [0.5, 0.5],
                    "initial_covariance_diagonal": [1.0, 1.0, 0.5, 0.5, 0.5],
                    "dynamic_observation_scale": 3.0,
                    "output_units": {
                        "acceleration": "m/s^2",
                        "gyroscope": "rad/s",
                        "jerk": "m/s^3",
                    },
                    "required_axes": 6,
                    "failure_action": "fail_closed",
                },
                "dl_resampling": {
                    "enabled": False,
                    "target_fs_hz": 400.0,
                    "method": "polyphase_anti_alias",
                    "preserve_feature_grid_hz": 400.0,
                },
                "normalization": {
                    "raw_ppg": "per_window_median_iqr_over_1p349_sd_finite",
                    "raw_imu": (
                        "outer_training_participant_only_median_iqr_over_1p349_"
                        "population_sd_then_one_axes6"
                    ),
                    "iqr_fallback": "standard_deviation_then_finite_one",
                    "clip_after_scale": [-8.0, 8.0],
                },
            },
            "quality": {"long_gap_max_samples": 100, "flatline_duration_s": 1.0},
        }
        views = build_signal_views(
            {
                "record_id": "p01_runtime",
                "participant_id": "p01",
                "fs_hz": self.fs,
                "ppg": ppg,
                "acc": self.acc_g,
                "gyro": self.gyro_dps,
                "acc_unit": "g",
                "gyro_unit": "deg/s",
                "imu_calibration": self.calibration,
            },
            config,
        )
        self.assertEqual(
            views.metadata["gravity_method"],
            CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID,
        )
        self.assertFalse(views.metadata["imu_diagnostics"]["silent_fallback"])
        self.assertEqual(
            {
                "dynamic_acc_mps2",
                "gyro_rads",
                "dynamic_magnitude",
                "gyro_magnitude",
                "jerk_magnitude",
                "roll_rad",
                "pitch_rad",
            }
            - set(views.imu_processed),
            set(),
        )

    def test_known_constant_z_rotation_preserves_units_and_motion_intensity(self) -> None:
        runtime_gyro = self.gyro_dps + np.array([0.0, 0.0, 10.0])
        result = preprocess_motion_imu_calibrated_ekf(
            self.acc_g,
            runtime_gyro,
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            participant_id="p01",
            calibration=self.calibration,
            config=self.config,
        )
        expected = float(np.deg2rad(10.0))
        self.assertTrue(
            np.allclose(result.values[:, 5], expected, rtol=0.0, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(result.values[:, 7], expected, rtol=0.0, atol=1e-12)
        )
        self.assertLess(float(np.max(np.abs(result.values[:, :3]))), 1e-8)
        self.assertLess(float(np.max(np.abs(result.values[:, 8]))), 1e-8)

    def test_lpf_is_separate_named_ablation_and_ekf_failure_cannot_fallback(self) -> None:
        lpf = preprocess_motion_imu_lpf_ablation(
            self.acc_g,
            self.gyro_dps,
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            participant_id="p01",
            calibration=self.calibration,
            config=self.config,
        )
        self.assertEqual(lpf.profile_id, PROFILE_A_LPF_ID)
        self.assertEqual(lpf.diagnostics["executed_as"], "named_ablation_only")
        lpf.validate()
        with self.assertRaisesRegex(ValueError, "stale or unknown"):
            replace(lpf, profile_id="profile_a_sensor_lpf_order4_stale").validate()
        mismatched_diagnostics = dict(lpf.diagnostics)
        mismatched_diagnostics["profile_id"] = CALIBRATED_ROLL_PITCH_EKF_PROFILE_ID
        with self.assertRaisesRegex(ValueError, "profile identity drift"):
            replace(lpf, diagnostics=mismatched_diagnostics).validate()
        broken = self.acc_g.copy()
        broken[10, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            preprocess_motion_imu_calibrated_ekf(
                broken,
                self.gyro_dps,
                fs_hz=self.fs,
                acceleration_unit="g",
                gyroscope_unit="deg/s",
                participant_id="p01",
                calibration=self.calibration,
                config=self.config,
            )

    def test_ptt_sit_static_calibration_is_explicit_and_participant_bound(self) -> None:
        calibration = fit_motion_imu_calibration(
            self.acc_g,
            self.gyro_dps,
            participant_id="ptt01",
            file_id="ptt01_sit",
            source_role=PTT_STATIC_CALIBRATION_ROLE,
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            config=self.config,
        )
        self.assertEqual(calibration.source_role, PTT_STATIC_CALIBRATION_ROLE)
        result = preprocess_motion_imu_calibrated_ekf(
            self.acc_g,
            self.gyro_dps,
            fs_hz=self.fs,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
            participant_id="ptt01",
            calibration=calibration,
            config=self.config,
        )
        self.assertEqual(
            result.diagnostics["calibration_source_role"],
            PTT_STATIC_CALIBRATION_ROLE,
        )
        with self.assertRaisesRegex(ValueError, "cross-participant"):
            preprocess_motion_imu_calibrated_ekf(
                self.acc_g,
                self.gyro_dps,
                fs_hz=self.fs,
                acceleration_unit="g",
                gyroscope_unit="deg/s",
                participant_id="ptt02",
                calibration=calibration,
                config=self.config,
            )
        with self.assertRaisesRegex(ValueError, "cross-participant"):
            preprocess_motion_imu_calibrated_ekf(
                self.acc_g,
                self.gyro_dps,
                fs_hz=self.fs,
                acceleration_unit="g",
                gyroscope_unit="deg/s",
                participant_id="p02",
                calibration=self.calibration,
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
