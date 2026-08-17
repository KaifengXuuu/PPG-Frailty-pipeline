"""视图、窗计划和 IMU contract 测试 / Signal-view, window, and IMU tests."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from ppg_frailty.data.windows import WindowPlan
from ppg_frailty.signal import (
    CANONICAL_FS_HZ,
    CausalImuProcessor,
    ImuProfile,
    build_signal_views,
    estimate_gravity_lpf,
    estimate_gravity_no_precalibration_ekf,
    preprocess_imu,
)


def synthetic_record(seconds: float = 12.0) -> dict[str, object]:
    """构建确定性同步 RED/IR/IMU / Build deterministic synchronized inputs."""

    samples = int(seconds * CANONICAL_FS_HZ)
    time = np.arange(samples) / CANONICAL_FS_HZ
    pulse = np.sin(2.0 * np.pi * 1.2 * time)
    ppg = np.column_stack((1000.0 + 20.0 * pulse, 1200.0 + 15.0 * np.sin(2.0 * np.pi * 1.2 * time + 0.1)))
    acc = np.column_stack((0.05 * np.sin(2 * np.pi * 0.7 * time), np.zeros(samples), np.full(samples, 9.80665)))
    gyro = np.zeros((samples, 3), dtype=np.float64)
    return {
        "record_id": "synthetic",
        "ppg": ppg,
        "acc": acc,
        "gyro": gyro,
        "acc_unit": "m/s2",
        "gyro_unit": "rad/s",
        "fs_hz": CANONICAL_FS_HZ,
        "timestamps_s": time,
    }


def signal_config() -> dict[str, object]:
    """构建显式resolved signal profile / Build an explicit resolved profile."""

    return {
        "signal": {
            "internal_fs_hz": 400.0,
            "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "ppg_native_unit": "raw_counts",
            "accelerometer_input_unit": "m/s2",
            "gyroscope_input_unit": "rad/s",
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
                "direct_source": "x_filter_0p2_to_8hz", "non_identity_source": "aligned_x_ar",
                "non_identity_semantics": "rate_only", "additional_filter": "none",
            },
            "gap_repair": {
                "method": "linear_inside_only", "max_gap_samples": 100,
                "edge_extrapolation": False, "all_missing_channel_action": "reject_record",
            },
            "imu": {
                "gravity_method": "quaternion_error_state_ekf",
                "initialization": "online_no_precalibration",
                "comparison_method": "lowpass_0p3hz",
                "sensor_lowpass_acc_hz": 20.0,
                "sensor_lowpass_gyro_hz": 40.0,
                "gravity_lowpass_hz": 0.3,
                "output_units": {"acceleration": "m/s^2", "gyroscope": "rad/s", "jerk": "m/s^3"},
                "required_axes": 6,
                "failure_action": "fail_closed",
            },
            "dl_resampling": {
                "enabled": False, "target_fs_hz": 400.0, "method": "polyphase_anti_alias",
                "preserve_feature_grid_hz": 400.0,
            },
            "normalization": {
                "raw_ppg": "per_window_median_iqr", "raw_imu": "outer_train_fold_robust_scaler",
                "iqr_fallback": "median_absolute_deviation_then_one", "clip_after_scale": None,
            },
        },
        "quality": {"long_gap_max_samples": 100, "flatline_duration_s": 1.0},
    }


class SignalViewsTest(unittest.TestCase):
    """验证公开 build_signal_views facade / Verify public construction facade."""

    def test_build_views_preserves_grid_and_direct_contract(self) -> None:
        views = build_signal_views(synthetic_record(), signal_config())
        views.validate()
        self.assertEqual(views.x_native.shape, (4800, 2))
        self.assertTrue(np.array_equal(views.x_analysis_rate, views.x_filter))
        self.assertEqual(views.metadata["fs_hz"], 400.0)
        self.assertEqual(views.to_contract().x_analysis.shape, (4800, 2))

    def test_unknown_imu_unit_fails_closed(self) -> None:
        record = synthetic_record()
        record["acc_unit"] = "raw_counts"
        with self.assertRaises(ValueError):
            build_signal_views(record, signal_config())

    def test_resolved_gap_limit_is_executed_not_hidden(self) -> None:
        """同一 gap 随配置阈值改变结果 / Resolved gap limit changes the result."""

        record = synthetic_record()
        ppg = np.asarray(record["ppg"], dtype=np.float64).copy()
        ppg[1000:1050, 0] = np.nan
        record["ppg"] = ppg
        accepted = build_signal_views(record, signal_config())
        self.assertTrue(accepted.repair_mask[1000:1050, 0].all())
        strict = copy.deepcopy(signal_config())
        strict["signal"]["gap_repair"]["max_gap_samples"] = 40
        strict["quality"]["long_gap_max_samples"] = 40
        with self.assertRaisesRegex(ValueError, "long_gap"):
            build_signal_views(record, strict)

    def test_unknown_signal_config_field_fails_closed(self) -> None:
        """未知 signal knob 不得被忽略 / Unknown signal knobs cannot be ignored."""

        config = copy.deepcopy(signal_config())
        config["signal"]["silent_new_knob"] = 1
        with self.assertRaisesRegex(ValueError, "signal key mismatch"):
            build_signal_views(synthetic_record(), config)

    def test_window_plan_masks_padding_and_complete_windows(self) -> None:
        complete_plan = WindowPlan(
            source_record_id="r", window_seconds=10.0, hop_seconds=5.0,
            end_alignment="start", short_record_action="reject",
            include_padded_tail=False, max_windows=None, cap_policy="not_applicable",
        )
        complete = complete_plan.plan(6000, 400.0)
        self.assertEqual([item.start_sample for item in complete], [0, 2000])
        self.assertFalse(any(any(item.padding_mask) for item in complete))
        padded_plan = WindowPlan(
            source_record_id="short", window_seconds=5.0, hop_seconds=5.0,
            end_alignment="start", short_record_action="pad_right",
            include_padded_tail=False, max_windows=None, cap_policy="not_applicable",
        )
        padded = padded_plan.plan(1000, 400.0)
        self.assertEqual(padded[0].valid_length, 1000)
        self.assertEqual(int(np.sum(padded[0].padding_mask)), 1000)


class ImuTest(unittest.TestCase):
    """验证 EKF 主路线和 LPF comparator / Verify primary EKF and LPF comparator."""

    def test_stationary_gravity_estimators(self) -> None:
        samples = 1600
        acc = np.tile([0.0, 0.0, 9.80665], (samples, 1))
        gyro = np.zeros_like(acc)
        ekf, diagnostics = estimate_gravity_no_precalibration_ekf(acc, gyro)
        lpf = estimate_gravity_lpf(acc)
        states = np.asarray(diagnostics["state_per_sample"])
        tracking = np.isin(states, ["tracking", "prediction_only"])
        self.assertGreater(int(np.count_nonzero(tracking)), 1000)
        self.assertLess(float(np.max(np.abs(ekf[tracking] - acc[tracking]))), 1e-8)
        self.assertLess(float(np.max(np.abs(lpf - acc))), 1e-8)
        self.assertAlmostEqual(diagnostics["gravity_norm_mean_mps2"], 9.80665, places=5)

    def test_preprocess_outputs_motion_quantities(self) -> None:
        record = synthetic_record(5.0)
        result = preprocess_imu(
            record["acc"],
            record["gyro"],
            fs_hz=400.0,
            acc_unit="m/s2",
            gyro_unit="rad/s",
            gravity_method="no_precalibration_ekf",
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.processed["dynamic_acc_mps2"].shape, (2000, 3))
        self.assertEqual(result.processed["jerk_magnitude"].shape, (2000,))

    def test_stateful_prefix_chunk_parity_for_both_routes(self) -> None:
        """任意合法分块与one-shot一致 / Legal chunking equals one-shot."""

        record = synthetic_record(8.0)
        timestamps = np.asarray(record["timestamps_s"])
        for method in ("no_precalibration_ekf", "lpf_0p3"):
            with self.subTest(method=method):
                profile = ImuProfile(gravity_method=method)
                full = CausalImuProcessor(
                    fs_hz=400.0,
                    acceleration_unit="m/s2",
                    gyroscope_unit="rad/s",
                    profile=profile,
                ).process_chunk(
                    record["acc"],
                    record["gyro"],
                    timestamps_s=timestamps,
                )
                chunked_processor = CausalImuProcessor(
                    fs_hz=400.0,
                    acceleration_unit="m/s2",
                    gyroscope_unit="rad/s",
                    profile=profile,
                )
                boundaries = (0, 333, 1111, 2000, 3200)
                chunks = [
                    chunked_processor.process_chunk(
                        record["acc"][left:right],
                        record["gyro"][left:right],
                        timestamps_s=timestamps[left:right],
                    )
                    for left, right in zip(boundaries[:-1], boundaries[1:])
                ]
                for key in (
                    "gravity_mps2",
                    "dynamic_acc_mps2",
                    "gyro_rads",
                    "jerk_mps3",
                ):
                    combined = np.concatenate(
                        [item.processed[key] for item in chunks], axis=0
                    )
                    np.testing.assert_allclose(
                        combined,
                        full.processed[key],
                        rtol=0.0,
                        atol=1e-12,
                        equal_nan=True,
                    )

    def test_ekf_and_lpf_comparison_metrics_are_explicit(self) -> None:
        """在平移burst上记录两路线RMSE / Compare both routes on a known burst."""

        time = np.arange(0.0, 20.0, 1.0 / 400.0)
        dynamic = np.zeros((time.size, 3))
        active = (time >= 5.0) & (time <= 15.0)
        dynamic[active, 0] = 2.0 * np.sin(
            2 * np.pi * 0.2 * time[active]
        )
        acc = dynamic + np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros_like(acc)
        outputs = {}
        for method in ("no_precalibration_ekf", "lpf_0p3"):
            result = preprocess_imu(
                acc,
                gyro,
                fs_hz=400.0,
                acc_unit="m/s2",
                gyro_unit="rad/s",
                gravity_method=method,
            )
            valid = result.valid_mask & active
            outputs[method] = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            result.processed["dynamic_acc_mps2"][valid, 0]
                            - dynamic[valid, 0]
                        )
                    )
                )
            )
        self.assertTrue(all(np.isfinite(value) for value in outputs.values()))
        self.assertNotAlmostEqual(
            outputs["no_precalibration_ekf"],
            outputs["lpf_0p3"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
