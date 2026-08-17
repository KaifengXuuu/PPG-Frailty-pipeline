"""Router no-fallback and rate-only integration tests / 路由集成测试。"""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.artifacts import run_artifact_route
from ppg_frailty.contracts import QualityState, SignalRoute
from ppg_frailty.signal import SqiConfig, build_signal_views, evaluate_quality


def views_fixture():
    """构建可路由的同步视图 / Build synchronized canonical views."""

    samples = 4800
    time = np.arange(samples) / 400.0
    cardiac = np.sin(2 * np.pi * 1.2 * time)
    motion = 0.4 * np.sin(2 * np.pi * 2.1 * time)
    ppg = np.column_stack((1000 + 20 * cardiac + 10 * motion, 1200 + 15 * cardiac - 8 * motion))
    acc = np.column_stack((motion, 0.5 * motion, np.full(samples, 9.80665)))
    gyro = np.column_stack((0.1 * motion, np.zeros(samples), np.zeros(samples)))
    config = {
        "signal": {
            "internal_fs_hz": 400.0,
            "channel_order": ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"],
            "ppg_native_unit": "raw_counts",
            "accelerometer_input_unit": "m/s2",
            "gyroscope_input_unit": "rad/s",
            "ppg_filter": {
                "family": "butterworth_sos", "order": 3,
                "low_hz": 0.2, "high_hz": 8.0,
                "phase": "zero_phase", "short_signal_policy": "reject",
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
    return build_signal_views(
        {"record_id": "route_fixture", "ppg": ppg, "acc": acc, "gyro": gyro,
         "acc_unit": "m/s2", "gyro_unit": "rad/s", "fs_hz": 400.0},
        config,
    )


class RouterTest(unittest.TestCase):
    """验证 route 和 Q_morph 边界 / Verify route and morphology-quality boundary."""

    def test_identity_remains_direct_eligible(self) -> None:
        outcome = run_artifact_route(views_fixture(), "identity")
        self.assertIs(outcome.route, SignalRoute.IDENTITY)
        self.assertIsNotNone(outcome.views)
        quality = evaluate_quality(outcome.views, config=SqiConfig())
        self.assertIsNot(quality.q_morph.state, QualityState.NOT_APPLICABLE)

    def test_nonidentity_forces_rate_only(self) -> None:
        outcome = run_artifact_route(
            views_fixture(),
            "spectral_mask",
            parameters={
                "stft_window_s": 4.0,
                "stft_hop_s": 1.0,
                "imu_mask_quantile": 0.75,
                "mask_strength": 0.8,
                "preserve_band_hz": [0.5, 3.0],
            },
        )
        self.assertIs(outcome.route, SignalRoute.ARTIFACT_RATE_ONLY)
        self.assertTrue(outcome.views.metadata["rate_only"])
        self.assertIs(
            evaluate_quality(
                outcome.views, config=SqiConfig()
            ).q_morph.state,
            QualityState.NOT_APPLICABLE,
        )

    def test_failure_drops_without_direct_fallback(self) -> None:
        views = views_fixture()
        outcome = run_artifact_route(
            views.x_filter, "nlms", imu_processed=None,
            parameters={"taps_per_delay": 2, "delay_taps": [0, 2]},
        )
        self.assertIs(outcome.route, SignalRoute.DROPPED)
        self.assertIsNone(outcome.views)
        self.assertIsNone(outcome.result.x_ar)


if __name__ == "__main__":
    unittest.main()
