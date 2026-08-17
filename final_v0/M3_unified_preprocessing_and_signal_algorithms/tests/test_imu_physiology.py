"""ESKF/LPF、peak/PPI/PRV 固定测试 / IMU and physiology reference tests."""

from __future__ import annotations

import unittest

import numpy as np

from ._support import load_fixture
from m3_signal_core import (
    CausalImuProcessor,
    PeakResult,
    ProcessingStatus,
    STANDARD_GRAVITY_MPS2,
    choose_primary_channel,
    compute_prv,
    convert_imu_to_si,
    derive_ppi,
    detect_peaks_corrected,
    dual_channel_agreement,
    preprocess_imu,
    preprocess_ppg,
    vector_jerk,
)
from m3_signal_core.imu_math import (
    NoPrecalibrationEskf,
    filter_axes,
)


class ImuReferenceTests(unittest.TestCase):
    """验证显式单位、causal state 和 ESKF 状态 / Validate IMU contracts."""

    def test_unit_equivalence(self) -> None:
        """g/deg/s 与 SI 输入必须严格等价 / Equivalent units yield equal SI."""

        acc_g = np.array([[0.1, -0.2, 1.0], [0.2, 0.0, 0.98]])
        gyro_dps = np.array([[10.0, -5.0, 1.0], [0.0, 2.0, -3.0]])
        acc_si, gyro_si = convert_imu_to_si(
            acc_g,
            gyro_dps,
            acceleration_unit="g",
            gyroscope_unit="deg/s",
        )
        acc_same, gyro_same = convert_imu_to_si(
            acc_si,
            gyro_si,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
        )
        np.testing.assert_allclose(acc_same, acc_si, atol=1e-12)
        np.testing.assert_allclose(gyro_same, gyro_si, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "unit_unknown"):
            convert_imu_to_si(
                acc_g,
                gyro_dps,
                acceleration_unit="guessed",
                gyroscope_unit="deg/s",
            )

    def test_causal_filter_chunk_parity(self) -> None:
        """任意分块与整段 causal SOS 一致 / Chunked and full filtering agree."""

        rng = np.random.default_rng(42)
        values = rng.normal(size=(2000, 3))
        full, _ = filter_axes(
            values, 400.0, 20.0, order=3, phase_mode="causal_stateful"
        )
        first, state = filter_axes(
            values[:731], 400.0, 20.0, order=3, phase_mode="causal_stateful"
        )
        second, _ = filter_axes(
            values[731:],
            400.0,
            20.0,
            order=3,
            phase_mode="causal_stateful",
            initial_state=state,
        )
        np.testing.assert_allclose(np.vstack([first, second]), full, atol=1e-12)

    def test_static_ekf_reaches_tracking_without_precalibration(self) -> None:
        """水平静态在线初始化后重力准确 / Static online ESKF becomes valid."""

        count = 2400
        acceleration = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (count, 1))
        gyroscope = np.zeros((count, 3))
        result = preprocess_imu(
            acceleration,
            gyroscope,
            400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
            profile_id="imu_ekf_si_400_causal_v1",
        )
        self.assertEqual(result.status, ProcessingStatus.PARTIAL)
        self.assertGreater(int(np.sum(result.sample_valid_mask)), 1000)
        error = np.linalg.norm(
            result.gravity_mps2[result.sample_valid_mask]
            - np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2]),
            axis=1,
        )
        self.assertLess(float(np.max(error)), 1e-8)
        self.assertTrue(result.diagnostics["no_static_precalibration"])

    def test_invalid_acceleration_stays_pending(self) -> None:
        """无物理重力向量时不伪造输出 / Invalid norm remains pending."""

        estimator = NoPrecalibrationEskf(400.0)
        for index in range(100):
            state = estimator.step(np.zeros(3), np.zeros(3), index)
        self.assertEqual(state["state"], "initialization_pending")
        self.assertFalse(state["valid"])

    def test_ekf_and_lpf_have_separate_results(self) -> None:
        """同一 fixture 并行产生独立主/对照结果 / Routes remain isolated."""

        fixture = load_fixture("imu_reference_v1.npy")
        acceleration = fixture[:, :3]
        gyroscope = fixture[:, 3:6]
        ekf = preprocess_imu(
            acceleration,
            gyroscope,
            400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
            profile_id="imu_ekf_si_400_causal_v1",
        )
        lpf = preprocess_imu(
            acceleration,
            gyroscope,
            400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
            profile_id="imu_lpf_si_400_causal_v1",
        )
        self.assertEqual(
            ekf.diagnostics["gravity_method"],
            "quaternion_error_state_ekf_without_precalibration",
        )
        self.assertEqual(
            lpf.diagnostics["gravity_method"], "second_order_lowpass_0p3_hz"
        )
        self.assertFalse(ekf.diagnostics["silent_fallback"])
        self.assertFalse(lpf.diagnostics["silent_fallback"])
        self.assertTrue(np.any(ekf.sample_valid_mask))
        self.assertTrue(np.all(np.isfinite(lpf.gravity_mps2)))

    def test_stateful_imu_chunk_parity(self) -> None:
        """整段与两块 runtime 必须一致 / Stateful chunks must equal one-shot."""

        fixture = load_fixture("imu_reference_v1.npy")
        acceleration = fixture[:, :3]
        gyroscope = fixture[:, 3:6]
        one_shot = preprocess_imu(
            acceleration,
            gyroscope,
            400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
            profile_id="imu_ekf_si_400_causal_v1",
        )
        processor = CausalImuProcessor(
            "imu_ekf_si_400_causal_v1",
            fs_hz=400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
        )
        first = processor.process_chunk(acceleration[:2400], gyroscope[:2400])
        second = processor.process_chunk(acceleration[2400:], gyroscope[2400:])
        np.testing.assert_allclose(
            np.vstack([first.gravity_mps2, second.gravity_mps2]),
            one_shot.gravity_mps2,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.vstack([first.jerk_mps3, second.jerk_mps3]),
            one_shot.jerk_mps3,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.concatenate([first.sample_valid_mask, second.sample_valid_mask]),
            one_shot.sample_valid_mask,
        )

    def test_profile_mismatch_and_m1_unit_alias(self) -> None:
        """拒绝采样率伪装并兼容 M1 m/s2 / Reject mismatch; accept M1 unit."""

        acceleration = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (1200, 1))
        gyroscope = np.zeros((1200, 3))
        with self.assertRaisesRegex(ValueError, "profile_mismatch"):
            preprocess_imu(
                acceleration,
                gyroscope,
                200.0,
                acceleration_unit="m/s2",
                gyroscope_unit="rad/s",
                profile_id="imu_ekf_si_400_causal_v1",
            )
        result = preprocess_imu(
            acceleration,
            gyroscope,
            400.0,
            acceleration_unit="m/s2",
            gyroscope_unit="rad/s",
            profile_id="imu_ekf_si_400_causal_v1",
        )
        self.assertTrue(np.any(result.sample_valid_mask))

    def test_ekf_synthetic_truth_and_finite_common_mask(self) -> None:
        """合成真值门与公共 mask 有限性 / Assert truth gate and finite outputs."""

        fixture = load_fixture("imu_reference_v1.npy")
        result = preprocess_imu(
            fixture[:, :3],
            fixture[:, 3:6],
            400.0,
            acceleration_unit="m/s^2",
            gyroscope_unit="rad/s",
            profile_id="imu_ekf_si_400_causal_v1",
        )
        mask = result.sample_valid_mask
        error = np.linalg.norm(
            result.dynamic_acc_mps2[mask] - fixture[mask, 9:12], axis=1
        )
        self.assertLess(float(np.sqrt(np.mean(error**2))), 0.35)
        self.assertTrue(np.isfinite(result.jerk_mps3[mask]).all())

    def test_no_estimate_is_latched_until_explicit_reset(self) -> None:
        """超时后不可单帧假恢复 / Timeout remains latched until explicit reset."""

        estimator = NoPrecalibrationEskf(400.0)
        for index in range(800):
            state = estimator.step(
                np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2]),
                np.zeros(3),
                index,
            )
        self.assertTrue(state["valid"])
        for index in range(800, 1805):
            state = estimator.step(
                np.array([20.0, 0.0, 0.0]), np.zeros(3), index
            )
        self.assertEqual(state["state"], "no_estimate")
        recovered = estimator.step(
            np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2]),
            np.zeros(3),
            1806,
        )
        self.assertEqual(recovered["state"], "no_estimate")
        self.assertFalse(recovered["valid"])

    def test_vector_jerk_definition(self) -> None:
        """jerk 使用逐轴 backward difference / Verify vector definition."""

        acceleration = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
        jerk = vector_jerk(acceleration, 400.0)
        self.assertTrue(np.isnan(jerk[0]).all())
        np.testing.assert_allclose(jerk[1], [40.0, -80.0, 120.0])


def manual_peak_result(ppi_sec: np.ndarray, fs_hz: float = 400.0) -> PeakResult:
    """从 PPI 构造公式 fixture / Build a formula-level PeakResult."""

    ppi = np.asarray(ppi_sec, dtype=np.float64)
    peaks = np.concatenate(
        ([0], np.rint(np.cumsum(ppi) * float(fs_hz)).astype(np.int64))
    )
    mask = (ppi >= 0.30) & (ppi <= 2.00)
    return PeakResult(
        ProcessingStatus.VALID,
        peaks,
        np.ones(peaks.size),
        1,
        ppi,
        mask,
        ppi[mask],
        ppi[mask].copy(),
        float(mask.sum()),
        ["manual_formula_fixture"],
    )


class PhysiologyReferenceTests(unittest.TestCase):
    """验证双极性、PPI 边界和 PRV 公式 / Validate corrected physiology."""

    def test_polarity_invariance(self) -> None:
        """波形反相只改变 polarity / Inversion preserves peak indices."""

        raw = load_fixture("ppg_reference_v1.npy")
        preprocessed = preprocess_ppg(
            raw,
            400.0,
            profile_id="frailty3_motion_ppg_400_offline_v1",
        )
        positive = detect_peaks_corrected(preprocessed.filtered, 400.0)
        negative = detect_peaks_corrected(-preprocessed.filtered, 400.0)
        np.testing.assert_array_equal(positive.peaks, negative.peaks)
        self.assertEqual(positive.polarity, -negative.polarity)

    def test_peak_fixture_event_recall(self) -> None:
        """合成真值峰 recall 达到工程门 / Check synthetic event recall."""

        raw = load_fixture("ppg_reference_v1.npy")
        expected = load_fixture("ppg_expected_peaks_v1.npy")
        filtered = preprocess_ppg(
            raw,
            400.0,
            profile_id="frailty3_motion_ppg_400_offline_v1",
        ).filtered
        detected = detect_peaks_corrected(filtered, 400.0).peaks
        matched = sum(
            np.any(np.abs(detected - int(reference)) <= 20)
            for reference in expected
        )
        self.assertGreaterEqual(matched / expected.size, 0.90)

    def test_ppi_hard_boundaries(self) -> None:
        """0.30 和 2.00 秒包含，外侧拒绝 / Verify inclusive bounds."""

        peaks = np.array([0, 119, 239, 1039, 1840], dtype=np.int64)
        raw, mask = derive_ppi(peaks, 400.0)
        np.testing.assert_allclose(raw, [0.2975, 0.30, 2.00, 2.0025])
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_time_domain_prv_formula(self) -> None:
        """SDNN/RMSSD/pNN50 精确遵循定义 / Check exact PRV formulas."""

        base = np.array([0.90, 1.00, 1.10, 0.95, 1.05])
        ppi = np.tile(base, 14)
        result = compute_prv(manual_peak_result(ppi), 70.0)
        expected_ms = ppi * 1000.0
        expected_diff = np.diff(expected_ms)
        self.assertAlmostEqual(
            result.metrics["sdnn_ms"], float(np.std(expected_ms, ddof=1))
        )
        self.assertAlmostEqual(
            result.metrics["rmssd_ms"],
            float(np.sqrt(np.mean(expected_diff**2))),
        )
        self.assertAlmostEqual(
            result.metrics["pnn50_fraction"],
            float(np.mean(np.abs(expected_diff) > 50.0)),
        )

    def test_frequency_tiers(self) -> None:
        """120/300 秒 frequency tier 分开 / Separate exploratory/confirmatory tiers."""

        exploratory = compute_prv(manual_peak_result(np.ones(130)), 130.0)
        confirmatory = compute_prv(manual_peak_result(np.ones(310)), 310.0)
        self.assertEqual(
            exploratory.metrics["frequency_tier"], "exploratory_120s"
        )
        self.assertEqual(
            confirmatory.metrics["frequency_tier"], "confirmatory_300s"
        )

    def test_dual_channel_tie_break_and_no_shift(self) -> None:
        """SQI 平局选 RED 且不生成共识峰 / Deterministic dual-channel semantics."""

        red = manual_peak_result(np.ones(10))
        infrared = manual_peak_result(np.ones(10))
        infrared.peaks = infrared.peaks + 8
        result = choose_primary_channel(
            red,
            infrared,
            red_sqi=0.8,
            infrared_sqi=0.8,
            fs_hz=400.0,
        )
        self.assertEqual(result["selected_channel"], "RED")
        self.assertFalse(result["consensus_peak_generation"])
        self.assertEqual(
            dual_channel_agreement(red, infrared, 400.0)["f1_50ms"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
