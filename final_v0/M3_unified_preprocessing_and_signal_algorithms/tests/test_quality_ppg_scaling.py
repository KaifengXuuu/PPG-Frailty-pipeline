"""质量门、PPG 与 fold scaling 测试 / Quality, PPG, and scaling tests."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import signal

from ._support import load_fixture
from m3_signal_core import (
    FoldScaler,
    ProcessingStatus,
    detect_peaks_corrected,
    design_ppg_sos,
    inspect_and_repair_signal,
    preprocess_ppg,
    raw_ppg_metrics,
    resample_external_ppg_to_400,
    robust_window_scale,
    validate_channel_contract,
)


class QualityGateTests(unittest.TestCase):
    """验证冻结的 NaN、gap、flatline 和通道语义 / Validate quality semantics."""

    def setUp(self) -> None:
        """创建非平线参考波形 / Build a non-flat reference waveform."""

        self.fs = 400.0
        time = np.arange(4000) / self.fs
        self.signal = np.sin(2.0 * np.pi * 1.2 * time)

    def test_gap_boundary_at_025_seconds(self) -> None:
        """100 samples 修复、101 samples 拒绝 / Check exact gap boundary."""

        # 中文：10,000 样本使 100 点同时满足恰好 1% 总量和 0.25 s 单 gap。
        # English: 10,000 samples make 100 points exactly 1% and exactly 0.25 s.
        time = np.arange(10000) / self.fs
        reference = np.sin(2.0 * np.pi * 1.2 * time)
        repairable = reference.copy()
        repairable[500:600] = np.nan
        repaired = inspect_and_repair_signal(
            repairable,
            self.fs,
            channel_names=["PPG"],
            flatline_channels=["PPG"],
        )
        self.assertEqual(repaired.status, ProcessingStatus.REPAIRED)
        self.assertEqual(int(np.sum(repaired.repair_mask)), 100)
        excessive = reference.copy()
        excessive[500:601] = np.nan
        rejected = inspect_and_repair_signal(
            excessive,
            self.fs,
            channel_names=["PPG"],
            flatline_channels=["PPG"],
        )
        self.assertEqual(rejected.status, ProcessingStatus.INVALID)
        self.assertIn("excessive_gap", [issue.code for issue in rejected.issues])

    def test_boundary_gap_and_nonfinite_fraction(self) -> None:
        """边界不外推且 >1% 无效 / Boundary gaps and excessive fraction fail."""

        boundary = self.signal.copy()
        boundary[:5] = np.nan
        result = inspect_and_repair_signal(
            boundary, self.fs, channel_names=["PPG"], flatline_channels=["PPG"]
        )
        self.assertEqual(result.status, ProcessingStatus.INVALID)
        self.assertIn("boundary_gap", [issue.code for issue in result.issues])
        excessive = self.signal.copy()
        excessive[100:150] = np.inf
        result = inspect_and_repair_signal(
            excessive, self.fs, channel_names=["PPG"], flatline_channels=["PPG"]
        )
        self.assertIn("excessive_nonfinite", [issue.code for issue in result.issues])

    def test_exact_flatline_threshold(self) -> None:
        """399 samples 通过，400 samples 拒绝 / Validate one-second threshold."""

        below = self.signal.copy()
        below[500:899] = 0.25
        result_below = inspect_and_repair_signal(
            below, self.fs, channel_names=["PPG"], flatline_channels=["PPG"]
        )
        self.assertNotIn("flatline", [issue.code for issue in result_below.issues])
        exact = self.signal.copy()
        exact[500:900] = 0.25
        result_exact = inspect_and_repair_signal(
            exact, self.fs, channel_names=["PPG"], flatline_channels=["PPG"]
        )
        self.assertIn("flatline", [issue.code for issue in result_exact.issues])
        self.assertEqual(result_exact.status, ProcessingStatus.INVALID)

    def test_channel_contract_is_order_sensitive(self) -> None:
        """交换通道必须失败 / Swapped channels must be detected."""

        issues = validate_channel_contract(
            ["IR", "RED", "AX"], ["RED", "IR", "AX"]
        )
        self.assertIn("channel_order_mismatch", [issue.code for issue in issues])


class PpgProfileTests(unittest.TestCase):
    """验证滤波 profile、重采样和幅值保留 / Validate PPG profiles."""

    def test_frozen_filter_response(self) -> None:
        """检查 35 bpm 保留与 8 Hz 截止 / Check frozen response anchors."""

        static_sos = design_ppg_sos(400.0, 0.2, 8.0, order=3)
        motion_sos = design_ppg_sos(400.0, 0.4, 8.0, order=3)
        frequencies = np.array([35.0 / 60.0, 8.0])
        _, static_response = signal.sosfreqz(
            static_sos, worN=frequencies, fs=400.0
        )
        _, motion_response = signal.sosfreqz(
            motion_sos, worN=frequencies, fs=400.0
        )
        self.assertGreater(abs(static_response[0]) ** 2, 0.99)
        self.assertGreater(abs(motion_response[0]) ** 2, 0.90)
        self.assertAlmostEqual(abs(static_response[1]) ** 2, 0.5, places=6)

    def test_ppg_preprocessing_preserves_raw_metrics(self) -> None:
        """滤波前 DC 必须保留 / Preserve pre-filter DC descriptors."""

        raw = load_fixture("ppg_reference_v1.npy")
        result = preprocess_ppg(
            raw,
            400.0,
            profile_id="frailty3_motion_ppg_400_offline_v1",
        )
        self.assertIn(result.status, {ProcessingStatus.VALID, ProcessingStatus.REPAIRED})
        self.assertIsNotNone(result.filtered)
        expected = raw_ppg_metrics(raw)
        self.assertAlmostEqual(result.raw_metrics["dc_median"], expected["dc_median"])
        self.assertEqual(result.filter_metadata["notch"], "disabled")

    def test_resampling_maps_all_payloads_with_one_profile(self) -> None:
        """500→400 同步映射波形、时间、mask 和峰 / Map every payload."""

        values = np.sin(2.0 * np.pi * 1.0 * np.arange(5000) / 500.0)
        timestamps = 10.0 + np.arange(values.size) / 500.0
        valid = np.ones(values.size, dtype=bool)
        valid[250] = False
        result = resample_external_ppg_to_400(
            values,
            500.0,
            timestamps_s=timestamps,
            valid_mask=valid,
            peak_annotations=np.array([500, 1000], dtype=np.int64),
        )
        self.assertEqual(result.signal.size, 4000)
        self.assertEqual(result.metadata["up"], 4)
        self.assertEqual(result.metadata["down"], 5)
        self.assertEqual(result.timestamps_s[0], 10.0)
        np.testing.assert_array_equal(result.peak_annotations, [400, 800])
        self.assertFalse(bool(result.valid_mask[200]))
        self.assertEqual(result.status, ProcessingStatus.PARTIAL)
        self.assertEqual(result.reason_codes, ["source_valid_mask_partial"])

    def test_resampling_rejects_unregistered_source_rate(self) -> None:
        """125 Hz 不得冒充已登记 external profile / Reject unregistered 125 Hz."""

        values = np.arange(1250, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "source_fs_hz"):
            resample_external_ppg_to_400(
                values,
                125.0,
                timestamps_s=np.arange(values.size) / 125.0,
                valid_mask=np.ones(values.size, dtype=bool),
            )

    def test_sim_256_to_400_resampling_is_registered(self) -> None:
        """Sim 256→400 使用冻结 25/16 比例 / Verify registered Sim route."""

        values = np.sin(2.0 * np.pi * np.arange(2560) / 256.0)
        result = resample_external_ppg_to_400(
            values,
            256.0,
            timestamps_s=np.arange(values.size) / 256.0,
            valid_mask=np.ones(values.size, dtype=bool),
            peak_annotations=np.array([256, 512], dtype=np.int64),
        )
        self.assertEqual(result.signal.size, 4000)
        self.assertEqual(result.metadata["up"], 25)
        self.assertEqual(result.metadata["down"], 16)
        np.testing.assert_array_equal(result.peak_annotations, [400, 800])
        self.assertEqual(result.status, ProcessingStatus.VALID)

    def test_peak_detector_requires_peak_purpose_profile(self) -> None:
        """motion 预处理 profile 不能冒充 peak 输入 / Enforce profile purpose."""

        time = np.arange(4000) / 400.0
        filtered = np.sin(2.0 * np.pi * 1.2 * time)
        with self.assertRaisesRegex(ValueError, "not_future_peak_input"):
            detect_peaks_corrected(
                filtered,
                400.0,
                profile_id="frailty3_motion_ppg_400_offline_v1",
            )
        with self.assertRaisesRegex(ValueError, "fs_hz"):
            detect_peaks_corrected(filtered, 256.0)

    def test_deprecated_mobile_alias_cannot_run_preprocessing(self) -> None:
        """旧 mobile 泛化 alias 不得进入新实验 / Reject deprecated alias."""

        values = np.sin(2.0 * np.pi * 1.2 * np.arange(4000) / 400.0)
        with self.assertRaisesRegex(ValueError, "not_future_ppg_preprocessing"):
            preprocess_ppg(
                values,
                400.0,
                profile_id="mobile_ppg_400_causal_v1",
            )


class ScalingTests(unittest.TestCase):
    """验证训练折隔离和零 IQR fail-closed / Validate scaler boundaries."""

    def test_fit_rejects_non_training_role(self) -> None:
        """OOF 不得拟合 scaler / OOF data cannot fit a scaler."""

        with self.assertRaises(ValueError):
            FoldScaler().fit(
                np.arange(12, dtype=float).reshape(4, 3),
                fit_role="oof_validation",
                training_ids=["oof_subject"],
            )

    def test_test_distribution_does_not_refit(self) -> None:
        """修改测试分布不改变训练 artifact / No transductive refit."""

        train = np.arange(30, dtype=float).reshape(10, 3)
        scaler = FoldScaler().fit(
            train, fit_role="training", training_ids=["A", "B"]
        )
        before = scaler.to_dict()
        scaler.transform(np.full((5, 3), 1e9))
        self.assertEqual(before, scaler.to_dict())

    def test_reversible_window_scaling_and_zero_iqr(self) -> None:
        """robust view 可逆，零 IQR 明确失败 / Check reversibility and failure."""

        values = np.column_stack(
            [np.arange(20, dtype=float), np.arange(20, dtype=float) ** 2]
        )
        normalized, center, scale = robust_window_scale(values)
        np.testing.assert_allclose(normalized * scale + center, values)
        with self.assertRaisesRegex(ValueError, "zero_iqr"):
            robust_window_scale(np.ones((20, 2)))


if __name__ == "__main__":
    unittest.main()
