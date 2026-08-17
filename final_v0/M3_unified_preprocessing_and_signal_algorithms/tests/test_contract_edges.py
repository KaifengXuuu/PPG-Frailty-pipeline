"""M3 合同边界回归 / M3 contract-edge regressions."""

from __future__ import annotations

import json
import unittest

import numpy as np

from ._support import load_fixture
from m3_signal_core import (
    FoldAmplitudeRiskModel,
    PeakResult,
    ProcessingStatus,
    choose_primary_channel,
    compute_prv,
    detect_peaks_corrected,
    dual_ppg_raw_metrics,
    map_processing_status_to_m1,
    preprocess_ppg,
    to_serializable,
)


def peak_result(status: ProcessingStatus, ppi: np.ndarray) -> PeakResult:
    """构造最小峰合同 / Build a minimal peak contract."""

    values = np.asarray(ppi, dtype=np.float64)
    peaks = np.concatenate(([0], np.rint(np.cumsum(values) * 400.0).astype(np.int64)))
    valid = (values >= 0.30) & (values <= 2.00)
    return PeakResult(
        status,
        peaks,
        np.ones(peaks.size),
        1,
        values,
        valid,
        values[valid],
        values[valid],
        float(valid.sum()),
        [],
    )


class ContractEdgeTests(unittest.TestCase):
    """验证 registry、JSON、状态与 SQI 边界 / Validate contract boundaries."""

    def test_strict_json_replaces_nonfinite_with_null(self) -> None:
        """NaN/Inf 必须转 null / Non-finite values must serialize as null."""

        result = PeakResult(
            ProcessingStatus.INSUFFICIENT,
            np.empty(0, dtype=np.int64),
            np.array([np.nan]),
            0,
            np.empty(0),
            np.empty(0, dtype=bool),
            np.empty(0),
            np.empty(0),
            float("-inf"),
            ["DURATION_LT_8S"],
        )
        payload = to_serializable(result)
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload["score"])
        self.assertIsNone(payload["peak_confidence"][0])

    def test_profile_provenance_is_not_a_reason_code(self) -> None:
        """profile 单独序列化 / Profile provenance stays outside reasons."""

        result = detect_peaks_corrected(
            np.zeros(8 * 400 - 1, dtype=np.float64),
            400.0,
            profile_id="frailty3_peak_ppg_400_offline_v1",
        )
        payload = to_serializable(result)
        self.assertEqual(payload["profile_id"], "frailty3_peak_ppg_400_offline_v1")
        self.assertNotIn("frailty3_peak_ppg_400_offline_v1", payload["reason_codes"])
        self.assertEqual(payload["nni_semantics"], "hard_valid_ppi_no_imputation_v1")

    def test_m1_status_mapping_is_end_of_stream_aware(self) -> None:
        """pending 在流中与结束时语义不同 / Pending differs at end of stream."""

        self.assertEqual(
            map_processing_status_to_m1(
                ProcessingStatus.INITIALIZATION_PENDING, end_of_stream=False
            ),
            "processing_lag",
        )
        self.assertEqual(
            map_processing_status_to_m1(
                ProcessingStatus.INITIALIZATION_PENDING, end_of_stream=True
            ),
            "insufficient_quality",
        )
        self.assertEqual(
            map_processing_status_to_m1(ProcessingStatus.PARTIAL, end_of_stream=True),
            "partial",
        )

    def test_ppg_profile_and_timestamp_are_enforced(self) -> None:
        """拒绝伪采样率和重复时间 / Reject fake fs and repeated timestamps."""

        values = np.sin(2.0 * np.pi * np.arange(1600) / 400.0)
        with self.assertRaisesRegex(ValueError, "profile_mismatch"):
            preprocess_ppg(
                values,
                256.0,
                profile_id="frailty3_motion_ppg_400_offline_v1",
            )
        timestamps = np.arange(values.size, dtype=np.float64) / 400.0
        timestamps[800] = timestamps[799]
        result = preprocess_ppg(
            values,
            400.0,
            profile_id="frailty3_motion_ppg_400_offline_v1",
            timestamps_s=timestamps,
        )
        self.assertEqual(result.status, ProcessingStatus.INVALID)
        self.assertIn(
            "timestamp_not_strictly_increasing",
            {issue.code for issue in result.quality.issues},
        )

    def test_dual_raw_metrics_and_training_fold_amplitude_gate(self) -> None:
        """比例保留且幅值模型只准训练折拟合 / Preserve ratios; fit on train."""

        red = np.array([10.0, 12.0, 8.0, 10.0])
        infrared = np.array([20.0, 24.0, 16.0, 20.0])
        metrics = dual_ppg_raw_metrics(red, infrared)
        self.assertAlmostEqual(metrics["red_ir_dc_ratio"], 0.5)
        self.assertAlmostEqual(metrics["red_ir_ac_ratio"], 0.5)
        model = FoldAmplitudeRiskModel().fit(
            np.array([[1.0, 2.0, -1.0], [1.1, 2.1, -0.9], [0.9, 1.9, -1.1]]),
            fit_role="training",
            training_ids=["S1", "S2", "S3"],
        )
        self.assertFalse(model.evaluate(np.array([1.0, 2.0, -1.0]))["sqi_risk"])
        with self.assertRaisesRegex(ValueError, "restricted to training"):
            FoldAmplitudeRiskModel().fit(
                np.ones((2, 3)),
                fit_role="oof_validation",
                training_ids=["S4", "S5"],
            )

    def test_source_raw_and_repaired_ppg_views_are_distinct(self) -> None:
        """插值不得覆盖 source raw / Interpolation must not overwrite source raw."""

        values = np.sin(2.0 * np.pi * np.arange(1600) / 400.0)
        values[800] = np.nan
        result = preprocess_ppg(
            values,
            400.0,
            profile_id="frailty3_motion_ppg_400_offline_v1",
        )
        self.assertEqual(result.status, ProcessingStatus.REPAIRED)
        self.assertTrue(np.isnan(result.source_raw[800]))
        self.assertTrue(np.isfinite(result.repaired_raw).all())
        self.assertAlmostEqual(result.raw_metrics["source_nonfinite_fraction"], 1 / 1600)
        self.assertIn("source_dc_median", result.raw_metrics)
        self.assertIn("repaired_dc_median", result.raw_metrics)

    def test_prv_low_coverage_is_partial(self) -> None:
        """60 s 声称不能掩盖 5 s PPI / Claimed duration cannot hide low coverage."""

        result = compute_prv(peak_result(ProcessingStatus.VALID, np.ones(5)), 60.0)
        self.assertEqual(result.status, ProcessingStatus.PARTIAL)
        self.assertIsNone(result.metrics["sdnn_ms"])
        self.assertIn("TIME_DOMAIN_PRV_COVERAGE_LT_0P80", result.reason_codes)

    def test_invalid_channel_and_nan_sqi_cannot_win(self) -> None:
        """无效通道或 NaN SQI 不得成为 primary / Invalid inputs cannot win."""

        invalid = peak_result(ProcessingStatus.INVALID, np.ones(5))
        valid = peak_result(ProcessingStatus.VALID, np.ones(5))
        selected = choose_primary_channel(
            invalid, valid, red_sqi=0.9, infrared_sqi=0.8, fs_hz=400.0
        )
        self.assertEqual(selected["selected_channel"], "IR")
        unavailable = choose_primary_channel(
            valid, valid, red_sqi=np.nan, infrared_sqi=np.nan, fs_hz=400.0
        )
        self.assertIsNone(unavailable["selected_channel"])
        self.assertEqual(unavailable["selection_reason"], "no_valid_channel")

    def test_peak_confidence_is_bounded(self) -> None:
        """峰 confidence 必须严格在 0–1 / Peak confidence must be bounded."""

        raw = load_fixture("ppg_reference_v1.npy")
        filtered = preprocess_ppg(
            raw,
            400.0,
            profile_id="frailty3_peak_ppg_400_offline_v1",
        ).filtered
        result = detect_peaks_corrected(filtered, 400.0)
        self.assertTrue(np.all(result.peak_confidence >= 0.0))
        self.assertTrue(np.all(result.peak_confidence <= 1.0))


if __name__ == "__main__":
    # 中文：保留 unittest 直接执行，同时兼容 pytest discovery。
    # English: Keep direct unittest execution and pytest discovery.
    unittest.main()
