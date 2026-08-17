"""Raw PPG/IMU normalization boundary tests / raw PPG 与 IMU 归一化边界测试。"""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from ppg_frailty.contracts import SignalRoute
from ppg_frailty.representations import (
    RawWindows,
    build_raw_windows,
    fit_fold_imu_channel_transform,
    transform_raw_windows_imu,
)
from ppg_frailty.representations.imu_transform import (
    IMU_TRANSFORM_SCHEMA_VERSION,
    IQR_NORMAL_CONSISTENCY_DIVISOR,
)
from ppg_frailty.signal.views import CanonicalSignalViews, WindowPlan


class RawImuTransformTests(unittest.TestCase):
    """PPG window scaling and train-only IMU scaling / 两阶段缩放。"""

    def test_raw_builder_scales_only_red_ir(self) -> None:
        samples = 800
        time = np.arange(samples, dtype=np.float64) / 400.0
        ppg = np.column_stack((
            1000.0 + 20.0 * np.sin(2.0 * np.pi * 1.2 * time),
            1200.0 + 15.0 * np.sin(2.0 * np.pi * 1.1 * time),
        ))
        dynamic = np.column_stack((time, 2.0 * time, -3.0 * time))
        gyro = np.column_stack((0.1 + time, 0.2 - time, 0.3 + 0.5 * time))
        dynamic_magnitude = np.linalg.norm(dynamic, axis=1)
        gyro_magnitude = np.linalg.norm(gyro, axis=1)
        jerk_magnitude = np.linalg.norm(
            np.diff(dynamic, axis=0, prepend=dynamic[:1]) * 400.0,
            axis=1,
        )
        views = CanonicalSignalViews(
            x_native=ppg.copy(),
            x_filter=ppg.copy(),
            x_analysis_rate=ppg.copy(),
            imu_processed={
                "dynamic_acc_mps2": dynamic,
                "gyro_rads": gyro,
                "dynamic_magnitude": dynamic_magnitude,
                "gyro_magnitude": gyro_magnitude,
                "jerk_magnitude": jerk_magnitude,
                "imu_valid_mask": np.ones(samples, dtype=bool),
            },
            metadata={"record_id": "raw_test", "fs_hz": 400.0},
            source_valid_mask=np.ones_like(ppg, dtype=bool),
            repair_mask=np.zeros_like(ppg, dtype=bool),
            route=SignalRoute.DIRECT,
        )
        plan = WindowPlan(
            source_record_id="raw_test",
            window_seconds=1.0,
            hop_seconds=1.0,
            end_alignment="start",
            short_record_action="reject",
            include_padded_tail=False,
            max_windows=None,
            cap_policy="not_applicable",
        )
        raw = build_raw_windows(views, plan)
        original_imu = np.column_stack(
            (
                dynamic,
                gyro,
            )
        )
        self.assertTrue(np.array_equal(raw.values[0, 2:], original_imu[:400].T.astype(np.float32)))
        self.assertTrue(np.array_equal(raw.values[1, 2:], original_imu[400:].T.astype(np.float32)))
        self.assertTrue(np.allclose(np.median(raw.values[:, :2], axis=2), 0.0, atol=1e-6))
        self.assertEqual(raw.values.shape, (2, 8, 400))
        first_red = ppg[:400, 0]
        q25, q75 = np.percentile(first_red, [25.0, 75.0])
        expected_red = np.clip(
            (first_red - np.median(first_red)) / ((q75 - q25) / 1.349),
            -8.0,
            8.0,
        )
        np.testing.assert_allclose(raw.values[0, 0], expected_red, rtol=1e-6)
        self.assertEqual(
            raw.provenance["imu_normalization"],
            "unscaled_si_requires_outer_train_transform",
        )
        self.assertEqual(len(raw.provenance["imu_channel_schema"]), 6)
        self.assertFalse(raw.provenance["derived_motion_channels_in_frailty_tensor"])

        routed = replace(
            views,
            x_ar=np.zeros_like(ppg),
            route=SignalRoute.ARTIFACT_RATE_ONLY,
            metadata={
                **views.metadata,
                "rate_only": True,
                "q_morph_state": "not_applicable",
                "artifact_output_valid_mask": np.ones(samples, dtype=bool),
            },
        )
        routed_raw = build_raw_windows(routed, plan)
        np.testing.assert_array_equal(routed_raw.values, raw.values)
        self.assertEqual(routed_raw.provenance["ppg_source_view"], "x_filter")

    def test_fold_transform_is_train_only_hash_bound_and_preserves_ppg(self) -> None:
        values = np.zeros((3, 8, 20), dtype=np.float32)
        values[:, 0, :] = np.arange(20, dtype=np.float32)
        values[:, 1, :] = 2.0
        for channel in range(6):
            values[0, channel + 2, :] = np.linspace(channel, channel + 2.0, 20)
            values[1, channel + 2, :] = np.linspace(channel + 2.0, channel + 4.0, 20)
            values[2, channel + 2, :] = 1000.0 + channel
        artifact = fit_fold_imu_channel_transform(
            values,
            ["train_a", "train_b", "heldout"],
            fitted_on_participant_ids=["train_a", "train_b"],
            outer_train_participant_ids=["train_a", "train_b"],
            outer_oof_participant_ids=["heldout"],
        )
        raw = RawWindows(
            values=values,
            valid_mask=np.ones((3, 20), dtype=bool),
            start_samples=np.array([0, 20, 40]),
            candidate_count=3,
            dropped_invalid_count=0,
        )
        transformed = transform_raw_windows_imu(raw, artifact)
        self.assertTrue(np.array_equal(transformed.values[:, :2], values[:, :2]))
        self.assertTrue(np.isfinite(transformed.values).all())
        self.assertEqual(transformed.provenance["imu_transform_sha256"], artifact.artifact_sha256)
        self.assertEqual(artifact.valid_count.tolist(), [40] * 6)
        train_samples = np.asarray(values[:2, 2, :], dtype=np.float64).reshape(-1)
        q25, q75 = np.percentile(train_samples, [25.0, 75.0])
        self.assertAlmostEqual(
            float(artifact.scale[0]),
            float(q75 - q25) / IQR_NORMAL_CONSISTENCY_DIVISOR,
        )
        with self.assertRaisesRegex(ValueError, "artifact identity drift"):
            replace(artifact, valid_count=artifact.valid_count + 1).validate()
        with self.assertRaisesRegex(ValueError, "artifact identity drift"):
            replace(
                artifact,
                schema_version=(
                    "raw_frailty_imu_axes6_outer_train_"
                    "median_iqr_population_sd_v2"
                ),
            ).validate()

    def test_oof_participant_cannot_fit_imu_transform(self) -> None:
        values = np.ones((2, 8, 12), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-training participant"):
            fit_fold_imu_channel_transform(
                values,
                ["train", "heldout"],
                fitted_on_participant_ids=["train", "heldout"],
                outer_train_participant_ids=["train"],
                outer_oof_participant_ids=["heldout"],
            )

    def test_fold_transform_falls_back_from_iqr_to_population_sd_then_one(
        self,
    ) -> None:
        values = np.zeros((1, 8, 20), dtype=np.float32)
        values[0, 2, -1] = 10.0
        artifact = fit_fold_imu_channel_transform(
            values,
            ["train"],
            fitted_on_participant_ids=["train"],
            outer_train_participant_ids=["train"],
            outer_oof_participant_ids=[],
        )
        expected_sd = float(
            np.std(np.asarray(values[0, 2], dtype=np.float64), ddof=0)
        )
        self.assertAlmostEqual(float(artifact.scale[0]), expected_sd)
        self.assertEqual(float(artifact.scale[1]), 1.0)
        self.assertEqual(
            artifact.schema_version,
            IMU_TRANSFORM_SCHEMA_VERSION,
        )

        raw = RawWindows(
            values=values,
            valid_mask=np.ones((1, 20), dtype=bool),
            start_samples=np.array([0]),
            candidate_count=1,
            dropped_invalid_count=0,
        )
        transformed = transform_raw_windows_imu(raw, artifact)
        self.assertEqual(
            transformed.provenance["imu_normalization"],
            "outer_train_median_iqr_over_1p349_population_sd_then_one",
        )


if __name__ == "__main__":
    unittest.main()
