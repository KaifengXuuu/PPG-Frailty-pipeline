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
        views = CanonicalSignalViews(
            x_native=ppg.copy(),
            x_filter=ppg.copy(),
            x_analysis_rate=ppg.copy(),
            imu_processed={
                "dynamic_acc_mps2": dynamic,
                "gyro_rads": gyro,
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
        original_imu = np.column_stack((dynamic, gyro))
        self.assertTrue(np.array_equal(raw.values[0, 2:], original_imu[:400].T.astype(np.float32)))
        self.assertTrue(np.array_equal(raw.values[1, 2:], original_imu[400:].T.astype(np.float32)))
        self.assertTrue(np.allclose(np.median(raw.values[:, :2], axis=2), 0.0, atol=1e-6))
        self.assertEqual(raw.provenance["imu_normalization"], "unscaled_si_requires_outer_train_transform")

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
        with self.assertRaisesRegex(ValueError, "artifact identity drift"):
            replace(artifact, valid_count=artifact.valid_count + 1).validate()

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


if __name__ == "__main__":
    unittest.main()
