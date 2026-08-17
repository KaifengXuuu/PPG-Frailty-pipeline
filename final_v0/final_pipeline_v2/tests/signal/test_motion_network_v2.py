"""Motion 8-channel reference and named 11-channel augmentation tests."""

from __future__ import annotations

import unittest

import numpy as np

from ppg_frailty.representations.motion import (
    MOTION_AUGMENTED_CHANNEL_SCHEMA,
    MOTION_AUGMENTED_SCHEMA_SHA256,
    MOTION_DERIVED_AUGMENTATION_PROFILE_ID,
    MOTION_NETWORK_CHANNEL_SCHEMA,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_SCALER_SCHEMA,
    apply_motion_fold_imu_transform,
    build_motion_window_tensors,
    fit_motion_fold_imu_transform,
)
from ppg_frailty.signal.motion_imu import (
    MOTION_IMU_CHANNEL_SCHEMA,
    MOTION_IMU_CHANNEL_UNITS,
    MotionImuResult,
)


def _imu(values: np.ndarray) -> MotionImuResult:
    samples = values.shape[0]
    result = MotionImuResult(
        values=values,
        channel_schema=MOTION_IMU_CHANNEL_SCHEMA,
        channel_units=MOTION_IMU_CHANNEL_UNITS,
        roll_rad=np.zeros(samples),
        pitch_rad=np.zeros(samples),
        gravity_mps2=np.tile([0.0, 0.0, 9.80665], (samples, 1)),
        valid_mask=np.ones(samples, dtype=bool),
        profile_id="unit_test_calibrated_ekf",
        diagnostics={"silent_fallback": False},
    )
    result.validate()
    return result


class MotionNetworkTests(unittest.TestCase):
    def test_exact_schema_window_hop_and_ppg_only_window_scaling(self) -> None:
        samples = 4000
        time = np.arange(samples, dtype=np.float64) / 400.0
        ppg = np.column_stack(
            (
                1000.0 + 20.0 * np.sin(2.0 * np.pi * time),
                1200.0 + 15.0 * np.cos(2.0 * np.pi * time),
            )
        )
        imu_values = np.column_stack(
            [time * (index + 1.0) for index in range(9)]
        )
        windows = build_motion_window_tensors(
            ppg,
            _imu(imu_values),
            record_id="p1_B",
            participant_id="p1",
            role_or_activity="B",
            dataset_id="frailty29",
        )
        self.assertEqual(MOTION_NETWORK_CHANNEL_SCHEMA[:2], ("RED", "IR"))
        self.assertEqual(len(MOTION_NETWORK_CHANNEL_SCHEMA), 8)
        self.assertEqual(windows.schema_sha256, MOTION_NETWORK_SCHEMA_SHA256)
        self.assertEqual(windows.values.shape, (2, 8, 3200))
        self.assertTrue(np.array_equal(windows.start_samples, [0, 800]))
        self.assertTrue(
            np.allclose(np.median(windows.values[:, :2], axis=2), 0.0, atol=1e-6)
        )
        self.assertTrue(
            np.array_equal(
                windows.values[0, 2:],
                imu_values[:3200, :6].T.astype(np.float32),
            )
        )
        augmented = build_motion_window_tensors(
            ppg,
            _imu(imu_values),
            record_id="p1_B",
            participant_id="p1",
            role_or_activity="B",
            dataset_id="frailty29",
            profile_id=MOTION_DERIVED_AUGMENTATION_PROFILE_ID,
        )
        self.assertEqual(augmented.channel_schema, MOTION_AUGMENTED_CHANNEL_SCHEMA)
        self.assertEqual(augmented.schema_sha256, MOTION_AUGMENTED_SCHEMA_SHA256)
        self.assertEqual(augmented.values.shape, (2, 11, 3200))
        np.testing.assert_array_equal(
            augmented.values[0, 8:],
            imu_values[:3200, 6:].T.astype(np.float32),
        )

    def test_reference_scaler_uses_train_participants_and_six_axes(self) -> None:
        values = np.zeros((3, 8, 3200), dtype=np.float32)
        values[:, :2, :] = np.arange(3200, dtype=np.float32)
        for channel in range(6):
            values[0, channel + 2] = np.linspace(channel, channel + 1.0, 3200)
            values[1, channel + 2] = np.linspace(channel + 1.0, channel + 2.0, 3200)
            values[2, channel + 2] = 10000.0 + channel
        transform = fit_motion_fold_imu_transform(
            values,
            ("train1", "train2", "heldout"),
            fitted_on_participant_ids=("train1", "train2"),
            outer_train_participant_ids=("train1", "train2"),
            outer_oof_participant_ids=("heldout",),
        )
        transformed = apply_motion_fold_imu_transform(values, transform)
        self.assertEqual(transform.center.shape, (6,))
        self.assertEqual(transform.valid_count.tolist(), [6400] * 6)
        self.assertTrue(np.array_equal(transformed[:, :2], values[:, :2]))
        self.assertGreater(float(np.min(transformed[2, 2:])), 1000.0)
        with self.assertRaisesRegex(ValueError, "non-training participant"):
            fit_motion_fold_imu_transform(
                values,
                ("train1", "train2", "heldout"),
                fitted_on_participant_ids=("train1", "heldout"),
                outer_train_participant_ids=("train1", "train2"),
                outer_oof_participant_ids=("heldout",),
            )

    def test_named_augmentation_scaler_covers_all_nine_motion_channels(self) -> None:
        values = np.zeros((2, 11, 3200), dtype=np.float32)
        for channel in range(9):
            values[:, channel + 2] = channel + np.linspace(0.0, 2.0, 3200)
        transform = fit_motion_fold_imu_transform(
            values,
            ("train1", "train2"),
            fitted_on_participant_ids=("train1", "train2"),
            outer_train_participant_ids=("train1", "train2"),
            outer_oof_participant_ids=(),
            profile_id=MOTION_DERIVED_AUGMENTATION_PROFILE_ID,
        )
        self.assertEqual(transform.center.shape, (9,))
        self.assertEqual(
            transform.profile_id,
            MOTION_DERIVED_AUGMENTATION_PROFILE_ID,
        )
        self.assertEqual(apply_motion_fold_imu_transform(values, transform).shape, values.shape)

    def test_scaler_iqr_fallback_is_population_sd_then_one(self) -> None:
        values = np.zeros((1, 8, 3200), dtype=np.float32)
        values[0, 2, -1] = 10.0
        transform = fit_motion_fold_imu_transform(
            values,
            ("train",),
            fitted_on_participant_ids=("train",),
            outer_train_participant_ids=("train",),
            outer_oof_participant_ids=(),
        )
        expected_sd = float(
            np.std(np.asarray(values[0, 2], dtype=np.float64), ddof=0)
        )
        self.assertAlmostEqual(float(transform.scale[0]), expected_sd)
        self.assertEqual(float(transform.scale[1]), 1.0)
        self.assertEqual(transform.schema_version, MOTION_SCALER_SCHEMA)


if __name__ == "__main__":
    unittest.main()
