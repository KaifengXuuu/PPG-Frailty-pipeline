"""Synthetic contracts for the isolated L0--L7 Legacy Bridge mechanics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppg_frailty.legacy_bridge import (
    _raw_windows_from_matrix,
    _window_starts,
    build_legacy_bridge_raw_windows,
    resolve_legacy_bridge_profile,
)
from ppg_frailty.provenance import stable_payload_sha256
from ppg_frailty.training.datasets import RawWindowDataset, SampleIdentity
from ppg_frailty.training.legacy_bridge import (
    LegacyBridgeTrainer,
    outer_train_window_inverse_frequency_weights,
    sampling_diagnostics,
)
from ppg_frailty.training.trainer import outer_train_inverse_frequency_weights


def _dataset() -> RawWindowDataset:
    roster = (
        ("p0", 0, 5),
        ("p1", 0, 1),
        ("p2", 0, 1),
        ("p3", 1, 2),
        ("p4", 1, 1),
        ("p5", 2, 2),
    )
    identities: list[SampleIdentity] = []
    for participant, label, count in roster:
        for index in range(count):
            role = "B" if index == 0 else f"R{min(index, 4)}"
            identities.append(
                SampleIdentity(
                    participant_id=participant,
                    file_id=f"{participant}_{role}",
                    role=role,
                    label=label,
                    signal_route="DIRECT",
                    window_id=f"{participant}_{index}",
                )
            )
    values = np.arange(len(identities) * 8 * 16, dtype=np.float32).reshape(
        len(identities), 8, 16
    )
    return RawWindowDataset(values, identities)


def _star_profile(profile_id: str, **overrides: object):
    controls = {
        "ppg_preprocessing": "legacy_detrend_bandpass_0p2_8",
        "imu_preprocessing": "legacy_filtered_axes",
        "target_fs_hz": 64.0,
        "window_seconds": 15.0,
        "hop_seconds": 3.0,
        "historical_retained_fraction": 0.9,
        "max_windows_per_file": None,
        "allow_short_record_padding": True,
        "normalization": "per_window_all_eight",
        "sampler": "exhaustive_shuffle_without_replacement",
        "class_weighting": "outer_train_window_inverse_frequency",
        "optimizer": "adamw",
        "batch_size": 32,
        "fixed_epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "training_metric_aggregation_rule": "line_a_equal_files",
        "primary_report_aggregation_view": "window_balanced_to_participant",
    }
    controls.update(overrides)
    definition = {"profile_id": profile_id, "controls": controls}
    return resolve_legacy_bridge_profile(
        profile_id,
        protocol_design="centered_star_v1",
        profile_definition=definition,
        profile_definition_sha256=stable_payload_sha256(controls),
    )


class LegacyBridgeRuntimeTests(unittest.TestCase):
    def test_centered_star_factor_diffs_are_exactly_the_declared_star_arms(
        self,
    ) -> None:
        baseline = _star_profile("B0")
        profiles = {
            "B1": _star_profile("B1", target_fs_hz=400.0),
            "B2": _star_profile(
                "B2", window_seconds=5.0, hop_seconds=2.5
            ),
            "B3": _star_profile("B3", imu_preprocessing="calibrated_ekf_adyn"),
            "B4": _star_profile(
                "B4", normalization="ppg_window_imu_outer_train_fold"
            ),
            "B5": _star_profile("B5", sampler="balance_line_weighted_v2"),
            "B6": _star_profile("B6", optimizer="adam", batch_size=64),
            "B7": _star_profile(
                "B7", primary_report_aggregation_view="line_b_equal_role_families"
            ),
        }
        expected = {
            "B1": {"target_fs_hz"},
            "B2": {"window_seconds", "hop_seconds"},
            "B3": {"imu_preprocessing"},
            "B4": {"normalization"},
            "B5": {"sampler"},
            "B6": {"optimizer", "batch_size"},
            "B7": set(),
        }
        for profile_id, profile in profiles.items():
            changed = {
                key
                for key in baseline.training_identity_payload
                if baseline.training_identity_payload[key]
                != profile.training_identity_payload[key]
            }
            self.assertEqual(changed, expected[profile_id], profile_id)

        b1 = profiles["B1"]
        self.assertEqual(
            (b1.window_seconds, b1.hop_seconds),
            (15.0, 3.0),
        )
        b2 = profiles["B2"]
        self.assertEqual(b2.target_fs_hz, 64.0)
        self.assertEqual(b2.historical_retained_fraction, 0.9)
        self.assertIsNone(b2.max_windows_per_file)
        self.assertTrue(b2.resolved_allow_short_record_padding)
        full_starts = np.arange(0, 3_840 - 320 + 1, 160, dtype=np.int64)
        if int(full_starts[-1]) != 3_840 - 320:
            full_starts = np.append(full_starts, 3_840 - 320)
        self.assertEqual(
            len(_window_starts(3_840, fs_hz=64.0, profile=b2)),
            int(np.ceil(0.9 * len(full_starts))),
        )
        self.assertEqual(
            profiles["B5"].class_weighting,
            "outer_train_window_inverse_frequency",
        )

    def test_centered_star_profiles_are_field_driven_and_b0_b7_train_identical(
        self,
    ) -> None:
        b0 = _star_profile("B0")
        b3 = _star_profile("B3", imu_preprocessing="calibrated_ekf_adyn")
        b4 = _star_profile(
            "B4", normalization="ppg_window_imu_outer_train_fold"
        )
        b7 = _star_profile(
            "B7", primary_report_aggregation_view="line_b_equal_role_families"
        )

        self.assertTrue(b0.builds_windows_from_raw_record)
        self.assertFalse(b0.requires_calibrated_imu_views)
        self.assertTrue(b3.requires_calibrated_imu_views)
        self.assertEqual(b3.channel_schema[2:5], ("A_dyn_x", "A_dyn_y", "A_dyn_z"))
        self.assertTrue(b4.uses_fold_imu_transform)
        self.assertEqual(b4.resolved_imu_preprocessing, "legacy_filtered_axes")
        self.assertEqual(b0.training_identity_sha256, b7.training_identity_sha256)
        self.assertNotEqual(
            b0.profile_definition_sha256,
            b7.profile_definition_sha256,
        )
        self.assertEqual(
            b7.training_config().expected_aggregation_rule,
            "line_a_equal_files",
        )

    def test_centered_star_definition_is_complete_and_sha_bound(self) -> None:
        b0 = _star_profile("B0")
        controls = dict(b0.declared_controls)
        controls.pop("optimizer")
        with self.assertRaisesRegex(ValueError, "must be complete"):
            resolve_legacy_bridge_profile(
                "B0",
                protocol_design="centered_star_v1",
                profile_definition={"profile_id": "B0", "controls": controls},
                profile_definition_sha256=stable_payload_sha256(controls),
            )
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            resolve_legacy_bridge_profile(
                "B0",
                protocol_design="centered_star_v1",
                profile_definition={
                    "profile_id": "B0",
                    "controls": b0.declared_controls,
                },
                profile_definition_sha256="0" * 64,
            )

    def test_shared_builder_switches_only_declared_imu_and_normalization(self) -> None:
        sample_count = 6_400
        time_axis = np.arange(sample_count, dtype=np.float64) / 400.0
        record = {
            "fs_hz": 400.0,
            "ppg": np.column_stack(
                (
                    np.sin(2.0 * np.pi * time_axis),
                    np.cos(2.0 * np.pi * time_axis),
                )
            ),
            "acc": np.column_stack(
                (0.1 * time_axis, 0.2 * time_axis, 1.0 + 0.05 * time_axis)
            ),
            "gyro": np.column_stack(
                (0.01 * time_axis, 0.02 * time_axis, 0.03 * time_axis)
            ),
        }
        dynamic = np.column_stack(
            (2.0 + time_axis, 4.0 + 2.0 * time_axis, 8.0 + 3.0 * time_axis)
        )
        calibrated_views = type(
            "Views",
            (),
            {
                "imu_processed": {
                    "dynamic_acc_mps2": dynamic,
                    "gyro_rads": np.column_stack(
                        (time_axis, 2.0 * time_axis, 3.0 * time_axis)
                    ),
                    "imu_valid_mask": np.ones(sample_count, dtype=bool),
                },
                "validate": lambda self: None,
            },
        )()

        b0 = _star_profile("B0")
        b0_windows = build_legacy_bridge_raw_windows(record, b0)
        b3 = _star_profile("B3", imu_preprocessing="calibrated_ekf_adyn")
        b3_windows = build_legacy_bridge_raw_windows(
            record,
            b3,
            calibrated_views=calibrated_views,
        )
        self.assertEqual(b3_windows.values.shape[1:], (8, 960))
        self.assertTrue(
            np.allclose(np.median(b3_windows.values[0], axis=1), 0.0, atol=2e-3)
        )
        np.testing.assert_array_equal(
            b0_windows.values[:, :2],
            b3_windows.values[:, :2],
        )

        b4 = _star_profile(
            "B4", normalization="ppg_window_imu_outer_train_fold"
        )
        b4_windows = build_legacy_bridge_raw_windows(record, b4)
        np.testing.assert_array_equal(
            b0_windows.values[:, :2],
            b4_windows.values[:, :2],
        )
        self.assertTrue(
            np.allclose(np.median(b4_windows.values[0, :2], axis=1), 0.0, atol=2e-3)
        )
        self.assertGreater(abs(float(np.median(b4_windows.values[0, 4]))), 0.1)

    def test_profiles_are_the_frozen_cumulative_chain(self) -> None:
        profiles = [resolve_legacy_bridge_profile(f"L{level}") for level in range(8)]
        self.assertEqual([profile.target_fs_hz for profile in profiles], [64, 64, 400, 400, 400, 400, 400, 400])
        self.assertEqual(profiles[0].historical_retained_fraction, 0.9)
        self.assertEqual(profiles[1].max_windows_per_file, 128)
        self.assertTrue(all(profile.fixed_epochs == 10 for profile in profiles))
        self.assertEqual(
            [profile.sampler for profile in profiles],
            [
                "exhaustive_shuffle_without_replacement",
                "exhaustive_shuffle_without_replacement",
                "exhaustive_shuffle_without_replacement",
                "exhaustive_shuffle_without_replacement",
                "exhaustive_shuffle_without_replacement",
                "uniform_replacement",
                "balance_line_weighted_v2",
                "balance_line_weighted_v2",
            ],
        )
        self.assertEqual(profiles[6].class_weighting, "outer_train_inverse_frequency")
        self.assertEqual(profiles[7].optimizer, "adam")
        self.assertEqual(profiles[7].batch_size, 64)

    def test_historical_window_retention_and_l0_only_padding_are_exact(self) -> None:
        l0 = resolve_legacy_bridge_profile("L0")
        starts = _window_starts(3_840, fs_hz=64.0, profile=l0)
        self.assertEqual(len(starts), 15)
        self.assertEqual((int(starts[0]), int(starts[-1])), (0, 2_880))

        l1 = resolve_legacy_bridge_profile("L1")
        matrix = np.linspace(-1.0, 1.0, 100 * 8, dtype=np.float64).reshape(100, 8)
        with self.assertRaisesRegex(ValueError, "no valid windows"):
            _raw_windows_from_matrix(
                matrix,
                fs_hz=64.0,
                profile=l1,
                valid_mask=None,
                provenance={"fixture": True},
            )
        self.assertEqual(
            _window_starts(100, fs_hz=64.0, profile=l1).tolist(),
            [],
        )

    def test_window_and_participant_class_weight_populations_are_distinct(self) -> None:
        dataset = _dataset()
        window_weights = outer_train_window_inverse_frequency_weights(dataset)
        participant_weights = outer_train_inverse_frequency_weights(
            dataset, 3, class_count_basis="participant"
        )
        np.testing.assert_allclose(
            window_weights,
            np.asarray([12 / 21, 12 / 9, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            participant_weights,
            np.asarray([2 / 3, 1.0, 2.0], dtype=np.float32),
        )
        self.assertFalse(np.array_equal(window_weights, participant_weights))

    def test_three_samplers_and_eleven_diagnostics_are_deterministic(self) -> None:
        dataset = _dataset()
        expected_keys = {
            "dataset_row_count",
            "draw_count",
            "unique_row_draw_count",
            "duplicate_draw_fraction",
            "never_drawn_row_fraction",
            "draw_counts_by_participant",
            "draw_counts_by_class",
            "draw_counts_by_B_R_family",
            "draw_counts_by_file",
            "class_weight_vector",
            "sampler_identity",
        }
        for level in (0, 5, 6):
            profile = resolve_legacy_bridge_profile(f"L{level}")
            trainer = LegacyBridgeTrainer(profile.training_config())
            first = trainer._draw_indices(dataset, epoch_seed=1_000_042)
            second = trainer._draw_indices(dataset, epoch_seed=1_000_042)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(len(first), len(dataset))
            diagnostics = sampling_diagnostics(
                dataset,
                first,
                sampler_identity=profile.sampler,
                class_weight_vector=trainer._class_weight_vector(dataset),
            )
            self.assertEqual(set(diagnostics), expected_keys)
            self.assertEqual(diagnostics["draw_count"], len(dataset))
            if level == 0:
                self.assertEqual(diagnostics["unique_row_draw_count"], len(dataset))
                self.assertEqual(diagnostics["duplicate_draw_fraction"], 0.0)
            else:
                self.assertLess(diagnostics["unique_row_draw_count"], len(dataset))

    def test_optimizer_switch_is_confined_to_l7(self) -> None:
        member = torch.nn.Linear(4, 3)
        l6 = LegacyBridgeTrainer(resolve_legacy_bridge_profile("L6").training_config())
        l7 = LegacyBridgeTrainer(resolve_legacy_bridge_profile("L7").training_config())
        self.assertIsInstance(l6._optimizer(member), torch.optim.AdamW)
        self.assertIsInstance(l7._optimizer(member), torch.optim.Adam)


if __name__ == "__main__":
    unittest.main()
