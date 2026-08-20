"""Synthetic contracts for the isolated L0--L7 Legacy Bridge mechanics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ppg_frailty.legacy_bridge import (
    _raw_windows_from_matrix,
    _window_starts,
    resolve_legacy_bridge_profile,
)
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


class LegacyBridgeRuntimeTests(unittest.TestCase):
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
        participant_weights = outer_train_inverse_frequency_weights(dataset, 3)
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
