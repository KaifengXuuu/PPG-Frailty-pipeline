"""Runtime and config-hash tests for optimizer and sampler parameterization."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from ppg_frailty.config import (
    canonical_json_bytes,
    load_config,
    validate_config_payload,
)
from ppg_frailty.training import (
    FrozenOuterSplit,
    RawWindowDataset,
    SampleIdentity,
    TrainingConfig,
    UnifiedTrainer,
    configured_row_sampling_weights,
    subject_epoch_sampling_indices,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = PIPELINE_ROOT / "configs" / "reference_static_role_aware_v2.yaml"


def _identity(participant: str, label: int, row: int) -> SampleIdentity:
    return SampleIdentity(
        participant_id=participant,
        file_id=f"{participant}_B",
        role="B",
        label=label,
        signal_route="direct",
        window_id=f"{participant}_w{row}",
    )


def _unequal_subject_dataset() -> RawWindowDataset:
    identities = (
        _identity("P0", 0, 0),
        _identity("P0", 0, 1),
        _identity("P0", 0, 2),
        _identity("P1", 0, 0),
        _identity("P2", 1, 0),
        _identity("P2", 1, 1),
    )
    values = np.arange(6 * 2 * 8, dtype=np.float32).reshape(6, 2, 8)
    return RawWindowDataset(values, identities)


class OptimizerParameterizationTests(unittest.TestCase):
    def test_defaults_and_partial_mapping_are_complete_and_hash_bound(self) -> None:
        raw = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
        raw["training"]["optimizer_parameters"] = {"eps": 1e-7}
        resolved = validate_config_payload(raw)
        self.assertEqual(
            resolved["training"]["optimizer_parameters"],
            {
                "betas": [0.9, 0.999],
                "eps": 1e-7,
                "amsgrad": False,
                "maximize": False,
            },
        )
        base = load_config(REFERENCE_CONFIG)
        changed = base.to_dict()
        changed["training"]["optimizer"] = "sgd"
        changed["training"]["optimizer_parameters"] = {"momentum": 0.8}
        changed = validate_config_payload(changed)
        changed_hash = hashlib.sha256(canonical_json_bytes(changed)).hexdigest()
        self.assertNotEqual(base.sha256, changed_hash)
        self.assertEqual(
            changed["training"]["optimizer_parameters"],
            {
                "momentum": 0.8,
                "dampening": 0.0,
                "nesterov": False,
                "maximize": False,
            },
        )

        raw_sgd = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
        raw_sgd["training"]["optimizer"] = "sgd"
        resolved_sgd = validate_config_payload(raw_sgd)
        self.assertEqual(
            resolved_sgd["training"]["optimizer_parameters"],
            {
                "momentum": 0.0,
                "dampening": 0.0,
                "nesterov": False,
                "maximize": False,
            },
        )

    def test_effective_json_round_trip_preserves_scientific_numbers(self) -> None:
        original = load_config(REFERENCE_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effective_config.yaml"
            path.write_text(json.dumps(original.payload), encoding="utf-8")
            round_trip = load_config(path)
        self.assertEqual(round_trip.sha256, original.sha256)
        self.assertIsInstance(
            round_trip.payload["training"]["optimizer_parameters"]["eps"],
            float,
        )

    def test_all_registered_optimizers_apply_declared_parameters(self) -> None:
        cases = (
            (
                "adam",
                {"betas": [0.8, 0.95], "eps": 1e-7, "amsgrad": True},
                torch.optim.Adam,
                {"betas": (0.8, 0.95), "eps": 1e-7, "amsgrad": True},
            ),
            (
                "adamw",
                {"betas": [0.7, 0.9], "maximize": True},
                torch.optim.AdamW,
                {"betas": (0.7, 0.9), "maximize": True},
            ),
            (
                "sgd",
                {"momentum": 0.85, "nesterov": True},
                torch.optim.SGD,
                {"momentum": 0.85, "nesterov": True},
            ),
            (
                "rmsprop",
                {"alpha": 0.9, "eps": 1e-6, "momentum": 0.2, "centered": True},
                torch.optim.RMSprop,
                {"alpha": 0.9, "eps": 1e-6, "momentum": 0.2, "centered": True},
            ),
        )
        for name, parameters, expected_type, expected_values in cases:
            with self.subTest(optimizer=name):
                config = TrainingConfig(
                    optimizer=name,
                    optimizer_parameters=parameters,
                    learning_rate=0.0123,
                    weight_decay=0.0045,
                )
                optimizer = UnifiedTrainer(config)._optimizer(nn.Linear(2, 3))
                self.assertIsInstance(optimizer, expected_type)
                group = optimizer.param_groups[0]
                self.assertEqual(group["lr"], 0.0123)
                self.assertEqual(group["weight_decay"], 0.0045)
                for key, expected in expected_values.items():
                    self.assertEqual(group[key], expected)

    def test_optimizer_specific_keys_and_ranges_fail_closed(self) -> None:
        invalid = (
            {"optimizer": "sgd", "optimizer_parameters": {"betas": [0.9, 0.99]}},
            {"optimizer": "adam", "optimizer_parameters": {"eps": 0.0}},
            {"optimizer": "adam", "optimizer_parameters": {"betas": [1.0, 0.9]}},
            {"optimizer": "sgd", "optimizer_parameters": {"momentum": -0.1}},
            {"optimizer": "sgd", "optimizer_parameters": {"nesterov": True}},
            {"optimizer": "rmsprop", "optimizer_parameters": {"alpha": 1.0}},
            {"optimizer": "rmsprop", "optimizer_parameters": {"centered": 1}},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


class SamplerParameterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _unequal_subject_dataset()

    def test_replacement_sampler_honours_exact_epoch_budget(self) -> None:
        config = TrainingConfig(
            sampler="uniform_replacement",
            samples_per_epoch=11,
        )
        loader = UnifiedTrainer(config)._loader(self.dataset, shuffle=True)
        self.assertIsInstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
        self.assertEqual(len(loader.sampler), 11)
        self.assertEqual(len(list(loader.sampler)), 11)

        split = FrozenOuterSplit(
            repeat=0,
            fold=0,
            seed=42,
            train_participant_ids=("P0", "P1", "P2"),
            oof_participant_ids=("P3",),
            registry_hash="registry",
            fold_hash="fold",
        )
        with self.assertRaisesRegex(
            ValueError,
            "execution_backend=estimator does not support.*samples_per_epoch",
        ):
            UnifiedTrainer(config).fit_estimator(object(), self.dataset, split)

    def test_subject_quota_matches_migrated_legacy_semantics(self) -> None:
        config = TrainingConfig(
            sampler="subject_balanced",
            participant_window_quota=2,
        )
        trainer = UnifiedTrainer(config)
        loader = trainer._loader(self.dataset, shuffle=True)
        indices = tuple(int(value) for value in loader.sampler)
        participant_ids = np.asarray(self.dataset.participant_ids, dtype=object)
        counts = {
            participant: int(np.sum(participant_ids[list(indices)] == participant))
            for participant in ("P0", "P1", "P2")
        }
        self.assertEqual(counts, {"P0": 2, "P1": 2, "P2": 2})
        np.testing.assert_allclose(
            configured_row_sampling_weights(
                self.dataset,
                sampler="subject_balanced",
                training_balance="equal_role_families",
                allowed_role_families=("B", "R"),
                participant_window_quota=2,
            ),
            (1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 6, 1 / 6),
        )

    def test_class_subject_sampler_balances_subject_slots_by_class(self) -> None:
        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)
        first = subject_epoch_sampling_indices(
            self.dataset,
            sampler="class_subject_balanced",
            participant_window_quota=1,
            generator=first_generator,
        )
        second = subject_epoch_sampling_indices(
            self.dataset,
            sampler="class_subject_balanced",
            participant_window_quota=1,
            generator=second_generator,
        )
        self.assertEqual(first, second)
        labels = np.asarray([identity.label for identity in self.dataset.identities])
        np.testing.assert_array_equal(np.bincount(labels[list(first)]), (2, 2))
        np.testing.assert_allclose(
            configured_row_sampling_weights(
                self.dataset,
                sampler="class_subject_balanced",
                training_balance="equal_role_families",
                allowed_role_families=("B", "R"),
                participant_window_quota=1,
            ),
            (1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 4, 1 / 4),
        )

    def test_sampler_parameters_cannot_be_silently_ignored(self) -> None:
        invalid = (
            {"sampler": "exhaustive_shuffle_without_replacement", "samples_per_epoch": 5},
            {"sampler": "subject_balanced", "samples_per_epoch": 5},
            {"sampler": "uniform_replacement", "participant_window_quota": 2},
            {"sampler": "subject_balanced", "participant_window_quota": "101%"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
