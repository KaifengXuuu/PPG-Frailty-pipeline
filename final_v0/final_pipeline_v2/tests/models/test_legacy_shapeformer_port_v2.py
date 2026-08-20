"""Numerical and fold-isolation tests for the legacy ShapeFormer parallel port."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from ppg_frailty.models.factory import (
    FRAILTY_RAW_CHANNEL_SCHEMA,
    ModelInputSpec,
    materialize_architecture_parameters,
    prepare_model_factory,
)
from ppg_frailty.models.shapeformer_legacy import (
    LEGACY_DISCOVERY_BALANCE,
    LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
    LegacyEffectSizeShapeFormer,
    discover_legacy_effect_size_shapelets,
)
from ppg_frailty.module_registry import validate_model_config
from ppg_frailty.pipeline import run_model_comparison
from ppg_frailty.training.datasets import (
    FileBagDataset,
    RawWindowDataset,
    SampleIdentity,
)
from ppg_frailty.training.trainer import FrozenOuterSplit, FullCohortRefitScope


def _load_historical_module() -> object:
    source = Path(__file__).resolve().parents[4] / "shapeformer_port.py"
    spec = importlib.util.spec_from_file_location(
        "_v2_test_historical_shapeformer_port", source
    )
    if spec is None or spec.loader is None:  # pragma: no cover - path invariant
        raise RuntimeError(f"cannot load historical ShapeFormer source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_fixture() -> tuple[
    RawWindowDataset, FrozenOuterSplit, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(123)
    labels = np.repeat(np.arange(3, dtype=np.int64), 4)
    values = (
        rng.normal(size=(12, 8, 64))
        + labels[:, None, None] * 0.20
    ).astype(np.float32)
    identities = tuple(
        SampleIdentity(
            participant_id=f"p{index // 4}",
            file_id=f"file_{index:02d}",
            role="B",
            label=int(labels[index]),
            signal_route="direct",
            window_id=f"window_{index:02d}",
        )
        for index in range(12)
    )
    dataset = RawWindowDataset(values, identities)
    split = FrozenOuterSplit(
        repeat=1,
        fold=2,
        seed=9,
        train_participant_ids=("p0", "p1", "p2"),
        oof_participant_ids=("p3",),
        registry_hash="registry",
        fold_hash="fold",
    ).bind_training_dataset(dataset)
    return dataset, split, values, labels


def _encoder_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_id": "shapeformer_legacy_effect_size_port",
        "seed": 9,
        "input_fs_hz": 100.0,
        "sequence_length_samples": 64,
        "shapelet_length_samples": 16,
        "discovery_stride_samples": 8,
        "shapelets_per_class": 2,
        "max_discovery_windows": 6,
        "candidates_per_class_channel": 2,
        "local_kernel_width_samples": 8,
        "local_embedding_channels": 8,
        "shape_embedding_channels": 8,
        "attention_feedforward_channels": 12,
        "attention_heads": 2,
        "dropout": 0.0,
        "shapelet_search_window_samples": 8,
        "complexity_norm": 1000.0,
        "max_complexity_ratio": 3.0,
    }
    config.update(overrides)
    return config


class LegacyShapeFormerPortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, self.split, self.values, self.labels = _raw_fixture()
        self.raw_spec = ModelInputSpec(
            "raw",
            n_channels=8,
            n_classes=3,
            channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
        )

    def _discover(self, **overrides: object):
        config = _encoder_config(**overrides)
        return discover_legacy_effect_size_shapelets(
            self.values,
            self.labels,
            self.dataset.participant_ids,
            tuple(identity.file_id for identity in self.dataset.identities),
            tuple(identity.window_id for identity in self.dataset.identities),
            discovery_method=LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
            input_fs_hz=float(config["input_fs_hz"]),
            sequence_length_samples=int(config["sequence_length_samples"]),
            shapelet_length_samples=int(config["shapelet_length_samples"]),
            discovery_stride_samples=int(config["discovery_stride_samples"]),
            shapelets_per_class=int(config["shapelets_per_class"]),
            max_discovery_windows=int(config["max_discovery_windows"]),
            candidates_per_class_channel=int(
                config["candidates_per_class_channel"]
            ),
            outer_repeat_index=self.split.repeat,
            outer_fold_index=self.split.fold,
            seed=int(config["seed"]),
        )

    def test_discovery_is_numerically_identical_to_historical_effect_size(self) -> None:
        historical = _load_historical_module()
        expected = historical.discover_shapelets(
            self.values,
            self.labels,
            n_shapelets_per_class=2,
            shapelet_len=16,
            stride=8,
            max_discovery_windows=6,
            candidates_per_class_channel=2,
            seed=9,
        )
        observed = self._discover()
        np.testing.assert_array_equal(
            observed.discovery_indices, expected.discovery_indices
        )
        np.testing.assert_allclose(observed.info, expected.info, atol=0.0)
        for actual, reference in zip(observed.values, expected.shapelets):
            np.testing.assert_allclose(actual, reference, atol=0.0)
        self.assertEqual(observed.discovery_indices.size, 6)
        self.assertEqual(observed.enumerated_candidate_count, 3 * 8 * 7)
        self.assertEqual(observed.retained_candidate_count, 3 * 8 * 2)

    def test_downstream_features_are_numerically_identical_to_legacy_port(self) -> None:
        historical = _load_historical_module()
        bank = self._discover()
        old_model = historical.PortedShapeFormer(
            8,
            64,
            3,
            bank.info,
            bank.values,
            len_w=64,
            local_embed_dim=8,
            shape_embed_dim=8,
            dim_ff=12,
            num_heads=2,
            dropout=0.0,
            shapelet_search_window=8,
        ).eval()
        new_model = LegacyEffectSizeShapeFormer(
            8,
            3,
            bank,
            sequence_length_samples=64,
            local_kernel_width_samples=8,
            local_embedding_channels=8,
            shape_embedding_channels=8,
            attention_feedforward_channels=12,
            attention_heads=2,
            dropout=0.0,
            shapelet_search_window_samples=8,
            input_fs_hz=100.0,
        ).eval()
        with torch.no_grad():
            for model in (old_model, new_model):
                for parameter in model.parameters():
                    parameter.fill_(0.013)
            inputs = torch.from_numpy(self.values[:3])
            expected = old_model.forward_features(inputs)
            observed = new_model.forward_features(inputs)
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=0.0)

    def test_raw_prepare_forward_hash_and_outer_fold_isolation(self) -> None:
        prepared = prepare_model_factory(
            _encoder_config(), self.raw_spec, self.dataset, self.split
        )
        self.assertEqual(
            prepared.provenance["discovery_method"],
            LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
        )
        self.assertEqual(
            prepared.provenance["discovery_balance"],
            LEGACY_DISCOVERY_BALANCE,
        )
        self.assertEqual(
            tuple(
                prepared.resolved_model_config["shapelets"].fitted_participant_ids
            ),
            ("p0", "p1", "p2"),
        )
        self.assertNotIn(
            "p3", prepared.resolved_model_config["shapelets"].fitted_participant_ids
        )
        model = prepared.factory().eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(self.values[:2]),
                torch.ones((2, 64), dtype=torch.bool),
            )
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertEqual(model.local_temporal[0].kernel_size, (1, 8))
        self.assertEqual(model.local_feedforward[0].out_features, 12)

        architecture = materialize_architecture_parameters(
            _encoder_config(candidates_per_class_channel=1), self.raw_spec
        )
        changed = materialize_architecture_parameters(
            _encoder_config(candidates_per_class_channel=2), self.raw_spec
        )
        self.assertNotEqual(architecture, changed)
        self.assertEqual(changed["candidates_per_class_channel"], 2)

        contaminated_values = np.concatenate(
            (self.values, self.values[:1]), axis=0
        )
        contaminated_identities = self.dataset.identities + (
            SampleIdentity(
                "p3", "oof_file", "B", 0, "direct", window_id="oof_window"
            ),
        )
        contaminated = RawWindowDataset(
            contaminated_values, contaminated_identities
        )
        with self.assertRaisesRegex(ValueError, "outer-train"):
            prepare_model_factory(
                _encoder_config(), self.raw_spec, contaminated, self.split
            )

    def test_generic_file_bag_fusion_and_no_historical_ui_parameters(self) -> None:
        bags = tuple(
            self.values[index : index + 4] for index in range(0, 12, 4)
        )
        masks = tuple(np.ones((4, 64), dtype=bool) for _ in bags)
        identities = tuple(
            SampleIdentity(
                f"p{index}", f"bag_{index}", "B", index, "direct"
            )
            for index in range(3)
        )
        file_bags = FileBagDataset(
            bags,
            np.arange(15, dtype=np.float32).reshape(3, 5),
            identities,
            masks,
        )
        split = FrozenOuterSplit(
            1,
            2,
            9,
            ("p0", "p1", "p2"),
            ("p3",),
            "registry",
            "fold",
        ).bind_training_dataset(file_bags)
        spec = ModelInputSpec(
            "fusion",
            n_channels=8,
            n_classes=3,
            n_file_features=5,
            channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
        )
        nested = _encoder_config()
        nested.pop("seed")
        prepared = prepare_model_factory(
            {
                "model_id": "file_bag_fusion",
                "seed": 9,
                "signal_encoder": nested,
                "feature_hidden_dim": 4,
                "fusion_hidden_dim": 6,
                "pooling": "mean",
                "dropout": 0.0,
            },
            spec,
            file_bags,
            split,
        )
        self.assertFalse(prepared.provenance["file_features_used_for_discovery"])
        model = prepared.factory().eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(np.stack(bags)),
                torch.ones((3, 4), dtype=torch.bool),
                torch.from_numpy(file_bags.file_features),
                torch.from_numpy(np.stack(masks)),
            )
        self.assertEqual(tuple(output.shape), (3, 3))
        self.assertEqual(
            model.signal_encoder.fitted_participant_ids,
            ("p0", "p1", "p2"),
        )
        self.assertEqual(
            model.resolved_architecture_parameters["signal_encoder"]["model_id"],
            "shapeformer_legacy_effect_size_port",
        )

        section = {
            "model_id": "ShapeFormerLegacyEffectSizePort",
            "input_channels": 8,
            "input_channels_resolution": "canonical_frailty_raw_8",
            "input_channel_order": list(FRAILTY_RAW_CHANNEL_SCHEMA),
            "n_classes": 3,
            "seed_policy": "outer_repeat",
            "processes": 2,
        }
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_model_config(section, "raw")
        with self.assertRaisesRegex(ValueError, "unknown legacy"):
            prepare_model_factory(
                _encoder_config(verbose=True),
                self.raw_spec,
                self.dataset,
                self.split,
            )

    def test_explicit_model_comparison_smoke_uses_the_legacy_route(self) -> None:
        report = run_model_comparison(
            ("ShapeFormerLegacyEffectSizePort",), seed=7
        )
        row = report["results"][0]
        self.assertEqual(row["status"], "passed")
        self.assertEqual(
            row["machine_model_id"],
            "shapeformer_legacy_effect_size_port",
        )
        self.assertEqual(
            row["discovery_method"],
            LEGACY_EFFECT_SIZE_DISCOVERY_METHOD,
        )

    def test_final_refit_repeats_discovery_on_exactly_all_29(self) -> None:
        rng = np.random.default_rng(55)
        participant_ids = tuple(f"p{index:02d}" for index in range(29))
        labels = np.arange(29, dtype=np.int64) % 3
        values = (
            rng.normal(size=(29, 8, 32))
            + labels[:, None, None] * 0.1
        ).astype(np.float32)
        identities = tuple(
            SampleIdentity(
                participant_id,
                f"{participant_id}_file",
                "B",
                int(labels[index]),
                "direct",
                window_id=f"{participant_id}_window",
            )
            for index, participant_id in enumerate(participant_ids)
        )
        dataset = RawWindowDataset(values, identities)
        scope = FullCohortRefitScope(
            participant_ids,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ).bind_training_dataset(dataset)
        prepared = prepare_model_factory(
            _encoder_config(
                input_fs_hz=64.0,
                sequence_length_samples=32,
                shapelet_length_samples=8,
                discovery_stride_samples=4,
                shapelets_per_class=1,
                max_discovery_windows=12,
                candidates_per_class_channel=2,
                local_kernel_width_samples=5,
                shapelet_search_window_samples=4,
            ),
            self.raw_spec,
            dataset,
            scope,
        )
        bank = prepared.resolved_model_config["shapelets"]
        self.assertEqual(bank.fitted_participant_ids, participant_ids)
        self.assertEqual(
            prepared.provenance["outer_train_dataset_hash"],
            scope.train_dataset_hash,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
