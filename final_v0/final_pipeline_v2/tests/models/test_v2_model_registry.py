"""V2 candidate registry and fold-local ShapeFormer preparation smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppg_frailty.models import (
    FIVE_MEMBER_ENSEMBLE_COMPARISONS,
    NONENSEMBLE_MODEL_CANDIDATES,
    ModelInputSpec,
    create_model,
    materialize_architecture_parameters,
    prepare_model_factory,
)
from ppg_frailty.models.factory import FRAILTY_RAW_CHANNEL_SCHEMA
from ppg_frailty.models.shapeformer import ExperimentalShapeFormer
from ppg_frailty.training.datasets import (
    FileBagDataset,
    RawWindowDataset,
    SampleIdentity,
)
from ppg_frailty.training.trainer import FrozenOuterSplit, FullCohortRefitScope


EXPECTED_IDS = (
    "compact_cnn",
    "inception_full",
    "inception_small",
    "inception_matrix",
    "rocket_numpy",
    "minirocket_ablation",
    "logistic_regression",
    "rbf_svm",
    "extra_trees",
    "shapeformer_channel_specific_osd",
    "shapeformer_effect_size_fixed_v1",
    "fusion_compact",
    "fusion_inception",
)


def _raw_outer_train(length: int = 64) -> tuple[RawWindowDataset, FrozenOuterSplit]:
    axis = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float32)
    phases = (0.0, 0.2, np.pi, np.pi + 0.2, 0.7, 0.9)
    values = np.stack([
        np.stack([
            np.sin(axis + phase + channel * 0.05)
            for channel in range(8)
        ])
        for phase in phases
    ]).astype(np.float32)
    identities = (
        SampleIdentity("p0", "p0_b1", "B", 0, "direct", window_id="w0"),
        SampleIdentity("p0", "p0_b2", "B2", 0, "direct", window_id="w1"),
        SampleIdentity("p1", "p1_r1", "R1", 1, "direct", window_id="w0"),
        SampleIdentity("p1", "p1_r2", "R2", 1, "direct", window_id="w1"),
        SampleIdentity("p2", "p2_s1", "S1", 2, "direct", window_id="w0"),
        SampleIdentity("p2", "p2_s2", "S2", 2, "direct", window_id="w1"),
    )
    dataset = RawWindowDataset(values, identities)
    split = FrozenOuterSplit(
        repeat=2,
        fold=3,
        seed=20042,
        train_participant_ids=("p0", "p1", "p2"),
        oof_participant_ids=("p3",),
        registry_hash="registry",
        fold_hash="fold",
    )
    return dataset, split


def _fusion_outer_train(length: int = 64) -> tuple[FileBagDataset, FrozenOuterSplit]:
    raw, split = _raw_outer_train(length)
    bags = tuple(
        raw.values[index : index + 2]
        for index in range(0, len(raw), 2)
    )
    masks = tuple(
        raw.sample_mask[index : index + 2]
        for index in range(0, len(raw), 2)
    )
    identities = tuple(
        SampleIdentity(
            participant_id=f"p{index}",
            file_id=f"p{index}_fusion_file",
            role="B",
            label=index,
            signal_route="direct",
        )
        for index in range(3)
    )
    dataset = FileBagDataset(
        bags,
        np.arange(15, dtype=np.float32).reshape(3, 5),
        identities,
        masks,
    )
    return dataset, split.bind_training_dataset(dataset)


def _shapeformer_config(model_id: str) -> dict[str, object]:
    method = (
        "channel_specific_osd"
        if model_id == "shapeformer_channel_specific_osd"
        else "effect_size_fixed_v1"
    )
    result: dict[str, object] = {
        "model_id": model_id,
        "discovery_method": method,
        "input_fs_hz": 100.0,
        "dropout": 0.0,
        "seed": 42,
    }
    if model_id == "shapeformer_channel_specific_osd":
        result.update(
            {
                "num_pip_ratio": 0.20,
                "shapelets_per_class": 3,
                "max_discovery_windows": 180,
                "discovery_balance": "participant_file_balanced",
                "position_search_neighbourhood_samples": 128,
                "pip_rounding_rule": "floor_ratio_minimum_5_capped_at_actual_T",
                "pip_selection_rule": (
                    "upstream_zscored_time_index_perpendicular_distance_first_max"
                ),
                "candidate_generation_rule": (
                    "insertion_stage_three_consecutive_pips_half_open"
                ),
                "candidate_enumeration_rule": (
                    "upstream_class_channel_source_sample_insertion_order"
                ),
                "candidate_ranking_rule": (
                    "upstream_numpy_default_argsort_then_reverse"
                ),
                "selected_bank_order_rule": (
                    "upstream_per_class_start_sample_default_argsort"
                ),
                "discovery_position_search_boundary_rule": (
                    "upstream_pcs_start_minus_w_plus_1_end_plus_w_half_open"
                ),
                "information_gain_split_rule": (
                    "upstream_positive_recall_grid_0p2"
                ),
                "sequence_length_samples": 64,
                "local_kernel_width_samples": 8,
                "local_embedding_channels": 8,
                "shape_embedding_channels": 8,
                "attention_feedforward_channels": 16,
                "attention_heads": 2,
                "attention_query_chunk_size": 16,
                "distance_position_chunk_size": 16,
                "complexity_norm": 1000.0,
                "max_complexity_ratio": 3.0,
            }
        )
    else:
        result.update(
            {
                "shapelet_length_samples": 128,
                "shapelets_per_class": 3,
                "discovery_stride_samples": 64,
                "max_candidates_per_class": 8,
                "hidden_channels": 8,
                "patch_size_samples": 4,
                "attention_heads": 2,
                "attention_layers": 1,
                "distance_position_chunk_size": 16,
            }
        )
    return result


def test_file_bag_split_binding_includes_temporal_sample_masks() -> None:
    dataset, split = _fusion_outer_train(64)
    changed_masks = [mask.copy() for mask in dataset.sample_masks]
    changed_masks[0][0, -1] = False
    changed = FileBagDataset(
        dataset.window_bags,
        dataset.file_features,
        dataset.identities,
        changed_masks,
    )
    with pytest.raises(ValueError, match="identity/content hash"):
        split.assert_training_dataset(changed, exact=True)


def test_exact_13_candidate_registry_and_separate_ensemble() -> None:
    assert tuple(item.machine_id for item in NONENSEMBLE_MODEL_CANDIDATES) == EXPECTED_IDS
    assert len(NONENSEMBLE_MODEL_CANDIDATES) == 13
    assert sum(item.registry_role == "reference" for item in NONENSEMBLE_MODEL_CANDIDATES) == 11
    assert sum(item.registry_role == "ablation" for item in NONENSEMBLE_MODEL_CANDIDATES) == 2
    by_id = {item.machine_id: item for item in NONENSEMBLE_MODEL_CANDIDATES}
    assert by_id["shapeformer_channel_specific_osd"].registry_role == "reference"
    assert by_id["shapeformer_effect_size_fixed_v1"].registry_role == "ablation"
    assert all(item.machine_id not in by_id for item in FIVE_MEMBER_ENSEMBLE_COMPARISONS)
    assert all(item.registry_role == "comparison" for item in FIVE_MEMBER_ENSEMBLE_COMPARISONS)


@pytest.mark.parametrize("model_id", ["logistic_regression", "rocket_numpy"])
def test_classical_and_rocket_epoch_requests_fail_closed(model_id: str) -> None:
    mode = "feature_vector" if model_id == "logistic_regression" else "feature_matrix"
    with pytest.raises(ValueError, match="do not accept epoch settings"):
        create_model(
            {"model_id": model_id, "epoch_profile": "ablation_7"},
            ModelInputSpec(mode, n_channels=2, feature_names=("a", "b")),
        )


@pytest.mark.parametrize(
    ("model_id", "registry_role"),
    [
        ("shapeformer_channel_specific_osd", "reference"),
        ("shapeformer_effect_size_fixed_v1", "ablation"),
    ],
)
def test_fold_local_shapeformer_preparation_is_reproducible(
    model_id: str, registry_role: str
) -> None:
    dataset, split = _raw_outer_train(64 if registry_role == "reference" else 256)
    spec = ModelInputSpec(
        "raw", n_channels=8, n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    config = _shapeformer_config(model_id)
    config["architecture_parameters"] = materialize_architecture_parameters(config, spec)
    prepared = prepare_model_factory(
        config,
        spec,
        dataset,
        split,
    )
    assert prepared.provenance["registry_role"] == registry_role
    assert prepared.provenance["outer_repeat_index"] == 2
    assert prepared.provenance["outer_fold_index"] == 3
    assert prepared.provenance["fallback_used"] is False
    if registry_role == "reference":
        architecture = prepared.resolved_model_config["architecture_parameters"]
        assert architecture["attention_probability_dropout_applied"] is False
        assert (
            architecture["shape_position_embedding_width_policy"]
            == "upstream_observed_max_plus_1_per_axis"
        )
        assert architecture["shape_channel_position_width"] >= 1
        assert architecture["shape_start_position_width"] >= 1
        assert architecture["shape_end_position_width"] >= 1
        assert (
            prepared.provenance["candidate_ranking_rule"]
            == "upstream_numpy_default_argsort_then_reverse"
        )
    first = prepared.factory()
    second = prepared.factory()
    first_state = next(first.parameters()).detach()
    second_state = next(second.parameters()).detach()
    assert torch.equal(first_state, second_state)
    first.eval()
    with torch.no_grad():
        output = first(torch.from_numpy(dataset.values[:1]), torch.from_numpy(dataset.sample_mask[:1]))
    assert tuple(output.shape) == (1, 3)


@pytest.mark.parametrize(
    "signal_model_id",
    ["shapeformer_channel_specific_osd", "shapeformer_effect_size_fixed_v1"],
)
def test_file_bag_fusion_composes_fold_local_shapeformer_and_file_features(
    signal_model_id: str,
) -> None:
    """Both ShapeFormer routes discover on outer-train bags and fuse once/file."""

    dataset, split = _fusion_outer_train(64)
    spec = ModelInputSpec(
        "fusion",
        n_channels=8,
        n_file_features=5,
        n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    signal = _shapeformer_config(signal_model_id)
    if signal_model_id == "shapeformer_channel_specific_osd":
        signal.update(
            {
                "shapelets_per_class": 1,
                "max_discovery_windows": 6,
                "position_search_neighbourhood_samples": 7,
                "local_embedding_channels": 8,
                "shape_embedding_channels": 8,
                "attention_feedforward_channels": 16,
                "attention_heads": 2,
                "attention_query_chunk_size": 8,
                "distance_position_chunk_size": 8,
            }
        )
    else:
        signal.update(
            {
                "shapelet_length_samples": 16,
                "shapelets_per_class": 1,
                "discovery_stride_samples": 8,
                "max_candidates_per_class": 4,
                "attention_feedforward_channels": 18,
            }
        )
    signal.pop("seed")
    config = {
        "model_id": "file_bag_fusion",
        "seed": 42,
        "signal_encoder": signal,
        "feature_hidden_dim": 4,
        "fusion_hidden_dim": 6,
        "pooling": "attention",
        "dropout": 0.0,
    }
    prepared = prepare_model_factory(config, spec, dataset, split)
    assert prepared.provenance["fold_local_preparation"] == (
        "signal_encoder_shapelet_discovery"
    )
    assert prepared.provenance["signal_encoder_model_id"] == signal_model_id
    assert prepared.provenance["file_features_used_for_discovery"] is False
    assert prepared.provenance["outer_train_dataset_hash"] == split.train_dataset_hash
    model = prepared.factory().eval()
    assert model.signal_encoder.fitted_participant_ids == ("p0", "p1", "p2")
    assert "p3" not in model.signal_encoder.fitted_participant_ids
    assert model.feature_encoder[0].in_features == 5
    with torch.no_grad():
        output = model(
            torch.from_numpy(np.stack(dataset.window_bags)),
            torch.ones((3, 2), dtype=torch.bool),
            torch.from_numpy(dataset.file_features),
            torch.from_numpy(np.stack(dataset.sample_masks)),
        )
    assert tuple(output.shape) == (3, 3)
    assert model.resolved_architecture_parameters["signal_encoder"][
        "model_id"
    ] == signal_model_id
    if signal_model_id == "shapeformer_effect_size_fixed_v1":
        assert model.signal_encoder.attention_feedforward_channels == 18
        assert model.signal_encoder.patch_attention.layers[0].linear1.out_features == 18


def test_file_bag_shapeformer_final_refit_uses_the_same_verified_preparation() -> None:
    """Final refit repeats discovery on exactly the bound all-participant bags."""

    length = 32
    axis = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float32)
    participant_ids = tuple(f"p{index:02d}" for index in range(29))
    identities = tuple(
        SampleIdentity(
            participant_id=participant_id,
            file_id=f"{participant_id}_file",
            role="B",
            label=index % 3,
            signal_route="direct",
        )
        for index, participant_id in enumerate(participant_ids)
    )
    bags = tuple(
        np.stack(
            [
                np.stack(
                    [
                        np.sin(axis + index * 0.07 + channel * 0.03)
                        for channel in range(8)
                    ]
                ).astype(np.float32)
            ]
        )
        for index in range(29)
    )
    dataset = FileBagDataset(
        bags,
        np.arange(58, dtype=np.float32).reshape(29, 2),
        identities,
    )
    scope = FullCohortRefitScope(
        participant_ids=participant_ids,
        registry_hash="a" * 64,
        config_hash="b" * 64,
        oof_evidence_hash="c" * 64,
    ).bind_training_dataset(dataset)
    spec = ModelInputSpec(
        "fusion",
        n_channels=8,
        n_file_features=2,
        n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    prepared = prepare_model_factory(
        {
            "model_id": "file_bag_fusion",
            "seed": 42,
            "signal_encoder": {
                "model_id": "shapeformer_effect_size_fixed_v1",
                "input_fs_hz": 100.0,
                "shapelet_length_samples": 8,
                "discovery_stride_samples": 4,
                "shapelets_per_class": 1,
                "max_candidates_per_class": 3,
                "hidden_channels": 8,
                "attention_heads": 2,
                "attention_feedforward_channels": 12,
                "patch_size_samples": 4,
                "dropout": 0.0,
            },
        },
        spec,
        dataset,
        scope,
    )
    assert prepared.provenance["outer_train_dataset_hash"] == scope.train_dataset_hash
    model = prepared.factory()
    assert model.signal_encoder.fitted_participant_ids == participant_ids
    assert model.signal_encoder.attention_feedforward_channels == 12


def test_shapeformer_method_and_representation_cannot_fallback() -> None:
    dataset, split = _raw_outer_train()
    wrong = _shapeformer_config("shapeformer_channel_specific_osd")
    wrong["discovery_method"] = "effect_size_fixed_v1"
    with pytest.raises(ValueError, match="no discovery fallback"):
        prepare_model_factory(
            wrong,
            ModelInputSpec(
                "raw", n_channels=8, n_classes=3,
                channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
            ),
            dataset,
            split,
        )
    with pytest.raises(ValueError, match="require raw representation"):
        create_model(
            {
                "model_id": "shapeformer_channel_specific_osd",
                "seed": 42,
            },
            ModelInputSpec("feature_matrix", n_channels=2, n_classes=3),
        )


def test_channel_specific_discovery_numeric_capacity_is_runtime_selectable() -> None:
    dataset, split = _raw_outer_train(64)
    spec = ModelInputSpec(
        "raw",
        n_channels=8,
        n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    config = _shapeformer_config("shapeformer_channel_specific_osd")
    config.update(
        {
            "num_pip_ratio": 0.15,
            "shapelets_per_class": 1,
            "max_discovery_windows": 6,
            "position_search_neighbourhood_samples": 7,
            "local_kernel_width_samples": 5,
            "local_embedding_channels": 12,
            "shape_embedding_channels": 20,
            "attention_feedforward_channels": 28,
            "attention_heads": 4,
            "attention_query_chunk_size": 9,
            "distance_position_chunk_size": 11,
            "dropout": 0.15,
            "complexity_norm": 700.0,
            "max_complexity_ratio": 2.5,
        }
    )
    config["architecture_parameters"] = materialize_architecture_parameters(
        config, spec
    )
    prepared = prepare_model_factory(config, spec, dataset, split)
    bank = prepared.resolved_model_config["shapelets"]
    assert bank.num_pip_ratio == 0.15
    assert bank.shapelets_per_class == 1
    assert bank.count == 3
    assert bank.max_discovery_windows == 6
    assert bank.position_search_neighbourhood_samples == 7
    model = prepared.factory()
    assert model.local_kernel_width_samples == 5
    assert model.local_embedding_channels == 12
    assert model.shape_embedding_channels == 20
    assert model.attention_feedforward_channels == 28
    assert model.attention_heads == 4


def test_effect_size_length_and_stride_are_runtime_selectable() -> None:
    dataset, split = _raw_outer_train(96)
    spec = ModelInputSpec(
        "raw",
        n_channels=8,
        n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    config = _shapeformer_config("shapeformer_effect_size_fixed_v1")
    config.update(
        {
            "shapelet_length_samples": 24,
            "discovery_stride_samples": 12,
            "shapelets_per_class": 1,
            "max_candidates_per_class": 4,
            "attention_feedforward_channels": 30,
        }
    )
    prepared = prepare_model_factory(config, spec, dataset, split)
    architecture = prepared.resolved_model_config["architecture_parameters"]
    assert architecture["shapelet_length_samples"] == 24
    assert architecture["candidate_stride_samples"] == 12
    assert architecture["attention_feedforward_channels"] == 30
    model = prepared.factory()
    assert model.shapelet_length_samples == 24
    assert model.discovery_stride_samples == 12
    assert model.attention_feedforward_channels == 30
    assert model.patch_attention.layers[0].linear1.out_features == 30
    assert model.provenance()["attention_feedforward_channels"] == 30


def test_channel_specific_scalar_distance_ablation_is_parallel_runtime_module() -> None:
    dataset, split = _raw_outer_train(64)
    spec = ModelInputSpec(
        "raw",
        n_channels=8,
        n_classes=3,
        channel_schema=FRAILTY_RAW_CHANNEL_SCHEMA,
    )
    config = _shapeformer_config("shapeformer_channel_specific_osd")
    config["model_id"] = "shapeformer_channel_specific_scalar_distance_ablation"
    for field in (
        "sequence_length_samples",
        "local_kernel_width_samples",
        "local_embedding_channels",
        "shape_embedding_channels",
        "attention_feedforward_channels",
        "attention_query_chunk_size",
        "complexity_norm",
        "max_complexity_ratio",
    ):
        config.pop(field)
    config.update(
        {
            "shapelets_per_class": 1,
            "max_discovery_windows": 6,
            "hidden_channels": 12,
            "patch_size_samples": 4,
            "attention_heads": 3,
            "attention_layers": 2,
            "attention_feedforward_channels": 31,
        }
    )
    config["architecture_parameters"] = materialize_architecture_parameters(
        config, spec
    )
    prepared = prepare_model_factory(config, spec, dataset, split)
    assert prepared.provenance["registry_role"] == "ablation"
    model = prepared.factory()
    assert isinstance(model, ExperimentalShapeFormer)
    assert model.channel_specific_osd_supported is True
    assert model.hidden_channels == 12
    assert model.attention_heads == 3
    assert model.attention_layers == 2
    assert model.attention_feedforward_channels == 31
