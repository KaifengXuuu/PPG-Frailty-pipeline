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
from ppg_frailty.training.datasets import RawWindowDataset, SampleIdentity
from ppg_frailty.training.trainer import FrozenOuterSplit


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
    values = np.stack(
        [
            np.stack((np.sin(axis), np.cos(axis))),
            np.stack((np.sin(axis + 0.2), np.cos(axis + 0.2))),
            np.stack((-np.sin(axis), -np.cos(axis))),
            np.stack((-np.sin(axis + 0.2), -np.cos(axis + 0.2))),
            np.stack((np.sin(axis + 0.7), -np.cos(axis + 0.7))),
            np.stack((np.sin(axis + 0.9), -np.cos(axis + 0.9))),
        ]
    ).astype(np.float32)
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
        "raw", n_channels=2, n_classes=3, channel_schema=("RED", "IR")
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


def test_shapeformer_method_and_representation_cannot_fallback() -> None:
    dataset, split = _raw_outer_train()
    wrong = _shapeformer_config("shapeformer_channel_specific_osd")
    wrong["discovery_method"] = "effect_size_fixed_v1"
    with pytest.raises(ValueError, match="no discovery fallback"):
        prepare_model_factory(
            wrong,
            ModelInputSpec(
                "raw", n_channels=2, n_classes=3, channel_schema=("RED", "IR")
            ),
            dataset,
            split,
        )
    with pytest.raises(ValueError, match="require raw representation"):
        create_model(
            {
                "model_id": "shapeformer_channel_specific_osd",
                "seed": 42,
                "architecture_parameters": {"guard_test": True},
            },
            ModelInputSpec("feature_matrix", n_channels=2, n_classes=3),
        )
