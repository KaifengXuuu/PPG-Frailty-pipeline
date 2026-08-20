"""Focused non-scientific tests for canonical channel-specific OSD discovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from ppg_frailty.models.pisd_port import (
    CANDIDATE_GENERATION_RULE,
    CANDIDATE_ENUMERATION_RULE,
    CANDIDATE_RANKING_RULE,
    DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE,
    INFORMATION_GAIN_SPLIT_RULE,
    PIP_ROUNDING_RULE,
    PIP_SELECTION_RULE,
    SELECTED_BANK_ORDER_RULE,
    POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES,
    PisdShapelets,
    _balanced_discovery_indices,
    _information_gain,
    _insertion_stage_three_pip_intervals,
)
from ppg_frailty.models.shapeformer import ExperimentalShapeFormer
from ppg_frailty.models.shapeformer_literature import (
    ChannelSpecificShapeBlock,
    LiteratureShapeFormerChannelSpecificOSD,
)


def _bank() -> PisdShapelets:
    classes = np.repeat(np.arange(3, dtype=np.int64), 3)
    participants = ("p0", "p1", "p2")
    files = ("f0", "f1", "f2")
    windows = ("w0", "w1", "w2")
    selection = "\n".join(
        f"{participant}\t{file_id}\t{window_id}"
        for participant, file_id, window_id in zip(participants, files, windows)
    ).encode("utf-8")
    roster = "\n".join(participants).encode("utf-8")
    return PisdShapelets(
        values=tuple(np.asarray((0.0, 1.0, 0.0), dtype=np.float32) for _ in range(9)),
        source_classes=classes,
        information_gains=np.ones(9),
        source_sample_indices=classes,
        source_channels=np.zeros(9, dtype=np.int64),
        source_starts=np.zeros(9, dtype=np.int64),
        source_ends=np.full(9, 3, dtype=np.int64),
        source_start_seconds=np.zeros(9),
        source_end_seconds=np.full(9, 3.0),
        candidate_lengths=np.full(9, 3, dtype=np.int64),
        source_channel_names=("RED",) * 9,
        source_participant_ids=tuple(participants[value] for value in classes),
        source_file_ids=tuple(files[value] for value in classes),
        source_window_ids=tuple(windows[value] for value in classes),
        discovery_sequence_lengths=np.full(9, 8, dtype=np.int64),
        pip_counts=np.full(9, 5, dtype=np.int64),
        fitted_participant_ids=participants,
        discovery_participant_ids=participants,
        discovery_file_ids=files,
        discovery_window_ids=windows,
        discovery_selection_hash=hashlib.sha256(selection).hexdigest(),
        channel_schema=("RED", "IR"),
        discovery_method="channel_specific_osd",
        discovery_balance="participant_file_balanced",
        input_fs_hz=1.0,
        num_pip_ratio=0.20,
        shapelets_per_class=3,
        max_discovery_windows=180,
        discovery_window_count=3,
        position_search_neighbourhood_samples=POSITION_SEARCH_NEIGHBOURHOOD_SAMPLES,
        pip_rounding_rule=PIP_ROUNDING_RULE,
        pip_selection_rule=PIP_SELECTION_RULE,
        candidate_generation_rule=CANDIDATE_GENERATION_RULE,
        candidate_enumeration_rule=CANDIDATE_ENUMERATION_RULE,
        candidate_ranking_rule=CANDIDATE_RANKING_RULE,
        selected_bank_order_rule=SELECTED_BANK_ORDER_RULE,
        discovery_position_search_boundary_rule=(
            DISCOVERY_POSITION_SEARCH_BOUNDARY_RULE
        ),
        information_gain_split_rule=INFORMATION_GAIN_SPLIT_RULE,
        outer_repeat_index=0,
        outer_fold_index=0,
        outer_train_participant_hash=hashlib.sha256(roster).hexdigest(),
    )


def test_discovery_cap_balances_participants_before_files() -> None:
    labels: list[int] = []
    participants: list[str] = []
    files: list[str] = []

    def add(label: int, participant: str, file_ids: tuple[str, ...], per_file: int) -> None:
        for file_id in file_ids:
            for _ in range(per_file):
                labels.append(label)
                participants.append(participant)
                files.append(file_id)

    # A flat participant/file-pair round robin would allocate 5:1 at quota six.
    # The confirmed two-level policy must allocate 3:3 within each class.
    add(0, "p0_many_files", ("a", "b", "c", "d", "e"), 4)
    add(0, "p0_one_file", ("z",), 20)
    add(1, "p1_many_files", ("a", "b", "c", "d", "e"), 4)
    add(1, "p1_one_file", ("z",), 20)

    selected = _balanced_discovery_indices(
        np.asarray(labels, dtype=np.int64),
        tuple(participants),
        tuple(files),
        maximum=12,
        seed=42,
    )
    assert selected.size == 12
    counts = Counter(participants[index] for index in selected.tolist())
    assert counts == {
        "p0_many_files": 3,
        "p0_one_file": 3,
        "p1_many_files": 3,
        "p1_one_file": 3,
    }
    for participant in ("p0_many_files", "p1_many_files"):
        selected_files = {
            files[index]
            for index in selected.tolist()
            if participants[index] == participant
        }
        assert len(selected_files) == 3


def test_persisted_bank_requires_exact_frailty3_classes_and_nine_shapelets() -> None:
    bank = _bank()
    with pytest.raises(ValueError, match="source_classes"):
        replace(bank, source_classes=np.asarray((0, 0, 0, 1, 1, 1, 1, 1, 1)))


def test_insertion_stage_candidate_endpoints_match_frozen_upstream_fixture() -> None:
    signal = np.asarray((0.0, 4.0, 1.0, 5.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0))
    # Captured from local upstream Shapelet.auto_pisd.auto_piss_extractor with
    # num_pip=0.5 (five PIPs for T=10). Repeated insertion-stage intervals are
    # intentional and must not be replaced by final-PIP-only triples.
    assert _insertion_stage_three_pip_intervals(signal, 5) == (
        (0, 9),
        (3, 9),
        (0, 4),
        (4, 9),
        (3, 5),
    )


def test_information_gain_matches_frozen_upstream_positive_recall_grid() -> None:
    distances = np.arange(1, 21, dtype=np.float64) / 100.0
    target_class = np.asarray(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1),
        dtype=bool,
    )
    # Captured from local upstream
    # Shapelet.shapelet_support_method.find_best_split_point_and_info_gain.
    expected = 0.10803154614559995
    assert _information_gain(distances, target_class) == expected
    # Upstream returns -1 when no 0.2-grid marker is attainable; preserving the
    # sentinel is required for identical ranking on tiny discovery cohorts.
    assert _information_gain(
        np.asarray((0.1, 0.2, 0.3)), np.asarray((True, False, False))
    ) == -1.0


def test_shape_block_selects_raw_neighbourhood_segment_and_emits_vector_formula() -> None:
    candidate = np.asarray((1.0, 2.0, 1.0), dtype=np.float32)
    block = ChannelSpecificShapeBlock(
        source_channel=1,
        source_start=3,
        source_end=6,
        shapelet=candidate,
        shape_embedding_channels=2,
        sequence_length=10,
        position_search_neighbourhood_samples=128,
        distance_position_chunk_size=2,
        complexity_norm=1000.0,
        max_complexity_ratio=3.0,
    ).eval()
    x = torch.zeros((1, 2, 10), dtype=torch.float32)
    x[0, 1] = torch.tensor((9.0, 1.0, 2.0, 1.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0))
    mask = torch.ones((1, 10), dtype=torch.bool)
    selected = block._best_segments(x[:, 1, :], mask)
    assert torch.equal(selected, torch.tensor(((1.0, 2.0, 1.0),)))
    expected = block.selected_projection(selected) - block.shapelet_projection(
        block.shapelet.unsqueeze(0)
    )
    assert torch.allclose(block(x, mask).squeeze(1), expected)


def test_faithful_shapeformer_fuses_shape_tokens_with_full_multivariate_branch() -> None:
    bank = _bank()
    model = LiteratureShapeFormerChannelSpecificOSD(
        n_channels=2,
        n_classes=3,
        sequence_length=8,
        shapelets=bank,
        local_kernel_width_samples=3,
        local_embedding_channels=8,
        shape_embedding_channels=8,
        attention_feedforward_channels=16,
        attention_heads=2,
        attention_query_chunk_size=3,
        distance_position_chunk_size=2,
        dropout=0.0,
        complexity_norm=1000.0,
        max_complexity_ratio=3.0,
        position_search_neighbourhood_samples=128,
        input_fs_hz=1.0,
    ).eval()
    x = torch.randn((2, 2, 8), generator=torch.Generator().manual_seed(42))
    mask = torch.ones((2, 8), dtype=torch.bool)
    mask[1, -2:] = False
    logits = model(x, mask)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    provenance = model.provenance()
    assert provenance["implementation_status"] == "implemented_not_benchmarked_high_compute"
    assert provenance["generic_branch_channel_count"] == 2
    assert provenance["position_search_neighbourhood_samples"] == 128


def test_shapeformer_numeric_controls_are_runtime_selectable() -> None:
    bank = replace(
        _bank(),
        num_pip_ratio=0.35,
        max_discovery_windows=12,
        position_search_neighbourhood_samples=4,
    )
    model = LiteratureShapeFormerChannelSpecificOSD(
        n_channels=2,
        n_classes=3,
        sequence_length=8,
        shapelets=bank,
        local_kernel_width_samples=1,
        local_embedding_channels=6,
        shape_embedding_channels=10,
        attention_feedforward_channels=14,
        attention_heads=2,
        attention_query_chunk_size=3,
        distance_position_chunk_size=2,
        dropout=0.15,
        complexity_norm=700.0,
        max_complexity_ratio=2.5,
        position_search_neighbourhood_samples=4,
        input_fs_hz=1.0,
    ).eval()
    assert model.num_pip_ratio == 0.35
    assert model.max_discovery_windows == 12
    assert model.position_search_neighbourhood_samples == 4
    assert model.local_kernel_width_samples == 1
    assert model.local_embedding_channels == 6
    assert model.shape_embedding_channels == 10
    assert model.attention_feedforward_channels == 14
    assert model.dropout_probability == 0.15
    x = torch.randn((1, 2, 8), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        output = model(x, torch.ones((1, 8), dtype=torch.bool))
    assert tuple(output.shape) == (1, 3)


def test_channel_specific_search_uses_nontrivial_global_time_mask() -> None:
    bank = _bank()
    model = ExperimentalShapeFormer(
        2,
        3,
        bank,
        hidden_channels=8,
        attention_heads=2,
        attention_layers=1,
        patch_size_samples=2,
        input_fs_hz=1.0,
        dropout=0.0,
        distance_position_chunk_size=2,
    ).eval()
    x = torch.tensor(
        [[[0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0], [7.0] * 8]]
    )
    mask = torch.zeros((1, 8), dtype=torch.bool)
    mask[:, :3] = True
    similarity = model._channel_specific_similarity(x, mask)
    assert similarity.shape == (1, 9)
    assert torch.isfinite(similarity).all()
    invalid = torch.zeros_like(mask)
    with pytest.raises(ValueError, match="fully valid"):
        model._channel_specific_similarity(x, invalid)
    with pytest.raises(ValueError, match=r"\[batch,time\]"):
        model.forward_features(x, mask[:, None, :].expand(-1, 2, -1))
