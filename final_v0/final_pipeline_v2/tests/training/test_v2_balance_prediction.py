"""Focused V2 balance-line, metadata and public prediction smoke tests."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ppg_frailty.training.aggregation import (
    CANONICAL_BALANCE_LINE,
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    aggregate_hierarchy,
    hierarchy_oof_rows,
)
from ppg_frailty.training.datasets import (
    FeatureMatrixDataset,
    FeatureVectorDataset,
    FileBagDataset,
    RawWindowDataset,
    SampleIdentity,
)
from ppg_frailty.training.oof import OofPredictionRow, validate_role_level_oof
from ppg_frailty.training.trainer import (
    FrozenOuterSplit,
    TrainingConfig,
    UnifiedTrainer,
    participant_file_window_sampling_weights,
)


def _identities() -> tuple[SampleIdentity, ...]:
    return (
        SampleIdentity("p0", "p0_b", "B", 0, "direct", window_id="w0"),
        SampleIdentity("p0", "p0_r1", "R1", 0, "direct", window_id="w0"),
        SampleIdentity("p0", "p0_r2", "R2", 0, "direct", window_id="w0"),
        SampleIdentity("p1", "p1_b", "B", 1, "direct", window_id="w0"),
    )


def _split() -> FrozenOuterSplit:
    return FrozenOuterSplit(
        repeat=0,
        fold=0,
        seed=42,
        train_participant_ids=("p0", "p1"),
        oof_participant_ids=("p2",),
        registry_hash="registry",
        fold_hash="fold",
    )


def _smoke_config(training_balance: str = "equal_role_families") -> TrainingConfig:
    return TrainingConfig(
        execution_mode="smoke",
        epoch_profile="smoke",
        fixed_epochs=1,
        batch_size=2,
        n_classes=2,
        training_balance=training_balance,
    )


def test_line_a_and_line_b_training_weights_have_declared_semantics() -> None:
    dataset = FeatureVectorDataset(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        ("a", "b"),
        _identities(),
    )
    line_a = participant_file_window_sampling_weights(
        dataset, training_balance="equal_files"
    )
    line_b = participant_file_window_sampling_weights(
        dataset, training_balance="equal_role_families"
    )
    np.testing.assert_allclose(line_a, (1 / 6, 1 / 6, 1 / 6, 1 / 2))
    np.testing.assert_allclose(line_b, (1 / 4, 1 / 8, 1 / 8, 1 / 2))
    assert np.isclose(line_a.sum(), 1.0)
    assert np.isclose(line_b.sum(), 1.0)


class _TinyRaw(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1))


class _TinyEnsemble(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def predict_probabilities(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.softmax(self.linear(x.mean(dim=-1)), dim=-1)


class _TinyProbabilityEnsemble(nn.Module):
    def __init__(self, member_count: int) -> None:
        super().__init__()
        self.members = nn.ModuleList(nn.Linear(2, 2) for _ in range(member_count))
        self.member_seeds = tuple(101 + index for index in range(member_count))

    def member_probabilities(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = x.mean(dim=-1)
        return torch.stack(
            [torch.softmax(member(pooled), dim=-1) for member in self.members],
            dim=0,
        )


class _TinyFileBag(nn.Module):
    def forward(
        self,
        window_bag: torch.Tensor,
        window_mask: torch.Tensor,
        file_features: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        score = window_bag.mean(dim=(1, 2, 3)) + file_features.mean(dim=1)
        return torch.stack((-score, score), dim=1)


def test_deep_fit_metadata_and_public_deep_ensemble_filebag_prediction() -> None:
    values = np.arange(64, dtype=np.float32).reshape(4, 2, 8) / 64.0
    raw = RawWindowDataset(values, _identities())
    trainer = UnifiedTrainer(_smoke_config())
    result = trainer.fit(_TinyRaw, raw, _split())
    assert result.selected_epoch == 1
    assert result.provenance is not None
    assert result.provenance.training_balance == "equal_role_families"
    assert result.provenance.expected_aggregation_rule == LINE_B_EQUAL_ROLE_FAMILIES
    assert result.model.training_balance_ == "equal_role_families"

    probability, labels, identities = trainer.predict_probabilities(result.model, raw)
    assert probability.shape == (4, 2)
    assert labels.tolist() == [0, 0, 0, 1]
    assert tuple(item.file_id for item in identities) == tuple(
        item.file_id for item in _identities()
    )

    ensemble_probability, _, _ = trainer.predict_probabilities(_TinyEnsemble(), raw)
    assert ensemble_probability.shape == (4, 2)

    file_bag = FileBagDataset(
        (
            np.ones((1, 2, 8), dtype=np.float32),
            -np.ones((2, 2, 8), dtype=np.float32),
        ),
        np.asarray(((0.1,), (-0.1,)), dtype=np.float32),
        (
            SampleIdentity("p0", "p0_bag", "B", 0, "direct"),
            SampleIdentity("p1", "p1_bag", "R", 1, "direct"),
        ),
    )
    bag_probability, bag_labels, bag_identities = trainer.predict_probabilities(
        _TinyFileBag(), file_bag
    )
    assert bag_probability.shape == (2, 2)
    assert bag_labels.tolist() == [0, 1]
    assert len(bag_identities) == 2


def test_public_ensemble_prediction_uses_declared_arbitrary_member_roster() -> None:
    values = np.arange(64, dtype=np.float32).reshape(4, 2, 8) / 64.0
    raw = RawWindowDataset(values, _identities())
    trainer = UnifiedTrainer(_smoke_config())

    member_probability, averaged, member_labels, member_identities = (
        trainer.predict_ensemble_members(_TinyProbabilityEnsemble(2), raw)
    )
    assert member_probability.shape == (2, 4, 2)
    torch.testing.assert_close(
        torch.from_numpy(averaged),
        torch.from_numpy(member_probability.mean(axis=0)),
    )
    assert member_labels.tolist() == [0, 0, 0, 1]
    assert len(member_identities) == 4


class _MaskEstimator:
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        participant_ids: tuple[str, ...],
        mask: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "_MaskEstimator":
        self.fitted_participant_ids_ = tuple(participant_ids)
        self.fitted_object_provenance_ = {
            "fake": {"fitted_participant_ids": tuple(participant_ids)}
        }
        self.observed_sample_weight_ = np.asarray(sample_weight)
        self.observed_fit_mask_ = np.asarray(mask)
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        assert mask is not None
        self.observed_predict_mask_ = np.asarray(mask)
        score = np.asarray(x, dtype=np.float64).mean(axis=(1, 2))
        positive = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack((1.0 - positive, positive))


def test_estimator_uses_same_weights_and_public_masked_prediction() -> None:
    values = np.arange(24, dtype=np.float32).reshape(4, 2, 3) / 24.0
    mask = np.asarray(
        ((True, True, True), (True, True, False), (True, False, False), (True, True, True))
    )
    dataset = FeatureMatrixDataset(values, mask, _identities(), ("c0", "c1"))
    estimator = _MaskEstimator()
    trainer = UnifiedTrainer(_smoke_config())
    result = trainer.fit_estimator(estimator, dataset, _split())
    assert result.provenance is not None
    assert result.provenance.epoch_profile == "not_applicable"
    expected = participant_file_window_sampling_weights(
        dataset, training_balance="equal_role_families"
    )
    np.testing.assert_allclose(estimator.observed_sample_weight_ / 4.0, expected)
    probability, labels, identities = trainer.predict_estimator_probabilities(
        estimator, dataset
    )
    assert probability.shape == (4, 2)
    assert labels.tolist() == [0, 0, 0, 1]
    assert len(identities) == 4
    np.testing.assert_array_equal(estimator.observed_predict_mask_, mask)


def _oof(file_id: str, role: str, probabilities: tuple[float, float]) -> OofPredictionRow:
    return OofPredictionRow(
        participant_id="p0",
        file_id=file_id,
        role=role,
        label=0,
        probabilities=probabilities,
        repeat=0,
        fold=0,
        split_seed=42,
        training_seed=42,
        config_hash="config",
        manifest_hash="manifest",
        fold_hash="fold",
        preprocessing_hash="preprocess",
        feature_hash="feature",
        model_hash="model",
        representation_mode="raw",
        signal_route="direct",
        quality_score=1.0,
        retained=True,
        level="file",
        class_order=(0, 1),
    )


def test_aggregation_line_a_equal_files_vs_line_b_equal_families() -> None:
    rows = (
        _oof("b", "B", (1.0, 0.0)),
        _oof("r1", "R1", (0.0, 1.0)),
        _oof("r2", "R2", (0.0, 1.0)),
        _oof("r3", "R3", (0.0, 1.0)),
    )
    line_a = aggregate_hierarchy(rows, balance_line=LINE_A_EQUAL_FILES)
    line_b = aggregate_hierarchy(rows, balance_line=LINE_B_EQUAL_ROLE_FAMILIES)
    np.testing.assert_allclose(line_a.participant_rows[0].probabilities, (0.25, 0.75))
    np.testing.assert_allclose(line_b.participant_rows[0].probabilities, (0.5, 0.5))
    assert line_a.role_rows == ()
    assert {row.role for row in line_b.role_rows} == {"B", "R"}
    assert line_a.participant_rows[0].aggregation_rule == LINE_A_EQUAL_FILES
    assert line_b.participant_rows[0].aggregation_rule == LINE_B_EQUAL_ROLE_FAMILIES


def test_role_aware_hierarchy_is_default_and_role_rows_are_persistable() -> None:
    rows = (
        _oof("b", "B", (1.0, 0.0)),
        _oof("r1", "R1", (0.0, 1.0)),
        _oof("r2", "R2", (0.0, 1.0)),
    )
    result = aggregate_hierarchy(rows)
    assert result.balance_line == CANONICAL_BALANCE_LINE
    persisted = hierarchy_oof_rows(result)
    assert {row.level for row in persisted} == {"file", "role", "participant"}
    validate_role_level_oof(persisted)
