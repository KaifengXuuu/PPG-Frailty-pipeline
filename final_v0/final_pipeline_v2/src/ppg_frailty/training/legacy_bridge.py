"""Reviewed bridge profiles using shared configurable training mechanics.

The bridge keeps its cumulative and centred-star sampling diagnostics here,
while optimizer, sampler, and class-weight algorithms remain shared with the
ordinary V2 trainer. Shared algorithms must not acquire bridge-only duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - deep entry points fail when called.
    torch = None
    nn = None
    DataLoader = None
    Dataset = Any

from sklearn.metrics import balanced_accuracy_score

from .aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    canonical_role_family,
)
from .datasets import collate_samples
from .trainer import (
    UnifiedTrainer,
    configured_class_weight_vector,
    dataset_identities,
    outer_train_window_inverse_frequency_weights,
    participant_file_window_sampling_weights,
)


BRIDGE_SAMPLERS = frozenset(
    {
        "exhaustive_shuffle_without_replacement",
        "uniform_replacement",
        "balance_line_weighted_v2",
    }
)
BRIDGE_CLASS_WEIGHTING = frozenset(
    {
        "outer_train_window_inverse_frequency",
        "outer_train_inverse_frequency",
    }
)


@dataclass(frozen=True)
class LegacyBridgeTrainingConfig:
    """Resolved training controls for a cumulative or centred-star profile."""

    profile_id: str
    sampler: str
    class_weighting: str
    optimizer: str
    training_balance: str
    expected_aggregation_rule: str
    batch_size: int
    fixed_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_workers: int = 0
    seed: int = 42
    n_classes: int = 3
    epoch_rule: str = "fixed_epoch"
    epoch_profile: str = "legacy_bridge_fixed10"
    execution_mode: str = "legacy_bridge"
    maximum_inner_epochs: int = 0
    inner_patience: int = 0
    classifier_role_families: tuple[str, ...] = ("B", "R")
    loss: str = "cross_entropy"
    focal_gamma: float = 2.0
    class_weight_beta: float = 0.999
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = None
    deterministic_algorithms: bool = True
    cache_policy: str = "fresh_raw_csv_no_training_cache"
    outer_labels_visible_to_trainer: bool = False
    inner_grouped_folds: int = 0
    refit_on_all_outer_training: bool = True
    legacy_epoch_rule_alias: str | None = None
    protocol_design: str = "cumulative_chain_v1"

    @property
    def class_count_basis(self) -> str:
        """Expose the basis encoded by the frozen historical profile name."""

        return (
            "row"
            if self.class_weighting == "outer_train_window_inverse_frequency"
            else "participant"
        )

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.replace("_", "").isalnum():
            raise ValueError("bridge profile_id must be a non-empty safe identifier")
        if self.protocol_design not in {
            "cumulative_chain_v1",
            "centered_star_v1",
            "field_driven_followup_v1",
        }:
            raise ValueError("unsupported bridge protocol design")
        if self.protocol_design == "cumulative_chain_v1" and self.profile_id not in {
            f"L{value}" for value in range(8)
        }:
            raise ValueError("cumulative legacy bridge profile_id must be L0..L7")
        if self.sampler not in BRIDGE_SAMPLERS:
            raise ValueError("unsupported legacy bridge sampler")
        if self.class_weighting not in BRIDGE_CLASS_WEIGHTING:
            raise ValueError("unsupported legacy bridge class weighting")
        if self.optimizer not in {"adamw", "adam"}:
            raise ValueError("legacy bridge optimizer must be adamw or adam")
        if self.expected_aggregation_rule not in {
            LINE_A_EQUAL_FILES,
            LINE_B_EQUAL_ROLE_FAMILIES,
        }:
            raise ValueError("legacy bridge aggregation rule must be Line A or Line B")
        if self.training_balance not in {
            "legacy_exhaustive",
            "uniform_replacement",
            "equal_role_families",
        }:
            raise ValueError("unsupported legacy bridge training balance")
        if self.batch_size <= 0 or self.fixed_epochs <= 0:
            raise ValueError("bridge requires positive batch size and epoch count")
        if self.seed != 42:
            raise ValueError("legacy bridge training seed is frozen at 42")
        if self.outer_labels_visible_to_trainer or self.inner_grouped_folds != 0:
            raise ValueError("legacy bridge keeps outer labels hidden and has no inner selection")


def _count_strings(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def sampling_diagnostics(
    dataset: Dataset,
    draw_indices: np.ndarray,
    *,
    sampler_identity: str,
    class_weight_vector: np.ndarray,
) -> dict[str, Any]:
    """Return the eleven required per-fold/per-epoch exposure diagnostics."""

    identities = dataset_identities(dataset)
    draws = np.asarray(draw_indices, dtype=np.int64)
    if draws.ndim != 1 or draws.size != len(dataset):
        raise ValueError("bridge sampler must draw exactly len(dataset) rows")
    if np.any(draws < 0) or np.any(draws >= len(dataset)):
        raise ValueError("bridge sampler emitted an invalid row index")
    unique = int(np.unique(draws).size)
    selected = [identities[int(index)] for index in draws]
    return {
        "dataset_row_count": int(len(dataset)),
        "draw_count": int(draws.size),
        "unique_row_draw_count": unique,
        "duplicate_draw_fraction": float((draws.size - unique) / max(draws.size, 1)),
        "never_drawn_row_fraction": float((len(dataset) - unique) / max(len(dataset), 1)),
        "draw_counts_by_participant": _count_strings(
            [str(value.participant_id) for value in selected]
        ),
        "draw_counts_by_class": _count_strings(
            [str(int(value.label)) for value in selected]
        ),
        "draw_counts_by_B_R_family": _count_strings(
            [canonical_role_family(str(value.role)) for value in selected]
        ),
        "draw_counts_by_file": _count_strings(
            [str(value.file_id) for value in selected]
        ),
        "class_weight_vector": [float(value) for value in class_weight_vector],
        "sampler_identity": str(sampler_identity),
    }


class LegacyBridgeTrainer(UnifiedTrainer):
    """UnifiedTrainer variant for hash-bound historical bridge profiles."""

    config: LegacyBridgeTrainingConfig

    def __init__(self, config: LegacyBridgeTrainingConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]

    def _class_weight_vector(self, dataset: Dataset) -> np.ndarray:
        return configured_class_weight_vector(
            dataset,
            class_weighting=self.config.class_weighting,
            n_classes=self.config.n_classes,
            class_weight_beta=self.config.class_weight_beta,
        )

    def _criterion(self, dataset: Dataset) -> nn.Module:
        return super()._criterion(dataset)

    def _optimizer(self, member: nn.Module) -> torch.optim.Optimizer:
        return super()._optimizer(member)

    def _draw_indices(self, dataset: Dataset, *, epoch_seed: int) -> np.ndarray:
        generator = torch.Generator()
        generator.manual_seed(int(epoch_seed))
        size = int(len(dataset))
        if self.config.sampler == "exhaustive_shuffle_without_replacement":
            return torch.randperm(size, generator=generator).cpu().numpy().astype(np.int64)
        if self.config.sampler == "uniform_replacement":
            weights = torch.ones(size, dtype=torch.double)
        else:
            weights = torch.as_tensor(
                participant_file_window_sampling_weights(
                    dataset,
                    training_balance="equal_role_families",
                    allowed_role_families=self.config.classifier_role_families,
                ),
                dtype=torch.double,
            )
        return (
            torch.multinomial(weights, size, replacement=True, generator=generator)
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    def _loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        seed_offset: int = 0,
        absolute_seed: int | None = None,
    ) -> DataLoader:
        """Prediction loader is sequential; training order is created explicitly."""

        if shuffle:
            seed = self.config.seed + seed_offset if absolute_seed is None else absolute_seed
            indices = self._draw_indices(dataset, epoch_seed=int(seed))
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=indices.tolist(),
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=collate_samples,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_samples,
        )

    def _train_epochs(
        self,
        model: nn.Module,
        dataset: Dataset,
        epochs: int,
        *,
        training_data_scope: str,
    ) -> list[dict[str, Any]]:
        """Train and persist exposure diagnostics plus descriptive train BA."""

        model.to(self.device)
        history: list[dict[str, Any]] = []
        members = list(model.members) if hasattr(model, "members") else [model]
        member_seeds = tuple(
            int(value) for value in getattr(model, "member_seeds", (self.config.seed,))
        )
        if len(member_seeds) != len(members):
            raise ValueError("bridge member seeds and members are not aligned")
        optimizers = [self._optimizer(member) for member in members]
        criterion = self._criterion(dataset)
        class_weights = self._class_weight_vector(dataset)
        for epoch in range(1, int(epochs) + 1):
            for member_index, (member, optimizer) in enumerate(zip(members, optimizers)):
                member_seed = member_seeds[member_index]
                epoch_seed = member_seed + epoch * 1_000_000
                self._set_absolute_seed(epoch_seed)
                indices = self._draw_indices(dataset, epoch_seed=epoch_seed)
                loader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    sampler=indices.tolist(),
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    collate_fn=collate_samples,
                )
                loss = self._train_member_epoch(member, loader, optimizer, criterion)
                history.append(
                    {
                        "epoch": epoch,
                        "member": member_index,
                        "training_seed": member_seed,
                        "epoch_rng_seed": epoch_seed,
                        "numpy_epoch_rng_seed": epoch_seed % (1 << 32),
                        "training_loss": float(loss),
                        "sampling_diagnostics": sampling_diagnostics(
                            dataset,
                            indices,
                            sampler_identity=self.config.sampler,
                            class_weight_vector=class_weights,
                        ),
                    }
                )
            probabilities, labels, identities = self._predict_probabilities_with_identities(
                model, dataset
            )
            participant_probability, participant_labels = (
                self._participant_training_predictions(
                    probabilities,
                    labels,
                    identities,
                    balance_line=self.config.expected_aggregation_rule,
                )
            )
            score = balanced_accuracy_score(
                participant_labels,
                participant_probability.argmax(axis=1),
            )
            history.append(
                {
                    "epoch": epoch,
                    "training_participant_balanced_accuracy": float(score),
                    "training_balanced_accuracy_unit": "participant",
                    "training_balanced_accuracy_aggregation_rule": (
                        self.config.expected_aggregation_rule
                    ),
                    "training_data_scope": training_data_scope,
                    "outer_heldout_used": False,
                    "metric_used_for_selection_or_checkpoint": False,
                }
            )
        return history


__all__ = [
    "BRIDGE_CLASS_WEIGHTING",
    "BRIDGE_SAMPLERS",
    "LegacyBridgeTrainer",
    "LegacyBridgeTrainingConfig",
    "outer_train_window_inverse_frequency_weights",
    "sampling_diagnostics",
]
