"""Unified frozen-membership trainer with outer-label isolation.

统一的冻结成员训练器，并严格隔离 outer 标签。
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.metrics import balanced_accuracy_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
except ImportError:  # pragma: no cover - covered by subprocess portability test
    torch = None
    nn = None
    DataLoader = None
    Dataset = Any
    Subset = None
    WeightedRandomSampler = None

from ..provenance import stable_payload_sha256
from ..data.schema import ROLE_FAMILIES
from .aggregation import (
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    aggregation_rule_for_training_balance,
    canonical_role_family,
)
from .datasets import FeatureMatrixDataset, FileBagDataset, collate_samples


def _require_torch() -> None:
    """Fail only for deep operations / 仅深度操作时失败。"""

    if torch is None or nn is None or DataLoader is None:
        raise ImportError(
            "deep training requires optional dependency torch; "
            "UnifiedTrainer.fit_estimator remains available without it"
        )


_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = {":16:8", ":4096:8"}


def resolve_torch_training_device(
    requested: str,
    *,
    deterministic_algorithms: bool,
) -> Any:
    """Resolve a Torch device and fail before model/data allocation.

    Deterministic CUDA matmul requires one of the two cuBLAS workspace
    settings documented by PyTorch.  Set the larger deterministic workspace
    only when the caller did not provide one; reject an invalid explicit value
    instead of silently disabling determinism or falling back to CPU.
    """

    _require_torch()
    device = torch.device(str(requested).strip())
    if device.type != "cuda":
        return device
    if deterministic_algorithms:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
            raise RuntimeError(
                "deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG "
                "to be :4096:8 or :16:8"
            )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but torch.cuda.is_available() is false; "
            "CPU fallback is forbidden"
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device.index} is unavailable; "
            f"visible device count={torch.cuda.device_count()}"
        )
    return device


@dataclass(frozen=True)
class FrozenOuterSplit:
    """Immutable subject membership for one repeat/fold / 单个 repeat/fold 的不可变成员。"""

    repeat: int
    fold: int
    seed: int
    train_participant_ids: tuple[str, ...]
    oof_participant_ids: tuple[str, ...]
    registry_hash: str
    fold_hash: str
    train_dataset_hash: str | None = None

    def __post_init__(self) -> None:
        train = tuple(sorted(set(str(value) for value in self.train_participant_ids)))
        oof = tuple(sorted(set(str(value) for value in self.oof_participant_ids)))
        if not train or not oof or set(train) & set(oof):
            raise ValueError("outer train and OOF participant sets must be non-empty and disjoint")
        object.__setattr__(self, "train_participant_ids", train)
        object.__setattr__(self, "oof_participant_ids", oof)

    def assert_train_only(
        self,
        participant_ids: list[str] | tuple[str, ...],
        *,
        exact: bool = False,
    ) -> None:
        """Fail on outer leakage and optionally require the complete train roster.

        若发生 outer 泄漏则关闭失败；最终 refit 时可进一步要求完整 train roster。
        """

        fitted = set(str(value) for value in participant_ids)
        if not fitted or not fitted <= set(self.train_participant_ids):
            raise ValueError("fitted rows must be a non-empty subset of frozen outer-train subjects")
        if fitted & set(self.oof_participant_ids):
            raise ValueError("outer-OOF participant reached a fitting operation")
        if exact and fitted != set(self.train_participant_ids):
            missing = sorted(set(self.train_participant_ids) - fitted)
            extra = sorted(fitted - set(self.train_participant_ids))
            raise ValueError(
                f"final fit must use the exact frozen outer-train roster; missing={missing}, extra={extra}"
            )

    def bind_training_dataset(self, dataset: Dataset) -> "FrozenOuterSplit":
        """Return a split bound to exact row identities and values.

        返回绑定到逐行身份与数值内容的 split。该哈希可识别把 held-out 数组伪装成
        train participant ID 的错误。
        """

        self.assert_train_only(dataset_participant_ids(dataset), exact=True)
        validate_dataset_identity_coherence(dataset)
        return replace(self, train_dataset_hash=dataset_binding_hash(dataset))

    def assert_training_dataset(self, dataset: Dataset, *, exact: bool = True) -> None:
        """Validate roster, row identity coherence and optional content binding.

        校验 roster、逐行身份一致性，以及可选的内容绑定哈希。
        """

        self.assert_train_only(dataset_participant_ids(dataset), exact=exact)
        validate_dataset_identity_coherence(dataset)
        if self.train_dataset_hash is not None:
            observed = dataset_binding_hash(dataset)
            if observed != self.train_dataset_hash:
                raise ValueError("training dataset identity/content hash does not match frozen split")

    @property
    def membership_hash(self) -> str:
        """Stable hash of exact membership / 精确成员关系的稳定哈希。"""

        return stable_payload_sha256(
            {
                "repeat": self.repeat,
                "fold": self.fold,
                "seed": self.seed,
                "train": self.train_participant_ids,
                "oof": self.oof_participant_ids,
                "registry_hash": self.registry_hash,
                "fold_hash": self.fold_hash,
                "train_dataset_hash": self.train_dataset_hash,
            }
        )


@dataclass(frozen=True)
class FullCohortRefitScope:
    """Exact all-29 final-refit membership guard with no pseudo holdout."""

    participant_ids: tuple[str, ...]
    registry_hash: str
    config_hash: str
    oof_evidence_hash: str
    train_dataset_hash: str | None = None
    scope_kind: str = "final_refit_all_29"

    def __post_init__(self) -> None:
        participants = tuple(sorted(set(map(str, self.participant_ids))))
        if len(participants) != 29:
            raise ValueError("full-cohort refit scope requires exactly 29 participants")
        if self.scope_kind != "final_refit_all_29":
            raise ValueError("full-cohort scope_kind is frozen")
        for name in ("registry_hash", "config_hash", "oof_evidence_hash"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "participant_ids", participants)

    @property
    def train_participant_ids(self) -> tuple[str, ...]:
        return self.participant_ids

    @property
    def oof_participant_ids(self) -> tuple[str, ...]:
        return ()

    @property
    def repeat(self) -> int:
        """Compatibility index for fold-local model preparation provenance."""

        return 0

    @property
    def fold(self) -> int:
        """Compatibility index; fold_hash remains explicitly full-cohort."""

        return 0

    @property
    def seed(self) -> int:
        return 42

    @property
    def fold_hash(self) -> str:
        return stable_payload_sha256(
            {
                "scope_kind": self.scope_kind,
                "participants": self.participant_ids,
                "registry_hash": self.registry_hash,
                "config_hash": self.config_hash,
                "oof_evidence_hash": self.oof_evidence_hash,
            }
        )

    @property
    def membership_hash(self) -> str:
        return stable_payload_sha256(
            {
                "scope_kind": self.scope_kind,
                "participants": self.participant_ids,
                "registry_hash": self.registry_hash,
                "config_hash": self.config_hash,
                "oof_evidence_hash": self.oof_evidence_hash,
                "train_dataset_hash": self.train_dataset_hash,
            }
        )

    def assert_train_only(
        self,
        participant_ids: list[str] | tuple[str, ...],
        *,
        exact: bool = False,
    ) -> None:
        fitted = set(map(str, participant_ids))
        expected = set(self.participant_ids)
        if not fitted or not fitted <= expected:
            raise ValueError("final-refit rows must be a non-empty subset of the all-29 roster")
        if exact and fitted != expected:
            raise ValueError("final refit must use the exact all-29 participant roster")

    def bind_training_dataset(self, dataset: Dataset) -> "FullCohortRefitScope":
        self.assert_train_only(dataset_participant_ids(dataset), exact=True)
        validate_dataset_identity_coherence(dataset)
        return replace(self, train_dataset_hash=dataset_binding_hash(dataset))

    def assert_training_dataset(self, dataset: Dataset, *, exact: bool = True) -> None:
        self.assert_train_only(dataset_participant_ids(dataset), exact=exact)
        validate_dataset_identity_coherence(dataset)
        if self.train_dataset_hash is not None:
            observed = dataset_binding_hash(dataset)
            if observed != self.train_dataset_hash:
                raise ValueError("final-refit dataset identity/content hash changed")


@dataclass(frozen=True)
class InnerGroupedSplit:
    """One deterministic validation fold from a grouped inner split.

    The split contains participant identities only.  It cannot carry labels or
    row arrays from the outer OOF partition, and ``validate`` requires an exact
    partition of the frozen outer-training roster.
    """

    train_participant_ids: tuple[str, ...]
    validation_participant_ids: tuple[str, ...]
    n_folds: int = 2
    validation_fold_index: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        train = tuple(str(value) for value in self.train_participant_ids)
        validation = tuple(str(value) for value in self.validation_participant_ids)
        if len(train) != len(set(train)) or len(validation) != len(set(validation)):
            raise ValueError("inner participant identities must be unique")
        if isinstance(self.n_folds, bool) or not isinstance(
            self.n_folds, (int, np.integer)
        ):
            raise ValueError("inner n_folds must be an integer")
        if isinstance(self.validation_fold_index, bool) or not isinstance(
            self.validation_fold_index, (int, np.integer)
        ):
            raise ValueError("inner validation_fold_index must be an integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("inner seed must be an integer")
        if int(self.n_folds) < 2:
            raise ValueError("inner n_folds must be at least two")
        if not 0 <= int(self.validation_fold_index) < int(self.n_folds):
            raise ValueError("inner validation_fold_index is outside n_folds")
        if not 0 <= int(self.seed) <= 0xFFFF_FFFF:
            raise ValueError("inner seed must be in [0, 2^32-1]")
        object.__setattr__(self, "train_participant_ids", train)
        object.__setattr__(self, "validation_participant_ids", validation)
        object.__setattr__(self, "n_folds", int(self.n_folds))
        object.__setattr__(self, "validation_fold_index", int(self.validation_fold_index))
        object.__setattr__(self, "seed", int(self.seed))

    def validate(self, outer: FrozenOuterSplit) -> None:
        """Require disjoint subsets of outer-train / 要求是 outer-train 的互斥子集。"""

        train = set(self.train_participant_ids)
        validation = set(self.validation_participant_ids)
        outer_train = set(outer.train_participant_ids)
        if not train or not validation or train & validation:
            raise ValueError("inner train/validation sets must be non-empty and disjoint")
        if train | validation != outer_train:
            raise ValueError("inner membership must exactly partition outer-train")

    @property
    def membership_hash(self) -> str:
        """Bind the selected fold identity and its exact participant roster."""

        return stable_payload_sha256(
            {
                "train_participant_ids": tuple(sorted(self.train_participant_ids)),
                "validation_participant_ids": tuple(
                    sorted(self.validation_participant_ids)
                ),
                "n_folds": self.n_folds,
                "validation_fold_index": self.validation_fold_index,
                "seed": self.seed,
            }
        )


DEEP_EPOCH_PROFILES: dict[str, int] = {
    "default_10": 10,
    "ablation_7": 7,
    "ablation_15": 15,
}


def derived_epoch_profile(epoch_rule: str, fixed_epochs: int) -> str:
    """Return provenance derived solely from executable epoch controls."""

    if epoch_rule == "inner_grouped_selection":
        return "inner_grouped_selection"
    named_profiles = {
        epoch_count: profile_name
        for profile_name, epoch_count in DEEP_EPOCH_PROFILES.items()
    }
    return named_profiles.get(int(fixed_epochs), f"configured_{int(fixed_epochs)}")

TRAINING_OPTIMIZERS = frozenset({"adam", "adamw", "sgd", "rmsprop"})
OPTIMIZER_PARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "adam": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    },
    "adamw": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    },
    "sgd": {
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "maximize": False,
    },
    "rmsprop": {
        "alpha": 0.99,
        "eps": 1e-8,
        "momentum": 0.0,
        "centered": False,
        "maximize": False,
    },
}
TRAINING_SAMPLERS = frozenset(
    {
        "balance_line_weighted_v2",
        "uniform_replacement",
        "exhaustive_shuffle_without_replacement",
        "subject_balanced",
        "class_subject_balanced",
    }
)
TRAINING_CLASS_WEIGHTINGS = frozenset(
    {
        "inverse_frequency",
        "effective_number",
        "none",
    }
)
TRAINING_CLASS_COUNT_BASES = frozenset({"participant", "row"})
_CLASS_WEIGHTING_INPUT_ALIASES = {
    "outer_train_inverse_frequency": ("inverse_frequency", "participant"),
    "outer_train_window_inverse_frequency": ("inverse_frequency", "row"),
}
TRAINING_LOSSES = frozenset(
    {"cross_entropy", "balanced_softmax", "focal_loss"}
)
_TRAINING_LOSS_INPUT_ALIASES = {"weighted_ce": "cross_entropy"}
# No training-cache I/O is wired into UnifiedTrainer.  Keep the field as an
# explicit effective-config statement, but expose only the behavior that exists.
TRAINING_CACHE_POLICIES = frozenset({"disabled"})


def _finite_optimizer_float(
    value: Any,
    *,
    field: str,
    minimum: float = 0.0,
    maximum_exclusive: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Normalize one optimizer scalar without accepting booleans or NaN/Inf."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"optimizer_parameters.{field} must be a finite number")
    resolved = float(value)
    if strictly_positive and resolved <= 0.0:
        raise ValueError(f"optimizer_parameters.{field} must be positive")
    if not strictly_positive and resolved < minimum:
        raise ValueError(
            f"optimizer_parameters.{field} must be at least {minimum:g}"
        )
    if maximum_exclusive is not None and resolved >= maximum_exclusive:
        raise ValueError(
            f"optimizer_parameters.{field} must be below {maximum_exclusive:g}"
        )
    return resolved


def resolve_optimizer_parameters(
    optimizer: str,
    parameters: Any,
) -> dict[str, Any]:
    """Return the complete, optimizer-specific parameter mapping.

    ``learning_rate`` and ``weight_decay`` remain common top-level training
    controls.  Every accepted optimizer-specific key is materialized here so it
    participates in the effective configuration hash and is never silently
    ignored by the torch constructor.
    """

    if optimizer not in OPTIMIZER_PARAMETER_DEFAULTS:
        raise ValueError(f"unsupported optimizer: {optimizer!r}")
    if not isinstance(parameters, Mapping):
        raise ValueError("optimizer_parameters must be a string-keyed mapping")
    if not all(isinstance(key, str) for key in parameters):
        raise ValueError("optimizer_parameters must be a string-keyed mapping")
    defaults = OPTIMIZER_PARAMETER_DEFAULTS[optimizer]
    unknown = sorted(set(parameters) - set(defaults))
    if unknown:
        raise ValueError(
            f"optimizer_parameters for {optimizer} contain unsupported keys: {unknown}"
        )
    resolved = {**defaults, **parameters}
    if optimizer in {"adam", "adamw"}:
        betas = resolved["betas"]
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
        ):
            raise ValueError("optimizer_parameters.betas must contain two values")
        beta1 = _finite_optimizer_float(
            betas[0], field="betas[0]", maximum_exclusive=1.0
        )
        beta2 = _finite_optimizer_float(
            betas[1], field="betas[1]", maximum_exclusive=1.0
        )
        resolved["betas"] = [beta1, beta2]
        resolved["eps"] = _finite_optimizer_float(
            resolved["eps"], field="eps", strictly_positive=True
        )
        for field_name in ("amsgrad", "maximize"):
            if not isinstance(resolved[field_name], bool):
                raise ValueError(
                    f"optimizer_parameters.{field_name} must be boolean"
                )
    elif optimizer == "sgd":
        resolved["momentum"] = _finite_optimizer_float(
            resolved["momentum"], field="momentum"
        )
        resolved["dampening"] = _finite_optimizer_float(
            resolved["dampening"], field="dampening"
        )
        for field_name in ("nesterov", "maximize"):
            if not isinstance(resolved[field_name], bool):
                raise ValueError(
                    f"optimizer_parameters.{field_name} must be boolean"
                )
        if resolved["nesterov"] and (
            resolved["momentum"] <= 0.0 or resolved["dampening"] != 0.0
        ):
            raise ValueError(
                "SGD nesterov requires positive momentum and zero dampening"
            )
    else:
        resolved["alpha"] = _finite_optimizer_float(
            resolved["alpha"], field="alpha", maximum_exclusive=1.0
        )
        resolved["eps"] = _finite_optimizer_float(
            resolved["eps"], field="eps", strictly_positive=True
        )
        resolved["momentum"] = _finite_optimizer_float(
            resolved["momentum"], field="momentum"
        )
        for field_name in ("centered", "maximize"):
            if not isinstance(resolved[field_name], bool):
                raise ValueError(
                    f"optimizer_parameters.{field_name} must be boolean"
                )
    return resolved


def normalize_participant_window_quota(value: Any) -> str:
    """Normalize the legacy per-subject epoch quota into a stable hash value."""

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "none", "0", "-1"}:
            return "all"
        if text.endswith("%"):
            try:
                percentage = float(text[:-1].strip())
            except ValueError as exc:
                raise ValueError(
                    "participant_window_quota percentage is not numeric"
                ) from exc
            if not np.isfinite(percentage) or not 0.0 < percentage <= 100.0:
                raise ValueError(
                    "participant_window_quota percentage must be in (0,100]"
                )
            return f"{percentage:g}%"
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(
                "participant_window_quota must be all, a positive count, or a fraction"
            ) from exc
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(
            "participant_window_quota must be all, a positive count, or a fraction"
        )
    numeric = float(value)
    if 0.0 < numeric <= 1.0 and not numeric.is_integer():
        return f"{numeric * 100.0:g}%"
    if numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(
            "participant_window_quota must be a positive integer or fraction in (0,1]"
        )
    return str(int(numeric))


@dataclass(frozen=True)
class TrainingConfig:
    """Executable training controls with conservative backward-compatible defaults.

    ``fixed_epoch`` and outer-train-only ``inner_grouped_selection`` are parallel
    strategies.  ``execution_mode`` and ``epoch_profile`` remain readable for
    legacy YAML compatibility, but ordinary configs materialize them as derived
    provenance. The executable controls are the epoch rule and its numeric
    parameters. Optimizer, row sampling, class weighting and loss remain
    independent runtime choices.
    """

    epoch_rule: str = "fixed_epoch"
    epoch_profile: str = "default_10"
    execution_mode: str = "formal"
    fixed_epochs: int = 10
    maximum_inner_epochs: int = 0
    inner_patience: int = 0
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_workers: int = 0
    seed: int = 42
    optimizer: str = "adam"
    optimizer_parameters: dict[str, Any] = field(default_factory=dict)
    class_weighting: str = "inverse_frequency"
    class_count_basis: str = "participant"
    training_balance: str = "equal_role_families"
    sampler: str = "balance_line_weighted_v2"
    samples_per_epoch: int | None = None
    participant_window_quota: str | int | float = "all"
    classifier_role_families: tuple[str, ...] = ("B", "R")
    loss: str = "cross_entropy"
    focal_gamma: float = 2.0
    class_weight_beta: float = 0.999
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = None
    deterministic_algorithms: bool = True
    cache_policy: str = "disabled"
    outer_labels_visible_to_trainer: bool = False
    inner_grouped_folds: int = 0
    refit_on_all_outer_training: bool = True
    n_classes: int = 3
    legacy_epoch_rule_alias: str | None = field(init=False, default=None)
    legacy_loss_alias: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.epoch_rule == "fixed":
            object.__setattr__(self, "epoch_rule", "fixed_epoch")
            object.__setattr__(self, "legacy_epoch_rule_alias", "fixed")
        if self.epoch_rule not in {"fixed_epoch", "inner_grouped_selection"}:
            raise ValueError(
                "epoch_rule must be fixed_epoch or inner_grouped_selection"
            )
        if self.execution_mode not in {"formal", "smoke"}:
            raise ValueError("execution_mode must be formal or smoke")
        integer_fields = {
            "fixed_epochs": self.fixed_epochs,
            "maximum_inner_epochs": self.maximum_inner_epochs,
            "inner_patience": self.inner_patience,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "inner_grouped_folds": self.inner_grouped_folds,
            "n_classes": self.n_classes,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.fixed_epochs <= 0:
            raise ValueError("fixed_epochs must be positive")
        if not isinstance(self.epoch_profile, str) or not self.epoch_profile.strip():
            raise ValueError("epoch_profile must be a non-empty provenance label")
        if self.epoch_rule == "fixed_epoch":
            if (
                self.maximum_inner_epochs != 0
                or self.inner_patience != 0
                or self.inner_grouped_folds != 0
            ):
                raise ValueError(
                    "fixed_epoch requires zero inner epochs, patience and folds"
                )
        else:
            if self.maximum_inner_epochs <= 0:
                raise ValueError(
                    "inner_grouped_selection requires positive maximum_inner_epochs"
                )
            if self.inner_patience <= 0:
                raise ValueError(
                    "inner_grouped_selection requires positive inner_patience"
                )
            if self.inner_grouped_folds < 2:
                raise ValueError(
                    "inner_grouped_selection requires at least two grouped folds"
                )
        object.__setattr__(self, "execution_mode", "formal")
        object.__setattr__(
            self,
            "epoch_profile",
            derived_epoch_profile(self.epoch_rule, int(self.fixed_epochs)),
        )
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        # ``numpy.random.seed`` is still used by ``_set_absolute_seed`` and its
        # public range is narrower than torch's.  Validate against the common
        # executable range so accepted configuration never fails later.
        if self.seed < 0 or self.seed > 0xFFFF_FFFF:
            raise ValueError("seed must be in [0, 2^32-1]")
        numeric_types = (int, float, np.integer, np.floating)
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, numeric_types)
            or not np.isfinite(self.learning_rate)
            or float(self.learning_rate) <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, numeric_types)
            or not np.isfinite(self.weight_decay)
            or float(self.weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            isinstance(self.focal_gamma, bool)
            or not isinstance(self.focal_gamma, numeric_types)
            or not np.isfinite(self.focal_gamma)
            or float(self.focal_gamma) < 0.0
        ):
            raise ValueError("focal_gamma must be finite and non-negative")
        if (
            isinstance(self.class_weight_beta, bool)
            or not isinstance(self.class_weight_beta, numeric_types)
            or not np.isfinite(self.class_weight_beta)
            or not 0.0 <= float(self.class_weight_beta) < 1.0
        ):
            raise ValueError("class_weight_beta must be finite in [0,1)")
        if not isinstance(self.optimizer, str) or self.optimizer not in TRAINING_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {sorted(TRAINING_OPTIMIZERS)}"
            )
        object.__setattr__(
            self,
            "optimizer_parameters",
            resolve_optimizer_parameters(self.optimizer, self.optimizer_parameters),
        )
        if self.class_weighting in _CLASS_WEIGHTING_INPUT_ALIASES:
            canonical_weighting, implied_basis = _CLASS_WEIGHTING_INPUT_ALIASES[
                self.class_weighting
            ]
            if self.class_count_basis not in {"participant", implied_basis}:
                raise ValueError(
                    f"legacy class_weighting={self.class_weighting} conflicts with "
                    f"class_count_basis={self.class_count_basis}"
                )
            object.__setattr__(self, "class_weighting", canonical_weighting)
            object.__setattr__(self, "class_count_basis", implied_basis)
        if (
            not isinstance(self.class_weighting, str)
            or self.class_weighting not in TRAINING_CLASS_WEIGHTINGS
        ):
            raise ValueError(
                "class_weighting must be one of "
                f"{sorted(TRAINING_CLASS_WEIGHTINGS)}"
            )
        if (
            not isinstance(self.class_count_basis, str)
            or self.class_count_basis not in TRAINING_CLASS_COUNT_BASES
        ):
            raise ValueError(
                "class_count_basis must be one of "
                f"{sorted(TRAINING_CLASS_COUNT_BASES)}"
            )
        if (
            isinstance(self.label_smoothing, bool)
            or not isinstance(self.label_smoothing, numeric_types)
            or not np.isfinite(self.label_smoothing)
            or not 0.0 <= float(self.label_smoothing) < 1.0
        ):
            raise ValueError("label_smoothing must be finite in [0,1)")
        if self.gradient_clip_norm is not None and (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, numeric_types)
            or not np.isfinite(self.gradient_clip_norm)
            or float(self.gradient_clip_norm) <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be null or finite and positive")
        if not isinstance(self.training_balance, str) or self.training_balance not in {
            "equal_files",
            "equal_role_families",
        }:
            raise ValueError("unsupported training_balance")
        if not isinstance(self.sampler, str) or self.sampler not in TRAINING_SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(TRAINING_SAMPLERS)}")
        if self.samples_per_epoch is not None and (
            isinstance(self.samples_per_epoch, bool)
            or not isinstance(self.samples_per_epoch, (int, np.integer))
            or int(self.samples_per_epoch) <= 0
        ):
            raise ValueError("samples_per_epoch must be null or a positive integer")
        if self.samples_per_epoch is not None and self.sampler not in {
            "balance_line_weighted_v2",
            "uniform_replacement",
        }:
            raise ValueError(
                "samples_per_epoch applies only to replacement samplers; "
                "subject samplers use participant_window_quota"
            )
        quota = normalize_participant_window_quota(self.participant_window_quota)
        if quota != "all" and self.sampler not in {
            "subject_balanced",
            "class_subject_balanced",
        }:
            raise ValueError(
                "participant_window_quota applies only to subject samplers"
            )
        object.__setattr__(self, "participant_window_quota", quota)
        if self.samples_per_epoch is not None:
            object.__setattr__(self, "samples_per_epoch", int(self.samples_per_epoch))
        families = tuple(canonical_role_family(value) for value in self.classifier_role_families)
        if not families or len(families) != len(set(families)):
            raise ValueError("classifier_role_families must be non-empty and unique")
        # The field is consumed as a membership set throughout sampling and
        # inference. Canonical order prevents equivalent permutations from
        # changing the effective configuration hash.
        object.__setattr__(
            self,
            "classifier_role_families",
            tuple(family for family in ROLE_FAMILIES if family in families),
        )
        if self.loss in _TRAINING_LOSS_INPUT_ALIASES:
            if self.class_weighting == "none":
                raise ValueError(
                    "weighted_ce compatibility alias requires an active "
                    "class_weighting strategy"
                )
            object.__setattr__(self, "legacy_loss_alias", str(self.loss))
            object.__setattr__(
                self,
                "loss",
                _TRAINING_LOSS_INPUT_ALIASES[str(self.loss)],
            )
        if not isinstance(self.loss, str) or self.loss not in TRAINING_LOSSES:
            accepted = sorted(TRAINING_LOSSES | set(_TRAINING_LOSS_INPUT_ALIASES))
            raise ValueError(f"loss must be one of {accepted}")
        if self.loss == "balanced_softmax" and self.class_weighting != "none":
            raise ValueError(
                "balanced_softmax owns its configured count-basis correction and "
                "requires class_weighting=none"
            )
        if (
            self.class_weighting == "none"
            and self.loss != "balanced_softmax"
            and self.class_count_basis != "participant"
        ):
            raise ValueError(
                "class_count_basis=row would be inactive when class_weighting=none "
                "and loss is not balanced_softmax"
            )
        if self.loss != "focal_loss" and float(self.focal_gamma) != 2.0:
            raise ValueError(
                "focal_gamma is configurable only when loss=focal_loss"
            )
        if (
            self.class_weighting != "effective_number"
            and float(self.class_weight_beta) != 0.999
        ):
            raise ValueError(
                "class_weight_beta is configurable only when "
                "class_weighting=effective_number"
            )
        if (
            not isinstance(self.cache_policy, str)
            or self.cache_policy not in TRAINING_CACHE_POLICIES
        ):
            raise ValueError(
                f"cache_policy must be one of {sorted(TRAINING_CACHE_POLICIES)}"
            )
        if not isinstance(self.deterministic_algorithms, bool):
            raise ValueError("deterministic_algorithms must be boolean")
        if not isinstance(self.outer_labels_visible_to_trainer, bool):
            raise ValueError("outer_labels_visible_to_trainer must be boolean")
        if self.outer_labels_visible_to_trainer:
            raise ValueError("outer_labels_visible_to_trainer must remain false")
        if not isinstance(self.refit_on_all_outer_training, bool):
            raise ValueError("refit_on_all_outer_training must be boolean")
        if not self.refit_on_all_outer_training:
            raise ValueError(
                "refit_on_all_outer_training must remain true to prevent partial refits"
            )
        if self.n_classes <= 1:
            raise ValueError("n_classes must exceed one")

    @property
    def expected_aggregation_rule(self) -> str:
        """Return the aggregation line that must accompany this training balance."""

        return aggregation_rule_for_training_balance(self.training_balance)

    def _with_epoch_override(self, fixed_epochs: int) -> "TrainingConfig":
        """Apply the experiment-only short-run override and label provenance."""

        if self.epoch_rule != "fixed_epoch":
            raise ValueError("epoch override requires epoch_rule=fixed_epoch")
        if (
            isinstance(fixed_epochs, bool)
            or not isinstance(fixed_epochs, (int, np.integer))
            or int(fixed_epochs) <= 0
        ):
            raise ValueError("epoch override must be a positive integer")
        resolved = replace(self, fixed_epochs=int(fixed_epochs))
        object.__setattr__(resolved, "execution_mode", "smoke")
        object.__setattr__(resolved, "epoch_profile", "smoke")
        return resolved

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TrainingConfig":
        """Construct from a resolved YAML training block without ignored keys.

        从 resolved YAML 的 training 区块构造，禁止忽略任何未知字段。
        """

        accepted = {
            field_name
            for field_name, definition in cls.__dataclass_fields__.items()
            if definition.init
        }
        unknown = sorted(set(value) - accepted)
        if unknown:
            raise ValueError(f"unknown training configuration fields: {unknown}")
        return cls(**dict(value))

    def to_mapping(self) -> dict[str, Any]:
        """Return every effective init field in JSON/YAML-facing form."""

        resolved: dict[str, Any] = {}
        for field_name, definition in self.__dataclass_fields__.items():
            if not definition.init:
                continue
            value = getattr(self, field_name)
            resolved[field_name] = list(value) if isinstance(value, tuple) else value
        return resolved

    def validate_for_execution_backend(self, execution_backend: str) -> None:
        """Reject training controls that the selected backend cannot consume.

        Estimator models still execute the common sampler, balance-line, role,
        participant-quota, class-weighting and seed controls.  Torch-only
        controls must remain at the single :class:`TrainingConfig` default
        mapping so changing one can never alter only the config hash.
        """

        if execution_backend == "torch":
            return
        if execution_backend != "estimator":
            raise ValueError(
                f"unknown training execution_backend: {execution_backend!r}"
            )
        torch_only_fields = (
            "epoch_rule",
            "epoch_profile",
            "fixed_epochs",
            "maximum_inner_epochs",
            "inner_patience",
            "inner_grouped_folds",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "device",
            "num_workers",
            "optimizer",
            "optimizer_parameters",
            "loss",
            "focal_gamma",
            "label_smoothing",
            "gradient_clip_norm",
            "deterministic_algorithms",
            "samples_per_epoch",
        )
        observed = self.to_mapping()
        neutral = type(self)().to_mapping()
        unsupported = [
            name
            for name in torch_only_fields
            if observed[name] != neutral[name]
        ]
        if unsupported:
            raise ValueError(
                "execution_backend=estimator does not support non-default "
                "Torch-only training fields: " + ", ".join(unsupported)
            )
        if self.sampler == "uniform_replacement":
            raise ValueError(
                "execution_backend=estimator does not support "
                "sampler=uniform_replacement because estimator sample_weight "
                "cannot execute replacement draws"
            )


DEEP_EPOCH_CONFIG_IDS: dict[str, str] = {
    "fixed_epoch_10_reference": "default_10",
    "epoch_7_ablation": "ablation_7",
    "epoch_15_ablation": "ablation_15",
}
def materialize_deep_epoch_config(
    model_id: str,
    config_id: str,
    *,
    base_config: TrainingConfig | None = None,
) -> TrainingConfig:
    """Materialize one exact 7/10/15 deep profile; reject non-deep models."""

    from ..module_registry import model_factory_contract

    machine_id = str(model_id)
    if model_factory_contract(machine_id)["execution_backend"] != "torch":
        raise ValueError(
            "epoch 7/10/15 profiles are deep-only; non-iterative estimators "
            "do not execute epoch controls"
        )
    try:
        epoch_profile = DEEP_EPOCH_CONFIG_IDS[str(config_id)]
    except KeyError as exc:
        raise ValueError(f"unknown deep epoch config identity: {config_id}") from exc
    fixed_epochs = DEEP_EPOCH_PROFILES[epoch_profile]
    base = TrainingConfig() if base_config is None else base_config
    return replace(
        base,
        epoch_rule="fixed_epoch",
        fixed_epochs=fixed_epochs,
        maximum_inner_epochs=0,
        inner_patience=0,
        inner_grouped_folds=0,
    )


def materialize_all_deep_epoch_configs(
    model_id: str,
    *,
    base_config: TrainingConfig | None = None,
) -> dict[str, TrainingConfig]:
    """Return the three named materializable identities without executing them."""

    return {
        config_id: materialize_deep_epoch_config(
            model_id, config_id, base_config=base_config
        )
        for config_id in DEEP_EPOCH_CONFIG_IDS
    }


@dataclass(frozen=True)
class FittedObjectProvenance:
    """Fold-local audit record for every fitted object / 每个拟合对象的 fold-local 审计记录。"""

    object_type: str
    fitted_participant_ids: tuple[str, ...]
    outer_membership_hash: str
    registry_hash: str
    fold_hash: str
    epoch_rule: str
    selected_epoch: int | None
    state_hash: str
    dataset_binding_hash: str = ""
    training_balance: str = ""
    expected_aggregation_rule: str = ""
    epoch_profile: str = ""
    execution_mode: str = ""
    training_seed: int = 42
    member_training_seeds: tuple[int, ...] = ()
    member_state_hashes: tuple[str, ...] = ()
    optimizer: str = ""
    optimizer_parameters: dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    loss: str = ""
    class_weighting: str = ""
    class_weight_beta: float = 0.0
    focal_gamma: float = 0.0
    class_count_basis: str = ""
    class_counts: tuple[float, ...] = ()
    class_weight_count_basis: str = ""
    class_weight_vector: tuple[float, ...] = ()
    sampler: str = ""
    samples_per_epoch: int | None = None
    participant_window_quota: str = "all"
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = None
    deterministic_algorithms: bool = True
    cache_policy: str = ""
    maximum_inner_epochs: int = 0
    inner_patience: int = 0
    inner_grouped_folds: int = 0
    inner_validation_fold_index: int | None = None
    inner_membership_hash: str = ""
    inner_train_participant_ids: tuple[str, ...] = ()
    inner_validation_participant_ids: tuple[str, ...] = ()
    refit_on_all_outer_training: bool = True
    notes: tuple[str, ...] = ()


@dataclass
class TrainingResult:
    """Fitted model plus auditable history / 已拟合模型及可审计历史。"""

    model: Any
    selected_epoch: int | None
    history: list[dict[str, Any]] = field(default_factory=list)
    provenance: FittedObjectProvenance | None = None


def dataset_participant_ids(dataset: Dataset) -> tuple[str, ...]:
    """Read row identities through nested ``Subset`` objects / 穿透 Subset 读取行身份。"""

    if Subset is not None and isinstance(dataset, Subset):
        parent = dataset_participant_ids(dataset.dataset)
        return tuple(parent[int(index)] for index in dataset.indices)
    if not hasattr(dataset, "participant_ids"):
        raise TypeError("dataset must expose row-aligned participant_ids")
    return tuple(str(value) for value in dataset.participant_ids)


def dataset_identities(dataset: Dataset) -> tuple[Any, ...]:
    """Return row identities through nested subsets / 穿透子集返回逐行身份。"""

    if Subset is not None and isinstance(dataset, Subset):
        parent = dataset_identities(dataset.dataset)
        return tuple(parent[int(index)] for index in dataset.indices)
    if not hasattr(dataset, "identities"):
        raise TypeError("dataset must expose row-aligned identities")
    return tuple(dataset.identities)


def validate_dataset_identity_coherence(dataset: Dataset) -> None:
    """Reject internally inconsistent participant/file/window identities.

    拒绝 participant、file、window 或标签彼此矛盾的数据身份。该校验不能由调用方
    单独提供 participant ID 绕过。
    """

    participant_ids = dataset_participant_ids(dataset)
    identities = dataset_identities(dataset)
    if len(participant_ids) != len(identities) or not identities:
        raise ValueError("dataset identities must be non-empty and row-aligned")
    labels_by_participant: dict[str, int] = {}
    files: dict[str, tuple[str, str, int]] = {}
    windows: dict[tuple[str, str], tuple[str, int]] = {}
    for declared, identity in zip(participant_ids, identities):
        participant = str(identity.participant_id)
        if declared != participant:
            raise ValueError("dataset participant_ids disagree with row identities")
        label = int(identity.label)
        previous_label = labels_by_participant.setdefault(participant, label)
        if previous_label != label:
            raise ValueError("one participant has inconsistent labels")
        file_signature = (participant, str(identity.role), label)
        previous_file = files.setdefault(str(identity.file_id), file_signature)
        if previous_file != file_signature:
            raise ValueError("one file_id has inconsistent participant, role or label")
        if identity.window_id is not None:
            window_key = (str(identity.file_id), str(identity.window_id))
            window_signature = (participant, label)
            previous_window = windows.setdefault(window_key, window_signature)
            if previous_window != window_signature:
                raise ValueError("one window identity has inconsistent participant or label")


def _hash_dataset_value(digest: "hashlib._Hash", name: str, value: Any) -> None:
    """Update one content digest deterministically / 确定性更新一个内容哈希。"""

    digest.update(name.encode("utf-8"))
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _hash_dataset_value(digest, f"{name}[{index}]", item)
    else:
        digest.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def dataset_binding_hash(dataset: Dataset) -> str:
    """Hash row identities together with all exposed model-input arrays.

    将逐行身份与数据集公开的全部模型输入数组共同哈希，绑定名称与真实数值。
    """

    validate_dataset_identity_coherence(dataset)
    digest = hashlib.sha256()
    identity_payload = tuple(
        (
            str(identity.participant_id),
            str(identity.file_id),
            str(identity.role),
            int(identity.label),
            str(identity.signal_route),
            None if identity.window_id is None else str(identity.window_id),
        )
        for identity in dataset_identities(dataset)
    )
    _hash_dataset_value(digest, "identities", identity_payload)
    found = False
    for name in (
        "values",
        "sample_mask",
        "sample_masks",
        "row_mask",
        "window_bags",
        "window_masks",
        "file_features",
    ):
        if hasattr(dataset, name):
            _hash_dataset_value(digest, name, getattr(dataset, name))
            found = True
    if not found:
        raise TypeError("dataset binding requires exposed model-input arrays")
    return digest.hexdigest()


def participant_file_window_sampling_weights(
    dataset: Dataset,
    *,
    training_balance: str = "equal_files",
    allowed_role_families: tuple[str, ...] = ("B", "R"),
) -> np.ndarray:
    """Return Line A or Line B row weights with participant mass fixed first.

    Line A is participant→file→row. Line B is
    participant→available-role-family→file→row. Numeric role suffixes identify
    files only and are canonicalised before Line B weighting.
    """

    validate_dataset_identity_coherence(dataset)
    identities = dataset_identities(dataset)
    if training_balance not in {"equal_files", "equal_role_families"}:
        raise ValueError(f"unsupported training_balance: {training_balance!r}")
    allowed = tuple(canonical_role_family(value) for value in allowed_role_families)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("allowed_role_families must be non-empty and unique")
    participants = sorted({str(identity.participant_id) for identity in identities})
    family_by_row = [canonical_role_family(identity.role) for identity in identities]
    outside = sorted(set(family_by_row) - set(allowed))
    if outside:
        raise ValueError(
            f"classifier rows contain role families outside the declared input set: {outside}"
        )
    files_by_participant: dict[str, set[str]] = {participant: set() for participant in participants}
    families_by_participant: dict[str, set[str]] = {
        participant: set() for participant in participants
    }
    files_by_participant_family: dict[tuple[str, str], set[str]] = {}
    row_counts: dict[tuple[str, str], int] = {}
    for identity, family in zip(identities, family_by_row):
        participant = str(identity.participant_id)
        file_id = str(identity.file_id)
        files_by_participant[participant].add(file_id)
        families_by_participant[participant].add(family)
        files_by_participant_family.setdefault((participant, family), set()).add(file_id)
        row_counts[(participant, file_id)] = row_counts.get((participant, file_id), 0) + 1

    values: list[float] = []
    for identity, family in zip(identities, family_by_row):
        participant = str(identity.participant_id)
        file_id = str(identity.file_id)
        if training_balance == "equal_files":
            denominator = (
                len(participants)
                * len(files_by_participant[participant])
                * row_counts[(participant, file_id)]
            )
        else:
            denominator = (
                len(participants)
                * len(families_by_participant[participant])
                * len(files_by_participant_family[(participant, family)])
                * row_counts[(participant, file_id)]
            )
        values.append(1.0 / denominator)
    weights = np.asarray(values, dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("V2 hierarchy sampling weights do not sum to one")
    return weights


def outer_train_class_counts(
    dataset: Dataset,
    n_classes: int,
    *,
    class_count_basis: str,
) -> np.ndarray:
    """Count classes from one explicit outer-train-only statistical unit."""

    if class_count_basis == "participant":
        return outer_train_participant_class_counts(dataset, n_classes)
    if class_count_basis == "row":
        labels = np.asarray(
            [int(identity.label) for identity in dataset_identities(dataset)],
            dtype=np.int64,
        )
        if labels.ndim != 1 or labels.size != len(dataset):
            raise ValueError("dataset labels must be one-dimensional and row aligned")
        if labels.size == 0 or np.any(labels < 0) or np.any(labels >= int(n_classes)):
            raise ValueError("outer-training row labels are invalid")
        return np.bincount(labels, minlength=int(n_classes)).astype(np.float64)
    raise ValueError(
        f"class_count_basis must be one of {sorted(TRAINING_CLASS_COUNT_BASES)}"
    )


def outer_train_inverse_frequency_weights(
    dataset: Dataset,
    n_classes: int,
    *,
    class_count_basis: str = "participant",
) -> np.ndarray:
    """Compute inverse-frequency weights from the configured outer-train unit."""

    counts = outer_train_class_counts(
        dataset,
        n_classes,
        class_count_basis=class_count_basis,
    )
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[present] = counts.sum() / (int(present.sum()) * counts[present])
    return weights


def outer_train_participant_class_counts(
    dataset: Dataset,
    n_classes: int,
) -> np.ndarray:
    """Count unique outer-training participants in each class."""

    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
        raise ValueError("n_classes must be an integer")
    if int(n_classes) <= 1:
        raise ValueError("n_classes must exceed one")
    labels_by_participant: dict[str, int] = {}
    for identity in dataset_identities(dataset):
        participant = str(identity.participant_id)
        label = int(identity.label)
        previous = labels_by_participant.setdefault(participant, label)
        if previous != label:
            raise ValueError("one participant has inconsistent class labels")
    labels = np.asarray(list(labels_by_participant.values()), dtype=np.int64)
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= int(n_classes)):
        raise ValueError("outer-training participant labels are invalid")
    return np.bincount(labels, minlength=int(n_classes)).astype(np.float64)


def outer_train_effective_number_weights(
    dataset: Dataset,
    n_classes: int,
    *,
    beta: float,
    class_count_basis: str = "participant",
) -> np.ndarray:
    """Return effective-number weights from the configured count basis.

    ``beta`` is validated by :class:`TrainingConfig` and again here because this
    helper is public.  Missing classes retain zero loss weight; the mean weight
    over classes present in this train-only population is one.
    """

    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float, np.integer, np.floating))
        or not np.isfinite(beta)
        or not 0.0 <= float(beta) < 1.0
    ):
        raise ValueError("effective-number beta must be finite in [0,1)")
    counts = outer_train_class_counts(
        dataset,
        n_classes,
        class_count_basis=class_count_basis,
    )
    present = counts > 0.0
    weights = np.zeros(int(n_classes), dtype=np.float64)
    beta_value = float(beta)
    if beta_value == 0.0:
        weights[present] = 1.0
    else:
        denominator = -np.expm1(counts[present] * np.log(beta_value))
        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
            raise ValueError("effective-number denominator is not finite and positive")
        weights[present] = (1.0 - beta_value) / denominator
        weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def outer_train_window_inverse_frequency_weights(
    dataset: Dataset,
    n_classes: int = 3,
) -> np.ndarray:
    """Compute inverse-frequency class weights from outer-training rows.

    This is the historical window-count alternative to participant-count class
    weighting.  Both policies are outer-training-only and share the same class
    order; selecting one is an explicit configuration choice.
    """

    return outer_train_inverse_frequency_weights(
        dataset,
        n_classes,
        class_count_basis="row",
    )


def configured_class_weight_vector(
    dataset: Dataset,
    *,
    class_weighting: str,
    n_classes: int,
    class_weight_beta: float = 0.999,
    class_count_basis: str = "participant",
) -> np.ndarray:
    """Resolve one declared class-weight policy to an explicit vector."""

    if class_weighting in _CLASS_WEIGHTING_INPUT_ALIASES:
        class_weighting, class_count_basis = _CLASS_WEIGHTING_INPUT_ALIASES[
            class_weighting
        ]
    if class_weighting == "inverse_frequency":
        return outer_train_inverse_frequency_weights(
            dataset,
            n_classes,
            class_count_basis=class_count_basis,
        )
    if class_weighting == "effective_number":
        return outer_train_effective_number_weights(
            dataset,
            n_classes,
            beta=class_weight_beta,
            class_count_basis=class_count_basis,
        )
    if class_weighting == "none":
        return np.ones(int(n_classes), dtype=np.float32)
    raise ValueError(f"unsupported class_weighting: {class_weighting!r}")


def _participant_quota(row_count: int, quota: Any) -> int:
    """Resolve one normalized per-participant quota against available rows."""

    normalized = normalize_participant_window_quota(quota)
    if row_count <= 0:
        return 0
    if normalized == "all":
        return int(row_count)
    if normalized.endswith("%"):
        percentage = float(normalized[:-1]) / 100.0
        return max(1, int(np.ceil(int(row_count) * percentage)))
    return int(normalized)


def _subject_sampling_structure(
    dataset: Dataset,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, int],
    dict[int, tuple[str, ...]],
]:
    """Build deterministic participant/label groups from train-only identities."""

    validate_dataset_identity_coherence(dataset)
    rows_by_subject: dict[str, list[int]] = {}
    label_by_subject: dict[str, int] = {}
    for row_index, identity in enumerate(dataset_identities(dataset)):
        subject = str(identity.participant_id)
        label = int(identity.label)
        previous = label_by_subject.setdefault(subject, label)
        if previous != label:
            raise ValueError("one participant has inconsistent class labels")
        rows_by_subject.setdefault(subject, []).append(row_index)
    if not rows_by_subject:
        raise ValueError("subject sampling requires a non-empty dataset")
    indices = {
        subject: np.asarray(rows_by_subject[subject], dtype=np.int64)
        for subject in sorted(rows_by_subject)
    }
    subjects_by_class: dict[int, tuple[str, ...]] = {}
    for label in sorted(set(label_by_subject.values())):
        subjects_by_class[label] = tuple(
            subject
            for subject in sorted(indices)
            if label_by_subject[subject] == label
        )
    return indices, label_by_subject, subjects_by_class


def participant_window_sampling_weights(
    dataset: Dataset,
    *,
    class_balanced: bool,
    participant_window_quota: Any = "all",
) -> np.ndarray:
    """Return the expected row mass of the migrated legacy subject sampler.

    ``subject_balanced`` gives each participant mass proportional to its resolved
    quota. ``class_subject_balanced`` additionally gives every participant an
    equal selection chance inside its class, matching the legacy sampler's
    repeated subject-slot construction.  The helper is deterministic and can be
    reused as ``sample_weight`` by non-torch estimators.
    """

    rows_by_subject, label_by_subject, subjects_by_class = _subject_sampling_structure(
        dataset
    )
    row_mass = np.zeros(len(dataset), dtype=np.float64)
    for subject, indices in rows_by_subject.items():
        quota = _participant_quota(len(indices), participant_window_quota)
        if class_balanced:
            class_size = len(subjects_by_class[label_by_subject[subject]])
            subject_mass = float(quota) / float(class_size)
        else:
            subject_mass = float(quota)
        row_mass[indices] = subject_mass / float(len(indices))
    total = float(row_mass.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("subject sampler produced zero or non-finite mass")
    return row_mass / total


def _torch_choice(
    candidates: np.ndarray,
    size: int,
    *,
    replacement: bool,
    generator: torch.Generator,
) -> list[int]:
    """Draw deterministic integer candidates with an explicit torch generator."""

    if size <= 0 or candidates.size <= 0:
        return []
    if replacement:
        positions = torch.randint(
            int(candidates.size),
            (int(size),),
            generator=generator,
        ).cpu().numpy()
    else:
        if size > candidates.size:
            raise ValueError("non-replacement sample exceeds available candidates")
        positions = torch.randperm(
            int(candidates.size), generator=generator
        )[: int(size)].cpu().numpy()
    return [int(candidates[int(position)]) for position in positions]


def subject_epoch_sampling_indices(
    dataset: Dataset,
    *,
    sampler: str,
    participant_window_quota: Any,
    generator: torch.Generator,
) -> tuple[int, ...]:
    """Materialize one legacy-compatible, deterministic subject-sampling epoch."""

    if sampler not in {"subject_balanced", "class_subject_balanced"}:
        raise ValueError("subject_epoch_sampling_indices requires a subject sampler")
    rows_by_subject, _label_by_subject, subjects_by_class = (
        _subject_sampling_structure(dataset)
    )
    selected_subjects: list[str] = []
    if sampler == "subject_balanced":
        subjects = np.asarray(sorted(rows_by_subject), dtype=object)
        order = torch.randperm(int(subjects.size), generator=generator).cpu().numpy()
        selected_subjects = [str(subjects[int(position)]) for position in order]
    else:
        nonempty = [subjects_by_class[label] for label in sorted(subjects_by_class)]
        target_slots = max(len(subjects) for subjects in nonempty)
        for subjects in nonempty:
            candidate_positions = np.arange(len(subjects), dtype=np.int64)
            chosen_positions = _torch_choice(
                candidate_positions,
                target_slots,
                replacement=len(subjects) < target_slots,
                generator=generator,
            )
            selected_subjects.extend(subjects[position] for position in chosen_positions)

    sampled: list[int] = []
    for subject in selected_subjects:
        candidates = rows_by_subject[subject]
        quota = _participant_quota(len(candidates), participant_window_quota)
        sampled.extend(
            _torch_choice(
                candidates,
                quota,
                replacement=quota > len(candidates),
                generator=generator,
            )
        )
    if not sampled:
        raise RuntimeError("subject sampler emitted an empty epoch")
    order = torch.randperm(len(sampled), generator=generator).cpu().numpy()
    return tuple(sampled[int(position)] for position in order)


def configured_row_sampling_weights(
    dataset: Dataset,
    *,
    sampler: str,
    training_balance: str,
    allowed_role_families: tuple[str, ...],
    participant_window_quota: Any = "all",
) -> np.ndarray:
    """Return the row distribution represented by a configured sampler.

    Replacement itself is a loader concern.  This helper exposes the exact row
    probability mass so deep loaders and estimator ``sample_weight`` handling use
    the same algorithm rather than merely sharing a configuration name.
    """

    if sampler == "balance_line_weighted_v2":
        return participant_file_window_sampling_weights(
            dataset,
            training_balance=training_balance,
            allowed_role_families=allowed_role_families,
        )
    if sampler in {
        "uniform_replacement",
        "exhaustive_shuffle_without_replacement",
    }:
        if len(dataset) <= 0:
            raise ValueError("sampling requires a non-empty dataset")
        return np.full(len(dataset), 1.0 / len(dataset), dtype=np.float64)
    if sampler in {"subject_balanced", "class_subject_balanced"}:
        return participant_window_sampling_weights(
            dataset,
            class_balanced=sampler == "class_subject_balanced",
            participant_window_quota=participant_window_quota,
        )
    raise ValueError(f"unsupported sampler: {sampler!r}")


def subset_by_participants(dataset: Dataset, participant_ids: set[str]) -> Subset:
    """Create a deterministic row subset / 创建确定性行子集。"""

    _require_torch()
    indices = [
        index
        for index, participant_id in enumerate(dataset_participant_ids(dataset))
        if participant_id in participant_ids
    ]
    if not indices:
        raise ValueError("participant subset contains no dataset rows")
    return Subset(dataset, indices)


def build_inner_grouped_split(
    outer_train_dataset: Dataset,
    outer: FrozenOuterSplit,
    *,
    n_folds: int,
    seed: int,
    validation_fold_index: int | None = None,
) -> InnerGroupedSplit:
    """Build one deterministic participant-grouped, class-stratified inner fold.

    Only identities and labels already attached to ``outer_train_dataset`` are
    read.  The frozen split first proves that every row belongs to outer train;
    class-wise round-robin assignment then keeps each participant wholly inside
    exactly one of ``n_folds`` validation partitions.
    """

    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError("inner n_folds must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("inner seed must be an integer")
    n_folds = int(n_folds)
    seed = int(seed)
    if n_folds < 2:
        raise ValueError("inner n_folds must be at least two")
    if not 0 <= seed <= 0xFFFF_FFFF:
        raise ValueError("inner seed must be in [0, 2^32-1]")
    outer.assert_training_dataset(outer_train_dataset, exact=True)
    labels_by_participant: dict[str, int] = {}
    for identity in dataset_identities(outer_train_dataset):
        participant = str(identity.participant_id)
        label = int(identity.label)
        previous = labels_by_participant.setdefault(participant, label)
        if previous != label:
            raise ValueError("one participant has inconsistent class labels")
    participants_by_class: dict[int, list[str]] = {}
    for participant, label in labels_by_participant.items():
        participants_by_class.setdefault(label, []).append(participant)
    if len(participants_by_class) < 2:
        raise ValueError("inner stratification requires at least two observed classes")
    smallest_class = min(len(values) for values in participants_by_class.values())
    if n_folds > smallest_class:
        raise ValueError(
            "inner n_folds cannot exceed the smallest outer-train participant class"
        )
    fold_index = (
        (int(outer.repeat) * 1_000 + int(outer.fold)) % n_folds
        if validation_fold_index is None
        else int(validation_fold_index)
    )
    if not 0 <= fold_index < n_folds:
        raise ValueError("inner validation_fold_index is outside n_folds")
    generator = np.random.default_rng(seed)
    validation: list[str] = []
    for label in sorted(participants_by_class):
        participants = np.asarray(
            sorted(participants_by_class[label]),
            dtype=object,
        )
        generator.shuffle(participants)
        validation.extend(
            str(value) for value in participants[fold_index::n_folds]
        )
    validation_ids = tuple(sorted(validation))
    train_ids = tuple(sorted(set(labels_by_participant) - set(validation_ids)))
    split = InnerGroupedSplit(
        train_participant_ids=train_ids,
        validation_participant_ids=validation_ids,
        n_folds=n_folds,
        validation_fold_index=fold_index,
        seed=seed,
    )
    split.validate(outer)
    return split


def forward_batch(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Dispatch a typed batch without representation guessing / 分派类型化批次。"""

    _require_torch()
    if "window_bag" in batch:
        return model(
            batch["window_bag"].to(device),
            batch["window_mask"].to(device),
            batch["file_features"].to(device),
            batch["sample_mask"].to(device),
        )
    x = batch["x"].to(device)
    mask = batch.get("mask")
    return model(x, None if mask is None else mask.to(device))


if nn is not None:

    class TrainingClassificationLoss(nn.Module):
        """Cross-entropy, balanced-softmax and focal-loss strategy module."""

        def __init__(
            self,
            *,
            loss: str,
            class_weight: torch.Tensor | None,
            class_counts: torch.Tensor,
            label_smoothing: float,
            focal_gamma: float,
        ) -> None:
            super().__init__()
            if loss not in TRAINING_LOSSES:
                raise ValueError(f"unsupported loss: {loss!r}")
            self.loss = str(loss)
            self.register_buffer(
                "weight",
                None if class_weight is None else class_weight.detach().clone(),
            )
            self.register_buffer(
                "class_counts",
                torch.clamp(class_counts.detach().clone(), min=1.0),
            )
            self.label_smoothing = float(label_smoothing)
            self.focal_gamma = float(focal_gamma)

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if self.loss == "balanced_softmax":
                adjusted = logits + torch.log(self.class_counts.view(1, -1))
                return nn.functional.cross_entropy(
                    adjusted,
                    target,
                    label_smoothing=self.label_smoothing,
                )
            if self.loss == "focal_loss":
                weighted_ce = nn.functional.cross_entropy(
                    logits,
                    target,
                    weight=self.weight,
                    reduction="none",
                    label_smoothing=self.label_smoothing,
                )
                unweighted_ce = nn.functional.cross_entropy(
                    logits,
                    target,
                    reduction="none",
                )
                probability_true_class = torch.exp(-unweighted_ce)
                return torch.mean(
                    torch.pow(1.0 - probability_true_class, self.focal_gamma)
                    * weighted_ce
                )
            return nn.functional.cross_entropy(
                logits,
                target,
                weight=self.weight,
                label_smoothing=self.label_smoothing,
            )


def _state_hash(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes and bytes / 哈希张量名称、类型、形状与字节。"""

    _require_torch()
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_member_state_hashes(model: nn.Module) -> tuple[str, ...]:
    """Return one state hash per independent member, or one for a single model."""

    members = tuple(model.members) if hasattr(model, "members") else (model,)
    return tuple(_state_hash(member) for member in members)


class UnifiedTrainer:
    """One trainer enforcing frozen membership and outer-label blindness.

    统一训练器，强制冻结成员且对 outer 标签保持不可见。公开 ``fit`` 签名没有
    ``outer_oof_dataset`` 或 ``outer_y`` 参数；epoch 只能固定，或由 outer-train 内部
    的 grouped validation 选择，随后在完整 outer-train 上从头重训。
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = (
            resolve_torch_training_device(
                config.device,
                deterministic_algorithms=config.deterministic_algorithms,
            )
            if torch is not None
            else None
        )

    def _loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        seed_offset: int = 0,
        absolute_seed: int | None = None,
    ) -> DataLoader:
        """Build a seeded deterministic loader / 构造带种子的确定性加载器。"""

        _require_torch()
        generator = torch.Generator()
        generator.manual_seed(
            self.config.seed + seed_offset if absolute_seed is None else int(absolute_seed)
        )
        sampler = None
        if shuffle and self.config.sampler in {
            "balance_line_weighted_v2",
            "uniform_replacement",
        }:
            sampler = WeightedRandomSampler(
                torch.as_tensor(
                    configured_row_sampling_weights(
                        dataset,
                        sampler=self.config.sampler,
                        training_balance=self.config.training_balance,
                        allowed_role_families=self.config.classifier_role_families,
                        participant_window_quota=self.config.participant_window_quota,
                    ),
                    dtype=torch.double,
                ),
                num_samples=(
                    len(dataset)
                    if self.config.samples_per_epoch is None
                    else int(self.config.samples_per_epoch)
                ),
                replacement=True,
                generator=generator,
            )
        elif shuffle and self.config.sampler in {
            "subject_balanced",
            "class_subject_balanced",
        }:
            sampler = list(
                subject_epoch_sampling_indices(
                    dataset,
                    sampler=self.config.sampler,
                    participant_window_quota=self.config.participant_window_quota,
                    generator=generator,
                )
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=collate_samples,
            generator=generator,
        )

    def _set_seed(self, offset: int = 0) -> None:
        """Set Python, NumPy and PyTorch seeds / 设置三类随机种子。"""

        self._set_absolute_seed(self.config.seed + offset)

    def _set_absolute_seed(self, seed: int) -> None:
        """Set all training RNGs to one explicitly archived seed."""

        seed = int(seed)
        if seed < 0:
            raise ValueError("absolute training seed must be non-negative")
        random.seed(seed)
        # NumPy's legacy global RNG accepts uint32 only, while a valid member
        # seed plus the explicit epoch offset may exceed that range.  Preserve
        # the full seed for Python/Torch and archive it, but map NumPy onto its
        # documented state space deterministically.
        np.random.seed(seed % (1 << 32))
        _require_torch()
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(self.config.deterministic_algorithms)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = self.config.deterministic_algorithms
            torch.backends.cudnn.benchmark = not self.config.deterministic_algorithms

    def _criterion(self, dataset: Dataset) -> nn.Module:
        """Build the declared loss entirely from train-only class statistics."""

        weights = configured_class_weight_vector(
            dataset,
            class_weighting=self.config.class_weighting,
            n_classes=self.config.n_classes,
            class_weight_beta=float(getattr(self.config, "class_weight_beta", 0.999)),
            class_count_basis=self.config.class_count_basis,
        )
        torch_weight = (
            None
            if self.config.class_weighting == "none"
            else torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        )
        class_counts = outer_train_class_counts(
            dataset,
            self.config.n_classes,
            class_count_basis=self.config.class_count_basis,
        )
        return TrainingClassificationLoss(
            loss=self.config.loss,
            class_weight=torch_weight,
            class_counts=torch.as_tensor(
                class_counts,
                dtype=torch.float32,
                device=self.device,
            ),
            label_smoothing=float(self.config.label_smoothing),
            focal_gamma=float(getattr(self.config, "focal_gamma", 2.0)),
        )

    def _optimizer(self, member: nn.Module) -> torch.optim.Optimizer:
        """Build the selected optimizer with every resolved parameter applied."""

        optimizer_type = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }[self.config.optimizer]
        parameters = resolve_optimizer_parameters(
            self.config.optimizer,
            getattr(self.config, "optimizer_parameters", {}),
        )
        if "betas" in parameters:
            parameters["betas"] = tuple(parameters["betas"])
        return optimizer_type(
            member.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **parameters,
        )

    def _train_member_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train one independent member for one epoch / 独立训练一个成员一个 epoch。"""

        model.train()
        total_loss, total_rows = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, self.device)
            labels = batch["y"].to(self.device)
            loss = criterion(logits, labels)
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(self.config.gradient_clip_norm)
                )
            optimizer.step()
            total_loss += float(loss.detach()) * labels.shape[0]
            total_rows += int(labels.shape[0])
        return total_loss / max(total_rows, 1)

    def _train_epochs(
        self,
        model: nn.Module,
        dataset: Dataset,
        epochs: int,
        *,
        training_data_scope: str,
    ) -> list[dict[str, bool | float | int | str]]:
        """Train members and archive one descriptive train BA per epoch.

        The balanced-accuracy pass is an endpoint measurement over the complete,
        deterministic training dataset.  It never changes the optimizer,
        checkpoint, selected epoch, or sampler state.
        """

        model.to(self.device)
        history: list[dict[str, bool | float | int | str]] = []
        members = list(model.members) if hasattr(model, "members") else [model]
        member_seeds = tuple(
            int(value) for value in getattr(model, "member_seeds", (self.config.seed,))
        )
        if len(member_seeds) != len(members):
            raise ValueError("model member_seeds must align with independently trained members")
        optimizers = [self._optimizer(member) for member in members]
        criterion = self._criterion(dataset)
        for epoch in range(1, epochs + 1):
            for member_index, (member, optimizer) in enumerate(zip(members, optimizers)):
                member_seed = member_seeds[member_index]
                epoch_seed = member_seed + epoch * 1_000_000
                self._set_absolute_seed(epoch_seed)
                loader = self._loader(dataset, shuffle=True, absolute_seed=epoch_seed)
                loss = self._train_member_epoch(member, loader, optimizer, criterion)
                history.append(
                    {
                        "epoch": epoch,
                        "member": member_index,
                        "training_seed": member_seed,
                        "epoch_rng_seed": epoch_seed,
                        "numpy_epoch_rng_seed": epoch_seed % (1 << 32),
                        "training_loss": float(loss),
                    }
                )
            probability, labels, identities = self._predict_probabilities_with_identities(
                model, dataset
            )
            participant_probability, participant_labels = (
                self._participant_training_predictions(
                    probability,
                    labels,
                    identities,
                    balance_line=self.config.expected_aggregation_rule,
                )
            )
            score = balanced_accuracy_score(
                participant_labels, participant_probability.argmax(axis=1)
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

    def _predict_probabilities_with_identities(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Predict inner probabilities with row-aligned immutable identities.

        预测 inner 概率并同时返回逐行不可变身份。
        """

        model.to(self.device)
        model.eval()
        probabilities, labels, identities = [], [], []
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                if hasattr(model, "predict_probabilities"):
                    if "window_bag" in batch:
                        raise TypeError("ensemble file-bag prediction is not defined")
                    x = batch["x"].to(self.device)
                    mask = batch.get("mask")
                    probability = model.predict_probabilities(
                        x, None if mask is None else mask.to(self.device)
                    )
                else:
                    probability = torch.softmax(forward_batch(model, batch, self.device), dim=-1)
                probabilities.append(probability.cpu().numpy())
                labels.append(batch["y"].numpy())
                identities.extend(batch["identities"])
        return np.concatenate(probabilities), np.concatenate(labels), tuple(identities)

    @staticmethod
    def _validate_row_aligned_probabilities(
        probability: np.ndarray,
        labels: np.ndarray,
        identities: tuple[Any, ...],
        *,
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Validate public prediction output / 校验公开预测结果的逐行对齐与概率语义。"""

        values = np.asarray(probability, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if (
            values.ndim != 2
            or values.shape != (targets.size, int(n_classes))
            or targets.size != len(identities)
        ):
            raise ValueError("probabilities, labels and identities are not row-aligned")
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("predict_probabilities must return finite normalised probabilities")
        return values, targets, tuple(identities)

    def predict_probabilities(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Predict a deep/ensemble/FileBag dataset without fitting.

        对 deep、五成员 ensemble 或 FileBag 数据执行无拟合预测，返回与输入行严格
        对齐的 probabilities、labels 和 identities。
        """

        _require_torch()
        probability, labels, identities = self._predict_probabilities_with_identities(
            model, dataset
        )
        return self._validate_row_aligned_probabilities(
            probability,
            labels,
            identities,
            n_classes=self.config.n_classes,
        )

    def predict_ensemble_members(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Return member and exact-average probabilities with row identities.

        The first result is ``[N,row,class]`` and the second is ``[row,class]``.
        This API lets the runner write N member OOF rows plus one
        independently checked ensemble row without retraining or re-predicting.
        """

        _require_torch()
        if not hasattr(model, "member_probabilities") or not hasattr(model, "member_seeds"):
            raise TypeError("predict_ensemble_members requires a seeded probability ensemble")
        model.to(self.device)
        model.eval()
        member_batches: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        identities: list[Any] = []
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                if "window_bag" in batch:
                    raise TypeError("Inception ensemble does not accept file-bag inputs")
                x = batch["x"].to(self.device)
                mask = batch.get("mask")
                members = model.member_probabilities(
                    x, None if mask is None else mask.to(self.device)
                )
                member_batches.append(members.cpu().numpy())
                labels.append(batch["y"].numpy())
                identities.extend(batch["identities"])
        member_probability = np.concatenate(member_batches, axis=1).astype(np.float64)
        averaged = member_probability.mean(axis=0)
        targets = np.concatenate(labels).astype(np.int64)
        self._validate_row_aligned_probabilities(
            averaged,
            targets,
            tuple(identities),
            n_classes=self.config.n_classes,
        )
        expected_member_count = len(tuple(model.member_seeds))
        if (
            expected_member_count <= 0
            or member_probability.shape[0] != expected_member_count
            or not np.allclose(
                member_probability.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6
            )
        ):
            raise RuntimeError(
                "ensemble member probabilities violate the declared roster contract"
            )
        return member_probability, averaged, targets, tuple(identities)

    def predict_estimator_probabilities(
        self, estimator: Any, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Predict estimator probabilities with the dataset mask when supported.

        通过统一公开入口预测 estimator；若 estimator 声明 mask 参数，则传入
        FeatureMatrix row mask 或 raw sample mask。该方法绝不执行 fit。
        """

        if Subset is not None and isinstance(dataset, Subset):
            raise TypeError("estimator prediction expects a materialised dataset")
        if not hasattr(dataset, "values") or not hasattr(estimator, "predict_proba"):
            raise TypeError("estimator and materialised values are required")
        keywords: dict[str, Any] = {}
        if "mask" in inspect.signature(estimator.predict_proba).parameters:
            if isinstance(dataset, FeatureMatrixDataset):
                keywords["mask"] = dataset.row_mask
            elif hasattr(dataset, "sample_mask"):
                keywords["mask"] = dataset.sample_mask
        probability = estimator.predict_proba(dataset.values, **keywords)
        labels = np.asarray(dataset.labels, dtype=np.int64)
        identities = dataset_identities(dataset)
        return self._validate_row_aligned_probabilities(
            probability,
            labels,
            identities,
            n_classes=self.config.n_classes,
        )

    def _predict_probabilities(
        self, model: nn.Module, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible probability-only helper / 向后兼容的概率辅助函数。"""

        probability, labels, _ = self.predict_probabilities(model, dataset)
        return probability, labels

    @staticmethod
    def _participant_validation_predictions(
        probability: np.ndarray,
        labels: np.ndarray,
        identities: tuple[Any, ...],
        *,
        balance_line: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate inner validation with the configured Line A/Line B rule."""

        return UnifiedTrainer._participant_training_predictions(
            probability,
            labels,
            identities,
            balance_line=balance_line,
        )

    @staticmethod
    def _participant_training_predictions(
        probability: np.ndarray,
        labels: np.ndarray,
        identities: tuple[Any, ...],
        *,
        balance_line: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate train predictions by the case's declared Line A or Line B.

        This helper is deliberately separate from inner-validation aggregation:
        adding a descriptive training metric must not alter historical epoch
        selection semantics.
        """

        if balance_line not in {LINE_A_EQUAL_FILES, LINE_B_EQUAL_ROLE_FAMILIES}:
            raise ValueError("unsupported training balanced-accuracy aggregation line")
        if probability.shape[0] != labels.size or labels.size != len(identities):
            raise ValueError("training predictions, labels and identities are not row-aligned")
        file_groups: dict[tuple[str, str, str], list[int]] = {}
        labels_by_participant: dict[str, int] = {}
        for index, identity in enumerate(identities):
            participant = str(identity.participant_id)
            label = int(labels[index])
            if labels_by_participant.setdefault(participant, label) != label:
                raise ValueError("training participant has inconsistent labels")
            role = canonical_role_family(str(identity.role))
            file_groups.setdefault(
                (participant, role, str(identity.file_id)), []
            ).append(index)

        file_probabilities: dict[tuple[str, str], list[np.ndarray]] = {}
        for (participant, role, _), indices in file_groups.items():
            file_probabilities.setdefault((participant, role), []).append(
                probability[np.asarray(indices, dtype=np.int64)].mean(axis=0)
            )

        participant_groups: dict[str, list[np.ndarray]] = {}
        if balance_line == LINE_A_EQUAL_FILES:
            for (participant, _), values in file_probabilities.items():
                participant_groups.setdefault(participant, []).extend(values)
        else:
            for (participant, _), values in file_probabilities.items():
                participant_groups.setdefault(participant, []).append(
                    np.asarray(values, dtype=np.float64).mean(axis=0)
                )

        participants = sorted(participant_groups)
        if not participants:
            raise ValueError("training balanced accuracy requires participant predictions")
        aggregated = np.asarray(
            [
                np.asarray(participant_groups[participant], dtype=np.float64).mean(axis=0)
                for participant in participants
            ]
        )
        normalizer = aggregated.sum(axis=1, keepdims=True)
        if not np.isfinite(aggregated).all() or np.any(normalizer <= 0.0):
            raise ValueError("training participant probabilities must be finite and positive")
        aggregated /= normalizer
        participant_labels = np.asarray(
            [labels_by_participant[participant] for participant in participants],
            dtype=np.int64,
        )
        return aggregated, participant_labels

    def _select_epoch(
        self, model: nn.Module, inner_train: Dataset, inner_validation: Dataset
    ) -> tuple[int, list[dict[str, bool | float | int | str]]]:
        """Select an epoch using inner labels only / 仅使用 inner 标签选择 epoch。"""

        model.to(self.device)
        members = list(model.members) if hasattr(model, "members") else [model]
        member_seeds = tuple(
            int(value) for value in getattr(model, "member_seeds", (self.config.seed,))
        )
        if len(member_seeds) != len(members):
            raise ValueError("model member_seeds must align with inner-selection members")
        optimizers = [self._optimizer(member) for member in members]
        criterion = self._criterion(inner_train)
        best_epoch, best_score, stale = 1, -np.inf, 0
        history: list[dict[str, bool | float | int | str]] = []
        for epoch in range(1, self.config.maximum_inner_epochs + 1):
            for member_index, (member, optimizer) in enumerate(zip(members, optimizers)):
                member_seed = member_seeds[member_index]
                epoch_seed = member_seed + epoch * 1_000_000
                self._set_absolute_seed(epoch_seed)
                loss = self._train_member_epoch(
                    member,
                    self._loader(
                        inner_train,
                        shuffle=True,
                        absolute_seed=epoch_seed,
                    ),
                    optimizer,
                    criterion,
                )
                history.append(
                    {
                        "epoch": epoch,
                        "member": member_index,
                        "training_seed": member_seed,
                        "epoch_rng_seed": epoch_seed,
                        "numpy_epoch_rng_seed": epoch_seed % (1 << 32),
                        "inner_training_loss": float(loss),
                        "training_data_scope": "inner_train_only",
                        "outer_heldout_used": False,
                    }
                )
            probability, labels, identities = self._predict_probabilities_with_identities(
                model, inner_validation
            )
            participant_probability, participant_labels = (
                self._participant_validation_predictions(
                    probability,
                    labels,
                    identities,
                    balance_line=self.config.expected_aggregation_rule,
                )
            )
            score = balanced_accuracy_score(
                participant_labels, participant_probability.argmax(axis=1)
            )
            if not np.isfinite(score):
                raise RuntimeError("inner participant balanced accuracy is not finite")
            history.append(
                {
                    "epoch": epoch,
                    "inner_balanced_accuracy": float(score),
                    "inner_participant_balanced_accuracy": float(score),
                    "inner_selection_unit": "participant",
                    "inner_selection_aggregation_rule": (
                        self.config.expected_aggregation_rule
                    ),
                    "outer_heldout_used": False,
                }
            )
            if score > best_score + 1e-12:
                best_epoch, best_score, stale = epoch, float(score), 0
            else:
                stale += 1
                if stale >= self.config.inner_patience:
                    break
        return best_epoch, history

    def fit(
        self,
        model_factory: Callable[[], nn.Module],
        outer_train_dataset: Dataset,
        frozen_split: FrozenOuterSplit | FullCohortRefitScope,
        *,
        inner_split: InnerGroupedSplit | None = None,
    ) -> TrainingResult:
        """Fit without accepting or reading outer-OOF labels.

        在不接受也不读取 outer-OOF 标签的前提下拟合。inner 路线选择 epoch 后必定
        重新创建模型并用全部 outer-train 从头训练，防止少用训练参与者。
        """

        _require_torch()
        fitted_ids = dataset_participant_ids(outer_train_dataset)
        frozen_split.assert_training_dataset(outer_train_dataset, exact=True)
        self._set_seed()
        selection_model: nn.Module | None = None
        selection_history: list[dict[str, bool | float | int | str]] = []
        if self.config.epoch_rule == "fixed_epoch":
            if inner_split is not None:
                raise ValueError("fixed_epoch training does not accept an inner split")
            selected_epoch = self.config.fixed_epochs
        else:
            if isinstance(frozen_split, FullCohortRefitScope):
                raise ValueError(
                    "inner epoch selection belongs to an outer fold, not final all-cohort refit"
                )
            if inner_split is None:
                raise ValueError(
                    "inner_grouped_selection requires an explicit train-only inner split"
                )
            inner_split.validate(frozen_split)
            if int(inner_split.n_folds) != int(self.config.inner_grouped_folds):
                raise ValueError("inner split n_folds differs from training configuration")
            if int(inner_split.seed) != int(self.config.seed):
                raise ValueError("inner split seed differs from training configuration")
            inner_train = subset_by_participants(
                outer_train_dataset,
                set(inner_split.train_participant_ids),
            )
            inner_validation = subset_by_participants(
                outer_train_dataset,
                set(inner_split.validation_participant_ids),
            )
            frozen_split.assert_train_only(dataset_participant_ids(inner_train))
            frozen_split.assert_train_only(dataset_participant_ids(inner_validation))
            outer_classes = set(
                int(identity.label) for identity in dataset_identities(outer_train_dataset)
            )
            inner_train_classes = set(
                int(identity.label) for identity in dataset_identities(inner_train)
            )
            inner_validation_classes = set(
                int(identity.label) for identity in dataset_identities(inner_validation)
            )
            if (
                inner_train_classes != outer_classes
                or inner_validation_classes != outer_classes
            ):
                raise ValueError(
                    "inner train and validation must each retain every outer-train class"
                )
            selection_model = model_factory()
            selected_epoch, selection_history = self._select_epoch(
                selection_model,
                inner_train,
                inner_validation,
            )

        # Reapply the orchestration seed and construct a fresh model.  In the
        # inner-selection route the selection object is deliberately discarded;
        # the returned model always sees the complete outer-training roster.
        self._set_seed()
        model = model_factory()
        if selection_model is not None and model is selection_model:
            raise RuntimeError(
                "model_factory must return a fresh model for full outer-train refit"
            )
        # Release the inner model before allocating optimizer state for the
        # full-data refit; large architectures must not remain resident twice.
        selection_model = None
        model.training_balance_ = self.config.training_balance
        model.expected_aggregation_rule_ = self.config.expected_aggregation_rule
        training_data_scope = (
            "full_cohort_refit_all_29"
            if isinstance(frozen_split, FullCohortRefitScope)
            else "full_outer_train_only"
        )
        fit_history = self._train_epochs(
            model,
            outer_train_dataset,
            selected_epoch,
            training_data_scope=training_data_scope,
        )
        member_training_seeds = tuple(
            int(value) for value in getattr(model, "member_seeds", ())
        )
        member_hashes = model_member_state_hashes(model)
        if member_training_seeds and len(member_training_seeds) != len(member_hashes):
            raise RuntimeError("ensemble seed/state provenance is not one-to-one")
        provenance = FittedObjectProvenance(
            object_type=type(model).__name__,
            fitted_participant_ids=tuple(sorted(set(fitted_ids))),
            outer_membership_hash=frozen_split.membership_hash,
            registry_hash=frozen_split.registry_hash,
            fold_hash=frozen_split.fold_hash,
            epoch_rule=self.config.epoch_rule,
            selected_epoch=selected_epoch,
            state_hash=_state_hash(model),
            dataset_binding_hash=dataset_binding_hash(outer_train_dataset),
            training_balance=self.config.training_balance,
            expected_aggregation_rule=self.config.expected_aggregation_rule,
            epoch_profile=self.config.epoch_profile,
            execution_mode=self.config.execution_mode,
            training_seed=int(self.config.seed),
            member_training_seeds=member_training_seeds,
            member_state_hashes=member_hashes,
            optimizer=self.config.optimizer,
            optimizer_parameters=resolve_optimizer_parameters(
                self.config.optimizer,
                getattr(self.config, "optimizer_parameters", {}),
            ),
            learning_rate=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            loss=self.config.loss,
            class_weighting=self.config.class_weighting,
            class_weight_beta=float(getattr(self.config, "class_weight_beta", 0.999)),
            focal_gamma=float(getattr(self.config, "focal_gamma", 2.0)),
            class_count_basis=(
                self.config.class_count_basis
                if self.config.loss == "balanced_softmax"
                or self.config.class_weighting != "none"
                else "archived_train_only_counts_not_used_by_loss"
            ),
            class_counts=tuple(
                float(value)
                for value in outer_train_class_counts(
                    outer_train_dataset,
                    self.config.n_classes,
                    class_count_basis=self.config.class_count_basis,
                )
            ),
            class_weight_vector=tuple(
                float(value)
                for value in configured_class_weight_vector(
                    outer_train_dataset,
                    class_weighting=self.config.class_weighting,
                    n_classes=self.config.n_classes,
                    class_weight_beta=float(
                        getattr(self.config, "class_weight_beta", 0.999)
                    ),
                    class_count_basis=self.config.class_count_basis,
                )
            ),
            class_weight_count_basis=(
                self.config.class_count_basis
                if self.config.class_weighting != "none"
                else "not_applicable_uniform"
            ),
            sampler=self.config.sampler,
            samples_per_epoch=getattr(self.config, "samples_per_epoch", None),
            participant_window_quota=str(
                getattr(self.config, "participant_window_quota", "all")
            ),
            label_smoothing=float(self.config.label_smoothing),
            gradient_clip_norm=self.config.gradient_clip_norm,
            deterministic_algorithms=bool(self.config.deterministic_algorithms),
            cache_policy=str(self.config.cache_policy),
            maximum_inner_epochs=int(self.config.maximum_inner_epochs),
            inner_patience=int(self.config.inner_patience),
            inner_grouped_folds=int(self.config.inner_grouped_folds),
            inner_validation_fold_index=(
                None if inner_split is None else int(inner_split.validation_fold_index)
            ),
            inner_membership_hash=(
                "" if inner_split is None else inner_split.membership_hash
            ),
            inner_train_participant_ids=(
                ()
                if inner_split is None
                else tuple(sorted(inner_split.train_participant_ids))
            ),
            inner_validation_participant_ids=(
                ()
                if inner_split is None
                else tuple(sorted(inner_split.validation_participant_ids))
            ),
            refit_on_all_outer_training=bool(
                self.config.refit_on_all_outer_training
            ),
            notes=(
                "outer_labels_not_accepted_by_fit_api",
                "full_outer_train_refit",
                "training_dataset_identity_bound",
                f"membership_scope={getattr(frozen_split, 'scope_kind', 'frozen_outer_split')}",
                f"sampler={self.config.sampler}",
                f"training_balance={self.config.training_balance}",
                f"expected_aggregation_rule={self.config.expected_aggregation_rule}",
                f"class_weighting={self.config.class_weighting}",
                f"optimizer={self.config.optimizer}",
                f"loss={self.config.loss}",
                f"class_weight_beta={getattr(self.config, 'class_weight_beta', 0.999)}",
                f"focal_gamma={getattr(self.config, 'focal_gamma', 2.0)}",
                f"cache_policy={self.config.cache_policy}",
            )
            + (
                (
                    "epoch_selected_from_outer_train_only_inner_fold",
                    "inner_selection_model_discarded_before_fresh_full_outer_refit",
                    f"inner_membership_hash={inner_split.membership_hash}",
                )
                if inner_split is not None
                else ()
            )
            + (
                (
                    "training_seed_field_is_ensemble_orchestration_seed_only",
                    "member_training_seeds_are_authoritative_for_member_stochasticity",
                )
                if member_training_seeds
                else ()
            )
            + (
                (f"legacy_epoch_rule_alias={self.config.legacy_epoch_rule_alias}",)
                if self.config.legacy_epoch_rule_alias
                else ()
            ),
        )
        return TrainingResult(
            model=model,
            selected_epoch=selected_epoch,
            history=selection_history + fit_history,
            provenance=provenance,
        )

    def fit_estimator(
        self,
        estimator: Any,
        outer_train_dataset: Dataset,
        frozen_split: FrozenOuterSplit | FullCohortRefitScope,
    ) -> TrainingResult:
        """Fit estimators under the same membership guard.

        在相同成员守卫下拟合 estimator。
        """

        self.config.validate_for_execution_backend("estimator")
        fitted_ids = dataset_participant_ids(outer_train_dataset)
        frozen_split.assert_training_dataset(outer_train_dataset, exact=True)
        if Subset is not None and isinstance(outer_train_dataset, Subset):
            raise TypeError("fit_estimator expects a materialised dataset, not Subset")
        if not hasattr(outer_train_dataset, "values"):
            raise TypeError("estimator datasets must expose a values array")
        keywords: dict[str, Any] = {"participant_ids": fitted_ids}
        if "mask" in inspect.signature(estimator.fit).parameters:
            if isinstance(outer_train_dataset, FeatureMatrixDataset):
                keywords["mask"] = outer_train_dataset.row_mask
            elif hasattr(outer_train_dataset, "sample_mask"):
                keywords["mask"] = outer_train_dataset.sample_mask
        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            raise TypeError(
                "estimator must accept sample_weight for the declared balanced training protocol"
            )
        sampling_weight = configured_row_sampling_weights(
            outer_train_dataset,
            sampler=self.config.sampler,
            training_balance=self.config.training_balance,
            allowed_role_families=self.config.classifier_role_families,
            participant_window_quota=self.config.participant_window_quota,
        )
        class_weight = configured_class_weight_vector(
            outer_train_dataset,
            class_weighting=self.config.class_weighting,
            n_classes=self.config.n_classes,
            class_weight_beta=float(self.config.class_weight_beta),
            class_count_basis=self.config.class_count_basis,
        )
        row_labels = np.asarray(outer_train_dataset.labels, dtype=np.int64)
        combined_weight = sampling_weight * class_weight[row_labels]
        if combined_weight.sum() <= 0:
            raise ValueError("combined estimator sample weights have zero mass")
        keywords["sample_weight"] = combined_weight / combined_weight.mean()
        estimator.fit(outer_train_dataset.values, outer_train_dataset.labels, **keywords)
        estimator.training_balance_ = self.config.training_balance
        estimator.expected_aggregation_rule_ = self.config.expected_aggregation_rule
        learned_ids = tuple(getattr(estimator, "fitted_participant_ids_", ()))
        if tuple(sorted(set(learned_ids))) != tuple(sorted(set(fitted_ids))):
            raise RuntimeError("fitted estimator did not preserve exact training provenance")
        expected_ids = tuple(sorted(set(fitted_ids)))
        nested = getattr(estimator, "fitted_object_provenance_", {})
        if not nested:
            raise RuntimeError("fitted estimator did not expose per-object provenance")
        for object_name, object_provenance in nested.items():
            observed_ids = tuple(
                sorted(set(str(value) for value in object_provenance["fitted_participant_ids"]))
            )
            if observed_ids != expected_ids:
                raise RuntimeError(
                    f"fitted object {object_name} has mismatched training provenance"
                )
        # English: Hash the learned estimator state, not merely hyperparameters.
        # 中文：哈希已经学习的 estimator 状态，而不只是超参数。
        state_hash = hashlib.sha256(
            pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest()
        provenance = FittedObjectProvenance(
            object_type=type(estimator).__name__,
            fitted_participant_ids=tuple(sorted(set(fitted_ids))),
            outer_membership_hash=frozen_split.membership_hash,
            registry_hash=frozen_split.registry_hash,
            fold_hash=frozen_split.fold_hash,
            epoch_rule="not_applicable",
            selected_epoch=None,
            state_hash=state_hash,
            dataset_binding_hash=dataset_binding_hash(outer_train_dataset),
            training_balance=self.config.training_balance,
            expected_aggregation_rule=self.config.expected_aggregation_rule,
            epoch_profile="not_applicable",
            execution_mode=self.config.execution_mode,
            training_seed=int(self.config.seed),
            optimizer="not_applicable",
            optimizer_parameters={},
            learning_rate=0.0,
            weight_decay=0.0,
            loss="estimator_native",
            class_weighting=self.config.class_weighting,
            class_weight_beta=float(self.config.class_weight_beta),
            focal_gamma=0.0,
            class_count_basis=(
                self.config.class_count_basis
                if self.config.class_weighting != "none"
                else "not_applicable_uniform"
            ),
            class_counts=tuple(
                float(value)
                for value in outer_train_class_counts(
                    outer_train_dataset,
                    self.config.n_classes,
                    class_count_basis=self.config.class_count_basis,
                )
            ),
            class_weight_vector=tuple(float(value) for value in class_weight),
            class_weight_count_basis=(
                self.config.class_count_basis
                if self.config.class_weighting != "none"
                else "not_applicable_uniform"
            ),
            sampler=self.config.sampler,
            samples_per_epoch=self.config.samples_per_epoch,
            participant_window_quota=str(self.config.participant_window_quota),
            label_smoothing=0.0,
            gradient_clip_norm=None,
            deterministic_algorithms=bool(self.config.deterministic_algorithms),
            cache_policy=str(self.config.cache_policy),
            maximum_inner_epochs=0,
            inner_patience=0,
            inner_grouped_folds=0,
            inner_validation_fold_index=None,
            inner_membership_hash="",
            inner_train_participant_ids=(),
            inner_validation_participant_ids=(),
            refit_on_all_outer_training=bool(
                self.config.refit_on_all_outer_training
            ),
            notes=(
                "all_stateful_transforms_fitted_on_outer_train",
                f"membership_scope={getattr(frozen_split, 'scope_kind', 'frozen_outer_split')}",
                f"sampler={self.config.sampler}",
                f"class_weighting={self.config.class_weighting}",
                f"training_balance={self.config.training_balance}",
                f"expected_aggregation_rule={self.config.expected_aggregation_rule}",
            ),
        )
        return TrainingResult(model=estimator, selected_epoch=None, provenance=provenance)
