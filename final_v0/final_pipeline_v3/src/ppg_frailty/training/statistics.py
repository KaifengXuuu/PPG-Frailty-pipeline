"""Participant-cluster inference and comparison-object report archives for V2."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping
import uuid

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score


DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_PERMUTATION_RESAMPLES = 100_000
_ARCHIVE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ParticipantPrediction:
    """One participant-level OOF probability vector for one repeat."""

    participant_id: str
    label: int
    repeat: int
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not str(self.participant_id).strip() or int(self.repeat) < 0:
            raise ValueError("participant_id must be non-empty and repeat non-negative")
        probability = np.asarray(self.probabilities, dtype=np.float64)
        if probability.ndim != 1 or probability.size < 2:
            raise ValueError("probabilities must be a one-dimensional multiclass vector")
        if not np.isfinite(probability).all() or np.any(probability < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("probabilities must sum to one")


@dataclass(frozen=True)
class ClusterBootstrapResult:
    """Two-sided 95% participant-cluster bootstrap interval."""

    metric: str
    estimate: float
    ci95_lower: float
    ci95_upper: float
    lcb95: float
    n_resamples: int
    seed: int
    n_participants: int
    n_repeats: int
    stratified_by_class: bool = True
    cluster_unit: str = "participant_with_all_repeats"
    interval_method: str = "percentile_two_sided_95"
    lcb_definition: str = "lower_2.5_percentile"


@dataclass(frozen=True)
class PairedPermutationResult:
    """Two-sided paired permutation result with participant as exchange unit."""

    metric: str
    observed_candidate_minus_reference: float
    two_sided_p_value: float
    n_resamples: int
    seed: int
    n_participants: int
    n_repeats: int
    null_mean: float
    null_standard_deviation: float
    exchange_unit: str = "participant_with_all_repeats"


@dataclass(frozen=True)
class HolmResult:
    """One family-wise Holm adjusted comparison."""

    comparison_id: str
    comparison_family: str
    metric: str
    raw_p_value: float
    adjusted_p_value: float
    rank: int
    family_size: int
    alpha: float
    reject_null: bool


@dataclass(frozen=True)
class ConfigMetrics:
    """All decision-facing metrics for one eligible or excluded configuration."""

    config_id: str
    registry_role: str
    participant_mean_balanced_accuracy: float
    participant_mean_macro_f1: float
    worst_fold_balanced_accuracy: float
    balanced_accuracy_lcb95: float
    macro_f1_lcb95: float
    worst_class_recall: float
    worst_class_f1: float
    expected_calibration_error: float
    variability: Mapping[str, float]
    confusion_matrices: Mapping[str, tuple[tuple[float, ...], ...]]
    inference_cost: Mapping[str, float | None]
    parameter_count: int | None
    eligible: bool = True
    exclusion_reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.config_id).strip():
            raise ValueError("config_id must be non-empty")
        if self.registry_role not in {"reference", "ablation", "comparison"}:
            raise ValueError("registry_role must preserve reference/ablation/comparison provenance")
        primary = np.asarray(
            (
                self.participant_mean_balanced_accuracy,
                self.participant_mean_macro_f1,
                self.worst_fold_balanced_accuracy,
                self.balanced_accuracy_lcb95,
                self.macro_f1_lcb95,
                self.worst_class_recall,
                self.worst_class_f1,
                self.expected_calibration_error,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(primary).all() or np.any(primary < 0.0) or np.any(primary > 1.0):
            raise ValueError("classification metrics and LCB95 must be finite in [0,1]")
        if not self.variability or not self.inference_cost or not self.confusion_matrices:
            raise ValueError(
                "variability, inference_cost and confusion_matrices must be explicitly reported"
            )
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in self.variability.values()
        ):
            raise ValueError("variability values must be finite and non-negative")
        measured_costs = tuple(
            value for value in self.inference_cost.values() if value is not None
        )
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in measured_costs
        ):
            raise ValueError("measured inference costs must be finite and non-negative")
        if self.parameter_count is not None and (
            isinstance(self.parameter_count, bool) or int(self.parameter_count) < 0
        ):
            raise ValueError("measured parameter_count must be a non-negative integer")
        operational_missing = self.parameter_count is None or any(
            value is None for value in self.inference_cost.values()
        )
        if operational_missing and self.eligible:
            raise ValueError(
                "missing operational measurements require eligible=false"
            )
        if not self.eligible and not str(self.exclusion_reason).strip():
            raise ValueError("ineligible configurations require exclusion_reason")
        for matrix in self.confusion_matrices.values():
            values = np.asarray(matrix, dtype=np.float64)
            if (
                values.ndim != 2
                or values.shape[0] < 2
                or values.shape[0] != values.shape[1]
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                raise ValueError("confusion matrices must be finite non-negative square matrices")

    @property
    def lcb95(self) -> float:
        """Compatibility read alias; ranking LCB is the BA lower bound."""

        return self.balanced_accuracy_lcb95


@dataclass(frozen=True)
class ManualFinalSelection:
    """One human-selected purpose-specific final while preserving provenance."""

    purpose: str
    config_id: str
    registry_role: str
    rationale: str

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in (self.purpose, self.config_id, self.rationale)):
            raise ValueError("manual final selections require purpose, config_id and rationale")
        if self.registry_role not in {"reference", "ablation", "comparison"}:
            raise ValueError("manual selection must retain the original registry_role")


@dataclass(frozen=True)
class ComparisonArchive:
    """Complete report payload for one comparison object and run."""

    comparison_id: str
    run_id: str
    configs: tuple[ConfigMetrics, ...]
    bootstrap_results: Mapping[str, tuple[ClusterBootstrapResult, ...]]
    paired_permutation_results: Mapping[str, PairedPermutationResult]
    holm_results: tuple[HolmResult, ...]
    selections: tuple[ManualFinalSelection, ...]
    run_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        for value, label in ((self.comparison_id, "comparison_id"), (self.run_id, "run_id")):
            if _ARCHIVE_ID.fullmatch(str(value)) is None:
                raise ValueError(f"{label} contains unsafe path characters")
        if not self.configs:
            raise ValueError("comparison archive requires at least one configuration")
        config_by_id = {item.config_id: item for item in self.configs}
        if len(config_by_id) != len(self.configs):
            raise ValueError("configuration ids must be unique")
        for selection in self.selections:
            item = config_by_id.get(selection.config_id)
            if item is None or item.registry_role != selection.registry_role:
                raise ValueError("selection must reference a config and preserve its registry_role")
        for config in self.configs:
            bootstrap = self.bootstrap_results.get(config.config_id, ())
            by_metric = {result.metric: result for result in bootstrap}
            if set(by_metric) != {"balanced_accuracy", "macro_f1"}:
                raise ValueError(
                    f"{config.config_id} must archive bootstrap results for BA and macro-F1"
                )
            if not np.isclose(
                config.balanced_accuracy_lcb95,
                by_metric["balanced_accuracy"].lcb95,
                rtol=0.0,
                atol=1e-12,
            ) or not np.isclose(
                config.macro_f1_lcb95,
                by_metric["macro_f1"].lcb95,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("ConfigMetrics LCB columns must equal archived bootstrap results")
        holm_groups: dict[tuple[str, str], list[HolmResult]] = {}
        for result in self.holm_results:
            if result.metric not in {"balanced_accuracy", "macro_f1"}:
                raise ValueError("Holm correction is limited to BA and macro-F1")
            if not result.comparison_family.strip():
                raise ValueError("Holm result requires comparison_family")
            holm_groups.setdefault((result.comparison_family, result.metric), []).append(result)
        for values in holm_groups.values():
            if any(value.family_size != len(values) for value in values):
                raise ValueError("Holm family_size must match comparison-family x metric group")
        required = {"independent_test", "fold_protocol", "seeds"}
        missing = sorted(required - set(self.run_manifest))
        if missing:
            raise ValueError(f"run_manifest is missing required fields: {missing}")
        if not isinstance(self.run_manifest["independent_test"], bool):
            raise ValueError("run_manifest independent_test must be boolean")
        if not str(self.run_manifest["fold_protocol"]).strip():
            raise ValueError("run_manifest fold_protocol must be non-empty")
        seeds = tuple(int(value) for value in self.run_manifest["seeds"])
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("run_manifest seeds must be non-empty and unique")
        _jsonable(self.run_manifest)


def _freeze_predictions(
    rows: Iterable[ParticipantPrediction],
    class_order: tuple[int, ...],
) -> tuple[ParticipantPrediction, ...]:
    frozen = tuple(rows)
    if not frozen or len(class_order) < 2 or len(class_order) != len(set(class_order)):
        raise ValueError("predictions and a unique multiclass class_order are required")
    keys: set[tuple[str, int]] = set()
    label_by_participant: dict[str, int] = {}
    repeats_by_participant: dict[str, set[int]] = {}
    for row in frozen:
        if len(row.probabilities) != len(class_order) or row.label not in class_order:
            raise ValueError("prediction width/label differs from class_order")
        key = (row.participant_id, row.repeat)
        if key in keys:
            raise ValueError("each participant may appear only once in each repeat")
        keys.add(key)
        previous = label_by_participant.setdefault(row.participant_id, row.label)
        if previous != row.label:
            raise ValueError("one participant has inconsistent labels")
        repeats_by_participant.setdefault(row.participant_id, set()).add(row.repeat)
    repeat_sets = {tuple(sorted(values)) for values in repeats_by_participant.values()}
    if len(repeat_sets) != 1:
        raise ValueError("every participant must carry the same complete repeat set")
    return frozen


def _arrays(
    rows: tuple[ParticipantPrediction, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    participant = np.asarray([row.participant_id for row in rows], dtype=object)
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    repeats = np.asarray([row.repeat for row in rows], dtype=np.int64)
    probability = np.asarray([row.probabilities for row in rows], dtype=np.float64)
    return participant, labels, repeats, probability


def _mean_repeat_metric(
    labels: np.ndarray,
    repeats: np.ndarray,
    probability: np.ndarray,
    class_order: tuple[int, ...],
    metric: str,
) -> float:
    predicted = np.asarray(class_order, dtype=np.int64)[probability.argmax(axis=1)]
    values: list[float] = []
    for repeat in sorted(np.unique(repeats).tolist()):
        selected = repeats == repeat
        if metric == "balanced_accuracy":
            value = balanced_accuracy_score(labels[selected], predicted[selected])
        elif metric == "macro_f1":
            value = f1_score(
                labels[selected],
                predicted[selected],
                labels=np.asarray(class_order),
                average="macro",
                zero_division=0,
            )
        else:
            raise ValueError("metric must be balanced_accuracy or macro_f1")
        values.append(float(value))
    return float(np.mean(values))


def participant_cluster_bootstrap(
    rows: Iterable[ParticipantPrediction],
    *,
    class_order: tuple[int, ...] = (0, 1, 2),
    metric: str = "balanced_accuracy",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> ClusterBootstrapResult:
    """Resample participants within class and carry every repeat prediction."""

    frozen = _freeze_predictions(rows, class_order)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    participant, labels, repeats, probability = _arrays(frozen)
    unique_participants = sorted(set(participant.tolist()))
    label_by_participant = {
        value: int(labels[np.flatnonzero(participant == value)[0]])
        for value in unique_participants
    }
    strata = {
        label: [value for value in unique_participants if label_by_participant[value] == label]
        for label in class_order
    }
    if any(not values for values in strata.values()):
        raise ValueError("stratified bootstrap requires at least one participant in every class")
    indices_by_participant = {
        value: np.flatnonzero(participant == value) for value in unique_participants
    }
    estimate = _mean_repeat_metric(labels, repeats, probability, class_order, metric)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled: list[str] = []
        for label in class_order:
            values = strata[label]
            sampled.extend(rng.choice(values, size=len(values), replace=True).tolist())
        row_indices = np.concatenate([indices_by_participant[value] for value in sampled])
        draws[index] = _mean_repeat_metric(
            labels[row_indices],
            repeats[row_indices],
            probability[row_indices],
            class_order,
            metric,
        )
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return ClusterBootstrapResult(
        metric=metric,
        estimate=estimate,
        ci95_lower=float(lower),
        ci95_upper=float(upper),
        lcb95=float(lower),
        n_resamples=int(n_resamples),
        seed=int(seed),
        n_participants=len(unique_participants),
        n_repeats=len(set(repeats.tolist())),
    )


def paired_participant_permutation(
    reference: Iterable[ParticipantPrediction],
    candidate: Iterable[ParticipantPrediction],
    *,
    class_order: tuple[int, ...] = (0, 1, 2),
    metric: str = "balanced_accuracy",
    n_resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    seed: int = 42,
) -> PairedPermutationResult:
    """Swap whole participant clusters between paired configurations."""

    ref = _freeze_predictions(reference, class_order)
    cand = _freeze_predictions(candidate, class_order)
    ref_by_key = {(row.participant_id, row.repeat): row for row in ref}
    cand_by_key = {(row.participant_id, row.repeat): row for row in cand}
    if set(ref_by_key) != set(cand_by_key):
        raise ValueError("paired permutation requires identical participant/repeat keys")
    keys = tuple(sorted(ref_by_key))
    if any(ref_by_key[key].label != cand_by_key[key].label for key in keys):
        raise ValueError("paired configurations disagree on participant labels")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    ordered_ref = tuple(ref_by_key[key] for key in keys)
    ordered_cand = tuple(cand_by_key[key] for key in keys)
    participant, labels, repeats, ref_probability = _arrays(ordered_ref)
    _, _, _, cand_probability = _arrays(ordered_cand)
    ref_metric = _mean_repeat_metric(labels, repeats, ref_probability, class_order, metric)
    cand_metric = _mean_repeat_metric(labels, repeats, cand_probability, class_order, metric)
    observed = cand_metric - ref_metric
    unique_participants = np.asarray(sorted(set(participant.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    null = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        swapped = set(
            unique_participants[
                rng.integers(0, 2, size=unique_participants.size, dtype=np.int8).astype(bool)
            ].tolist()
        )
        mask = np.asarray([value in swapped for value in participant], dtype=bool)
        perm_ref = ref_probability.copy()
        perm_cand = cand_probability.copy()
        perm_ref[mask] = cand_probability[mask]
        perm_cand[mask] = ref_probability[mask]
        null[index] = (
            _mean_repeat_metric(labels, repeats, perm_cand, class_order, metric)
            - _mean_repeat_metric(labels, repeats, perm_ref, class_order, metric)
        )
    extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
    p_value = float((extreme + 1) / (n_resamples + 1))
    return PairedPermutationResult(
        metric=metric,
        observed_candidate_minus_reference=float(observed),
        two_sided_p_value=p_value,
        n_resamples=int(n_resamples),
        seed=int(seed),
        n_participants=int(unique_participants.size),
        n_repeats=len(set(repeats.tolist())),
        null_mean=float(null.mean()),
        null_standard_deviation=float(null.std(ddof=0)),
    )


def holm_adjust(
    p_values: Mapping[str, float],
    *,
    comparison_family: str,
    metric: str,
    alpha: float = 0.05,
) -> tuple[HolmResult, ...]:
    """Apply Holm correction within exactly one declared comparison family."""

    if not p_values or not 0.0 < alpha < 1.0:
        raise ValueError("non-empty p_values and alpha in (0,1) are required")
    if not str(comparison_family).strip():
        raise ValueError("comparison_family is required")
    if metric not in {"balanced_accuracy", "macro_f1"}:
        raise ValueError("Holm metric must be balanced_accuracy or macro_f1")
    ordered = sorted((float(value), str(key)) for key, value in p_values.items())
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("p-values must be finite in [0,1]")
    size = len(ordered)
    adjusted_by_id: dict[str, tuple[float, int, float]] = {}
    running = 0.0
    for rank, (raw, comparison_id) in enumerate(ordered, start=1):
        running = max(running, (size - rank + 1) * raw)
        adjusted_by_id[comparison_id] = (min(1.0, running), rank, raw)
    return tuple(
        HolmResult(
            comparison_id=comparison_id,
            comparison_family=str(comparison_family),
            metric=metric,
            raw_p_value=raw,
            adjusted_p_value=adjusted,
            rank=rank,
            family_size=size,
            alpha=float(alpha),
            reject_null=bool(adjusted <= alpha),
        )
        for comparison_id, (adjusted, rank, raw) in sorted(adjusted_by_id.items())
    )


def holm_adjust_by_family_metric(
    p_values: Mapping[tuple[str, str, str], float],
    *,
    alpha: float = 0.05,
) -> tuple[HolmResult, ...]:
    """Apply independent Holm corrections to each family x metric group.

    Input keys are ``(comparison_family, metric, comparison_id)``. This helper
    prevents accidentally combining BA and macro-F1 or unrelated ablation
    families into one multiplicity correction.
    """

    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for (family, metric, comparison_id), value in p_values.items():
        grouped.setdefault((str(family), str(metric)), {})[str(comparison_id)] = float(value)
    if not grouped:
        raise ValueError("at least one family/metric p-value is required")
    return tuple(
        result
        for (family, metric), values in sorted(grouped.items())
        for result in holm_adjust(
            values,
            comparison_family=family,
            metric=metric,
            alpha=alpha,
        )
    )


def _top_label_ece(
    labels: np.ndarray,
    probability: np.ndarray,
    class_order: tuple[int, ...],
    *,
    n_bins: int,
) -> float:
    """Compute pooled top-label ECE over participant-repeat predictions."""

    if int(n_bins) <= 0:
        raise ValueError("ece_bins must be positive")
    classes = np.asarray(class_order, dtype=np.int64)
    predicted = classes[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for index in range(int(n_bins)):
        if index == int(n_bins) - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected):
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(ece)


def build_config_metrics_from_predictions_and_fold_summaries(
    *,
    config_id: str,
    registry_role: str,
    predictions: Iterable[ParticipantPrediction],
    fold_balanced_accuracies: Mapping[str, float],
    fold_confusion_matrices: Mapping[str, Iterable[Iterable[float]]],
    fold_participant_rosters: Mapping[str, Iterable[str]],
    inference_cost: Mapping[str, float | None],
    parameter_count: int | None,
    class_order: tuple[int, ...] = (0, 1, 2),
    n_bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = 42,
    ece_bins: int = 10,
    eligible: bool = True,
    exclusion_reason: str = "",
) -> tuple[ConfigMetrics, tuple[ClusterBootstrapResult, ...]]:
    """Build one complete record without duplicating statistical calculations.

    Missing or invalid operational measurements become None and make the
    configuration ineligible. They are never coerced to zero. Predictive
    metrics are recomputed from participant OOF probabilities and explicit
    fold BA summaries.
    """

    frozen = _freeze_predictions(predictions, class_order)
    if not fold_balanced_accuracies:
        raise ValueError("fold_balanced_accuracies must be explicitly reported")
    expected_fold_keys = {
        f"r{repeat}f{fold}" for repeat in range(5) for fold in range(5)
    }
    if set(fold_balanced_accuracies) != expected_fold_keys:
        raise ValueError("fold_balanced_accuracies requires exact keys r0f0..r4f4")
    if set(fold_confusion_matrices) != expected_fold_keys:
        raise ValueError("fold_confusion_matrices requires exact keys r0f0..r4f4")
    if set(fold_participant_rosters) != expected_fold_keys:
        raise ValueError("fold_participant_rosters requires exact keys r0f0..r4f4")
    fold_values = np.asarray(
        [float(value) for value in fold_balanced_accuracies.values()],
        dtype=np.float64,
    )
    if (
        not np.isfinite(fold_values).all()
        or np.any(fold_values < 0.0)
        or np.any(fold_values > 1.0)
    ):
        raise ValueError("fold balanced accuracies must be finite in [0,1]")

    _, labels, repeats, probability = _arrays(frozen)
    classes = np.asarray(class_order, dtype=np.int64)
    predicted = classes[probability.argmax(axis=1)]
    unique_repeats = tuple(sorted(np.unique(repeats).tolist()))
    repeat_ba = np.asarray(
        [
            balanced_accuracy_score(labels[repeats == repeat], predicted[repeats == repeat])
            for repeat in unique_repeats
        ],
        dtype=np.float64,
    )
    repeat_f1 = np.asarray(
        [
            f1_score(
                labels[repeats == repeat],
                predicted[repeats == repeat],
                labels=classes,
                average="macro",
                zero_division=0,
            )
            for repeat in unique_repeats
        ],
        dtype=np.float64,
    )
    pooled_recall = recall_score(
        labels,
        predicted,
        labels=classes,
        average=None,
        zero_division=0,
    )
    pooled_f1 = f1_score(
        labels,
        predicted,
        labels=classes,
        average=None,
        zero_division=0,
    )
    confusion: dict[str, tuple[tuple[float, ...], ...]] = {
        "pooled_participant_repeat": tuple(
            tuple(float(value) for value in row)
            for row in confusion_matrix(labels, predicted, labels=classes)
        )
    }
    prediction_rosters = {
        repeat: {
            item.participant_id for item in frozen if item.repeat == repeat
        }
        for repeat in range(5)
    }
    prediction_labels = {
        (item.repeat, item.participant_id): int(item.label) for item in frozen
    }
    for repeat in range(5):
        seen: set[str] = set()
        for fold in range(5):
            key = f"r{repeat}f{fold}"
            roster = tuple(str(value) for value in fold_participant_rosters[key])
            if (
                not roster
                or len(roster) != len(set(roster))
                or any(not value.strip() for value in roster)
            ):
                raise ValueError(f"{key} fold participant roster is empty or invalid")
            if seen.intersection(roster):
                raise ValueError(f"{key} overlaps another fold roster in repeat {repeat}")
            seen.update(roster)
            matrix = np.asarray(fold_confusion_matrices[key], dtype=np.float64)
            if (
                matrix.shape != (len(class_order), len(class_order))
                or not np.isfinite(matrix).all()
                or np.any(matrix < 0.0)
            ):
                raise ValueError(f"{key} confusion matrix must be finite non-negative 3x3")
            if not np.isclose(matrix.sum(), len(roster), rtol=0.0, atol=1e-12):
                raise ValueError(f"{key} confusion total does not match held-out roster")
            row_totals = matrix.sum(axis=1)
            expected_row_totals = np.asarray(
                [
                    sum(
                        prediction_labels[(repeat, participant)] == class_value
                        for participant in roster
                    )
                    for class_value in class_order
                ],
                dtype=np.float64,
            )
            if not np.array_equal(row_totals, expected_row_totals):
                raise ValueError(
                    f"{key} confusion row totals differ from held-out roster labels"
                )
            supported = row_totals > 0.0
            matrix_ba = float(
                np.mean(np.diag(matrix)[supported] / row_totals[supported])
            )
            if not np.isclose(
                matrix_ba,
                float(fold_balanced_accuracies[key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{key} balanced accuracy differs from its confusion matrix"
                )
            confusion[key] = tuple(
                tuple(float(value) for value in row) for row in matrix
            )
        if seen != prediction_rosters[repeat]:
            raise ValueError(
                f"repeat {repeat} fold rosters differ from participant OOF predictions"
            )
    for repeat in unique_repeats:
        selected = repeats == repeat
        confusion[f"repeat_{repeat}"] = tuple(
            tuple(float(value) for value in row)
            for row in confusion_matrix(
                labels[selected],
                predicted[selected],
                labels=classes,
            )
        )

    bootstrap_ba = participant_cluster_bootstrap(
        frozen,
        class_order=class_order,
        metric="balanced_accuracy",
        n_resamples=n_bootstrap_resamples,
        seed=bootstrap_seed,
    )
    bootstrap_f1 = participant_cluster_bootstrap(
        frozen,
        class_order=class_order,
        metric="macro_f1",
        n_resamples=n_bootstrap_resamples,
        seed=bootstrap_seed,
    )

    normalized_cost: dict[str, float | None] = {}
    for name, value in inference_cost.items():
        key = str(name).strip()
        if not key:
            raise ValueError("inference_cost names must be non-empty")
        if value is None:
            normalized_cost[key] = None
            continue
        numeric = float(value)
        normalized_cost[key] = (
            numeric if np.isfinite(numeric) and numeric >= 0.0 else None
        )
    if not normalized_cost:
        normalized_cost = {"not_measured": None}

    if (
        parameter_count is None
        or isinstance(parameter_count, bool)
        or int(parameter_count) < 0
    ):
        normalized_parameter_count = None
    else:
        normalized_parameter_count = int(parameter_count)
    operational_missing = normalized_parameter_count is None or any(
        value is None for value in normalized_cost.values()
    )
    final_eligible = bool(eligible) and not operational_missing
    final_reason = str(exclusion_reason).strip()
    if not final_eligible and not final_reason:
        final_reason = (
            "operational_measurements_not_measured"
            if operational_missing
            else "excluded_by_comparison_protocol"
        )

    metrics = ConfigMetrics(
        config_id=str(config_id),
        registry_role=str(registry_role),
        participant_mean_balanced_accuracy=float(repeat_ba.mean()),
        participant_mean_macro_f1=float(repeat_f1.mean()),
        worst_fold_balanced_accuracy=float(fold_values.min()),
        balanced_accuracy_lcb95=bootstrap_ba.lcb95,
        macro_f1_lcb95=bootstrap_f1.lcb95,
        worst_class_recall=float(np.min(pooled_recall)),
        worst_class_f1=float(np.min(pooled_f1)),
        expected_calibration_error=_top_label_ece(
            labels,
            probability,
            class_order,
            n_bins=ece_bins,
        ),
        variability={
            "repeat_balanced_accuracy_population_sd": float(repeat_ba.std(ddof=0)),
            "repeat_macro_f1_population_sd": float(repeat_f1.std(ddof=0)),
            "fold_balanced_accuracy_population_sd": float(fold_values.std(ddof=0)),
        },
        confusion_matrices=confusion,
        inference_cost=normalized_cost,
        parameter_count=normalized_parameter_count,
        eligible=final_eligible,
        exclusion_reason=final_reason,
    )
    return metrics, (bootstrap_ba, bootstrap_f1)


def rank_top10(
    configs: Iterable[ConfigMetrics],
    *,
    limit: int = 10,
) -> tuple[ConfigMetrics, ...]:
    """Return a BA-sorted review list, never an automatically selected winner."""

    if not 1 <= int(limit) <= 10:
        raise ValueError("per-comparison ranking limit must be between 1 and 10")
    eligible = [item for item in configs if item.eligible]
    return tuple(
        sorted(
            eligible,
            key=lambda item: (-item.participant_mean_balanced_accuracy, item.config_id),
        )[: int(limit)]
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("comparison archives forbid NaN and infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"comparison archive value is not strict-JSON compatible: {type(value)}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_comparison_artifact_index(directory: Path) -> None:
    files = tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.name != "artifact_index.json"
        )
    )
    entries = tuple(
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    )
    encoded_entries = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _write_json(
        directory / "artifact_index.json",
        {
            "schema_version": "comparison_artifact_index_v2",
            "overwrite": False,
            "payload_sha256": hashlib.sha256(encoded_entries).hexdigest(),
            "artifacts": entries,
        },
    )


def verify_comparison_archive(path: str | Path) -> Mapping[str, Any]:
    """Verify every indexed byte and reject missing, extra, or tampered files."""

    directory = Path(path)
    index_path = directory / "artifact_index.json"
    if not directory.is_dir() or not index_path.is_file():
        raise ValueError("comparison archive is missing artifact_index.json")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("comparison artifact index is unreadable") from exc
    if (
        not isinstance(index, Mapping)
        or set(index) != {
            "schema_version",
            "overwrite",
            "payload_sha256",
            "artifacts",
        }
        or index["schema_version"] != "comparison_artifact_index_v2"
        or index["overwrite"] is not False
        or not isinstance(index["artifacts"], list)
    ):
        raise ValueError("comparison artifact index schema is invalid")
    entries = index["artifacts"]
    encoded_entries = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if index["payload_sha256"] != hashlib.sha256(encoded_entries).hexdigest():
        raise ValueError("comparison artifact index payload hash mismatch")
    indexed_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("comparison artifact index entry schema is invalid")
        name = str(entry["path"])
        if (
            Path(name).name != name
            or name == "artifact_index.json"
            or name in indexed_names
        ):
            raise ValueError("comparison artifact index contains an unsafe/duplicate path")
        indexed_names.add(name)
        artifact = directory / name
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(entry["bytes"])
            or _file_sha256(artifact) != str(entry["sha256"])
        ):
            raise ValueError(f"comparison artifact failed integrity verification: {name}")
    observed_names = {
        item.name for item in directory.iterdir() if item.is_file()
    }
    if observed_names != indexed_names | {"artifact_index.json"}:
        raise ValueError("comparison archive contains missing or unindexed files")
    return index


def read_verified_manual_selections(
    path: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Read manual selections only after full archive integrity verification."""

    verify_comparison_archive(path)
    payload = json.loads(
        (Path(path) / "selection_record.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, list) or any(not isinstance(item, Mapping) for item in payload):
        raise ValueError("verified selection_record.json must contain a list of mappings")
    return tuple(dict(item) for item in payload)


def _markdown_report(archive: ComparisonArchive, ranking: tuple[ConfigMetrics, ...]) -> str:
    def format_cost(value: float | None) -> str:
        return "not_measured" if value is None else f"{float(value):.6g}"

    lines = [
        f"# Comparison report: {archive.comparison_id}",
        "",
        f"Run: {archive.run_id}",
        "",
        (
            "This is a BA-sorted human-review report; it never selects a winner "
            "automatically. / 本报告仅按 BA 排序供人工审阅，绝不自动选出 winner。"
        ),
        "",
        (
            "| Rank | Config | Role | Mean BA | Worst-fold BA | Macro-F1 | BA LCB95 | Macro-F1 LCB95 | "
            "Worst recall | Worst F1 | ECE | Params | Inference cost |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | {item.config_id} | {item.registry_role} | "
            f"{item.participant_mean_balanced_accuracy:.6f} | "
            f"{item.worst_fold_balanced_accuracy:.6f} | "
            f"{item.participant_mean_macro_f1:.6f} | "
            f"{item.balanced_accuracy_lcb95:.6f} | {item.macro_f1_lcb95:.6f} | "
            f"{item.worst_class_recall:.6f} | {item.worst_class_f1:.6f} | "
            f"{item.expected_calibration_error:.6f} | {item.parameter_count} | "
            f"{', '.join(f'{key}={format_cost(value)}' for key, value in sorted(item.inference_cost.items()))} |"
        )
    lines.extend(
        [
            "",
            f"Bootstrap result groups: {len(archive.bootstrap_results)}",
            f"Paired permutation comparisons: {len(archive.paired_permutation_results)}",
            f"Holm-adjusted comparisons: {len(archive.holm_results)}",
            f"Manual purpose-specific selections: {len(archive.selections)}",
            "",
        ]
    )
    return "\n".join(lines)


def _formal_authority_payload(
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    authority = run_manifest.get("producer_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("formal comparison producer authority missing")
    required = {
        "schema_version", "pipeline_generation", "status", "operation",
        "authority_config_id", "authority_config_hash",
        "formal_execution_gate", "formal_execution_gate_sha256",
        "producer_authority_sha256",
    }
    payload = dict(authority)
    if (
        set(payload) != required
        or payload["schema_version"]
        != "ppg_frailty.formal_comparison_producer_authority.v2"
        or payload["pipeline_generation"] != "final_pipeline_v2"
        or payload["status"]
        != "formal_clean_source_exact_dependency_toc_guarded"
        or payload["operation"] != "formal_benchmark"
    ):
        raise ValueError("formal comparison producer authority schema drift")
    unsigned = {
        key: value for key, value in payload.items()
        if key != "producer_authority_sha256"
    }
    encoded = json.dumps(
        _jsonable(unsigned),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if payload["producer_authority_sha256"] != hashlib.sha256(encoded).hexdigest():
        raise ValueError("formal comparison producer authority seal drift")
    gate = payload["formal_execution_gate"]
    if (
        not isinstance(gate, Mapping)
        or gate.get("formal_execution_gate_evidence_sha256")
        != payload["formal_execution_gate_sha256"]
    ):
        raise ValueError("formal comparison producer gate binding drift")
    return payload


def _write_comparison_archive_impl(
    archive: ComparisonArchive,
    root: str | Path,
    *,
    formal: bool,
    prepublish_guard: Callable[[], object] | None = None,
) -> Path:
    """Atomically archive one comparison object under an explicit authority."""

    ranking = rank_top10(archive.configs)
    run_manifest = dict(archive.run_manifest)
    if formal:
        authority = _formal_authority_payload(run_manifest)
    else:
        authority = {
            "schema_version":
                "ppg_frailty.diagnostic_comparison_producer_authority.v2",
            "pipeline_generation": "final_pipeline_v2",
            "status": "diagnostic_unverified_generic_writer",
            "manual_selection_eligible": False,
        }
        run_manifest["producer_authority"] = authority
    target = Path(root) / archive.comparison_id / archive.run_id
    if target.exists():
        raise FileExistsError(f"comparison archive already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{archive.run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        metrics = tuple(sorted(archive.configs, key=lambda item: item.config_id))
        _write_json(staging / "metrics_all_configs.json", metrics)
        _write_json(staging / "bootstrap_confidence_intervals.json", archive.bootstrap_results)
        _write_json(staging / "paired_permutation_results.json", archive.paired_permutation_results)
        _write_json(staging / "holm_adjustment.json", archive.holm_results)
        _write_json(
            staging / "confusion_matrices.json",
            {item.config_id: item.confusion_matrices for item in metrics},
        )
        _write_json(
            staging / "variability.json",
            {item.config_id: item.variability for item in metrics},
        )
        _write_json(staging / "selection_record.json", archive.selections)
        _write_json(
            staging / "run_manifest.json",
            {
                **run_manifest,
                "comparison_id": archive.comparison_id,
                "run_id": archive.run_id,
                "ranking_rule": "participant_mean_balanced_accuracy_descending",
                "top_n_maximum": 10,
                "automatic_selection": False,
            },
        )
        if formal:
            _write_json(staging / "formal_producer_attestation.json", authority)
        with (staging / "ranking_top10.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "rank",
                    "config_id",
                    "registry_role",
                    "participant_mean_balanced_accuracy",
                    "participant_mean_macro_f1",
                    "worst_fold_balanced_accuracy",
                    "balanced_accuracy_lcb95",
                    "macro_f1_lcb95",
                    "worst_class_recall",
                    "worst_class_f1",
                    "expected_calibration_error",
                    "parameter_count",
                    "inference_cost_json",
                )
            )
            for rank, item in enumerate(ranking, start=1):
                writer.writerow(
                    (
                        rank,
                        item.config_id,
                        item.registry_role,
                        item.participant_mean_balanced_accuracy,
                        item.participant_mean_macro_f1,
                        item.worst_fold_balanced_accuracy,
                        item.balanced_accuracy_lcb95,
                        item.macro_f1_lcb95,
                        item.worst_class_recall,
                        item.worst_class_f1,
                        item.expected_calibration_error,
                        item.parameter_count,
                        json.dumps(
                            _jsonable(item.inference_cost),
                            sort_keys=True,
                            allow_nan=False,
                        ),
                    )
                )
        (staging / "comparison_report.md").write_text(
            _markdown_report(archive, ranking), encoding="utf-8"
        )
        _write_comparison_artifact_index(staging)
        verify_comparison_archive(staging)
        if formal:
            if prepublish_guard is None:
                raise ValueError("formal comparison prepublish guard missing")
            prepublish_guard()
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def write_comparison_archive(
    archive: ComparisonArchive,
    root: str | Path,
) -> Path:
    """Write a diagnostic archive that is never eligible for final selection."""

    return _write_comparison_archive_impl(
        archive,
        root,
        formal=False,
        prepublish_guard=None,
    )


def _write_formal_comparison_archive(
    archive: ComparisonArchive,
    root: str | Path,
    *,
    prepublish_guard: Callable[[], object],
) -> Path:
    """Private source/dependency-gated writer used by the reviewed orchestrator."""

    return _write_comparison_archive_impl(
        archive,
        root,
        formal=True,
        prepublish_guard=prepublish_guard,
    )


__all__ = [
    "ClusterBootstrapResult",
    "ComparisonArchive",
    "ConfigMetrics",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_PERMUTATION_RESAMPLES",
    "HolmResult",
    "ManualFinalSelection",
    "PairedPermutationResult",
    "ParticipantPrediction",
    "build_config_metrics_from_predictions_and_fold_summaries",
    "holm_adjust",
    "holm_adjust_by_family_metric",
    "paired_participant_permutation",
    "participant_cluster_bootstrap",
    "rank_top10",
    "read_verified_manual_selections",
    "verify_comparison_archive",
    "write_comparison_archive",
]
