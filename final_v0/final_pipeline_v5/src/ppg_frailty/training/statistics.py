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
from typing import Any, Iterable, Mapping
import uuid
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_PERMUTATION_RESAMPLES = 100000
PAIRED_PERMUTATION_IMPLEMENTATION_VERSION = "participant_confusion_vectorized_rowwise_rng_v2"
PAIRED_PERMUTATION_RNG_CONTRACT = "numpy_default_rng_pcg64_int8_one_participant_mask_call_per_resample"
CLUSTER_BOOTSTRAP_IMPLEMENTATION_VERSION = "stratified_participant_count_weighted_vectorized_metrics_v2"
CLUSTER_BOOTSTRAP_RNG_CONTRACT = "numpy_default_rng_pcg64_one_choice_per_resample_per_class_in_class_order"
_CLUSTER_BOOTSTRAP_CHUNK_SIZE = 10000
_ARCHIVE_ID = re.compile("^[A-Za-z0-9][A-Za-z0-9_.-]*$")

@dataclass(frozen=True)
class ParticipantPrediction:
    """One participant-level OOF probability vector for one repeat."""

    participant_id: str
    label: int
    repeat: int
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        participant_id = str(self.participant_id).strip()
        if not participant_id or int(self.repeat) < 0:
            raise ValueError("participant_id must be non-empty and repeat non-negative")
        object.__setattr__(self, "participant_id", participant_id)
        probability = np.asarray(self.probabilities, dtype=np.float64)
        if probability.ndim != 1 or probability.size < 2:
            raise ValueError("probabilities must be a one-dimensional multiclass vector")
        if not np.isfinite(probability).all() or np.any(probability < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=1e-06):
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
    valid_resamples: int | None = None
    implementation_version: str = CLUSTER_BOOTSTRAP_IMPLEMENTATION_VERSION
    rng_contract: str = CLUSTER_BOOTSTRAP_RNG_CONTRACT

@dataclass(frozen=True)
class PairedClusterBootstrapResult:
    """Paired percentile interval with participant as the shared resampling unit."""

    metric: str
    observed_candidate_minus_reference: float
    ci95_lower: float
    ci95_upper: float
    n_resamples: int
    seed: int
    n_participants: int
    n_repeats: int
    valid_resamples: int
    stratified_by_class: bool = True
    cluster_unit: str = "participant_with_all_repeats"
    interval_method: str = "percentile_two_sided_95"
    comparison_direction: str = "candidate_minus_reference"
    implementation_version: str = CLUSTER_BOOTSTRAP_IMPLEMENTATION_VERSION
    rng_contract: str = CLUSTER_BOOTSTRAP_RNG_CONTRACT

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
    implementation_version: str = PAIRED_PERMUTATION_IMPLEMENTATION_VERSION
    rng_contract: str = PAIRED_PERMUTATION_RNG_CONTRACT

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
        if self.registry_role not in {"reference", "ablation", "comparison", "optional"}:
            raise ValueError("registry_role must preserve reference/ablation/comparison/optional provenance")
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
            raise ValueError("variability, inference_cost and confusion_matrices must be explicitly reported")
        if any((not np.isfinite(float(value)) or float(value) < 0.0 for value in self.variability.values())):
            raise ValueError("variability values must be finite and non-negative")
        if not self.eligible and (not str(self.exclusion_reason).strip()):
            raise ValueError("ineligible configurations require exclusion_reason")
        for matrix in self.confusion_matrices.values():
            values = np.asarray(matrix, dtype=np.float64)
            square = values.ndim == 2 and values.shape[0] >= 2 and values.shape[0] == values.shape[1]
            if not square or not np.isfinite(values).all() or np.any(values < 0.0):
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
        if not all((str(value).strip() for value in (self.purpose, self.config_id, self.rationale))):
            raise ValueError("manual final selections require purpose, config_id and rationale")
        if self.registry_role not in {"reference", "ablation", "comparison", "optional"}:
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
        required = {"independent_test", "fold_protocol", "seeds"}
        if not required <= set(self.run_manifest):
            raise ValueError(f"run_manifest is missing required fields: {sorted(required - set(self.run_manifest))}")
        _jsonable(self.run_manifest)

def _freeze_predictions(
    rows: Iterable[ParticipantPrediction], class_order: tuple[int, ...]
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

def _arrays(rows: tuple[ParticipantPrediction, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    participant = np.asarray([row.participant_id for row in rows], dtype=object)
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    repeats = np.asarray([row.repeat for row in rows], dtype=np.int64)
    probability = np.asarray([row.probabilities for row in rows], dtype=np.float64)
    return (participant, labels, repeats, probability)

def _mean_repeat_metric(
    labels: np.ndarray, repeats: np.ndarray, probability: np.ndarray, class_order: tuple[int, ...], metric: str
) -> float:
    predicted = np.asarray(class_order, dtype=np.int64)[probability.argmax(axis=1)]
    values: list[float] = []
    for repeat in sorted(np.unique(repeats).tolist()):
        selected = repeats == repeat
        if metric == "balanced_accuracy":
            value = balanced_accuracy_score(labels[selected], predicted[selected])
        elif metric == "macro_f1":
            value = f1_score(
                labels[selected], predicted[selected], labels=np.asarray(class_order), average="macro", zero_division=0
            )
        elif metric == "macro_roc_auc_ovr":
            value = np.mean(
                [
                    roc_auc_score(labels[selected] == class_label, probability[selected, class_index])
                    for class_index, class_label in enumerate(class_order)
                ]
            )
        else:
            raise ValueError("metric must be balanced_accuracy, macro_f1 or macro_roc_auc_ovr")
        values.append(float(value))
    return float(np.mean(values))

def _validate_bootstrap_controls(n_resamples: int, seed: int) -> tuple[int, int]:
    """Normalize deterministic bootstrap controls without accepting booleans."""
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or int(n_resamples) <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return (int(n_resamples), int(seed))

def _participant_repeat_metric_arrays(
    participant: np.ndarray,
    labels: np.ndarray,
    repeats: np.ndarray,
    probability: np.ndarray,
    class_order: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Materialize the complete participant-by-repeat OOF tensor once.

    ``_freeze_predictions`` has already established that every participant has
    exactly one row for every repeat.  The dense layout lets a bootstrap draw
    be represented by one participant-count vector; duplicate participant
    draws therefore carry every repeat without physically duplicating rows.
    """
    unique_participants = np.asarray(sorted(set(participant.tolist())), dtype=object)
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=np.int64)
    participant_position = {value: index for index, value in enumerate(unique_participants.tolist())}
    repeat_position = {int(value): index for index, value in enumerate(repeat_values.tolist())}
    class_position = {int(value): index for index, value in enumerate(class_order)}
    n_participants = int(unique_participants.size)
    n_repeats = int(repeat_values.size)
    n_classes = len(class_order)
    participant_labels = np.full(n_participants, -1, dtype=np.int64)
    probabilities = np.empty((n_participants, n_repeats, n_classes), dtype=np.float64)
    populated = np.zeros((n_participants, n_repeats), dtype=bool)
    for row_index in range(participant.size):
        participant_index = participant_position[str(participant[row_index])]
        repeat_index = repeat_position[int(repeats[row_index])]
        participant_labels[participant_index] = class_position[int(labels[row_index])]
        probabilities[participant_index, repeat_index] = probability[row_index]
        populated[participant_index, repeat_index] = True
    if not populated.all():
        raise ValueError("participant-by-repeat prediction grid is incomplete")
    predicted = probabilities.argmax(axis=-1)
    confusion_flat = np.zeros((n_participants, n_repeats * n_classes * n_classes), dtype=np.float64)
    repeat_offsets = np.arange(n_repeats, dtype=np.int64) * n_classes * n_classes
    for participant_index in range(n_participants):
        cells = repeat_offsets + participant_labels[participant_index] * n_classes + predicted[participant_index]
        confusion_flat[participant_index, cells] = 1.0
    strata_positions = tuple((np.flatnonzero(participant_labels == class_index) for class_index in range(n_classes)))
    if any((values.size == 0 for values in strata_positions)):
        raise ValueError("stratified bootstrap requires at least one participant in every class")
    return (unique_participants, repeat_values, participant_labels, probabilities, strata_positions, confusion_flat)

def _stratified_count_chunks(
    *, n_participants: int, strata_positions: tuple[np.ndarray, ...], n_resamples: int, seed: int
) -> Iterable[np.ndarray]:
    """Yield participant multiplicities while preserving the V1 RNG order.

    There remains exactly one ``Generator.choice`` call for each
    resample-by-class pair, ordered first by resample and then by
    ``class_order``.  Only deterministic metric arithmetic is batched, so
    chunk boundaries cannot change same-seed draws.
    """
    rng = np.random.default_rng(seed)
    processed = 0
    while processed < n_resamples:
        current = min(_CLUSTER_BOOTSTRAP_CHUNK_SIZE, n_resamples - processed)
        counts = np.zeros((current, n_participants), dtype=np.int32)
        for draw_index in range(current):
            for positions in strata_positions:
                sampled = rng.choice(positions, size=int(positions.size), replace=True)
                counts[draw_index] += np.bincount(sampled, minlength=n_participants).astype(np.int32, copy=False)
        yield counts
        processed += current

def _cluster_metric_from_counts(
    counts: np.ndarray,
    *,
    participant_labels: np.ndarray,
    probabilities: np.ndarray,
    confusion_flat: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Evaluate equal-repeat metrics for weighted participant draws.

    For ROC-AUC, participant multiplicities are observation weights.  The
    weighted Mann--Whitney statistic counts a positive/negative score tie as
    one half, exactly matching sklearn's binary ROC-AUC tie convention.
    """
    n_draws = int(counts.shape[0])
    n_repeats = int(probabilities.shape[1])
    n_classes = int(probabilities.shape[2])
    if metric in {"balanced_accuracy", "macro_f1"}:
        confusion = (counts @ confusion_flat).reshape(n_draws, n_repeats, n_classes, n_classes)
        diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
        support = confusion.sum(axis=-1)
        if metric == "balanced_accuracy":
            recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0.0)
            supported = support > 0.0
            per_repeat = np.divide(
                (recall * supported).sum(axis=-1),
                supported.sum(axis=-1),
                out=np.zeros(recall.shape[:-1], dtype=np.float64),
                where=supported.sum(axis=-1) > 0,
            )
        else:
            predicted_support = confusion.sum(axis=-2)
            denominator = support + predicted_support
            class_f1 = np.divide(2.0 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0.0)
            per_repeat = class_f1.mean(axis=-1)
        return per_repeat.mean(axis=-1)
    if metric != "macro_roc_auc_ovr":
        raise ValueError("metric must be balanced_accuracy, macro_f1 or macro_roc_auc_ovr")
    per_repeat_class = np.empty((n_draws, n_repeats, n_classes), dtype=np.float64)
    float_counts = counts.astype(np.float64, copy=False)
    for class_index in range(n_classes):
        positive = np.flatnonzero(participant_labels == class_index)
        negative = np.flatnonzero(participant_labels != class_index)
        positive_weights = float_counts[:, positive]
        negative_weights = float_counts[:, negative]
        denominator = positive_weights.sum(axis=1) * negative_weights.sum(axis=1)
        for repeat_index in range(n_repeats):
            positive_scores = probabilities[positive, repeat_index, class_index]
            negative_scores = probabilities[negative, repeat_index, class_index]
            comparison = (positive_scores[:, None] > negative_scores[None, :]).astype(np.float64) + 0.5 * (
                positive_scores[:, None] == negative_scores[None, :]
            )
            numerator = np.einsum("bi,ij,bj->b", positive_weights, comparison, negative_weights, optimize=True)
            per_repeat_class[:, repeat_index, class_index] = np.divide(
                numerator, denominator, out=np.full(n_draws, np.nan, dtype=np.float64), where=denominator > 0.0
            )
    return per_repeat_class.mean(axis=(1, 2))

def _cluster_bootstrap_draws(
    participant: np.ndarray,
    labels: np.ndarray,
    repeats: np.ndarray,
    probability_sets: tuple[np.ndarray, ...],
    class_order: tuple[int, ...],
    metric: str,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Evaluate one or more models on the same stratified bootstrap draws."""
    prepared = [
        _participant_repeat_metric_arrays(participant, labels, repeats, probability, class_order)
        for probability in probability_sets
    ]
    unique_participants = prepared[0][0]
    participant_labels = prepared[0][2]
    strata_positions = prepared[0][4]
    if any(
        (
            not np.array_equal(item[0], unique_participants)
            or not np.array_equal(item[2], participant_labels)
            or any((not np.array_equal(left, right) for left, right in zip(item[4], strata_positions, strict=True)))
            for item in prepared[1:]
        )
    ):
        raise ValueError("bootstrap model tensors disagree on participant roster")
    outputs = [np.empty(n_resamples, dtype=np.float64) for _ in prepared]
    offset = 0
    for counts in _stratified_count_chunks(
        n_participants=int(unique_participants.size),
        strata_positions=strata_positions,
        n_resamples=n_resamples,
        seed=seed,
    ):
        stop = offset + int(counts.shape[0])
        for output, item in zip(outputs, prepared, strict=True):
            output[offset:stop] = _cluster_metric_from_counts(
                counts, participant_labels=item[2], probabilities=item[3], confusion_flat=item[5], metric=metric
            )
        offset = stop
    return tuple(outputs)

def participant_cluster_bootstrap(
    rows: Iterable[ParticipantPrediction],
    *,
    class_order: tuple[int, ...] = (0, 1, 2),
    metric: str = "balanced_accuracy",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> ClusterBootstrapResult:
    """Resample participants within class and carry every repeat prediction.

    Participant multiplicities are evaluated in vectorised chunks.  This is
    algebraically equivalent to materialising duplicate OOF rows, but avoids
    calling sklearn once per resample.
    """
    frozen = _freeze_predictions(rows, class_order)
    n_resamples, seed = _validate_bootstrap_controls(n_resamples, seed)
    participant, labels, repeats, probability = _arrays(frozen)
    estimate = _mean_repeat_metric(labels, repeats, probability, class_order, metric)
    (draws,) = _cluster_bootstrap_draws(
        participant, labels, repeats, (probability,), class_order, metric, n_resamples, seed
    )
    lower, upper = np.quantile(draws, (0.025, 0.975))
    unique_participants = sorted(set(participant.tolist()))
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
        valid_resamples=int(n_resamples),
    )

def paired_participant_cluster_bootstrap(
    reference: Iterable[ParticipantPrediction],
    candidate: Iterable[ParticipantPrediction],
    *,
    class_order: tuple[int, ...] = (0, 1, 2),
    metric: str = "balanced_accuracy",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> PairedClusterBootstrapResult:
    """Bootstrap a candidate-minus-reference metric with shared clusters.

    Reference and candidate must contain exactly the same participant-by-repeat
    OOF keys and labels. Each stratified participant draw is applied to both
    configurations, and every selected participant contributes all of their
    repeat predictions. The interval therefore preserves both the repeated-CV
    cluster and the within-participant model pairing.
    """
    ref = _freeze_predictions(reference, class_order)
    cand = _freeze_predictions(candidate, class_order)
    ref_by_key = {(row.participant_id, row.repeat): row for row in ref}
    cand_by_key = {(row.participant_id, row.repeat): row for row in cand}
    if set(ref_by_key) != set(cand_by_key):
        raise ValueError("paired bootstrap requires identical participant/repeat keys")
    keys = tuple(sorted(ref_by_key))
    if any((ref_by_key[key].label != cand_by_key[key].label for key in keys)):
        raise ValueError("paired configurations disagree on participant labels")
    n_resamples, seed = _validate_bootstrap_controls(n_resamples, seed)
    ordered_ref = tuple((ref_by_key[key] for key in keys))
    ordered_cand = tuple((cand_by_key[key] for key in keys))
    participant, labels, repeats, ref_probability = _arrays(ordered_ref)
    _, _, _, cand_probability = _arrays(ordered_cand)
    reference_estimate = _mean_repeat_metric(labels, repeats, ref_probability, class_order, metric)
    candidate_estimate = _mean_repeat_metric(labels, repeats, cand_probability, class_order, metric)
    observed = candidate_estimate - reference_estimate
    ref_draws, cand_draws = _cluster_bootstrap_draws(
        participant, labels, repeats, (ref_probability, cand_probability), class_order, metric, n_resamples, seed
    )
    draws = cand_draws - ref_draws
    lower, upper = np.quantile(draws, (0.025, 0.975))
    unique_participants = sorted(set(participant.tolist()))
    return PairedClusterBootstrapResult(
        metric=metric,
        observed_candidate_minus_reference=float(observed),
        ci95_lower=float(lower),
        ci95_upper=float(upper),
        n_resamples=int(n_resamples),
        seed=int(seed),
        n_participants=len(unique_participants),
        n_repeats=len(set(repeats.tolist())),
        valid_resamples=int(n_resamples),
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
    """Swap whole participant clusters between paired configurations.

    The null distribution is evaluated from vectorised repeat-level confusion
    matrices.  This is algebraically identical to rebuilding predictions after
    every participant-cluster swap, but keeps the registered 100,000-resample
    report contract practical for multi-candidate studies.
    """
    ref = _freeze_predictions(reference, class_order)
    cand = _freeze_predictions(candidate, class_order)
    ref_by_key = {(row.participant_id, row.repeat): row for row in ref}
    cand_by_key = {(row.participant_id, row.repeat): row for row in cand}
    if set(ref_by_key) != set(cand_by_key):
        raise ValueError("paired permutation requires identical participant/repeat keys")
    keys = tuple(sorted(ref_by_key))
    if any((ref_by_key[key].label != cand_by_key[key].label for key in keys)):
        raise ValueError("paired configurations disagree on participant labels")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    ordered_ref = tuple((ref_by_key[key] for key in keys))
    ordered_cand = tuple((cand_by_key[key] for key in keys))
    participant, labels, repeats, ref_probability = _arrays(ordered_ref)
    _, _, _, cand_probability = _arrays(ordered_cand)
    ref_metric = _mean_repeat_metric(labels, repeats, ref_probability, class_order, metric)
    cand_metric = _mean_repeat_metric(labels, repeats, cand_probability, class_order, metric)
    observed = cand_metric - ref_metric
    unique_participants = np.asarray(sorted(set(participant.tolist())), dtype=object)
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=np.int64)
    participant_index = {value: index for index, value in enumerate(unique_participants.tolist())}
    repeat_index = {int(value): index for index, value in enumerate(repeat_values.tolist())}
    class_index = {int(value): index for index, value in enumerate(class_order)}
    n_participants = int(unique_participants.size)
    n_repeats = int(repeat_values.size)
    n_classes = len(class_order)
    ref_confusion = np.zeros((n_participants, n_repeats, n_classes, n_classes), dtype=np.float64)
    cand_confusion = np.zeros_like(ref_confusion)
    ref_predictions = ref_probability.argmax(axis=1)
    cand_predictions = cand_probability.argmax(axis=1)
    for row_index in range(len(keys)):
        participant_position = participant_index[str(participant[row_index])]
        repeat_position = repeat_index[int(repeats[row_index])]
        true_position = class_index[int(labels[row_index])]
        ref_confusion[participant_position, repeat_position, true_position, int(ref_predictions[row_index])] += 1.0
        cand_confusion[participant_position, repeat_position, true_position, int(cand_predictions[row_index])] += 1.0
    base_ref = ref_confusion.sum(axis=0)
    base_cand = cand_confusion.sum(axis=0)
    delta_flat = (ref_confusion - cand_confusion).reshape(n_participants, -1)

    def metric_from_confusion(confusion: np.ndarray) -> np.ndarray:
        diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
        support = confusion.sum(axis=-1)
        predicted_count = confusion.sum(axis=-2)
        if metric == "balanced_accuracy":
            recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0.0)
            supported = support > 0.0
            per_repeat = np.divide(
                (recall * supported).sum(axis=-1),
                supported.sum(axis=-1),
                out=np.zeros(recall.shape[:-1], dtype=np.float64),
                where=supported.sum(axis=-1) > 0,
            )
        else:
            denominator = support + predicted_count
            f1 = np.divide(2.0 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0.0)
            per_repeat = f1.mean(axis=-1)
        return per_repeat.mean(axis=-1)

    rng = np.random.default_rng(seed)
    extreme = 0
    null_sum = 0.0
    null_sum_squares = 0.0
    processed = 0
    chunk_size = 10000
    while processed < n_resamples:
        current = min(chunk_size, n_resamples - processed)
        swap_masks = np.empty((current, n_participants), dtype=np.int8)
        for row_index in range(current):
            swap_masks[row_index] = rng.integers(0, 2, size=n_participants, dtype=np.int8)
        swaps = swap_masks.astype(np.float64)
        delta = (swaps @ delta_flat).reshape(current, n_repeats, n_classes, n_classes)
        perm_cand_confusion = base_cand[None, ...] + delta
        perm_ref_confusion = base_ref[None, ...] - delta
        null = metric_from_confusion(perm_cand_confusion) - metric_from_confusion(perm_ref_confusion)
        extreme += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
        null_sum += float(null.sum())
        null_sum_squares += float(np.square(null).sum())
        processed += current
    p_value = float((extreme + 1) / (n_resamples + 1))
    null_mean = null_sum / float(n_resamples)
    null_variance = max(0.0, null_sum_squares / float(n_resamples) - null_mean * null_mean)
    return PairedPermutationResult(
        metric=metric,
        observed_candidate_minus_reference=float(observed),
        two_sided_p_value=p_value,
        n_resamples=int(n_resamples),
        seed=int(seed),
        n_participants=n_participants,
        n_repeats=n_repeats,
        null_mean=float(null_mean),
        null_standard_deviation=float(np.sqrt(null_variance)),
    )

def holm_adjust(
    p_values: Mapping[str, float], *, comparison_family: str, metric: str, alpha: float = 0.05
) -> tuple[HolmResult, ...]:
    """Apply Holm correction within exactly one declared comparison family."""
    if not p_values or not 0.0 < alpha < 1.0:
        raise ValueError("non-empty p_values and alpha in (0,1) are required")
    if not str(comparison_family).strip():
        raise ValueError("comparison_family is required")
    if metric not in {"balanced_accuracy", "macro_f1"}:
        raise ValueError("Holm metric must be balanced_accuracy or macro_f1")
    ordered = sorted(((float(value), str(key)) for key, value in p_values.items()))
    if any((not np.isfinite(value) or not 0.0 <= value <= 1.0 for value, _ in ordered)):
        raise ValueError("p-values must be finite in [0,1]")
    size = len(ordered)
    adjusted_by_id: dict[str, tuple[float, int, float]] = {}
    running = 0.0
    for rank, (raw, comparison_id) in enumerate(ordered, start=1):
        running = max(running, (size - rank + 1) * raw)
        adjusted_by_id[comparison_id] = (min(1.0, running), rank, raw)
    return tuple(
        (
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
    )

def holm_adjust_by_family_metric(
    p_values: Mapping[tuple[str, str, str], float], *, alpha: float = 0.05
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
        (
            result
            for (family, metric), values in sorted(grouped.items())
            for result in holm_adjust(values, comparison_family=family, metric=metric, alpha=alpha)
        )
    )

def _top_label_ece(labels: np.ndarray, probability: np.ndarray, class_order: tuple[int, ...], *, n_bins: int) -> float:
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
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
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
    expected_fold_keys = {f"r{repeat}f{fold}" for repeat in range(5) for fold in range(5)}
    if set(fold_balanced_accuracies) != expected_fold_keys:
        raise ValueError("fold_balanced_accuracies requires exact keys r0f0..r4f4")
    if set(fold_confusion_matrices) != expected_fold_keys:
        raise ValueError("fold_confusion_matrices requires exact keys r0f0..r4f4")
    if set(fold_participant_rosters) != expected_fold_keys:
        raise ValueError("fold_participant_rosters requires exact keys r0f0..r4f4")
    fold_values = np.asarray([float(value) for value in fold_balanced_accuracies.values()], dtype=np.float64)
    if not np.isfinite(fold_values).all() or np.any(fold_values < 0.0) or np.any(fold_values > 1.0):
        raise ValueError("fold balanced accuracies must be finite in [0,1]")
    _, labels, repeats, probability = _arrays(frozen)
    classes = np.asarray(class_order, dtype=np.int64)
    predicted = classes[probability.argmax(axis=1)]
    unique_repeats = tuple(sorted(np.unique(repeats).tolist()))
    repeat_ba = np.asarray(
        [balanced_accuracy_score(labels[repeats == repeat], predicted[repeats == repeat]) for repeat in unique_repeats],
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
    pooled_recall = recall_score(labels, predicted, labels=classes, average=None, zero_division=0)
    pooled_f1 = f1_score(labels, predicted, labels=classes, average=None, zero_division=0)
    confusion: dict[str, tuple[tuple[float, ...], ...]] = {
        "pooled_participant_repeat": tuple(
            (tuple((float(value) for value in row)) for row in confusion_matrix(labels, predicted, labels=classes))
        )
    }
    prediction_rosters = {
        repeat: {item.participant_id for item in frozen if item.repeat == repeat} for repeat in range(5)
    }
    prediction_labels = {(item.repeat, item.participant_id): int(item.label) for item in frozen}
    for repeat in range(5):
        seen: set[str] = set()
        for fold in range(5):
            key = f"r{repeat}f{fold}"
            roster = tuple((str(value) for value in fold_participant_rosters[key]))
            if not roster or len(roster) != len(set(roster)) or any((not value.strip() for value in roster)):
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
                    sum((prediction_labels[repeat, participant] == class_value for participant in roster))
                    for class_value in class_order
                ],
                dtype=np.float64,
            )
            if not np.array_equal(row_totals, expected_row_totals):
                raise ValueError(f"{key} confusion row totals differ from held-out roster labels")
            supported = row_totals > 0.0
            matrix_ba = float(np.mean(np.diag(matrix)[supported] / row_totals[supported]))
            if not np.isclose(matrix_ba, float(fold_balanced_accuracies[key]), rtol=0.0, atol=1e-12):
                raise ValueError(f"{key} balanced accuracy differs from its confusion matrix")
            confusion[key] = tuple((tuple((float(value) for value in row)) for row in matrix))
        if seen != prediction_rosters[repeat]:
            raise ValueError(f"repeat {repeat} fold rosters differ from participant OOF predictions")
    for repeat in unique_repeats:
        selected = repeats == repeat
        confusion[f"repeat_{repeat}"] = tuple(
            (
                tuple((float(value) for value in row))
                for row in confusion_matrix(labels[selected], predicted[selected], labels=classes)
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
        frozen, class_order=class_order, metric="macro_f1", n_resamples=n_bootstrap_resamples, seed=bootstrap_seed
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
        normalized_cost[key] = numeric if np.isfinite(numeric) and numeric >= 0.0 else None
    if not normalized_cost:
        normalized_cost = {"not_measured": None}
    if parameter_count is None or isinstance(parameter_count, bool) or int(parameter_count) < 0:
        normalized_parameter_count = None
    else:
        normalized_parameter_count = int(parameter_count)
    operational_missing = normalized_parameter_count is None or any(
        (value is None for value in normalized_cost.values())
    )
    final_eligible = bool(eligible) and (not operational_missing)
    final_reason = str(exclusion_reason).strip()
    if not final_eligible and (not final_reason):
        final_reason = (
            "operational_measurements_not_measured" if operational_missing else "excluded_by_comparison_protocol"
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
        expected_calibration_error=_top_label_ece(labels, probability, class_order, n_bins=ece_bins),
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
    return (metrics, (bootstrap_ba, bootstrap_f1))

def rank_top10(configs: Iterable[ConfigMetrics], *, limit: int = 10) -> tuple[ConfigMetrics, ...]:
    """Return a BA-sorted review list, never an automatically selected winner."""
    if not 1 <= int(limit) <= 10:
        raise ValueError("per-comparison ranking limit must be between 1 and 10")
    eligible = [item for item in configs if item.eligible]
    return tuple(
        sorted(eligible, key=lambda item: (-item.participant_mean_balanced_accuracy, item.config_id))[: int(limit)]
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
    if isinstance(value, float) and (not np.isfinite(value)):
        raise ValueError("comparison archives forbid NaN and infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"comparison archive value is not strict-JSON compatible: {type(value)}")

def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _write_comparison_artifact_index(directory: Path) -> None:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != "artifact_index.json")
    entries = [{"path": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)} for path in files]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    _write_json(
        directory / "artifact_index.json",
        {
            "schema_version": "comparison_artifact_index_v2",
            "overwrite": False,
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "artifacts": entries,
        },
    )

def verify_comparison_archive(path: str | Path) -> Mapping[str, Any]:
    """Verify every indexed byte using the archive's single integrity index."""
    directory = Path(path)
    index_path = directory / "artifact_index.json"
    if not directory.is_dir() or not index_path.is_file():
        raise ValueError("comparison archive is missing artifact_index.json")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index["artifacts"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("comparison artifact index is unreadable") from exc
    if index.get("schema_version") != "comparison_artifact_index_v2" or index.get("overwrite") is not False:
        raise ValueError("comparison artifact index schema is invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if index.get("payload_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("comparison artifact index payload hash mismatch")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        name = str(entry["path"])
        if Path(name).name != name or name == "artifact_index.json" or name in indexed:
            raise ValueError("comparison artifact index contains an unsafe/duplicate path")
        indexed[name] = entry
        artifact = directory / name
        if (
            not artifact.is_file()
            or artifact.stat().st_size != entry["bytes"]
            or _file_sha256(artifact) != entry["sha256"]
        ):
            raise ValueError(f"comparison artifact failed integrity verification: {name}")
    observed_names = {item.name for item in directory.iterdir() if item.is_file()}
    if observed_names != set(indexed) | {"artifact_index.json"}:
        raise ValueError("comparison archive contains missing or unindexed files")
    return index

def read_verified_manual_selections(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Read manual selections only after full archive integrity verification."""
    verify_comparison_archive(path)
    payload = json.loads((Path(path) / "selection_record.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any((not isinstance(item, Mapping) for item in payload)):
        raise ValueError("verified selection_record.json must contain a list of mappings")
    return tuple((dict(item) for item in payload))

def _markdown_report(archive: ComparisonArchive, ranking: tuple[ConfigMetrics, ...]) -> str:
    lines = [
        f"# Comparison report: {archive.comparison_id}",
        "",
        f"Run: {archive.run_id}",
        "",
        "This BA-sorted report never selects a winner automatically. / 本报告仅按 BA 排序供人工审阅，绝不自动选出 winner。",
        "",
        "| Rank | Config | Role | Mean BA | Worst-fold BA | Macro-F1 | BA LCB95 | Macro-F1 LCB95 | "
        "Worst recall | Worst F1 | ECE | Params | Inference cost |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(ranking, start=1):
        cost = ", ".join(
            f"{key}={'not_measured' if value is None else f'{float(value):.6g}'}"
            for key, value in sorted(item.inference_cost.items())
        )
        lines.append(
            f"| {rank} | {item.config_id} | {item.registry_role} | "
            f"{item.participant_mean_balanced_accuracy:.6f} | {item.worst_fold_balanced_accuracy:.6f} | "
            f"{item.participant_mean_macro_f1:.6f} | {item.balanced_accuracy_lcb95:.6f} | "
            f"{item.macro_f1_lcb95:.6f} | {item.worst_class_recall:.6f} | {item.worst_class_f1:.6f} | "
            f"{item.expected_calibration_error:.6f} | {item.parameter_count} | {cost} |"
        )
    lines += [
        "",
        f"Bootstrap result groups: {len(archive.bootstrap_results)}",
        f"Paired permutation comparisons: {len(archive.paired_permutation_results)}",
        f"Holm-adjusted comparisons: {len(archive.holm_results)}",
        f"Manual purpose-specific selections: {len(archive.selections)}",
        "",
    ]
    return "\n".join(lines)

def _write_comparison_archive_impl(archive: ComparisonArchive, root: str | Path, *, formal: bool) -> Path:
    """Atomically archive one comparison; ``formal`` remains API-compatible."""
    del formal
    ranking = rank_top10(archive.configs)
    run_manifest = dict(archive.run_manifest)
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
        _write_json(staging / "confusion_matrices.json", {item.config_id: item.confusion_matrices for item in metrics})
        _write_json(staging / "variability.json", {item.config_id: item.variability for item in metrics})
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
        with (staging / "ranking_top10.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            metric_columns = (
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
            )
            writer.writerow(("rank", *metric_columns, "inference_cost_json"))
            for rank, item in enumerate(ranking, start=1):
                writer.writerow(
                    (
                        rank,
                        *(getattr(item, name) for name in metric_columns),
                        json.dumps(_jsonable(item.inference_cost), sort_keys=True, allow_nan=False),
                    )
                )
        (staging / "comparison_report.md").write_text(_markdown_report(archive, ranking), encoding="utf-8")
        _write_comparison_artifact_index(staging)
        verify_comparison_archive(staging)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target

def write_comparison_archive(archive: ComparisonArchive, root: str | Path) -> Path:
    """Write a diagnostic archive that is never eligible for final selection."""
    return _write_comparison_archive_impl(archive, root, formal=False)

def _write_formal_comparison_archive(archive: ComparisonArchive, root: str | Path) -> Path:
    """Private writer bound to reference config and source-run identities."""
    return _write_comparison_archive_impl(archive, root, formal=True)


__all__ = (
    "CLUSTER_BOOTSTRAP_IMPLEMENTATION_VERSION CLUSTER_BOOTSTRAP_RNG_CONTRACT ClusterBootstrapResult ComparisonArchive "
    "ConfigMetrics DEFAULT_BOOTSTRAP_RESAMPLES DEFAULT_PERMUTATION_RESAMPLES HolmResult ManualFinalSelection "
    "PairedClusterBootstrapResult PairedPermutationResult ParticipantPrediction "
    "build_config_metrics_from_predictions_and_fold_summaries holm_adjust holm_adjust_by_family_metric "
    "paired_participant_cluster_bootstrap paired_participant_permutation participant_cluster_bootstrap rank_top10 "
    "read_verified_manual_selections verify_comparison_archive write_comparison_archive"
).split()
