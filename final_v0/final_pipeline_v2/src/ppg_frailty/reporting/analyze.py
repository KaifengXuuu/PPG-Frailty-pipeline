"""Auditable study-level metric tables built from real OOF or cell artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..data.schema import CANONICAL_CLASS_NAMES
from ..provenance import stable_payload_sha256
from ..training.aggregation import (
    BALANCE_LINES,
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
    QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW,
    QUALITY_WEIGHT_SOURCE_NONE,
    QUALITY_WEIGHT_SOURCE_ROUTE_FILE,
    aggregate_hierarchy,
)
from ..training.oof import OofPredictionRow
from ..training.evaluator import evaluate_predictions_with_abstentions
from ..training.statistics import (
    ParticipantPrediction,
    holm_adjust_by_family_metric,
    paired_participant_cluster_bootstrap,
    paired_participant_permutation,
    participant_cluster_bootstrap,
)
from .collect import CollectedStudy
from .classification_diagnostics import (
    classification_diagnostic_status_rows,
    classification_per_class_metric_rows,
    classification_roc_curve_rows,
    classification_tsne_rows,
    normalize_classification_rows,
)


WINDOW_BALANCED_TO_PARTICIPANT = "window_balanced_to_participant"
AGGREGATION_REPORT_VIEWS = (
    WINDOW_BALANCED_TO_PARTICIPANT,
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
)

_ABSTENTION_AWARE_METRICS = (
    "abstention_aware_balanced_accuracy",
    "abstention_aware_macro_precision",
    "abstention_aware_macro_recall",
    "abstention_aware_macro_f1",
)


@dataclass(frozen=True)
class _AggregationReplaySemantics:
    resolved_config_balance_line: str
    quality_weighted: bool
    quality_weight_source: str
    resolved_config_path: str


@dataclass(frozen=True)
class _AggregationReplayContext:
    """Runtime-effective source line plus fail-closed replay evidence."""

    source_line: str
    semantics: _AggregationReplaySemantics
    source_participant_rows: tuple[OofPredictionRow, ...]
    selected_cell_validation: str
    source_replay_validation: str


def _aggregation_replay_semantics(
    collected: CollectedStudy,
    case_id: str,
) -> _AggregationReplaySemantics:
    """Resolve report replay controls from exactly one persisted case config."""

    failures = [
        row
        for row in getattr(collected, "resolved_config_failures", ())
        if str(row.get("case_id")) == case_id
    ]
    if failures:
        raise ValueError(
            "resolved config is unavailable: "
            + "; ".join(str(row.get("error", "read failed")) for row in failures)
        )
    matches = [
        row
        for row in getattr(collected, "resolved_aggregation_configs", ())
        if str(row.get("case_id")) == case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "aggregation replay requires exactly one persisted resolved config; "
            f"observed {len(matches)}"
        )
    record = matches[0]
    aggregation = record.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("resolved config has no aggregation mapping")
    balance_line = str(aggregation.get("balance_line", ""))
    if balance_line not in BALANCE_LINES:
        raise ValueError("resolved config has no supported aggregation balance_line")
    quality_weighted = aggregation.get("quality_weighting")
    if not isinstance(quality_weighted, bool):
        raise ValueError("resolved config quality_weighting must be boolean")
    declared_source = aggregation.get("quality_weight_source")
    if declared_source is None:
        # Compatibility for artifacts materialized before the explicit source
        # field existed.  This mirrors the config materializer's historical
        # default while keeping a wholly missing config fail-closed above.
        quality_weight_source = (
            QUALITY_WEIGHT_SOURCE_ROUTE_FILE
            if quality_weighted
            else QUALITY_WEIGHT_SOURCE_NONE
        )
    else:
        quality_weight_source = str(declared_source)
    if quality_weight_source not in {
        QUALITY_WEIGHT_SOURCE_NONE,
        QUALITY_WEIGHT_SOURCE_ROUTE_FILE,
        QUALITY_WEIGHT_SOURCE_LEGACY_WINDOW,
    }:
        raise ValueError("resolved config has an unsupported quality_weight_source")
    if quality_weighted == (quality_weight_source == QUALITY_WEIGHT_SOURCE_NONE):
        raise ValueError(
            "resolved config quality_weighting and quality_weight_source disagree"
        )
    return _AggregationReplaySemantics(
        resolved_config_balance_line=balance_line,
        quality_weighted=quality_weighted,
        quality_weight_source=quality_weight_source,
        resolved_config_path=str(record.get("resolved_config_path", "")),
    )


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for raw in values if (value := _number(raw)) is not None]
    return float(fmean(clean)) if clean else None


def _sum_numbers(values: Iterable[Any]) -> int | None:
    clean = [value for raw in values if (value := _number(raw)) is not None]
    return int(sum(clean)) if clean else None


def _sum_abstention_counts(values: Iterable[Any]) -> list[list[Any]] | None:
    """Normalize and sum persisted per-class abstention count encodings."""

    totals: dict[Any, int] = {}
    for raw in values:
        if isinstance(raw, Mapping):
            items = raw.items()
        elif isinstance(raw, (list, tuple)):
            items = []
            for item in raw:
                if isinstance(item, Mapping):
                    label = item.get("class_label", item.get("label"))
                    count = item.get("abstention_count", item.get("count"))
                    items.append((label, count))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    items.append((item[0], item[1]))
        else:
            continue
        for label, raw_count in items:
            count = _number(raw_count)
            if label is None or count is None:
                continue
            totals[label] = totals.get(label, 0) + int(count)
    if not totals:
        return None
    return [
        [label, totals[label]]
        for label in sorted(totals, key=lambda value: str(value))
    ]


def _sd(values: Iterable[Any]) -> float | None:
    clean = [value for raw in values if (value := _number(raw)) is not None]
    return float(pstdev(clean)) if clean else None


def _lcb95(values: Iterable[Any]) -> float | None:
    clean = np.asarray(
        [value for raw in values if (value := _number(raw)) is not None],
        dtype=np.float64,
    )
    if clean.size < 2:
        return None
    half_width = float(
        student_t.ppf(0.95, df=int(clean.size - 1))
        * clean.std(ddof=1)
        / math.sqrt(clean.size)
    )
    return float(max(0.0, clean.mean() - half_width))


def _descriptive_statistics(values: Iterable[Any]) -> dict[str, Any]:
    """Return repeat-level descriptive statistics with a two-sided t-CI."""

    clean = np.asarray(
        [value for raw in values if (value := _number(raw)) is not None],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_sd": None,
            "population_sd": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci95_margin": None,
            "minimum": None,
            "maximum": None,
        }
    mean = float(clean.mean())
    sample_sd = float(clean.std(ddof=1)) if clean.size >= 2 else None
    margin = (
        float(
            student_t.ppf(0.975, df=int(clean.size - 1))
            * float(sample_sd)
            / math.sqrt(clean.size)
        )
        if sample_sd is not None
        else None
    )
    return {
        "n": int(clean.size),
        "mean": mean,
        "sample_sd": sample_sd,
        "population_sd": float(clean.std(ddof=0)),
        "ci95_low": None if margin is None else float(mean - margin),
        "ci95_high": None if margin is None else float(mean + margin),
        "ci95_margin": margin,
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
    }


def _percent_interval_text(lower: Any, upper: Any) -> str:
    """Return one concise percentage interval for human-facing tables."""

    low = _number(lower)
    high = _number(upper)
    if low is None or high is None:
        return "N/A"
    return f"[{100.0 * low:.1f}, {100.0 * high:.1f}]"


def _participant_prediction_contract(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[ParticipantPrediction, ...],
    dict[tuple[str, int], tuple[int, int, int]],
]:
    """Normalize one complete participant OOF roster for paired inference."""

    predictions: list[ParticipantPrediction] = []
    membership: dict[tuple[str, int], tuple[int, int, int]] = {}
    for row in rows:
        if str(row.get("level", "participant")) != "participant":
            raise ValueError("paired inference received non-participant OOF")
        if not _as_bool(row.get("retained", True)):
            raise ValueError(
                "paired conditional inference is unavailable with abstentions"
            )
        participant_id = str(row.get("participant_id", "")).strip()
        repeat = int(row.get("repeat", -1))
        fold = int(row.get("fold", -1))
        split_seed = int(row.get("split_seed", -1))
        label = int(row.get("label", -1))
        probabilities = tuple(
            float(value) for value in row.get("probabilities", ())
        )
        class_order = tuple(int(value) for value in row.get("class_order", ()))
        if (
            not participant_id
            or repeat < 0
            or fold < 0
            or split_seed < 0
            or class_order != (0, 1, 2)
            or label not in class_order
            or len(probabilities) != len(class_order)
        ):
            raise ValueError("paired inference participant OOF contract is incomplete")
        key = (participant_id, repeat)
        authority = (fold, split_seed, label)
        if key in membership:
            raise ValueError("paired inference participant/repeat roster is duplicated")
        membership[key] = authority
        predictions.append(
            ParticipantPrediction(
                participant_id=participant_id,
                label=label,
                repeat=repeat,
                probabilities=probabilities,
            )
        )
    if not predictions:
        raise ValueError("paired inference requires participant OOF predictions")
    return tuple(predictions), membership


def _participant_cluster_interval_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute absolute BA/F1/ROC-AUC CIs from participant OOF clusters."""

    predictions, _membership = _participant_prediction_contract(rows)
    n_resamples = int(policy["bootstrap_replicates"])
    seed = int(policy["seed"])
    if n_resamples <= 0 or seed < 0:
        raise ValueError("participant-cluster bootstrap controls are invalid")
    output: dict[str, Any] = {}
    for metric in (
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc_ovr",
    ):
        result = participant_cluster_bootstrap(
            predictions,
            metric=metric,
            n_resamples=n_resamples,
            seed=seed,
        )
        prefix = f"participant_cluster_{metric}"
        output.update(
            {
                f"{prefix}_estimate": result.estimate,
                f"{prefix}_ci95_low": result.ci95_lower,
                f"{prefix}_ci95_high": result.ci95_upper,
                f"{prefix}_n_resamples": result.n_resamples,
                f"{prefix}_valid_resamples": result.valid_resamples,
                f"{prefix}_seed": result.seed,
                f"{prefix}_n_participants": result.n_participants,
                f"{prefix}_n_repeats": result.n_repeats,
                f"{prefix}_cluster_unit": result.cluster_unit,
                f"{prefix}_interval_method": result.interval_method,
                f"{prefix}_implementation_version": (
                    result.implementation_version
                ),
                f"{prefix}_rng_contract": result.rng_contract,
            }
        )
    return output


def _paired_participant_inference(
    collected: CollectedStudy,
    *,
    oof_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    case_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare every eligible case with the declared reference using V2 policy."""

    reference_case_id = collected.manifest.get("reference_case_id")
    if reference_case_id in (None, ""):
        return [], []
    reference_case_id = str(reference_case_id)
    candidate_case_ids = sorted(
        set(str(value) for value in case_ids) - {reference_case_id}
    )
    family = (
        str(collected.plan.get("study", {}).get("study_id", "study"))
        + "__declared_reference"
    )

    def unavailable_rows(candidate_case_id: str, reason: str) -> list[dict[str, Any]]:
        comparison_id = f"{candidate_case_id}_vs_{reference_case_id}"
        return [
            {
                "comparison_family": family,
                "comparison_id": comparison_id,
                "reference_case_id": reference_case_id,
                "candidate_case_id": candidate_case_id,
                "metric": metric,
                "candidate_minus_reference": None,
                "participant_cluster_delta_ci95_low": None,
                "participant_cluster_delta_ci95_high": None,
                "raw_two_sided_p_value": None,
                "holm_adjusted_p_value": None,
                "holm_rank": None,
                "holm_family_size": None,
                "alpha": None,
                "reject_null_after_holm": None,
                "participant_count": None,
                "repeat_count": None,
                "test_method": "N/A_incompatible_declared_pair",
                "p_value_applicability": f"N/A_{reason}",
                "comparison_contract_status": f"N/A_{reason}",
                "automatic_selection": False,
            }
            for metric in (
                "balanced_accuracy",
                "macro_f1",
                "macro_roc_auc_ovr",
            )
        ]
    status_by_case = {
        str(row.get("case_id")): str(row.get("status", "unknown"))
        for row in collected.case_records
        if row.get("case_id") is not None
    }
    if status_by_case.get(reference_case_id, "passed") != "passed":
        reason = "declared_reference_case_did_not_pass"
        return (
            [
                row
                for candidate_case_id in candidate_case_ids
                for row in unavailable_rows(candidate_case_id, reason)
            ],
            ["Paired CI/P are N/A because the declared reference case did not pass."],
        )
    policies = {
        str(row.get("case_id")): dict(row["evaluation_statistics"])
        for row in collected.resolved_aggregation_configs
        if isinstance(row.get("evaluation_statistics"), Mapping)
    }
    reference_policy = policies.get(reference_case_id)
    if reference_policy is None:
        reason = "reference_evaluation_statistics_policy_unavailable"
        return (
            [
                row
                for candidate_case_id in candidate_case_ids
                for row in unavailable_rows(candidate_case_id, reason)
            ],
            ["Paired CI/P are N/A because the persisted reference evaluation-statistics policy is unavailable."],
        )
    required_policy = {
        "cluster_unit": "participant_with_all_five_repeat_oof_predictions",
        "paired_exchange_unit": "participant",
        "multiplicity_correction": "holm_within_comparison_family",
        "affects_automatic_selection": False,
    }
    if any(
        reference_policy.get(key) != value
        for key, value in required_policy.items()
    ):
        reason = "reference_policy_is_not_registered_participant_cluster_protocol"
        return (
            [
                row
                for candidate_case_id in candidate_case_ids
                for row in unavailable_rows(candidate_case_id, reason)
            ],
            ["Paired CI/P are N/A because the reference evaluation-statistics policy is not the implemented participant-cluster/Holm protocol."],
        )
    try:
        n_resamples = int(reference_policy["paired_permutation_replicates"])
        bootstrap_resamples = int(
            reference_policy.get("bootstrap_replicates", n_resamples)
        )
        seed = int(reference_policy["seed"])
        reference_predictions, reference_membership = (
            _participant_prediction_contract(oof_by_case.get(reference_case_id, ()))
        )
        raw_expected_repeats = collected.plan.get("execution", {}).get(
            "repeats", ()
        )
        expected_repeats = (
            {int(value) for value in raw_expected_repeats}
            if isinstance(raw_expected_repeats, (list, tuple))
            and raw_expected_repeats
            else {prediction.repeat for prediction in reference_predictions}
        )
        observed_repeats = {
            prediction.repeat for prediction in reference_predictions
        }
        if expected_repeats != observed_repeats:
            raise ValueError("reference declared repeat roster differs from OOF")
        if any(
            len(
                {
                    split_seed
                    for (_participant_id, row_repeat), (
                        _fold,
                        split_seed,
                        _label,
                    ) in reference_membership.items()
                    if row_repeat == repeat
                }
            )
            != 1
            for repeat in expected_repeats
        ):
            raise ValueError("reference repeat contains multiple split seeds")
    except (KeyError, TypeError, ValueError) as error:
        reason = "reference_participant_oof_contract_unavailable"
        return (
            [
                row
                for candidate_case_id in candidate_case_ids
                for row in unavailable_rows(candidate_case_id, reason)
            ],
            [
                "Paired CI/P are N/A for the declared reference: "
                f"{type(error).__name__}: {error}"
            ],
        )
    if n_resamples <= 0 or bootstrap_resamples <= 0:
        reason = "invalid_permutation_or_bootstrap_resample_budget"
        return (
            [
                row
                for candidate_case_id in candidate_case_ids
                for row in unavailable_rows(candidate_case_id, reason)
            ],
            ["Paired inference is N/A because a permutation/bootstrap resample budget is invalid."],
        )
    raw_p_values: dict[tuple[str, str, str], float] = {}
    raw_rows: dict[tuple[str, str], dict[str, Any]] = {}
    unavailable: list[dict[str, Any]] = []
    limitations: list[str] = []
    for candidate_case_id in sorted(set(str(value) for value in case_ids)):
        if candidate_case_id == reference_case_id:
            continue
        if status_by_case.get(candidate_case_id, "passed") != "passed":
            unavailable.extend(
                unavailable_rows(candidate_case_id, "candidate_case_did_not_pass")
            )
            limitations.append(
                f"{candidate_case_id}: paired P values are N/A because the case did not pass."
            )
            continue
        if policies.get(candidate_case_id) != reference_policy:
            unavailable.extend(
                unavailable_rows(
                    candidate_case_id,
                    "candidate_evaluation_statistics_policy_differs_from_reference",
                )
            )
            limitations.append(
                f"{candidate_case_id}: paired P values are N/A because its evaluation-statistics policy differs from the reference."
            )
            continue
        try:
            candidate_predictions, candidate_membership = (
                _participant_prediction_contract(
                    oof_by_case.get(candidate_case_id, ())
                )
            )
            if candidate_membership != reference_membership:
                raise ValueError(
                    "participant/repeat/fold/split-seed/label roster differs"
                )
            comparison_id = f"{candidate_case_id}_vs_{reference_case_id}"
            for metric in (
                "balanced_accuracy",
                "macro_f1",
                "macro_roc_auc_ovr",
            ):
                interval = paired_participant_cluster_bootstrap(
                    reference_predictions,
                    candidate_predictions,
                    metric=metric,
                    n_resamples=bootstrap_resamples,
                    seed=seed,
                )
                permutation = None
                if metric in {"balanced_accuracy", "macro_f1"}:
                    permutation = paired_participant_permutation(
                        reference_predictions,
                        candidate_predictions,
                        metric=metric,
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                    raw_p_values[(family, metric, comparison_id)] = (
                        permutation.two_sided_p_value
                    )
                raw_rows[(comparison_id, metric)] = {
                    "comparison_family": family,
                    "comparison_id": comparison_id,
                    "reference_case_id": reference_case_id,
                    "candidate_case_id": candidate_case_id,
                    "metric": metric,
                    "candidate_minus_reference": (
                        interval.observed_candidate_minus_reference
                    ),
                    "participant_cluster_delta_ci95_low": interval.ci95_lower,
                    "participant_cluster_delta_ci95_high": interval.ci95_upper,
                    "bootstrap_resamples": interval.n_resamples,
                    "bootstrap_valid_resamples": interval.valid_resamples,
                    "bootstrap_seed": interval.seed,
                    "bootstrap_cluster_unit": interval.cluster_unit,
                    "bootstrap_interval_method": interval.interval_method,
                    "bootstrap_implementation_version": (
                        interval.implementation_version
                    ),
                    "bootstrap_rng_contract": interval.rng_contract,
                    "raw_two_sided_p_value": (
                        None
                        if permutation is None
                        else permutation.two_sided_p_value
                    ),
                    "n_resamples": (
                        None if permutation is None else permutation.n_resamples
                    ),
                    "seed": seed,
                    "participant_count": interval.n_participants,
                    "repeat_count": interval.n_repeats,
                    "exchange_unit": (
                        interval.cluster_unit
                        if permutation is None
                        else permutation.exchange_unit
                    ),
                    "test_method": (
                        "paired_participant_cluster_bootstrap_ci_only"
                        if permutation is None
                        else "paired_participant_cluster_bootstrap_and_permutation"
                    ),
                    "permutation_implementation_version": (
                        None
                        if permutation is None
                        else permutation.implementation_version
                    ),
                    "permutation_rng_contract": (
                        None if permutation is None else permutation.rng_contract
                    ),
                    "p_value_applicability": (
                        "N/A_no_registered_roc_auc_permutation_test"
                        if permutation is None
                        else "available_two_sided_participant_cluster_permutation"
                    ),
                    "comparison_contract_status": "matched_complete_roster",
                    "automatic_selection": False,
                }
        except (TypeError, ValueError) as error:
            unavailable.extend(
                unavailable_rows(
                    candidate_case_id,
                    "candidate_reference_matched_oof_contract_unavailable",
                )
            )
            limitations.append(
                f"{candidate_case_id}: paired P values are N/A: "
                f"{type(error).__name__}: {error}"
            )
    adjusted = (
        holm_adjust_by_family_metric(raw_p_values, alpha=0.05)
        if raw_p_values
        else ()
    )
    adjusted_by_key = {
        (row.comparison_id, row.metric): row for row in adjusted
    }
    output: list[dict[str, Any]] = list(unavailable)
    for key, row in sorted(raw_rows.items()):
        holm = adjusted_by_key.get(key)
        output.append(
            {
                **row,
                "holm_adjusted_p_value": (
                    None if holm is None else holm.adjusted_p_value
                ),
                "holm_rank": None if holm is None else holm.rank,
                "holm_family_size": None if holm is None else holm.family_size,
                "alpha": None if holm is None else holm.alpha,
                "reject_null_after_holm": (
                    None if holm is None else holm.reject_null
                ),
                "interpretation": (
                    "paired outer-OOF cluster interval; BA/F1 additionally use Holm-adjusted permutation P; no automatic winner or causal claim"
                ),
            }
        )
    return output, limitations


def _per_class_metric_distributions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "balanced_accuracy_ovr",
        "f1",
        "recall",
        "specificity",
        "roc_auc_ovr",
        "pr_auc_ovr",
    )
    groups: dict[tuple[str, Any, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("case_id", "")),
            row.get("class_label"),
            str(row.get("class_name", "")),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (case_id, class_label, class_name), selected in sorted(
        groups.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        for metric in metrics:
            output.append(
                {
                    "case_id": case_id,
                    "class_label": class_label,
                    "class_name": class_name,
                    "metric": metric,
                    **_descriptive_statistics(
                        row.get(metric) for row in selected
                    ),
                }
            )
    return output


def _class_order(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    for row in rows:
        raw = row.get("class_order")
        if isinstance(raw, (list, tuple)) and raw:
            return tuple(int(value) for value in raw)
    for row in rows:
        probability = row.get("probabilities")
        if isinstance(probability, (list, tuple)) and probability:
            return tuple(range(len(probability)))
    return ()


def _valid_oof(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[int, ...]]:
    retained = [
        row
        for row in rows
        if row.get("retained", True) is not False
        and isinstance(row.get("probabilities"), (list, tuple))
    ]
    order = _class_order(retained)
    clean: list[Mapping[str, Any]] = []
    for row in retained:
        probability = np.asarray(row["probabilities"], dtype=np.float64)
        if (
            probability.shape == (len(order),)
            and np.isfinite(probability).all()
            and np.all(probability >= 0.0)
            and np.isclose(probability.sum(), 1.0, atol=1e-6)
            and int(row.get("label", -1)) in order
        ):
            clean.append(row)
    return clean, order


def _oof_metric_row(
    case_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    repeat: int | None,
) -> dict[str, Any] | None:
    clean, order = _valid_oof(rows)
    if not clean or not order:
        return None
    labels = np.asarray([int(row["label"]) for row in clean], dtype=np.int64)
    probabilities = np.asarray(
        [row["probabilities"] for row in clean], dtype=np.float64
    )
    predictions = np.asarray(
        [order[int(index)] for index in probabilities.argmax(axis=1)],
        dtype=np.int64,
    )
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(order),
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(order))
    total = float(matrix.sum())
    specificity: list[float] = []
    class_balanced_accuracy: list[float] = []
    class_roc_auc: list[float | None] = []
    class_pr_auc: list[float | None] = []
    for index, label in enumerate(order):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        negative_total = true_negative + false_positive
        current_specificity = (
            true_negative / negative_total if negative_total else 0.0
        )
        specificity.append(float(current_specificity))
        class_balanced_accuracy.append(
            float((float(recall[index]) + current_specificity) / 2.0)
        )
        binary_label = labels == int(label)
        if np.unique(binary_label).size < 2:
            class_roc_auc.append(None)
            class_pr_auc.append(None)
        else:
            class_roc_auc.append(
                float(roc_auc_score(binary_label, probabilities[:, index]))
            )
            class_pr_auc.append(
                float(
                    average_precision_score(
                        binary_label, probabilities[:, index]
                    )
                )
            )
    valid_roc = [value for value in class_roc_auc if value is not None]
    valid_pr = [value for value in class_pr_auc if value is not None]
    return {
        "case_id": case_id,
        "repeat": repeat,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(order), average="macro", zero_division=0)
        ),
        "macro_roc_auc_ovr": float(fmean(valid_roc)) if valid_roc else None,
        "macro_pr_auc_ovr": float(fmean(valid_pr)) if valid_pr else None,
        "worst_class_recall": float(np.min(recall)),
        "worst_class_f1": float(np.min(class_f1)),
        "n_predictions": int(labels.size),
        "class_order": list(order),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "case_id": case_id,
                "repeat": repeat,
                "class_label": int(label),
                "class_name": CANONICAL_CLASS_NAMES.get(int(label), str(label)),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "specificity": specificity[index],
                "balanced_accuracy_ovr": class_balanced_accuracy[index],
                "f1": float(class_f1[index]),
                "roc_auc_ovr": class_roc_auc[index],
                "pr_auc_ovr": class_pr_auc[index],
                "support": int(support[index]),
            }
            for index, label in enumerate(order)
        ],
    }


def _oof_abstention_metric_row(
    case_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    repeat: int | None,
) -> dict[str, Any] | None:
    """Recompute complete-roster participant metrics for one repeat or pool."""

    if not rows:
        return None
    class_order = tuple(sorted(int(value) for value in CANONICAL_CLASS_NAMES))
    retained, dropped = [], []
    for row in rows:
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError):
            return None
        if label not in class_order:
            return None
        if not _as_bool(row.get("retained", True)):
            dropped.append(label)
            continue
        try:
            probability = np.asarray(
                row.get("probabilities"), dtype=np.float64
            )
        except (TypeError, ValueError):
            return None
        raw_order = row.get("class_order")
        order = (
            tuple(int(value) for value in raw_order)
            if isinstance(raw_order, (list, tuple)) and raw_order
            else class_order
        )
        if (
            order != class_order
            or probability.shape != (len(class_order),)
            or not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or not np.isclose(probability.sum(), 1.0, atol=1e-6)
        ):
            return None
        retained.append((label, probability))
    probability = np.asarray(
        [value for _, value in retained], dtype=np.float64
    ).reshape((len(retained), len(class_order)))
    metric = evaluate_predictions_with_abstentions(
        np.asarray([label for label, _ in retained], dtype=np.int64),
        probability,
        np.asarray(dropped, dtype=np.int64),
        class_order=class_order,
    )
    return {
        "case_id": case_id,
        "repeat": repeat,
        "abstention_aware_balanced_accuracy": metric.balanced_accuracy,
        "abstention_aware_macro_precision": metric.macro_precision,
        "abstention_aware_macro_recall": metric.macro_recall,
        "abstention_aware_macro_f1": metric.macro_f1,
        "abstention_count": metric.n_abstained,
        "abstention_counts_by_class": metric.abstention_counts_by_class,
        "abstention_aware_per_class": tuple(
            asdict(value) for value in metric.per_class
        ),
        "coverage_rate": metric.coverage_rate,
        "abstention_probability_metrics_scope": metric.probability_metrics_scope,
    }


def _calibration_rows(
    case_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> tuple[list[dict[str, Any]], float | None]:
    clean, order = _valid_oof(rows)
    if not clean:
        return [], None
    probability = np.asarray([row["probabilities"] for row in clean], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in clean], dtype=np.int64)
    prediction_indices = probability.argmax(axis=1)
    predicted = np.asarray([order[int(index)] for index in prediction_indices])
    confidence = probability.max(axis=1)
    correct = predicted == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        left, right = float(edges[index]), float(edges[index + 1])
        mask = (confidence >= left) & (
            confidence <= right if index == bins - 1 else confidence < right
        )
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        fraction = float(count / confidence.size)
        if count:
            ece += fraction * abs(float(accuracy) - float(mean_confidence))
        result.append(
            {
                "case_id": case_id,
                "bin_index": index,
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "fraction": fraction,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return result, float(ece)


def _per_class_from_confusion(
    case_id: str,
    matrix_value: Any,
    *,
    class_order: Sequence[Any] | None = None,
    metric_source: str = "config_metrics_v2_pooled_confusion",
) -> list[dict[str, Any]]:
    matrix = np.asarray(matrix_value, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
    ):
        return []
    order = (
        list(class_order)
        if class_order is not None and len(class_order) == matrix.shape[0]
        else list(range(matrix.shape[0]))
    )
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    class_f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    true_negative = matrix.sum() - true_positive - false_positive - false_negative
    specificity = np.divide(
        true_negative,
        true_negative + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_negative + false_positive) > 0,
    )
    return [
        {
            "case_id": case_id,
            "repeat": None,
            "class_label": order[index],
            "class_name": CANONICAL_CLASS_NAMES.get(
                int(order[index]), str(order[index])
            ),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "true_positive": int(true_positive[index]),
            "false_positive": int(false_positive[index]),
            "true_negative": int(true_negative[index]),
            "false_negative": int(false_negative[index]),
            "predicted_support": int(predicted[index]),
            "observation_count": int(matrix.sum()),
            "specificity": float(specificity[index]),
            "balanced_accuracy_ovr": float(
                (recall[index] + specificity[index]) / 2.0
            ),
            "f1": float(class_f1[index]),
            "roc_auc_ovr": None,
            "pr_auc_ovr": None,
            "support": int(support[index]),
            "metric_source": metric_source,
        }
        for index in range(matrix.shape[0])
    ]


def _cell_repeat_rows(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for cell in cells:
        case_id = str(cell.get("case_id", ""))
        repeat = cell.get("repeat")
        fold = cell.get("fold")
        if repeat is None or fold is None:
            continue
        row = {
            "case_id": case_id,
            "repeat": int(repeat),
            "fold": int(fold),
            "balanced_accuracy": _number(cell.get("balanced_accuracy")),
            "macro_f1": _number(cell.get("macro_f1")),
            **{
                name: _number(cell.get(name))
                for name in _ABSTENTION_AWARE_METRICS
            },
            "abstention_count": _number(cell.get("abstention_count")),
            "abstention_counts_by_class": cell.get(
                "abstention_counts_by_class"
            ),
            "abstention_aware_per_class": cell.get(
                "abstention_aware_per_class"
            ),
            "abstention_probability_metrics_scope": cell.get(
                "abstention_probability_metrics_scope"
            ),
            "coverage_rate": _number(cell.get("coverage_rate")),
            "expected_calibration_error": _number(
                cell.get("expected_calibration_error")
            ),
            "confusion_matrix": cell.get("confusion_matrix"),
            "class_order": cell.get("class_order"),
            "per_class": cell.get("per_class"),
            "status": cell.get("status"),
        }
        fold_rows.append(row)
        groups.setdefault((case_id, int(repeat)), []).append(row)
    repeat_rows = [
        {
            "case_id": case_id,
            "repeat": repeat,
            "balanced_accuracy": _mean(
                row.get("balanced_accuracy") for row in values
            ),
            "macro_f1": _mean(row.get("macro_f1") for row in values),
            **{
                name: _mean(row.get(name) for row in values)
                for name in _ABSTENTION_AWARE_METRICS
            },
            "abstention_count": _sum_numbers(
                row.get("abstention_count") for row in values
            ),
            "abstention_counts_by_class": _sum_abstention_counts(
                row.get("abstention_counts_by_class") for row in values
            ),
            "coverage_rate": _mean(row.get("coverage_rate") for row in values),
            "metric_source": "mean_cell_metrics_fallback",
        }
        for (case_id, repeat), values in sorted(groups.items())
    ]
    return repeat_rows, fold_rows


def _pooled_cell_confusion(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], list[list[float]], int] | None:
    """Pool compatible passed-cell confusion matrices as a labeled fallback."""

    order: list[Any] | None = None
    matrices: list[np.ndarray] = []
    for row in rows:
        if str(row.get("status", "passed")) != "passed":
            continue
        matrix = np.asarray(row.get("confusion_matrix"), dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or matrix.size == 0
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0.0)
        ):
            continue
        raw_order = row.get("class_order")
        current_order = (
            list(raw_order)
            if isinstance(raw_order, (list, tuple))
            and len(raw_order) == matrix.shape[0]
            else list(range(matrix.shape[0]))
        )
        if matrices and (
            current_order != order or matrix.shape != matrices[0].shape
        ):
            return None
        if order is None:
            order = current_order
        matrices.append(matrix)
    if not matrices or order is None:
        return None
    return order, np.sum(matrices, axis=0).tolist(), len(matrices)


def _confusion_long_tables(
    matrices: Sequence[Mapping[str, Any]],
    predictive_ranks: Mapping[str, int],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    counts: list[Mapping[str, Any]] = []
    normalized: list[Mapping[str, Any]] = []
    for row in matrices:
        case_id = str(row.get("case_id", ""))
        matrix = np.asarray(row.get("confusion_matrix"), dtype=np.float64)
        order = list(row.get("class_order", ()))
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] != len(order)
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0.0)
        ):
            continue
        rank = predictive_ranks.get(case_id)
        row_totals = matrix.sum(axis=1)
        for true_index, true_label in enumerate(order):
            for predicted_index, predicted_label in enumerate(order):
                common = {
                    "case_id": case_id,
                    "predictive_rank": rank,
                    "true_class": true_label,
                    "predicted_class": predicted_label,
                    "metric_source": row.get("metric_source"),
                }
                counts.append(
                    {
                        **common,
                        "count": float(matrix[true_index, predicted_index]),
                    }
                )
                normalized.append(
                    {
                        **common,
                        "row_fraction": (
                            float(matrix[true_index, predicted_index] / row_totals[true_index])
                            if row_totals[true_index] > 0.0
                            else None
                        ),
                    }
                )
    return counts, normalized


def _manifest_cases(collected: CollectedStudy) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in collected.manifest.get("cases", ())
        if isinstance(row, Mapping) and row.get("case_id") is not None
    }


def _record_statuses(collected: CollectedStudy) -> dict[str, str]:
    return {
        str(row.get("case_id")): str(row.get("status", "unknown"))
        for row in collected.case_records
    }


def _trusted_by_case(collected: CollectedStudy) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in collected.trusted_config_metrics
        if row.get("case_id") is not None
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "retained"}


def _joined_text(values: Iterable[Any], *, default: str = "not_reported") -> str:
    observed = sorted(
        {
            str(value)
            for value in values
            if value is not None and str(value).strip()
        }
    )
    return " | ".join(observed) if observed else default


def _case_motion_evidence_scope(
    collected: CollectedStudy,
    case_id: str,
) -> tuple[bool | None, str]:
    """Separate auxiliary motion provenance from frailty-label OOF validity."""

    provenances: list[Mapping[str, Any]] = []
    for row in collected.quality_rows:
        if str(row.get("case_id")) != case_id:
            continue
        artifact = row.get("route_artifact")
        if not isinstance(artifact, Mapping):
            continue
        provenance = artifact.get("motion_provenance")
        if isinstance(provenance, Mapping) and _as_bool(
            provenance.get("enabled", False)
        ):
            provenances.append(provenance)
    if not provenances:
        return None, "not_applicable_no_auxiliary_motion_evidence"
    valid = all(
        _as_bool(row.get("valid_outer_oof_claim", False))
        and str(row.get("frailty29_evaluation_relation", ""))
        != "in_sample_for_frailty29"
        for row in provenances
    )
    return (
        valid,
        (
            "outer_oof_auxiliary_motion_evidence"
            if valid
            else "comparison_only_in_sample_auxiliary"
        ),
    )


def _route_role_quality_tables(
    collected: CollectedStudy,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str, str, str, str, str], list[Mapping[str, Any]]] = {}
    component_groups: dict[
        tuple[str, str, str, str], list[tuple[float | None, bool]]
    ] = {}
    role_oof_counts: dict[tuple[str, str], tuple[int, int]] = {}
    for row in collected.role_oof_rows:
        key = (str(row.get("case_id", "")), str(row.get("role", "unknown")))
        total, retained = role_oof_counts.get(key, (0, 0))
        role_oof_counts[key] = (
            total + 1,
            retained + int(_as_bool(row.get("retained", True))),
        )

    has_outer_partition = any(
        str(row.get("outer_partition", "")) in {"outer_train", "outer_oof"}
        for row in collected.quality_rows
    )
    for row in collected.quality_rows:
        if has_outer_partition and str(row.get("outer_partition")) != "outer_oof":
            continue
        artifact = (
            row.get("route_artifact")
            if isinstance(row.get("route_artifact"), Mapping)
            else {}
        )
        case_id = str(row.get("case_id", ""))
        role = str(row.get("role", "unknown"))
        route_state = str(
            artifact.get("state")
            or row.get("route_status")
            or "not_reported"
        )
        signal_route = str(
            row.get("signal_route")
            or artifact.get("source_signal")
            or "not_reported"
        )
        quality_tier = str(
            artifact.get("quality_tier")
            or row.get("quality_tier")
            or "not_reported"
        )
        motion_state = str(
            artifact.get("motion_state")
            or row.get("motion_state")
            or "not_reported"
        )
        groups.setdefault(
            (
                case_id,
                role,
                route_state,
                signal_route,
                quality_tier,
                motion_state,
            ), []
        ).append(row)
        components = (
            row.get("components")
            if isinstance(row.get("components"), Mapping)
            else {}
        )
        non_predictors = (
            components.get("non_predictor_features")
            if isinstance(components.get("non_predictor_features"), Mapping)
            else {}
        )
        for component, raw in non_predictors.items():
            payload = raw if isinstance(raw, Mapping) else {}
            value = _number(payload.get("value"))
            valid = _as_bool(payload.get("valid", value is not None))
            component_groups.setdefault(
                (case_id, role, route_state, str(component)), []
            ).append((value, valid))
        for component, field in (
            ("sqi.direct_q_rate_score", "direct_q_rate_score"),
            ("sqi.direct_q_rate_coverage", "direct_q_rate_coverage"),
            ("sqi.direct_q_morph_score", "direct_q_morph_score"),
            ("sqi.direct_q_morph_coverage", "direct_q_morph_coverage"),
            ("sqi.post_q_rate_score", "post_q_rate_score"),
            ("sqi.post_q_rate_coverage", "post_q_rate_coverage"),
            ("motion.record_probability_diagnostic", "motion_record_probability"),
        ):
            value = _number(artifact.get(field))
            component_groups.setdefault(
                (case_id, role, route_state, component), []
            ).append((value, value is not None))

    coverage: list[Mapping[str, Any]] = []
    for (
        case_id,
        role,
        route_state,
        signal_route,
        quality_tier,
        motion_state,
    ), rows in sorted(groups.items()):
        retained_count = sum(_as_bool(row.get("retained", True)) for row in rows)
        direct_count = sum(
            str(
                row.get("signal_route")
                or (
                    row.get("route_artifact", {}).get("source_signal")
                    if isinstance(row.get("route_artifact"), Mapping)
                    else ""
                )
            ).lower()
            in {"direct", "identity", "x_filter"}
            for row in rows
        )
        processed_count = sum(
            str(
                row.get("signal_route")
                or (
                    row.get("route_artifact", {}).get("source_signal")
                    if isinstance(row.get("route_artifact"), Mapping)
                    else ""
                )
            ).lower()
            in {"artifact_reduced", "rate_only_recovered", "x_ar"}
            for row in rows
        )
        predictor_counts: list[float] = []
        unavailable_counts: list[float] = []
        motion_probabilities: list[float] = []
        motion_thresholds: list[float] = []
        motion_window_counts: list[float] = []
        motion_provenance_rows: list[Mapping[str, Any]] = []
        sqi_states: dict[str, list[Any]] = {
            name: []
            for name in (
                "direct_q_rate_state",
                "direct_q_morph_state",
                "post_q_rate_state",
            )
        }
        sqi_values: dict[str, list[float]] = {
            name: []
            for name in (
                "direct_q_rate_score",
                "direct_q_rate_coverage",
                "direct_q_morph_score",
                "direct_q_morph_coverage",
                "post_q_rate_score",
                "post_q_rate_coverage",
            )
        }
        abstention_reasons: list[Any] = []
        denoiser_ids: list[Any] = []
        denoiser_statuses: list[Any] = []
        abstention_count = 0
        denoiser_attempt_count = 0
        denoiser_success_count = 0
        reducer_failures = 0
        post_q_rate_pass_count = 0
        post_q_rate_recovery_eligible_count = 0
        post_q_rate_recovery_count = 0
        denoiser_requested_cell_count = 0
        denoiser_success_cell_count = 0
        post_q_rate_pass_cell_count = 0
        post_q_rate_recovery_eligible_cell_count = 0
        post_q_rate_recovered_cell_count = 0
        motion_high_cell_count = 0
        motion_low_cell_count = 0
        for row in rows:
            components = (
                row.get("components")
                if isinstance(row.get("components"), Mapping)
                else {}
            )
            availability = (
                components.get("predictor_availability")
                if isinstance(components.get("predictor_availability"), Mapping)
                else {}
            )
            if (value := _number(availability.get("predictor_count"))) is not None:
                predictor_counts.append(value)
            if (
                value := _number(availability.get("unavailable_predictor_count"))
            ) is not None:
                unavailable_counts.append(value)
            artifact = (
                row.get("route_artifact")
                if isinstance(row.get("route_artifact"), Mapping)
                else {}
            )
            for source, target in (
                (artifact.get("motion_record_probability"), motion_probabilities),
                (artifact.get("motion_threshold"), motion_thresholds),
                (artifact.get("motion_window_count"), motion_window_counts),
            ):
                if (value := _number(source)) is not None:
                    target.append(value)
            provenance = artifact.get("motion_provenance")
            if isinstance(provenance, Mapping) and _as_bool(
                provenance.get("enabled", False)
            ):
                motion_provenance_rows.append(provenance)
            for name, values in sqi_states.items():
                if artifact.get(name) is not None:
                    values.append(artifact[name])
            for name, values in sqi_values.items():
                if (value := _number(artifact.get(name))) is not None:
                    values.append(value)
            # Final representation eligibility is authoritative.  The route
            # artifact describes the earlier SQI/motion decision and may have
            # preceded a later feature/window construction failure.
            if not _as_bool(row.get("retained", True)):
                abstention_count += 1
                abstention_reasons.append(
                    row.get("reason") or artifact.get("abstention_reason")
                )
            if _as_bool(artifact.get("denoiser_attempted", False)):
                denoiser_attempt_count += 1
                denoiser_ids.append(artifact.get("denoiser_id"))
                denoiser_status = artifact.get("denoiser_status")
                denoiser_statuses.append(denoiser_status)
                if str(denoiser_status).lower() == "success":
                    denoiser_success_count += 1
                direct_q_rate_state = str(
                    artifact.get("direct_q_rate_state") or ""
                ).lower()
                post_q_rate_state = str(
                    artifact.get("post_q_rate_state") or ""
                ).lower()
                if post_q_rate_state == "pass":
                    post_q_rate_pass_count += 1
                if direct_q_rate_state and direct_q_rate_state != "pass":
                    post_q_rate_recovery_eligible_count += 1
                    if post_q_rate_state == "pass":
                        post_q_rate_recovery_count += 1
                reducer_status = str(
                    denoiser_status
                    or artifact.get("reducer_status")
                    or artifact.get("artifact_reducer_status")
                    or ""
                ).lower()
                if reducer_status != "success":
                    reducer_failures += 1
            denoiser_requested_cell_count += int(
                _number(artifact.get("denoiser_requested_cell_count")) or 0
            )
            denoiser_success_cell_count += int(
                _number(artifact.get("denoiser_success_cell_count")) or 0
            )
            post_q_rate_pass_cell_count += int(
                _number(artifact.get("post_q_rate_pass_cell_count")) or 0
            )
            post_q_rate_recovery_eligible_cell_count += int(
                _number(
                    artifact.get("post_q_rate_recovery_eligible_cell_count")
                )
                or 0
            )
            post_q_rate_recovered_cell_count += int(
                _number(artifact.get("post_q_rate_recovered_cell_count")) or 0
            )
            motion_high_cell_count += int(
                _number(artifact.get("motion_high_cell_count")) or 0
            )
            motion_low_cell_count += int(
                _number(artifact.get("motion_low_cell_count")) or 0
            )
        total_predictors = float(sum(predictor_counts))
        total_unavailable = float(sum(unavailable_counts))
        role_oof_total, role_oof_retained = role_oof_counts.get(
            (case_id, role), (0, 0)
        )
        coverage.append(
            {
                "case_id": case_id,
                "evaluation_partition": (
                    "outer_oof" if has_outer_partition else "not_reported"
                ),
                "role": role,
                "route_state": route_state,
                "signal_route": signal_route,
                "quality_tier": quality_tier,
                "motion_state": motion_state,
                "record_count": len(rows),
                "retained_record_count": retained_count,
                "retained_coverage": retained_count / len(rows),
                "direct_rate_record_count": direct_count,
                "processed_rate_record_count": processed_count,
                "dropped_record_count": len(rows) - retained_count,
                "abstention_count": abstention_count,
                "abstention_rate": abstention_count / len(rows),
                "abstention_reasons": _joined_text(abstention_reasons),
                "mean_motion_record_probability": (
                    float(np.mean(motion_probabilities))
                    if motion_probabilities else None
                ),
                "mean_motion_threshold": (
                    float(np.mean(motion_thresholds))
                    if motion_thresholds else None
                ),
                "mean_motion_window_count": (
                    float(np.mean(motion_window_counts))
                    if motion_window_counts else None
                ),
                "motion_evidence_sha256": _joined_text(
                    row.get("evidence_sha256")
                    for row in motion_provenance_rows
                ),
                "motion_model_artifact_sha256": _joined_text(
                    row.get("model_artifact_sha256")
                    for row in motion_provenance_rows
                ),
                "motion_training_scope": _joined_text(
                    row.get("training_scope")
                    for row in motion_provenance_rows
                ),
                "motion_frailty29_relation": _joined_text(
                    row.get("frailty29_evaluation_relation")
                    for row in motion_provenance_rows
                ),
                "auxiliary_motion_evidence_valid_outer_oof": (
                    all(
                        _as_bool(row.get("valid_outer_oof_claim", False))
                        and str(row.get("frailty29_evaluation_relation", ""))
                        != "in_sample_for_frailty29"
                        for row in motion_provenance_rows
                    )
                    if motion_provenance_rows
                    else None
                ),
                **{
                    f"{name}s": _joined_text(values)
                    for name, values in sqi_states.items()
                },
                **{
                    f"mean_{name}": (
                        float(np.mean(values)) if values else None
                    )
                    for name, values in sqi_values.items()
                },
                "denoiser_attempt_count": denoiser_attempt_count,
                "denoiser_success_count": denoiser_success_count,
                "denoiser_requested_cell_count": denoiser_requested_cell_count,
                "denoiser_success_cell_count": denoiser_success_cell_count,
                "denoiser_failed_cell_count": (
                    denoiser_requested_cell_count - denoiser_success_cell_count
                ),
                "denoiser_cell_failure_rate": (
                    (denoiser_requested_cell_count - denoiser_success_cell_count)
                    / denoiser_requested_cell_count
                    if denoiser_requested_cell_count
                    else None
                ),
                "post_q_rate_pass_count": post_q_rate_pass_count,
                "post_q_rate_pass_rate": (
                    post_q_rate_pass_count / denoiser_attempt_count
                    if denoiser_attempt_count
                    else None
                ),
                "post_q_rate_recovery_eligible_count": (
                    post_q_rate_recovery_eligible_count
                ),
                "post_q_rate_recovery_count": post_q_rate_recovery_count,
                "post_q_rate_recovery_rate": (
                    post_q_rate_recovery_count
                    / post_q_rate_recovery_eligible_count
                    if post_q_rate_recovery_eligible_count
                    else None
                ),
                "post_q_rate_pass_cell_count": post_q_rate_pass_cell_count,
                "post_q_rate_pass_cell_rate": (
                    post_q_rate_pass_cell_count / denoiser_requested_cell_count
                    if denoiser_requested_cell_count
                    else None
                ),
                "post_q_rate_recovery_eligible_cell_count": (
                    post_q_rate_recovery_eligible_cell_count
                ),
                "post_q_rate_recovered_cell_count": (
                    post_q_rate_recovered_cell_count
                ),
                "post_q_rate_recovery_cell_rate": (
                    post_q_rate_recovered_cell_count
                    / post_q_rate_recovery_eligible_cell_count
                    if post_q_rate_recovery_eligible_cell_count
                    else None
                ),
                "motion_high_cell_count": motion_high_cell_count,
                "motion_low_cell_count": motion_low_cell_count,
                "denoiser_ids": _joined_text(denoiser_ids),
                "denoiser_statuses": _joined_text(denoiser_statuses),
                "mean_unavailable_predictor_count": (
                    float(np.mean(unavailable_counts))
                    if unavailable_counts
                    else None
                ),
                "unavailable_predictor_rate": (
                    total_unavailable / total_predictors
                    if total_predictors > 0.0
                    else None
                ),
                "reducer_failure_count": reducer_failures,
                "reducer_failure_rate": (
                    reducer_failures / denoiser_attempt_count
                    if denoiser_attempt_count
                    else None
                ),
                "role_oof_prediction_count": role_oof_total,
                "retained_role_oof_prediction_count": role_oof_retained,
            }
        )

    distributions: list[Mapping[str, Any]] = []
    for (case_id, role, route_state, component), values in sorted(
        component_groups.items()
    ):
        finite = [value for value, valid in values if valid and value is not None]
        distributions.append(
            {
                "case_id": case_id,
                "role": role,
                "route_state": route_state,
                "component": component,
                "reported_count": len(values),
                "valid_count": len(finite),
                "unavailable_rate": 1.0 - len(finite) / len(values),
                "mean": float(np.mean(finite)) if finite else None,
                "population_sd": float(np.std(finite)) if finite else None,
                "minimum": min(finite, default=None),
                "maximum": max(finite, default=None),
            }
        )
    return coverage, distributions


def _denoiser_hr_tables(
    collected: CollectedStudy,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Build paired per-record and participant-macro direct/post HR tables."""

    records: list[dict[str, Any]] = []
    for row in collected.quality_rows:
        artifact = (
            row.get("route_artifact")
            if isinstance(row.get("route_artifact"), Mapping)
            else {}
        )
        if not _as_bool(artifact.get("denoiser_attempted", False)):
            continue
        direct_hr = _number(artifact.get("direct_hr_bpm"))
        post_hr = _number(artifact.get("post_denoise_hr_bpm"))
        delta = (
            float(post_hr - direct_hr)
            if direct_hr is not None and post_hr is not None
            else None
        )
        direct_ppi_s = _number(artifact.get("direct_median_valid_ppi_s"))
        post_ppi_s = _number(
            artifact.get("post_denoise_median_valid_ppi_s")
        )
        ppi_delta_ms = (
            float((post_ppi_s - direct_ppi_s) * 1000.0)
            if direct_ppi_s is not None and post_ppi_s is not None
            else None
        )
        direct_q_rate_state = str(
            artifact.get("direct_q_rate_state") or "not_evaluated"
        ).lower()
        post_q_rate_state = str(
            artifact.get("post_q_rate_state") or "not_evaluated"
        ).lower()
        recovery_eligible = direct_q_rate_state not in {
            "pass",
            "not_evaluated",
        }
        denoiser_status = str(
            artifact.get("denoiser_status")
            or artifact.get("reducer_status")
            or artifact.get("artifact_reducer_status")
            or "not_reported"
        ).lower()
        records.append(
            {
                "case_id": str(row.get("case_id", "")),
                "repeat": row.get("repeat"),
                "fold": row.get("fold"),
                "outer_partition": str(
                    row.get("outer_partition", "not_reported")
                ),
                "participant_id": str(row.get("participant_id", "")),
                "record_id": str(row.get("record_id", "")),
                "role": str(row.get("role", "unknown")),
                "denoiser_id": str(artifact.get("denoiser_id", "unknown")),
                "denoiser_status": denoiser_status,
                "heart_rate_estimator": str(
                    artifact.get(
                        "heart_rate_estimator",
                        "60_over_median_valid_ppi_s",
                    )
                ),
                "direct_hr_bpm": direct_hr,
                "post_denoise_hr_bpm": post_hr,
                "post_minus_direct_hr_bpm": delta,
                "absolute_post_minus_direct_hr_bpm": (
                    None if delta is None else abs(delta)
                ),
                "direct_median_valid_ppi_ms": (
                    None if direct_ppi_s is None else direct_ppi_s * 1000.0
                ),
                "post_denoise_median_valid_ppi_ms": (
                    None if post_ppi_s is None else post_ppi_s * 1000.0
                ),
                "post_minus_direct_ppi_ms": ppi_delta_ms,
                "absolute_post_minus_direct_ppi_ms": (
                    None if ppi_delta_ms is None else abs(ppi_delta_ms)
                ),
                "direct_valid_ppi_count": artifact.get(
                    "direct_valid_ppi_count"
                ),
                "post_denoise_valid_ppi_count": artifact.get(
                    "post_denoise_valid_ppi_count"
                ),
                "direct_peak_count": artifact.get("direct_peak_count"),
                "post_denoise_peak_count": artifact.get(
                    "post_denoise_peak_count"
                ),
                "direct_reference_wavelength": artifact.get(
                    "direct_reference_wavelength"
                ),
                "post_denoise_reference_wavelength": artifact.get(
                    "post_denoise_reference_wavelength"
                ),
                "direct_q_rate_state": direct_q_rate_state,
                "post_q_rate_state": post_q_rate_state,
                "post_q_rate_recovery_eligible": recovery_eligible,
                "post_q_rate_recovered": (
                    recovery_eligible and post_q_rate_state == "pass"
                ),
                "reducer_failed": denoiser_status != "success",
                "retained_for_classifier": _as_bool(row.get("retained", False)),
            }
        )

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        for role_scope in (str(row["role"]), "ALL"):
            grouped.setdefault(
                (
                    str(row["case_id"]),
                    str(row["denoiser_id"]),
                    str(row["outer_partition"]),
                    role_scope,
                ),
                [],
            ).append(row)

    summary: list[Mapping[str, Any]] = []
    for (case_id, denoiser_id, outer_partition, role_scope), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] != "outer_oof",
            item[0][2],
            item[0][3] != "ALL",
            item[0][3],
        ),
    ):
        paired = [
            row
            for row in rows
            if row["direct_hr_bpm"] is not None
            and row["post_denoise_hr_bpm"] is not None
        ]
        participant_rows: dict[str, list[Mapping[str, Any]]] = {}
        for row in paired:
            participant_rows.setdefault(str(row["participant_id"]), []).append(row)
        participant_direct = [
            float(fmean(float(row["direct_hr_bpm"]) for row in values))
            for values in participant_rows.values()
        ]
        participant_post = [
            float(fmean(float(row["post_denoise_hr_bpm"]) for row in values))
            for values in participant_rows.values()
        ]
        participant_delta = [
            float(
                fmean(float(row["post_minus_direct_hr_bpm"]) for row in values)
            )
            for values in participant_rows.values()
        ]
        participant_absolute_delta = [
            float(
                fmean(
                    float(row["absolute_post_minus_direct_hr_bpm"])
                    for row in values
                )
            )
            for values in participant_rows.values()
        ]
        ppi_paired = [
            row
            for row in rows
            if row["direct_median_valid_ppi_ms"] is not None
            and row["post_denoise_median_valid_ppi_ms"] is not None
        ]
        participant_ppi_rows: dict[str, list[Mapping[str, Any]]] = {}
        for row in ppi_paired:
            participant_ppi_rows.setdefault(
                str(row["participant_id"]), []
            ).append(row)
        participant_direct_ppi = [
            float(
                fmean(
                    float(row["direct_median_valid_ppi_ms"])
                    for row in values
                )
            )
            for values in participant_ppi_rows.values()
        ]
        participant_post_ppi = [
            float(
                fmean(
                    float(row["post_denoise_median_valid_ppi_ms"])
                    for row in values
                )
            )
            for values in participant_ppi_rows.values()
        ]
        participant_ppi_delta = [
            float(
                fmean(
                    float(row["post_minus_direct_ppi_ms"])
                    for row in values
                    if row["post_minus_direct_ppi_ms"] is not None
                )
            )
            for values in participant_ppi_rows.values()
            if any(row["post_minus_direct_ppi_ms"] is not None for row in values)
        ]
        participant_absolute_ppi_delta = [
            float(
                fmean(
                    float(row["absolute_post_minus_direct_ppi_ms"])
                    for row in values
                    if row["absolute_post_minus_direct_ppi_ms"] is not None
                )
            )
            for values in participant_ppi_rows.values()
            if any(
                row["absolute_post_minus_direct_ppi_ms"] is not None
                for row in values
            )
        ]
        direct_stats = _descriptive_statistics(participant_direct)
        post_stats = _descriptive_statistics(participant_post)
        delta_stats = _descriptive_statistics(participant_delta)
        absolute_stats = _descriptive_statistics(participant_absolute_delta)
        direct_ppi_stats = _descriptive_statistics(participant_direct_ppi)
        post_ppi_stats = _descriptive_statistics(participant_post_ppi)
        ppi_delta_stats = _descriptive_statistics(participant_ppi_delta)
        absolute_ppi_delta_stats = _descriptive_statistics(
            participant_absolute_ppi_delta
        )
        recovery_eligible = [
            row for row in rows if row["post_q_rate_recovery_eligible"]
        ]
        summary.append(
            {
                "case_id": case_id,
                "denoiser_id": denoiser_id,
                "outer_partition": outer_partition,
                "role_scope": role_scope,
                "attempted_record_count": len(rows),
                "successful_reducer_record_count": sum(
                    str(row["denoiser_status"]).lower() == "success"
                    for row in rows
                ),
                "reducer_failure_count": sum(
                    bool(row["reducer_failed"]) for row in rows
                ),
                "reducer_failure_rate": (
                    sum(bool(row["reducer_failed"]) for row in rows)
                    / len(rows)
                    if rows
                    else None
                ),
                "post_q_rate_pass_count": sum(
                    str(row["post_q_rate_state"]) == "pass" for row in rows
                ),
                "post_q_rate_pass_rate": (
                    sum(
                        str(row["post_q_rate_state"]) == "pass"
                        for row in rows
                    )
                    / len(rows)
                    if rows
                    else None
                ),
                "post_q_rate_recovery_eligible_count": len(recovery_eligible),
                "post_q_rate_recovery_count": sum(
                    bool(row["post_q_rate_recovered"])
                    for row in recovery_eligible
                ),
                "post_q_rate_recovery_rate": (
                    sum(
                        bool(row["post_q_rate_recovered"])
                        for row in recovery_eligible
                    )
                    / len(recovery_eligible)
                    if recovery_eligible
                    else None
                ),
                "paired_hr_record_count": len(paired),
                "paired_participant_count": len(participant_rows),
                "paired_ppi_record_count": len(ppi_paired),
                "paired_ppi_participant_count": len(participant_ppi_rows),
                "participant_macro_direct_hr_bpm": direct_stats["mean"],
                "participant_sd_direct_hr_bpm": direct_stats["population_sd"],
                "participant_macro_post_denoise_hr_bpm": post_stats["mean"],
                "participant_sd_post_denoise_hr_bpm": post_stats[
                    "population_sd"
                ],
                "participant_macro_post_minus_direct_hr_bpm": delta_stats[
                    "mean"
                ],
                "participant_sd_post_minus_direct_hr_bpm": delta_stats[
                    "population_sd"
                ],
                "participant_macro_absolute_hr_change_bpm": absolute_stats[
                    "mean"
                ],
                "participant_sd_absolute_hr_change_bpm": absolute_stats[
                    "population_sd"
                ],
                "participant_macro_direct_median_ppi_ms": direct_ppi_stats[
                    "mean"
                ],
                "participant_sd_direct_median_ppi_ms": direct_ppi_stats[
                    "population_sd"
                ],
                "participant_macro_post_denoise_median_ppi_ms": (
                    post_ppi_stats["mean"]
                ),
                "participant_sd_post_denoise_median_ppi_ms": (
                    post_ppi_stats["population_sd"]
                ),
                "participant_macro_ppi_endpoint_error_ms": (
                    absolute_ppi_delta_stats["mean"]
                ),
                "participant_sd_ppi_endpoint_error_ms": (
                    absolute_ppi_delta_stats["population_sd"]
                ),
                "participant_macro_post_minus_direct_ppi_ms": (
                    ppi_delta_stats["mean"]
                ),
                "participant_sd_post_minus_direct_ppi_ms": (
                    ppi_delta_stats["population_sd"]
                ),
                "endpoint_reference": (
                    "same_record_direct_ppg_no_ecg_reference"
                ),
                "heart_rate_estimator": (
                    str(rows[0]["heart_rate_estimator"])
                    if rows
                    else "60_over_median_valid_ppi_s"
                ),
                "comparison_scope": (
                    "paired_same_record_same_outer_partition_before_after_"
                    "denoising_then_participant_macro"
                ),
            }
        )
    return records, summary


_OOF_ROW_FIELDS = frozenset(field.name for field in fields(OofPredictionRow))


def _oof_object(row: Mapping[str, Any]) -> OofPredictionRow:
    """Rebuild one strict OOF row from the report-safe mapping projection."""

    payload = {
        name: row[name]
        for name in _OOF_ROW_FIELDS
        if name in row
    }
    for name in ("probabilities", "member_training_seeds", "class_order"):
        if name in payload:
            payload[name] = tuple(payload[name] or ())
    return OofPredictionRow(**payload)


def _participant_replay_key(row: OofPredictionRow) -> tuple[Any, ...]:
    return (
        row.repeat,
        row.fold,
        row.split_seed,
        row.participant_id,
        row.prediction_kind,
        row.member_index,
        row.training_seed,
        row.model_hash,
        row.signal_route,
    )


def _validate_source_line_replay(
    *,
    case_id: str,
    source_line: str,
    source_rows: Sequence[OofPredictionRow],
    replayed: Sequence[OofPredictionRow],
    persisted_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Verify retained probabilities and fully-dropped participant coverage."""

    if not persisted_rows:
        return "not_checked_no_persisted_subject_oof"
    persisted = tuple(_oof_object(row) for row in persisted_rows)
    if any(row.level != "participant" for row in persisted):
        raise ValueError("persisted subject OOF contains a non-participant row")
    if any(row.aggregation_rule != source_line for row in persisted):
        raise ValueError("persisted subject OOF disagrees with the file-OOF source line")
    retained_persisted = tuple(row for row in persisted if row.retained)
    dropped_persisted = tuple(row for row in persisted if not row.retained)
    expected = {
        _participant_replay_key(row): row
        for row in retained_persisted
    }
    actual = {_participant_replay_key(row): row for row in replayed}
    if len(expected) != len(retained_persisted) or len(actual) != len(replayed):
        raise ValueError("duplicate participant OOF identity during aggregation replay")
    if expected.keys() != actual.keys():
        raise ValueError("file-OOF replay participant roster differs from persisted subject OOF")
    for key, observed in actual.items():
        reference = expected[key]
        if (
            observed.label != reference.label
            or tuple(observed.class_order) != tuple(reference.class_order)
            or observed.aggregation_rule != reference.aggregation_rule
        ):
            raise ValueError("file-OOF replay participant metadata differs from persisted OOF")
        if not np.allclose(
            np.asarray(observed.probabilities, dtype=np.float64),
            np.asarray(reference.probabilities, dtype=np.float64),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                f"{case_id}: file-OOF source-line replay probability mismatch"
            )

    source_by_participant: dict[
        tuple[Any, ...], list[OofPredictionRow]
    ] = {}
    for row in source_rows:
        source_by_participant.setdefault(_participant_replay_key(row), []).append(row)
    retained_source_keys = {
        key
        for key, rows in source_by_participant.items()
        if any(row.retained for row in rows)
    }
    dropped_source_keys = set(source_by_participant) - retained_source_keys
    if retained_source_keys != set(actual):
        raise ValueError("file-OOF retained participant coverage differs from replay")
    persisted_dropped_keys = {
        _participant_replay_key(row)
        for row in dropped_persisted
    }
    if len(persisted_dropped_keys) != len(dropped_persisted):
        raise ValueError("duplicate dropped participant OOF identity during replay")
    if persisted_dropped_keys != dropped_source_keys:
        raise ValueError(
            "file-OOF fully-dropped participant coverage differs from persisted OOF"
        )
    return "exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage"


def _validate_selected_cell_oof_provenance(
    collected: CollectedStudy,
    *,
    case_id: str,
    source_rows: Sequence[OofPredictionRow],
) -> str:
    """Reject stale/mixed OOF using the selected case-result cell keys."""

    raw_cells = [
        row
        for row in getattr(collected, "cell_rows", ())
        if str(row.get("case_id")) == case_id
        and str(row.get("status", "passed")) == "passed"
        and row.get("repeat") is not None
        and row.get("fold") is not None
    ]
    if not raw_cells:
        return "not_checked_no_selected_cell_rows"

    cells: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in raw_cells:
        key = (int(row["repeat"]), int(row["fold"]))
        if key in cells:
            raise ValueError("selected case result contains duplicate execution cells")
        cells[key] = row
    source_keys = {(int(row.repeat), int(row.fold)) for row in source_rows}
    if source_keys != set(cells):
        raise ValueError("selected cell roster disagrees with file OOF execution keys")

    for row in source_rows:
        selected = cells[(int(row.repeat), int(row.fold))]
        for field, observed in (
            ("split_seed", row.split_seed),
            ("training_seed", row.training_seed),
            ("config_hash", row.config_hash),
        ):
            if field == "training_seed" and observed is None:
                continue
            expected = selected.get(field)
            if expected in (None, ""):
                continue
            if field in {"split_seed", "training_seed"}:
                expected = int(expected)
            else:
                expected = str(expected)
            if observed != expected:
                raise ValueError(
                    f"selected case-result {field} disagrees with the file OOF"
                )
    return "exact_match_selected_case_result_cell_roster_and_available_provenance"


def _aggregation_replay_context(
    collected: CollectedStudy,
    *,
    case_id: str,
    source_rows: Sequence[OofPredictionRow],
    persisted_rows: Sequence[Mapping[str, Any]],
) -> _AggregationReplayContext:
    """Resolve an effective OOF line and prove it against participant OOF."""

    if not source_rows or any(row.level != "file" for row in source_rows):
        raise ValueError("aggregation replay requires file-level OOF rows")
    source_lines = {row.aggregation_rule for row in source_rows}
    if len(source_lines) != 1 or next(iter(source_lines)) not in BALANCE_LINES:
        raise ValueError("file OOF has multiple or unsupported effective source lines")
    source_line = next(iter(source_lines))
    semantics = _aggregation_replay_semantics(collected, case_id)
    selected_validation = _validate_selected_cell_oof_provenance(
        collected,
        case_id=case_id,
        source_rows=source_rows,
    )
    replay_source = tuple(
        replace(row, aggregation_rule=source_line) for row in source_rows
    )
    source_participants = aggregate_hierarchy(
        replay_source,
        balance_line=source_line,
        quality_weighted=semantics.quality_weighted,
        quality_weight_source=semantics.quality_weight_source,
    ).participant_rows
    source_validation = _validate_source_line_replay(
        case_id=case_id,
        source_line=source_line,
        source_rows=source_rows,
        replayed=source_participants,
        persisted_rows=persisted_rows,
    )
    return _AggregationReplayContext(
        source_line=source_line,
        semantics=semantics,
        source_participant_rows=tuple(source_participants),
        selected_cell_validation=selected_validation,
        source_replay_validation=source_validation,
    )


def _window_balanced_participants(
    source_rows: Sequence[OofPredictionRow],
) -> tuple[OofPredictionRow, ...]:
    """Average all retained windows equally within each held-out participant.

    This is a report-only sensitivity view.  It deliberately bypasses the
    canonical window→file invariant and therefore cannot become training or
    model-selection evidence.
    """

    if any(row.level != "window" for row in source_rows):
        raise ValueError("window-balanced reporting requires window-level OOF rows")
    groups: dict[tuple[Any, ...], list[OofPredictionRow]] = {}
    for row in source_rows:
        if not row.retained:
            continue
        key = (
            row.repeat,
            row.fold,
            row.split_seed,
            row.seed,
            row.participant_id,
            row.prediction_kind,
            row.member_index,
            row.training_seed,
            row.config_hash,
            row.manifest_hash,
            row.fold_hash,
            row.preprocessing_hash,
            row.feature_hash,
            row.model_hash,
            row.representation_mode,
            row.signal_route,
        )
        groups.setdefault(key, []).append(row)
    output: list[OofPredictionRow] = []
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        del key
        if len({row.label for row in rows}) != 1:
            raise ValueError("labels disagree within one window-balanced participant")
        if len({tuple(row.class_order) for row in rows}) != 1:
            raise ValueError("class orders disagree within one window-balanced participant")
        probabilities = np.asarray(
            [row.probabilities for row in rows], dtype=np.float64
        ).mean(axis=0)
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
            raise ValueError("window-balanced participant probabilities are invalid")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("window-balanced participant probabilities sum to zero")
        probabilities /= total
        reference = rows[0]
        output.append(
            replace(
                reference,
                probabilities=tuple(float(value) for value in probabilities),
                file_id=f"participant::{reference.participant_id}",
                role="participant",
                level="participant",
                window_id=None,
                member_index=None,
                quality_score=float(np.mean([row.quality_score for row in rows])),
                aggregation_rule=WINDOW_BALANCED_TO_PARTICIPANT,
            )
        )
    return tuple(output)


def _raw_recording_role(row: Mapping[str, Any]) -> str:
    """Recover B/R1… identifiers from the immutable file id for report labels."""

    import re

    file_id = str(row.get("file_id", "")).strip().upper()
    match = re.search(r"(?:^|[:/\\_-])(B|R[0-9]+|S[0-9]+|W[0-9]+)$", file_id)
    if match is not None:
        return str(match.group(1))
    return str(row.get("role", "unknown"))


def _aggregation_hierarchy_coverage(
    collected: CollectedStudy,
) -> list[Mapping[str, Any]]:
    """Expose the distinct window/file/role/participant report populations."""

    failed_levels = {
        (str(row.get("case_id")), str(row.get("oof_level")))
        for row in collected.oof_read_failures
    }
    files_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.file_oof_rows:
        files_by_case.setdefault(str(row.get("case_id")), []).append(row)
    subjects_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.subject_oof_rows:
        subjects_by_case.setdefault(str(row.get("case_id")), []).append(row)

    # A Line-A source run has no persisted role rows by construction.  Rebuild
    # the report-only Line-B role layer from that same fitted model's file OOF so
    # every case exposes the B/R population used by the parallel role-balanced
    # view.  Never combine this replay with the persisted source-role table: a
    # Line-B source case would otherwise be counted twice.
    replayed_role_rows: list[Mapping[str, Any]] = []
    for case_id, raw_rows in sorted(files_by_case.items()):
        if (case_id, "file") in failed_levels:
            continue
        try:
            file_rows = tuple(_oof_object(row) for row in raw_rows)
            context = _aggregation_replay_context(
                collected,
                case_id=case_id,
                source_rows=file_rows,
                persisted_rows=subjects_by_case.get(case_id, ()),
            )
            replay = tuple(
                replace(row, aggregation_rule=LINE_B_EQUAL_ROLE_FAMILIES)
                for row in file_rows
            )
            replayed_role_rows.extend(
                {"case_id": case_id, **asdict(row)}
                for row in aggregate_hierarchy(
                    replay,
                    balance_line=LINE_B_EQUAL_ROLE_FAMILIES,
                    quality_weighted=context.semantics.quality_weighted,
                    quality_weight_source=context.semantics.quality_weight_source,
                ).role_rows
            )
        except Exception:  # noqa: BLE001 - other report tables retain the limitation.
            continue

    sources = (
        (
            "window",
            WINDOW_BALANCED_TO_PARTICIPANT,
            collected.window_oof_rows,
            _raw_recording_role,
        ),
        ("file", LINE_A_EQUAL_FILES, collected.file_oof_rows, _raw_recording_role),
        (
            "role",
            LINE_B_EQUAL_ROLE_FAMILIES,
            tuple(replayed_role_rows),
            lambda row: str(row.get("role", "unknown")),
        ),
        (
            "participant",
            "participant_balanced_endpoint",
            collected.subject_oof_rows,
            lambda row: "ALL",
        ),
    )
    grouped: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = {}
    for level, view, rows, labeler in sources:
        for row in rows:
            if (str(row.get("case_id")), level) in failed_levels:
                continue
            labels = (str(labeler(row)), "ALL")
            for group_label in dict.fromkeys(labels):
                key = (
                    str(row.get("case_id")),
                    int(row.get("repeat", 0)),
                    level,
                    view,
                    group_label,
                )
                grouped.setdefault(key, []).append(row)
    output: list[Mapping[str, Any]] = []
    for (case_id, repeat, level, view, group_label), rows in sorted(grouped.items()):
        retained = [row for row in rows if row.get("retained", True) is not False]
        all_participants = {
            str(row.get("participant_id"))
            for row in rows
            if row.get("participant_id") is not None
        }
        retained_participants = {
            str(row.get("participant_id"))
            for row in retained
            if row.get("participant_id") is not None
        }
        output.append(
            {
                "case_id": case_id,
                "repeat": repeat,
                "aggregation_level": level,
                "aggregation_view": view,
                "group_label": group_label,
                "oof_unit_count": len(rows),
                "retained_oof_unit_count": len(retained),
                "dropped_oof_unit_count": len(rows) - len(retained),
                "retained_coverage": len(retained) / len(rows),
                # Preserve the historical participant_count meaning used by
                # existing plots: participants represented after retention.
                "participant_count": len(retained_participants),
                "total_participant_count": len(all_participants),
                "retained_participant_count": len(retained_participants),
                "dropped_participant_count": (
                    len(all_participants - retained_participants)
                ),
            }
        )
    return output


def _aggregation_report_view_tables(
    collected: CollectedStudy,
    *,
    source_eligibility_by_case: Mapping[str, bool],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[str],
]:
    """Build three report-only participant views from the same fitted OOF."""

    windows_by_case: dict[str, list[Mapping[str, Any]]] = {}
    files_by_case: dict[str, list[Mapping[str, Any]]] = {}
    subjects_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.window_oof_rows:
        windows_by_case.setdefault(str(row.get("case_id")), []).append(row)
    for row in collected.file_oof_rows:
        files_by_case.setdefault(str(row.get("case_id")), []).append(row)
    for row in collected.subject_oof_rows:
        subjects_by_case.setdefault(str(row.get("case_id")), []).append(row)
    failures = {
        (str(row.get("case_id")), str(row.get("oof_level")))
        for row in collected.oof_read_failures
    }
    bins = int(collected.plan.get("report", {}).get("calibration_bins", 10))
    summaries: list[Mapping[str, Any]] = []
    repeat_metrics: list[Mapping[str, Any]] = []
    fold_metrics: list[Mapping[str, Any]] = []
    per_class_metrics: list[Mapping[str, Any]] = []
    confusion_matrices: list[Mapping[str, Any]] = []
    notes: list[str] = []

    for case_id in sorted(set(windows_by_case) | set(files_by_case)):
        views: dict[str, tuple[OofPredictionRow, ...]] = {}
        source_line: str | None = None
        replay_semantics: _AggregationReplaySemantics | None = None
        replay_context: _AggregationReplayContext | None = None
        raw_files = files_by_case.get(case_id, ())
        if raw_files and (case_id, "file") not in failures:
            try:
                file_rows = tuple(_oof_object(row) for row in raw_files)
                replay_context = _aggregation_replay_context(
                    collected,
                    case_id=case_id,
                    source_rows=file_rows,
                    persisted_rows=subjects_by_case.get(case_id, ()),
                )
                source_line = replay_context.source_line
                replay_semantics = replay_context.semantics
                for balance_line in BALANCE_LINES:
                    if balance_line == source_line:
                        views[balance_line] = replay_context.source_participant_rows
                    else:
                        replay = tuple(
                            replace(row, aggregation_rule=balance_line)
                            for row in file_rows
                        )
                        views[balance_line] = aggregate_hierarchy(
                            replay,
                            balance_line=balance_line,
                            quality_weighted=replay_semantics.quality_weighted,
                            quality_weight_source=replay_semantics.quality_weight_source,
                        ).participant_rows
            except Exception as error:  # noqa: BLE001 - preserve report limitation.
                notes.append(
                    f"{case_id}: file/role-balanced report views unavailable: "
                    f"{type(error).__name__}: {error}"
                )
        raw_windows = windows_by_case.get(case_id, ())
        if raw_windows and (case_id, "window") not in failures:
            try:
                views[WINDOW_BALANCED_TO_PARTICIPANT] = _window_balanced_participants(
                    tuple(_oof_object(row) for row in raw_windows)
                )
            except Exception as error:  # noqa: BLE001 - preserve report limitation.
                notes.append(
                    f"{case_id}: window-balanced report view unavailable: "
                    f"{type(error).__name__}: {error}"
                )
        elif not raw_windows:
            notes.append(
                f"{case_id}: window-balanced report view is N/A because no window OOF was persisted"
            )

        for view in AGGREGATION_REPORT_VIEWS:
            participant_rows = views.get(view)
            if not participant_rows:
                continue
            participant_maps = [
                {"case_id": case_id, **asdict(row)} for row in participant_rows
            ]
            by_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
            for row in participant_maps:
                by_cell.setdefault(
                    (int(row["repeat"]), int(row["fold"])), []
                ).append(row)
            by_repeat: dict[int, list[Mapping[str, Any]]] = {}
            for row in participant_maps:
                by_repeat.setdefault(int(row["repeat"]), []).append(row)
            current_repeats: list[Mapping[str, Any]] = []
            evidence_role = (
                "declared_training_aggregation"
                if view == source_line
                else "posthoc_same_oof_sensitivity_only"
            )
            view_quality_weighted = bool(
                view in BALANCE_LINES
                and replay_semantics is not None
                and replay_semantics.quality_weighted
            )
            view_quality_weight_source = (
                replay_semantics.quality_weight_source
                if view_quality_weighted and replay_semantics is not None
                else QUALITY_WEIGHT_SOURCE_NONE
            )
            for (repeat, fold), rows in sorted(by_cell.items()):
                metric = _oof_metric_row(case_id, rows, repeat=repeat)
                if metric is None:
                    continue
                fold_metrics.append(
                    {
                        key: value
                        for key, value in metric.items()
                        if key not in {"per_class", "confusion_matrix"}
                    }
                    | {
                        "fold": fold,
                        "aggregation_view": view,
                        "declared_source_line": source_line,
                        "metric_source": "same_fitted_oof_report_reaggregation",
                    }
                )
            for repeat, rows in sorted(by_repeat.items()):
                metric = _oof_metric_row(case_id, rows, repeat=repeat)
                if metric is None:
                    continue
                projected = {
                    key: value
                    for key, value in metric.items()
                    if key not in {"per_class", "confusion_matrix"}
                }
                _, repeat_ece = _calibration_rows(case_id, rows, bins=bins)
                projected.update(
                    {
                        "aggregation_view": view,
                        "declared_source_line": source_line,
                        "evidence_role": evidence_role,
                        "quality_weighting": view_quality_weighted,
                        "quality_weight_source": view_quality_weight_source,
                        "expected_calibration_error": repeat_ece,
                        "metric_source": "same_fitted_oof_report_reaggregation",
                    }
                )
                current_repeats.append(projected)
                repeat_metrics.append(projected)
                for class_row in metric["per_class"]:
                    per_class_metrics.append(
                        {
                            **class_row,
                            "aggregation_view": view,
                            "declared_source_line": source_line,
                            "evidence_role": evidence_role,
                            "quality_weighting": view_quality_weighted,
                            "quality_weight_source": view_quality_weight_source,
                            "metric_source": "same_fitted_oof_report_reaggregation",
                        }
                    )
            if not current_repeats:
                continue
            pooled = _oof_metric_row(case_id, participant_maps, repeat=None)
            if pooled is not None:
                confusion_matrices.append(
                    {
                        "case_id": case_id,
                        "aggregation_view": view,
                        "declared_source_line": source_line,
                        "evidence_role": evidence_role,
                        "quality_weighting": view_quality_weighted,
                        "quality_weight_source": view_quality_weight_source,
                        "class_order": pooled["class_order"],
                        "class_names": [
                            CANONICAL_CLASS_NAMES.get(int(value), str(value))
                            for value in pooled["class_order"]
                        ],
                        "confusion_matrix": pooled["confusion_matrix"],
                        "metric_source": "pooled_same_fitted_oof_report_reaggregation",
                    }
                )
            ba = _descriptive_statistics(
                row.get("balanced_accuracy") for row in current_repeats
            )
            f1 = _descriptive_statistics(row.get("macro_f1") for row in current_repeats)
            worst_recall = _descriptive_statistics(
                row.get("worst_class_recall") for row in current_repeats
            )
            worst_f1 = _descriptive_statistics(
                row.get("worst_class_f1") for row in current_repeats
            )
            summaries.append(
                {
                    "case_id": case_id,
                    "aggregation_view": view,
                    "declared_source_line": source_line,
                    "resolved_config_balance_line": (
                        replay_semantics.resolved_config_balance_line
                        if replay_semantics is not None
                        else None
                    ),
                    "source_line_provenance": (
                        "selected_file_oof_effective_line"
                        if replay_context is not None
                        else None
                    ),
                    "source_replay_validation": (
                        replay_context.source_replay_validation
                        if replay_context is not None
                        else None
                    ),
                    "evidence_role": evidence_role,
                    "quality_weighting": view_quality_weighted,
                    "quality_weight_source": view_quality_weight_source,
                    "primary_ranking_eligible": bool(
                        view == source_line
                        and source_eligibility_by_case.get(case_id, False)
                    ),
                    "participant_mean_balanced_accuracy": ba["mean"],
                    "participant_mean_macro_f1": f1["mean"],
                    "repeat_balanced_accuracy_population_sd": ba["population_sd"],
                    "repeat_macro_f1_population_sd": f1["population_sd"],
                    "worst_class_recall": worst_recall["mean"],
                    "worst_class_f1": worst_f1["mean"],
                    "repeat_count": len(current_repeats),
                    "participant_oof_prediction_count": len(participant_maps),
                    "metric_source": "same_fitted_oof_report_reaggregation",
                }
            )

    return (
        summaries,
        repeat_metrics,
        fold_metrics,
        per_class_metrics,
        confusion_matrices,
        _aggregation_hierarchy_coverage(collected),
        notes,
    )


def _aggregation_line_tables(
    collected: CollectedStudy,
    *,
    complete_by_case: Mapping[str, bool],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[str],
]:
    """Build Line A/Line B views from one fitted model's file-level OOF.

    The source line reproduces the persisted participant OOF. The other line is
    explicitly post-hoc aggregation sensitivity only; it is not a separately
    trained Line A/Line B model and is never eligible for the primary ranking.
    """

    file_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.file_oof_rows:
        file_by_case.setdefault(str(row.get("case_id")), []).append(row)
    subject_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.subject_oof_rows:
        subject_by_case.setdefault(str(row.get("case_id")), []).append(row)
    statuses = _record_statuses(collected)
    bins = int(collected.plan.get("report", {}).get("calibration_bins", 10))
    summaries: list[Mapping[str, Any]] = []
    repeat_metrics: list[Mapping[str, Any]] = []
    per_class_metrics: list[Mapping[str, Any]] = []
    notes: list[str] = []
    read_failures_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for failure in collected.oof_read_failures:
        case_id = str(failure.get("case_id", ""))
        if case_id and str(failure.get("oof_level")) in {"file", "participant"}:
            read_failures_by_case.setdefault(case_id, []).append(failure)
    for case_id, failures in sorted(read_failures_by_case.items()):
        details = "; ".join(
            f"{row.get('oof_level', 'unknown')}: {row.get('error', 'read failed')}"
            for row in failures
        )
        notes.append(
            f"{case_id}: Line A/Line B file-OOF reaggregation suppressed "
            f"because OOF input was incomplete or unreadable: {details}"
        )

    for case_id, raw_rows in sorted(file_by_case.items()):
        if case_id in read_failures_by_case:
            continue
        try:
            source_rows = tuple(_oof_object(row) for row in raw_rows)
            context = _aggregation_replay_context(
                collected,
                case_id=case_id,
                source_rows=source_rows,
                persisted_rows=subject_by_case.get(case_id, ()),
            )
            source_line = context.source_line
            replay_semantics = context.semantics
            participant_by_line: dict[str, tuple[OofPredictionRow, ...]] = {
                source_line: context.source_participant_rows
            }
            for balance_line in BALANCE_LINES:
                if balance_line == source_line:
                    continue
                replay_source = tuple(
                    replace(row, aggregation_rule=balance_line)
                    for row in source_rows
                )
                participant_by_line[balance_line] = aggregate_hierarchy(
                    replay_source,
                    balance_line=balance_line,
                    quality_weighted=replay_semantics.quality_weighted,
                    quality_weight_source=replay_semantics.quality_weight_source,
                ).participant_rows
            source_validation = context.source_replay_validation
            participant_source_keys = {
                _participant_replay_key(row)
                for row in source_rows
            }
            retained_participant_source_keys = {
                _participant_replay_key(row)
                for row in source_rows
                if row.retained
            }
            dropped_participant_count = (
                len(participant_source_keys)
                - len(retained_participant_source_keys)
            )
            case_rows: list[dict[str, Any]] = []
            for balance_line in BALANCE_LINES:
                participant_maps = [
                    {"case_id": case_id, **asdict(row)}
                    for row in participant_by_line[balance_line]
                ]
                by_repeat: dict[int, list[Mapping[str, Any]]] = {}
                for row in participant_maps:
                    by_repeat.setdefault(int(row["repeat"]), []).append(row)
                current_repeats: list[Mapping[str, Any]] = []
                for repeat, rows in sorted(by_repeat.items()):
                    metric = _oof_metric_row(case_id, rows, repeat=repeat)
                    if metric is None:
                        continue
                    projected = {
                        key: value
                        for key, value in metric.items()
                        if key not in {"per_class", "confusion_matrix"}
                    }
                    projected.update(
                        {
                            "balance_line": balance_line,
                            "declared_source_line": source_line,
                            "view_role": (
                                "declared_source_line"
                                if balance_line == source_line
                                else "posthoc_aggregation_only"
                            ),
                            "quality_weighting": replay_semantics.quality_weighted,
                            "quality_weight_source": (
                                replay_semantics.quality_weight_source
                            ),
                            "metric_source": "file_oof_reaggregation",
                        }
                    )
                    _, repeat_ece = _calibration_rows(
                        case_id,
                        rows,
                        bins=bins,
                    )
                    projected["expected_calibration_error"] = repeat_ece
                    current_repeats.append(projected)
                    repeat_metrics.append(projected)
                    for row in metric["per_class"]:
                        per_class_metrics.append(
                            {
                                **row,
                                "balance_line": balance_line,
                                "declared_source_line": source_line,
                                "view_role": (
                                    "declared_source_line"
                                    if balance_line == source_line
                                    else "posthoc_aggregation_only"
                                ),
                                "quality_weighting": (
                                    replay_semantics.quality_weighted
                                ),
                                "quality_weight_source": (
                                    replay_semantics.quality_weight_source
                                ),
                                "metric_source": "per_repeat_file_oof_reaggregation",
                            }
                        )
                if not current_repeats:
                    raise ValueError("file-level OOF produced no per-repeat metrics")
                ba_statistics = _descriptive_statistics(
                    row.get("balanced_accuracy") for row in current_repeats
                )
                f1_statistics = _descriptive_statistics(
                    row.get("macro_f1") for row in current_repeats
                )
                worst_recall_statistics = _descriptive_statistics(
                    row.get("worst_class_recall") for row in current_repeats
                )
                worst_f1_statistics = _descriptive_statistics(
                    row.get("worst_class_f1") for row in current_repeats
                )
                ece_statistics = _descriptive_statistics(
                    row.get("expected_calibration_error")
                    for row in current_repeats
                )
                primary_eligible = (
                    balance_line == source_line
                    and statuses.get(case_id) == "passed"
                    and source_validation.startswith("exact_match")
                    and bool(complete_by_case.get(case_id, False))
                )
                case_rows.append(
                    {
                        "case_id": case_id,
                        "balance_line": balance_line,
                        "declared_source_line": source_line,
                        "resolved_config_balance_line": (
                            replay_semantics.resolved_config_balance_line
                        ),
                        "source_line_provenance": "selected_file_oof_effective_line",
                        "selected_cell_validation": context.selected_cell_validation,
                        "view_role": (
                            "declared_source_line"
                            if balance_line == source_line
                            else "posthoc_aggregation_only"
                        ),
                        "case_status": statuses.get(case_id, "not_run"),
                        "quality_weighting": replay_semantics.quality_weighted,
                        "quality_weight_source": replay_semantics.quality_weight_source,
                        "resolved_config_path": replay_semantics.resolved_config_path,
                        "primary_ranking_eligible": primary_eligible,
                        "participant_mean_balanced_accuracy": ba_statistics["mean"],
                        "participant_mean_macro_f1": f1_statistics["mean"],
                        "repeat_balanced_accuracy_population_sd": ba_statistics[
                            "population_sd"
                        ],
                        "repeat_balanced_accuracy_ci95_low": ba_statistics["ci95_low"],
                        "repeat_balanced_accuracy_ci95_high": ba_statistics["ci95_high"],
                        "repeat_macro_f1_population_sd": f1_statistics[
                            "population_sd"
                        ],
                        "repeat_macro_f1_ci95_low": f1_statistics["ci95_low"],
                        "repeat_macro_f1_ci95_high": f1_statistics["ci95_high"],
                        "worst_class_recall": worst_recall_statistics["mean"],
                        "worst_class_f1": worst_f1_statistics["mean"],
                        "expected_calibration_error": ece_statistics["mean"],
                        "repeat_worst_class_recall_population_sd": (
                            worst_recall_statistics["population_sd"]
                        ),
                        "repeat_worst_class_f1_population_sd": (
                            worst_f1_statistics["population_sd"]
                        ),
                        "repeat_expected_calibration_error_population_sd": (
                            ece_statistics["population_sd"]
                        ),
                        "repeat_count": len(current_repeats),
                        "participant_oof_prediction_count": len(participant_maps),
                        "participant_oof_total_count": len(
                            participant_source_keys
                        ),
                        "dropped_participant_oof_count": dropped_participant_count,
                        "file_oof_prediction_count": len(source_rows),
                        "retained_file_oof_prediction_count": sum(
                            row.retained for row in source_rows
                        ),
                        "dropped_file_oof_prediction_count": sum(
                            not row.retained for row in source_rows
                        ),
                        "source_replay_validation": (
                            source_validation
                            if balance_line == source_line
                            else "not_applicable_posthoc_view"
                        ),
                        "metric_source": "file_oof_reaggregation",
                    }
                )
            lookup = {str(row["balance_line"]): row for row in case_rows}
            line_a = lookup[LINE_A_EQUAL_FILES]
            line_b = lookup[LINE_B_EQUAL_ROLE_FAMILIES]
            ba_delta = (
                float(line_a["participant_mean_balanced_accuracy"])
                - float(line_b["participant_mean_balanced_accuracy"])
            )
            f1_delta = (
                float(line_a["participant_mean_macro_f1"])
                - float(line_b["participant_mean_macro_f1"])
            )
            for row in case_rows:
                row["line_a_minus_line_b_balanced_accuracy"] = ba_delta
                row["line_a_minus_line_b_macro_f1"] = f1_delta
            summaries.extend(case_rows)
        except Exception as error:  # noqa: BLE001 - preserve the exact report limitation.
            notes.append(
                f"{case_id}: Line A/Line B file-OOF reaggregation unavailable: "
                f"{type(error).__name__}: {error}"
            )
    return summaries, repeat_metrics, per_class_metrics, notes


_LEGACY_BRIDGE_VIEW_COLUMNS = (
    (
        WINDOW_BALANCED_TO_PARTICIPANT,
        "legacy_aggregation",
    ),
    (LINE_A_EQUAL_FILES, "line_a_aggregation"),
    (LINE_B_EQUAL_ROLE_FAMILIES, "v2_aggregation"),
)


def _legacy_bridge_report_tables(
    collected: CollectedStudy,
    case_summary: Sequence[Mapping[str, Any]],
    aggregation_view_comparison: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[str],
]:
    """Build the two distinct Stage-3 bridge reports without conflating order.

    The numeric report contains the L0 baseline followed by exactly seven
    predefined adjacent CompactCNN contrasts.  The execution report contains
    the same eight absolute metric rows in the requested run order, but never
    computes a delta between neighbouring execution rows.  In particular,
    ``L7 -> L5`` is scheduling information, not causal ablation evidence.
    """

    raw_bridge = collected.plan.get("legacy_bridge")
    if not isinstance(raw_bridge, Mapping):
        return [], [], []
    if str(raw_bridge.get("design", "cumulative_chain_v1")) != (
        "cumulative_chain_v1"
    ):
        return [], [], []

    try:
        profiles = raw_bridge.get("profiles")
        if not isinstance(profiles, (list, tuple)):
            raise ValueError("legacy_bridge.profiles must be a sequence")
        profile_by_case = {
            str(row["case_id"]): row
            for row in profiles
            if isinstance(row, Mapping) and row.get("case_id") is not None
        }
        if len(profile_by_case) != len(profiles):
            raise ValueError(
                "legacy_bridge profiles contain missing or duplicate case_id"
            )
        catalog_case_ids = tuple(
            str(row.get("catalog_case_id", ""))
            for row in profile_by_case.values()
        )
        if (
            any(not case_id for case_id in catalog_case_ids)
            or len(set(catalog_case_ids)) != len(catalog_case_ids)
        ):
            raise ValueError(
                "legacy_bridge profiles contain missing or duplicate catalog_case_id"
            )
        compact_profile_count = sum(
            str(row.get("model_id")) == "CompactCNN1D"
            for row in profile_by_case.values()
        )
        if compact_profile_count != 8:
            raise ValueError(
                "legacy_bridge must declare exactly eight CompactCNN profiles"
            )

        numeric_raw = raw_bridge.get("numeric_profile_order")
        execution_raw = raw_bridge.get("execution_order")
        if not isinstance(numeric_raw, (list, tuple)) or not isinstance(
            execution_raw, (list, tuple)
        ):
            raise ValueError(
                "legacy_bridge numeric_profile_order and execution_order must be sequences"
            )
        numeric_cases = [
            str(case_id)
            for case_id in numeric_raw
            if str(profile_by_case.get(str(case_id), {}).get("model_id"))
            == "CompactCNN1D"
        ]
        execution_cases = [
            str(case_id)
            for case_id in execution_raw
            if str(profile_by_case.get(str(case_id), {}).get("model_id"))
            == "CompactCNN1D"
        ]
        numeric_profiles = tuple(
            str(profile_by_case[case_id].get("profile_id"))
            for case_id in numeric_cases
        )
        execution_profiles = tuple(
            str(profile_by_case[case_id].get("profile_id"))
            for case_id in execution_cases
        )
        if numeric_profiles != tuple(f"L{level}" for level in range(8)):
            raise ValueError(
                "numeric CompactCNN reporting order must be exactly L0..L7"
            )
        if execution_profiles != (
            "L7",
            "L5",
            "L6",
            "L4",
            "L3",
            "L2",
            "L1",
            "L0",
        ):
            raise ValueError(
                "CompactCNN execution report order must be exactly "
                "L7,L5,L6,L4,L3,L2,L1,L0"
            )
        if set(numeric_cases) != set(execution_cases) or len(numeric_cases) != 8:
            raise ValueError(
                "numeric and execution reports must contain the same eight CompactCNN cases"
            )
        expected_pairs = tuple(
            f"{left}->{right}"
            for left, right in zip(numeric_cases, numeric_cases[1:])
        )
        adjacent_raw = raw_bridge.get("adjacent_comparisons")
        if not isinstance(adjacent_raw, (list, tuple)) or tuple(
            str(value) for value in adjacent_raw
        ) != expected_pairs:
            raise ValueError(
                "legacy_bridge adjacent_comparisons must be the seven L0->L7 pairs"
            )
    except (KeyError, TypeError, ValueError) as error:
        return [], [], [
            "legacy bridge dual reports are N/A because their frozen order "
            f"contract is invalid: {type(error).__name__}: {error}"
        ]

    summary_by_case = {
        str(row.get("case_id")): row
        for row in case_summary
        if row.get("case_id") is not None
    }
    view_by_case = {
        (str(row.get("case_id")), str(row.get("aggregation_view"))): row
        for row in aggregation_view_comparison
        if row.get("case_id") is not None
        and row.get("aggregation_view") is not None
    }

    def absolute_row(case_id: str) -> dict[str, Any]:
        profile = profile_by_case[case_id]
        catalog_case_id = str(profile["catalog_case_id"])
        summary = summary_by_case.get(catalog_case_id, {})
        result: dict[str, Any] = {
            "model": profile.get("model_id"),
            "profile": profile.get("profile_id"),
            "case_id": case_id,
            "display_case_id": case_id,
            "catalog_case_id": catalog_case_id,
            "case_status": summary.get("status", "not_run"),
            "complete_for_requested_execution": bool(
                summary.get("complete_for_requested_execution", False)
            ),
        }
        views_complete = True
        for view_name, suffix in _LEGACY_BRIDGE_VIEW_COLUMNS:
            view = view_by_case.get((catalog_case_id, view_name), {})
            ba = _number(view.get("participant_mean_balanced_accuracy"))
            macro_f1 = _number(view.get("participant_mean_macro_f1"))
            worst_f1 = _number(view.get("worst_class_f1"))
            result[f"BA_{suffix}"] = ba
            result[f"macroF1_{suffix}"] = macro_f1
            result[f"worst_class_F1_{suffix}"] = worst_f1
            result[f"metric_source_{suffix}"] = view.get("metric_source")
            result[f"evidence_role_{suffix}"] = view.get("evidence_role")
            views_complete = (
                views_complete and ba is not None and macro_f1 is not None
            )
        result["worst_class_F1"] = result["worst_class_F1_v2_aggregation"]
        result["aggregation_views_complete"] = views_complete
        return result

    numeric_rows: list[Mapping[str, Any]] = []
    previous: dict[str, Any] | None = None
    delta_fields = tuple(
        field
        for suffix in (
            "legacy_aggregation",
            "line_a_aggregation",
            "v2_aggregation",
        )
        for field in (f"BA_{suffix}", f"macroF1_{suffix}")
    ) + ("worst_class_F1",)
    for index, case_id in enumerate(numeric_cases):
        current = absolute_row(case_id)
        deltas = (
            None
            if previous is None
            else {
                field: (
                    _number(current.get(field)) - _number(previous.get(field))
                    if _number(current.get(field)) is not None
                    and _number(previous.get(field)) is not None
                    else None
                )
                for field in delta_fields
            }
        )
        row = {
            "numeric_profile_order": index,
            **current,
            "previous_numeric_case_id": (
                None if previous is None else previous["case_id"]
            ),
            "previous_numeric_profile": (
                None if previous is None else previous["profile"]
            ),
            "numeric_comparison": (
                "baseline"
                if previous is None
                else f"{previous['profile']}->{current['profile']}"
            ),
            "comparison_role": (
                "numeric_baseline_no_contrast"
                if previous is None
                else "predefined_adjacent_numeric_ablation"
            ),
            "contrast_metrics_available": (
                False
                if previous is None
                else bool(previous["aggregation_views_complete"])
                and bool(current["aggregation_views_complete"])
            ),
            "delta_from_previous_numeric_profile": deltas,
            "interpretation": (
                "baseline_for_seven_predefined_adjacent_ablation_contrasts"
                if previous is None
                else "predefined_adjacent_numeric_profile_ablation_only"
            ),
            **{
                f"delta_{field}": None if deltas is None else deltas[field]
                for field in delta_fields
            },
        }
        numeric_rows.append(row)
        previous = current

    execution_rows: list[Mapping[str, Any]] = []
    previous_execution: dict[str, Any] | None = None
    for index, case_id in enumerate(execution_cases, start=1):
        current = absolute_row(case_id)
        execution_rows.append(
            {
                "execution_order": index,
                **current,
                "previous_execution_case_id": (
                    None
                    if previous_execution is None
                    else previous_execution["case_id"]
                ),
                "previous_execution_profile": (
                    None
                    if previous_execution is None
                    else previous_execution["profile"]
                ),
                "execution_transition": (
                    "start"
                    if previous_execution is None
                    else f"{previous_execution['profile']}->{current['profile']}"
                ),
                "execution_transition_is_ablation": False,
                "interpretation": (
                    "execution_start_absolute_metrics_only"
                    if previous_execution is None
                    else "execution_order_only_not_a_causal_ablation"
                ),
            }
        )
        previous_execution = current

    notes = [
        "legacy bridge report A uses only the seven predefined numeric "
        "CompactCNN contrasts L0->L1 through L6->L7; report B lists absolute "
        "metrics in execution order and never interprets execution jumps, "
        "including L7->L5, as ablations"
    ]
    return numeric_rows, execution_rows, notes


_STAGE3_STAR_DESIGN = "centered_star_v1"
_STAGE3_STAR_VIEWS = {
    WINDOW_BALANCED_TO_PARTICIPANT: "W",
    LINE_A_EQUAL_FILES: "A",
    LINE_B_EQUAL_ROLE_FAMILIES: "B",
}
_STAGE3_STAR_METRICS = (
    ("balanced_accuracy", "BA", "participant_mean_balanced_accuracy"),
    ("macro_f1", "macroF1", "participant_mean_macro_f1"),
    ("worst_class_f1", "worst_class_F1", "worst_class_f1"),
)
_STAGE3_STAR_IDENTITY_FIELDS = (
    "repeat", "fold", "split_seed", "training_seed", "participant_id",
    "file_id", "role", "window_id", "window_start_sample", "label",
    "retained", "class_order", "prediction_kind", "member_index",
)


def _flatten_control_paths(
    value: Any, *, prefix: str = "controls"
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        path = f"{prefix}.{key}"
        output.update(
            _flatten_control_paths(item, prefix=path)
            if isinstance(item, Mapping)
            else {path: item}
        )
    return output


def _normalized_control_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(sorted({
        path if path.startswith("controls.") else f"controls.{path}"
        for raw in value if (path := str(raw).strip())
    }))


def _changed_control_paths(
    reference: Mapping[str, Any], variant: Mapping[str, Any]
) -> tuple[str, ...]:
    left, right = _flatten_control_paths(reference), _flatten_control_paths(variant)
    missing = object()
    return tuple(sorted(
        path for path in set(left) | set(right)
        if left.get(path, missing) != right.get(path, missing)
    ))


def _cell_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    output: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("repeat") is not None and row.get("fold") is not None:
            output.setdefault((int(row["repeat"]), int(row["fold"])), []).append(row)
    return output


def _star_cell_evidence(
    cell_rows: Sequence[Mapping[str, Any]],
    oof_rows: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    cells, oof = _cell_groups(cell_rows), _cell_groups(oof_rows)
    unexpected = len((set(cells) | set(oof)) - set(expected))
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for key in expected:
        current_cells, current_oof = cells.get(key, ()), oof.get(key, ())

        def unique(field: str, include_cell: bool = False) -> str | None:
            sources = (*current_oof, *current_cells) if include_cell else current_oof
            values = {
                str(row[field]) for row in sources
                if row.get(field) not in (None, "")
            }
            return next(iter(values)) if len(values) == 1 else None

        participants = sorted({
            str(row["participant_id"]) for row in current_oof
            if row.get("participant_id") not in (None, "")
        })
        result[key] = {
            "cell_passed": len(current_cells) == 1
            and str(current_cells[0].get("status")) == "passed",
            "oof_present": bool(current_oof),
            "split_seed": unique("split_seed", True),
            "training_seed": unique("training_seed", True),
            "manifest_hash": unique("manifest_hash"),
            "fold_hash": unique("fold_hash"),
            "roster_hash": stable_payload_sha256(participants) if participants else None,
            "unexpected_cell_count": unexpected,
        }
    return result


def _star_probability_audit(
    reference_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[int, int]],
) -> Mapping[str, Any]:
    expected_set = set(expected)

    def index(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, np.ndarray], bool, int]:
        observed = {
            (int(row["repeat"]), int(row["fold"])) for row in rows
            if row.get("repeat") is not None and row.get("fold") is not None
        }
        values: dict[str, np.ndarray] = {}
        duplicate = False
        for row in rows:
            if row.get("repeat") is None or row.get("fold") is None:
                continue
            identity = {field: row.get(field) for field in _STAGE3_STAR_IDENTITY_FIELDS}
            key = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
            duplicate |= key in values
            values[key] = np.asarray(row.get("probabilities"), dtype=np.float64)
        return values, observed == expected_set and not duplicate, len(rows)

    reference, reference_valid, reference_count = index(reference_rows)
    variant, variant_valid, variant_count = index(variant_rows)
    common = sorted(set(reference) & set(variant))
    differences = [
        float(np.max(np.abs(reference[key] - variant[key])))
        for key in common if reference[key].shape == variant[key].shape
    ]

    def hashes(values: Mapping[str, np.ndarray]) -> tuple[str | None, str | None]:
        keys = sorted(values)
        if not keys:
            return None, None
        return (
            stable_payload_sha256(keys),
            stable_payload_sha256([[key, values[key].tolist()] for key in keys]),
        )

    reference_identity, reference_probability = hashes(reference)
    variant_identity, variant_probability = hashes(variant)
    exact = bool(
        reference_valid and variant_valid
        and reference.keys() == variant.keys()
        and len(differences) == len(reference)
        and differences and max(differences) == 0.0
    )
    return {
        "reference_window_oof_row_count": reference_count,
        "variant_window_oof_row_count": variant_count,
        "matched_window_oof_row_count": len(common),
        "window_oof_probability_max_abs_diff": max(differences, default=None),
        "reference_window_oof_identity_sha256": reference_identity,
        "variant_window_oof_identity_sha256": variant_identity,
        "reference_window_oof_probability_sha256": reference_probability,
        "variant_window_oof_probability_sha256": variant_probability,
        "report_view_factor_window_oof_probabilities_identical": exact,
        "window_oof_identity_audit_status": (
            "exact_row_identity_and_bitwise_probability_match"
            if exact else "mismatch_or_incomplete"
        ),
    }


def _stage3_star_report_tables(
    collected: CollectedStudy,
    case_summary: Sequence[Mapping[str, Any]],
    aggregation_view_comparison: Sequence[Mapping[str, Any]],
    aggregation_view_fold_metrics: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]], list[Mapping[str, Any]],
    list[Mapping[str, Any]], list[Mapping[str, Any]], list[str],
]:
    """Build 16 absolutes and only the 14 legal same-model B0 contrasts."""

    bridge = collected.plan.get("legacy_bridge")
    if not isinstance(bridge, Mapping) or bridge.get("design") != _STAGE3_STAR_DESIGN:
        return [], [], [], [], []
    try:
        profiles = [
            dict(row) for row in bridge.get("profiles", ())
            if isinstance(row, Mapping)
        ]
        by_display = {str(row.get("case_id")): row for row in profiles}
        catalog_ids = {str(row.get("catalog_case_id")) for row in profiles}
        execution = tuple(map(str, bridge.get("execution_order", ())))
        budget = bridge.get("budget")
        if not isinstance(budget, Mapping):
            raise ValueError("centered star requires a budget")
        expected = tuple(
            (int(repeat), int(fold))
            for repeat in budget.get("repeat_indices", ())
            for fold in budget.get("fold_indices", ())
        )
        if (
            len(profiles) != 16 or len(by_display) != 16
            or len(catalog_ids) != 16 or "" in catalog_ids
            or not expected or len(set(expected)) != len(expected)
            or len(execution) != 16 or set(execution) != set(by_display)
        ):
            raise ValueError(
                "centered star requires 16 unique cases and a non-empty unique "
                "repeat/fold budget"
            )
        if any(
            not isinstance(row.get("controls"), Mapping)
            or row["controls"].get("primary_report_aggregation_view")
            not in (WINDOW_BALANCED_TO_PARTICIPANT, LINE_B_EQUAL_ROLE_FAMILIES)
            for row in profiles
        ):
            raise ValueError("every profile requires full controls and a W/B native view")
        references = [row for row in profiles if row.get("reference_case_id") is None]
        variants = [row for row in profiles if row.get("reference_case_id") is not None]
        models = {str(row.get("model_id")) for row in profiles}
        required_profiles = {f"B{level}" for level in range(8)}
        if (
            len(references) != 2 or len(variants) != 14 or len(models) != 2
            or {str(row.get("profile_id")) for row in references} != {"B0"}
            or any(
                {str(row.get("profile_id")) for row in profiles if str(row.get("model_id")) == model}
                != required_profiles for model in models
            )
            or any(
                str(row.get("reference_case_id")) not in by_display
                or by_display[str(row["reference_case_id"])].get("model_id") != row.get("model_id")
                for row in variants
            )
        ):
            raise ValueError("centered-star contrasts must be B0-centred within each model")
        declared = bridge.get("centered_comparisons")
        expected_comparisons = [{
            "model_id": row.get("model_id"),
            "reference_case_id": row.get("reference_case_id"),
            "variant_case_id": row.get("case_id"),
            "profile_id": row.get("profile_id"),
            "factor_id": row.get("factor_id"),
            "changed_control_paths": list(row.get("changed_control_paths", ())),
        } for row in variants]
        if (
            not isinstance(declared, (list, tuple))
            or [dict(row) for row in declared] != expected_comparisons
        ):
            raise ValueError("centered_comparisons disagrees with profiles")
    except (KeyError, TypeError, ValueError) as error:
        return [], [], [], [], [
            "Stage-3 centered-star reports are N/A because the materialized "
            f"design contract is invalid: {type(error).__name__}: {error}"
        ]

    summary = {str(row.get("case_id")): row for row in case_summary}
    views = {
        (str(row.get("case_id")), str(row.get("aggregation_view"))): row
        for row in aggregation_view_comparison
    }
    fold_views = {
        (
            str(row.get("case_id")), str(row.get("aggregation_view")),
            int(row.get("repeat", -1)), int(row.get("fold", -1)),
        ): row for row in aggregation_view_fold_metrics
    }

    def by_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
        output: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            output.setdefault(str(row.get("case_id")), []).append(row)
        return output

    cell_rows = by_case(collected.cell_rows)
    windows = by_case(collected.window_oof_rows)
    files = by_case(collected.file_oof_rows)
    evidence = {}
    for profile in profiles:
        catalog = str(profile["catalog_case_id"])
        oof = windows.get(catalog) or files.get(catalog, [])
        evidence[str(profile["case_id"])] = _star_cell_evidence(
            cell_rows.get(catalog, []), oof, expected
        )
    cross_model_controls = {}
    for profile in profiles:
        peers = [
            row for row in profiles
            if row.get("profile_id") == profile.get("profile_id")
        ]
        cross_model_controls[str(profile["case_id"])] = (
            len(peers) == 2
            and len({stable_payload_sha256(row["controls"]) for row in peers}) == 1
        )

    absolute: list[dict[str, Any]] = []
    for profile in profiles:
        case_id, catalog = str(profile["case_id"]), str(profile["catalog_case_id"])
        controls = profile["controls"]
        reference = (
            profile if profile.get("reference_case_id") is None
            else by_display[str(profile["reference_case_id"])]
        )
        actual_paths = (
            () if reference is profile
            else _changed_control_paths(reference["controls"], controls)
        )
        declared_paths = _normalized_control_paths(profile.get("changed_control_paths"))
        calculated_hash = stable_payload_sha256(controls)
        declared_hash = profile.get("controls_sha256")
        current_summary, current_evidence = summary.get(catalog, {}), evidence[case_id]
        row: dict[str, Any] = {
            "model": profile.get("model_id"),
            "profile": profile.get("profile_id"),
            "factor_id": profile.get("factor_id"),
            "case_id": case_id,
            "catalog_case_id": catalog,
            "reference_case_id": profile.get("reference_case_id"),
            "native_aggregation_view": controls["primary_report_aggregation_view"],
            "case_status": current_summary.get("status", "not_run"),
            "complete_for_requested_execution": bool(
                current_summary.get("complete_for_requested_execution", False)
            ),
            "expected_cell_count": len(expected),
            "passed_cell_count": sum(
                bool(value["cell_passed"]) for value in current_evidence.values()
            ),
            "declared_changed_control_paths": declared_paths,
            "actual_changed_control_paths": actual_paths,
            "single_factor_audit": (
                "baseline_no_contrast" if reference is profile
                else "pass_exact_declared_paths"
                if actual_paths == declared_paths and actual_paths
                else "fail_changed_paths_mismatch"
            ),
            "controls_sha256": calculated_hash,
            "declared_controls_sha256": declared_hash,
            "declared_controls_hash_matches": bool(
                isinstance(declared_hash, str) and len(declared_hash) == 64
                and set(declared_hash.lower()) <= set("0123456789abcdef")
                and declared_hash == calculated_hash
            ),
            "cross_model_profile_controls_match": cross_model_controls[case_id],
            "interpretation": profile.get("interpretation"),
        }
        for view, short in _STAGE3_STAR_VIEWS.items():
            metric_row = views.get((catalog, view), {})
            for metric, prefix, source in _STAGE3_STAR_METRICS:
                del metric
                row[f"{prefix}_{short}_sensitivity"] = _number(metric_row.get(source))
        native_short = _STAGE3_STAR_VIEWS[row["native_aggregation_view"]]
        for metric, prefix, _source in _STAGE3_STAR_METRICS:
            row[f"native_{metric}"] = row[f"{prefix}_{native_short}_sensitivity"]
        row["native_metrics_available"] = all(
            row[f"native_{metric}"] is not None
            for metric, _prefix, _source in _STAGE3_STAR_METRICS
        )
        absolute.append(row)

    absolute_by_case = {str(row["case_id"]): row for row in absolute}
    contrasts: list[dict[str, Any]] = []
    fold_contrasts: list[dict[str, Any]] = []
    reason_codes = {
        "all_requested_cells": "not_all_requested_cells_passed_with_oof",
        "seed_match": "seed_mismatch_or_missing",
        "split_hash_match": "split_hash_mismatch_or_missing",
        "heldout_roster_hash_match": "heldout_roster_mismatch_or_missing",
        "single_factor_match": "single_factor_path_audit_failed",
        "controls_integrity_match": "controls_hash_or_cross_model_profile_mismatch",
        "all_requested_native_fold_metrics_available": "native_fold_metrics_incomplete",
        "cases_complete": "case_not_passed_or_incomplete",
    }
    for variant in variants:
        variant_id, reference_id = str(variant["case_id"]), str(variant["reference_case_id"])
        reference, left_abs, right_abs = (
            by_display[reference_id], absolute_by_case[reference_id], absolute_by_case[variant_id]
        )
        left_evidence, right_evidence = evidence[reference_id], evidence[variant_id]
        actual_paths = tuple(right_abs["actual_changed_control_paths"])
        declared_paths = tuple(right_abs["declared_changed_control_paths"])
        complete = all(
            left_evidence[key]["cell_passed"] and right_evidence[key]["cell_passed"]
            and left_evidence[key]["oof_present"] and right_evidence[key]["oof_present"]
            and left_evidence[key]["unexpected_cell_count"] == 0
            and right_evidence[key]["unexpected_cell_count"] == 0
            for key in expected
        )

        def match(fields: Sequence[str]) -> bool:
            return bool(complete and all(
                left_evidence[key][field] is not None
                and left_evidence[key][field] == right_evidence[key][field]
                for key in expected for field in fields
            ))

        reference_view = str(reference["controls"]["primary_report_aggregation_view"])
        variant_view = str(variant["controls"]["primary_report_aggregation_view"])
        pairs = {
            key: (
                fold_views.get((str(reference["catalog_case_id"]), reference_view, *key)),
                fold_views.get((str(variant["catalog_case_id"]), variant_view, *key)),
            ) for key in expected
        }
        native_folds = all(
            left is not None and right is not None
            and all(
                _number(side.get(metric)) is not None
                for side in (left, right)
                for metric, _prefix, _source in _STAGE3_STAR_METRICS
            )
            for left, right in pairs.values()
        )
        checks = {
            "all_requested_cells": complete,
            "seed_match": match(("split_seed", "training_seed")),
            "split_hash_match": match(("manifest_hash", "fold_hash")),
            "heldout_roster_hash_match": match(("roster_hash",)),
            "single_factor_match": actual_paths == declared_paths and bool(actual_paths),
            "controls_integrity_match": bool(
                left_abs["declared_controls_hash_matches"]
                and right_abs["declared_controls_hash_matches"]
                and left_abs["cross_model_profile_controls_match"]
                and right_abs["cross_model_profile_controls_match"]
            ),
            "all_requested_native_fold_metrics_available": native_folds,
            "cases_complete": bool(
                left_abs["case_status"] == right_abs["case_status"] == "passed"
                and left_abs["complete_for_requested_execution"]
                and right_abs["complete_for_requested_execution"]
                and left_abs["native_metrics_available"]
                and right_abs["native_metrics_available"]
            ),
        }
        is_report_view_factor = actual_paths == (
            "controls.primary_report_aggregation_view",
        )
        probability_audit: Mapping[str, Any] = {
            "window_oof_identity_audit_status": "not_applicable_non_report_view_factor"
        }
        training_identity = None
        if is_report_view_factor:
            blocked = set(actual_paths)
            training_identity = (
                {key: value for key, value in _flatten_control_paths(reference["controls"]).items() if key not in blocked}
                == {key: value for key, value in _flatten_control_paths(variant["controls"]).items() if key not in blocked}
            )
            probability_audit = _star_probability_audit(
                windows.get(str(reference["catalog_case_id"]), []),
                windows.get(str(variant["catalog_case_id"]), []),
                expected,
            )
            checks["report_view_factor_training_controls_identical"] = training_identity
            checks["report_view_factor_window_oof_probabilities_identical"] = bool(
                probability_audit["report_view_factor_window_oof_probabilities_identical"]
            )
        available = all(checks.values())
        reasons = [
            reason_codes.get(name, name) for name, passed in checks.items() if not passed
        ]
        contrast: dict[str, Any] = {
            "model": variant.get("model_id"),
            "factor_id": variant.get("factor_id"),
            "reference_profile": reference.get("profile_id"),
            "variant_profile": variant.get("profile_id"),
            "reference_case_id": reference_id,
            "variant_case_id": variant_id,
            "reference_catalog_case_id": reference.get("catalog_case_id"),
            "variant_catalog_case_id": variant.get("catalog_case_id"),
            "reference_native_aggregation_view": reference_view,
            "variant_native_aggregation_view": variant_view,
            "native_comparison_semantics": (
                "B0_window_endpoint_to_variant_line_b_endpoint"
                if is_report_view_factor else "same_window_balanced_endpoint"
            ),
            "declared_changed_control_paths": declared_paths,
            "actual_changed_control_paths": actual_paths,
            "single_factor_audit": (
                "pass_exact_declared_paths"
                if checks["single_factor_match"] else "fail_changed_paths_mismatch"
            ),
            **checks,
            "contrast_metrics_available": available,
            "unavailable_reasons": reasons,
            "fold_delta_inference": "descriptive_only_no_ci_no_significance",
            "shared_reference_warning": "same_model_B0_is_reused_across_seven_correlated_contrasts",
            "report_view_factor_training_controls_identical": training_identity,
            **probability_audit,
        }
        for metric, prefix, _source in _STAGE3_STAR_METRICS:
            left, right = left_abs[f"native_{metric}"], right_abs[f"native_{metric}"]
            contrast[f"reference_native_{metric}"] = left
            contrast[f"variant_native_{metric}"] = right
            contrast[f"delta_native_{metric}"] = right - left if available else None
            for short in _STAGE3_STAR_VIEWS.values():
                left_view = _number(left_abs.get(f"{prefix}_{short}_sensitivity"))
                right_view = _number(right_abs.get(f"{prefix}_{short}_sensitivity"))
                contrast[f"delta_{metric}_{short}_sensitivity_only"] = (
                    right_view - left_view
                    if available and left_view is not None and right_view is not None
                    else None
                )
        contrasts.append(contrast)
        for repeat, fold in expected:
            pair = pairs[(repeat, fold)]
            fold_row: dict[str, Any] = {
                "model": variant.get("model_id"),
                "factor_id": variant.get("factor_id"),
                "reference_profile": reference.get("profile_id"),
                "variant_profile": variant.get("profile_id"),
                "reference_case_id": reference_id,
                "variant_case_id": variant_id,
                "repeat": repeat,
                "fold": fold,
                "reference_native_aggregation_view": reference_view,
                "variant_native_aggregation_view": variant_view,
                "contrast_metrics_available": available,
                "inference": "descriptive_only_no_ci_no_significance",
            }
            for metric, _prefix, _source in _STAGE3_STAR_METRICS:
                left = None if pair[0] is None else _number(pair[0].get(metric))
                right = None if pair[1] is None else _number(pair[1].get(metric))
                fold_row[f"reference_native_{metric}"] = left
                fold_row[f"variant_native_{metric}"] = right
                fold_row[f"delta_native_{metric}"] = (
                    right - left
                    if available and left is not None and right is not None
                    else None
                )
            fold_contrasts.append(fold_row)

    execution_rows = []
    previous = None
    for index, case_id in enumerate(execution, start=1):
        execution_rows.append({
            "execution_order": index,
            **absolute_by_case[case_id],
            "previous_execution_case_id": previous,
            "execution_transition": "start" if previous is None else f"{previous}->{case_id}",
            "execution_transition_is_ablation": False,
            "execution_interpretation": "absolute_metrics_only_no_execution_order_delta",
        })
        previous = case_id
    repeat_count = len({repeat for repeat, _fold in expected})
    fold_count = len({fold for _repeat, fold in expected})
    notes = [
        f"Stage-3 has {repeat_count} repeat(s) and {fold_count} grouped folds; "
        "fold deltas are descriptive only, with no per-fold independence or "
        "significance claim.",
        "Each model reuses one B0 across seven correlated contrasts.",
        "B7 compares the B0 W endpoint with Line B; W/A/B same-view deltas remain sensitivity-only.",
        "B0/B7 OOF uses sorted row identities and zero probability tolerance, not Parquet bytes or row order.",
        f"The 16-case two-model B0-B7 design contains "
        f"{16 * len(expected)} repeat/fold cells.",
    ]
    return absolute, contrasts, fold_contrasts, execution_rows, notes


def _stage3_star_presentation_tables(
    collected: CollectedStudy,
    absolute: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    aggregation_view_repeat_metrics: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[str],
]:
    """Build two within-model B0--B7 tables and one side-by-side table.

    Human-facing means and population SDs use the declared repeat-level native
    participant-OOF endpoints. Variant deltas are paired to the same model's B0
    by repeat. The cross-model table is deliberately descriptive and is never
    relabelled as a causal ablation.
    """

    if not absolute:
        return [], [], [], []
    bridge = collected.plan.get("legacy_bridge")
    if not isinstance(bridge, Mapping) or bridge.get("design") != _STAGE3_STAR_DESIGN:
        return [], [], [], []
    budget = bridge.get("budget")
    if not isinstance(budget, Mapping):
        return [], [], [], [
            "Stage-3 presentation tables are N/A because the repeat budget is missing."
        ]
    try:
        expected_repeats = tuple(int(value) for value in budget["repeat_indices"])
    except (KeyError, TypeError, ValueError):
        return [], [], [], [
            "Stage-3 presentation tables are N/A because repeat indices are invalid."
        ]
    if not expected_repeats or len(set(expected_repeats)) != len(expected_repeats):
        return [], [], [], [
            "Stage-3 presentation tables are N/A because repeat indices are empty or duplicated."
        ]

    model_order = ("InceptionTimeFull", "CompactCNN1D")
    profile_order = tuple(f"B{level}" for level in range(8))
    absolute_by_model_profile = {
        (str(row.get("model")), str(row.get("profile"))): row
        for row in absolute
    }
    contrast_by_model_profile = {
        (str(row.get("model")), str(row.get("variant_profile"))): row
        for row in contrasts
    }
    repeats = {
        (
            str(row.get("case_id")),
            str(row.get("aggregation_view")),
            int(row.get("repeat", -1)),
        ): row
        for row in aggregation_view_repeat_metrics
        if row.get("repeat") is not None
    }
    metrics = ("balanced_accuracy", "macro_f1", "worst_class_f1")
    native_series: dict[tuple[str, str, str], tuple[float, ...] | None] = {}

    def complete_series(
        absolute_row: Mapping[str, Any], metric: str
    ) -> tuple[float, ...] | None:
        values: list[float] = []
        for repeat in expected_repeats:
            row = repeats.get(
                (
                    str(absolute_row.get("catalog_case_id")),
                    str(absolute_row.get("native_aggregation_view")),
                    repeat,
                )
            )
            value = None if row is None else _number(row.get(metric))
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    model_tables: dict[str, list[Mapping[str, Any]]] = {
        model: [] for model in model_order
    }
    notes: list[str] = []
    for model in model_order:
        baseline = absolute_by_model_profile.get((model, "B0"))
        if baseline is None:
            notes.append(
                f"Stage-3 {model} B0--B7 table is N/A because B0 is missing."
            )
            continue
        baseline_series = {
            metric: complete_series(baseline, metric) for metric in metrics
        }
        for profile in profile_order:
            absolute_row = absolute_by_model_profile.get((model, profile))
            if absolute_row is None:
                notes.append(
                    f"Stage-3 {model} table is incomplete because {profile} is missing."
                )
                continue
            contrast = (
                None
                if profile == "B0"
                else contrast_by_model_profile.get((model, profile))
            )
            row: dict[str, Any] = {
                "model": model,
                "profile": profile,
                "factor_id": absolute_row.get("factor_id"),
                "reference_profile": "B0",
                "native_aggregation_view": absolute_row.get(
                    "native_aggregation_view"
                ),
                "comparison_type": (
                    "baseline"
                    if profile == "B0"
                    else "within_model_B0_centered_ablation"
                ),
                "repeat_count": len(expected_repeats),
                "passed_cell_count": absolute_row.get("passed_cell_count"),
                "changed_control_paths": absolute_row.get(
                    "actual_changed_control_paths"
                ),
                "single_factor_audit": absolute_row.get("single_factor_audit"),
                "contrast_metrics_available": (
                    bool(absolute_row.get("native_metrics_available"))
                    if profile == "B0"
                    else bool(
                        contrast is not None
                        and contrast.get("contrast_metrics_available")
                    )
                ),
                "inference": "descriptive_repeat_mean_population_sd",
            }
            for metric in metrics:
                series = complete_series(absolute_row, metric)
                native_series[(model, profile, metric)] = series
                baseline_values = baseline_series[metric]
                absolute_available = series is not None
                deltas = (
                    tuple(0.0 for _ in expected_repeats)
                    if profile == "B0" and absolute_available
                    else tuple(
                        right - left
                        for left, right in zip(baseline_values, series)
                    )
                    if (
                        series is not None
                        and baseline_values is not None
                        and row["contrast_metrics_available"]
                    )
                    else None
                )
                row[f"native_{metric}"] = (
                    float(fmean(series)) if series is not None else None
                )
                row[f"native_{metric}_sd"] = (
                    _sd(series) if series is not None else None
                )
                row[f"delta_vs_B0_{metric}"] = (
                    float(fmean(deltas)) if deltas is not None else None
                )
                row[f"delta_vs_B0_{metric}_sd"] = (
                    _sd(deltas) if deltas is not None else None
                )
            row["repeat_metrics_complete"] = all(
                native_series.get((model, profile, metric)) is not None
                for metric in metrics
            )
            model_tables[model].append(row)

    side_by_side: list[Mapping[str, Any]] = []
    inception_model, cnn_model = model_order
    for profile in profile_order:
        inception = absolute_by_model_profile.get((inception_model, profile))
        cnn = absolute_by_model_profile.get((cnn_model, profile))
        if inception is None or cnn is None:
            notes.append(
                f"Stage-3 cross-model row {profile} is N/A because one model is missing."
            )
            continue
        row: dict[str, Any] = {
            "profile": profile,
            "factor_id": inception.get("factor_id"),
            "native_aggregation_view": inception.get("native_aggregation_view"),
            "comparison_type": "matched_architecture_comparison_not_ablation",
            "repeat_count": len(expected_repeats),
            "cross_model_profile_controls_match": bool(
                inception.get("cross_model_profile_controls_match")
                and cnn.get("cross_model_profile_controls_match")
            ),
            "both_cases_complete": bool(
                inception.get("complete_for_requested_execution")
                and cnn.get("complete_for_requested_execution")
            ),
            "inference": "descriptive_paired_repeat_difference_no_significance",
        }
        for metric in metrics:
            inception_values = native_series.get(
                (inception_model, profile, metric)
            )
            cnn_values = native_series.get((cnn_model, profile, metric))
            paired = (
                tuple(
                    inception_value - cnn_value
                    for inception_value, cnn_value in zip(
                        inception_values, cnn_values
                    )
                )
                if inception_values is not None and cnn_values is not None
                else None
            )
            for prefix, values in (
                ("inception", inception_values),
                ("cnn", cnn_values),
                ("inception_minus_cnn", paired),
            ):
                row[f"{prefix}_{metric}"] = (
                    float(fmean(values)) if values is not None else None
                )
                row[f"{prefix}_{metric}_sd"] = (
                    _sd(values) if values is not None else None
                )
        row["comparison_metrics_available"] = bool(
            row["cross_model_profile_controls_match"]
            and row["both_cases_complete"]
            and all(
                native_series.get((model, profile, metric)) is not None
                for model in model_order
                for metric in metrics
            )
        )
        side_by_side.append(row)

    return (
        model_tables[inception_model],
        model_tables[cnn_model],
        side_by_side,
        notes,
    )


@dataclass(frozen=True)
class StudyAnalysis:
    """All normalized tables consumed by CSV, Markdown, HTML, and plots."""

    case_summary: tuple[Mapping[str, Any], ...]
    metric_distribution_summary: tuple[Mapping[str, Any], ...]
    repeat_metrics: tuple[Mapping[str, Any], ...]
    fold_metrics: tuple[Mapping[str, Any], ...]
    per_class_metrics: tuple[Mapping[str, Any], ...]
    confusion_matrices: tuple[Mapping[str, Any], ...]
    confusion_counts: tuple[Mapping[str, Any], ...]
    confusion_row_normalized: tuple[Mapping[str, Any], ...]
    calibration_bins: tuple[Mapping[str, Any], ...]
    paired_deltas: tuple[Mapping[str, Any], ...]
    paired_participant_inference: tuple[Mapping[str, Any], ...]
    aggregation_line_comparison: tuple[Mapping[str, Any], ...]
    aggregation_line_repeat_metrics: tuple[Mapping[str, Any], ...]
    aggregation_line_per_class_metrics: tuple[Mapping[str, Any], ...]
    aggregation_view_comparison: tuple[Mapping[str, Any], ...]
    aggregation_view_repeat_metrics: tuple[Mapping[str, Any], ...]
    aggregation_view_per_class_metrics: tuple[Mapping[str, Any], ...]
    aggregation_view_confusion_matrices: tuple[Mapping[str, Any], ...]
    aggregation_hierarchy_coverage: tuple[Mapping[str, Any], ...]
    legacy_bridge_numeric_ablation_report: tuple[Mapping[str, Any], ...]
    legacy_bridge_execution_order_report: tuple[Mapping[str, Any], ...]
    coverage: tuple[Mapping[str, Any], ...]
    route_role_coverage: tuple[Mapping[str, Any], ...]
    quality_distributions: tuple[Mapping[str, Any], ...]
    predictive_leaderboard: tuple[Mapping[str, Any], ...]
    worst_class_f1_stability: tuple[Mapping[str, Any], ...]
    incomplete_cases: tuple[Mapping[str, Any], ...]
    deployment_table: tuple[Mapping[str, Any], ...]
    notes: tuple[str, ...]
    denoiser_hr_record_pairs: tuple[Mapping[str, Any], ...] = ()
    denoiser_hr_comparison: tuple[Mapping[str, Any], ...] = ()
    repeat_per_class_metrics: tuple[Mapping[str, Any], ...] = ()
    per_class_metric_distribution_summary: tuple[Mapping[str, Any], ...] = ()
    aggregation_view_fold_metrics: tuple[Mapping[str, Any], ...] = ()
    stage3_star_absolute: tuple[Mapping[str, Any], ...] = ()
    stage3_star_contrasts: tuple[Mapping[str, Any], ...] = ()
    stage3_star_fold_contrasts: tuple[Mapping[str, Any], ...] = ()
    stage3_star_execution: tuple[Mapping[str, Any], ...] = ()
    stage3_star_inception_comparison: tuple[Mapping[str, Any], ...] = ()
    stage3_star_cnn_comparison: tuple[Mapping[str, Any], ...] = ()
    stage3_star_model_comparison: tuple[Mapping[str, Any], ...] = ()
    classification_prediction_scores: tuple[Mapping[str, Any], ...] = ()
    classification_roc_curves: tuple[Mapping[str, Any], ...] = ()
    classification_prediction_tsne: tuple[Mapping[str, Any], ...] = ()
    classification_diagnostic_status: tuple[Mapping[str, Any], ...] = ()
    classifier_per_class_results: tuple[Mapping[str, Any], ...] = ()


def analyze_study(collected: CollectedStudy) -> StudyAnalysis:
    """Build descriptive comparison tables without selecting a winner."""

    manifest_cases = _manifest_cases(collected)
    statuses = _record_statuses(collected)
    trusted = _trusted_by_case(collected)
    evaluation_policy_by_case = {
        str(row.get("case_id")): dict(row["evaluation_statistics"])
        for row in collected.resolved_aggregation_configs
        if isinstance(row.get("evaluation_statistics"), Mapping)
    }
    oof_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in collected.subject_oof_rows:
        oof_by_case.setdefault(str(row.get("case_id")), []).append(row)
    cell_repeat, fold_rows = _cell_repeat_rows(collected.cell_rows)
    fallback_repeat_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in cell_repeat:
        fallback_repeat_by_case.setdefault(str(row["case_id"]), []).append(row)
    fold_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in fold_rows:
        fold_by_case.setdefault(str(row["case_id"]), []).append(row)

    execution = collected.plan.get("execution", {})
    execution = execution if isinstance(execution, Mapping) else {}
    requested_repeats = execution.get("repeats")
    requested_folds = execution.get("folds")
    expected_repeat_count = (
        len(requested_repeats)
        if isinstance(requested_repeats, (list, tuple))
        else None
    )
    expected_fold_cell_count = (
        len(requested_repeats) * len(requested_folds)
        if isinstance(requested_repeats, (list, tuple))
        and isinstance(requested_folds, (list, tuple))
        else None
    )

    all_repeat: list[Mapping[str, Any]] = []
    all_per_class: list[Mapping[str, Any]] = []
    repeat_per_class: list[Mapping[str, Any]] = []
    matrices: list[Mapping[str, Any]] = []
    calibration: list[Mapping[str, Any]] = []
    summaries: list[Mapping[str, Any]] = []
    metric_distributions: list[Mapping[str, Any]] = []
    coverage_rows: list[Mapping[str, Any]] = []
    notes = list(collected.limitations)

    report_options = collected.plan.get("report", {})
    report_options = report_options if isinstance(report_options, Mapping) else {}
    classification_prediction_scores = normalize_classification_rows(
        collected.subject_oof_rows,
        evaluation_id="participant_outer_oof",
        aggregation_level="participant",
    )
    classifier_per_class_results = tuple(
        {
            **row,
            "case_execution_status": statuses.get(
                str(row.get("classifier_id")), "unknown"
            ),
        }
        for row in classification_per_class_metric_rows(
            classification_prediction_scores,
            class_names=CANONICAL_CLASS_NAMES,
        )
    )
    abstention_aware_classifier_per_class_results: list[Mapping[str, Any]] = []
    classification_roc_curves = classification_roc_curve_rows(
        classification_prediction_scores,
        macro_grid_points=int(
            report_options.get("classification_roc_macro_grid_points", 201)
        ),
    )
    classification_prediction_tsne = classification_tsne_rows(
        classification_prediction_scores,
        random_state=int(
            report_options.get("classification_tsne_random_state", 42)
        ),
        perplexity=float(
            report_options.get("classification_tsne_perplexity", 30.0)
        ),
        max_samples=int(
            report_options.get("classification_tsne_max_samples", 5000)
        ),
    )
    classification_diagnostic_status = classification_diagnostic_status_rows(
        tuple(manifest_cases),
        classification_prediction_scores,
        classification_roc_curves,
        classification_prediction_tsne,
    )
    if classification_prediction_tsne:
        notes.append(
            "Classification t-SNE is a report-only embedding of persisted OOF "
            "prediction-probability vectors, not a hidden-feature embedding and "
            "not evidence of separability in the model representation space."
        )

    for case_id, case in manifest_cases.items():
        oof_rows = oof_by_case.get(case_id, [])
        trusted_row = trusted.get(case_id, {})
        participant_cluster_fields: dict[str, Any] = {}
        cluster_policy = evaluation_policy_by_case.get(case_id)
        participant_cluster_ci_reason = ""
        if oof_rows and cluster_policy is not None:
            try:
                participant_cluster_fields = _participant_cluster_interval_fields(
                    oof_rows,
                    policy=cluster_policy,
                )
            except (KeyError, TypeError, ValueError) as error:
                participant_cluster_ci_reason = (
                    f"N/A_{type(error).__name__}: {error}"
                )
                notes.append(
                    f"{case_id}: participant-cluster CI is N/A: "
                    f"{type(error).__name__}: {error}"
                )
        elif not oof_rows:
            participant_cluster_ci_reason = (
                "N/A_no_persisted_participant_outer_oof_predictions"
            )
        else:
            participant_cluster_ci_reason = (
                "N/A_no_registered_participant_cluster_bootstrap_policy"
            )
        cluster_source = {**dict(trusted_row), **participant_cluster_fields}
        participant_cluster_ci_available = all(
            _number(cluster_source.get(field)) is not None
            for field in (
                "participant_cluster_balanced_accuracy_ci95_low",
                "participant_cluster_balanced_accuracy_ci95_high",
                "participant_cluster_macro_f1_ci95_low",
                "participant_cluster_macro_f1_ci95_high",
                "participant_cluster_macro_roc_auc_ovr_ci95_low",
                "participant_cluster_macro_roc_auc_ovr_ci95_high",
            )
        )
        if participant_cluster_ci_available:
            participant_cluster_ci_reason = ""
        repeat_rows: list[Mapping[str, Any]] = []
        case_folds = fold_by_case.get(case_id, [])
        if oof_rows:
            repeat_groups: dict[int, list[Mapping[str, Any]]] = {}
            for row in oof_rows:
                repeat_groups.setdefault(int(row.get("repeat", -1)), []).append(row)
            for repeat, values in sorted(repeat_groups.items()):
                conditional = _oof_metric_row(case_id, values, repeat=repeat)
                aware = _oof_abstention_metric_row(
                    case_id, values, repeat=repeat
                )
                if aware is None:
                    continue
                if conditional is not None:
                    repeat_per_class.extend(
                        {
                            **class_row,
                            "metric_source": "participant_oof_per_repeat",
                        }
                        for class_row in conditional["per_class"]
                    )
                repeat_rows.append(
                    (
                        {
                            key: value
                            for key, value in conditional.items()
                            if key not in {"per_class", "confusion_matrix"}
                        }
                        if conditional is not None
                        else {
                            "case_id": case_id,
                            "repeat": repeat,
                            "balanced_accuracy": None,
                            "macro_f1": None,
                            "macro_roc_auc_ovr": None,
                            "macro_pr_auc_ovr": None,
                            "worst_class_recall": None,
                            "worst_class_f1": None,
                            "n_predictions": 0,
                            "class_order": list(CANONICAL_CLASS_NAMES),
                        }
                    )
                    | aware
                    | {
                        "metric_source": (
                            "participant_oof_recomputed_conditional_and_"
                            "abstention_aware"
                        )
                    }
                )
            pooled = _oof_metric_row(case_id, oof_rows, repeat=None)
            pooled_aware = _oof_abstention_metric_row(
                case_id, oof_rows, repeat=None
            )
            if pooled is not None:
                all_per_class.extend(pooled["per_class"])
                matrices.append(
                    {
                        "case_id": case_id,
                        "class_order": pooled["class_order"],
                        "confusion_matrix": pooled["confusion_matrix"],
                        "metric_source": "participant_oof_pooled_repeats",
                    }
                )
            current_calibration, calculated_ece = _calibration_rows(
                case_id,
                oof_rows,
                bins=int(collected.plan.get("report", {}).get("calibration_bins", 10)),
            )
            calibration.extend(current_calibration)
            metric_source = "participant_oof"
        else:
            repeat_rows = fallback_repeat_by_case.get(case_id, [])
            pooled = None
            pooled_aware = None
            calculated_ece = None
            metric_source = "cell_metric_fallback_no_subject_oof"
            trusted_confusions = trusted_row.get("confusion_matrices")
            confusion_added = False
            if isinstance(trusted_confusions, Mapping):
                trusted_matrix = trusted_confusions.get(
                    "pooled_participant_repeat"
                )
                per_class = _per_class_from_confusion(case_id, trusted_matrix)
                if per_class:
                    all_per_class.extend(per_class)
                    pooled = {
                        "worst_class_recall": min(
                            float(row["recall"]) for row in per_class
                        ),
                        "worst_class_f1": min(float(row["f1"]) for row in per_class),
                    }
                    matrices.append(
                        {
                            "case_id": case_id,
                            "class_order": list(range(len(per_class))),
                            "confusion_matrix": trusted_matrix,
                            "metric_source": "config_metrics_v2_pooled_confusion",
                        }
                    )
                    confusion_added = True
            if not confusion_added:
                cell_confusion = _pooled_cell_confusion(case_folds)
                if cell_confusion is not None:
                    class_order, pooled_matrix, source_cell_count = cell_confusion
                    per_class = _per_class_from_confusion(
                        case_id,
                        pooled_matrix,
                        class_order=class_order,
                        metric_source="summed_cell_confusion_fallback",
                    )
                    if per_class:
                        all_per_class.extend(per_class)
                        pooled = {
                            "worst_class_recall": min(
                                float(row["recall"]) for row in per_class
                            ),
                            "worst_class_f1": min(
                                float(row["f1"]) for row in per_class
                            ),
                        }
                        matrices.append(
                            {
                                "case_id": case_id,
                                "class_order": class_order,
                                "confusion_matrix": pooled_matrix,
                                "metric_source": "summed_cell_confusion_fallback",
                                "source_cell_count": source_cell_count,
                            }
                        )
        cell_repeat_lookup = {
            int(row["repeat"]): row
            for row in fallback_repeat_by_case.get(case_id, ())
        }
        enriched_repeat_rows: list[Mapping[str, Any]] = []
        for row in repeat_rows:
            projected = dict(row)
            fallback = cell_repeat_lookup.get(int(row["repeat"]))
            if fallback is not None and not oof_rows:
                for key in (
                    *_ABSTENTION_AWARE_METRICS,
                    "abstention_count",
                    "abstention_counts_by_class",
                    "coverage_rate",
                ):
                    if key in fallback:
                        projected[key] = fallback[key]
            enriched_repeat_rows.append(projected)
        repeat_rows = enriched_repeat_rows
        all_repeat.extend(repeat_rows)
        repeat_ba = [row.get("balanced_accuracy") for row in repeat_rows]
        repeat_f1 = [row.get("macro_f1") for row in repeat_rows]
        repeat_roc_auc = [row.get("macro_roc_auc_ovr") for row in repeat_rows]
        repeat_pr_auc = [row.get("macro_pr_auc_ovr") for row in repeat_rows]
        repeat_abstention_metrics = {
            name: [row.get(name) for row in repeat_rows]
            for name in _ABSTENTION_AWARE_METRICS
        }
        participant_mean_coverage = _mean(
            row.get("coverage_rate") for row in repeat_rows
        )
        total_abstentions = _sum_numbers(
            row.get("abstention_count") for row in repeat_rows
        )
        mean_ba = _number(
            trusted_row.get("participant_mean_balanced_accuracy")
        )
        mean_f1 = _number(trusted_row.get("participant_mean_macro_f1"))
        aware_means = {
            name: _mean(values)
            for name, values in repeat_abstention_metrics.items()
        }
        for name in _ABSTENTION_AWARE_METRICS:
            trusted_value = _number(trusted_row.get(f"participant_mean_{name}"))
            if not oof_rows and trusted_value is not None:
                aware_means[name] = trusted_value
        aware_metric_source = (
            "participant_oof_complete_roster_recomputation"
            if oof_rows
            else "config_metrics_v2_complete_roster_recomputation"
            if any(
                _number(trusted_row.get(f"participant_mean_{name}")) is not None
                for name in _ABSTENTION_AWARE_METRICS
            )
            else "mean_cell_metrics_compatibility_fallback"
            if any(value is not None for value in aware_means.values())
            else "legacy_conditional_no_abstention_metadata"
        )
        if (
            aware_means["abstention_aware_balanced_accuracy"] is None
            and total_abstentions in {None, 0}
        ):
            aware_means["abstention_aware_balanced_accuracy"] = (
                mean_ba if mean_ba is not None else _mean(repeat_ba)
            )
            aware_means["abstention_aware_macro_recall"] = aware_means[
                "abstention_aware_balanced_accuracy"
            ]
            aware_means["abstention_aware_macro_f1"] = (
                mean_f1 if mean_f1 is not None else _mean(repeat_f1)
            )
        worst_recall = _number(trusted_row.get("worst_class_recall"))
        worst_f1 = _number(trusted_row.get("worst_class_f1"))
        conditional_worst_recall = (
            worst_recall
            if worst_recall is not None
            else (pooled or {}).get("worst_class_recall")
        )
        conditional_worst_f1 = (
            worst_f1
            if worst_f1 is not None
            else (pooled or {}).get("worst_class_f1")
        )
        aware_worst_recall = _number(
            trusted_row.get("abstention_aware_worst_class_recall")
        )
        aware_worst_f1 = _number(
            trusted_row.get("abstention_aware_worst_class_f1")
        )
        if pooled_aware is not None:
            aware_per_class = pooled_aware["abstention_aware_per_class"]
            aware_worst_recall = min(
                float(row["recall"]) for row in aware_per_class
            )
            aware_worst_f1 = min(float(row["f1"]) for row in aware_per_class)
            # Conditional and abstention-aware per-class results are different
            # estimands.  Avoid duplicating the table for routes with complete
            # coverage, but make the full-roster result explicit whenever a
            # route abstains.  An abstained observation is a false negative for
            # its true one-vs-rest class; no probability score is invented for
            # ROC/PR calculations.
            if int(pooled_aware.get("abstention_count", 0)) > 0:
                abstained_total = int(pooled_aware["abstention_count"])
                total = sum(int(row["support"]) for row in aware_per_class)
                retained_total = total - abstained_total
                for class_row in aware_per_class:
                    true_positive = int(class_row["true_positive"])
                    false_positive = int(class_row["false_positive"])
                    false_negative = int(class_row["false_negative"])
                    true_negative = (
                        total
                        - true_positive
                        - false_positive
                        - false_negative
                    )
                    if true_negative < 0:
                        raise ValueError(
                            f"{case_id}: abstention-aware one-vs-rest counts "
                            "exceed the pooled participant roster"
                        )
                    specificity_denominator = true_negative + false_positive
                    specificity = (
                        float(true_negative / specificity_denominator)
                        if specificity_denominator
                        else None
                    )
                    recall = float(class_row["recall"])
                    abstention_aware_classifier_per_class_results.append(
                        {
                            "classifier_id": case_id,
                            "evaluation_id": (
                                "participant_outer_oof_abstention_aware"
                            ),
                            "aggregation_level": "participant",
                            "class_label": int(class_row["label"]),
                            "class_name": CANONICAL_CLASS_NAMES.get(
                                int(class_row["label"]),
                                str(class_row["label"]),
                            ),
                            "true_positive": true_positive,
                            "false_positive": false_positive,
                            "true_negative": true_negative,
                            "false_negative": false_negative,
                            "support": int(class_row["support"]),
                            "retained_support": int(
                                class_row["retained_support"]
                            ),
                            "abstention_count": int(
                                class_row["abstention_count"]
                            ),
                            "predicted_support": (
                                true_positive + false_positive
                            ),
                            "observation_count": total,
                            "input_observation_count": total,
                            "retained_observation_count": retained_total,
                            "excluded_observation_count": abstained_total,
                            "precision": float(class_row["precision"]),
                            "sensitivity": recall,
                            "recall": recall,
                            "specificity": specificity,
                            "balanced_accuracy_ovr": (
                                (recall + specificity) / 2.0
                                if specificity is not None
                                else None
                            ),
                            "f1": float(class_row["f1"]),
                            "roc_auc_ovr": None,
                            "pr_auc_ovr": None,
                            "probability_metric_applicability": (
                                "N/A_abstained_observations_have_no_frozen_"
                                "class_probability"
                            ),
                            "result_applicability": "available",
                            "case_execution_status": statuses.get(
                                case_id, "unknown"
                            ),
                            "metric_scope": (
                                "one_vs_rest_abstention_aware_full_roster;_"
                                "abstentions_are_false_negatives_for_their_"
                                "true_class"
                            ),
                            "metric_source": (
                                "participant_oof_pooled_repeats_with_"
                                "abstentions_as_true_class_false_negatives"
                            ),
                            "prediction_rule_source": (
                                "persisted_decision_rule_for_retained_rows;_"
                                "abstention_rule_for_excluded_rows"
                            ),
                        }
                    )
        elif total_abstentions in {None, 0}:
            aware_worst_recall = (
                aware_worst_recall
                if aware_worst_recall is not None
                else conditional_worst_recall
            )
            aware_worst_f1 = (
                aware_worst_f1
                if aware_worst_f1 is not None
                else conditional_worst_f1
            )
        ece = _number(trusted_row.get("expected_calibration_error"))
        variability = trusted_row.get("variability")
        variability = (
            variability if isinstance(variability, Mapping) else {}
        )
        ba_statistics = _descriptive_statistics(repeat_ba)
        f1_statistics = _descriptive_statistics(repeat_f1)
        roc_auc_statistics = _descriptive_statistics(repeat_roc_auc)
        pr_auc_statistics = _descriptive_statistics(repeat_pr_auc)
        aware_ba_statistics = _descriptive_statistics(
            repeat_abstention_metrics["abstention_aware_balanced_accuracy"]
        )
        aware_f1_statistics = _descriptive_statistics(
            repeat_abstention_metrics["abstention_aware_macro_f1"]
        )
        metric_distributions.extend(
            (
                {
                    "case_id": case_id,
                    "metric": "balanced_accuracy",
                    "metric_source": metric_source,
                    **ba_statistics,
                },
                {
                    "case_id": case_id,
                    "metric": "macro_f1",
                    "metric_source": metric_source,
                    **f1_statistics,
                },
                {
                    "case_id": case_id,
                    "metric": "macro_roc_auc_ovr",
                    "metric_source": metric_source,
                    **roc_auc_statistics,
                },
                {
                    "case_id": case_id,
                    "metric": "macro_pr_auc_ovr",
                    "metric_source": metric_source,
                    **pr_auc_statistics,
                },
                {
                    "case_id": case_id,
                    "metric": "abstention_aware_balanced_accuracy",
                    "metric_source": metric_source,
                    **aware_ba_statistics,
                },
                {
                    "case_id": case_id,
                    "metric": "abstention_aware_macro_f1",
                    "metric_source": metric_source,
                    **aware_f1_statistics,
                },
            )
        )
        status = statuses.get(case_id, "not_run")
        auxiliary_motion_valid_oof, motion_ranking_interpretation = (
            _case_motion_evidence_scope(collected, case_id)
        )
        passed_fold_cell_count = sum(
            str(row.get("status")) == "passed" for row in case_folds
        )
        incompleteness_reasons: list[str] = []
        if status != "passed":
            incompleteness_reasons.append(f"case_status={status}")
        if (
            expected_repeat_count is not None
            and len(repeat_rows) != expected_repeat_count
        ):
            incompleteness_reasons.append(
                "repeat_metric_count="
                f"{len(repeat_rows)}/{expected_repeat_count}"
            )
        if (
            expected_fold_cell_count is not None
            and passed_fold_cell_count != expected_fold_cell_count
        ):
            incompleteness_reasons.append(
                "passed_fold_cell_count="
                f"{passed_fold_cell_count}/{expected_fold_cell_count}"
            )
        summary = {
            "case_id": case_id,
            "status": status,
            "is_reference": bool(case.get("is_reference", False)),
            "changed_values": case.get("changed_values", {}),
            "complete_for_requested_execution": not incompleteness_reasons,
            "incompleteness_reasons": incompleteness_reasons,
            "expected_repeat_count": expected_repeat_count,
            "expected_fold_cell_count": expected_fold_cell_count,
            "passed_fold_cell_count": passed_fold_cell_count,
            "metric_source": (
                "config_metrics_v2"
                if trusted_row
                else metric_source
            ),
            "participant_mean_balanced_accuracy": (
                mean_ba if mean_ba is not None else _mean(repeat_ba)
            ),
            "participant_mean_macro_f1": (
                mean_f1 if mean_f1 is not None else _mean(repeat_f1)
            ),
            "participant_mean_macro_roc_auc_ovr": _mean(repeat_roc_auc),
            "participant_mean_macro_pr_auc_ovr": _mean(repeat_pr_auc),
            **{
                f"participant_mean_{name}": value
                for name, value in aware_means.items()
            },
            "primary_ranking_metric": (
                "participant_mean_abstention_aware_balanced_accuracy"
            ),
            "primary_ranking_metric_source": aware_metric_source,
            "frailty_classification_evaluation_scope": (
                "outer_heldout_participant_oof"
                if oof_rows
                else "cell_or_config_metric_fallback"
            ),
            "auxiliary_motion_evidence_valid_outer_oof": (
                auxiliary_motion_valid_oof
            ),
            "ranking_interpretation": motion_ranking_interpretation,
            "participant_mean_coverage_rate": participant_mean_coverage,
            "abstention_count": total_abstentions,
            "abstention_counts_by_class": _sum_abstention_counts(
                row.get("abstention_counts_by_class") for row in repeat_rows
            ),
            "balanced_accuracy_lcb95": (
                _number(trusted_row.get("balanced_accuracy_lcb95"))
                if trusted_row
                else _lcb95(repeat_ba)
            ),
            "macro_f1_lcb95": (
                _number(trusted_row.get("macro_f1_lcb95"))
                if trusted_row
                else _lcb95(repeat_f1)
            ),
            "participant_cluster_balanced_accuracy_ci95_low": _number(
                cluster_source.get(
                    "participant_cluster_balanced_accuracy_ci95_low"
                )
            ),
            "participant_cluster_balanced_accuracy_ci95_high": _number(
                cluster_source.get(
                    "participant_cluster_balanced_accuracy_ci95_high"
                )
            ),
            "participant_cluster_balanced_accuracy_ci95": (
                _percent_interval_text(
                    cluster_source.get(
                        "participant_cluster_balanced_accuracy_ci95_low"
                    ),
                    cluster_source.get(
                        "participant_cluster_balanced_accuracy_ci95_high"
                    ),
                )
            ),
            "participant_cluster_macro_f1_ci95_low": _number(
                cluster_source.get("participant_cluster_macro_f1_ci95_low")
            ),
            "participant_cluster_macro_f1_ci95_high": _number(
                cluster_source.get("participant_cluster_macro_f1_ci95_high")
            ),
            "participant_cluster_macro_f1_ci95": _percent_interval_text(
                cluster_source.get("participant_cluster_macro_f1_ci95_low"),
                cluster_source.get("participant_cluster_macro_f1_ci95_high"),
            ),
            "participant_cluster_macro_roc_auc_ovr_ci95_low": _number(
                cluster_source.get(
                    "participant_cluster_macro_roc_auc_ovr_ci95_low"
                )
            ),
            "participant_cluster_macro_roc_auc_ovr_ci95_high": _number(
                cluster_source.get(
                    "participant_cluster_macro_roc_auc_ovr_ci95_high"
                )
            ),
            "participant_cluster_macro_roc_auc_ovr_ci95": _percent_interval_text(
                cluster_source.get(
                    "participant_cluster_macro_roc_auc_ovr_ci95_low"
                ),
                cluster_source.get(
                    "participant_cluster_macro_roc_auc_ovr_ci95_high"
                ),
            ),
            "participant_cluster_ci_applicability": (
                "available_recomputed_from_participant_outer_oof"
                if participant_cluster_fields
                else "available_from_persisted_trusted_metrics"
                if participant_cluster_ci_available
                else "N/A"
            ),
            "participant_cluster_ci_reason": participant_cluster_ci_reason,
            "participant_cluster_bootstrap_resamples": _number(
                cluster_source.get(
                    "participant_cluster_balanced_accuracy_n_resamples"
                )
            ),
            "participant_cluster_bootstrap_valid_resamples": _number(
                cluster_source.get(
                    "participant_cluster_balanced_accuracy_valid_resamples"
                )
            ),
            "participant_cluster_bootstrap_seed": _number(
                cluster_source.get("participant_cluster_balanced_accuracy_seed")
            ),
            "participant_cluster_bootstrap_cluster_unit": cluster_source.get(
                "participant_cluster_balanced_accuracy_cluster_unit"
            ),
            "participant_cluster_bootstrap_interval_method": cluster_source.get(
                "participant_cluster_balanced_accuracy_interval_method"
            ),
            "participant_cluster_bootstrap_implementation_version": (
                cluster_source.get(
                    "participant_cluster_balanced_accuracy_implementation_version"
                )
            ),
            "repeat_balanced_accuracy_population_sd": (
                _number(
                    variability.get("repeat_balanced_accuracy_population_sd")
                )
                if trusted_row
                else _sd(repeat_ba)
            ),
            "repeat_balanced_accuracy_sample_sd": ba_statistics["sample_sd"],
            "repeat_balanced_accuracy_ci95_low": ba_statistics["ci95_low"],
            "repeat_balanced_accuracy_ci95_high": ba_statistics["ci95_high"],
            "repeat_balanced_accuracy_ci95_margin": ba_statistics["ci95_margin"],
            "repeat_balanced_accuracy_minimum": ba_statistics["minimum"],
            "repeat_balanced_accuracy_maximum": ba_statistics["maximum"],
            "repeat_macro_f1_population_sd": (
                _number(
                    variability.get("repeat_macro_f1_population_sd")
                )
                if trusted_row
                else _sd(repeat_f1)
            ),
            "repeat_macro_f1_sample_sd": f1_statistics["sample_sd"],
            "repeat_macro_f1_ci95_low": f1_statistics["ci95_low"],
            "repeat_macro_f1_ci95_high": f1_statistics["ci95_high"],
            "repeat_macro_f1_ci95_margin": f1_statistics["ci95_margin"],
            "repeat_macro_f1_minimum": f1_statistics["minimum"],
            "repeat_macro_f1_maximum": f1_statistics["maximum"],
            "repeat_macro_roc_auc_ovr_population_sd": roc_auc_statistics[
                "population_sd"
            ],
            "repeat_macro_roc_auc_ovr_sample_sd": roc_auc_statistics["sample_sd"],
            "repeat_macro_roc_auc_ovr_ci95_low": roc_auc_statistics["ci95_low"],
            "repeat_macro_roc_auc_ovr_ci95_high": roc_auc_statistics["ci95_high"],
            "repeat_macro_pr_auc_ovr_population_sd": pr_auc_statistics[
                "population_sd"
            ],
            "repeat_macro_pr_auc_ovr_sample_sd": pr_auc_statistics["sample_sd"],
            "repeat_macro_pr_auc_ovr_ci95_low": pr_auc_statistics["ci95_low"],
            "repeat_macro_pr_auc_ovr_ci95_high": pr_auc_statistics["ci95_high"],
            "repeat_abstention_aware_balanced_accuracy_population_sd": (
                aware_ba_statistics["population_sd"]
            ),
            "repeat_abstention_aware_balanced_accuracy_sample_sd": (
                aware_ba_statistics["sample_sd"]
            ),
            "repeat_abstention_aware_balanced_accuracy_ci95_low": (
                aware_ba_statistics["ci95_low"]
            ),
            "repeat_abstention_aware_balanced_accuracy_ci95_high": (
                aware_ba_statistics["ci95_high"]
            ),
            "repeat_abstention_aware_macro_f1_population_sd": (
                aware_f1_statistics["population_sd"]
            ),
            "repeat_abstention_aware_macro_f1_sample_sd": (
                aware_f1_statistics["sample_sd"]
            ),
            "repeat_abstention_aware_macro_f1_ci95_low": (
                aware_f1_statistics["ci95_low"]
            ),
            "repeat_abstention_aware_macro_f1_ci95_high": (
                aware_f1_statistics["ci95_high"]
            ),
            "worst_fold_balanced_accuracy": (
                _number(trusted_row.get("worst_fold_balanced_accuracy"))
                if trusted_row
                else min(
                    (
                        value
                        for row in case_folds
                        if (value := _number(row.get("balanced_accuracy"))) is not None
                    ),
                    default=None,
                )
            ),
            "worst_fold_abstention_aware_balanced_accuracy": min(
                (
                    value
                    for row in case_folds
                    if (
                        value := _number(
                            row.get("abstention_aware_balanced_accuracy")
                        )
                    )
                    is not None
                ),
                default=None,
            ),
            "worst_class_recall": conditional_worst_recall,
            "worst_class_f1": conditional_worst_f1,
            "abstention_aware_worst_class_recall": aware_worst_recall,
            "abstention_aware_worst_class_f1": aware_worst_f1,
            "expected_calibration_error": (
                ece if ece is not None else calculated_ece
            ),
            "repeat_count": len(repeat_rows),
            "fold_cell_count": len(case_folds),
            "subject_oof_prediction_count": len(oof_rows),
            "ci_method": (
                "repeat_student_t_abstention_aware_primary"
            ),
            "conditional_ci_method": (
                (
                    "participant_cluster_bootstrap_recomputed_from_persisted_participant_oof"
                    if participant_cluster_fields
                    else "participant_cluster_bootstrap_config_metrics_v2"
                )
                if _number(
                    cluster_source.get(
                        "participant_cluster_balanced_accuracy_ci95_low"
                    )
                )
                is not None
                else "repeat_student_t_conditional"
            ),
            "repeat_ci95_method": "two_sided_student_t_0.95",
        }
        summaries.append(summary)
        cell_coverage = [
            row.get("coverage_rate") for row in case_folds
        ]
        coverage_rows.append(
            {
                "case_id": case_id,
                "mean_coverage_rate": _mean(cell_coverage),
                "minimum_coverage_rate": min(
                    (
                        value
                        for raw in cell_coverage
                        if (value := _number(raw)) is not None
                    ),
                    default=None,
                ),
                "reported_cell_count": len(case_folds),
                "quality_diagnostic_row_count": sum(
                    str(row.get("case_id")) == case_id
                    for row in collected.quality_rows
                ),
            }
        )

    leaderboard = sorted(
        [
            row
            for row in summaries
            if row.get("status") == "passed"
            if bool(row.get("complete_for_requested_execution"))
            if _number(
                row.get(
                    "participant_mean_abstention_aware_balanced_accuracy"
                )
            )
            is not None
        ],
        key=lambda row: (
            -(
                _number(
                    row.get(
                        "participant_mean_abstention_aware_balanced_accuracy"
                    )
                )
                if _number(
                    row.get(
                        "participant_mean_abstention_aware_balanced_accuracy"
                    )
                )
                is not None
                else -math.inf
            ),
            -(
                _number(row.get("participant_mean_coverage_rate"))
                if _number(row.get("participant_mean_coverage_rate")) is not None
                else -math.inf
            ),
            -(
                _number(row.get("participant_mean_abstention_aware_macro_f1"))
                if _number(
                    row.get("participant_mean_abstention_aware_macro_f1")
                )
                is not None
                else -math.inf
            ),
            -(
                _number(row.get("participant_mean_balanced_accuracy"))
                if _number(row.get("participant_mean_balanced_accuracy")) is not None
                else -math.inf
            ),
            str(row.get("case_id")),
        ),
    )
    top_k = int(collected.plan.get("report", {}).get("top_k", 10))
    predictive = [
        {
            "predictive_rank": index,
            **dict(row),
            "decision": "manual_review_only_no_automatic_winner",
        }
        for index, row in enumerate(leaderboard[:top_k], start=1)
    ]
    predictive_ranks = {
        str(row["case_id"]): int(row["predictive_rank"])
        for row in predictive
    }
    stability_candidates = [
        row
        for row in predictive
        if _number(row.get("abstention_aware_worst_class_f1")) is not None
    ]
    stability_candidates.sort(
        key=lambda row: (
            -float(
                _number(row.get("abstention_aware_worst_class_f1")) or 0.0
            ),
            (
                float(
                    _number(
                        row.get(
                            "repeat_abstention_aware_balanced_accuracy_population_sd"
                        )
                    )
                )
                if _number(
                    row.get(
                        "repeat_abstention_aware_balanced_accuracy_population_sd"
                    )
                )
                is not None
                else math.inf
            ),
            -float(
                _number(
                    row.get(
                        "participant_mean_abstention_aware_balanced_accuracy"
                    )
                )
                or 0.0
            ),
            str(row.get("case_id")),
        )
    )
    worst_class_f1_stability = [
        {
            "worst_class_f1_stability_rank": index,
            "predictive_rank": row["predictive_rank"],
            "case_id": row["case_id"],
            "abstention_aware_worst_class_f1": row.get(
                "abstention_aware_worst_class_f1"
            ),
            "abstention_aware_worst_class_recall": row.get(
                "abstention_aware_worst_class_recall"
            ),
            "worst_class_f1": row.get("worst_class_f1"),
            "worst_class_recall": row.get("worst_class_recall"),
            "participant_mean_abstention_aware_balanced_accuracy": row.get(
                "participant_mean_abstention_aware_balanced_accuracy"
            ),
            "participant_mean_balanced_accuracy": row.get(
                "participant_mean_balanced_accuracy"
            ),
            "repeat_abstention_aware_balanced_accuracy_population_sd": row.get(
                "repeat_abstention_aware_balanced_accuracy_population_sd"
            ),
            "repeat_balanced_accuracy_population_sd": row.get(
                "repeat_balanced_accuracy_population_sd"
            ),
            "balanced_accuracy_lcb95": row.get("balanced_accuracy_lcb95"),
        }
        for index, row in enumerate(stability_candidates[:10], start=1)
    ]
    incomplete_cases = [
        row
        for row in summaries
        if not bool(row.get("complete_for_requested_execution"))
    ]
    confusion_counts, confusion_row_normalized = _confusion_long_tables(
        matrices,
        predictive_ranks,
    )

    deployment: list[Mapping[str, Any]] = []
    for row in summaries:
        trusted_row = trusted.get(str(row["case_id"]), {})
        inference = trusted_row.get("inference_cost")
        inference = dict(inference) if isinstance(inference, Mapping) else {}
        parameter_count = trusted_row.get("parameter_count")
        measurements_complete = bool(inference) and parameter_count is not None and all(
            value is not None for value in inference.values()
        )
        deployment.append(
            {
                "case_id": row["case_id"],
                "parameter_count": parameter_count,
                "inference_cost": inference,
                "operational_measurements_complete": measurements_complete,
                "reported_eligible": trusted_row.get("eligible"),
                "reported_exclusion_reason": trusted_row.get("exclusion_reason", ""),
                "deployment_readiness": (
                    "measured"
                    if measurements_complete
                    else "N/A_pending_hardware_evidence_V2_026"
                ),
            }
        )

    reference = collected.manifest.get("reference_case_id")
    paired: list[Mapping[str, Any]] = []
    bridge = collected.plan.get("legacy_bridge")
    centered_star = isinstance(bridge, Mapping) and str(
        bridge.get("design", "")
    ) == _STAGE3_STAR_DESIGN
    if reference and not centered_star:
        repeat_lookup = {
            (str(row["case_id"]), int(row["repeat"])): row
            for row in all_repeat
            if row.get("repeat") is not None
        }
        for row in all_repeat:
            case_id = str(row["case_id"])
            repeat = int(row["repeat"])
            if case_id == reference:
                continue
            baseline = repeat_lookup.get((str(reference), repeat))
            if baseline is None:
                continue
            paired.append(
                {
                    "reference_case_id": reference,
                    "case_id": case_id,
                    "repeat": repeat,
                    "balanced_accuracy_delta": (
                        _number(row.get("balanced_accuracy"))
                        - _number(baseline.get("balanced_accuracy"))
                        if _number(row.get("balanced_accuracy")) is not None
                        and _number(baseline.get("balanced_accuracy")) is not None
                        else None
                    ),
                    "macro_f1_delta": (
                        _number(row.get("macro_f1"))
                        - _number(baseline.get("macro_f1"))
                        if _number(row.get("macro_f1")) is not None
                        and _number(baseline.get("macro_f1")) is not None
                        else None
                    ),
                    "macro_roc_auc_ovr_delta": (
                        _number(row.get("macro_roc_auc_ovr"))
                        - _number(baseline.get("macro_roc_auc_ovr"))
                        if _number(row.get("macro_roc_auc_ovr")) is not None
                        and _number(baseline.get("macro_roc_auc_ovr")) is not None
                        else None
                    ),
                    "macro_pr_auc_ovr_delta": (
                        _number(row.get("macro_pr_auc_ovr"))
                        - _number(baseline.get("macro_pr_auc_ovr"))
                        if _number(row.get("macro_pr_auc_ovr")) is not None
                        and _number(baseline.get("macro_pr_auc_ovr")) is not None
                        else None
                    ),
                }
            )
    elif centered_star:
        notes.append(
            "generic single-reference paired deltas are disabled for the two-model centered star; use only the fourteen same-model Stage-3 star contrasts"
        )
    elif len(summaries) > 1:
        notes.append("paired deltas are N/A because no reference case was declared")

    paired_participant_inference: list[dict[str, Any]] = []
    if not centered_star:
        paired_participant_inference, paired_inference_notes = (
            _paired_participant_inference(
                collected,
                oof_by_case=oof_by_case,
                case_ids=tuple(manifest_cases),
            )
        )
        notes.extend(paired_inference_notes)
        if paired_participant_inference:
            notes.append(
                "Paired BA, Macro-F1 and macro ROC-AUC differences use a shared-draw "
                "participant-cluster bootstrap CI. BA/F1 additionally use the "
                "registered participant-cluster permutation budget with metric-wise "
                "Holm correction; ROC-AUC P remains explicitly N/A."
            )

    route_role_coverage, quality_distributions = _route_role_quality_tables(
        collected
    )
    denoiser_hr_record_pairs, denoiser_hr_comparison = _denoiser_hr_tables(
        collected
    )
    (
        aggregation_line_comparison,
        aggregation_line_repeat_metrics,
        aggregation_line_per_class_metrics,
        aggregation_line_notes,
    ) = _aggregation_line_tables(
        collected,
        complete_by_case={
            str(row.get("case_id")): bool(
                row.get("complete_for_requested_execution")
            )
            for row in summaries
        },
    )
    notes.extend(aggregation_line_notes)
    (
        aggregation_view_comparison,
        aggregation_view_repeat_metrics,
        aggregation_view_fold_metrics,
        aggregation_view_per_class_metrics,
        aggregation_view_confusion_matrices,
        aggregation_hierarchy_coverage,
        aggregation_view_notes,
    ) = _aggregation_report_view_tables(
        collected,
        source_eligibility_by_case={
            str(row.get("case_id")): bool(row.get("primary_ranking_eligible"))
            for row in aggregation_line_comparison
            if row.get("view_role") == "declared_source_line"
        },
    )
    notes.extend(aggregation_view_notes)
    (
        stage3_star_absolute,
        stage3_star_contrasts,
        stage3_star_fold_contrasts,
        stage3_star_execution,
        stage3_star_notes,
    ) = _stage3_star_report_tables(
        collected,
        summaries,
        aggregation_view_comparison,
        aggregation_view_fold_metrics,
    )
    notes.extend(stage3_star_notes)
    (
        stage3_star_inception_comparison,
        stage3_star_cnn_comparison,
        stage3_star_model_comparison,
        stage3_star_presentation_notes,
    ) = _stage3_star_presentation_tables(
        collected,
        stage3_star_absolute,
        stage3_star_contrasts,
        aggregation_view_repeat_metrics,
    )
    notes.extend(stage3_star_presentation_notes)
    (
        legacy_bridge_numeric_ablation_report,
        legacy_bridge_execution_order_report,
        legacy_bridge_report_notes,
    ) = _legacy_bridge_report_tables(
        collected,
        summaries,
        aggregation_view_comparison,
    )
    notes.extend(legacy_bridge_report_notes)
    classifier_per_class_results += tuple(
        abstention_aware_classifier_per_class_results
    )
    represented_classifier_ids = {
        str(row.get("classifier_id"))
        for row in classifier_per_class_results
        if not str(row.get("metric_scope", "")).startswith(
            "one_vs_rest_abstention_aware"
        )
    }
    for row in all_per_class:
        case_id = str(row.get("case_id", ""))
        if (
            not case_id
            or case_id in represented_classifier_ids
            or row.get("repeat") is not None
        ):
            continue
        classifier_per_class_results += (
            {
                "classifier_id": case_id,
                "evaluation_id": "participant_outer_oof_fallback",
                "aggregation_level": "participant",
                **{
                    key: row.get(key)
                    for key in (
                        "class_label",
                        "class_name",
                        "true_positive",
                        "false_positive",
                        "true_negative",
                        "false_negative",
                        "support",
                        "predicted_support",
                        "observation_count",
                        "precision",
                        "specificity",
                        "balanced_accuracy_ovr",
                        "f1",
                        "roc_auc_ovr",
                        "pr_auc_ovr",
                    )
                },
                "sensitivity": row.get("recall"),
                "recall": row.get("recall"),
                "probability_metric_applicability": (
                    "available"
                    if row.get("roc_auc_ovr") is not None
                    else "N/A_no_persisted_probability_scores"
                ),
                "result_applicability": "available_confusion_fallback",
                "metric_scope": "one_vs_rest_pooled_fallback",
                "metric_source": row.get("metric_source"),
                "prediction_rule_source": "persisted_confusion_fallback",
            },
        )
    represented_classifier_ids = {
        str(row.get("classifier_id"))
        for row in classifier_per_class_results
        if not str(row.get("metric_scope", "")).startswith(
            "one_vs_rest_abstention_aware"
        )
    }
    for case_id in sorted(set(manifest_cases) - represented_classifier_ids):
        for class_label, class_name in sorted(CANONICAL_CLASS_NAMES.items()):
            classifier_per_class_results += (
                {
                    "classifier_id": case_id,
                    "evaluation_id": "participant_outer_oof",
                    "aggregation_level": "participant",
                    "class_label": class_label,
                    "class_name": class_name,
                    "true_positive": None,
                    "false_positive": None,
                    "true_negative": None,
                    "false_negative": None,
                    "support": None,
                    "predicted_support": None,
                    "observation_count": 0,
                    "input_observation_count": 0,
                    "retained_observation_count": 0,
                    "excluded_observation_count": 0,
                    "precision": None,
                    "sensitivity": None,
                    "recall": None,
                    "specificity": None,
                    "balanced_accuracy_ovr": None,
                    "f1": None,
                    "roc_auc_ovr": None,
                    "pr_auc_ovr": None,
                    "probability_metric_applicability": (
                        "N/A_no_persisted_probability_or_confusion_evidence"
                    ),
                    "result_applicability": (
                        "N/A_no_persisted_probability_or_confusion_evidence"
                    ),
                    "case_execution_status": statuses.get(case_id, "unknown"),
                    "metric_scope": "one_vs_rest_not_computable",
                    "metric_source": "N/A_no_classifier_evidence",
                    "prediction_rule_source": "N/A_no_classifier_evidence",
                },
            )
    if repeat_per_class:
        notes.append(
            "Report-only ROC AUC is one-vs-rest macro AUC; PR AUC is "
            "one-vs-rest average precision. Per-class BA is (sensitivity + "
            "specificity) / 2. All are recomputed conditional on retained "
            "outer-participant OOF predictions; input, retained, and excluded "
            "counts remain explicit in classifier_per_class_results."
        )
    return StudyAnalysis(
        case_summary=tuple(summaries),
        metric_distribution_summary=tuple(metric_distributions),
        repeat_metrics=tuple(all_repeat),
        fold_metrics=tuple(fold_rows),
        per_class_metrics=tuple(all_per_class),
        confusion_matrices=tuple(matrices),
        confusion_counts=tuple(confusion_counts),
        confusion_row_normalized=tuple(confusion_row_normalized),
        calibration_bins=tuple(calibration),
        paired_deltas=tuple(paired),
        paired_participant_inference=tuple(paired_participant_inference),
        aggregation_line_comparison=tuple(aggregation_line_comparison),
        aggregation_line_repeat_metrics=tuple(aggregation_line_repeat_metrics),
        aggregation_line_per_class_metrics=tuple(
            aggregation_line_per_class_metrics
        ),
        aggregation_view_comparison=tuple(aggregation_view_comparison),
        aggregation_view_repeat_metrics=tuple(aggregation_view_repeat_metrics),
        aggregation_view_fold_metrics=tuple(aggregation_view_fold_metrics),
        aggregation_view_per_class_metrics=tuple(
            aggregation_view_per_class_metrics
        ),
        aggregation_view_confusion_matrices=tuple(
            aggregation_view_confusion_matrices
        ),
        aggregation_hierarchy_coverage=tuple(aggregation_hierarchy_coverage),
        legacy_bridge_numeric_ablation_report=tuple(
            legacy_bridge_numeric_ablation_report
        ),
        legacy_bridge_execution_order_report=tuple(
            legacy_bridge_execution_order_report
        ),
        coverage=tuple(coverage_rows),
        route_role_coverage=tuple(route_role_coverage),
        quality_distributions=tuple(quality_distributions),
        predictive_leaderboard=tuple(predictive),
        worst_class_f1_stability=tuple(worst_class_f1_stability),
        incomplete_cases=tuple(incomplete_cases),
        deployment_table=tuple(deployment),
        notes=tuple(dict.fromkeys(notes)),
        denoiser_hr_record_pairs=tuple(denoiser_hr_record_pairs),
        denoiser_hr_comparison=tuple(denoiser_hr_comparison),
        repeat_per_class_metrics=tuple(repeat_per_class),
        per_class_metric_distribution_summary=tuple(
            _per_class_metric_distributions(repeat_per_class)
        ),
        stage3_star_absolute=tuple(stage3_star_absolute),
        stage3_star_contrasts=tuple(stage3_star_contrasts),
        stage3_star_fold_contrasts=tuple(stage3_star_fold_contrasts),
        stage3_star_execution=tuple(stage3_star_execution),
        stage3_star_inception_comparison=tuple(
            stage3_star_inception_comparison
        ),
        stage3_star_cnn_comparison=tuple(stage3_star_cnn_comparison),
        stage3_star_model_comparison=tuple(stage3_star_model_comparison),
        classification_prediction_scores=tuple(
            classification_prediction_scores
        ),
        classification_roc_curves=tuple(classification_roc_curves),
        classification_prediction_tsne=tuple(
            classification_prediction_tsne
        ),
        classification_diagnostic_status=tuple(
            classification_diagnostic_status
        ),
        classifier_per_class_results=tuple(classifier_per_class_results),
    )


__all__ = ["StudyAnalysis", "analyze_study"]
