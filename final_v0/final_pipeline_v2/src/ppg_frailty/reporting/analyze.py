"""Auditable study-level metric tables built from real OOF or cell artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .collect import CollectedStudy


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for raw in values if (value := _number(raw)) is not None]
    return float(fmean(clean)) if clean else None


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
    return {
        "case_id": case_id,
        "repeat": repeat,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(order), average="macro", zero_division=0)
        ),
        "worst_class_recall": float(np.min(recall)),
        "worst_class_f1": float(np.min(class_f1)),
        "n_predictions": int(labels.size),
        "class_order": list(order),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(order)
        ).tolist(),
        "per_class": [
            {
                "case_id": case_id,
                "repeat": repeat,
                "class_label": int(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(order)
        ],
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
    return [
        {
            "case_id": case_id,
            "repeat": None,
            "class_label": order[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
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


def _route_role_quality_tables(
    collected: CollectedStudy,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
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

    for row in collected.quality_rows:
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
        groups.setdefault(
            (case_id, role, route_state, signal_route), []
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

    coverage: list[Mapping[str, Any]] = []
    for (case_id, role, route_state, signal_route), rows in sorted(groups.items()):
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
        reducer_failures = 0
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
            reducer_status = str(
                artifact.get("reducer_status")
                or artifact.get("artifact_reducer_status")
                or ""
            ).lower()
            state = str(artifact.get("state") or row.get("route_status") or "").lower()
            if (
                reducer_status
                and reducer_status
                not in {"success", "identity", "not_applied", "not_requested"}
            ) or state == "rejected_after_reduction":
                reducer_failures += 1
        total_predictors = float(sum(predictor_counts))
        total_unavailable = float(sum(unavailable_counts))
        role_oof_total, role_oof_retained = role_oof_counts.get(
            (case_id, role), (0, 0)
        )
        coverage.append(
            {
                "case_id": case_id,
                "role": role,
                "route_state": route_state,
                "signal_route": signal_route,
                "record_count": len(rows),
                "retained_record_count": retained_count,
                "retained_coverage": retained_count / len(rows),
                "direct_rate_record_count": direct_count,
                "processed_rate_record_count": processed_count,
                "dropped_record_count": len(rows) - retained_count,
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
                "reducer_failure_rate": reducer_failures / len(rows),
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
    coverage: tuple[Mapping[str, Any], ...]
    route_role_coverage: tuple[Mapping[str, Any], ...]
    quality_distributions: tuple[Mapping[str, Any], ...]
    predictive_leaderboard: tuple[Mapping[str, Any], ...]
    worst_class_f1_stability: tuple[Mapping[str, Any], ...]
    incomplete_cases: tuple[Mapping[str, Any], ...]
    deployment_table: tuple[Mapping[str, Any], ...]
    notes: tuple[str, ...]


def analyze_study(collected: CollectedStudy) -> StudyAnalysis:
    """Build descriptive comparison tables without selecting a winner."""

    manifest_cases = _manifest_cases(collected)
    statuses = _record_statuses(collected)
    trusted = _trusted_by_case(collected)
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
    matrices: list[Mapping[str, Any]] = []
    calibration: list[Mapping[str, Any]] = []
    summaries: list[Mapping[str, Any]] = []
    metric_distributions: list[Mapping[str, Any]] = []
    coverage_rows: list[Mapping[str, Any]] = []
    notes = list(collected.limitations)

    for case_id, case in manifest_cases.items():
        oof_rows = oof_by_case.get(case_id, [])
        trusted_row = trusted.get(case_id, {})
        repeat_rows: list[Mapping[str, Any]] = []
        case_folds = fold_by_case.get(case_id, [])
        if oof_rows:
            repeat_groups: dict[int, list[Mapping[str, Any]]] = {}
            for row in oof_rows:
                repeat_groups.setdefault(int(row.get("repeat", -1)), []).append(row)
            for repeat, values in sorted(repeat_groups.items()):
                metric = _oof_metric_row(case_id, values, repeat=repeat)
                if metric is not None:
                    repeat_rows.append(
                        {
                            key: value
                            for key, value in metric.items()
                            if key not in {"per_class", "confusion_matrix"}
                        }
                        | {"metric_source": "participant_oof"}
                    )
            pooled = _oof_metric_row(case_id, oof_rows, repeat=None)
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
        all_repeat.extend(repeat_rows)
        repeat_ba = [row.get("balanced_accuracy") for row in repeat_rows]
        repeat_f1 = [row.get("macro_f1") for row in repeat_rows]
        mean_ba = _number(
            trusted_row.get("participant_mean_balanced_accuracy")
        )
        mean_f1 = _number(trusted_row.get("participant_mean_macro_f1"))
        worst_recall = _number(trusted_row.get("worst_class_recall"))
        worst_f1 = _number(trusted_row.get("worst_class_f1"))
        ece = _number(trusted_row.get("expected_calibration_error"))
        variability = trusted_row.get("variability")
        variability = (
            variability if isinstance(variability, Mapping) else {}
        )
        ba_statistics = _descriptive_statistics(repeat_ba)
        f1_statistics = _descriptive_statistics(repeat_f1)
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
            )
        )
        status = statuses.get(case_id, "not_run")
        passed_fold_cell_count = sum(
            str(row.get("status")) == "passed" for row in case_folds
        )
        incompleteness_reasons: list[str] = []
        if status != "passed":
            incompleteness_reasons.append(f"case_status={status}")
        if (
            expected_repeat_count is not None
            and int(ba_statistics["n"]) != expected_repeat_count
        ):
            incompleteness_reasons.append(
                "repeat_metric_count="
                f"{ba_statistics['n']}/{expected_repeat_count}"
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
            "worst_class_recall": (
                worst_recall
                if worst_recall is not None
                else (pooled or {}).get("worst_class_recall")
            ),
            "worst_class_f1": (
                worst_f1
                if worst_f1 is not None
                else (pooled or {}).get("worst_class_f1")
            ),
            "expected_calibration_error": (
                ece if ece is not None else calculated_ece
            ),
            "repeat_count": len(repeat_rows),
            "fold_cell_count": len(case_folds),
            "subject_oof_prediction_count": len(oof_rows),
            "ci_method": (
                "participant_cluster_bootstrap_config_metrics_v2"
                if trusted_row
                else "repeat_student_t_fallback"
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
            if _number(row.get("participant_mean_balanced_accuracy")) is not None
        ],
        key=lambda row: (
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
        if _number(row.get("worst_class_f1")) is not None
    ]
    stability_candidates.sort(
        key=lambda row: (
            -float(_number(row.get("worst_class_f1")) or 0.0),
            (
                float(_number(row.get("repeat_balanced_accuracy_population_sd")))
                if _number(row.get("repeat_balanced_accuracy_population_sd"))
                is not None
                else math.inf
            ),
            -float(_number(row.get("participant_mean_balanced_accuracy")) or 0.0),
            str(row.get("case_id")),
        )
    )
    worst_class_f1_stability = [
        {
            "worst_class_f1_stability_rank": index,
            "predictive_rank": row["predictive_rank"],
            "case_id": row["case_id"],
            "worst_class_f1": row.get("worst_class_f1"),
            "worst_class_recall": row.get("worst_class_recall"),
            "participant_mean_balanced_accuracy": row.get(
                "participant_mean_balanced_accuracy"
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
                    else "N/A_pending_hardware_gate_V2_026"
                ),
            }
        )

    reference = collected.manifest.get("reference_case_id")
    paired: list[Mapping[str, Any]] = []
    if reference:
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
                }
            )
    elif len(summaries) > 1:
        notes.append("paired deltas are N/A because no reference case was declared")

    route_role_coverage, quality_distributions = _route_role_quality_tables(
        collected
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
        coverage=tuple(coverage_rows),
        route_role_coverage=tuple(route_role_coverage),
        quality_distributions=tuple(quality_distributions),
        predictive_leaderboard=tuple(predictive),
        worst_class_f1_stability=tuple(worst_class_f1_stability),
        incomplete_cases=tuple(incomplete_cases),
        deployment_table=tuple(deployment),
        notes=tuple(dict.fromkeys(notes)),
    )


__all__ = ["StudyAnalysis", "analyze_study"]
