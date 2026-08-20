"""Auditable study-level metric tables built from real OOF or cell artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
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

from ..data.schema import CANONICAL_CLASS_NAMES
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
from .collect import CollectedStudy


WINDOW_BALANCED_TO_PARTICIPANT = "window_balanced_to_participant"
AGGREGATION_REPORT_VIEWS = (
    WINDOW_BALANCED_TO_PARTICIPANT,
    LINE_A_EQUAL_FILES,
    LINE_B_EQUAL_ROLE_FAMILIES,
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
                "class_name": CANONICAL_CLASS_NAMES.get(int(label), str(label)),
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
            "class_name": CANONICAL_CLASS_NAMES.get(
                int(order[index]), str(order[index])
            ),
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
            lambda row: "participant",
        ),
    )
    grouped: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = {}
    for level, view, rows, labeler in sources:
        for row in rows:
            if (str(row.get("case_id")), level) in failed_levels:
                continue
            key = (
                str(row.get("case_id")),
                int(row.get("repeat", 0)),
                level,
                view,
                labeler(row),
            )
            grouped.setdefault(key, []).append(row)
    output: list[Mapping[str, Any]] = []
    for (case_id, repeat, level, view, group_label), rows in sorted(grouped.items()):
        retained = [row for row in rows if row.get("retained", True) is not False]
        output.append(
            {
                "case_id": case_id,
                "repeat": repeat,
                "aggregation_level": level,
                "aggregation_view": view,
                "group_label": group_label,
                "oof_unit_count": len(rows),
                "retained_oof_unit_count": len(retained),
                "participant_count": len(
                    {
                        str(row.get("participant_id"))
                        for row in retained
                        if row.get("participant_id") is not None
                    }
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
                    else "N/A_pending_hardware_evidence_V2_026"
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
        legacy_bridge_numeric_ablation_report,
        legacy_bridge_execution_order_report,
        legacy_bridge_report_notes,
    ) = _legacy_bridge_report_tables(
        collected,
        summaries,
        aggregation_view_comparison,
    )
    notes.extend(legacy_bridge_report_notes)
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
        aggregation_line_comparison=tuple(aggregation_line_comparison),
        aggregation_line_repeat_metrics=tuple(aggregation_line_repeat_metrics),
        aggregation_line_per_class_metrics=tuple(
            aggregation_line_per_class_metrics
        ),
        aggregation_view_comparison=tuple(aggregation_view_comparison),
        aggregation_view_repeat_metrics=tuple(aggregation_view_repeat_metrics),
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
    )


__all__ = ["StudyAnalysis", "analyze_study"]
