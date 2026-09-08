"""Reusable prediction-space diagnostics for every fitted classifier.

The module deliberately consumes persisted out-of-fold probabilities rather
than model-specific hidden activations.  This keeps the figures reproducible
for historical studies and prevents a probability-space t-SNE from being
misreported as a learned feature embedding.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, auc, roc_auc_score, roc_curve


@dataclass(frozen=True)
class ClassificationDiagnosticConfig:
    """Report-only controls; none of these values alter fitted predictions."""

    tsne_random_state: int = 42
    tsne_perplexity: float = 30.0
    tsne_max_samples: int = 5000
    roc_macro_grid_points: int = 201
    score_histogram_bins: int = 40

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.tsne_perplexity)) or float(self.tsne_perplexity) <= 0:
            raise ValueError("classification t-SNE perplexity must be positive")
        if int(self.tsne_max_samples) < 3:
            raise ValueError("classification t-SNE max samples must be at least 3")
        if int(self.roc_macro_grid_points) < 2:
            raise ValueError("classification ROC macro grid must contain at least 2 points")
        if int(self.score_histogram_bins) < 2:
            raise ValueError("classification score histogram bins must be at least 2")


def _as_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _probabilities(row: Mapping[str, Any]) -> tuple[float, ...] | None:
    raw = row.get("probabilities")
    if raw is None and row.get("p_active") is not None:
        active = float(row["p_active"])
        raw = (1.0 - active, active)
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return None
    total = float(sum(values))
    if total <= 0:
        return None
    return tuple(value / total for value in values)


def normalize_classification_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    classifier_id: str | None = None,
    evaluation_id: str | None = None,
    aggregation_level: str | None = None,
    label_field: str = "label",
) -> tuple[dict[str, Any], ...]:
    """Normalize multiclass OOF rows and binary motion-score rows."""

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        probabilities = _probabilities(row)
        label = _as_int(row.get(label_field, row.get("activity_label")))
        if probabilities is None or label is None:
            continue
        raw_order = row.get("class_order")
        try:
            class_order = (tuple(int(value) for value in raw_order) if raw_order is not None
                           and not isinstance(raw_order, (str, bytes)) else tuple(range(len(probabilities))))
        except (TypeError, ValueError):
            class_order = tuple(range(len(probabilities)))
        if len(class_order) != len(probabilities) or label not in class_order:
            continue
        threshold = row.get("threshold")
        try:
            threshold_value = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            threshold_value = None
        if threshold_value is not None and not math.isfinite(threshold_value):
            threshold_value = None
        if threshold_value is not None and len(class_order) == 2:
            predicted_label = class_order[-1] if probabilities[-1] >= threshold_value else class_order[0]
        else:
            predicted_label = class_order[int(np.argmax(probabilities))]
        group_classifier = str(classifier_id if classifier_id is not None else row.get("case_id", "classifier"))
        group_evaluation = str(evaluation_id if evaluation_id is not None else row.get("evaluation_id", "outer_oof"))
        group_level = str(aggregation_level if aggregation_level is not None else row.get("level", "participant"))
        sample_id = str(
            row.get(
                "sample_id",
                row.get(
                    "window_id",
                    row.get(
                        "file_id",
                        row.get("participant_id", f"row_{index:08d}"),
                    ),
                ),
            ))
        normalized = {
            "classifier_id":
            group_classifier,
            "evaluation_id":
            group_evaluation,
            "aggregation_level":
            group_level,
            "sample_id":
            sample_id,
            "participant_id":
            row.get("participant_id"),
            "file_id":
            row.get("file_id"),
            "repeat":
            row.get("repeat"),
            "fold":
            row.get("fold"),
            "split_seed":
            row.get("split_seed"),
            "training_seed":
            row.get("training_seed"),
            "retained":
            bool(row.get("retained", True)),
            "true_label":
            label,
            "predicted_label":
            int(predicted_label),
            "prediction_correct":
            bool(predicted_label == label),
            "class_order":
            class_order,
            "probabilities":
            probabilities,
            "predicted_confidence":
            float(max(probabilities)),
            "true_class_probability":
            float(probabilities[class_order.index(label)]),
            "decision_threshold":
            threshold_value,
            "threshold_source": ("persisted_frozen_threshold" if threshold_value is not None else
                                 "none_multiclass_argmax" if len(class_order) > 2 else "binary_argmax_equivalent_0.5"),
        }
        if threshold_value is None and len(class_order) == 2:
            normalized["decision_threshold"] = 0.5
        for class_label, probability in zip(class_order, probabilities, strict=True):
            normalized[f"probability_class_{class_label}"] = float(probability)
        output.append(normalized)
    return tuple(output)


def _groups(rows: Sequence[Mapping[str, Any]], ) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["classifier_id"]),
            str(row["evaluation_id"]),
            str(row["aggregation_level"]),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def classification_per_class_metric_rows(
    normalized_rows: Sequence[Mapping[str, Any]],
    *,
    class_names: Mapping[int, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Recompute auditable one-vs-rest metrics for every classifier class.

    ``normalized_rows`` must follow :func:`normalize_classification_rows`.
    In particular, hard-label metrics consume the persisted normalized
    ``predicted_label`` rather than taking a second argmax.  This preserves a
    binary classifier's frozen decision threshold while the probability vector
    remains the source for one-vs-rest ROC-AUC and average precision.
    """

    output: list[dict[str, Any]] = []
    resolved_names = {int(label): str(name) for label, name in (class_names or {}).items()}
    for key, source_group in sorted(_groups(normalized_rows).items()):
        classifier_id, evaluation_id, aggregation_level = key
        try:
            class_order = tuple(int(value) for value in source_group[0]["class_order"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{key}: normalized classification class_order is invalid") from error
        if len(class_order) < 2 or len(set(class_order)) != len(class_order):
            raise ValueError(f"{key}: normalized classification class_order must contain " "at least two unique labels")

        group = [row for row in source_group if row.get("retained", True) is not False]
        excluded_observation_count = len(source_group) - len(group)
        if not group:
            for class_label in class_order:
                output.append({
                    "classifier_id": classifier_id,
                    "evaluation_id": evaluation_id,
                    "aggregation_level": aggregation_level,
                    "class_label": class_label,
                    "class_name": resolved_names.get(class_label, str(class_label)),
                    "true_positive": None,
                    "false_positive": None,
                    "true_negative": None,
                    "false_negative": None,
                    "support": None,
                    "predicted_support": None,
                    "observation_count": 0,
                    "input_observation_count": len(source_group),
                    "retained_observation_count": 0,
                    "excluded_observation_count": excluded_observation_count,
                    "precision": None,
                    "sensitivity": None,
                    "recall": None,
                    "specificity": None,
                    "balanced_accuracy_ovr": None,
                    "f1": None,
                    "roc_auc_ovr": None,
                    "pr_auc_ovr": None,
                    "probability_metric_applicability": ("N/A_no_retained_classification_observations"),
                    "result_applicability": ("N/A_no_retained_classification_observations"),
                    "metric_scope": ("one_vs_rest_equal_weight_conditional_on_retention"),
                    "metric_source": ("normalized_persisted_probabilities_and_predicted_labels"),
                    "prediction_rule_source": ("normalized_predicted_label_preserves_frozen_threshold"),
                })
            continue

        labels: list[int] = []
        predictions: list[int] = []
        probabilities: list[tuple[float, ...]] = []
        for row in group:
            try:
                row_order = tuple(int(value) for value in row["class_order"])
                label = int(row["true_label"])
                predicted = int(row["predicted_label"])
                probability = tuple(float(value) for value in row["probabilities"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{key}: normalized classification row is incomplete") from error
            if row_order != class_order:
                raise ValueError(f"{key}: normalized classification class_order differs within group")
            if label not in class_order or predicted not in class_order:
                raise ValueError(f"{key}: true_label or predicted_label is outside class_order")
            if (len(probability) != len(class_order)
                    or not all(math.isfinite(value) and value >= 0.0 for value in probability)
                    or not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-6)):
                raise ValueError(f"{key}: normalized classification probability vector is invalid")
            labels.append(label)
            predictions.append(predicted)
            probabilities.append(probability)

        label_array = np.asarray(labels, dtype=np.int64)
        prediction_array = np.asarray(predictions, dtype=np.int64)
        probability_matrix = np.asarray(probabilities, dtype=np.float64)
        observation_count = int(label_array.size)
        for column, class_label in enumerate(class_order):
            positive = label_array == class_label
            predicted_positive = prediction_array == class_label
            true_positive = int(np.count_nonzero(positive & predicted_positive))
            false_positive = int(np.count_nonzero(~positive & predicted_positive))
            false_negative = int(np.count_nonzero(positive & ~predicted_positive))
            true_negative = int(np.count_nonzero(~positive & ~predicted_positive))
            support = true_positive + false_negative
            predicted_support = true_positive + false_positive
            precision = float(true_positive / predicted_support) if predicted_support else 0.0
            sensitivity = float(true_positive / support) if support else 0.0
            negative_support = true_negative + false_positive
            specificity = float(true_negative / negative_support) if negative_support else 0.0
            f1 = float(2.0 * precision * sensitivity / (precision + sensitivity)) if precision + sensitivity else 0.0
            if np.unique(positive).size < 2:
                roc_auc_ovr = None
                pr_auc_ovr = None
                probability_metric_applicability = "N/A_group_lacks_positive_or_negative_class"
            else:
                scores = probability_matrix[:, column]
                roc_auc_ovr = float(roc_auc_score(positive, scores))
                pr_auc_ovr = float(average_precision_score(positive, scores))
                probability_metric_applicability = "available"
            output.append({
                "classifier_id": classifier_id,
                "evaluation_id": evaluation_id,
                "aggregation_level": aggregation_level,
                "class_label": class_label,
                "class_name": resolved_names.get(class_label, str(class_label)),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "support": support,
                "predicted_support": predicted_support,
                "observation_count": observation_count,
                "input_observation_count": len(source_group),
                "retained_observation_count": observation_count,
                "excluded_observation_count": excluded_observation_count,
                "precision": precision,
                "sensitivity": sensitivity,
                "recall": sensitivity,
                "specificity": specificity,
                "balanced_accuracy_ovr": float(0.5 * (sensitivity + specificity)),
                "f1": f1,
                "roc_auc_ovr": roc_auc_ovr,
                "pr_auc_ovr": pr_auc_ovr,
                "probability_metric_applicability": (probability_metric_applicability),
                "result_applicability": "available",
                "metric_scope": ("one_vs_rest_equal_weight_conditional_on_retention"),
                "metric_source": ("normalized_persisted_probabilities_and_predicted_labels"),
                "prediction_rule_source": ("normalized_predicted_label_preserves_frozen_threshold"),
            })
    return tuple(output)


def classification_roc_curve_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    macro_grid_points: int = 201,
) -> tuple[dict[str, Any], ...]:
    """Build empirical one-vs-rest and macro-average ROC curve coordinates."""

    if int(macro_grid_points) < 2:
        raise ValueError("macro_grid_points must be at least 2")
    output: list[dict[str, Any]] = []
    for key, group in sorted(_groups(rows).items()):
        classifier_id, evaluation_id, aggregation_level = key
        class_order = tuple(int(value) for value in group[0]["class_order"])
        labels = np.asarray([int(row["true_label"]) for row in group], dtype=np.int64)
        matrix = np.asarray([row["probabilities"] for row in group], dtype=np.float64)
        class_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for column, class_label in enumerate(class_order):
            binary = (labels == class_label).astype(np.int64)
            if np.unique(binary).size < 2:
                continue
            fpr, tpr, thresholds = roc_curve(binary, matrix[:, column])
            curve_auc = float(auc(fpr, tpr))
            class_curves.append((fpr, tpr))
            for point_index, (x_value, y_value, threshold) in enumerate(zip(fpr, tpr, thresholds, strict=True)):
                output.append({
                    "classifier_id": classifier_id,
                    "evaluation_id": evaluation_id,
                    "aggregation_level": aggregation_level,
                    "curve": "one_vs_rest",
                    "class_label": class_label,
                    "point_index": point_index,
                    "false_positive_rate": float(x_value),
                    "true_positive_rate": float(y_value),
                    "score_threshold": (float(threshold) if math.isfinite(float(threshold)) else None),
                    "roc_auc": curve_auc,
                    "observation_count": len(group),
                })
        if class_curves:
            grid = np.linspace(0.0, 1.0, int(macro_grid_points), dtype=np.float64)
            macro_tpr = np.mean(
                np.vstack([np.interp(grid, fpr, tpr) for fpr, tpr in class_curves]),
                axis=0,
            )
            macro_auc = float(auc(grid, macro_tpr))
            for point_index, (x_value, y_value) in enumerate(zip(grid, macro_tpr, strict=True)):
                output.append({
                    "classifier_id": classifier_id,
                    "evaluation_id": evaluation_id,
                    "aggregation_level": aggregation_level,
                    "curve": "macro_average_ovr",
                    "class_label": "macro",
                    "point_index": point_index,
                    "false_positive_rate": float(x_value),
                    "true_positive_rate": float(y_value),
                    "score_threshold": None,
                    "roc_auc": macro_auc,
                    "observation_count": len(group),
                })
    return tuple(output)


def _group_seed(base_seed: int, key: tuple[str, str, str]) -> int:
    digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def classification_tsne_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    random_state: int = 42,
    perplexity: float = 30.0,
    max_samples: int = 5000,
) -> tuple[dict[str, Any], ...]:
    """Embed persisted probability vectors with deterministic report-only t-SNE."""

    output: list[dict[str, Any]] = []
    for key, raw_group in sorted(_groups(rows).items()):
        seed = _group_seed(int(random_state), key)
        rng = np.random.default_rng(seed)
        indices = np.arange(len(raw_group), dtype=np.int64)
        if indices.size > int(max_samples):
            indices = np.sort(rng.choice(indices, size=int(max_samples), replace=False))
        group = [raw_group[int(index)] for index in indices]
        matrix = np.asarray([row["probabilities"] for row in group], dtype=np.float64)
        if len(group) < 3 or np.unique(matrix, axis=0).shape[0] < 3:
            continue
        effective_perplexity = min(float(perplexity), float(len(group) - 1))
        coordinates = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=seed,
            init="random",
            learning_rate="auto",
        ).fit_transform(matrix)
        for row, (x_value, y_value) in zip(group, coordinates, strict=True):
            output.append({
                **dict(row),
                "tsne_x": float(x_value),
                "tsne_y": float(y_value),
                "tsne_input_space": "persisted_prediction_probability_vector",
                "tsne_base_random_state": int(random_state),
                "tsne_effective_random_state": seed,
                "tsne_requested_perplexity": float(perplexity),
                "tsne_effective_perplexity": effective_perplexity,
                "tsne_max_samples": int(max_samples),
                "tsne_group_sample_count": len(group),
            })
    return tuple(output)


def classification_diagnostic_status_rows(
    expected_classifier_ids: Sequence[str],
    normalized_rows: Sequence[Mapping[str, Any]],
    roc_rows: Sequence[Mapping[str, Any]],
    tsne_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Record availability so classifiers with missing OOF are not silently omitted."""

    normalized_counts: dict[str, int] = {}
    roc_counts: dict[str, int] = {}
    tsne_counts: dict[str, int] = {}
    for target, source in (
        (normalized_counts, normalized_rows),
        (roc_counts, roc_rows),
        (tsne_counts, tsne_rows),
    ):
        for row in source:
            classifier_id = str(row["classifier_id"])
            target[classifier_id] = target.get(classifier_id, 0) + 1
    return tuple({
        "classifier_id":
        classifier_id,
        "prediction_score_status": (
            "available" if normalized_counts.get(classifier_id, 0) else "N/A_no_oof_probabilities"),
        "roc_auc_curve_status": ("available" if roc_counts.get(classifier_id, 0) else "N/A_requires_both_classes"),
        "prediction_tsne_status": (
            "available" if tsne_counts.get(classifier_id, 0) else "N/A_requires_three_unique_probability_vectors"),
        "prediction_row_count":
        normalized_counts.get(classifier_id, 0),
        "roc_curve_point_count":
        roc_counts.get(classifier_id, 0),
        "tsne_point_count":
        tsne_counts.get(classifier_id, 0),
    } for classifier_id in dict.fromkeys(str(value) for value in expected_classifier_ids))


__all__ = [
    "ClassificationDiagnosticConfig",
    "classification_diagnostic_status_rows",
    "classification_per_class_metric_rows",
    "classification_roc_curve_rows",
    "classification_tsne_rows",
    "normalize_classification_rows",
]
