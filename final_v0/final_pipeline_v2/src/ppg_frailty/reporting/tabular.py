"""Shared CSV, compact-display, workbook, and table/figure pairing helpers.

The JSON report artifacts remain the lossless numerical audit source.  CSV,
Markdown, HTML, and XLSX may use :func:`compact_rows` to collapse a reported
mean and its SD into one human-facing ``mean_sd`` field.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class ReportTable:
    """One independently exported report table."""

    name: str
    rows: Sequence[Mapping[str, Any]]
    description: str = ""
    compact: bool = True


@dataclass(frozen=True)
class ColumnDefinition:
    """Auditable human-readable semantics for one displayed table column."""

    column_name: str
    display_label: str
    source_fields: tuple[str, ...]
    definition: str
    formula: str
    documentation_kind: str


_SCORE_WORDS = (
    "accuracy",
    "auc",
    "coverage",
    "f1",
    "precision",
    "predictive_value",
    "recall",
    "sensitivity",
    "specificity",
)


_NON_ARITHMETIC_FORMULA = (
    "N/A — identifier, provenance, configuration, status, or other "
    "non-arithmetic field."
)
_DIRECT_VALUE_FORMULA = (
    "N/A — direct persisted/source-defined value; the table renderer applies "
    "no additional arithmetic."
)


_EXACT_COLUMN_DEFINITIONS: Mapping[str, tuple[str, str]] = {
    "ir_red": (
        "Optical PPG channel for the denoiser endpoint row.",
        "N/A — categorical channel identifier, either IR or RED",
    ),
    "rmse_sd_ms": (
        "Participant-macro IBI–PPI RMSE mean and between-participant sample SD in milliseconds; a trailing star marks the minimum mean in the activity table.",
        "mean_RMSE ± sqrt[sum_i(RMSE_i-mean_RMSE)^2/(n-1)] ms",
    ),
    "f1_sd": (
        "Participant-macro ECG-aligned beat-detection F1 mean and between-participant sample SD in percent; a trailing star marks the maximum mean in the activity table.",
        "100*mean_i(F1_i) ± 100*sqrt[sum_i(F1_i-mean_i(F1_i))^2/(n-1)]",
    ),
    "rmse_p_versus_identity": (
        "Holm-adjusted two-sided participant-paired sign-flip P value for the denoiser RMSE versus identity on identical successful segments; the identity row is the reference.",
        "raw p=(1+sum_b I(|mean(s_b*d)|>=|mean(d)|))/(B+1); Holm p_(i)=max_(j<=i)[(m-j+1)p_(j)], capped at 1",
    ),
    "participant_macro_mean_sd": (
        "Participant-macro arithmetic mean and between-participant sample SD, rendered in the endpoint unit named by the table title.",
        "display = s*mean_i(m_i) ± s*sqrt[sum_i(m_i-mean_i(m_i))^2/(n-1)]; s=100 for unitless score endpoints and s=1 for native-unit endpoints",
    ),
    "mean_sd": (
        "Arithmetic mean and sample SD rendered in the endpoint unit named by the table title.",
        "display = s*mean(x) ± s*sqrt[sum_i(x_i-mean(x))^2/(n-1)]; s=100 for unitless score endpoints and s=1 for native-unit endpoints",
    ),
    "participant_bootstrap_ci95": (
        "Percentile-bootstrap 95% interval for the participant-macro arithmetic mean.",
        "CI95 = [Q_0.025(mean(x_i*)), Q_0.975(mean(x_i*))], where participant IDs are sampled with replacement; use the same endpoint scale s as the adjacent mean ± SD column",
    ),
    "holm_p_vs_identity": (
        "Holm-adjusted two-sided participant-paired sign-flip P value for the denoiser versus identity; the identity row is the reference.",
        "raw p=(1+sum_b I(|mean(s_b*d)|>=|mean(d)|))/(B+1); Holm p_(i)=max_(j<=i)[(m-j+1)p_(j)], capped at 1",
    ),
    "holm_p_vs_frailty29_trained": (
        "Holm-adjusted two-sided participant-paired sign-flip P value for the PTT22-trained detector versus the Frailty29-trained reference on the identical target roster.",
        "raw p=(1+sum_b I(|mean(s_b*d)|>=|mean(d)|))/(B+1); Holm p_(i)=max_(j<=i)[(m-j+1)p_(j)], capped at 1",
    ),
    "holm_p_vs_reference": (
        "Holm-adjusted two-sided participant-paired sign-flip P value for the table's named candidate versus configurable reference on an identical participant roster.",
        "raw p=(1+sum_b I(|mean(s_b*d)|>=|mean(d)|))/(B+1); Holm p_(i)=max_(j<=i)[(m-j+1)p_(j)], capped at 1",
    ),
    "accuracy": (
        "Fraction of evaluated units assigned to their true class.",
        "accuracy = (sum_c TP_c) / N",
    ),
    "balanced_accuracy": (
        "Macro-average recall across the K declared classes.",
        "BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]",
    ),
    "balanced_accuracy_ovr": (
        "One-vs-rest balanced accuracy for the named class.",
        "BA_c = 0.5 * [TP_c/(TP_c+FN_c) + TN_c/(TN_c+FP_c)]",
    ),
    "precision": (
        "Positive predictive value for the named class.",
        "precision = TP / (TP + FP)",
    ),
    "positive_predictive_value": (
        "Positive predictive value for the named class or beat endpoint.",
        "PPV = TP / (TP + FP)",
    ),
    "recall": (
        "Sensitivity for the named class.",
        "recall = TP / (TP + FN)",
    ),
    "sensitivity": (
        "True-positive rate for the positive class.",
        "sensitivity = TP / (TP + FN)",
    ),
    "specificity": (
        "True-negative rate for the negative class.",
        "specificity = TN / (TN + FP)",
    ),
    "false_positive_rate": (
        "False-positive rate for the positive class.",
        "FPR = FP / (FP + TN) = 1 - specificity",
    ),
    "true_positive_rate": (
        "True-positive rate for the positive class.",
        "TPR = TP / (TP + FN) = sensitivity",
    ),
    "f1": (
        "Harmonic mean of precision and recall for the named class.",
        "F1 = 2 * precision * recall / (precision + recall)",
    ),
    "macro_f1": (
        "Unweighted mean of the K class-specific F1 scores.",
        "macro-F1 = (1/K) * sum_c F1_c",
    ),
    "macro_precision": (
        "Unweighted mean of class-specific precision.",
        "macro-precision = (1/K) * sum_c precision_c",
    ),
    "macro_recall": (
        "Unweighted mean of class-specific recall.",
        "macro-recall = (1/K) * sum_c recall_c",
    ),
    "roc_auc": (
        "Area under the empirical receiver-operating-characteristic curve.",
        "ROC-AUC = integral_0^1 TPR(FPR) dFPR (empirical trapezoidal area)",
    ),
    "roc_auc_ovr": (
        "One-vs-rest ROC area for the named class.",
        "ROC-AUC_c = integral_0^1 TPR_c(FPR_c) dFPR_c",
    ),
    "macro_roc_auc_ovr": (
        "Unweighted mean of valid one-vs-rest class ROC areas.",
        "macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c",
    ),
    "pr_auc": (
        "Area/average precision under the empirical precision-recall curve.",
        "AP = sum_n (recall_n - recall_(n-1)) * precision_n",
    ),
    "pr_auc_ovr": (
        "One-vs-rest average precision for the named class.",
        "AP_c = sum_n (recall_(c,n) - recall_(c,n-1)) * precision_(c,n)",
    ),
    "macro_pr_auc_ovr": (
        "Unweighted mean of valid one-vs-rest class average precision values.",
        "macro PR-AUC = (1/K_valid) * sum_c AP_c",
    ),
    "coverage_rate": (
        "Fraction of eligible evaluation units retained for prediction.",
        "coverage = n_retained / n_total",
    ),
    "participant_coverage_rate": (
        "Fraction of eligible participants retained for prediction.",
        "participant coverage = retained participants / eligible participants",
    ),
    "retained_coverage": (
        "Fraction of source units retained by the declared route.",
        "retained coverage = retained units / total eligible units",
    ),
    "abstention_rate": (
        "Fraction of eligible units for which the classifier abstained.",
        "abstention rate = n_abstained / n_total = 1 - coverage",
    ),
    "expected_calibration_error": (
        "Top-label equal-width-bin expected calibration error.",
        "ECE = sum_b (n_b/N) * abs(accuracy_b - mean_confidence_b)",
    ),
    "multiclass_brier": (
        "Mean class-summed squared probability error.",
        "Brier = (1/N) * sum_i sum_c (p_ic - 1[y_i=c])^2",
    ),
    "multiclass_log_loss": (
        "Mean negative log probability assigned to the true class.",
        "log loss = -(1/N) * sum_i log(p_i,y_i)",
    ),
    "ibi_ppi_rmse_ms": (
        "Root-mean-square paired IBI–PPI error in milliseconds.",
        "RMSE = sqrt[(1/M) * sum_j (IBI_j - PPI_j)^2] * 1000",
    ),
    "ibi_ppi_mae_ms": (
        "Mean absolute paired IBI–PPI error in milliseconds.",
        "MAE = (1/M) * sum_j abs(IBI_j - PPI_j) * 1000",
    ),
    "prediction_correct": (
        "Indicator that the predicted class equals the true class.",
        "prediction_correct = 1[predicted_label = true_label]",
    ),
    "predicted_label": (
        "Class selected by the persisted classifier decision rule.",
        "binary: predicted=positive if p_positive >= frozen threshold, else negative; multiclass: predicted=class_order[argmax_c p_c]",
    ),
    "predicted_class": (
        "Class selected from the persisted probability vector.",
        "predicted_class = class_order[argmax_c p_c]",
    ),
    "predicted_confidence": (
        "Largest class probability for the prediction.",
        "predicted_confidence = max_c p_c",
    ),
    "true_class_probability": (
        "Persisted probability assigned to the true class.",
        "true_class_probability = p_(true class)",
    ),
    "row_fraction": (
        "Fraction of all table rows represented by the current group.",
        "row fraction = group row count / total row count",
    ),
    "runtime_fraction_of_signal": (
        "Execution time divided by signal duration.",
        "runtime fraction = execution seconds / signal-duration seconds",
    ),
    "execution_time_percent": (
        "Execution time expressed as a percentage of signal duration.",
        "execution time (%) = 100 * execution seconds / signal-duration seconds",
    ),
    "rank_sum_z": (
        "Normal-approximation Z statistic from the declared Wilcoxon rank-sum test.",
        "Z = [R_ref - n_ref*(n_ref+n_cmp+1)/2] / sqrt[n_ref*n_cmp*(n_ref+n_cmp+1)/12] for the declared scipy.stats.ranksums implementation",
    ),
    "common_subject_recordings": (
        "Number of finite subject-recordings present for both compared detectors.",
        "n_common = count(keys_reference intersection keys_comparator)",
    ),
    "reference_advantage": (
        "Median contrast oriented so a positive value favors the reference detector.",
        "higher-is-better: median_reference - median_comparator; lower-is-better: median_comparator - median_reference",
    ),
    "msptdfast_advantage": (
        "Median contrast oriented so a positive value favors MSPTDfast.",
        "higher-is-better: median_MSPTDfast - median_comparator; lower-is-better: median_comparator - median_MSPTDfast",
    ),
    "true_positives": (
        "Count of positive-class units predicted positive.",
        "TP = sum_i 1[y_i=positive and predicted_i=positive]",
    ),
    "false_positives": (
        "Count of negative-class units predicted positive.",
        "FP = sum_i 1[y_i=negative and predicted_i=positive]",
    ),
    "false_negatives": (
        "Count of positive-class units predicted negative or abstained when the "
        "declared abstention-aware rule applies.",
        "FN = sum_i 1[y_i=positive and predicted_i is not positive]",
    ),
    "true_negatives": (
        "Count of negative-class units predicted negative.",
        "TN = sum_i 1[y_i=negative and predicted_i=negative]",
    ),
    "support": (
        "Number of evaluated units whose true label is the named class.",
        "support_c = TP_c + FN_c",
    ),
    "predicted_support": (
        "Number of evaluated units predicted as the named class.",
        "predicted support_c = TP_c + FP_c",
    ),
    "true_positive": (
        "Count of named-class units predicted as that class.",
        "TP_c = sum_i 1[y_i=c and predicted_i=c]",
    ),
    "false_positive": (
        "Count of non-class units predicted as the named class.",
        "FP_c = sum_i 1[y_i!=c and predicted_i=c]",
    ),
    "false_negative": (
        "Count of named-class units not predicted as that class.",
        "FN_c = sum_i 1[y_i=c and predicted_i!=c]",
    ),
    "true_negative": (
        "Count of non-class units not predicted as the named class.",
        "TN_c = sum_i 1[y_i!=c and predicted_i!=c]",
    ),
}


_DIRECT_FIELD_TOKENS = (
    "activity",
    "algorithm",
    "artifact",
    "case",
    "channel",
    "class_label",
    "class_name",
    "config",
    "dataset",
    "description",
    "device",
    "error",
    "evidence",
    "execution_state",
    "fold",
    "group",
    "hash",
    "id",
    "interpretation",
    "label",
    "literature",
    "method",
    "model",
    "module",
    "name",
    "parameter",
    "path",
    "profile",
    "reason",
    "reference",
    "repeat",
    "role",
    "scope",
    "seed",
    "sha256",
    "source",
    "split",
    "status",
    "threshold",
    "timestamp",
    "title",
    "unit",
    "version",
)


def _normalized_column_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    normalized = normalized.replace("macrof1", "macro_f1")
    normalized = re.sub(
        r"(?:^|_)ba(?=_|$)",
        lambda match: ("_" if match.group(0).startswith("_") else "")
        + "balanced_accuracy",
        normalized,
    )
    return re.sub(
        r"(?:^|_)ppv(?=_|$)",
        lambda match: ("_" if match.group(0).startswith("_") else "")
        + "positive_predictive_value",
        normalized,
    )


def _metric_definition(name: str) -> tuple[str, str] | None:
    normalized = _normalized_column_name(name)
    direct = _EXACT_COLUMN_DEFINITIONS.get(normalized)
    if direct is not None:
        return direct
    for metric in sorted(_EXACT_COLUMN_DEFINITIONS, key=len, reverse=True):
        if re.search(rf"(?:^|_){re.escape(metric)}(?:_|$)", normalized):
            return _EXACT_COLUMN_DEFINITIONS[metric]
    return None


def _field_definition(field: str) -> tuple[str, str, str]:
    """Infer safe semantics without inventing producer-specific arithmetic."""

    normalized = _normalized_column_name(field)
    exact = _EXACT_COLUMN_DEFINITIONS.get(normalized)
    if exact is not None:
        return (*exact, "explicit_metric_formula")

    metric = _metric_definition(normalized)
    metric_text = metric[0] if metric is not None else "the reported statistic"
    scale = "100" if "percent" in normalized else "1"
    is_delta = (
        normalized.startswith("delta_")
        or normalized.endswith("_delta")
        or "_delta_" in normalized
        or "paired_delta" in normalized
        or "candidate_minus_reference" in normalized
        or "_minus_" in normalized
    )

    if normalized.endswith(
        ("_method", "_applicability", "_reason", "_metrics", "_cluster_unit")
    ):
        return (
            "Direct method, applicability, reason, metric-roster, or cluster-unit "
            "provenance for the reported statistic.",
            _NON_ARITHMETIC_FORMULA,
            "non_arithmetic_audit_field",
        )

    if (
        "ci95" in normalized
        and (
            "participant_cluster" in normalized
            or "paired_delta_cluster" in normalized
        )
    ):
        endpoint = (
            "lower endpoint"
            if normalized.endswith("_low")
            else "upper endpoint"
            if normalized.endswith("_high")
            else "two-sided interval"
        )
        draw = "100 * T_b" if "percent" in normalized else "T_b"
        statistic = (
            "T_b = metric_candidate,b - metric_reference,b from the same "
            "participant draw; "
            if is_delta
            else "T_b is the metric recomputed from bootstrap draw b; "
        )
        return (
            f"Participant-cluster percentile-bootstrap 95% CI {endpoint} for "
            f"{metric_text}",
            f"CI95 = [Q_0.025({draw}), Q_0.975({draw})], b=1..B; "
            + statistic
            + "each draw resamples participant IDs with replacement under the "
            "declared strata and carries every repeat/row belonging to each "
            "sampled participant cluster",
            "participant_cluster_interval_formula",
        )

    if is_delta:
        return (
            f"Paired candidate-minus-reference difference in {metric_text}",
            "delta = metric_candidate - metric_reference on the declared matched unit",
            "paired_difference_formula",
        )

    if "holm_sidak_adjusted_p" in normalized:
        return (
            "Holm–Sidak step-down multiplicity-adjusted P value.",
            "ordered adjusted p_(i) = max_(j<=i) {1 - (1 - p_(j))^(m-j+1)}, "
            "capped at 1",
            "holm_sidak_adjustment_formula",
        )
    if "holm_adjusted_p" in normalized:
        return (
            "Holm step-down multiplicity-adjusted P value.",
            "ordered adjusted p_(i) = max_(j<=i) [(m-j+1) * p_(j)], capped at 1",
            "holm_adjustment_formula",
        )
    if "p_value" in normalized or normalized.endswith("_p"):
        return (
            "Null-hypothesis tail probability from the table's declared test.",
            "two-sided p = Pr_H0(|T*| >= |T_observed|); exact statistic and "
            "resampling/rank distribution follow the declared test_method",
            "declared_p_value_formula",
        )

    if "ci95" in normalized or "lcb95" in normalized:
        if "repeat" in normalized or "_mean" in normalized or normalized == "ci95":
            interval = "mean +/- t_(0.975,n-1) * sample_SD / sqrt(n)"
            formula = f"repeat CI95 = {scale} * ({interval})"
            kind = "repeat_student_t_interval_formula"
        else:
            formula = (
                "endpoint(s) produced by the table's declared ci_method; the "
                "renderer does not substitute a different interval estimator"
            )
            kind = "declared_interval_method"
        return (
            f"Reported 95% confidence bound or interval for {metric_text}",
            formula,
            kind,
        )

    if "mean_sd" in normalized:
        return (
            f"Compact mean and sample-standard-deviation display for {metric_text}",
            f"display = {scale} * mean +/- {scale} * sqrt[sum_i "
            "(x_i - mean)^2 / (n - 1)]",
            "mean_sample_sd_display_formula",
        )
    if normalized.endswith(("_sample_sd", "_std")) or normalized in {"sample_sd", "sd", "std"}:
        return (
            f"Sample standard deviation of {metric_text}",
            "sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]",
            "sample_sd_formula",
        )
    if normalized.endswith("_population_sd") or normalized == "population_sd":
        return (
            f"Population-form standard deviation of {metric_text}",
            "population SD = sqrt[sum_i (x_i - mean)^2 / n]",
            "population_sd_formula",
        )
    if normalized.endswith("_mean") or normalized.startswith(
        ("mean_", "participant_mean_", "participant_macro_")
    ):
        return (
            f"Arithmetic mean of {metric_text} over the table's declared units",
            "mean = (1/n) * sum_i x_i",
            "arithmetic_mean_formula",
        )
    if "median" in normalized:
        return (
            f"Median of {metric_text} over the table's declared units",
            "median = Q_0.50(x)",
            "quantile_formula",
        )
    if normalized.endswith("_iqr") or "_q1" in normalized or "_q3" in normalized:
        return (
            f"Interquartile summary of {metric_text}",
            "Q1 = Q_0.25(x); Q3 = Q_0.75(x); IQR = Q3 - Q1",
            "quantile_formula",
        )
    percentile = re.search(r"(?:^|_)p(\d{1,2})(?:_|$)", normalized)
    if percentile is not None:
        probability = int(percentile.group(1)) / 100.0
        return (
            f"Empirical percentile of {metric_text}",
            f"percentile = Q_{probability:.2f}(x)",
            "quantile_formula",
        )
    if normalized.endswith("_minimum") or normalized == "minimum":
        return (f"Minimum observed {metric_text}", "minimum = min_i x_i", "extreme_formula")
    if normalized.endswith("_maximum") or normalized == "maximum":
        return (f"Maximum observed {metric_text}", "maximum = max_i x_i", "extreme_formula")

    if (
        normalized == "n"
        or normalized == "count"
        or normalized.endswith("_count")
        or normalized.startswith("n_")
    ):
        return (
            "Number of units satisfying the column's stated inclusion condition.",
            "count = sum_i 1[unit i satisfies the stated condition]",
            "count_formula",
        )
    if normalized.endswith(("_rate", "_fraction")):
        return (
            "Proportion for the numerator and denominator named by this column.",
            "rate = stated numerator count / stated eligible denominator count",
            "rate_formula",
        )
    if metric is not None:
        definition, formula = metric
        if "percent" in normalized:
            return (
                f"{definition} Displayed as percent.",
                f"displayed value = 100 * ({formula})",
                "scaled_metric_formula",
            )
        return definition, formula, "inferred_metric_formula"
    if normalized.endswith("_rank") or normalized == "rank":
        return (
            "Ordinal position after applying the table's declared sorting rule.",
            _NON_ARITHMETIC_FORMULA,
            "non_arithmetic_ordinal",
        )
    if any(
        re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", normalized)
        for token in _DIRECT_FIELD_TOKENS
    ):
        return (
            "Direct identifier, provenance, configuration, grouping, or status value.",
            _NON_ARITHMETIC_FORMULA,
            "non_arithmetic_audit_field",
        )
    return (
        f"Persisted source-table value for `{field}`; producer-specific semantics "
        "are not reinterpreted by the shared table renderer.",
        _DIRECT_VALUE_FORMULA,
        "safe_source_defined_fallback",
    )


def column_definition(
    column: str | tuple[str, str] | tuple[str, str, bool],
    *,
    display_label: str | None = None,
) -> ColumnDefinition:
    """Describe one raw field or one displayed mean/SD composite column."""

    if isinstance(column, str):
        if not column.strip():
            raise ValueError("column name must be non-empty")
        definition, formula, kind = _field_definition(column)
        return ColumnDefinition(
            column_name=column,
            display_label=display_label or column,
            source_fields=(column,),
            definition=definition,
            formula=formula,
            documentation_kind=kind,
        )
    if not isinstance(column, tuple) or len(column) not in {2, 3}:
        raise TypeError("column must be a field name or a (mean, SD[, percent]) tuple")
    mean_field, sd_field = column[:2]
    if not isinstance(mean_field, str) or not mean_field.strip():
        raise ValueError("mean field must be a non-empty string")
    if not isinstance(sd_field, str) or not sd_field.strip():
        raise ValueError("SD field must be a non-empty string")
    if len(column) == 3 and not isinstance(column[2], bool):
        raise TypeError("mean/SD tuple percent flag must be boolean")
    percent = True if len(column) == 2 else column[2]
    scale = "100" if percent else "1"
    metric = _metric_definition(mean_field)
    metric_text = metric[0] if metric is not None else f"`{mean_field}`"
    return ColumnDefinition(
        column_name=f"{mean_field} + {sd_field}",
        display_label=display_label or f"{mean_field} mean +/- SD",
        source_fields=(mean_field, sd_field),
        definition=f"Compact mean +/- sample-SD display for {metric_text}",
        formula=(
            f"display = {scale} * {mean_field} +/- {scale} * {sd_field}; "
            "sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]"
        ),
        documentation_kind="explicit_mean_sd_composite",
    )


def column_definition_rows(
    columns: Sequence[str | tuple[str, str] | tuple[str, str, bool]],
    *,
    display_labels: Sequence[str] | None = None,
    table_name: str = "",
    table_description: str = "",
) -> list[dict[str, Any]]:
    """Return one complete definition/formula row for every input column."""

    labels = list(display_labels) if display_labels is not None else [None] * len(columns)
    if len(labels) != len(columns):
        raise ValueError("display_labels must align one-to-one with columns")
    output: list[dict[str, Any]] = []
    for position, (column, label) in enumerate(zip(columns, labels, strict=True), start=1):
        item = column_definition(column, display_label=label)
        output.append(
            {
                "table_name": table_name,
                "table_description": table_description,
                "ordinal_position": position,
                "column_name": item.column_name,
                "display_label": item.display_label,
                "source_fields": "; ".join(item.source_fields),
                "definition": item.definition,
                "formula": item.formula,
                "documentation_kind": item.documentation_kind,
            }
        )
    return output


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def markdown_column_definitions_block(
    columns: Sequence[str | tuple[str, str] | tuple[str, str, bool]],
    *,
    display_labels: Sequence[str] | None = None,
) -> str:
    """Render a compact collapsible definition/formula block for a table."""

    rows = column_definition_rows(columns, display_labels=display_labels)
    lines = ["<details><summary>Column definitions and formulas</summary>", ""]
    for row in rows:
        lines.append(
            f"- **{_markdown_text(row['display_label'])}** "
            f"(`{_markdown_text(row['source_fields'])}`): "
            f"{_markdown_text(row['definition'])} Formula: "
            f"`{_markdown_text(row['formula'])}`"
        )
    lines.extend(("", "</details>"))
    return "\n".join(lines)


def html_column_definitions_block(
    columns: Sequence[str | tuple[str, str] | tuple[str, str, bool]],
    *,
    display_labels: Sequence[str] | None = None,
) -> str:
    """Render an HTML definition/formula block for a table."""

    rows = column_definition_rows(columns, display_labels=display_labels)
    items = "".join(
        "<li><strong>"
        + escape(str(row["display_label"]))
        + "</strong> (<code>"
        + escape(str(row["source_fields"]))
        + "</code>): "
        + escape(str(row["definition"]))
        + " <em>Formula:</em> <code>"
        + escape(str(row["formula"]))
        + "</code></li>"
        for row in rows
    )
    return (
        '<details class="column-definitions"><summary>Column definitions and '
        f"formulas</summary><ul>{items}</ul></details>"
    )


_COLUMN_DEFINITION_CATALOG_STEM = "table_column_definitions"


def table_column_definition_rows(
    tables: Sequence[ReportTable],
) -> list[dict[str, Any]]:
    """Describe every actual output field in a collection of report tables."""

    output: list[dict[str, Any]] = []
    for table in tables:
        if table.name == _COLUMN_DEFINITION_CATALOG_STEM:
            continue
        rows = compact_rows(table.rows) if table.compact else [dict(row) for row in table.rows]
        fields = list(dict.fromkeys(str(key) for row in rows for key in row))
        output.extend(
            column_definition_rows(
                fields,
                table_name=table.name,
                table_description=table.description,
            )
        )
    return output


def table_column_definition_rows_from_csv_directory(
    csv_directory: str | Path,
    *,
    excluded_stems: Sequence[str] = (_COLUMN_DEFINITION_CATALOG_STEM,),
) -> list[dict[str, Any]]:
    """Describe root CSV headers without reading or changing their data rows."""

    source = Path(csv_directory)
    excluded = set(excluded_stems)
    output: list[dict[str, Any]] = []
    for path in sorted(source.glob("*.csv"), key=lambda item: item.name.encode("utf-8")):
        if not path.is_file() or path.stem in excluded:
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            fields = next(csv.reader(stream), [])
        output.extend(
            column_definition_rows(
                fields,
                table_name=path.stem,
                table_description="Persisted root CSV report table",
            )
        )
    return output


def _column_catalog_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Table column definitions and formulas",
        "",
        "This documentation catalog describes source report tables. The catalog "
        "artifact excludes itself to prevent recursive documentation rows.",
        "",
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("table_name", "")), []).append(row)
    for table_name, selected in grouped.items():
        lines.extend((f"## `{_markdown_text(table_name)}`", ""))
        description = str(selected[0].get("table_description", "")).strip()
        if description:
            lines.extend((_markdown_text(description), ""))
        for row in selected:
            lines.append(
                f"- **{_markdown_text(row['display_label'])}** "
                f"(`{_markdown_text(row['source_fields'])}`): "
                f"{_markdown_text(row['definition'])} Formula: "
                f"`{_markdown_text(row['formula'])}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_table_column_definitions(
    output_directory: str | Path,
    *,
    tables: Sequence[ReportTable] | None = None,
    csv_directory: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write CSV, JSON, and Markdown column documentation without recursion."""

    if (tables is None) == (csv_directory is None):
        raise ValueError("provide exactly one of tables or csv_directory")
    rows = (
        table_column_definition_rows(tables or ())
        if tables is not None
        else table_column_definition_rows_from_csv_directory(csv_directory)
    )
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(target / f"{_COLUMN_DEFINITION_CATALOG_STEM}.csv", rows)
    json_path = target / f"{_COLUMN_DEFINITION_CATALOG_STEM}.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = target / "TABLE_COLUMN_DEFINITIONS.md"
    markdown_path.write_text(_column_catalog_markdown(rows), encoding="utf-8")
    return csv_path, json_path, markdown_path


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_score_field(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _SCORE_WORDS) and not any(
        word in lowered for word in ("count", "rank", "runtime")
    )


def format_mean_sd(
    mean: Any,
    sd: Any,
    *,
    percent: bool = False,
    decimals: int = 1,
) -> str:
    """Format one mean/SD pair without manufacturing an SD when unavailable."""

    mean_value = _number(mean)
    sd_value = _number(sd)
    if mean_value is None:
        return "N/A"
    scale = 100.0 if percent else 1.0
    rendered_mean = f"{mean_value * scale:.{decimals}f}"
    if sd_value is None:
        return rendered_mean
    return f"{rendered_mean} ± {sd_value * scale:.{decimals}f}"


def format_interval(
    lower: Any,
    upper: Any,
    *,
    percent: bool = False,
    decimals: int = 1,
) -> str:
    """Format one two-sided interval without hiding unavailable bounds."""

    lower_value = _number(lower)
    upper_value = _number(upper)
    if lower_value is None or upper_value is None:
        return "N/A"
    scale = 100.0 if percent else 1.0
    return (
        f"[{lower_value * scale:.{decimals}f}, "
        f"{upper_value * scale:.{decimals}f}]"
    )


def _mean_sd_pairs(fields: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Discover conventional report mean/SD pairs without table-name checks."""

    available = set(fields)
    pairs: dict[str, tuple[str, str]] = {}
    if "mean" in available:
        for candidate in ("sample_sd", "population_sd", "sd", "std"):
            if candidate in available:
                pairs["mean"] = ("mean", candidate)
                break
    for mean_field in fields:
        candidates: list[str] = []
        candidates.append(f"{mean_field}_sd")
        if mean_field.endswith("_mean"):
            stem = mean_field[: -len("_mean")]
            candidates.extend((f"{stem}_sample_sd", f"{stem}_population_sd", f"{stem}_sd"))
        if mean_field.startswith("participant_mean_"):
            metric = mean_field[len("participant_mean_") :]
            candidates.extend(
                (
                    f"repeat_{metric}_sample_sd",
                    f"repeat_{metric}_population_sd",
                )
            )
        for candidate in candidates:
            if candidate in available:
                pairs[mean_field] = (mean_field, candidate)
                break
    return pairs


def compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a presentation projection with every available mean/SD collapsed.

    Raw JSON retains numeric CI bounds, extrema, and the individual mean/SD
    columns.  The compact projection keeps CI95 as one readable interval while
    removing only redundant distribution columns belonging to a successfully
    paired metric.
    """

    source = [dict(row) for row in rows]
    fields = list(dict.fromkeys(str(key) for row in source for key in row))
    pairs = _mean_sd_pairs(fields)
    if not pairs:
        return source
    paired_sd = {sd for _, sd in pairs.values()}
    removable: set[str] = set(paired_sd)
    interval_fields: dict[str, tuple[str, str, str]] = {}
    for mean_field in pairs:
        if mean_field == "mean":
            if {"ci95_low", "ci95_high"} <= set(fields):
                interval_fields[mean_field] = (
                    "ci95_low", "ci95_high", "ci95"
                )
            removable.update(
                {
                    "population_sd",
                    "ci95_low",
                    "ci95_high",
                    "ci95_margin",
                    "minimum",
                    "maximum",
                }
            )
        elif mean_field.startswith("participant_mean_"):
            metric = mean_field[len("participant_mean_") :]
            low = f"repeat_{metric}_ci95_low"
            high = f"repeat_{metric}_ci95_high"
            if {low, high} <= set(fields):
                interval_fields[mean_field] = (
                    low,
                    high,
                    f"participant_{metric}_repeat_ci95",
                )
            removable.update(
                {
                    f"repeat_{metric}_population_sd",
                    f"repeat_{metric}_sample_sd",
                    f"repeat_{metric}_ci95_low",
                    f"repeat_{metric}_ci95_high",
                    f"repeat_{metric}_ci95_margin",
                    f"repeat_{metric}_minimum",
                    f"repeat_{metric}_maximum",
                }
            )
    output: list[dict[str, Any]] = []
    for row in source:
        projected: dict[str, Any] = {}
        for field in fields:
            if field in removable:
                continue
            if field in pairs:
                mean_field, sd_field = pairs[field]
                rendered_name = (
                    "mean_sd"
                    if field == "mean"
                    else (
                        "participant_"
                        + field[len("participant_mean_") :]
                        + "_mean_sd"
                    )
                    if field.startswith("participant_mean_")
                    else f"{field}_mean_sd"
                    if sd_field == f"{field}_sd"
                    else f"{field}_sd"
                )
                projected[rendered_name] = format_mean_sd(
                    row.get(mean_field),
                    row.get(sd_field),
                    percent=_is_score_field(
                        str(row.get("metric", mean_field))
                        if mean_field == "mean"
                        else mean_field
                    ),
                )
                interval = interval_fields.get(mean_field)
                if interval is not None:
                    low, high, rendered_interval_name = interval
                    projected[rendered_interval_name] = format_interval(
                        row.get(low),
                        row.get(high),
                        percent=_is_score_field(
                            str(row.get("metric", mean_field))
                            if mean_field == "mean"
                            else mean_field
                        ),
                    )
            else:
                projected[field] = row.get(field)
        output.append(projected)
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write one table to one RFC-4180-style UTF-8 CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    if not fields:
        target.write_text("\n", encoding="utf-8")
        return target
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return target


def _sheet_name(raw: str, used: set[str]) -> str:
    base = re.sub(r"[\\/*?:\[\]]", "_", raw).strip(" '") or "table"
    base = base[:31]
    candidate = base
    index = 2
    while candidate.casefold() in used:
        suffix = f"_{index}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


_EXCEL_CELL_TEXT_LIMIT = 32_767
_EXCEL_MAX_COLUMNS = 16_384


def _cell_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif value is None:
        return ""
    return str(value)


def _cell_text_chunks(value: Any) -> tuple[str, ...]:
    """Split long text across lossless Excel-safe continuation cells."""

    text = _cell_text(value)
    return tuple(
        text[offset : offset + _EXCEL_CELL_TEXT_LIMIT]
        for offset in range(0, len(text), _EXCEL_CELL_TEXT_LIMIT)
    ) or ("",)


def _column_name(index: int) -> str:
    value = index + 1
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(65 + remainder) + output
    return output


def _worksheet_xml(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    if not fields:
        fields = ["status"]
        rows = ({"status": "N/A_no_rows"},)
    chunked_rows = [
        {
            field: _cell_text_chunks(row.get(field))
            for field in fields
        }
        for row in rows
    ]
    chunk_counts = {
        field: max(len(row[field]) for row in chunked_rows)
        for field in fields
    }
    columns = [
        (field, chunk_index)
        for field in fields
        for chunk_index in range(chunk_counts[field])
    ]
    if len(columns) > _EXCEL_MAX_COLUMNS:
        raise ValueError(
            "lossless workbook continuation columns exceed the Excel limit: "
            f"{len(columns)} > {_EXCEL_MAX_COLUMNS}"
        )
    header = [
        field if chunk_index == 0 else f"{field}__continuation_{chunk_index + 1}"
        for field, chunk_index in columns
    ]
    xml_rows: list[str] = []
    for row_index in range(1, len(rows) + 2):
        cells: list[str] = []
        for column_index, (field, chunk_index) in enumerate(columns):
            reference = f"{_column_name(column_index)}{row_index}"
            if row_index == 1:
                text = escape(header[column_index])
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is>'
                    f'<t xml:space="preserve">{text}</t></is></c>'
                )
                continue
            source_row = rows[row_index - 2]
            chunks = chunked_rows[row_index - 2][field]
            value = source_row.get(field) if chunk_index == 0 else None
            number = _number(value) if not isinstance(value, bool) else None
            if (
                chunk_index == 0
                and len(chunks) == 1
                and number is not None
                and not isinstance(value, str)
            ):
                cells.append(f'<c r="{reference}"><v>{number:.17g}</v></c>')
            else:
                text = escape(
                    chunks[chunk_index] if chunk_index < len(chunks) else ""
                )
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is>'
                    f'<t xml:space="preserve">{text}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def write_excel_workbook(path: str | Path, tables: Sequence[ReportTable]) -> Path:
    """Write a dependency-free XLSX workbook with exactly one sheet per table."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    prepared: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = []
    for table in tables:
        sheet = _sheet_name(table.name, used)
        rows = compact_rows(table.rows) if table.compact else [dict(row) for row in table.rows]
        prepared.append((table.name, sheet, rows))
    workbook_sheets = "".join(
        f'<sheet name="{escape(sheet)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (_name, sheet, _rows) in enumerate(prepared, start=1)
    )
    relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(prepared) + 1)
    )
    styles_id = len(prepared) + 1
    relationships += (
        '<Relationship '
        f'Id="rId{styles_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(prepared) + 1)
    )
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{overrides}</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}</Relationships>"
        ),
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/>'
            '</border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
            'borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
            'borderId="0" xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" '
            'builtinId="0"/></cellStyles></styleSheet>'
        ),
    }
    for index, (_name, _sheet, rows) in enumerate(prepared, start=1):
        files[f"xl/worksheets/sheet{index}.xml"] = _worksheet_xml(rows)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return target


def write_excel_workbook_from_csv_directory(
    path: str | Path,
    csv_directory: str | Path,
) -> Path:
    """Write exactly one worksheet for every root CSV in ``csv_directory``.

    The directory is enumerated only after callers have finished exporting all
    report CSVs.  Reading the persisted presentation tables back prevents a
    late table such as ``table_figure_pairs.csv`` from being silently omitted
    from the workbook and also includes any auditable root CSV retained from an
    earlier compatible reporter version.  Nested auxiliary CSVs are excluded.
    The process-global CSV parser limit is raised using the largest platform-
    accepted integer and restored in ``finally``; long Excel text is preserved
    across explicit continuation columns rather than silently truncated.
    """

    source = Path(csv_directory)
    csv_paths = tuple(
        sorted(
            (
                candidate
                for candidate in source.glob("*.csv")
                if candidate.is_file()
            ),
            key=lambda candidate: candidate.name.encode("utf-8"),
        )
    )
    if not csv_paths:
        raise ValueError(f"workbook source contains no root CSV tables: {source}")
    tables: list[ReportTable] = []
    previous_field_limit = csv.field_size_limit()
    candidate_field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(candidate_field_limit)
            break
        except OverflowError:
            candidate_field_limit //= 10
    try:
        for csv_path in csv_paths:
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            tables.append(
                ReportTable(
                    name=csv_path.stem,
                    rows=rows,
                    description="Persisted root CSV report table",
                    compact=False,
                )
            )
    finally:
        csv.field_size_limit(previous_field_limit)
    return write_excel_workbook(path, tables)


__all__ = [
    "ColumnDefinition",
    "ReportTable",
    "column_definition",
    "column_definition_rows",
    "compact_rows",
    "format_mean_sd",
    "format_interval",
    "html_column_definitions_block",
    "markdown_column_definitions_block",
    "table_column_definition_rows",
    "table_column_definition_rows_from_csv_directory",
    "write_csv",
    "write_excel_workbook",
    "write_excel_workbook_from_csv_directory",
    "write_table_column_definitions",
]
