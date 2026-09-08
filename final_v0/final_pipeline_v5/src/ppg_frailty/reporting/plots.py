"""Static report figures with explicit N/A artifacts when inputs are absent."""

from __future__ import annotations

import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .analyze import StudyAnalysis
from .collect import CollectedStudy
from ..data.schema import CANONICAL_CLASS_NAMES

LEGACY_BRIDGE_FIGURE_NAMES = (
    "legacy_bridge_numeric_ablation_report",
    "legacy_bridge_execution_order_report",
)

STAGE3_STAR_FIGURE_NAMES = (
    "stage3_star_model_deltas",
    "stage3_star_fold_delta_heatmap",
)

FIGURE_TABLE_SOURCES: Mapping[str, tuple[str, ...]] = {
    "classification_prediction_scores": (
        "classification_prediction_scores",
        "classification_diagnostic_status",
    ),
    "classification_prediction_tsne": (
        "classification_prediction_tsne",
        "classification_diagnostic_status",
    ),
    "classification_roc_auc_curves": (
        "classification_roc_curves",
        "classification_diagnostic_status",
    ),
    "leaderboard": ("predictive_leaderboard", ),
    "stability": ("repeat_metrics", ),
    "macro_f1_stability": ("repeat_metrics", ),
    "roc_pr_auc_stability": ("repeat_metrics", ),
    "per_class_metric_stability": (
        "repeat_per_class_metrics",
        "per_class_metric_distribution_summary",
    ),
    "worst_class_f1_stability": ("worst_class_f1_stability", ),
    "fold_heatmap": ("fold_metrics", ),
    "paired_deltas": ("paired_deltas", ),
    "ablation_sensitivity_metrics": (
        "paired_deltas",
        "stage3_star_fold_contrasts",
    ),
    "coverage": ("coverage", ),
    "route_role_coverage": ("route_role_coverage", ),
    "denoiser_hr_comparison": ("denoiser_hr_comparison", ),
    "quality_distributions": ("quality_distributions", ),
    "calibration": ("calibration_bins", ),
    "confusion_matrices": ("confusion_matrices", ),
    "confusion_matrices_row_normalized": ("confusion_row_normalized", ),
    "per_class": ("per_class_metrics", ),
    "aggregation_view_metrics": ("aggregation_view_comparison", ),
    "aggregation_hierarchy_coverage": ("aggregation_hierarchy_coverage", ),
    "aggregation_view_confusion_matrices": ("aggregation_view_confusion_matrices", ),
    "aggregation_view_confusion_matrices_row_normalized": ("aggregation_view_confusion_matrices", ),
    "aggregation_view_per_class": ("aggregation_view_per_class_metrics", ),
    "learning_curves": ("training_history_raw", ),
    "top_learning_curves": ("training_history_raw", "predictive_leaderboard"),
    "balanced_accuracy_learning_curves": ("training_history_raw", ),
    "top_balanced_accuracy_learning_curves": (
        "training_history_raw",
        "predictive_leaderboard",
    ),
    "parameter_effects": ("varied_parameters", "repeat_metrics"),
    "parameter_interaction": ("varied_parameters", "repeat_metrics"),
    "legacy_bridge_numeric_ablation_report": ("legacy_bridge_numeric_ablation_report", ),
    "legacy_bridge_execution_order_report": ("legacy_bridge_execution_order_report", ),
    "stage3_star_model_deltas": ("stage3_star_contrasts", ),
    "stage3_star_fold_delta_heatmap": ("stage3_star_fold_contrasts", ),
}

STATIC_FIGURE_NAMES = (
    "classification_prediction_scores",
    "classification_prediction_tsne",
    "classification_roc_auc_curves",
    "leaderboard",
    "stability",
    "macro_f1_stability",
    "roc_pr_auc_stability",
    "per_class_metric_stability",
    "worst_class_f1_stability",
    "fold_heatmap",
    "paired_deltas",
    "ablation_sensitivity_metrics",
    "coverage",
    "route_role_coverage",
    "denoiser_hr_comparison",
    "quality_distributions",
    "calibration",
    "confusion_matrices",
    "confusion_matrices_row_normalized",
    "per_class",
    "aggregation_view_metrics",
    "aggregation_hierarchy_coverage",
    "aggregation_view_confusion_matrices",
    "aggregation_view_confusion_matrices_row_normalized",
    "aggregation_view_per_class",
    "learning_curves",
    "top_learning_curves",
    "balanced_accuracy_learning_curves",
    "top_balanced_accuracy_learning_curves",
    "parameter_effects",
    "parameter_interaction",
    *LEGACY_BRIDGE_FIGURE_NAMES,
    *STAGE3_STAR_FIGURE_NAMES,
)


def _is_stage3_centered_star(plan: Mapping[str, Any]) -> bool:
    bridge = plan.get("legacy_bridge")
    return isinstance(bridge, Mapping) and str(bridge.get("design", "")) == ("centered_star_v1")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _class_tick_labels(order: Sequence[Any]) -> list[str]:
    return [
        f"{value} · {CANONICAL_CLASS_NAMES.get(int(value), str(value))}"
        if str(value).lstrip("-").isdigit() else str(value) for value in order
    ]


def _category_tick_label(value: Any) -> str:
    """Wrap structured case identifiers without moving their tick anchor."""

    return "\n".join(str(value).split("__"))


def _diagnostic_groups(
    rows: Sequence[Mapping[str, Any]], ) -> list[tuple[tuple[str, str, str], list[Mapping[str, Any]]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("classifier_id")),
            str(row.get("evaluation_id")),
            str(row.get("aggregation_level")),
        )
        grouped.setdefault(key, []).append(row)
    return sorted(grouped.items())


def _diagnostic_panel_grid(
    pyplot: Any, panel_count: int, *, width: float = 5.2, height: float = 4.2
) -> tuple[Any, np.ndarray]:
    if panel_count <= 0:
        raise ValueError("classification diagnostic requires OOF prediction rows")
    columns = min(4, int(math.ceil(math.sqrt(panel_count))))
    rows = int(math.ceil(panel_count / columns))
    figure, axes = pyplot.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
    )
    for axis in axes.flat[panel_count:]:
        axis.axis("off")
    return figure, axes


def _classification_prediction_scores(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    pyplot: Any,
) -> Any:
    groups = _diagnostic_groups(analysis.classification_prediction_scores)
    figure, axes = _diagnostic_panel_grid(pyplot, len(groups))
    report = collected.plan.get("report", {})
    report = report if isinstance(report, Mapping) else {}
    bins = int(report.get("classification_score_histogram_bins", 40))
    edges = np.linspace(0.0, 1.0, bins + 1)
    for axis, (key, rows) in zip(axes.flat, groups, strict=False):
        classifier_id, evaluation_id, level = key
        class_order = tuple(int(value) for value in rows[0]["class_order"])
        if len(class_order) == 2:
            positive = class_order[-1]
            field = f"probability_class_{positive}"
            for true_label, color in zip(class_order, ("tab:blue", "tab:orange")):
                values = [float(row[field]) for row in rows if int(row["true_label"]) == true_label]
                axis.hist(
                    values,
                    bins=edges,
                    alpha=0.58,
                    color=color,
                    label=f"true={true_label} (n={len(values)})",
                )
            thresholds = [float(row["decision_threshold"]) for row in rows if row.get("decision_threshold") is not None]
            if thresholds:
                threshold = float(np.median(thresholds))
                axis.axvline(
                    threshold,
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    label=f"threshold={threshold:.4g}",
                )
            xlabel = f"P(class={positive})"
        else:
            for correct, color, label in (
                (True, "tab:green", "correct"),
                (False, "tab:red", "incorrect"),
            ):
                values = [
                    float(row["predicted_confidence"]) for row in rows if bool(row["prediction_correct"]) is correct
                ]
                if values:
                    axis.hist(
                        values,
                        bins=edges,
                        alpha=0.58,
                        color=color,
                        label=f"{label} (n={len(values)})",
                    )
            axis.text(
                0.02,
                0.97,
                "multiclass argmax · no scalar threshold",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
            xlabel = "Maximum predicted class probability"
        axis.set(xlim=(0.0, 1.0), xlabel=xlabel, ylabel="Count", title=f"{classifier_id}\n{evaluation_id} · {level}")
        axis.title.set_fontsize(9)
        axis.grid(axis="y", alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    figure.suptitle("OOF prediction-score distributions and decision threshold")
    figure.tight_layout()
    return figure


def _classification_prediction_tsne(analysis: StudyAnalysis, pyplot: Any) -> Any:
    groups = _diagnostic_groups(analysis.classification_prediction_tsne)
    figure, axes = _diagnostic_panel_grid(pyplot, len(groups))
    for axis, (key, rows) in zip(axes.flat, groups, strict=False):
        classifier_id, evaluation_id, level = key
        labels = sorted({int(row["true_label"]) for row in rows})
        for label in labels:
            selected = [row for row in rows if int(row["true_label"]) == label]
            axis.scatter(
                [float(row["tsne_x"]) for row in selected],
                [float(row["tsne_y"]) for row in selected],
                s=22,
                alpha=0.72,
                label=f"true={label}",
            )
        incorrect = [row for row in rows if not bool(row["prediction_correct"])]
        if incorrect:
            axis.scatter(
                [float(row["tsne_x"]) for row in incorrect],
                [float(row["tsne_y"]) for row in incorrect],
                s=36,
                marker="x",
                linewidths=1.0,
                color="black",
                label="misclassified",
            )
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=f"{classifier_id}\n{evaluation_id} · {level}")
        axis.title.set_fontsize(9)
        axis.grid(alpha=0.15)
        axis.legend(fontsize=7)
    figure.suptitle("Prediction-space t-SNE (persisted OOF probabilities; not hidden features)")
    figure.tight_layout()
    return figure


def _classification_roc_auc_curves(analysis: StudyAnalysis, pyplot: Any) -> Any:
    groups = _diagnostic_groups(analysis.classification_roc_curves)
    figure, axes = _diagnostic_panel_grid(pyplot, len(groups))
    for axis, (key, rows) in zip(axes.flat, groups, strict=False):
        classifier_id, evaluation_id, level = key
        curves: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in rows:
            curve_key = (str(row["curve"]), str(row["class_label"]))
            curves.setdefault(curve_key, []).append(row)
        for (curve, class_label), points in sorted(curves.items()):
            points = sorted(points, key=lambda row: int(row["point_index"]))
            curve_auc = float(points[0]["roc_auc"])
            is_macro = curve == "macro_average_ovr"
            axis.plot(
                [float(row["false_positive_rate"]) for row in points],
                [float(row["true_positive_rate"]) for row in points],
                linewidth=2.2 if is_macro else 1.2,
                linestyle="-" if is_macro else "--",
                label=f"{'macro OvR' if is_macro else f'class {class_label} OvR'} (AUC={curve_auc:.3f})",
            )
        axis.plot([0, 1], [0, 1], color="0.45", linestyle=":", linewidth=1.0)
        axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="False-positive rate", ylabel="True-positive rate")
        axis.set_title(f"{classifier_id}\n{evaluation_id} · {level}", fontsize=9)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="lower right")
    figure.suptitle("Empirical OOF ROC curves (pooled participant-repeat display; AUC shown)")
    figure.tight_layout()
    return figure


def _set_centered_category_ticks(
    axis: Any,
    positions: Sequence[float],
    labels: Sequence[Any],
    *,
    rotation: float = 0.0,
) -> None:
    """Bind categorical labels to explicit numeric centres on every plot."""

    numeric = np.asarray(positions, dtype=np.float64)
    axis.set_xticks(numeric)
    axis.set_xticklabels(
        [_category_tick_label(value) for value in labels],
        rotation=rotation,
        ha="center",
        rotation_mode="anchor",
    )
    axis.tick_params(axis="x", labelsize="small", pad=6)
    if numeric.size:
        axis.set_xlim(float(numeric.min()) - 0.5, float(numeric.max()) + 0.5)


def _unlink_if_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()


def clear_static_figure_artifacts(directory: str | Path) -> None:
    """Remove only reporter-owned PNG/N/A counterparts for deterministic reruns."""

    target = Path(directory)
    for name in STATIC_FIGURE_NAMES:
        _unlink_if_file(target / f"{name}.png")
        _unlink_if_file(target / f"{name}.NA.txt")


def _figure_status(directory: Path, name: str, status: str, target: Path, reason: str = "") -> dict[str, Any]:
    return dict(figure=name, status=status, path=str(target.relative_to(directory.parent)), reason=reason.strip())


def _na(
    directory: Path,
    name: str,
    reason: str,
    *,
    pyplot: Any | None = None,
) -> dict[str, Any]:
    render_png = pyplot is not None
    target = directory / f"{name}{'.png' if render_png else '.NA.txt'}"
    counterpart = directory / f"{name}{'.NA.txt' if render_png else '.png'}"
    temporary = directory / f".{name}.NA.tmp-{os.getpid()}-{time.time_ns()}{target.suffix}"
    figure = None
    try:
        if not render_png:
            temporary.write_text(f"N/A: {reason.strip()}\n", encoding="utf-8")
        else:
            figure, axis = pyplot.subplots(figsize=(9.0, 3.8))
            axis.axis("off")
            if name in {
                    "balanced_accuracy_learning_curves",
                    "top_balanced_accuracy_learning_curves",
            }:
                scope = "top-ranked " if name.startswith("top_") else ""
                na_title = f"N/A — {scope}balanced-accuracy learning curve"
            else:
                na_title = f"N/A — {name.replace('_', ' ')}"
            for y, text, size, options in (
                (0.57, na_title, 16, {"weight": "bold"}),
                (0.38, reason.strip(), 10, {"wrap": True}),
            ):
                axis.text(0.5, y, text, ha="center", va="center", fontsize=size, transform=axis.transAxes, **options)
            figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        os.replace(temporary, target)
        _unlink_if_file(counterpart)
    finally:
        _unlink_if_file(temporary)
        if figure is not None:
            pyplot.close(figure)
    return _figure_status(directory, name, "N/A", target, reason)


def _save(
    directory: Path,
    name: str,
    draw: Callable[[Any], Any],
    pyplot: Any,
    *,
    render_na_png: bool = False,
) -> dict[str, Any]:
    figure = None
    temporary = directory / f".{name}.tmp-{os.getpid()}-{time.time_ns()}.png"
    try:
        figure = draw(pyplot)
        target = directory / f"{name}.png"
        figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        os.replace(temporary, target)
        _unlink_if_file(directory / f"{name}.NA.txt")
        return _figure_status(directory, name, "generated", target)
    except Exception as error:  # noqa: BLE001 - report remains usable.
        return _na(directory, name, f"{type(error).__name__}: {error}",
                   pyplot=pyplot if render_na_png else None)
    finally:
        _unlink_if_file(temporary)
        if figure is not None:
            pyplot.close(figure)


def _paired_horizontal_bars(
    pyplot: Any,
    labels: Sequence[str],
    first: Sequence[float],
    second: Sequence[float],
    *,
    first_label: str,
    second_label: str,
    title: str,
    min_height: float,
    row_height: float,
    second_errors: Sequence[float] | None = None,
    tight: bool = False,
) -> Any:
    positions = np.arange(len(labels), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(9, max(min_height, len(labels) * row_height)))
    axis.barh(positions - 0.18, first, height=0.34, label=first_label)
    error = {"xerr": second_errors, "capsize": 3} if second_errors is not None else {}
    axis.barh(positions + 0.18, second, height=0.34, label=second_label, **error)
    axis.set_yticks(positions, labels)
    axis.set(xlim=(0.0, 1.0), xlabel="Participant-level score", title=title)
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    if tight:
        figure.tight_layout()
    return figure


def _leaderboard(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row for row in analysis.predictive_leaderboard
        if _number(row.get("participant_mean_abstention_aware_balanced_accuracy")) is not None
    ]
    if not rows:
        raise ValueError("no finite case-level predictive metrics")
    rows.reverse()
    return _paired_horizontal_bars(
        pyplot,
        [str(row["case_id"]) for row in rows],
        [float(row["participant_mean_abstention_aware_balanced_accuracy"]) for row in rows],
        [_number(row.get("participant_mean_abstention_aware_macro_f1")) or 0.0 for row in rows],
        first_label="Abstention-aware balanced accuracy",
        second_label="Abstention-aware Macro-F1",
        title="Predictive leaderboard (manual review; no automatic winner)",
        min_height=3.5,
        row_height=0.55,
    )


def _stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    cases: dict[str, list[float]] = {}
    for row in analysis.repeat_metrics:
        value = _number(row.get("balanced_accuracy"))
        if value is not None:
            cases.setdefault(str(row["case_id"]), []).append(value)
    if not cases:
        raise ValueError("repeat-level balanced accuracy unavailable")
    labels = sorted(cases)
    positions = np.arange(len(labels), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(max(7, len(labels) * 1.35), 4.8))
    axis.boxplot(
        [cases[label] for label in labels],
        positions=positions,
        showmeans=True,
    )
    _set_centered_category_ticks(axis, positions, labels)
    axis.set(ylim=(0.0, 1.0), ylabel="Balanced accuracy", title="Repeat stability")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _metric_boxplot_panels(
    rows: Sequence[Mapping[str, Any]],
    pyplot: Any,
    *,
    metrics: Sequence[tuple[str, str]],
    group_fields: Sequence[str],
    title: str,
) -> Any:
    """Draw configurable metric boxplots from arbitrary report rows."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = " · ".join(str(row.get(field, "N/A")) for field in group_fields)
        grouped.setdefault(label, []).append(row)
    available = [(field, label) for field, label in metrics if any(_number(row.get(field)) is not None for row in rows)]
    if not grouped or not available:
        raise ValueError("configured repeat-level metrics unavailable")
    labels = sorted(grouped)
    figure, axes = pyplot.subplots(
        len(available),
        1,
        figsize=(max(8.0,
                     len(labels) * 1.3), max(4.2,
                                             len(available) * 3.6)),
        squeeze=False,
    )
    positions = np.arange(len(labels), dtype=np.float64)
    for axis, (metric, metric_label) in zip(axes.flat, available):
        values = [[value for row in grouped[label] if (value := _number(row.get(metric))) is not None]
                  for label in labels]
        values = [current if current else [np.nan] for current in values]
        axis.boxplot(values, positions=positions, showmeans=True)
        _set_centered_category_ticks(axis, positions, labels, rotation=30.0)
        if "delta" in metric:
            finite = [abs(value) for group in values for value in group if np.isfinite(value)]
            limit = max(finite, default=0.05) * 1.15
            axis.set_ylim(-limit, limit)
            axis.axhline(0.0, color="black", linewidth=1)
        else:
            axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(metric_label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _macro_f1_stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    return _metric_boxplot_panels(
        analysis.repeat_metrics,
        pyplot,
        metrics=(("macro_f1", "Macro-F1"), ),
        group_fields=("case_id", ),
        title="Repeat-level Macro-F1 stability",
    )


def _roc_pr_auc_stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    return _metric_boxplot_panels(
        analysis.repeat_metrics,
        pyplot,
        metrics=(
            ("macro_roc_auc_ovr", "Macro ROC AUC (one-vs-rest)"),
            ("macro_pr_auc_ovr", "Macro PR AUC (one-vs-rest)"),
        ),
        group_fields=("case_id", ),
        title="Repeat-level AUC scalar stability (not an ROC curve)",
    )


def _per_class_metric_stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    return _metric_boxplot_panels(
        analysis.repeat_per_class_metrics,
        pyplot,
        metrics=(
            ("balanced_accuracy_ovr", "Per-class BA (one-vs-rest)"),
            ("f1", "Per-class F1"),
            ("roc_auc_ovr", "Per-class ROC AUC"),
            ("pr_auc_ovr", "Per-class PR AUC"),
        ),
        group_fields=("case_id", "class_name"),
        title="Per-class repeat stability",
    )


def _worst_class_f1_stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.worst_class_f1_stability)
    if not rows:
        raise ValueError("worst-class F1 stability ranking unavailable")
    rows.reverse()
    return _paired_horizontal_bars(
        pyplot,
        [f"S{row['worst_class_f1_stability_rank']} · {row['case_id']}" for row in rows],
        [float(row["abstention_aware_worst_class_f1"]) for row in rows],
        [float(row["participant_mean_abstention_aware_balanced_accuracy"]) for row in rows],
        first_label="Abstention-aware worst-class F1",
        second_label="Abstention-aware mean BA ± repeat population SD",
        second_errors=[
            _number(row.get("repeat_abstention_aware_balanced_accuracy_population_sd")) or 0.0 for row in rows
        ],
        title="Abstention-aware worst-class F1 stability review (top 10)",
        min_height=3.8,
        row_height=0.58,
        tight=True,
    )


def _fold_heatmap(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [row for row in analysis.fold_metrics if _number(row.get("balanced_accuracy")) is not None]
    if not rows:
        raise ValueError("fold/cell balanced accuracy unavailable")
    cases = sorted({str(row["case_id"]) for row in rows})
    cells = sorted({(int(row["repeat"]), int(row["fold"])) for row in rows})
    matrix = np.full((len(cases), len(cells)), np.nan, dtype=np.float64)
    case_index = {value: index for index, value in enumerate(cases)}
    cell_index = {value: index for index, value in enumerate(cells)}
    for row in rows:
        matrix[case_index[str(row["case_id"])],
               cell_index[(int(row["repeat"]), int(row["fold"]))], ] = float(row["balanced_accuracy"])
    figure, axis = pyplot.subplots(figsize=(max(9, len(cells) * 0.42), max(3.5, len(cases) * 0.55)))
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_yticks(range(len(cases)), cases)
    axis.set_xticks(
        range(len(cells)),
        [f"r{repeat}/f{fold}" for repeat, fold in cells],
        rotation=60,
        ha="right",
    )
    axis.set_title("Fold × repeat balanced accuracy")
    figure.colorbar(image, ax=axis, label="Balanced accuracy")
    return figure


def _paired_deltas(analysis: StudyAnalysis, pyplot: Any) -> Any:
    groups: dict[str, list[float]] = {}
    for row in analysis.paired_deltas:
        value = _number(row.get("balanced_accuracy_delta"))
        if value is not None:
            groups.setdefault(str(row["case_id"]), []).append(value)
    if not groups:
        raise ValueError("reference-paired repeat deltas unavailable")
    labels = sorted(groups)
    positions = np.arange(len(labels), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(max(7, len(labels) * 1.35), 4.8))
    axis.boxplot(
        [groups[label] for label in labels],
        positions=positions,
        showmeans=True,
    )
    _set_centered_category_ticks(axis, positions, labels)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(ylabel="Δ balanced accuracy vs reference", title="Paired repeat deltas")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _ablation_sensitivity_metrics(analysis: StudyAnalysis, pyplot: Any) -> Any:
    if analysis.stage3_star_fold_contrasts:
        rows = analysis.stage3_star_fold_contrasts
        return _metric_boxplot_panels(
            rows,
            pyplot,
            metrics=(
                ("delta_native_balanced_accuracy", "Δ BA"),
                ("delta_native_macro_f1", "Δ Macro-F1"),
                ("delta_native_worst_class_f1", "Δ worst-class F1"),
            ),
            group_fields=("model", "factor_id"),
            title="Centered-star matched-fold ablation sensitivity",
        )
    return _metric_boxplot_panels(
        analysis.paired_deltas,
        pyplot,
        metrics=(
            ("balanced_accuracy_delta", "Δ BA"),
            ("macro_f1_delta", "Δ Macro-F1"),
            ("macro_roc_auc_ovr_delta", "Δ macro ROC AUC"),
            ("macro_pr_auc_ovr_delta", "Δ macro PR AUC"),
        ),
        group_fields=("case_id", ),
        title="Reference-paired ablation sensitivity",
    )


def _coverage(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [row for row in analysis.coverage if _number(row.get("mean_coverage_rate")) is not None]
    if not rows:
        raise ValueError("coverage metrics unavailable")
    labels = [str(row["case_id"]) for row in rows]
    values = [float(row["mean_coverage_rate"]) for row in rows]
    positions = np.arange(len(labels), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(max(7, len(labels) * 1.35), 4.5))
    axis.bar(positions, values)
    _set_centered_category_ticks(axis, positions, labels)
    axis.set(ylim=(0.0, 1.0), ylabel="Mean coverage rate", title="Coverage by case")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _route_role_coverage(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [row for row in analysis.route_role_coverage if _number(row.get("retained_coverage")) is not None]
    if not rows:
        raise ValueError("route/role coverage diagnostics unavailable")
    labels = [f"{row['case_id']}\n{row['role']} · {row['route_state']}" for row in rows]
    retained = [float(row["retained_coverage"]) for row in rows]
    available = [(1.0 - float(row["unavailable_predictor_rate"])
                  if _number(row.get("unavailable_predictor_rate")) is not None else np.nan) for row in rows]
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(max(8, len(rows) * 1.15), 5.2))
    axis.bar(
        positions - 0.18,
        retained,
        width=0.36,
        label="Retained coverage",
    )
    axis.bar(
        positions + 0.18,
        available,
        width=0.36,
        label="Available predictor fraction",
    )
    _set_centered_category_ticks(axis, positions, labels)
    axis.set(ylim=(0.0, 1.0), ylabel="Fraction", title="Route × role coverage and feature availability")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _quality_distributions(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [row for row in analysis.quality_distributions if _number(row.get("mean")) is not None]
    if not rows:
        raise ValueError("route/role quality-component distributions unavailable")
    labels = [f"{row['case_id']} / {row['role']} / {row['route_state']}\n{row['component']}" for row in rows]
    means = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
    deviations = np.asarray(
        [_number(row.get("population_sd")) or 0.0 for row in rows],
        dtype=np.float64,
    )
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(max(9, len(rows) * 0.75), 5.4))
    axis.errorbar(
        positions,
        means,
        yerr=deviations,
        fmt="o",
        capsize=3,
    )
    _set_centered_category_ticks(axis, positions, labels)
    axis.set(ylabel="Component value (mean ± population SD)", title="Quality distributions by route and role")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _paired_bars(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[tuple[str, str, str]],
    *,
    ylabel: str,
    title: str,
) -> None:
    positions = np.arange(len(rows), dtype=np.float64)
    for index, (field, sd_field, label) in enumerate(specs):
        axis.bar(
            positions + (-0.19 if index == 0 else 0.19),
            [float(row[field]) for row in rows],
            width=0.38,
            yerr=[_number(row.get(sd_field)) or 0.0 for row in rows],
            capsize=3,
            label=label,
        )
    _set_centered_category_ticks(axis, positions, [f"{row['denoiser_id']}\n{row['case_id']}" for row in rows])
    axis.set(ylabel=ylabel, title=title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)


def _denoiser_hr_comparison(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row for row in getattr(analysis, "denoiser_hr_comparison", ())
        if str(row.get("role_scope")) == "ALL" and str(row.get("outer_partition")) == "outer_oof"
        and _number(row.get("participant_macro_direct_hr_bpm")) is not None
        and _number(row.get("participant_macro_post_denoise_hr_bpm")) is not None
    ]
    if not rows:
        raise ValueError("paired direct/post-denoiser HR evidence unavailable")
    ppi_rows = [
        row for row in rows if _number(row.get("participant_macro_direct_median_ppi_ms")) is not None
        and _number(row.get("participant_macro_post_denoise_median_ppi_ms")) is not None
    ]
    panel_count = 2 if ppi_rows else 1
    figure, axes = pyplot.subplots(
        1,
        panel_count,
        figsize=(max(7.5,
                     len(rows) * 2.0) * panel_count, 5.2),
        squeeze=False,
    )
    _paired_bars(
        axes[0, 0],
        rows,
        (
            ("participant_macro_direct_hr_bpm", "participant_sd_direct_hr_bpm", "Direct PPG HR"),
            ("participant_macro_post_denoise_hr_bpm", "participant_sd_post_denoise_hr_bpm", "Post-denoiser PPG HR"),
        ),
        ylabel="Participant-macro HR (bpm; mean ± participant SD)",
        title="Paired direct versus post-denoiser heart rate",
    )
    if ppi_rows:
        _paired_bars(
            axes[0, 1],
            ppi_rows,
            (
                (
                    "participant_macro_direct_median_ppi_ms",
                    "participant_sd_direct_median_ppi_ms",
                    "Direct PPG median PPI",
                ),
                (
                    "participant_macro_post_denoise_median_ppi_ms",
                    "participant_sd_post_denoise_median_ppi_ms",
                    "Post-denoiser median PPI",
                ),
            ),
            ylabel="Participant-macro median PPI (ms; mean ± participant SD)",
            title="Paired direct versus post-denoiser PPI",
        )
    figure.tight_layout()
    return figure


def _calibration(analysis: StudyAnalysis, pyplot: Any) -> Any:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in analysis.calibration_bins:
        if _number(row.get("mean_confidence")) is not None:
            groups.setdefault(str(row["case_id"]), []).append(row)
    if not groups:
        raise ValueError("participant OOF calibration bins unavailable")
    figure, axis = pyplot.subplots(figsize=(6.5, 5.6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    for case_id, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["bin_index"]))
        axis.plot(
            [float(row["mean_confidence"]) for row in ordered],
            [float(row["accuracy"]) for row in ordered],
            marker="o",
            label=case_id,
        )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Mean confidence", ylabel="Observed accuracy",
             title="Participant-level calibration")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.25)
    return figure


def _annotate_matrix(
    axis: Any,
    matrix: np.ndarray,
    format_spec: str,
    color: Callable[[float], str] | None = None,
) -> None:
    for y, x in np.ndindex(matrix.shape):
        value = matrix[y, x]
        if np.isfinite(value):
            options = {"color": color(value)} if color else {}
            axis.text(x, y, format(value, format_spec), ha="center", va="center", **options)


def _confusion_panel(
    axis: Any,
    figure: Any,
    matrix: np.ndarray,
    order: Sequence[Any],
    title: str,
    *,
    normalized: bool = False,
    contrast_text: bool = False,
) -> None:
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0 if normalized else None,
        vmax=1.0 if normalized else None,
    )
    axis.set_xticks(range(len(order)), [str(value) for value in order])
    axis.set_yticks(range(len(order)), _class_tick_labels(order))
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    color = lambda value: "white" if contrast_text and value >= 0.55 else "black"
    _annotate_matrix(axis, matrix, ".1%" if normalized else ".0f", color)
    figure.colorbar(image, ax=axis, fraction=0.046)


def _confusion(analysis: StudyAnalysis, pyplot: Any, *, row_normalized: bool = False) -> Any:
    rows = list(analysis.confusion_matrices)
    ranks: dict[str, int] = {}
    if row_normalized:
        ranks = {str(row["case_id"]): int(row["predictive_rank"]) for row in analysis.predictive_leaderboard}
        rows = [row for row in rows if str(row.get("case_id")) in ranks]
        rows.sort(key=lambda row: ranks[str(row["case_id"])])
        rows = rows[:6]
    if not rows:
        qualifier = "top-case pooled" if row_normalized else "participant OOF or labeled cell"
        raise ValueError(f"{qualifier} confusion matrices unavailable")
    columns = min(3, len(rows))
    rows_n = int(math.ceil(len(rows) / columns))
    width, height = ((4.1, 3.9) if row_normalized else (4.0, 3.8))
    figure, axes = pyplot.subplots(
        rows_n,
        columns,
        figsize=(columns * width, rows_n * height),
        squeeze=False,
    )
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray(row["confusion_matrix"], dtype=np.float64)
        if row_normalized:
            totals = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0.0)
        raw_order = list(row["class_order"])
        case_id = str(row["case_id"])
        heading = f"#{ranks[case_id]} · " if row_normalized else ""
        _confusion_panel(
            axis,
            figure,
            matrix,
            raw_order,
            f"{heading}{case_id}\nclass_order={','.join(str(value) for value in raw_order)}",
            normalized=row_normalized,
            contrast_text=row_normalized,
        )
    for axis in axes.flat[len(rows):]:
        axis.axis("off")
    title = (
        "Top-case row-normalized pooled confusion matrices"
        if row_normalized
        else "Pooled confusion matrices (participant OOF or labeled cell fallback)"
    )
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _normalized_confusion(analysis: StudyAnalysis, pyplot: Any) -> Any:
    return _confusion(analysis, pyplot, row_normalized=True)


def _per_class(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.per_class_metrics)
    if not rows:
        raise ValueError("participant OOF or labeled cell per-class metrics unavailable")
    observed_cases = {str(row["case_id"]) for row in rows}
    ranked = [str(row["case_id"]) for row in analysis.predictive_leaderboard if str(row["case_id"]) in observed_cases]
    cases = ranked + sorted(observed_cases - set(ranked))
    classes = sorted(
        {str(row["class_label"])
         for row in rows},
        key=lambda value: ((0, int(value)) if value.lstrip("-").isdigit() else (1, value)),
    )
    lookup = {(str(row["case_id"]), str(row["class_label"])): row for row in rows}
    positions = np.arange(len(cases), dtype=np.float64)
    metrics = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
    width = 0.25
    figure, axes = pyplot.subplots(
        len(classes),
        1,
        figsize=(max(10,
                     len(cases) * 1.35), max(4.8,
                                             len(classes) * 3.2)),
        squeeze=False,
        sharex=True,
    )
    for axis, class_label in zip(axes.flat, classes):
        for metric_index, (metric, metric_label) in enumerate(metrics):
            values = [_number(lookup.get((case_id, class_label), {}).get(metric)) for case_id in cases]
            axis.bar(
                positions + (metric_index - 1) * width,
                [np.nan if value is None else value for value in values],
                width=width,
                label=metric_label,
            )
        axis.set(ylim=(0.0, 1.0), ylabel="Score", title=f"Class {_class_tick_labels([class_label])[0]}")
        axis.grid(axis="y", alpha=0.25)
    _set_centered_category_ticks(axes.flat[-1], positions, cases)
    axes.flat[0].legend(loc="upper right")
    figure.suptitle("Per-class pooled metrics", y=0.995)
    figure.tight_layout()
    return figure


def _aggregation_view_metrics(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row for row in analysis.aggregation_view_comparison
        if _number(row.get("participant_mean_balanced_accuracy")) is not None
    ]
    if not rows:
        raise ValueError("window/file/role-balanced participant views unavailable")
    views = (
        "window_balanced_to_participant",
        "line_a_equal_files",
        "line_b_equal_role_families",
    )
    cases = sorted({str(row["case_id"]) for row in rows})
    lookup = {(str(row["case_id"]), str(row["aggregation_view"])): row for row in rows}
    positions = np.arange(len(cases), dtype=np.float64)
    width = 0.25
    figure, axes = pyplot.subplots(
        2,
        1,
        figsize=(max(10,
                     len(cases) * 1.8), 8.8),
        sharex=True,
    )
    for axis, metric, label in (
        (axes[0], "participant_mean_balanced_accuracy", "BA"),
        (axes[1], "participant_mean_macro_f1", "Macro-F1"),
    ):
        available_by_case = {
            case_id: tuple(view for view in views if _number(lookup.get((case_id, view), {}).get(metric)) is not None)
            for case_id in cases
        }
        for view in views:
            bar_positions: list[float] = []
            values: list[float] = []
            for case_position, case_id in zip(positions, cases):
                available = available_by_case[case_id]
                if view not in available:
                    continue
                slot = available.index(view) - (len(available) - 1) / 2.0
                bar_positions.append(float(case_position + slot * width))
                values.append(float(lookup[(case_id, view)][metric]))
            axis.bar(
                bar_positions,
                values,
                width=width,
                label=view if label == "BA" else None,
            )
            axis.set(ylim=(0.0, 1.0), ylabel=label)
            axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Same fitted OOF: window-, file-, and role-balanced participant views")
    axes[0].legend(fontsize="small")
    _set_centered_category_ticks(axes[1], positions, cases, rotation=25.0)
    figure.tight_layout()
    return figure


def _aggregation_hierarchy_coverage(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.aggregation_hierarchy_coverage)
    if not rows:
        raise ValueError("window/file/role hierarchy coverage unavailable")
    requested = (
        ("window", "Window OOF: B/R1–R4"),
        ("file", "File-balanced input: B/R1–R4"),
        ("role", "Role-balanced input: B/R"),
    )
    available = [(level, title) for level, title in requested if any(
        str(row.get("aggregation_level")) == level for row in rows)]
    if not available:
        raise ValueError("no window/file/role coverage rows")
    cases = sorted({str(row["case_id"]) for row in rows})
    figure, axes = pyplot.subplots(
        len(available),
        1,
        figsize=(max(9,
                     len(cases) * 0.75), 3.4 * len(available)),
        squeeze=False,
    )
    for axis, (level, title) in zip(axes.flat, available):
        current = [row for row in rows if str(row.get("aggregation_level")) == level]
        labels = sorted(
            {str(row["group_label"])
             for row in current},
            key=lambda value: (value[:1], int(value[1:]) if value[1:].isdigit() else -1),
        )
        values_by_key: dict[tuple[str, str], list[float]] = {}
        for row in current:
            values_by_key.setdefault((str(row["case_id"]), str(row["group_label"])),
                                     []).append(float(row["participant_count"]))
        matrix = np.full((len(cases), len(labels)), np.nan, dtype=np.float64)
        for row_index, case_id in enumerate(cases):
            for column_index, label in enumerate(labels):
                values = values_by_key.get((case_id, label), ())
                if values:
                    matrix[row_index, column_index] = float(np.mean(values))
        image = axis.imshow(matrix, aspect="auto", cmap="Blues")
        axis.set_yticks(range(len(cases)), cases)
        axis.set_xticks(range(len(labels)), labels)
        axis.set_title(f"{title} — mean distinct participants per repeat")
        _annotate_matrix(axis, matrix, ".0f")
        figure.colorbar(image, ax=axis, fraction=0.025, label="Participants")
    figure.tight_layout()
    return figure


def _aggregation_view_confusions(
    analysis: StudyAnalysis,
    pyplot: Any,
    *,
    row_normalized: bool = False,
) -> Any:
    rows = list(analysis.aggregation_view_confusion_matrices)
    if not rows:
        raise ValueError("aggregation-view confusion matrices unavailable")
    views = (
        "window_balanced_to_participant",
        "line_a_equal_files",
        "line_b_equal_role_families",
    )
    ranked = [str(row["case_id"]) for row in analysis.predictive_leaderboard]
    cases = ranked + sorted({str(row["case_id"]) for row in rows} - set(ranked))
    lookup = {(str(row["case_id"]), str(row["aggregation_view"])): row for row in rows}
    figure, axes = pyplot.subplots(
        len(cases),
        len(views),
        figsize=(4.0 * len(views), max(3.6, 3.35 * len(cases))),
        squeeze=False,
    )
    for row_index, case_id in enumerate(cases):
        for column_index, view in enumerate(views):
            axis = axes[row_index, column_index]
            row = lookup.get((case_id, view))
            if row is None:
                axis.axis("off")
                axis.set_title(f"{case_id}\n{view}\nN/A")
                continue
            counts = np.asarray(row["confusion_matrix"], dtype=np.float64)
            matrix = counts
            if row_normalized:
                totals = counts.sum(axis=1, keepdims=True)
                matrix = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0.0)
            raw_order = list(row["class_order"])
            _confusion_panel(
                axis,
                figure,
                matrix,
                raw_order,
                f"{case_id}\n{view}\nclass_order={','.join(str(value) for value in raw_order)}",
                normalized=row_normalized,
            )
            axis.tick_params(axis="both", labelsize="x-small")
            axis.title.set_fontsize("small")
    figure.suptitle(
        "Participant confusion matrices from the same fitted OOF "
        f"(three report views; {'row-normalized' if row_normalized else 'counts'})",
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.96,
        bottom=0.025,
        top=0.975,
        wspace=0.55,
        hspace=0.75,
    )
    return figure


def _aggregation_view_per_class(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.aggregation_view_per_class_metrics)
    if not rows:
        raise ValueError("aggregation-view per-class metrics unavailable")
    groups: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["aggregation_view"]),
            str(row["class_label"]),
        )
        bucket = groups.setdefault(key, {"precision": [], "recall": [], "f1": []})
        for metric in ("precision", "recall", "f1"):
            value = _number(row.get(metric))
            if value is not None:
                bucket[metric].append(value)
    view_labels = {
        "window_balanced_to_participant": "W · equal-window",
        "line_a_equal_files": "A · equal-file",
        "line_b_equal_role_families": "B · equal-role",
    }
    view_order = {view: index for index, view in enumerate(view_labels)}
    ranked_cases = [str(row["case_id"]) for row in analysis.predictive_leaderboard]
    all_cases = {key[0] for key in groups}
    case_order = ranked_cases + sorted(all_cases - set(ranked_cases))
    case_rank = {case_id: index for index, case_id in enumerate(case_order)}
    row_labels = sorted(
        {(key[0], key[1])
         for key in groups},
        key=lambda item: (
            case_rank.get(item[0], len(case_rank)),
            view_order.get(item[1], len(view_order)),
        ),
    )
    classes = sorted({key[2] for key in groups})
    figure, axes = pyplot.subplots(
        1,
        3,
        figsize=(22, max(8,
                         len(row_labels) * 0.45)),
        squeeze=False,
        sharey=True,
    )
    images = []
    for axis_index, (axis, metric) in enumerate(zip(axes.flat, ("precision", "recall", "f1"))):
        matrix = np.full((len(row_labels), len(classes)), np.nan, dtype=np.float64)
        for row_index, (case_id, view) in enumerate(row_labels):
            for column_index, class_label in enumerate(classes):
                values = groups.get((case_id, view, class_label), {}).get(metric, ())
                if values:
                    matrix[row_index, column_index] = float(np.mean(values))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        images.append(image)
        axis.set_yticks(range(len(row_labels)))
        if axis_index == 0:
            axis.set_yticklabels(
                [f"{case_id} · {view_labels.get(view, view)}" for case_id, view in row_labels],
                fontsize="x-small",
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_xticks(
            range(len(classes)),
            [f"{value}\n{CANONICAL_CLASS_NAMES.get(int(value), value)}" for value in classes],
        )
        axis.set(xlabel="Class label", title=metric.capitalize())
    figure.suptitle(
        "All-class metrics from the same fitted OOF aggregation views",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.34,
        right=0.95,
        bottom=0.07,
        top=0.94,
        wspace=0.08,
    )
    figure.colorbar(
        images[-1],
        ax=list(axes.flat),
        fraction=0.018,
        pad=0.015,
        label="Mean per-repeat score",
    )
    return figure


def _loss_history_metric_names(rows: Sequence[Mapping[str, Any]], ) -> tuple[str, ...]:
    preferred = (
        "training_loss",
        "inner_training_loss",
        "train_loss",
        "loss",
    )
    return tuple(name for name in preferred if any(_number(row.get(name)) is not None for row in rows))


def _balanced_accuracy_history_metric_names(rows: Sequence[Mapping[str, Any]], ) -> tuple[str, ...]:
    """Return only explicitly inner/train BA fields, never outer OOF metrics."""

    aliases = _BALANCED_ACCURACY_HISTORY_ALIASES
    selected: list[str] = []
    for alternatives in aliases:
        for name in alternatives:
            if any(_number(row.get(name)) is not None for row in rows):
                selected.append(name)
                break
    return tuple(selected)


_BALANCED_ACCURACY_HISTORY_ALIASES = (
    (
        "inner_participant_balanced_accuracy",
        "inner_balanced_accuracy",
    ),
    (
        "training_participant_balanced_accuracy",
        "train_participant_balanced_accuracy",
        "train_balanced_accuracy",
    ),
)


def _has_explicit_inner_or_train_balanced_accuracy(row: Mapping[str, Any], ) -> bool:
    return any(_number(row.get(name)) is not None for aliases in _BALANCED_ACCURACY_HISTORY_ALIASES for name in aliases)


def _explicitly_uses_outer_heldout(row: Mapping[str, Any]) -> bool:
    value = row.get("learning_curve_outer_heldout_used")
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "true",
        "yes",
        "1",
    }


def _learning_curves(collected: CollectedStudy, pyplot: Any) -> Any:
    rows = list(collected.history_rows)
    if not rows:
        raise ValueError("training history unavailable")
    x_name = "epoch" if any("epoch" in row for row in rows) else "step"
    metrics = _loss_history_metric_names(rows)
    if not metrics:
        raise ValueError("training history has no declared train/inner-train loss")
    groups: dict[tuple[str, Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("case_id")), row.get("repeat"), row.get("fold")), []).append(row)
    figure, axes = pyplot.subplots(len(metrics), 1, figsize=(8.5, max(3.2, len(metrics) * 2.8)), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        seen_labels: set[str] = set()
        for (case_id, repeat, fold), values in sorted(groups.items(), key=str):
            points = [(_number(row.get(x_name)), _number(row.get(metric))) for row in values]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if not points:
                continue
            points.sort()
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                alpha=0.45,
                linewidth=1,
                label=case_id if case_id not in seen_labels else None,
            )
            seen_labels.add(case_id)
        axis.set(xlabel=x_name, ylabel=metric)
        axis.grid(alpha=0.25)
    axes.flat[0].set_title("Learning curves (individual repeat/fold traces)")
    figure.tight_layout()
    return figure


def _mean_history_figure(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[str],
    metrics: Sequence[str],
    pyplot: Any,
    *,
    title: str,
    unavailable: str,
    bounded: bool = False,
) -> Any:
    x_name = "epoch" if any("epoch" in row for row in rows) else "step"
    figure, axes = pyplot.subplots(len(metrics), 1, figsize=(9.0, max(3.4, len(metrics) * 3.0)), squeeze=False)
    drew_any = False
    for axis, metric in zip(axes.flat, metrics):
        for case_id in cases:
            by_step: dict[float, list[float]] = {}
            for row in rows:
                if str(row.get("case_id")) != case_id:
                    continue
                x_value = _number(row.get(x_name))
                y_value = _number(row.get(metric))
                if x_value is None or y_value is None:
                    continue
                by_step.setdefault(x_value, []).append(y_value)
            if not by_step:
                continue
            ordered = sorted(by_step)
            axis.plot(
                ordered,
                [float(np.mean(by_step[value])) for value in ordered],
                marker="o",
                markersize=2.5,
                linewidth=1.6,
                label=case_id,
            )
            drew_any = True
        axis.set(xlabel=x_name, ylabel=metric, ylim=(0.0, 1.0) if bounded else None)
        axis.grid(alpha=0.25)
    if not drew_any:
        pyplot.close(figure)
        raise ValueError(unavailable)
    axes.flat[0].set_title(title)
    axes.flat[0].legend(fontsize="small")
    figure.tight_layout()
    return figure


def _top_learning_curves(collected: CollectedStudy, analysis: StudyAnalysis, pyplot: Any) -> Any:
    cases = [str(row["case_id"]) for row in analysis.predictive_leaderboard[:6]]
    rows = [row for row in collected.history_rows if str(row.get("case_id")) in cases]
    if not rows:
        raise ValueError("top-ranked cases have no training history")
    metrics = _loss_history_metric_names(rows)
    if not metrics:
        raise ValueError("top-ranked training history has no declared train/inner-train loss")
    return _mean_history_figure(
        rows,
        cases,
        metrics,
        pyplot,
        title="Top-ranked learning curves (mean across repeat/fold traces)",
        unavailable="top-ranked learning curves contain no finite points",
    )


def _balanced_accuracy_learning_curves(
    collected: CollectedStudy,
    pyplot: Any,
    *,
    top_cases: Sequence[str] | None = None,
) -> Any:
    """Plot provenance-safe inner/train BA histories only.

    A generic ``balanced_accuracy`` or ``val_balanced_accuracy`` field is
    intentionally rejected: without an explicit inner/train name it could be
    derived from the outer held-out fold and would make a training-time curve
    invalid.
    """

    allowed = None if top_cases is None else set(top_cases)
    rows = [row for row in collected.history_rows if allowed is None or str(row.get("case_id")) in allowed]
    if not rows:
        raise ValueError("training history unavailable for requested cases")
    rejected_rows = [
        row for row in rows
        if _explicitly_uses_outer_heldout(row) and _has_explicit_inner_or_train_balanced_accuracy(row)
    ]
    if rejected_rows:
        raise ValueError("at least one inner/train balanced-accuracy history row explicitly "
                         "marks outer held-out data as used; the complete learning-curve "
                         "figure is rejected")
    metrics = _balanced_accuracy_history_metric_names(rows)
    if not metrics:
        raise ValueError("no provenance-safe inner/train balanced-accuracy history; "
                         "outer held-out metrics are never converted into a learning curve")
    cases = sorted({str(row.get("case_id")) for row in rows})
    qualifier = "top-ranked " if top_cases is not None else ""
    return _mean_history_figure(
        rows,
        cases,
        metrics,
        pyplot,
        title=f"Provenance-safe {qualifier}balanced-accuracy learning curves (mean across repeat/fold traces)",
        unavailable="balanced-accuracy history contains no finite points",
        bounded=True,
    )


def _axis_case_values(collected: CollectedStudy, ) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    axes = [
        str(axis.get("path")) for axis in collected.plan.get("axes", ())
        if isinstance(axis, Mapping) and axis.get("path")
    ]
    cases = {
        str(case["case_id"]): (case.get("changed_values") if isinstance(case.get("changed_values"), Mapping) else {})
        for case in collected.manifest.get("cases", ()) if isinstance(case, Mapping) and case.get("case_id") is not None
    }
    return axes, cases


def _parameter_effects(collected: CollectedStudy, analysis: StudyAnalysis, pyplot: Any) -> Any:
    axes, cases = _axis_case_values(collected)
    if not axes:
        raise ValueError("no declared varied axis")
    repeat_lookup: dict[str, list[Mapping[str, Any]]] = {}
    for row in analysis.repeat_metrics:
        repeat_lookup.setdefault(str(row["case_id"]), []).append(row)
    figure, subplot_axes = pyplot.subplots(
        len(axes),
        1,
        figsize=(max(7.5,
                     len(cases) * 0.85), max(3.5,
                                             len(axes) * 3.2)),
        squeeze=False,
    )
    drew_any = False
    for subplot, path in zip(subplot_axes.flat, axes):
        ordered_values: list[Any] = []
        for changed in cases.values():
            if path in changed and repr(changed[path]) not in {repr(value) for value in ordered_values}:
                ordered_values.append(changed[path])
        for index, value in enumerate(ordered_values):
            ba: list[float] = []
            macro_f1: list[float] = []
            for case_id, changed in cases.items():
                if repr(changed.get(path)) != repr(value):
                    continue
                for row in repeat_lookup.get(case_id, ()):
                    if (score := _number(row.get("balanced_accuracy"))) is not None:
                        ba.append(score)
                    if (score := _number(row.get("macro_f1"))) is not None:
                        macro_f1.append(score)
            if ba:
                subplot.scatter(
                    [index - 0.08] * len(ba),
                    ba,
                    alpha=0.55,
                    marker="o",
                    color="#3b6fb6",
                )
                subplot.plot(index - 0.08, float(np.mean(ba)), marker="D", color="#173b69")
                drew_any = True
            if macro_f1:
                subplot.scatter(
                    [index + 0.08] * len(macro_f1),
                    macro_f1,
                    alpha=0.55,
                    marker="s",
                    color="#d27d2d",
                )
                subplot.plot(
                    index + 0.08,
                    float(np.mean(macro_f1)),
                    marker="D",
                    color="#7c3f0d",
                )
                drew_any = True
        subplot.set_xticks(range(len(ordered_values)), [str(value) for value in ordered_values])
        subplot.set(ylim=(0.0, 1.0), ylabel="Repeat score", title=f"Descriptive parameter view: {path}")
        subplot.grid(axis="y", alpha=0.25)
    if not drew_any:
        pyplot.close(figure)
        raise ValueError("declared axes have no repeat-level predictive metrics")
    figure.text(
        0.5,
        0.005,
        "Circles: BA; squares: macro-F1; diamonds: group means. "
        "Descriptive only, not a causal effect.",
        ha="center",
        fontsize="small",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def _parameter_interaction(collected: CollectedStudy, analysis: StudyAnalysis, pyplot: Any) -> Any:
    axes, cases = _axis_case_values(collected)
    if len(axes) < 2:
        raise ValueError("requires at least two declared varied axes")
    first, second = axes[:2]
    summary = {
        str(row["case_id"]): _number(row.get("participant_mean_balanced_accuracy"))
        for row in analysis.case_summary
    }
    first_values: list[Any] = []
    second_values: list[Any] = []
    groups: dict[tuple[str, str], list[float]] = {}
    for case_id, changed in cases.items():
        if first not in changed or second not in changed:
            continue
        score = summary.get(case_id)
        if score is None:
            continue
        first_key, second_key = repr(changed[first]), repr(changed[second])
        if first_key not in [repr(value) for value in first_values]:
            first_values.append(changed[first])
        if second_key not in [repr(value) for value in second_values]:
            second_values.append(changed[second])
        groups.setdefault((first_key, second_key), []).append(score)
    if not groups:
        raise ValueError("two-axis cases have no finite case-level BA")
    matrix = np.full((len(first_values), len(second_values)), np.nan, dtype=np.float64)
    for row_index, first_value in enumerate(first_values):
        for column_index, second_value in enumerate(second_values):
            values = groups.get((repr(first_value), repr(second_value)), [])
            if values:
                matrix[row_index, column_index] = float(np.mean(values))
    figure, axis = pyplot.subplots(figsize=(max(6.5, len(second_values) * 1.0), max(4.5, len(first_values) * 0.8)))
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(second_values)), [str(value) for value in second_values])
    axis.set_yticks(range(len(first_values)), [str(value) for value in first_values])
    suffix = "; averaged over remaining axes" if len(axes) > 2 else ""
    axis.set(xlabel=second, ylabel=first, title=f"Descriptive BA interaction view{suffix}; no causal claim")
    _annotate_matrix(axis, matrix, ".3f", lambda value: "white" if value < 0.55 else "black")
    figure.colorbar(image, ax=axis, label="Mean participant balanced accuracy")
    figure.tight_layout()
    return figure


def _bridge_panels(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    pyplot: Any,
    *,
    field_prefix: str,
    title: str,
    xlabel: str,
) -> Any:
    positions = np.arange(len(rows), dtype=np.float64)
    views = (
        ("legacy_aggregation", "Window-balanced legacy"),
        ("line_a_aggregation", "File-balanced Line A"),
        ("v2_aggregation", "Role-balanced Line B"),
    )
    drew_any, delta = False, bool(field_prefix)
    width_factor = 1.45 if delta else 1.35
    figure, axes = pyplot.subplots(2, 1, figsize=(max(10.0, len(rows) * width_factor), 7.4), sharex=True)
    ylabels = ("Delta balanced accuracy", "Delta Macro-F1") if delta else ("Balanced accuracy", "Macro-F1")
    for axis, metric, label in zip(axes, ("BA", "macroF1"), ylabels):
        for view_index, (suffix, view_label) in enumerate(views):
            values = [_number(row.get(f"{field_prefix}{metric}_{suffix}")) for row in rows]
            heights = [np.nan if value is None else value for value in values]
            if any(value is not None for value in values):
                axis.bar(positions + (view_index - 1) * 0.24, heights, width=0.24, label=view_label)
                drew_any = True
        if delta:
            axis.axhline(0.0, color="black", linewidth=0.8)
        else:
            axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    if not drew_any:
        pyplot.close(figure)
        kind = "numeric bridge ablation deltas" if delta else "execution-order bridge absolute metrics"
        raise ValueError(f"{kind} are unavailable")
    _set_centered_category_ticks(axes[-1], positions, labels)
    figure.suptitle(title)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), fontsize="small", ncol=3)
    axes[-1].set_xlabel(xlabel)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return figure


def _legacy_bridge_numeric_ablation_report(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row for row in analysis.legacy_bridge_numeric_ablation_report
        if row.get("comparison_role") == "predefined_adjacent_numeric_ablation"
    ]
    if len(rows) != 7:
        raise ValueError("numeric bridge report requires exactly seven adjacent ablation rows")
    return _bridge_panels(
        rows,
        [str(row.get("numeric_comparison")).replace("->", "→") for row in rows],
        pyplot,
        field_prefix="delta_",
        title="Bridge report A — seven predefined adjacent ablations (L0→L7)",
        xlabel="Predefined numeric-profile comparison",
    )


def _legacy_bridge_execution_order_report(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.legacy_bridge_execution_order_report)
    profiles = [str(row.get("profile")) for row in rows]
    if profiles != ["L7", "L5", "L6", "L4", "L3", "L2", "L1", "L0"]:
        raise ValueError("execution bridge report requires L7,L5,L6,L4,L3,L2,L1,L0")
    if any(bool(row.get("execution_transition_is_ablation")) for row in rows):
        raise ValueError("execution-order transitions must not be marked as ablations")
    return _bridge_panels(
        rows,
        profiles,
        pyplot,
        field_prefix="",
        title="Bridge report B — CompactCNN execution order (absolute metrics; transitions are not causal ablations)",
        xlabel="Execution profile (L7→L5 jump is scheduling only)",
    )


def _stage3_star_model_deltas(
    analysis: StudyAnalysis,
    pyplot: Any,
) -> Any:
    rows = list(analysis.stage3_star_contrasts)
    models = sorted({str(row.get("model")) for row in rows})
    if len(rows) != 14 or len(models) != 2:
        raise ValueError("centered-star delta figure requires 14 rows and two models")
    figure, axes = pyplot.subplots(
        len(models),
        1,
        figsize=(12.0, 4.2 * len(models)),
        squeeze=False,
    )
    drew_any = False
    metrics = (
        ("delta_native_balanced_accuracy", "Δ BA", "o"),
        ("delta_native_macro_f1", "Δ Macro-F1", "s"),
        ("delta_native_worst_class_f1", "Δ worst-class F1", "^"),
    )
    for axis, model in zip(axes.flat, models):
        axis_drew = False
        current = [row for row in rows if str(row.get("model")) == model]
        if len(current) != 7:
            pyplot.close(figure)
            raise ValueError("each centered-star model requires seven contrasts")
        labels = [f"{row.get('variant_profile')}\n{row.get('factor_id')}" for row in current]
        positions = np.arange(len(current), dtype=np.float64)
        for field, label, marker in metrics:
            values = [_number(row.get(field)) for row in current]
            if any(value is not None for value in values):
                axis.plot(
                    positions,
                    [np.nan if value is None else value for value in values],
                    marker=marker,
                    linewidth=1.5,
                    label=label,
                )
                drew_any = True
                axis_drew = True
        axis.axhline(0.0, color="black", linewidth=0.9)
        _set_centered_category_ticks(axis, positions, labels)
        axis.set(ylabel="Variant − same-model B0", title=model)
        axis.grid(axis="y", alpha=0.25)
        if axis_drew:
            axis.legend(fontsize="small")
    if not drew_any:
        pyplot.close(figure)
        raise ValueError("no available centered-star deltas")
    figure.suptitle("Stage 3 centered star — native participant-OOF effects " "(B0 reused; contrasts correlated)")
    figure.tight_layout()
    return figure


def _stage3_star_fold_delta_heatmap(
    analysis: StudyAnalysis,
    pyplot: Any,
) -> Any:
    rows = list(analysis.stage3_star_fold_contrasts)
    contrast_keys: list[tuple[str, str, str]] = []
    cell_keys = sorted({(int(row["repeat"]), int(row["fold"])) for row in rows})
    for row in rows:
        key = (
            str(row.get("model")),
            str(row.get("variant_profile")),
            str(row.get("factor_id")),
        )
        if key not in contrast_keys:
            contrast_keys.append(key)
    if len(contrast_keys) != 14 or not cell_keys or len(rows) != len(contrast_keys) * len(cell_keys):
        raise ValueError("centered-star fold rows do not form a complete 14×N " "repeat/fold matrix")
    matrix = np.full((len(contrast_keys), len(cell_keys)), np.nan, dtype=np.float64)
    row_index = {key: index for index, key in enumerate(contrast_keys)}
    cell_index = {key: index for index, key in enumerate(cell_keys)}
    for row in rows:
        value = _number(row.get("delta_native_balanced_accuracy"))
        if value is None:
            continue
        key = (
            str(row.get("model")),
            str(row.get("variant_profile")),
            str(row.get("factor_id")),
        )
        matrix[row_index[key], cell_index[(int(row["repeat"]), int(row["fold"]))]] = value
    if not np.isfinite(matrix).any():
        raise ValueError("no available centered-star matched-fold deltas")
    maximum = max(float(np.nanmax(np.abs(matrix))), 1e-9)
    figure, axis = pyplot.subplots(figsize=(max(8.5, 4.0 + 0.48 * len(cell_keys)), 8.5))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )
    axis.set_yticks(
        range(len(contrast_keys)),
        [f"{model} · {profile}\n{factor}" for model, profile, factor in contrast_keys],
        fontsize="x-small",
    )
    axis.set_xticks(
        range(len(cell_keys)),
        [f"r{repeat}/f{fold}" for repeat, fold in cell_keys],
    )
    axis.set_title("Stage 3 centered star — matched-fold native Δ BA\n" "descriptive only; no CI or significance test")
    figure.colorbar(image, ax=axis, label="Variant − same-model B0 BA")
    figure.tight_layout()
    return figure


def generate_static_figures(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    directory: str | Path,
    *,
    modules: Sequence[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Generate selected modular PNG figures, or one N/A marker per missing view."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    requested = None if not modules or "all" in modules else set(modules)
    if _is_stage3_centered_star(collected.plan):
        profile_figures = STAGE3_STAR_FIGURE_NAMES
    elif isinstance(collected.plan.get("legacy_bridge"), Mapping):
        profile_figures = LEGACY_BRIDGE_FIGURE_NAMES
    else:
        profile_figures = ()
    applicable = tuple(
        name for name in STATIC_FIGURE_NAMES
        if name not in (*LEGACY_BRIDGE_FIGURE_NAMES, *STAGE3_STAR_FIGURE_NAMES) or name in profile_figures
        if requested is None or name in requested)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except Exception as error:  # noqa: BLE001
        reason = f"matplotlib unavailable: {type(error).__name__}: {error}"
        return tuple(_na(target, name, reason) for name in applicable)
    analysis_renderers = {
        "classification_prediction_tsne": _classification_prediction_tsne,
        "classification_roc_auc_curves": _classification_roc_auc_curves,
        "leaderboard": _leaderboard,
        "stability": _stability,
        "macro_f1_stability": _macro_f1_stability,
        "roc_pr_auc_stability": _roc_pr_auc_stability,
        "per_class_metric_stability": _per_class_metric_stability,
        "worst_class_f1_stability": _worst_class_f1_stability,
        "fold_heatmap": _fold_heatmap,
        "paired_deltas": _paired_deltas,
        "ablation_sensitivity_metrics": _ablation_sensitivity_metrics,
        "coverage": _coverage,
        "route_role_coverage": _route_role_coverage,
        "denoiser_hr_comparison": _denoiser_hr_comparison,
        "quality_distributions": _quality_distributions,
        "calibration": _calibration,
        "confusion_matrices": _confusion,
        "confusion_matrices_row_normalized": _normalized_confusion,
        "per_class": _per_class,
        "aggregation_view_metrics": _aggregation_view_metrics,
        "aggregation_hierarchy_coverage": _aggregation_hierarchy_coverage,
        "aggregation_view_confusion_matrices": _aggregation_view_confusions,
        "aggregation_view_per_class": _aggregation_view_per_class,
        "legacy_bridge_numeric_ablation_report": _legacy_bridge_numeric_ablation_report,
        "legacy_bridge_execution_order_report": _legacy_bridge_execution_order_report,
        "stage3_star_model_deltas": _stage3_star_model_deltas,
        "stage3_star_fold_delta_heatmap": _stage3_star_fold_delta_heatmap,
    }
    plots: dict[str, Callable[[Any], Any]] = {
        name: partial(renderer, analysis) for name, renderer in analysis_renderers.items()
    }
    plots.update({
        "classification_prediction_scores": partial(_classification_prediction_scores, collected, analysis),
        "aggregation_view_confusion_matrices_row_normalized": partial(
            _aggregation_view_confusions, analysis, row_normalized=True),
        "learning_curves": partial(_learning_curves, collected),
        "top_learning_curves": partial(_top_learning_curves, collected, analysis),
        "balanced_accuracy_learning_curves": partial(_balanced_accuracy_learning_curves, collected),
        "top_balanced_accuracy_learning_curves": lambda p: _balanced_accuracy_learning_curves(
            collected, p, top_cases=[str(row["case_id"]) for row in analysis.predictive_leaderboard[:6]]),
        "parameter_effects": partial(_parameter_effects, collected, analysis),
        "parameter_interaction": partial(_parameter_interaction, collected, analysis),
    })
    na_png_names = {
        "balanced_accuracy_learning_curves",
        "top_balanced_accuracy_learning_curves",
        *LEGACY_BRIDGE_FIGURE_NAMES,
        *STAGE3_STAR_FIGURE_NAMES,
    }
    return tuple(_save(target, name, plots[name], pyplot, render_na_png=name in na_png_names) for name in applicable)


__all__ = [
    "FIGURE_TABLE_SOURCES",
    "LEGACY_BRIDGE_FIGURE_NAMES",
    "STAGE3_STAR_FIGURE_NAMES",
    "STATIC_FIGURE_NAMES",
    "clear_static_figure_artifacts",
    "generate_static_figures",
]
