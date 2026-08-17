"""Static report figures with explicit N/A artifacts when inputs are absent."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .analyze import StudyAnalysis
from .collect import CollectedStudy


STATIC_FIGURE_NAMES = (
    "leaderboard",
    "stability",
    "worst_class_f1_stability",
    "fold_heatmap",
    "paired_deltas",
    "coverage",
    "route_role_coverage",
    "quality_distributions",
    "calibration",
    "confusion_matrices",
    "confusion_matrices_row_normalized",
    "per_class",
    "learning_curves",
    "top_learning_curves",
    "parameter_effects",
    "parameter_interaction",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unlink_if_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()


def clear_static_figure_artifacts(directory: str | Path) -> None:
    """Remove only reporter-owned PNG/N/A counterparts for deterministic reruns."""

    target = Path(directory)
    for name in STATIC_FIGURE_NAMES:
        _unlink_if_file(target / f"{name}.png")
        _unlink_if_file(target / f"{name}.NA.txt")


def _na(directory: Path, name: str, reason: str) -> dict[str, Any]:
    target = directory / f"{name}.NA.txt"
    temporary = directory / f".{name}.NA.tmp-{os.getpid()}-{time.time_ns()}.txt"
    try:
        temporary.write_text(f"N/A: {reason.strip()}\n", encoding="utf-8")
        os.replace(temporary, target)
        _unlink_if_file(directory / f"{name}.png")
    finally:
        _unlink_if_file(temporary)
    return {
        "figure": name,
        "status": "N/A",
        "path": str(target.relative_to(directory.parent)),
        "reason": reason.strip(),
    }


def _save(
    directory: Path,
    name: str,
    draw: Callable[[Any], Any],
    pyplot: Any,
) -> dict[str, Any]:
    figure = None
    temporary = directory / f".{name}.tmp-{os.getpid()}-{time.time_ns()}.png"
    try:
        figure = draw(pyplot)
        target = directory / f"{name}.png"
        figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        os.replace(temporary, target)
        _unlink_if_file(directory / f"{name}.NA.txt")
        return {
            "figure": name,
            "status": "generated",
            "path": str(target.relative_to(directory.parent)),
            "reason": "",
        }
    except Exception as error:  # noqa: BLE001 - report remains usable.
        return _na(directory, name, f"{type(error).__name__}: {error}")
    finally:
        _unlink_if_file(temporary)
        if figure is not None:
            pyplot.close(figure)


def _leaderboard(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row
        for row in analysis.predictive_leaderboard
        if _number(row.get("participant_mean_balanced_accuracy")) is not None
    ]
    if not rows:
        raise ValueError("no finite case-level predictive metrics")
    labels = [str(row["case_id"]) for row in reversed(rows)]
    ba = [
        float(row["participant_mean_balanced_accuracy"]) for row in reversed(rows)
    ]
    f1 = [
        _number(row.get("participant_mean_macro_f1")) or 0.0
        for row in reversed(rows)
    ]
    positions = np.arange(len(labels), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(9, max(3.5, len(labels) * 0.55)))
    axis.barh(positions - 0.18, ba, height=0.34, label="Balanced accuracy")
    axis.barh(positions + 0.18, f1, height=0.34, label="Macro-F1")
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Participant-level score")
    axis.set_title("Predictive leaderboard (manual review; no automatic winner)")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    return figure


def _stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    cases: dict[str, list[float]] = {}
    for row in analysis.repeat_metrics:
        value = _number(row.get("balanced_accuracy"))
        if value is not None:
            cases.setdefault(str(row["case_id"]), []).append(value)
    if not cases:
        raise ValueError("repeat-level balanced accuracy unavailable")
    labels = sorted(cases)
    figure, axis = pyplot.subplots(figsize=(max(7, len(labels) * 1.15), 4.8))
    axis.boxplot([cases[label] for label in labels], tick_labels=labels, showmeans=True)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Repeat stability")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _worst_class_f1_stability(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.worst_class_f1_stability)
    if not rows:
        raise ValueError("worst-class F1 stability ranking unavailable")
    rows.reverse()
    labels = [
        f"S{row['worst_class_f1_stability_rank']} · {row['case_id']}"
        for row in rows
    ]
    worst_f1 = [float(row["worst_class_f1"]) for row in rows]
    mean_ba = [float(row["participant_mean_balanced_accuracy"]) for row in rows]
    ba_sd = [
        _number(row.get("repeat_balanced_accuracy_population_sd")) or 0.0
        for row in rows
    ]
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axis = pyplot.subplots(figsize=(9, max(3.8, len(rows) * 0.58)))
    axis.barh(
        positions - 0.18,
        worst_f1,
        height=0.34,
        label="Worst-class F1",
    )
    axis.barh(
        positions + 0.18,
        mean_ba,
        height=0.34,
        xerr=ba_sd,
        capsize=3,
        label="Mean BA ± repeat population SD",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Participant-level score")
    axis.set_title("Worst-class F1 stability review (top 10)")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return figure


def _fold_heatmap(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row
        for row in analysis.fold_metrics
        if _number(row.get("balanced_accuracy")) is not None
    ]
    if not rows:
        raise ValueError("fold/cell balanced accuracy unavailable")
    cases = sorted({str(row["case_id"]) for row in rows})
    cells = sorted({(int(row["repeat"]), int(row["fold"])) for row in rows})
    matrix = np.full((len(cases), len(cells)), np.nan, dtype=np.float64)
    case_index = {value: index for index, value in enumerate(cases)}
    cell_index = {value: index for index, value in enumerate(cells)}
    for row in rows:
        matrix[
            case_index[str(row["case_id"])],
            cell_index[(int(row["repeat"]), int(row["fold"]))],
        ] = float(row["balanced_accuracy"])
    figure, axis = pyplot.subplots(
        figsize=(max(9, len(cells) * 0.42), max(3.5, len(cases) * 0.55))
    )
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
    figure, axis = pyplot.subplots(figsize=(max(7, len(labels) * 1.15), 4.8))
    axis.boxplot([groups[label] for label in labels], tick_labels=labels, showmeans=True)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Δ balanced accuracy vs reference")
    axis.set_title("Paired repeat deltas")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _coverage(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row
        for row in analysis.coverage
        if _number(row.get("mean_coverage_rate")) is not None
    ]
    if not rows:
        raise ValueError("coverage metrics unavailable")
    labels = [str(row["case_id"]) for row in rows]
    values = [float(row["mean_coverage_rate"]) for row in rows]
    figure, axis = pyplot.subplots(figsize=(max(7, len(labels) * 1.1), 4.5))
    axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean coverage rate")
    axis.set_title("Coverage by case")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _route_role_coverage(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row
        for row in analysis.route_role_coverage
        if _number(row.get("retained_coverage")) is not None
    ]
    if not rows:
        raise ValueError("route/role coverage diagnostics unavailable")
    labels = [
        f"{row['case_id']}\n{row['role']} · {row['route_state']}"
        for row in rows
    ]
    retained = [float(row["retained_coverage"]) for row in rows]
    available = [
        (
            1.0 - float(row["unavailable_predictor_rate"])
            if _number(row.get("unavailable_predictor_rate")) is not None
            else np.nan
        )
        for row in rows
    ]
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axis = pyplot.subplots(
        figsize=(max(8, len(rows) * 0.75), 5.2)
    )
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
    axis.set_xticks(positions, labels, rotation=55, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction")
    axis.set_title("Route × role coverage and feature availability")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _quality_distributions(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = [
        row
        for row in analysis.quality_distributions
        if _number(row.get("mean")) is not None
    ]
    if not rows:
        raise ValueError("route/role quality-component distributions unavailable")
    labels = [
        f"{row['case_id']} / {row['role']} / {row['route_state']}\n{row['component']}"
        for row in rows
    ]
    means = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
    deviations = np.asarray(
        [
            _number(row.get("population_sd")) or 0.0
            for row in rows
        ],
        dtype=np.float64,
    )
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axis = pyplot.subplots(
        figsize=(max(9, len(rows) * 0.75), 5.4)
    )
    axis.errorbar(
        positions,
        means,
        yerr=deviations,
        fmt="o",
        capsize=3,
    )
    axis.set_xticks(positions, labels, rotation=60, ha="right")
    axis.set_ylabel("Component value (mean ± population SD)")
    axis.set_title("Quality distributions by route and role")
    axis.grid(axis="y", alpha=0.25)
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
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Observed accuracy")
    axis.set_title("Participant-level calibration")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.25)
    return figure


def _confusion(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.confusion_matrices)
    if not rows:
        raise ValueError("participant OOF or labeled cell confusion matrices unavailable")
    columns = min(3, len(rows))
    rows_n = int(math.ceil(len(rows) / columns))
    figure, axes = pyplot.subplots(
        rows_n,
        columns,
        figsize=(columns * 4.0, rows_n * 3.8),
        squeeze=False,
    )
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray(row["confusion_matrix"], dtype=np.float64)
        image = axis.imshow(matrix, cmap="Blues")
        order = [str(value) for value in row["class_order"]]
        axis.set_xticks(range(len(order)), order)
        axis.set_yticks(range(len(order)), order)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(str(row["case_id"]))
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                axis.text(x, y, f"{matrix[y, x]:.0f}", ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat[len(rows) :]:
        axis.axis("off")
    figure.suptitle("Pooled confusion matrices (participant OOF or labeled cell fallback)")
    figure.tight_layout()
    return figure


def _normalized_confusion(analysis: StudyAnalysis, pyplot: Any) -> Any:
    ranks = {
        str(row["case_id"]): int(row["predictive_rank"])
        for row in analysis.predictive_leaderboard
    }
    rows = [
        row
        for row in analysis.confusion_matrices
        if str(row.get("case_id")) in ranks
    ]
    rows.sort(key=lambda row: ranks[str(row["case_id"])])
    rows = rows[:6]
    if not rows:
        raise ValueError("top-case pooled confusion matrices unavailable")
    columns = min(3, len(rows))
    rows_n = int(math.ceil(len(rows) / columns))
    figure, axes = pyplot.subplots(
        rows_n,
        columns,
        figsize=(columns * 4.1, rows_n * 3.9),
        squeeze=False,
    )
    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray(row["confusion_matrix"], dtype=np.float64)
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix),
            where=totals > 0.0,
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        order = [str(value) for value in row["class_order"]]
        axis.set_xticks(range(len(order)), order)
        axis.set_yticks(range(len(order)), order)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        case_id = str(row["case_id"])
        axis.set_title(f"#{ranks[case_id]} · {case_id}")
        for y in range(normalized.shape[0]):
            for x in range(normalized.shape[1]):
                value = normalized[y, x]
                axis.text(
                    x,
                    y,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat[len(rows) :]:
        axis.axis("off")
    figure.suptitle("Top-case row-normalized pooled confusion matrices")
    figure.tight_layout()
    return figure


def _per_class(analysis: StudyAnalysis, pyplot: Any) -> Any:
    rows = list(analysis.per_class_metrics)
    if not rows:
        raise ValueError("participant OOF or labeled cell per-class metrics unavailable")
    labels = [
        f"{row['case_id']} / {row['class_label']}"
        for row in rows
    ]
    recall = [float(row["recall"]) for row in rows]
    f1 = [float(row["f1"]) for row in rows]
    positions = np.arange(len(rows))
    figure, axis = pyplot.subplots(figsize=(max(8, len(rows) * 0.55), 4.8))
    axis.bar(positions - 0.18, recall, width=0.36, label="Recall")
    axis.bar(positions + 0.18, f1, width=0.36, label="F1")
    axis.set_xticks(positions, labels, rotation=60, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Per-class pooled metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return figure


def _history_metric_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    excluded = {
        "case_id",
        "repeat",
        "fold",
        "epoch",
        "step",
        "member",
        "training_seed",
        "epoch_rng_seed",
    }
    preferred = (
        "training_loss",
        "inner_training_loss",
        "inner_participant_balanced_accuracy",
        "loss",
        "train_loss",
        "val_loss",
        "validation_loss",
        "balanced_accuracy",
        "val_balanced_accuracy",
        "macro_f1",
        "val_macro_f1",
    )
    present = []
    for name in preferred:
        if any(_number(row.get(name)) is not None for row in rows):
            present.append(name)
    if present:
        return tuple(present[:4])
    candidates = []
    for key in dict.fromkeys(key for row in rows for key in row):
        if key in excluded:
            continue
        if any(_number(row.get(key)) is not None for row in rows):
            candidates.append(key)
    return tuple(candidates[:4])


def _learning_curves(collected: CollectedStudy, pyplot: Any) -> Any:
    rows = list(collected.history_rows)
    if not rows:
        raise ValueError("training history unavailable")
    x_name = "epoch" if any("epoch" in row for row in rows) else "step"
    metrics = _history_metric_names(rows)
    if not metrics:
        raise ValueError("training history has no numeric learning metric")
    groups: dict[tuple[str, Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row.get("case_id")), row.get("repeat"), row.get("fold")), []
        ).append(row)
    figure, axes = pyplot.subplots(
        len(metrics), 1, figsize=(8.5, max(3.2, len(metrics) * 2.8)), squeeze=False
    )
    for axis, metric in zip(axes.flat, metrics):
        drew = False
        for (case_id, repeat, fold), values in sorted(groups.items(), key=str):
            points = [
                (_number(row.get(x_name)), _number(row.get(metric)))
                for row in values
            ]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if not points:
                continue
            points.sort()
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                alpha=0.45,
                linewidth=1,
                label=case_id if not drew else None,
            )
            drew = True
        axis.set_xlabel(x_name)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    axes.flat[0].set_title("Learning curves (individual repeat/fold traces)")
    figure.tight_layout()
    return figure


def _top_learning_curves(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    pyplot: Any,
) -> Any:
    top_cases = [
        str(row["case_id"])
        for row in analysis.predictive_leaderboard[:6]
    ]
    rows = [
        row
        for row in collected.history_rows
        if str(row.get("case_id")) in top_cases
    ]
    if not rows:
        raise ValueError("top-ranked cases have no training history")
    x_name = "epoch" if any("epoch" in row for row in rows) else "step"
    metrics = _history_metric_names(rows)
    if not metrics:
        raise ValueError("top-ranked training history has no numeric learning metric")
    figure, axes = pyplot.subplots(
        len(metrics),
        1,
        figsize=(8.8, max(3.2, len(metrics) * 2.9)),
        squeeze=False,
    )
    drew_any = False
    for axis, metric in zip(axes.flat, metrics):
        for case_id in top_cases:
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
        axis.set_xlabel(x_name)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    if not drew_any:
        pyplot.close(figure)
        raise ValueError("top-ranked learning curves contain no finite points")
    axes.flat[0].set_title("Top-ranked learning curves (mean across repeat/fold traces)")
    axes.flat[0].legend(fontsize="small")
    figure.tight_layout()
    return figure


def _axis_case_values(
    collected: CollectedStudy,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    axes = [
        str(axis.get("path"))
        for axis in collected.plan.get("axes", ())
        if isinstance(axis, Mapping) and axis.get("path")
    ]
    cases = {
        str(case["case_id"]): (
            case.get("changed_values")
            if isinstance(case.get("changed_values"), Mapping)
            else {}
        )
        for case in collected.manifest.get("cases", ())
        if isinstance(case, Mapping) and case.get("case_id") is not None
    }
    return axes, cases


def _parameter_effects(
    collected: CollectedStudy, analysis: StudyAnalysis, pyplot: Any
) -> Any:
    axes, cases = _axis_case_values(collected)
    if not axes:
        raise ValueError("no declared varied axis")
    repeat_lookup: dict[str, list[Mapping[str, Any]]] = {}
    for row in analysis.repeat_metrics:
        repeat_lookup.setdefault(str(row["case_id"]), []).append(row)
    figure, subplot_axes = pyplot.subplots(
        len(axes),
        1,
        figsize=(max(7.5, len(cases) * 0.85), max(3.5, len(axes) * 3.2)),
        squeeze=False,
    )
    drew_any = False
    for subplot, path in zip(subplot_axes.flat, axes):
        ordered_values: list[Any] = []
        for changed in cases.values():
            if path in changed and repr(changed[path]) not in {
                repr(value) for value in ordered_values
            }:
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
        subplot.set_xticks(
            range(len(ordered_values)), [str(value) for value in ordered_values]
        )
        subplot.set_ylim(0.0, 1.0)
        subplot.set_ylabel("Repeat score")
        subplot.set_title(f"Descriptive parameter view: {path}")
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


def _parameter_interaction(
    collected: CollectedStudy, analysis: StudyAnalysis, pyplot: Any
) -> Any:
    axes, cases = _axis_case_values(collected)
    if len(axes) < 2:
        raise ValueError("requires at least two declared varied axes")
    first, second = axes[:2]
    summary = {
        str(row["case_id"]): _number(
            row.get("participant_mean_balanced_accuracy")
        )
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
    matrix = np.full(
        (len(first_values), len(second_values)), np.nan, dtype=np.float64
    )
    for row_index, first_value in enumerate(first_values):
        for column_index, second_value in enumerate(second_values):
            values = groups.get((repr(first_value), repr(second_value)), [])
            if values:
                matrix[row_index, column_index] = float(np.mean(values))
    figure, axis = pyplot.subplots(
        figsize=(max(6.5, len(second_values) * 1.0), max(4.5, len(first_values) * 0.8))
    )
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(second_values)), [str(value) for value in second_values])
    axis.set_yticks(range(len(first_values)), [str(value) for value in first_values])
    axis.set_xlabel(second)
    axis.set_ylabel(first)
    suffix = (
        "; averaged over remaining axes"
        if len(axes) > 2
        else ""
    )
    axis.set_title(f"Descriptive BA interaction view{suffix}; no causal claim")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            if np.isfinite(matrix[row_index, column_index]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] < 0.55 else "black",
                )
    figure.colorbar(image, ax=axis, label="Mean participant balanced accuracy")
    figure.tight_layout()
    return figure


def generate_static_figures(
    collected: CollectedStudy,
    analysis: StudyAnalysis,
    directory: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Generate deterministic PNG figures, or one N/A marker per missing view."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except Exception as error:  # noqa: BLE001
        reason = f"matplotlib unavailable: {type(error).__name__}: {error}"
        return tuple(
            _na(target, name, reason)
            for name in STATIC_FIGURE_NAMES
        )
    plots: tuple[tuple[str, Callable[[Any], Any]], ...] = (
        ("leaderboard", lambda plot: _leaderboard(analysis, plot)),
        ("stability", lambda plot: _stability(analysis, plot)),
        (
            "worst_class_f1_stability",
            lambda plot: _worst_class_f1_stability(analysis, plot),
        ),
        ("fold_heatmap", lambda plot: _fold_heatmap(analysis, plot)),
        ("paired_deltas", lambda plot: _paired_deltas(analysis, plot)),
        ("coverage", lambda plot: _coverage(analysis, plot)),
        (
            "route_role_coverage",
            lambda plot: _route_role_coverage(analysis, plot),
        ),
        (
            "quality_distributions",
            lambda plot: _quality_distributions(analysis, plot),
        ),
        ("calibration", lambda plot: _calibration(analysis, plot)),
        ("confusion_matrices", lambda plot: _confusion(analysis, plot)),
        (
            "confusion_matrices_row_normalized",
            lambda plot: _normalized_confusion(analysis, plot),
        ),
        ("per_class", lambda plot: _per_class(analysis, plot)),
        ("learning_curves", lambda plot: _learning_curves(collected, plot)),
        (
            "top_learning_curves",
            lambda plot: _top_learning_curves(collected, analysis, plot),
        ),
        (
            "parameter_effects",
            lambda plot: _parameter_effects(collected, analysis, plot),
        ),
        (
            "parameter_interaction",
            lambda plot: _parameter_interaction(collected, analysis, plot),
        ),
    )
    return tuple(_save(target, name, draw, pyplot) for name, draw in plots)


__all__ = [
    "STATIC_FIGURE_NAMES",
    "clear_static_figure_artifacts",
    "generate_static_figures",
]
