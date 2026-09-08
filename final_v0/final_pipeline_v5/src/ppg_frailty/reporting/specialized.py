"""Reports for Stage-5, peak-detector and hyperparameter studies.

The training modules persist complete numerical data.  This module turns those
artifacts into tables and figures, so report layout can evolve without touching
the scientific workflow.
"""

from __future__ import annotations

import csv
from functools import partial
import json
import math
import warnings
from pathlib import Path
import shutil
from statistics import median
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .classification_diagnostics import (
    classification_diagnostic_status_rows,
    classification_per_class_metric_rows,
    classification_roc_curve_rows,
    classification_tsne_rows,
    normalize_classification_rows,
)
from .analyze import StudyAnalysis
from .plots import FIGURE_TABLE_SOURCES, STATIC_FIGURE_NAMES
from .tabular import ReportTable, write_excel_workbook

# model, evaluation, target, evaluation scope, manifest stage, data file, figure prefix
MOTION_DATASETS = (
    ("frailty29_trained_motion_detector", "frailty29_outer_oof", "frailty29", "source_grouped_oof",
     "internal_motion_oof", "motion_window_oof.parquet", "motion_internal"),
    ("frailty29_trained_motion_detector", "frailty29_trained_to_ptt22", "ptt22", "frozen_cross_dataset",
     "ptt_motion_external", "motion_ptt_window_predictions.parquet", "motion_ptt"),
    ("ptt22_trained_motion_detector", "ptt22_outer_oof", "ptt22", "source_grouped_oof", "ptt_motion_training_ablation",
     "motion_ptt_training_oof.parquet", "motion_ptt_training_oof"),
    ("ptt22_trained_motion_detector", "ptt22_trained_to_frailty29", "frailty29", "frozen_cross_dataset",
     "frailty29_reverse_evaluation", "motion_internal_reverse_predictions.parquet", "motion_internal_reverse"),
)
MOTION_METRICS = ("balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc", "pr_auc")
STAGE5_FIGURE_SOURCES: Mapping[str, tuple[str, ...]] = {
    "motion_detector_metrics": ("motion_detector_metrics", ),
    **{
        f"{prefix}{'_' + suffix if suffix else ''}_confusion_matrix": (
            f"motion_detector_{'window' if suffix == '' else 'file'}_confusion",
        )
        for prefix in (
            "motion_internal",
            "motion_ptt",
            "motion_ptt_training_oof",
            "motion_internal_reverse",
        ) for suffix in ("", "file")
    },
    **{
        f"{model}_{level}_{kind}": (table, )
        for model in ("frailty29_trained", "ptt22_trained") for level in ("window", "file") for kind, table in (
            ("score_distribution", "motion_detector_score_distributions"),
            ("prediction_tsne", "motion_detector_prediction_tsne"),
            ("roc_auc_curve", "motion_detector_roc_curves"),
        )
    },
    "motion_training_learning_curves": ("motion_training_history", ),
    "denoiser_interval_rmse": ("denoiser_static", "denoiser_dynamic"),
    "denoiser_beat_f1": ("denoiser_static", "denoiser_dynamic"),
    "denoiser_beat_sensitivity": ("denoiser_summary", ),
    "denoiser_beat_ppv": ("denoiser_summary", ),
    "denoiser_runtime": ("denoiser_summary", ),
    **{
        f"static_peak_detector_{name}": (
            "static_peak_detector_recording_metrics",
            "static_peak_detector_summary",
        )
        for name in ("f1", "sensitivity", "ppv", "interval_rmse", "runtime")
    },
}
HYPER_ROOT_TABLE_ALIASES = {
    "comprehensive_model_comparison": "case_summary",
    "model_comparison_performance": "predictive_leaderboard",
    "model_comparison_uncertainty": "metric_distribution_summary",
    "model_comparison_inference": "paired_participant_inference",
    "model_comparison_robustness": "worst_class_f1_stability",
    "exploratory_selected_paired_inference": "paired_participant_inference",
    "pairwise_repeat_metric_deltas": "pairwise_repeat_metric_deltas",
    "classifier_per_class_results": "classifier_per_class_results",
    "selection_conclusions": "selection_conclusions",
    "test_components": "test_components",
    "reporter_profiles": "reporter_profiles",
    "reproducibility": "reproducibility_summary",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return [dict(row) for row in value]
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    except ImportError as error:
        raise RuntimeError("reading report parquet requires pandas") from error


def _write_table(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_json(tables / f"{name}.json", values)
    fields = list(dict.fromkeys(str(key) for row in values for key in row))
    with (tables / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({
                key: json.dumps(value, ensure_ascii=False, default=str) if isinstance(value,
                                                                                      (dict, list, tuple)) else value
                for key, value in row.items()
            } for row in values)


def _add_table(root: Path, tables: list[ReportTable], name: str,
               rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    _write_table(root, name, values)
    tables.append(ReportTable(name, values, compact=False))
    return values


def _inventory_item(artifact_id: str, kind: str, path: Any, source: Any, status: Any) -> dict[str, Any]:
    return dict(v2_artifact_id=artifact_id, artifact_kind=kind, v5_artifact=path,
                semantic_source=source, status=status)


def _inventory_rows(
    tables: Sequence[ReportTable],
    statuses: Sequence[Mapping[str, Any]],
    *,
    figures_first: bool,
    figure_source: str | None = None,
    table_sources: Mapping[str, str] | None = None,
    default_table_source: str = "",
) -> list[dict[str, Any]]:
    figure_rows = [
        _inventory_item(str(row.get("figure")), "figure", row.get("path"),
                        row.get("source_tables") if figure_source is None else figure_source, row.get("status"))
        for row in statuses
    ]
    table_rows = [
        _inventory_item(table.name, "table", f"tables/{table.name}.csv",
                        table.name if table_sources is None else table_sources.get(table.name, default_table_source),
                        "generated")
        for table in tables
    ]
    return figure_rows + table_rows if figures_first else table_rows + figure_rows


def _pair_row(table: str, figure: Any, status: Any, path: Any) -> dict[str, Any]:
    return {"table": table, "figure": figure, "figure_status": status, "figure_path": path}


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(path: Path, pyplot: Any, figure: Any) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    pyplot.close(figure)


def _matching(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> list[Mapping[str, Any]]:
    return [row for row in rows if all(row[key] == value for key, value in criteria.items())]


def _plot_confusion(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    labels = sorted({int(row["true_label"]) for row in rows})
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    lookup = {value: index for index, value in enumerate(labels)}
    for row in rows:
        matrix[lookup[int(row["true_label"])], lookup[int(row["predicted_label"])]] += 1
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    figure.colorbar(image, ax=axis)
    _save_figure(path, plt, figure)


def _plot_roc(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    curves = classification_roc_curve_rows(rows)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    for label in sorted({row.get("class_label") for row in curves}, key=str):
        points = [row for row in curves if row.get("class_label") == label]
        if points:
            axis.plot(
                [float(row["false_positive_rate"]) for row in points],
                [float(row["true_positive_rate"]) for row in points],
                label=str(label),
            )
    axis.plot([0, 1], [0, 1], "--", color="0.6")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    if curves:
        axis.legend()
    _save_figure(path, plt, figure)


def _aggregate_files(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    """Exact V2 file endpoint: one score per repeat/fold/physical file."""

    aggregators = {
        "mean": lambda values: float(np.mean(values)),
        "median": lambda values: float(np.median(values)),
        "maximum": lambda values: float(np.max(values)),
    }
    if method not in aggregators:
        raise ValueError("file score aggregation must be median, mean, or maximum")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("repeat_index"),
            row.get("fold_index"),
            row.get("participant_id"),
            row.get("file_id", row.get("record_id")),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (repeat, fold, participant, file_id), group in sorted(groups.items(), key=str):
        scores = np.asarray([float(row["p_active"]) for row in group], dtype=np.float64)
        labels = {int(row["activity_label"]) for row in group}
        thresholds = {float(row["threshold"]) for row in group}
        if not np.all(np.isfinite(scores)) or len(labels) != 1 or len(thresholds) != 1:
            raise ValueError("file aggregation requires finite scores and one label/threshold")
        score, threshold = aggregators[method](scores), thresholds.pop()
        output.append({
            "participant_id": participant, "file_id": file_id,
            "activity": group[0].get("activity", group[0].get("role_family")), "activity_label": labels.pop(),
            "threshold": threshold, "p_active": score, "predicted_activity": int(score >= threshold),
            "score_aggregation": method, "window_count": len(scores),
            "repeat_index": repeat, "fold_index": fold,
        })
    return output


def _motion_datasets(
    source: Path, manifest: Mapping[str, Any]
) -> Iterable[tuple[str, str, str, str, str, list[dict[str, Any]]]]:
    stages = manifest.get("stages", {})
    for model, evaluation, target, scope, stage_name, filename, prefix in MOTION_DATASETS:
        stage = stages.get(stage_name, {}) if isinstance(stages, Mapping) else {}
        path = source / str(stage.get("artifact_dir", "")) / filename
        if path.is_file():
            yield model, evaluation, target, scope, prefix, _rows(path)


def _plot_box(path: Path, rows: Sequence[Mapping[str, Any]], metric: str, title: str) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is not None and row.get("status", "passed") == "passed":
            label = str(row.get("algorithm_or_reducer"))
            if row.get("channel") is not None:
                label += f"\n{row['channel']}"
            grouped.setdefault(label, []).append(float(value))
    if not grouped:
        return
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(max(7.0, len(grouped) * 1.2), 4.6))
    axis.boxplot(grouped.values(), tick_labels=grouped.keys(), whis=(10, 90))
    axis.set(title=title, ylabel=metric)
    axis.tick_params(axis="x", rotation=25)
    _save_figure(path, plt, figure)


def _plot_summary(path: Path, rows: Sequence[Mapping[str, Any]], metric: str, title: str) -> None:
    selected = [row for row in rows if row.get(metric) is not None]
    if not selected:
        return
    labels = [
        " · ".join(
            str(value) for value in (
                row.get("algorithm_or_reducer"),
                row.get("activity_group"),
                row.get("channel"),
            ) if value is not None) for row in selected
    ]
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(max(7.0, 0.7 * len(labels)), 4.8))
    axis.bar(range(len(labels)), [float(row[metric]) for row in selected])
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set(title=title, ylabel=metric)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(path, plt, figure)


def _render(path: Path, draw: Any) -> bool:
    """Render one semantic figure or leave an explicit N/A companion."""

    plt = _pyplot()
    figure = None
    try:
        figure = draw(plt)
        figure.savefig(path, dpi=160, bbox_inches="tight")
        return True
    except (KeyError, TypeError, ValueError) as error:
        path.with_suffix(".NA.txt").write_text(f"N/A: {type(error).__name__}: {error}\n", encoding="utf-8")
        return False
    finally:
        if figure is not None:
            plt.close(figure)


def _binary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    from ppg_frailty.quality.motion_runner import _binary_metrics as calculate

    thresholds = {float(row["threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise ValueError("motion metric group requires one frozen threshold")
    return calculate(
        np.asarray([int(row["activity_label"]) for row in rows], dtype=np.int64),
        np.asarray([float(row["p_active"]) for row in rows], dtype=np.float64),
        thresholds.pop(),
    )


def _bootstrap_ci(values: Sequence[float], resamples: int, seed: int) -> tuple[float | None, float | None]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        vector.size,
        np.full(vector.size, 1.0 / vector.size),
        size=resamples,
    )
    return tuple(float(value) for value in np.quantile(counts @ vector / vector.size, (0.025, 0.975)))


def _sign_flip(values: Sequence[float], resamples: int, seed: int) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size < 2:
        raise ValueError("paired sign-flip requires at least two participants")
    observed = abs(float(np.mean(vector)))
    if np.allclose(vector, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    rng, extreme, done = np.random.default_rng(seed), 0, 0
    while done < resamples:
        count = min(10_000, resamples - done)
        signs = 2.0 * rng.integers(0, 2, size=(count, vector.size)) - 1.0
        extreme += int(np.count_nonzero(np.abs(np.mean(signs * vector, axis=1)) >= observed - 1e-15))
        done += count
    return float((extreme + 1) / (resamples + 1))


def _holm(rows: list[dict[str, Any]], *, key: str, alpha: float) -> None:
    ordered = sorted(
        (float(row["raw_p_value"]), index) for index, row in enumerate(rows) if row.get("raw_p_value") is not None)
    running, count = 0.0, len(ordered)
    for rank, (raw, index) in enumerate(ordered, 1):
        running = max(running, (count - rank + 1) * raw)
        rows[index].update(
            holm_family=key,
            holm_family_size=count,
            holm_adjusted_p_value=min(1.0, running),
            alpha=alpha,
            reject_after_holm=min(1.0, running) <= alpha,
        )


def _paired_statistics(
    differences: Sequence[float], bootstrap_resamples: int, permutation_resamples: int, seed: int,
    between: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    low, high = _bootstrap_ci(differences, bootstrap_resamples, seed)
    return {
        "candidate_minus_reference": float(np.mean(differences)),
        "paired_participant_ci95_low": low, "paired_participant_ci95_high": high,
        "raw_p_value": _sign_flip(differences, permutation_resamples, seed),
        "paired_participant_count": len(differences),
        **dict(between or {}),
        "bootstrap_resamples": bootstrap_resamples, "permutation_resamples": permutation_resamples,
        "seed": seed,
    }


def _participant_motion_rows(
    datasets: Sequence[tuple[str, str, str, str, str, Sequence[Mapping[str, Any]]]],
    aggregation: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    output: list[dict[str, Any]] = []
    levels: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model, evaluation, target, scope, _prefix, raw in datasets:
        for level, rows in (("window", list(raw)), ("file", _aggregate_files(raw, aggregation))):
            levels[(evaluation, level)] = rows
            participants: dict[str, list[Mapping[str, Any]]] = {}
            for row in rows:
                participants.setdefault(str(row["participant_id"]), []).append(row)
            for participant, selected in sorted(participants.items()):
                try:
                    metrics = _binary_metrics(selected)
                except ValueError:
                    continue
                output.append({
                    "model_id": model, "dataset": evaluation, "target_dataset": target,
                    "evaluation_scope": scope, "aggregation_level": level,
                    "participant_id": participant, "observation_count": len(selected),
                    **metrics,
                })
    return output, levels


def _motion_metric_rows(
    datasets: Sequence[tuple[str, str, str, str, str, Sequence[Mapping[str, Any]]]],
    levels: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    participant_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, worst = [], []
    for model, evaluation, target, scope, _prefix, _raw in datasets:
        for level in ("window", "file"):
            rows = list(levels[(evaluation, level)])
            metrics = _binary_metrics(rows)
            grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
            for row in rows:
                if row.get("repeat_index") is not None and row.get("fold_index") is not None:
                    grouped.setdefault((int(row["repeat_index"]), int(row["fold_index"])), []).append(row)
            fold_scores = []
            for (repeat, fold), selected in sorted(grouped.items()):
                try:
                    score = _binary_metrics(selected)["balanced_accuracy"]
                except ValueError:
                    continue
                fold_scores.append(score)
                worst.append({
                    "model_id": model, "dataset": evaluation, "aggregation_level": level,
                    "repeat_index": repeat, "fold_index": fold, "balanced_accuracy": score,
                })
            row = {
                "model_id": model, "dataset": evaluation, "target_dataset": target,
                "evaluation_scope": scope, "aggregation_level": level, "observation_count": len(rows),
                "participant_count": len({str(value["participant_id"])
                                          for value in rows}),
                "file_count": len({(str(value["participant_id"]), str(value["file_id"]))
                                   for value in rows}),
                "worst_fold_balanced_accuracy": min(fold_scores) if fold_scores else None,
                **metrics,
            }
            selected_participants = _matching(participant_rows,
                                              model_id=model,
                                              dataset=evaluation,
                                              aggregation_level=level)
            for metric in MOTION_METRICS:
                values = [float(value[metric]) for value in selected_participants]
                low, high = _bootstrap_ci(values, bootstrap_resamples, bootstrap_seed)
                row.update({
                    f"participant_macro_{metric}": float(np.mean(values)) if values else None,
                    f"participant_macro_{metric}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    f"participant_macro_{metric}_ci95_low": low,
                    f"participant_macro_{metric}_ci95_high": high,
                })
            summary.append(row)
    return summary, worst


def _motion_inference(
    participant_rows: Sequence[Mapping[str, Any]],
    levels: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    reference: str,
    candidate: str,
    bootstrap_resamples: int,
    permutation_resamples: int,
    seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in sorted({str(row["target_dataset"]) for row in participant_rows}):
        for level in ("window", "file"):
            evaluations = {
                str(row["model_id"]): str(row["dataset"])
                for row in participant_rows if row["target_dataset"] == target and row["aggregation_level"] == level
            }
            if not {reference, candidate} <= evaluations.keys():
                continue

            def unit(row: Mapping[str, Any]) -> tuple[str, ...]:
                base = (str(row["participant_id"]), str(row["file_id"]))
                return (*base, str(row.get("window_id"))) if level == "window" else base

            by_model = {
                model: {unit(row): row
                        for row in levels[(evaluations[model], level)]}
                for model in (reference, candidate)
            }
            if set(by_model[reference]) != set(by_model[candidate]) or any(
                    int(by_model[reference][key]["activity_label"]) != int(by_model[candidate][key]["activity_label"])
                    for key in by_model[reference]):
                raise ValueError(f"training-source comparison units disagree for {target}/{level}")
            family: list[dict[str, Any]] = []
            for metric in MOTION_METRICS:
                values: dict[str, dict[str, float]] = {}
                for row in participant_rows:
                    if row["target_dataset"] == target and row["aggregation_level"] == level:
                        values.setdefault(str(row["model_id"]), {})[str(row["participant_id"])] = float(row[metric])
                left, right = values.get(reference, {}), values.get(candidate, {})
                common = sorted(set(left) & set(right))
                if len(common) < 2:
                    continue
                differences = [right[key] - left[key] for key in common]
                family.append({
                    "reference_model_id": reference,
                    "candidate_model_id": candidate,
                    "target_dataset": target,
                    "aggregation_level": level,
                    "metric": metric,
                    **_paired_statistics(differences, bootstrap_resamples, permutation_resamples, seed),
                    "analysis_role": "retrospective_training_source_ablation",
                })
            _holm(family, key=f"detector::{target}::{level}", alpha=alpha)
            output.extend(family)
    return output


def _confusion_row(model: str, evaluation: str, level: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in rows:
        matrix[int(row["activity_label"]), int(float(row["p_active"]) >= float(row["threshold"]))] += 1
    return {
        "model_id": model, "dataset": evaluation, "aggregation_level": level,
        "class_order": [0, 1], "confusion_matrix": matrix.tolist(),
        "true_static_predicted_static": int(matrix[0, 0]), "true_static_predicted_motion": int(matrix[0, 1]),
        "true_motion_predicted_static": int(matrix[1, 0]), "true_motion_predicted_motion": int(matrix[1, 1]),
    }


def _score_distribution_rows(
    datasets: Sequence[tuple[str, str, str, str, str, Sequence[Mapping[str, Any]]]],
    levels: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for model, evaluation, _target, scope, _prefix, _raw in datasets:
        for level in ("window", "file"):
            rows = levels[(evaluation, level)]
            thresholds = [float(row["threshold"]) for row in rows]
            for label, name in ((0, "static"), (1, "motion")):
                scores = np.asarray([float(row["p_active"]) for row in rows if int(row["activity_label"]) == label])
                if not scores.size:
                    continue
                q05, q25, q50, q75, q95 = np.quantile(scores, (0.05, 0.25, 0.5, 0.75, 0.95))
                output.append({
                    "model_id": model, "dataset": evaluation,
                    "evaluation_scope": scope, "aggregation_level": level,
                    "activity_class": name, "observation_count": scores.size,
                    "score_mean": float(np.mean(scores)),
                    "score_sd": float(np.std(scores, ddof=1)) if scores.size > 1 else None,
                    "score_q05": float(q05), "score_q25": float(q25), "score_median": float(q50),
                    "score_q75": float(q75), "score_q95": float(q95),
                    "threshold_min": min(thresholds), "threshold_median": median(thresholds),
                    "threshold_max": max(thresholds),
                })
    return output


def _motion_history(source: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    stages = manifest.get("stages", {})
    for stage_name in ("internal_motion_oof", "ptt_motion_training_ablation"):
        stage = stages.get(stage_name, {}) if isinstance(stages, Mapping) else {}
        directory = source / str(stage.get("artifact_dir", ""))
        for path in sorted(directory.rglob("motion_training_history.json")) if directory.is_dir() else ():
            payload = _json(path)
            for row in payload.get("rows", ()):
                rows.append({
                    "case_id": "ptt22" if "ptt" in stage_name else "frailty29",
                    "history_path": path.relative_to(source).as_posix(),
                    "repeat": payload.get("repeat_index", payload.get("repeat")),
                    "fold": payload.get("fold_index", payload.get("fold")),
                    "final_fit": payload.get("final_fit", False),
                    **dict(row),
                })
    return rows


def _plot_motion_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [f"{row['dataset']}\n{row['aggregation_level']}" for row in rows]
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(max(9.0, 2.0 * len(labels)), 5.0))
    x, width = np.arange(len(labels)), 0.8 / len(MOTION_METRICS)
    for index, metric in enumerate(MOTION_METRICS):
        axis.bar(
            x + (index - 2.5) * width,
            [float(row[metric]) for row in rows],
            width=width,
            label=metric.replace("_", " "),
        )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set(title="Motion detector window- and file-level metrics", ylabel="Score", ylim=(0, 1))
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(path, plt, figure)


def _denoiser_inference(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    bootstrap_resamples: int,
    permutation_resamples: int,
    seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """V2 participant-paired tests on identical successful segment rosters."""

    from ppg_frailty.quality.stage5_pre import _aggregate_benchmark

    endpoints = (
        "participant_macro_f1",
        "participant_macro_sensitivity",
        "participant_macro_positive_predictive_value",
        "participant_macro_ibi_ppi_rmse_ms",
    )
    algorithms = sorted({str(row["algorithm_or_reducer"]) for row in rows} - {reference})
    output: list[dict[str, Any]] = []
    for activity in sorted({str(row["activity_group"]) for row in rows}):
        for channel in sorted({str(row["channel"]) for row in rows}):
            for endpoint in endpoints:
                family: list[dict[str, Any]] = []
                for candidate in algorithms:
                    selected = [
                        row for row in rows if str(row["activity_group"]) == activity and str(row["channel"]) == channel
                        and str(row["algorithm_or_reducer"]) in {reference, candidate}
                    ]
                    by_algorithm: dict[str, dict[tuple[str, str, float], Mapping[str, Any]]] = {}
                    for row in selected:
                        key = (
                            str(row["participant_id"]),
                            str(row["record_id"]),
                            float(row["segment_start_s"]),
                        )
                        by_algorithm.setdefault(str(row["algorithm_or_reducer"]), {})[key] = row
                    left, right = by_algorithm.get(reference, {}), by_algorithm.get(candidate, {})
                    common = [
                        key for key in sorted(set(left) & set(right))
                        if left[key].get("status") == "passed" and right[key].get("status") == "passed" and (
                            endpoint != "participant_macro_ibi_ppi_rmse_ms" or all(
                                value.get("ibi_ppi_rmse_ms") is not None for value in (left[key], right[key])))
                    ]
                    paired = []
                    for participant in sorted({key[0] for key in common}):
                        keys = [key for key in common if key[0] == participant]
                        left_summary = _aggregate_benchmark([left[key] for key in keys])[0]
                        right_summary = _aggregate_benchmark([right[key] for key in keys])[0]
                        if left_summary.get(endpoint) is not None and right_summary.get(endpoint) is not None:
                            paired.append(float(right_summary[endpoint]) - float(left_summary[endpoint]))
                    if len(paired) < 2:
                        continue
                    family.append({
                        "reference_denoiser": reference,
                        "candidate_denoiser": candidate,
                        "activity_group": activity,
                        "channel": channel,
                        "metric": endpoint,
                        **_paired_statistics(
                            paired,
                            bootstrap_resamples,
                            permutation_resamples,
                            seed,
                            {"endpoint_common_segment_count": len(common)},
                        ),
                        "analysis_role": "retrospective_identity_controlled",
                    })
                _holm(
                    family,
                    key=f"denoiser::{activity}::{channel}::{endpoint}",
                    alpha=alpha,
                )
                output.extend(family)
    return output


def _component_rows(source: Path, plan: Mapping[str, Any],
                    manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from .components import build_motion_peak_test_component_rows
        from .profiles import reporter_profile_rows

        components = list(build_motion_peak_test_component_rows(plan, manifest, study_root=source))
        return components, list(reporter_profile_rows(components))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        components = [{
            "component_id": str(name),
            "component_role": "persisted_stage",
            "status": stage.get("status", "unknown") if isinstance(stage, Mapping) else "unknown",
        } for name, stage in (manifest.get("stages", {}) or {}).items()]
        return components, []


def _phase_report(
    source: Path,
    target: Path,
    *,
    preferred_reference: str | None,
) -> tuple[dict[str, Any], Mapping[str, tuple[Mapping[str, Any], ...]]]:
    """Render one hyperparameter phase through the ordinary 35-figure engine."""

    from ppg_frailty.v5_reporting.analysis import build_analysis
    from ppg_frailty.v5_reporting.collect import load_report_data
    from ppg_frailty.v5_reporting.contracts import ReportRequest, ResolvedSelection, RunSpec
    from ppg_frailty.v5_reporting.plots import generate_selected_figures
    from ppg_frailty.v5_reporting.registry import resolve_selection

    manifest = _json(source / "study_manifest.json")
    case_ids = tuple(
        str(row["case_id"]) for row in manifest.get("cases", ())
        if isinstance(row, Mapping) and row.get("case_id") not in (None, ""))
    reference = preferred_reference if preferred_reference in case_ids else (case_ids[0] if case_ids else None)
    mode = "comparison" if reference is not None and len(case_ids) > 1 else "single"
    request = ReportRequest(
        mode=mode,
        runs=(RunSpec(source.name, source), ),
        reference_case=reference if mode == "comparison" else None,
        comparison_family=str(manifest.get("study_id", source.name)),
        presets=("full", ),
        on_missing="na",
    )
    base = resolve_selection(mode=mode, presets=("full", ), modules=(), figures=None, tables=None)
    # The historical phase report exposed the complete ordinary 35-ID surface;
    # profile-inapplicable figures are explicit N/A artifacts.
    selection = ResolvedSelection(base.modules, base.tables, STATIC_FIGURE_NAMES)
    data = load_report_data(request)
    products = build_analysis(data, request, selection)
    tables: list[ReportTable] = []
    add_table = partial(_add_table, target, tables)
    for name, values in sorted(products.tables.items()):
        add_table(name, values)
    status = generate_selected_figures(data, products, request, STATIC_FIGURE_NAMES, target / "figures")
    add_table("figure_status", status)
    figures = [target / str(row["path"]) for row in status if row.get("status") == "generated" and row.get("path")]
    report = _finish_report(
        target, source=source, title=str(manifest.get("study_id", source.name)), tables=tables, figures=figures
    )
    report["figure_status"] = [dict(row) for row in status]
    _write_json(target / "report_manifest.json", report)
    return report, products.tables


def _report_directory(source: Path, output_dir: str | Path | None, api: str) -> Path:
    if output_dir is None:
        warnings.warn(
            f"{api}() without output_dir is an in-place V2 compatibility "
            "form; use analyse_report.py specialized-report",
            DeprecationWarning,
            stacklevel=3,
        )
        return source
    return Path(output_dir).resolve()


def _report_state(study_dir: str | Path, output_dir: str | Path | None,
                  api: str) -> tuple[Path, Path, dict[str, Any], Path, list[ReportTable], list[Path]]:
    source = Path(study_dir).resolve()
    output = _report_directory(source, output_dir, api)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    return source, output, _json(source / "study_manifest.json"), figures, [], []


def _finish_report(
    output: Path,
    *,
    source: Path,
    title: str,
    tables: Sequence[ReportTable],
    figures: Sequence[Path],
) -> dict[str, Any]:
    workbook = output / "tables" / "report_tables.xlsx"
    workbook.parent.mkdir(parents=True, exist_ok=True)
    write_excel_workbook(workbook, tables)
    relative_figures = [path.relative_to(output).as_posix() for path in figures if path.is_file()]
    markdown = [
        f"# {title}",
        "",
        f"Source: {source}",
        "",
        "## Figures",
        "",
        *[f"![{Path(path).stem}]({path})" for path in relative_figures],
        "",
        "## Tables",
        "",
        "See tables/report_tables.xlsx and CSV/JSON companions.",
    ]
    (output / "STUDY_SUMMARY.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output / "STUDY_SUMMARY.html").write_text(
        "<!doctype html><meta charset='utf-8'><h1>" + title + "</h1>" +
        "".join(f"<img src='{path}' alt='{Path(path).stem}'>" for path in relative_figures),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "ppg_frailty.specialized_report.v1",
        "source_pipeline_output": str(source),
        "tables": [table.name for table in tables],
        "figures": relative_figures,
    }
    _write_json(output / "report_manifest.json", manifest)
    return manifest


def generate_motion_peak_report(study_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Rebuild the complete Stage-5/static-peak analysis from immutable data."""

    source, output, manifest, figure_dir, tables, images = _report_state(study_dir, output_dir,
                                                                         "generate_motion_peak_report")
    plan = yaml.safe_load((source / "resolved_plan.yaml").read_text(encoding="utf-8")) or {}
    report = plan.get("report", {}) if isinstance(plan, Mapping) else {}
    report = report if isinstance(report, Mapping) else {}
    add_table = partial(_add_table, output, tables)
    components, profiles = _component_rows(source, plan, manifest)
    add_table("test_components", components)
    add_table("reporter_profiles", profiles)
    add_table(
        "denoiser_algorithms",
        [row for row in components if row.get("component_role") == "denoiser"],
    )

    if manifest.get("study_type") == "stage_ablation_01_static_peak_detectors":
        from ppg_frailty.quality.stage5_pre import _aggregate_static_peak_benchmark

        stage = manifest["stages"]["static_peak_ablation"]
        result = _json(source / str(stage["artifact_dir"]) / "static_peak_ablation.json")
        rows = add_table("rows", result.get("rows", ()))
        summary = list(result.get("summary_rows", ()))
        if rows and all(row.get("status") is not None for row in rows):
            summary = _aggregate_static_peak_benchmark(rows)
        comparisons = list(result.get("statistical_comparisons", ()))
        add_table("summary_rows", summary)
        add_table("statistical_comparisons", comparisons)
        conclusions = [{
            "study": "static_peak_detector_ablation",
            "comparison_count": len(comparisons),
            "significant_count": sum(bool(row.get("reject_at_alpha")) for row in comparisons),
            "interpretation_scope": "recording_level_rank_sum_with_holm_sidak",
        }]
        aliases = (
            ("static_peak_detector_recording_metrics", rows),
            ("static_peak_detector_distribution_statistics", summary),
            ("static_peak_detector_rank_sum_holm_sidak", comparisons),
            ("static_peak_detector_significance_summary", comparisons),
            ("static_peak_detector_endpoint_effects", comparisons),
            ("static_peak_detector_summary", summary),
            ("result_comparison", summary),
            ("result_comparison_compact", summary),
            ("result_conclusions", conclusions),
        )
        for name, values in aliases:
            add_table(name, values)
        specs = (
            ("f1_percent", "f1", "Static PTT recording beat F1 (%)"),
            ("sensitivity_percent", "sensitivity", "Static PTT beat sensitivity (%)"),
            ("positive_predictive_value_percent", "ppv", "Static PTT beat PPV (%)"),
            ("ibi_ppi_rmse_ms", "interval_rmse", "Static recording IBI–PPI RMSE (ms)"),
            ("execution_time_percent", "runtime", "Execution time (% signal duration)"),
        )
        for metric, figure_id, title in specs:
            path = figure_dir / f"static_peak_detector_{figure_id}.png"
            _plot_box(path, rows, metric, title)
            if path.is_file():
                images.append(path)
            # Short-lived V5 name retained as a file alias for existing consumers.
            compatibility = figure_dir / f"static_peak_{metric}.png"
            if path.is_file():
                shutil.copy2(path, compatibility)
                images.append(compatibility)
        relevant_figures = {
            key: value
            for key, value in STAGE5_FIGURE_SOURCES.items() if key.startswith("static_peak_detector_")
        }
    else:
        from ppg_frailty.quality.stage5_pre import _aggregate_benchmark
        from . import plots as semantic_plots

        aggregation = str(report.get("file_score_aggregation", "median"))
        bootstrap_resamples = int(report.get("participant_cluster_bootstrap_resamples", 10_000))
        bootstrap_seed = int(report.get("participant_cluster_bootstrap_seed", 42))
        permutation_resamples = int(report.get("participant_paired_permutation_resamples", 100_000))
        permutation_seed = int(report.get("participant_paired_permutation_seed", 42))
        alpha = float(report.get("alpha", 0.05))
        detector_reference = report.get("detector_inference_reference_model_id", "frailty29_trained_motion_detector")
        detector_candidate = report.get("detector_inference_candidate_model_id", "ptt22_trained_motion_detector")
        denoiser_reference = report.get("denoiser_inference_reference_id", "identity")
        datasets = list(_motion_datasets(source, manifest))
        participant_rows, levels = _participant_motion_rows(datasets, aggregation)
        metrics, fold_rows = _motion_metric_rows(
            datasets,
            levels,
            participant_rows,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        inference = _motion_inference(
            participant_rows,
            levels,
            reference=str(detector_reference),
            candidate=str(detector_candidate),
            bootstrap_resamples=bootstrap_resamples,
            permutation_resamples=permutation_resamples,
            seed=permutation_seed,
            alpha=alpha,
        )
        predictions, normalized, per_class = [], [], []
        confusion_window, confusion_file = [], []
        for model, evaluation, _target, _scope, prefix, raw in datasets:
            for level in ("window", "file"):
                values = list(levels[(evaluation, level)])
                prepared = [{
                    **row,
                    "repeat": row.get("repeat_index"),
                    "fold": row.get("fold_index"),
                } for row in values]
                current = list(
                    normalize_classification_rows(
                        prepared,
                        classifier_id=model,
                        evaluation_id=evaluation,
                        aggregation_level=level,
                        label_field="activity_label",
                    ))
                normalized.extend(current)
                predictions.extend({
                    "model_id": model,
                    "dataset": evaluation,
                    "aggregation_level": level,
                    **row
                } for row in values)
                current_per_class = list(
                    classification_per_class_metric_rows(current, class_names={
                        0: "static",
                        1: "motion"
                    }))
                per_class.extend(current_per_class)
                add_table(f"{evaluation}_{level}_predictions", current)
                add_table(f"{evaluation}_{level}_per_class", current_per_class)
                confusion = _confusion_row(model, evaluation, level, values)
                (confusion_window if level == "window" else confusion_file).append(confusion)
                canonical = figure_dir / f"{prefix}{'_file' if level == 'file' else ''}_confusion_matrix.png"
                if current:
                    _plot_confusion(canonical, current, f"{evaluation} · {level}")
                    images.append(canonical)
                    compatibility = figure_dir / f"{evaluation}_{level}_confusion.png"
                    shutil.copy2(canonical, compatibility)
                    images.append(compatibility)
        roc = list(
            classification_roc_curve_rows(
                normalized,
                macro_grid_points=int(report.get("classification_roc_macro_grid_points", 201)),
            ))
        tsne = list(
            classification_tsne_rows(
                normalized,
                random_state=int(report.get("classification_tsne_random_state", 42)),
                perplexity=float(report.get("classification_tsne_perplexity", 30.0)),
                max_samples=int(report.get("classification_tsne_max_samples", 2000)),
            ))
        diagnostic_status = list(
            classification_diagnostic_status_rows(tuple(dict.fromkeys(item[0] for item in datasets)), normalized, roc,
                                                  tsne))
        score_summary = _score_distribution_rows(datasets, levels)
        history = _motion_history(source, manifest)
        table_values = {
            "motion_detector_metrics": metrics,
            "motion_detector_participant_macro_statistics": metrics,
            "motion_detector_training_source_inference": inference,
            "motion_detector_worst_fold_ba": [{
                key: row.get(key)
                for key in (
                    "model_id",
                    "dataset",
                    "evaluation_scope",
                    "aggregation_level",
                    "worst_fold_balanced_accuracy",
                )
            } for row in metrics],
            "motion_detector_fold_metrics": fold_rows,
            "motion_detector_participant_metrics_raw": participant_rows,
            "motion_detector_file_predictions": [row for row in predictions if row["aggregation_level"] == "file"],
            "motion_detector_internal_evaluation": _matching(metrics, evaluation_scope="source_grouped_oof"),
            "motion_detector_cross_dataset_evaluation": _matching(metrics, evaluation_scope="frozen_cross_dataset"),
            "motion_detector_score_distributions": score_summary,
            "motion_detector_prediction_scores": normalized,
            "motion_detector_per_class_results": per_class,
            "motion_detector_per_class_performance": per_class,
            "motion_detector_per_class_discrimination": per_class,
            "motion_detector_roc_curves": roc,
            "motion_detector_prediction_tsne": tsne,
            "motion_detector_diagnostic_status": diagnostic_status,
            "motion_detector_window_confusion": confusion_window,
            "motion_detector_file_confusion": confusion_file,
            "motion_training_history": history,
            "inference_configuration": [
                {
                    "analysis": "motion_detector_training_source", "reference": detector_reference,
                    "candidate_or_family": detector_candidate,
                    "paired_resamples": permutation_resamples, "seed": permutation_seed,
                    "multiplicity": "Holm_within_target_x_level_across_endpoints",
                },
                {
                    "analysis": "denoiser_reducer_comparison", "reference": denoiser_reference,
                    "candidate_or_family": "all_non_reference_reducers",
                    "paired_resamples": permutation_resamples, "seed": permutation_seed,
                    "multiplicity": "Holm_within_activity_x_channel_x_endpoint",
                },
            ],
            "reproducibility_summary": [
                {
                    "evidence_scope": "participant_cluster_bootstrap_report",
                    "resample_count": bootstrap_resamples, "resampling_seed": bootstrap_seed,
                    "group_unit": "participant_id", "fit_or_recalibration_on_target": False,
                },
                {
                    "evidence_scope": "participant_paired_sign_flip_report",
                    "resample_count": permutation_resamples, "resampling_seed": permutation_seed,
                    "group_unit": "participant_id", "fit_or_recalibration_on_target": False,
                },
            ],
        }
        for metric in MOTION_METRICS:
            table_values[f"motion_detector_{metric}"] = [{
                "model_id": row["model_id"], "dataset": row["dataset"],
                "scope": row["evaluation_scope"], "level": row["aggregation_level"],
                "participant_id": row["participant_id"],
                "value": row[metric],
            } for row in participant_rows]
        for name, values in table_values.items():
            add_table(name, values)
        metric_path = figure_dir / "motion_detector_metrics.png"
        if metrics:
            _plot_motion_metrics(metric_path, metrics)
            images.append(metric_path)
        collected = SimpleNamespace(plan={"report": dict(report)}, history_rows=history)
        for model, file_prefix, title in (
            ("frailty29_trained_motion_detector", "frailty29_trained", "Frailty29-trained"),
            ("ptt22_trained_motion_detector", "ptt22_trained", "PTT22-trained"),
        ):
            for level in ("window", "file"):
                selected_scores = _matching(normalized, classifier_id=model, aggregation_level=level)
                selected_tsne = _matching(tsne, classifier_id=model, aggregation_level=level)
                selected_roc = _matching(roc, classifier_id=model, aggregation_level=level)
                analysis = StudyAnalysis(
                    classification_prediction_scores=tuple(selected_scores),
                    classification_prediction_tsne=tuple(selected_tsne),
                    classification_roc_curves=tuple(selected_roc),
                )
                for kind, draw in (
                    ("score_distribution",
                     lambda plt, a=analysis: semantic_plots._classification_prediction_scores(collected, a, plt)),
                    ("prediction_tsne", lambda plt, a=analysis: semantic_plots._classification_prediction_tsne(a, plt)),
                    ("roc_auc_curve", lambda plt, a=analysis: semantic_plots._classification_roc_auc_curves(a, plt)),
                ):
                    path = figure_dir / f"{file_prefix}_{level}_{kind}.png"
                    if _render(path, draw):
                        images.append(path)
        learning = figure_dir / "motion_training_learning_curves.png"
        if history and _render(learning, lambda plt: semantic_plots._learning_curves(collected, plt)):
            images.append(learning)

        denoiser_stage = (manifest.get("stages", {}) or {}).get("ptt_denoiser_benchmark", {})
        denoiser_path = source / str(denoiser_stage.get("artifact_dir", "")) / "denoiser_benchmark.json"
        denoiser_summary: list[dict[str, Any]] = []
        denoiser_inference: list[dict[str, Any]] = []
        if denoiser_path.is_file():
            denoiser = _json(denoiser_path)
            denoiser_rows = [dict(row) for row in denoiser.get("rows", ())]
            denoiser_summary = _aggregate_benchmark(denoiser_rows) if denoiser_rows else list(
                denoiser.get("summary_rows", ()))
            denoiser_inference = _denoiser_inference(
                denoiser_rows,
                reference=str(denoiser_reference),
                bootstrap_resamples=bootstrap_resamples,
                permutation_resamples=permutation_resamples,
                seed=permutation_seed,
                alpha=alpha,
            ) if denoiser_rows else []
            coverage = [{
                "denoiser": row["algorithm_or_reducer"], "activity": row["activity_group"],
                "channel": row["channel"],
                "participant_coverage_rate": row.get("participant_coverage_rate"),
                "segment_coverage_rate": row.get("segment_coverage_rate"),
                "failed_segment_count": row.get("failed_segment_count"),
            } for row in denoiser_summary]
            for name, values in (
                ("denoiser_rows", denoiser_rows),
                ("denoiser_summary", denoiser_summary),
                ("denoiser_compact_statistics", denoiser_summary),
                ("denoiser_paired_inference", denoiser_inference),
                ("denoiser_coverage", coverage),
                ("denoiser_static", _matching(denoiser_summary, activity_group="static")),
                ("denoiser_dynamic", _matching(denoiser_summary, activity_group="dynamic")),
            ):
                add_table(name, values)
            for metric, figure_id, title in (
                ("participant_macro_ibi_ppi_rmse_ms", "denoiser_interval_rmse", "IBI–PPI RMSE"),
                ("participant_macro_f1", "denoiser_beat_f1", "Beat F1"),
                ("participant_macro_sensitivity", "denoiser_beat_sensitivity", "Beat sensitivity"),
                ("participant_macro_positive_predictive_value", "denoiser_beat_ppv", "Beat PPV"),
                ("total_runtime_s", "denoiser_runtime", "Reducer + detector runtime"),
            ):
                path = figure_dir / f"{figure_id}.png"
                _plot_summary(path, denoiser_summary, metric, title)
                if path.is_file():
                    images.append(path)
        comparison_stage = (manifest.get("stages", {}) or {}).get("motion_model_comparison_package", {})
        comparison_path = source / str(comparison_stage.get("artifact_dir",
                                                            "")) / "motion_model_comparison_manifest.json"
        comparison = list(_json(comparison_path).get("candidates", ())) if comparison_path.is_file() else []
        add_table("motion_model_comparison_candidates", comparison)
        combined = [
            *({
                "evidence_type": "motion_detector",
                **row
            } for row in metrics),
            *({
                "evidence_type": "denoiser",
                **row
            } for row in denoiser_summary),
        ]
        add_table("result_comparison", combined)
        add_table("result_comparison_compact", combined)
        add_table(
            "result_conclusions",
            [
                {
                    "evidence_type": "motion_detector", "comparison_count": len(metrics),
                    "paired_inference_count": len(inference),
                    "scope": "window_and_file; source_oof_and_frozen_cross_dataset",
                },
                {
                    "evidence_type": "denoiser", "comparison_count": len(denoiser_summary),
                    "paired_inference_count": len(denoiser_inference),
                    "scope": "participant_macro_by_activity_and_wavelength",
                },
            ],
        )
        relevant_figures = {
            key: value
            for key, value in STAGE5_FIGURE_SOURCES.items() if not key.startswith("static_peak_detector_")
        }

    status_rows = []
    for figure_id, source_tables in relevant_figures.items():
        path = figure_dir / f"{figure_id}.png"
        na_path = figure_dir / f"{figure_id}.NA.txt"
        if not path.is_file() and not na_path.is_file():
            na_path.write_text("N/A: required source stage or diagnostic rows unavailable\n", encoding="utf-8")
        status_rows.append({
            "figure": figure_id,
            "status": "generated" if path.is_file() else "N/A",
            "path": (path if path.is_file() else na_path).relative_to(output).as_posix(),
            "source_tables": list(source_tables),
        })
    add_table("reporter_output_status", status_rows)
    add_table(
        "table_figure_pairs",
        [
            _pair_row(table, row["figure"], row["status"], row["path"]) for row in status_rows
            for table in row["source_tables"]
        ],
    )
    add_table("v2_v5_specialized_inventory", _inventory_rows(tables, status_rows, figures_first=True))
    return _finish_report(
        output, source=source, title=str(manifest.get("study_id", source.name)), tables=tables, figures=images
    )


def generate_hyperparameter_report(
    output: str | Path,
    *,
    plan: Mapping[str, Any] | None = None,
    ranking_tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    output_dir: str | Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Render root rankings and every nested phase through shared analysis."""

    source, target, manifest, figure_dir, tables, images = _report_state(output, output_dir,
                                                                         "generate_hyperparameter_report")
    resolved_plan = plan or yaml.safe_load((source / "study_plan.yaml").read_text(encoding="utf-8"))
    names = [str(name) for name in manifest.get("ranking_tables", ())]
    rankings = dict(ranking_tables or {})
    add_table = partial(_add_table, target, tables)
    metric = str(resolved_plan["resource"]["ranking_metric"])

    def ranking_figure(name: str, rows: Sequence[Mapping[str, Any]], field: str) -> None:
        selected = [row for row in rows if row.get(field) is not None]
        if not selected:
            return
        path = figure_dir / f"{name}.png"
        plt = _pyplot()
        figure, axis = plt.subplots(figsize=(max(7.0, len(selected) * 0.9), 4.6))
        axis.bar(
            [str(row["case_id"]) for row in selected],
            [100 * float(row[field]) for row in selected],
            yerr=[100 * float(row.get(field.replace("_mean", "_sd")) or 0) for row in selected],
        )
        axis.set(ylabel=f"{field} (%)", title=name.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=30)
        _save_figure(path, plt, figure)
        images.append(path)

    for name in names:
        path = source / "tables" / f"{name}.json"
        rows = [dict(row) for row in rankings.get(name, _rows(path))]
        add_table(name, rows)
        ranking_figure(name, rows, f"{metric}_mean")

    phase_directories = manifest.get("phase_directories", {})
    phase_reports: dict[str, dict[str, Any]] = {}
    phase_products: dict[str, Mapping[str, tuple[Mapping[str, Any], ...]]] = {}
    selected_case = str(manifest.get("selected_case_id", "")) or None
    if isinstance(phase_directories, Mapping):
        for phase, relative in phase_directories.items():
            phase_source = source / str(relative)
            if not (phase_source / "study_manifest.json").is_file():
                continue
            phase_target = target / "phases" / str(phase)
            phase_report, products = _phase_report(phase_source, phase_target, preferred_reference=selected_case)
            phase_reports[str(phase)] = phase_report
            phase_products[str(phase)] = products

    # Participant-OOF rankings use the ordinary analyzer's participant endpoint,
    # while persisted orchestration rankings remain the selection authority.
    participant_rankings: dict[str, list[dict[str, Any]]] = {}
    for phase, products in phase_products.items():
        rows = [dict(row) for row in products.get("case_summary", ())]
        field = f"participant_mean_{metric}"
        rows = [row for row in rows if row.get(field) is not None]
        rows.sort(key=lambda row: (-float(row[field]), str(row["case_id"])))
        participant_rankings[f"{phase}_participant_oof_ranking"] = [{
            "participant_oof_rank": index,
            **row
        } for index, row in enumerate(rows, 1)]
    if {"promotion", "completion"} <= phase_products.keys():
        combined = [
            *participant_rankings.get("promotion_participant_oof_ranking", ()),
            *participant_rankings.get("completion_participant_oof_ranking", ()),
        ]
        combined.sort(key=lambda row: (
            -float(row.get(f"participant_mean_{metric}", -math.inf)),
            str(row["case_id"]),
        ))
        participant_rankings["all_candidates_full_cv_participant_oof_ranking"] = [{
            **row, "participant_oof_rank": index
        } for index, row in enumerate(combined, 1)]
    for name, rows in participant_rankings.items():
        add_table(name, rows)
        ranking_figure(name, rows, f"participant_mean_{metric}")

    phase_order = ("completion", "promotion", "full_cv", "screen")
    authoritative = next((name for name in phase_order if name in phase_products), None)
    root_status: list[dict[str, Any]] = []
    if authoritative is not None:
        products = phase_products[authoritative]
        for alias, source_name in HYPER_ROOT_TABLE_ALIASES.items():
            add_table(alias, products.get(source_name, ()))
        for row in phase_reports[authoritative].get("figure_status", ()):
            status = {"phase": authoritative, **dict(row)}
            relative = str(row.get("path", ""))
            source_path = target / "phases" / authoritative / relative
            root_path = figure_dir / Path(relative).name
            if source_path.is_file():
                shutil.copy2(source_path, root_path)
                status["path"] = root_path.relative_to(target).as_posix()
                if root_path.suffix == ".png":
                    images.append(root_path)
            root_status.append(status)
    add_table("root_diagnostic_figures", root_status)
    add_table("root_reporter_artifact_status", root_status)
    selected_path = source / "selected_configuration.json"
    if selected_path.is_file():
        add_table("selected_configuration", [_json(selected_path)])
    add_table(
        "table_figure_pairs",
        [
            _pair_row(
                name,
                name,
                "generated" if (figure_dir / f"{name}.png").is_file() else "N/A",
                f"figures/{name}.png",
            ) for name in (*names, *participant_rankings)
        ] + [
            _pair_row(
                ";".join(FIGURE_TABLE_SOURCES.get(str(row.get("figure")), ())),
                row.get("figure"),
                row.get("status"),
                row.get("path"),
            ) for row in root_status
        ],
    )
    inventory = _inventory_rows(
        tables,
        root_status,
        figures_first=False,
        figure_source="ordinary_shared_35_figure_renderer",
        table_sources=HYPER_ROOT_TABLE_ALIASES,
        default_table_source="persisted_ranking_or_shared_analysis",
    )
    add_table("v2_v5_specialized_inventory", inventory)
    result = _finish_report(
        target, source=source, title=str(resolved_plan["study"]["study_id"]), tables=tables, figures=images
    )
    result.update(
        phase_reports={name: f"phases/{name}/report_manifest.json"
                       for name in phase_reports},
        ordinary_phase_figure_ids=list(STATIC_FIGURE_NAMES),
        ordinary_phase_figure_count=len(STATIC_FIGURE_NAMES),
    )
    _write_json(target / "report_manifest.json", result)
    return result


def rebuild_hyperparameter_report(study_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    return generate_hyperparameter_report(study_dir, output_dir=output_dir)


__all__ = ["generate_hyperparameter_report", "generate_motion_peak_report", "rebuild_hyperparameter_report"]
