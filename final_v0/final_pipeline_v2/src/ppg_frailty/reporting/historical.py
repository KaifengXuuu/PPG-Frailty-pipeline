"""Reproducible V2-oriented reporting for selected historical V1 searches.

The historical archives do not contain participant-level OOF probabilities.
This reporter therefore limits inference to descriptive repeat summaries and
records every unavailable V2 statistic explicitly instead of reconstructing it
from aggregate confusion matrices.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluate.benchmark import summarize_repeats
from .tabular import (
    markdown_column_definitions_block,
    write_csv,
    write_excel_workbook_from_csv_directory,
    write_table_column_definitions,
)


_PERFORMANCE_METRICS = (
    "window_balanced_accuracy",
    "window_macro_f1",
    "file_balanced_accuracy",
    "file_macro_f1",
    "subject_balanced_accuracy",
    "subject_macro_f1",
)
_SPLIT_SEEDS = (42, 10042, 20042, 30042, 40042)
_MAX_HUMAN_TABLE_COLUMNS = 8
_ARCHIVED_CLASS_FIELDS = (
    ("pre_frail", "pre_frail"),
    ("robust_non_frail", "robust_non_frail"),
    ("young", "young"),
)
_PARTICIPANT_CLUSTER_CI_UNAVAILABLE_REASON = (
    "participant-level OOF rows with stable participant_id, repeat, true_label, "
    "and predicted_label were not archived (class probabilities are additionally "
    "required for ROC-AUC); aggregate repeat metrics cannot reproduce participant-"
    "cluster resampling"
)
_PARTICIPANT_ROC_UNAVAILABLE_REASON = (
    "continuous participant-level OOF class probabilities were not archived"
)
_CONFIG_COLUMNS = (
    "model",
    "resolved_model",
    "extra_input",
    "cnn_epochs",
    "cnn_patience",
    "window_sec",
    "hop_sec",
    "overlap_pct",
    "max_windows_fraction",
    "cnn_lr",
    "cnn_weight_decay",
    "cnn_dropout",
    "cnn_label_smoothing",
    "sqi_mode",
    "aggregation",
    "manual_features",
    "loss_type",
    "class_weight_mode",
    "window_sampler",
    "windows_per_subject_per_epoch",
    "train_overlap_pct",
    "dynamic_data_mode",
    "train_role_mode",
    "validation_role_mode",
    "test_role_mode",
    "stage1_regularization_factor",
    "stage1_regularization_value",
    "cnn_target_fs",
)


def _pd():
    try:
        import pandas as pandas
    except ImportError as exc:  # pragma: no cover - optional reporting dependency
        raise ImportError("historical reporting requires pandas") from exc
    return pandas


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as exc:  # pragma: no cover - optional reporting dependency
        raise ImportError("historical reporting requires matplotlib") from exc
    return pyplot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_path(source: Path, raw: Any) -> Path:
    report_root = (source / "reports").resolve()
    if not report_root.is_dir():
        raise FileNotFoundError(f"historical report root is missing: {report_root}")
    candidate = Path(str(raw).replace("\\", "/"))
    if "reports" in candidate.parts:
        index = candidate.parts.index("reports")
        nested = (report_root / Path(*candidate.parts[index + 1 :])).resolve()
        if nested.is_relative_to(report_root) and nested.is_file():
            return nested
    local = (report_root / candidate.name).resolve()
    if local.is_relative_to(report_root) and local.is_file():
        return local
    matches = tuple(path.resolve() for path in report_root.glob(f"**/{candidate.name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve one historical report: {raw}")
    if not matches[0].is_relative_to(report_root):  # pragma: no cover - defensive
        raise ValueError(
            f"resolved historical report escapes source tree: {matches[0]}"
        )
    return matches[0]


def _as_records(frame: Any) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    return [dict(row) for row in clean.to_dict(orient="records")]


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    by_repeat = {
        f"r{index:02d}": {"value": float(value)}
        for index, value in enumerate(values, start=1)
    }
    summary = summarize_repeats(by_repeat)["value"]
    lower, upper = summary["ci95"]
    return {
        "n_repeats": int(summary["n"]),
        "mean": float(summary["mean"]),
        "sample_sd": float(summary["sd"]),
        "repeat_t_ci95_low": float(lower),
        "repeat_t_ci95_high": float(upper),
        "repeat_t_ci95_method": "two_sided_student_t_0.95",
    }


def _roster_signature(report: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for fold in report.get("folds", ()):
        subjects = fold.get("test_subjects", fold.get("val_subjects", ()))
        rows.append(
            {
                "fold": int(fold.get("fold", len(rows) + 1)),
                "participants": tuple(sorted(str(value) for value in subjects)),
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), rows


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.zeros(len(order), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def _early_exploratory_tests(repeats: Any) -> Any:
    pandas = _pd()
    rows: list[dict[str, Any]] = []
    models = sorted(repeats["model_display"].unique())
    for metric in ("subject_balanced_accuracy", "subject_macro_f1"):
        metric_rows: list[dict[str, Any]] = []
        for reference, candidate in itertools.combinations(models, 2):
            left = repeats.loc[
                repeats["model_display"].eq(reference), ["seed", metric]
            ].set_index("seed")
            right = repeats.loc[
                repeats["model_display"].eq(candidate), ["seed", metric]
            ].set_index("seed")
            if set(left.index) != set(right.index):
                raise ValueError("repeat-level paired comparison seed keys differ")
            delta = np.asarray(
                [
                    right.loc[seed, metric] - left.loc[seed, metric]
                    for seed in sorted(left.index)
                ],
                dtype=np.float64,
            )
            observed = abs(float(delta.mean()))
            null = np.asarray(
                [
                    float(np.mean(delta * np.asarray(signs, dtype=np.float64)))
                    for signs in itertools.product((-1.0, 1.0), repeat=len(delta))
                ]
            )
            raw_p = float(np.mean(np.abs(null) >= observed - 1e-15))
            metric_rows.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "candidate": candidate,
                    "candidate_minus_reference_mean_delta": float(delta.mean()),
                    "n_paired_repeat_seeds": len(delta),
                    "exact_sign_patterns": int(2 ** len(delta)),
                    "raw_two_sided_exact_sign_flip_p": raw_p,
                    "exchange_unit": "aggregate_repeat_seed_not_participant",
                    "formal_v2_inference": False,
                    "interpretation": "exploratory_only_selection_contaminated_legacy_cv",
                }
            )
        adjusted = _holm_adjust(
            [row["raw_two_sided_exact_sign_flip_p"] for row in metric_rows]
        )
        for row, value in zip(metric_rows, adjusted):
            row["holm_adjusted_p_within_metric_three_pairs"] = value
        rows.extend(metric_rows)
    return pandas.DataFrame(rows)


def _early_pairwise_repeat_deltas(repeats: Any) -> tuple[Any, Any]:
    """Materialize every historical model pair without inventing OOF CIs."""

    pandas = _pd()
    delta_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    models = sorted(repeats["model_display"].unique())
    for reference, candidate in itertools.combinations(models, 2):
        left = repeats.loc[repeats["model_display"].eq(reference)].set_index("seed")
        right = repeats.loc[repeats["model_display"].eq(candidate)].set_index("seed")
        if set(left.index) != set(right.index):
            raise ValueError("historical pair lacks the exact matched seed roster")
        comparison_id = f"{candidate}_vs_{reference}"
        for seed in sorted(left.index):
            reference_ba = float(left.loc[seed, "subject_balanced_accuracy"])
            candidate_ba = float(right.loc[seed, "subject_balanced_accuracy"])
            reference_f1 = float(left.loc[seed, "subject_macro_f1"])
            candidate_f1 = float(right.loc[seed, "subject_macro_f1"])
            delta_rows.append(
                {
                    "comparison_family": "historical_matched_three_model_all_pairs",
                    "comparison_id": comparison_id,
                    "comparison_role": "exploratory_legacy_model_comparison_not_ablation",
                    "reference_case_id": reference,
                    "candidate_case_id": candidate,
                    "repeat": int(left.loc[seed, "repeat"]),
                    "split_seed": int(seed),
                    "comparison_contract_status": "matched_aggregate_repeat_seed",
                    "difference_direction": "candidate_minus_reference",
                    "reference_balanced_accuracy": reference_ba,
                    "candidate_balanced_accuracy": candidate_ba,
                    "balanced_accuracy_delta": candidate_ba - reference_ba,
                    "reference_macro_f1": reference_f1,
                    "candidate_macro_f1": candidate_f1,
                    "macro_f1_delta": candidate_f1 - reference_f1,
                    "reference_macro_roc_auc_ovr": None,
                    "candidate_macro_roc_auc_ovr": None,
                    "macro_roc_auc_ovr_delta": None,
                    "macro_roc_auc_ovr_applicability": (
                        "N/A_probability_level_participant_oof_not_archived"
                    ),
                    "automatic_selection": False,
                }
            )
        for metric in (
            "balanced_accuracy",
            "macro_f1",
            "macro_roc_auc_ovr",
        ):
            inference_rows.append(
                {
                    "comparison_family": "historical_matched_three_model_all_pairs",
                    "comparison_id": comparison_id,
                    "reference_case_id": reference,
                    "candidate_case_id": candidate,
                    "metric": metric,
                    "candidate_minus_reference": None,
                    "participant_cluster_delta_ci95_low": None,
                    "participant_cluster_delta_ci95_high": None,
                    "raw_two_sided_p_value": None,
                    "holm_adjusted_p_value": None,
                    "comparison_contract_status": (
                        "N/A_participant_probability_oof_not_archived"
                    ),
                    "p_value_applicability": (
                        "N/A_participant_probability_oof_not_archived"
                    ),
                    "inference_role": "exploratory_legacy_not_formal_v2_inference",
                    "automatic_selection": False,
                }
            )
    return pandas.DataFrame(delta_rows), pandas.DataFrame(inference_rows)


def _summarise_groups(frame: Any, group_columns: Sequence[str]) -> Any:
    pandas = _pd()
    rows: list[dict[str, Any]] = []
    metrics = tuple(metric for metric in _PERFORMANCE_METRICS if metric in frame)
    for key, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values))
        row["n_runs"] = int(len(group))
        row["seeds"] = ",".join(
            str(int(value)) for value in sorted(group["seed"].unique())
        )
        for metric in metrics:
            stats = _metric_summary(group[metric].astype(float).tolist())
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        if "duration_sec" in group:
            stats = _metric_summary(group["duration_sec"].astype(float).tolist())
            for name, value in stats.items():
                row[f"duration_sec_{name}"] = value
        rows.append(row)
    return pandas.DataFrame(rows)


def _finite_archived_value(value: Any) -> float | int | None:
    """Return one archived numeric value without converting missing cells to zero."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _archived_per_class_tables(
    runs: Any,
    *,
    source_study: str,
    source: Path | None = None,
) -> tuple[Any, Any]:
    """Expose every archived config/repeat/class and its five-repeat summary.

    The root CSV contains precision, recall, F1 and support.  When ``source`` is
    supplied, the persisted 3x3 subject confusion matrix is also loaded from the
    run JSON, which recovers TP/FP/FN/TN, specificity and one-vs-rest BA without
    inventing participant rows.  ROC-AUC and PR-AUC still require continuous OOF
    probabilities and therefore remain explicit ``None``.
    """

    pandas = _pd()
    repeat_rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        config_id = str(run["overfit_config_id"])
        confusion: np.ndarray | None = None
        if source is not None:
            report = _json(_report_path(source, run["report_path"]))
            raw_confusion = np.asarray(
                report.get("subject_confusion_matrix", ()), dtype=np.int64
            )
            if raw_confusion.shape != (len(_ARCHIVED_CLASS_FIELDS),) * 2:
                raise ValueError(
                    f"{source_study}/{config_id}/seed={run['seed']} has an invalid "
                    "subject confusion matrix"
                )
            confusion = raw_confusion
        common = {
            "source_study": source_study,
            "classifier_id": f"{source_study}::{config_id}",
            "case_id": config_id,
            "config_name": run.get("overfit_config_name"),
            "model": run.get("model"),
            "resolved_model": run.get("resolved_model"),
            "repeat": int(run["repeat"]),
            "split_seed": int(run["seed"]),
            "metric_scope": "participant_hard_label_archived_repeat",
        }
        for class_index, (prefix, class_label) in enumerate(_ARCHIVED_CLASS_FIELDS):
            archived: dict[str, float | int | None] = {
                "precision": _finite_archived_value(run.get(f"{prefix}_precision")),
                "sensitivity": _finite_archived_value(run.get(f"{prefix}_recall")),
                "f1": _finite_archived_value(run.get(f"{prefix}_f1")),
                "support": _finite_archived_value(run.get(f"{prefix}_support")),
            }
            missing_fields = tuple(
                name for name, value in archived.items() if value is None
            )
            counts: dict[str, float | int | None]
            if confusion is None:
                counts = {
                    "true_positive": None,
                    "false_positive": None,
                    "false_negative": None,
                    "true_negative": None,
                    "predicted_support": None,
                    "specificity": None,
                    "balanced_accuracy": None,
                }
                confusion_status = "N/A_source_report_json_not_supplied"
            else:
                true_positive = int(confusion[class_index, class_index])
                false_negative = int(confusion[class_index, :].sum() - true_positive)
                false_positive = int(confusion[:, class_index].sum() - true_positive)
                true_negative = int(
                    confusion.sum() - true_positive - false_negative - false_positive
                )
                confusion_sensitivity = (
                    true_positive / (true_positive + false_negative)
                    if true_positive + false_negative
                    else None
                )
                if (
                    archived["support"] is not None
                    and int(archived["support"]) != true_positive + false_negative
                ):
                    raise ValueError(
                        f"{source_study}/{config_id}/seed={run['seed']}/{class_label} "
                        "support disagrees with the archived subject confusion matrix"
                    )
                if (
                    archived["sensitivity"] is not None
                    and confusion_sensitivity is not None
                    and not math.isclose(
                        float(archived["sensitivity"]),
                        float(confusion_sensitivity),
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"{source_study}/{config_id}/seed={run['seed']}/{class_label} "
                        "sensitivity disagrees with the archived subject confusion matrix"
                    )
                specificity = (
                    true_negative / (true_negative + false_positive)
                    if true_negative + false_positive
                    else None
                )
                sensitivity = confusion_sensitivity
                balanced_accuracy = (
                    0.5 * (float(sensitivity) + float(specificity))
                    if sensitivity is not None and specificity is not None
                    else None
                )
                counts = {
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_negative": true_negative,
                    "predicted_support": true_positive + false_positive,
                    "specificity": specificity,
                    "balanced_accuracy": balanced_accuracy,
                }
                confusion_status = "available_archived_subject_confusion_matrix"
            row = {
                **common,
                "class_label": class_label,
                **archived,
                **counts,
                "roc_auc": None,
                "pr_auc": None,
                "result_applicability": (
                    "available_archived_hard_label_per_class_fields"
                    if not missing_fields
                    else "N/A_one_or_more_archived_per_class_fields_missing"
                ),
                "missing_archived_fields": ",".join(missing_fields),
                "precision_applicability": (
                    "available_archived_field"
                    if archived["precision"] is not None
                    else f"N/A_archived_field_missing:{prefix}_precision"
                ),
                "sensitivity_applicability": (
                    "available_archived_field"
                    if archived["sensitivity"] is not None
                    else f"N/A_archived_field_missing:{prefix}_recall"
                ),
                "f1_applicability": (
                    "available_archived_field"
                    if archived["f1"] is not None
                    else f"N/A_archived_field_missing:{prefix}_f1"
                ),
                "support_applicability": (
                    "available_archived_field"
                    if archived["support"] is not None
                    else f"N/A_archived_field_missing:{prefix}_support"
                ),
                "confusion_count_applicability": confusion_status,
                "specificity_applicability": confusion_status,
                "balanced_accuracy_applicability": confusion_status,
                "roc_auc_applicability": (
                    "N/A_probability_level_participant_oof_not_archived"
                ),
                "pr_auc_applicability": (
                    "N/A_probability_level_participant_oof_not_archived"
                ),
            }
            repeat_rows.append(row)
    repeat_frame = pandas.DataFrame(repeat_rows)

    summary_rows: list[dict[str, Any]] = []
    group_columns = (
        "source_study",
        "classifier_id",
        "case_id",
        "config_name",
        "model",
        "resolved_model",
        "class_label",
        "metric_scope",
    )
    for key, group in repeat_frame.groupby(
        list(group_columns), dropna=False, sort=False
    ):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        row["expected_repeat_count"] = len(_SPLIT_SEEDS)
        row["split_seeds"] = ",".join(
            str(value) for value in sorted(group["split_seed"])
        )
        for metric in (
            "precision",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "support",
            "predicted_support",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ):
            values = [
                float(value)
                for value in group[metric].tolist()
                if _finite_archived_value(value) is not None
            ]
            row[f"{metric}_n_repeats_available"] = len(values)
            if values:
                stats = _metric_summary(values)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_sample_sd"] = stats["sample_sd"]
                row[f"{metric}_repeat_t_ci95_low"] = stats["repeat_t_ci95_low"]
                row[f"{metric}_repeat_t_ci95_high"] = stats["repeat_t_ci95_high"]
                row[f"{metric}_repeat_t_ci95_method"] = stats["repeat_t_ci95_method"]
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_sample_sd"] = None
                row[f"{metric}_repeat_t_ci95_low"] = None
                row[f"{metric}_repeat_t_ci95_high"] = None
                row[f"{metric}_repeat_t_ci95_method"] = None
            row[f"{metric}_applicability"] = (
                "available_all_five_archived_repeats"
                if len(values) == len(_SPLIT_SEEDS)
                else (
                    "N/A_archived_field_missing_all_repeats"
                    if not values
                    else "partial_archive_one_or_more_repeat_fields_missing"
                )
            )
        row.update(
            {
                "roc_auc": None,
                "pr_auc": None,
                "confusion_count_applicability": (
                    "available_all_five_archived_subject_confusion_matrices"
                    if source is not None
                    else "N/A_source_report_json_not_supplied"
                ),
                "specificity_applicability": (
                    "available_all_five_archived_subject_confusion_matrices"
                    if source is not None
                    else "N/A_source_report_json_not_supplied"
                ),
                "balanced_accuracy_applicability": (
                    "available_all_five_archived_subject_confusion_matrices"
                    if source is not None
                    else "N/A_source_report_json_not_supplied"
                ),
                "roc_auc_applicability": "N/A_probability_level_participant_oof_not_archived",
                "pr_auc_applicability": "N/A_probability_level_participant_oof_not_archived",
            }
        )
        summary_rows.append(row)
    return repeat_frame, pandas.DataFrame(summary_rows)


def _factor_signal_pair_tables(
    runs: Any,
    effects: Any,
    *,
    source_study: str,
) -> tuple[Any, Any]:
    """Materialize every declared factor-signal pair without pseudo-cluster CI."""

    pandas = _pd()
    delta_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for effect_index, effect in effects.reset_index(drop=True).iterrows():
        reference_id = str(effect["baseline_config_id"])
        candidate_id = str(effect["best_observed_config_id"])
        reference = runs.loc[runs["overfit_config_id"].astype(str).eq(reference_id)]
        candidate = runs.loc[runs["overfit_config_id"].astype(str).eq(candidate_id)]
        if (
            reference.groupby("seed").size().max() != 1
            or candidate.groupby("seed").size().max() != 1
        ):
            raise ValueError(
                f"{source_study} factor pair contains duplicate config/seed rows"
            )
        reference_by_seed = reference.set_index("seed")
        candidate_by_seed = candidate.set_index("seed")
        reference_seeds = tuple(sorted(int(value) for value in reference_by_seed.index))
        candidate_seeds = tuple(sorted(int(value) for value in candidate_by_seed.index))
        if reference_seeds != _SPLIT_SEEDS or candidate_seeds != _SPLIT_SEEDS:
            raise ValueError(
                f"{source_study} factor pair does not have the frozen five-seed roster"
            )
        epoch = effect.get("epoch")
        factor = str(effect.get("factor"))
        comparison_id = (
            f"{source_study}:factor={factor}:epoch={epoch}:"
            f"{candidate_id}_vs_{reference_id}:signal={effect_index + 1}"
        )
        paired_ba_deltas: list[float] = []
        paired_f1_deltas: list[float] = []
        for seed in _SPLIT_SEEDS:
            reference_row = reference_by_seed.loc[seed]
            candidate_row = candidate_by_seed.loc[seed]
            reference_repeat = int(reference_row["repeat"])
            candidate_repeat = int(candidate_row["repeat"])
            if reference_repeat != candidate_repeat:
                raise ValueError(
                    f"{source_study} factor pair repeat IDs differ at split seed {seed}"
                )
            reference_ba = float(reference_row["subject_balanced_accuracy"])
            candidate_ba = float(candidate_row["subject_balanced_accuracy"])
            reference_f1 = float(reference_row["subject_macro_f1"])
            candidate_f1 = float(candidate_row["subject_macro_f1"])
            paired_ba_deltas.append(candidate_ba - reference_ba)
            paired_f1_deltas.append(candidate_f1 - reference_f1)
            delta_rows.append(
                {
                    "source_study": source_study,
                    "comparison_family": f"{source_study}_factor_signals",
                    "comparison_id": comparison_id,
                    "comparison_role": "exploratory_post_hoc_factor_signal_not_ablation",
                    "factor": factor,
                    "epoch": epoch,
                    "reference_case_id": reference_id,
                    "candidate_case_id": candidate_id,
                    "repeat": reference_repeat,
                    "split_seed": int(seed),
                    "comparison_contract_status": "matched_aggregate_repeat_seed",
                    "difference_direction": "candidate_minus_reference",
                    "reference_balanced_accuracy": reference_ba,
                    "candidate_balanced_accuracy": candidate_ba,
                    "balanced_accuracy_delta": candidate_ba - reference_ba,
                    "reference_macro_f1": reference_f1,
                    "candidate_macro_f1": candidate_f1,
                    "macro_f1_delta": candidate_f1 - reference_f1,
                    "reference_macro_roc_auc_ovr": None,
                    "candidate_macro_roc_auc_ovr": None,
                    "macro_roc_auc_ovr_delta": None,
                    "macro_roc_auc_ovr_applicability": (
                        "N/A_probability_level_participant_oof_not_archived"
                    ),
                    "macro_roc_auc_ovr_unavailability_reason": (
                        _PARTICIPANT_ROC_UNAVAILABLE_REASON
                    ),
                    "automatic_selection": False,
                }
            )
        mean_deltas = {
            "balanced_accuracy": float(np.mean(paired_ba_deltas)),
            "macro_f1": float(np.mean(paired_f1_deltas)),
            "macro_roc_auc_ovr": None,
        }
        for metric, point_delta in mean_deltas.items():
            inference_rows.append(
                {
                    "source_study": source_study,
                    "comparison_family": f"{source_study}_factor_signals",
                    "comparison_id": comparison_id,
                    "comparison_role": "exploratory_post_hoc_factor_signal_not_ablation",
                    "factor": factor,
                    "epoch": epoch,
                    "reference_case_id": reference_id,
                    "candidate_case_id": candidate_id,
                    "metric": metric,
                    "candidate_minus_reference": point_delta,
                    "point_delta_source": (
                        "mean_of_five_matched_aggregate_repeat_deltas"
                        if point_delta is not None
                        else "N/A_probability_level_participant_oof_not_archived"
                    ),
                    "participant_cluster_delta_ci95_low": None,
                    "participant_cluster_delta_ci95_high": None,
                    "participant_cluster_ci_applicability": (
                        "N/A_participant_level_oof_rows_not_archived"
                    ),
                    "participant_cluster_ci_unavailability_reason": (
                        _PARTICIPANT_CLUSTER_CI_UNAVAILABLE_REASON
                    ),
                    "raw_two_sided_p_value": None,
                    "holm_adjusted_p_value": None,
                    "p_value_applicability": (
                        "N/A_matched_participant_level_predictions_not_archived"
                    ),
                    "inference_role": "exploratory_legacy_not_formal_v2_inference",
                    "automatic_selection": False,
                }
            )
    return pandas.DataFrame(delta_rows), pandas.DataFrame(inference_rows)


def _absolute_historical_cluster_ci_rows(
    summary: Any,
    *,
    source_study: str,
    classifier_id_column: str,
    model_column: str,
) -> Any:
    """Record absolute participant-cluster CI applicability for every classifier."""

    pandas = _pd()
    rows: list[dict[str, Any]] = []
    for _, classifier in summary.iterrows():
        classifier_id = str(classifier[classifier_id_column])
        points = {
            "balanced_accuracy": _finite_archived_value(
                classifier.get("subject_balanced_accuracy_mean")
            ),
            "macro_f1": _finite_archived_value(classifier.get("subject_macro_f1_mean")),
            "macro_roc_auc_ovr": None,
        }
        for metric, point in points.items():
            rows.append(
                {
                    "source_study": source_study,
                    "classifier_id": f"{source_study}::{classifier_id}",
                    "case_id": classifier_id,
                    "model": classifier.get(model_column),
                    "metric": metric,
                    "point_estimate": point,
                    "point_estimate_source": (
                        "mean_of_five_archived_repeat_summaries"
                        if point is not None
                        else "N/A_probability_level_participant_oof_not_archived"
                    ),
                    "participant_cluster_ci95_low": None,
                    "participant_cluster_ci95_high": None,
                    "participant_cluster_ci_applicability": (
                        "N/A_participant_level_oof_rows_not_archived"
                    ),
                    "participant_cluster_ci_unavailability_reason": (
                        _PARTICIPANT_CLUSTER_CI_UNAVAILABLE_REASON
                    ),
                    "formal_v2_inference": False,
                }
            )
    return pandas.DataFrame(rows)


def _early_contract_rows(
    early_source: Path,
    shape_source: Path,
    *,
    window_seconds: float,
    overlap_percent: float,
    patience: int,
    extra_input: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    pandas = _pd()
    frames = []
    for source in (early_source, shape_source):
        frame = pandas.read_csv(source / "sweep_runs.csv")
        numeric_extra = pandas.to_numeric(frame["extra_input"], errors="coerce")
        target_extra = (
            float(extra_input)
            if str(extra_input).replace(".", "", 1).isdigit()
            else math.nan
        )
        if math.isfinite(target_extra):
            extra_mask = numeric_extra.eq(target_extra)
        else:
            extra_mask = frame["extra_input"].astype(str).eq(str(extra_input))
        selected = frame[
            pandas.to_numeric(frame["window_sec"], errors="coerce").eq(window_seconds)
            & pandas.to_numeric(frame["overlap_pct"], errors="coerce").eq(
                overlap_percent
            )
            & pandas.to_numeric(frame["cnn_patience"], errors="coerce").eq(patience)
            & extra_mask
            & frame["status"].astype(str).str.lower().eq("ok")
        ].copy()
        selected["source_study"] = source.name
        selected["source_path"] = str(source)
        frames.append(selected)
    repeats = pandas.concat(frames, ignore_index=True)
    display = {
        "cnn": "CNN1D",
        "inceptiontime": "InceptionTime",
        "shapeformer_pisd": "ShapeFormer-PISD",
    }
    repeats["model_display"] = repeats["model"].map(display).fillna(repeats["model"])
    expected = set(display.values())
    if set(repeats["model_display"]) != expected or len(repeats) != 15:
        raise ValueError(
            "the matched three-model contract must resolve to 3 models x 5 repeats"
        )
    seed_sets = {
        tuple(sorted(group["seed"].astype(int).tolist()))
        for _, group in repeats.groupby("model_display")
    }
    if seed_sets != {_SPLIT_SEEDS}:
        raise ValueError(
            "matched historical models do not share the frozen five-seed schedule"
        )

    summary = _summarise_groups(repeats, ("model_display", "model", "resolved_model"))
    summary = summary.sort_values(
        ["subject_balanced_accuracy_mean", "subject_macro_f1_mean"],
        ascending=False,
    ).reset_index(drop=True)
    summary.insert(0, "descriptive_rank", np.arange(1, len(summary) + 1))
    summary["ranking_scope"] = "post_hoc_matched_historical_candidate_generation"
    summary["formal_v2_selection_eligible"] = False

    parameter_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    extra_description = (
        "no PPI/HRV extra input"
        if str(extra_input).strip() in {"0", "0.0"}
        else f"archived extra_input={extra_input}"
    )
    for model_name, group in repeats.groupby("model_display", sort=False):
        configs: list[dict[str, Any]] = []
        for _, run in group.sort_values("repeat").iterrows():
            report = _json(_report_path(Path(run["source_path"]), run["report_path"]))
            config = dict(report.get("config", {}))
            configs.append(
                {key: value for key, value in config.items() if key != "seed"}
            )
            roster_sha256, roster = _roster_signature(report)
            split_rows.append(
                {
                    "model": model_name,
                    "repeat": int(run["repeat"]),
                    "split_seed": int(run["seed"]),
                    "fold_count": len(roster),
                    "held_out_participant_count": len(
                        {
                            participant
                            for fold in roster
                            for participant in fold["participants"]
                        }
                    ),
                    "fold_sizes": ",".join(
                        str(len(fold["participants"])) for fold in roster
                    ),
                    "fold_roster_sha256": roster_sha256,
                    "roster": json.dumps(roster, ensure_ascii=False),
                }
            )
            subject_report = report.get("subject_classification_report", {})
            class_names = tuple(str(value) for value in report.get("class_names", ()))
            confusion = np.asarray(
                report.get("subject_confusion_matrix", ()), dtype=np.int64
            )
            if confusion.shape != (len(class_names), len(class_names)):
                raise ValueError(
                    f"{model_name}/seed={run['seed']} has an invalid subject confusion matrix"
                )
            for class_index, class_name in enumerate(class_names):
                metrics = subject_report.get(class_name, {})
                true_positive = int(confusion[class_index, class_index])
                false_negative = int(confusion[class_index, :].sum() - true_positive)
                false_positive = int(confusion[:, class_index].sum() - true_positive)
                true_negative = int(
                    confusion.sum() - true_positive - false_negative - false_positive
                )
                specificity = (
                    true_negative / (true_negative + false_positive)
                    if true_negative + false_positive
                    else None
                )
                sensitivity = metrics.get("recall")
                class_rows.append(
                    {
                        "model": model_name,
                        "repeat": int(run["repeat"]),
                        "seed": int(run["seed"]),
                        "class": class_name,
                        "precision": metrics.get("precision"),
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                        "balanced_accuracy": (
                            0.5 * (float(sensitivity) + float(specificity))
                            if sensitivity is not None and specificity is not None
                            else None
                        ),
                        "f1": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                        "predicted_support": true_positive + false_positive,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                        "true_negative": true_negative,
                        "roc_auc": None,
                        "pr_auc": None,
                        "roc_auc_applicability": (
                            "N/A_probability_level_participant_oof_not_archived"
                        ),
                        "pr_auc_applicability": (
                            "N/A_probability_level_participant_oof_not_archived"
                        ),
                    }
                )
        canonical = json.dumps(configs[0], sort_keys=True, default=str)
        if any(
            json.dumps(item, sort_keys=True, default=str) != canonical
            for item in configs[1:]
        ):
            raise ValueError(f"fixed parameters drift across repeats for {model_name}")
        first = group.iloc[0]
        common = {
            "input_data": "29-participant internal dataset; roles B,R1-R4; "
            f"8 raw channels; {extra_description}",
            "archived_window_count": first.get("n_windows"),
            "seed_schedule": ",".join(
                str(value) for value in sorted(group["seed"].astype(int))
            ),
            "split_scheme": "participant-grouped 5-fold within each repeat",
            "selection_rule": "best epoch from the historical validation trajectory; "
            f"patience {patience}",
        }
        for name, value in {**common, **configs[0]}.items():
            parameter_rows.append(
                {
                    "study": "matched_three_model_historical_comparison",
                    "model_or_module": model_name,
                    "parameter": name,
                    "value": json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value,
                    "provenance": "historical_report_json"
                    if name in configs[0]
                    else "derived_from_archived_run_contract",
                }
            )
    class_repeat = pandas.DataFrame(class_rows)
    class_summary_rows: list[dict[str, Any]] = []
    for (model_name, class_name), group in class_repeat.groupby(
        ["model", "class"], sort=False
    ):
        row: dict[str, Any] = {"model": model_name, "class": class_name}
        for metric in (
            "precision",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "support",
            "predicted_support",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ):
            stats = _metric_summary(group[metric].astype(float).tolist())
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        row["roc_auc"] = None
        row["pr_auc"] = None
        row[
            "roc_auc_applicability"
        ] = "N/A_probability_level_participant_oof_not_archived"
        row[
            "pr_auc_applicability"
        ] = "N/A_probability_level_participant_oof_not_archived"
        class_summary_rows.append(row)
    splits = pandas.DataFrame(split_rows)
    if splits.groupby("split_seed")["fold_roster_sha256"].nunique().max() != 1:
        raise ValueError(
            "matched historical models do not share identical participant folds"
        )
    if (
        not splits["fold_count"].eq(5).all()
        or not splits["held_out_participant_count"].eq(29).all()
    ):
        raise ValueError(
            "matched historical split registry is not 5-fold/29-participant"
        )
    paired_tests = _early_exploratory_tests(repeats)
    return (
        repeats,
        summary,
        pandas.DataFrame(parameter_rows),
        pandas.DataFrame(class_summary_rows),
        splits,
        paired_tests,
    )


def _overfit_summary(
    source: Path,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    pandas = _pd()
    runs = pandas.read_csv(source / "overfitting_runs.csv")
    seed_schedules = {
        tuple(sorted(group["seed"].astype(int).tolist()))
        for _, group in runs.groupby("overfit_config_id")
    }
    if seed_schedules != {_SPLIT_SEEDS}:
        raise ValueError(
            f"{source.name} does not contain exactly five frozen seeds per config"
        )
    group_columns = [
        column
        for column in (
            "overfit_config_id",
            "overfit_config_name",
            "model",
            "resolved_model",
            "overfit_stage",
            "stage1_screen_group",
            "stage1_regularization_factor",
            "stage1_regularization_value",
            "cnn_epochs",
            "cnn_lr",
            "cnn_weight_decay",
            "cnn_dropout",
            "cnn_label_smoothing",
            "max_windows_fraction",
            "sqi_mode",
            "aggregation",
            "manual_features",
            "loss_type",
            "class_weight_mode",
        )
        if column in runs
    ]
    summary = _summarise_groups(runs, group_columns)
    summary = summary.sort_values(
        ["subject_balanced_accuracy_mean", "subject_macro_f1_mean"],
        ascending=False,
    ).reset_index(drop=True)
    summary.insert(0, "descriptive_rank", np.arange(1, len(summary) + 1))
    summary["source_study"] = source.name
    summary["formal_v2_selection_eligible"] = False
    top = summary.head(15).copy()

    effect_rows: list[dict[str, Any]] = []
    for epoch, epoch_rows in summary.groupby("cnn_epochs", dropna=False):
        baseline = epoch_rows[
            epoch_rows["stage1_regularization_factor"].astype(str).eq("baseline")
        ]
        if baseline.empty:
            continue
        base = baseline.sort_values(
            "subject_balanced_accuracy_mean", ascending=False
        ).iloc[0]
        for factor, candidates in epoch_rows.groupby(
            "stage1_regularization_factor", dropna=False
        ):
            if str(factor) in {"baseline", "reference", "nan"}:
                continue
            candidate = candidates.sort_values(
                ["subject_balanced_accuracy_mean", "subject_macro_f1_mean"],
                ascending=False,
            ).iloc[0]
            effect_rows.append(
                {
                    "source_study": source.name,
                    "epoch": epoch,
                    "factor": factor,
                    "best_observed_value": candidate.get("stage1_regularization_value"),
                    "best_observed_config_id": candidate.get("overfit_config_id"),
                    "baseline_config_id": base.get("overfit_config_id"),
                    "baseline_subject_ba_mean": base["subject_balanced_accuracy_mean"],
                    "candidate_subject_ba_mean": candidate[
                        "subject_balanced_accuracy_mean"
                    ],
                    "descriptive_delta_ba": candidate["subject_balanced_accuracy_mean"]
                    - base["subject_balanced_accuracy_mean"],
                    "candidate_subject_ba_repeat_t_ci95_low": candidate[
                        "subject_balanced_accuracy_repeat_t_ci95_low"
                    ],
                    "candidate_subject_ba_repeat_t_ci95_high": candidate[
                        "subject_balanced_accuracy_repeat_t_ci95_high"
                    ],
                    "interpretation": "best_observed_post_hoc_factor_signal_not_confirmatory",
                }
            )
    effects = pandas.DataFrame(effect_rows)

    varied_rows: list[dict[str, Any]] = []
    for column in _CONFIG_COLUMNS:
        if column not in runs:
            continue
        values = [value for value in runs[column].dropna().unique().tolist()]
        varied_rows.append(
            {
                "source_study": source.name,
                "parameter": column,
                "unique_count": len(values),
                "values": json.dumps(
                    sorted(values, key=lambda value: str(value)), ensure_ascii=False
                ),
                "role": "varied" if len(values) > 1 else "fixed",
            }
        )
    manifest = _json(source / "overfitting_manifest.json")
    if (
        int(manifest.get("repeats", -1)) != 5
        or int(manifest.get("cv_folds", -1)) != 5
        or tuple(int(seed) for seed in manifest.get("seeds", ())) != _SPLIT_SEEDS
    ):
        raise ValueError(
            f"{source.name} manifest does not match the frozen 5x5 registry"
        )
    first_run = runs.iloc[0]
    first_report = _json(_report_path(source, first_run["report_path"]))
    fixed_rows = []
    for name, value in first_report.get("config", {}).items():
        if name == "seed":
            continue
        fixed_rows.append(
            {
                "source_study": source.name,
                "row_type": "archived_first_report_parameter",
                "parameter": name,
                "unique_count": None,
                "values": json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value,
                "role": "fixed_or_candidate_specific_see_run_table",
            }
        )
    fixed_rows.extend(
        [
            {
                "source_study": source.name,
                "row_type": "source_contract",
                "parameter": "input_data",
                "values": (
                    "29 participants; 145 static B/R1-R4 files; source columns "
                    "RED/IR/AX/AY/AZ/GX/GY/GZ; historical DL preprocessing and "
                    "target sampling rate vary by source study"
                ),
                "role": "fixed",
            },
            {
                "source_study": source.name,
                "row_type": "source_contract",
                "parameter": "optimizer",
                "values": "AdamW (supporting generator-code evidence; absent from report JSON)",
                "role": "provenance_limited",
            },
            {
                "source_study": source.name,
                "row_type": "source_contract",
                "parameter": "confidence_scope",
                "values": "historical_hypothesis_generation_only",
                "role": "interpretation",
            },
        ]
    )
    for row in varied_rows:
        row["row_type"] = "observed_run_parameter_space"
    contracts = pandas.DataFrame(fixed_rows + varied_rows)

    split_rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        report = _json(_report_path(source, run["report_path"]))
        signature, roster = _roster_signature(report)
        split_rows.append(
            {
                "source_study": source.name,
                "config_id": run["overfit_config_id"],
                "repeat": int(run["repeat"]),
                "split_seed": int(run["seed"]),
                "fold_count": len(roster),
                "fold_sizes": ",".join(
                    str(len(fold["participants"])) for fold in roster
                ),
                "held_out_participant_count": len(
                    {
                        participant
                        for fold in roster
                        for participant in fold["participants"]
                    }
                ),
                "fold_roster_sha256": signature,
            }
        )
    split_audit = pandas.DataFrame(split_rows)
    consistency = split_audit.groupby("split_seed")["fold_roster_sha256"].nunique()
    if consistency.max() != 1:
        raise ValueError(f"participant fold roster drift detected in {source.name}")
    split_audit = split_audit.groupby(
        ["source_study", "repeat", "split_seed"], as_index=False
    ).agg(
        config_count=("config_id", "nunique"),
        fold_count=("fold_count", "first"),
        fold_sizes=("fold_sizes", "first"),
        held_out_participant_count=("held_out_participant_count", "first"),
        fold_roster_sha256=("fold_roster_sha256", "first"),
    )
    split_audit["all_configs_share_roster"] = True
    if (
        not split_audit["fold_count"].eq(5).all()
        or not split_audit["held_out_participant_count"].eq(29).all()
    ):
        raise ValueError(f"{source.name} split registry is not 5-fold/29-participant")
    per_class_repeats, per_class_summary = _archived_per_class_tables(
        runs,
        source_study=source.name,
        source=source,
    )
    factor_repeat_deltas, factor_cluster_inference = _factor_signal_pair_tables(
        runs,
        effects,
        source_study=source.name,
    )
    return (
        summary,
        top,
        effects,
        contracts,
        split_audit,
        per_class_repeats,
        per_class_summary,
        factor_repeat_deltas,
        factor_cluster_inference,
    )


def _cross_study_split_audit(
    early: Any,
    fixed_epoch: Any,
    extension: Any,
) -> Any:
    pandas = _pd()
    sources = {
        "early_architecture": early,
        "fixed_epoch": fixed_epoch,
        "sqi_loss_feature_extension": extension,
    }
    rows: list[dict[str, Any]] = []
    for seed in _SPLIT_SEEDS:
        signatures: dict[str, str] = {}
        for label, frame in sources.items():
            values = frame.loc[
                frame["split_seed"].astype(int).eq(seed), "fold_roster_sha256"
            ].drop_duplicates()
            if len(values) != 1:
                raise ValueError(
                    f"{label} has an ambiguous roster for split seed {seed}"
                )
            signatures[label] = str(values.iloc[0])
        all_match = len(set(signatures.values())) == 1
        if not all_match:
            raise ValueError(f"historical study roster drift detected for seed {seed}")
        rows.append(
            {
                "split_seed": seed,
                **{
                    f"{label}_roster_sha256": value
                    for label, value in signatures.items()
                },
                "all_four_source_studies_share_roster": all_match,
                "interpretation": "same_split_registry_not_independent_evidence",
            }
        )
    return pandas.DataFrame(rows)


def _missing_statistics() -> list[dict[str, Any]]:
    return [
        {
            "requested_output": "participant-cluster bootstrap 95% CI",
            "status": "N/A",
            "reason": "participant-level OOF probabilities and stable participant keys were not archived",
            "required_source_fields": "participant_id, repeat, true_label, class_probability_vector",
            "v2_action": "do not substitute aggregate-confusion or fold-level pseudo-replicates",
        },
        {
            "requested_output": "paired participant-exchange permutation P and Holm adjustment",
            "status": "N/A",
            "reason": "matched participant-level predictions for candidate and reference are absent",
            "required_source_fields": "identical participant/repeat OOF keys for every compared model",
            "v2_action": "treat historical rankings as candidate/hypothesis generation",
        },
        {
            "requested_output": "ROC-AUC / PR-AUC and ROC/PR curves",
            "status": "N/A",
            "reason": "only hard-label confusion matrices and aggregate BA/F1 were archived",
            "required_source_fields": "true labels and continuous per-class OOF probabilities",
            "v2_action": "report only BA/F1/per-class hard-label metrics from the archive",
        },
        {
            "requested_output": "calibration curve, ECE, Brier score and score-distribution/t-SNE plots",
            "status": "N/A",
            "reason": "continuous OOF probabilities and embeddings were not archived",
            "required_source_fields": "per-participant probabilities; embeddings for t-SNE",
            "v2_action": "leave unavailable rather than infer from summary metrics",
        },
        {
            "requested_output": "selection-unbiased confirmatory P/CI for the best searched configuration",
            "status": "N/A",
            "reason": "the same historical resampling evidence was used to search and rank many configurations",
            "required_source_fields": "a new locked nested-CV or untouched confirmation study",
            "v2_action": "use V2 locked studies for confirmation",
        },
    ]


def _plot_early(
    repeats: Any,
    summary: Any,
    figures: Path,
    *,
    window_seconds: float,
    overlap_percent: float,
    patience: int,
) -> None:
    pyplot = _plt()
    order = summary["model_display"].tolist()
    fig, axes = pyplot.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("subject_balanced_accuracy", "subject_macro_f1"),
        ("Subject balanced accuracy", "Subject macro-F1"),
    ):
        groups = [
            repeats.loc[repeats["model_display"].eq(name), metric] * 100
            for name in order
        ]
        axis.boxplot(groups, labels=order, showmeans=True)
        for index, values in enumerate(groups, start=1):
            axis.scatter(np.full(len(values), index), values, s=22, alpha=0.7, zorder=3)
        axis.set_title(title)
        axis.set_ylabel("Score (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle(
        "Matched historical "
        f"{window_seconds:g} s / {overlap_percent:g}% overlap / patience {patience} comparison"
    )
    fig.tight_layout()
    fig.savefig(figures / "early_three_model_ba_f1_boxplots.png", dpi=180)
    pyplot.close(fig)

    fig, axis = pyplot.subplots(figsize=(7.5, 4.5))
    runtime = [
        repeats.loc[repeats["model_display"].eq(name), "duration_sec"] for name in order
    ]
    axis.boxplot(runtime, labels=order, showmeans=True)
    axis.set_yscale("log")
    axis.set_ylabel("Run duration (s, log scale)")
    axis.set_title("Historical computational burden (five archived repeats)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "early_three_model_runtime_boxplot.png", dpi=180)
    pyplot.close(fig)


def _plot_overfit(top: Any, effects: Any, figures: Path, prefix: str) -> None:
    pyplot = _plt()
    selected = top.head(12).iloc[::-1]
    fig, axis = pyplot.subplots(figsize=(10, 6.5))
    means = selected["subject_balanced_accuracy_mean"] * 100
    lower = selected["subject_balanced_accuracy_repeat_t_ci95_low"] * 100
    upper = selected["subject_balanced_accuracy_repeat_t_ci95_high"] * 100
    axis.barh(
        selected["overfit_config_id"],
        means,
        xerr=np.vstack((means - lower, upper - means)),
    )
    axis.set_xlabel("Subject balanced accuracy (%; repeat Student-t 95% CI)")
    axis.set_title(f"{prefix}: top observed configurations (exploratory)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}_top_config_ba_ci.png", dpi=180)
    pyplot.close(fig)

    if effects.empty:
        return
    plot = effects.sort_values(["epoch", "descriptive_delta_ba"])
    fig, axis = pyplot.subplots(figsize=(10, max(4.5, 0.28 * len(plot))))
    labels = [f"ep{row.epoch} · {row.factor}" for row in plot.itertuples()]
    colors = [
        "#2f7d32" if value >= 0 else "#b33a3a" for value in plot["descriptive_delta_ba"]
    ]
    axis.barh(labels, plot["descriptive_delta_ba"] * 100, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Best observed factor candidate − same-epoch baseline BA (pp)")
    axis.set_title(f"{prefix}: post-hoc factor signals, not confirmatory effects")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}_factor_signals.png", dpi=180)
    pyplot.close(fig)


def _consumed_report_paths(
    sources: Sequence[Path], early_repeats: Any
) -> tuple[Path, ...]:
    pandas = _pd()
    paths = {
        _report_path(Path(row.source_path), row.report_path)
        for row in early_repeats.itertuples()
    }
    for source in sources[2:]:
        runs = pandas.read_csv(source / "overfitting_runs.csv", usecols=["report_path"])
        paths.update(_report_path(source, raw) for raw in runs["report_path"])
    return tuple(sorted(paths, key=lambda path: str(path).encode("utf-8")))


def _source_rows(
    sources: Sequence[Path],
    consumed_reports: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for name in (
            "sweep_manifest.json",
            "sweep_runs.csv",
            "sweep_summary.csv",
            "overfitting_manifest.json",
            "overfitting_runs.csv",
            "overfitting_summary.csv",
        ):
            path = source / name
            if path.exists():
                rows.append(
                    {
                        "source_study": source.name,
                        "source_kind": "root_manifest_or_run_table",
                        "source_file": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    for path in consumed_reports:
        owner = next(
            source
            for source in sources
            if path.is_relative_to((source / "reports").resolve())
        )
        rows.append(
            {
                "source_study": owner.name,
                "source_kind": "consumed_report_json",
                "source_file": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if len(columns) > _MAX_HUMAN_TABLE_COLUMNS:
        raise ValueError(
            f"human-facing historical table has {len(columns)} columns; "
            f"maximum is {_MAX_HUMAN_TABLE_COLUMNS}"
        )
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if value is None:
                value = "N/A"
            elif isinstance(value, float):
                value = "N/A" if not math.isfinite(value) else f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(("", markdown_column_definitions_block(columns)))
    return "\n".join(lines)


def _unique_projection(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        projected = {column: row.get(column) for column in columns}
        identity = json.dumps(
            projected, ensure_ascii=False, sort_keys=True, default=str
        )
        if identity not in seen:
            seen.add(identity)
            output.append(projected)
    return output


def _historical_pairwise_display_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, list[dict[str, Any]], tuple[str, ...]], ...]:
    """Split wide historical repeat pairs into model-first semantic views."""

    contract_fields = (
        "candidate_case_id",
        "reference_case_id",
        "comparison_id",
        "comparison_family",
        "comparison_role",
        "comparison_contract_status",
        "difference_direction",
        "automatic_selection",
    )
    roster_fields = (
        "candidate_case_id",
        "reference_case_id",
        "comparison_id",
        "repeat",
        "split_seed",
        "comparison_contract_status",
        "automatic_selection",
    )
    metric_fields = (
        "candidate_case_id",
        "reference_case_id",
        "comparison_id",
        "repeat",
        "metric",
        "reference_value",
        "candidate_value",
        "candidate_minus_reference",
    )
    applicability_fields = (
        "candidate_case_id",
        "reference_case_id",
        "comparison_id",
        "repeat",
        "metric",
        "applicability",
        "difference_direction",
        "automatic_selection",
    )
    metrics = ("balanced_accuracy", "macro_f1", "macro_roc_auc_ovr")
    metric_rows: list[dict[str, Any]] = []
    applicability_rows: list[dict[str, Any]] = []
    for row in rows:
        for metric in metrics:
            reference = row.get(f"reference_{metric}")
            candidate = row.get(f"candidate_{metric}")
            delta = row.get(f"{metric}_delta")
            applicability = row.get(f"{metric}_applicability")
            if applicability is None:
                applicability = (
                    "available"
                    if any(value is not None for value in (reference, candidate, delta))
                    else "N/A_metric_not_available"
                )
            common = {
                "candidate_case_id": row.get("candidate_case_id"),
                "reference_case_id": row.get("reference_case_id"),
                "comparison_id": row.get("comparison_id"),
                "repeat": row.get("repeat"),
                "metric": metric,
            }
            metric_rows.append(
                {
                    **common,
                    "reference_value": reference,
                    "candidate_value": candidate,
                    "candidate_minus_reference": delta,
                }
            )
            applicability_rows.append(
                {
                    **common,
                    "applicability": applicability,
                    "difference_direction": row.get("difference_direction"),
                    "automatic_selection": row.get("automatic_selection"),
                }
            )
    return (
        (
            "Comparison contracts",
            _unique_projection(rows, contract_fields),
            contract_fields,
        ),
        (
            "Matched repeat roster",
            _unique_projection(rows, roster_fields),
            roster_fields,
        ),
        ("Per-repeat metric differences", metric_rows, metric_fields),
        (
            "Metric applicability and interpretation",
            applicability_rows,
            applicability_fields,
        ),
    )


def _md_display_tables(
    tables: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[str]]],
) -> str:
    return "\n\n".join(
        f"#### {title}\n\n{_md_table(rows, columns)}" for title, rows, columns in tables
    )


def _prepare_output_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"historical output target is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            "historical output target must be empty to prevent stale-artifact "
            f"contamination: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_historical_major_report(
    *,
    early_source: str | Path,
    shapeformer_source: str | Path,
    fixed_epoch_source: str | Path,
    extension_source: str | Path,
    output_dir: str | Path,
    window_seconds: float = 5.0,
    overlap_percent: float = 50.0,
    patience: int = 20,
    extra_input: str = "0",
) -> Path:
    """Generate the requested historical evidence bundle without retraining."""

    pandas = _pd()
    sources = tuple(
        Path(value).resolve()
        for value in (
            early_source,
            shapeformer_source,
            fixed_epoch_source,
            extension_source,
        )
    )
    if any(not source.is_dir() for source in sources):
        missing = [str(source) for source in sources if not source.is_dir()]
        raise FileNotFoundError(f"historical source directories are missing: {missing}")
    target = _prepare_output_directory(output_dir)
    tables = target / "tables"
    figures = target / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    (
        early_repeats,
        early_summary,
        early_parameters,
        early_class,
        early_splits,
        early_paired_tests,
    ) = _early_contract_rows(
        sources[0],
        sources[1],
        window_seconds=window_seconds,
        overlap_percent=overlap_percent,
        patience=patience,
        extra_input=extra_input,
    )
    (
        fixed_summary,
        fixed_top,
        fixed_effects,
        fixed_contract,
        fixed_splits,
        fixed_per_class_repeats,
        fixed_per_class_summary,
        fixed_factor_repeat_deltas,
        fixed_factor_cluster_inference,
    ) = _overfit_summary(sources[2])
    (
        extension_summary,
        extension_top,
        extension_effects,
        extension_contract,
        extension_splits,
        extension_per_class_repeats,
        extension_per_class_summary,
        extension_factor_repeat_deltas,
        extension_factor_cluster_inference,
    ) = _overfit_summary(sources[3])
    cross_study_splits = _cross_study_split_audit(
        early_splits,
        fixed_splits,
        extension_splits,
    )
    consumed_reports = _consumed_report_paths(sources, early_repeats)
    source_evidence = pandas.DataFrame(_source_rows(sources, consumed_reports))
    missing = pandas.DataFrame(_missing_statistics())
    early_repeat_deltas, early_cluster_inference = _early_pairwise_repeat_deltas(
        early_repeats
    )
    historical_repeat_deltas = pandas.DataFrame(
        _as_records(early_repeat_deltas)
        + _as_records(fixed_factor_repeat_deltas)
        + _as_records(extension_factor_repeat_deltas)
    )
    historical_cluster_inference = pandas.DataFrame(
        _as_records(early_cluster_inference)
        + _as_records(fixed_factor_cluster_inference)
        + _as_records(extension_factor_cluster_inference)
    )
    absolute_cluster_ci = pandas.DataFrame(
        _as_records(
            _absolute_historical_cluster_ci_rows(
                early_summary,
                source_study="matched_three_model_historical_comparison",
                classifier_id_column="model_display",
                model_column="model_display",
            )
        )
        + _as_records(
            _absolute_historical_cluster_ci_rows(
                fixed_summary,
                source_study=sources[2].name,
                classifier_id_column="overfit_config_id",
                model_column="resolved_model",
            )
        )
        + _as_records(
            _absolute_historical_cluster_ci_rows(
                extension_summary,
                source_study=sources[3].name,
                classifier_id_column="overfit_config_id",
                model_column="resolved_model",
            )
        )
    )

    exports = {
        "early_three_model_repeat_metrics": early_repeats,
        "early_three_model_summary": early_summary,
        "early_three_model_parameters": early_parameters,
        "early_three_model_per_class_summary": early_class,
        "early_three_model_split_audit": early_splits,
        "early_three_model_exploratory_paired_tests": early_paired_tests,
        "early_three_model_pairwise_repeat_metric_deltas": early_repeat_deltas,
        "early_three_model_paired_participant_inference": early_cluster_inference,
        "pairwise_repeat_metric_deltas": historical_repeat_deltas,
        "paired_participant_inference": historical_cluster_inference,
        "fixed_epoch_all_config_summary": fixed_summary,
        "fixed_epoch_top15": fixed_top,
        "fixed_epoch_factor_signals": fixed_effects,
        "fixed_epoch_per_class_repeat_results": fixed_per_class_repeats,
        "fixed_epoch_classifier_per_class_results": fixed_per_class_summary,
        "fixed_epoch_factor_pairwise_repeat_metric_deltas": fixed_factor_repeat_deltas,
        "fixed_epoch_factor_paired_participant_inference": fixed_factor_cluster_inference,
        "fixed_epoch_parameter_space": fixed_contract,
        "fixed_epoch_split_audit": fixed_splits,
        "extension_all_config_summary": extension_summary,
        "extension_top15": extension_top,
        "extension_factor_signals": extension_effects,
        "extension_per_class_repeat_results": extension_per_class_repeats,
        "extension_classifier_per_class_results": extension_per_class_summary,
        "extension_factor_pairwise_repeat_metric_deltas": extension_factor_repeat_deltas,
        "extension_factor_paired_participant_inference": extension_factor_cluster_inference,
        "extension_parameter_space": extension_contract,
        "extension_split_audit": extension_splits,
        "historical_cross_study_split_audit": cross_study_splits,
        "historical_absolute_participant_cluster_ci": absolute_cluster_ci,
        "missing_v2_statistics": missing,
        "source_evidence": source_evidence,
    }
    for name, frame in exports.items():
        write_csv(tables / f"{name}.csv", _as_records(frame))
    pairs = pandas.DataFrame(
        [
            {
                "table": "early_three_model_repeat_metrics.csv",
                "figure": "early_three_model_ba_f1_boxplots.png",
                "relationship": "same matched repeat evidence",
            },
            {
                "table": "early_three_model_repeat_metrics.csv",
                "figure": "early_three_model_runtime_boxplot.png",
                "relationship": "same matched runtime evidence",
            },
            {
                "table": "fixed_epoch_top15.csv",
                "figure": "fixed_epoch_top_config_ba_ci.png",
                "relationship": "same config-level ranking",
            },
            {
                "table": "fixed_epoch_factor_signals.csv",
                "figure": "fixed_epoch_factor_signals.png",
                "relationship": "same post-hoc factor contrasts",
            },
            {
                "table": "extension_top15.csv",
                "figure": "extension_top_config_ba_ci.png",
                "relationship": "same config-level ranking",
            },
            {
                "table": "extension_factor_signals.csv",
                "figure": "extension_factor_signals.png",
                "relationship": "same post-hoc factor contrasts",
            },
        ]
    )
    write_csv(tables / "table_figure_pairs.csv", _as_records(pairs))
    _plot_early(
        early_repeats,
        early_summary,
        figures,
        window_seconds=window_seconds,
        overlap_percent=overlap_percent,
        patience=patience,
    )
    _plot_overfit(fixed_top, fixed_effects, figures, "fixed_epoch")
    _plot_overfit(extension_top, extension_effects, figures, "extension")
    write_table_column_definitions(tables, csv_directory=tables)
    write_excel_workbook_from_csv_directory(tables / "report_tables.xlsx", tables)

    top_rows = _as_records(early_summary)
    early_display = []
    for row in top_rows:
        early_display.append(
            {
                "model": row["model_display"],
                "rank": row["descriptive_rank"],
                "subject BA mean ± SD": f"{100*row['subject_balanced_accuracy_mean']:.1f} ± {100*row['subject_balanced_accuracy_sample_sd']:.1f}",
                "subject BA repeat t-CI95": f"[{100*row['subject_balanced_accuracy_repeat_t_ci95_low']:.1f}, {100*row['subject_balanced_accuracy_repeat_t_ci95_high']:.1f}]",
                "subject macro-F1 mean ± SD": f"{100*row['subject_macro_f1_mean']:.1f} ± {100*row['subject_macro_f1_sample_sd']:.1f}",
                "runtime mean (s)": row["duration_sec_mean"],
            }
        )
    best = top_rows[0]
    shape = next(row for row in top_rows if row["model_display"] == "ShapeFormer-PISD")
    runtime_ratio = shape["duration_sec_mean"] / best["duration_sec_mean"]
    shape_repeat = early_repeats.loc[
        early_repeats["model_display"].eq("ShapeFormer-PISD"),
        ["seed", "subject_balanced_accuracy", "subject_macro_f1"],
    ].set_index("seed")
    shape_uniformly_lower = True
    for other_name in (
        name
        for name in early_repeats["model_display"].unique()
        if name != "ShapeFormer-PISD"
    ):
        other_repeat = early_repeats.loc[
            early_repeats["model_display"].eq(other_name),
            ["seed", "subject_balanced_accuracy", "subject_macro_f1"],
        ].set_index("seed")
        shape_uniformly_lower = shape_uniformly_lower and bool(
            (
                shape_repeat["subject_balanced_accuracy"]
                < other_repeat["subject_balanced_accuracy"]
            ).all()
            and (
                shape_repeat["subject_macro_f1"] < other_repeat["subject_macro_f1"]
            ).all()
        )
    if shape_uniformly_lower:
        shape_interpretation = (
            "ShapeFormer-PISD was below both alternatives on BA and macro-F1 "
            "in all five matched repeats. This supports not advancing this "
            "historical implementation on utility/cost grounds, but does not "
            "prove general ShapeFormer inferiority."
        )
    else:
        shape_interpretation = (
            "ShapeFormer-PISD was not uniformly below both alternatives across "
            "the matched repeats, so this archive alone does not support its exclusion."
        )
    extra_description = (
        "no extra PPI/HRV input"
        if str(extra_input).strip() in {"0", "0.0"}
        else f"extra_input={extra_input}"
    )
    fixed_count = len(fixed_summary)
    extension_count = len(extension_summary)
    fixed_leader = str(fixed_top.iloc[0]["overfit_config_id"])
    extension_leader = str(extension_top.iloc[0]["overfit_config_id"])
    report = [
        "# Historical major-search V2-oriented reanalysis",
        "",
        "## Scope and evidential status",
        "",
        "This is a read-only reanalysis of four archived searches. It does not retrain a model, alter an archived run, or upgrade a post-hoc search into a confirmatory V2 test.",
        "",
        f"The matched architecture comparison is restricted to `{window_seconds:g} s`, `{overlap_percent:g}%` overlap, patience `{patience}`, and {extra_description}. All four source studies share seeds `{', '.join(str(seed) for seed in _SPLIT_SEEDS)}` and the exact participant-fold roster for every seed; this improves pairing but does not make the searches independent evidence.",
        "",
        "## Matched CNN1D–InceptionTime–ShapeFormer comparison",
        "",
        _md_table(early_display, tuple(early_display[0])),
        "",
        f"{best['model_display']} is the descriptive leader. ShapeFormer-PISD is lower by `{100*(best['subject_balanced_accuracy_mean']-shape['subject_balanced_accuracy_mean']):.1f}` BA percentage points and `{100*(best['subject_macro_f1_mean']-shape['subject_macro_f1_mean']):.1f}` macro-F1 points, while requiring about `{runtime_ratio:.1f}×` the leader's mean archived run time. {shape_interpretation}",
        "",
        "The exact aggregate-repeat sign-flip tests are retained in `tables/early_three_model_exploratory_paired_tests.csv`. With only five sign pairs there are 32 possible sign patterns, so the smallest attainable two-sided P is 0.0625; these P values are exploratory and are not V2 participant-exchange inference.",
        "",
        "### Every matched pair × repeat difference",
        "",
        "The following rows keep all three model pairs and all five matched split seeds. BA and macro-F1 are candidate minus reference; ROC-AUC and participant-cluster CIs are explicit N/A because probability-level participant OOF was not archived.",
        "",
        _md_display_tables(
            _historical_pairwise_display_tables(
                _as_records(early_repeat_deltas),
            )
        ),
        "",
        "## Fixed-epoch regularization search (20260608)",
        "",
        f"The archive contains `{fixed_count}` complete five-repeat configurations. Its top observed configuration is `{fixed_leader}` with subject BA `{100*fixed_top.iloc[0]['subject_balanced_accuracy_mean']:.1f} ± {100*fixed_top.iloc[0]['subject_balanced_accuracy_sample_sd']:.1f}`. Because this winner was chosen from the same {fixed_count}-configuration evidence, it is hypothesis-generating. The defensible manuscript use is to motivate fixed-epoch and regularization hypotheses later retested in V2, not to report the winner as an unbiased final estimate.",
        "",
        f"All `{fixed_count}` classifiers/configurations have three-class repeat rows and five-repeat summaries in `tables/fixed_epoch_per_class_repeat_results.csv` and `tables/fixed_epoch_classifier_per_class_results.csv`. Each of the `{len(fixed_effects)}` explicit baseline-vs-candidate factor-signal pairs has five matched BA/F1/ROC-AUC comparison rows in `tables/fixed_epoch_factor_pairwise_repeat_metric_deltas.csv`; ROC-AUC and all participant-cluster intervals are explicit N/A with their archive limitation.",
        "",
        "## SQI/loss/feature extension (20260625)",
        "",
        f"The archive contains `{extension_count}` complete five-repeat configurations. Its top observed configuration is `{extension_leader}` with subject BA `{100*extension_top.iloc[0]['subject_balanced_accuracy_mean']:.1f} ± {100*extension_top.iloc[0]['subject_balanced_accuracy_sample_sd']:.1f}`. The complete post-hoc factor signals are retained in `tables/extension_factor_signals.csv`; they may motivate a matched V2 composition study but cannot freeze an SQI or aggregation route by themselves.",
        "",
        f"All `{extension_count}` classifiers/configurations have three-class repeat rows and five-repeat summaries in `tables/extension_per_class_repeat_results.csv` and `tables/extension_classifier_per_class_results.csv`. Each of the `{len(extension_effects)}` explicit baseline-vs-candidate factor-signal pairs has five matched rows in `tables/extension_factor_pairwise_repeat_metric_deltas.csv`, with participant-cluster CI applicability recorded pair-by-pair and metric-by-metric.",
        "",
        "## Statistical compatibility with V2",
        "",
        "Available: config-level means, sample SD, two-sided Student-t 95% intervals across the five archived repeat summaries, class-level hard-label metrics, confusion matrices, learning curves, and run duration.",
        "",
        "Unavailable: participant-cluster bootstrap intervals, formal participant-exchange paired permutation P values, ROC/PR curves, probability calibration and t-SNE. See `tables/missing_v2_statistics.csv`. A P value is a null-tail probability, not posterior confidence.",
        "",
        "Participant-cluster CI is defined as a true-class-stratified bootstrap of participant IDs: each sampled participant carries all of that participant's OOF rows across all repeats, the metric is recomputed within each repeat, repeat metrics are equally averaged, and the 2.5th/97.5th bootstrap percentiles form the 95% interval. It quantifies sampling uncertainty conditional on the observed participants, frozen splits, model and analysis contract; it is not a posterior probability and does not include dataset shift or model-selection uncertainty. Because these archives lack participant-keyed OOF rows, `tables/historical_absolute_participant_cluster_ci.csv` and every paired inference table record N/A with the exact reason.",
        "",
        "The early architecture archive also carries a selection-contamination risk: the persisted fold histories and archived generator structure indicate that the held-out fold supplied the validation trajectory used for best-epoch selection. It is therefore described as legacy fold-held-out CV with selection contamination, not untouched OOF confirmation. The fixed-epoch June searches disable best-epoch selection, but still have no CV-external test set.",
        "",
        "## Recommended writing order",
        "",
        "1. Use the matched three-model historical comparison to motivate CNN/InceptionTime as ordinary candidates and to separate ShapeFormer into a non-blocking diagnostic route.",
        "2. Use the 20260608 archive only to motivate fixed-epoch and regularization hypotheses.",
        "3. Use the 20260625 archive only to motivate SQI, aggregation, loss and engineered-feature hypotheses.",
        "4. Move every effectiveness claim to the later matched V2 ablations, hyperparameter studies and locked confirmation.",
        "",
        "## Method references",
        "",
        "- Student (1908), *The Probable Error of a Mean*, https://doi.org/10.1093/biomet/6.1.1",
        "- Bengio & Grandvalet (2004), *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, https://www.jmlr.org/papers/v5/grandvalet04a.html",
        "- Varma & Simon (2006), *Bias in error estimation when using cross-validation for model selection*, https://doi.org/10.1186/1471-2105-7-91",
        "- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, https://www.jmlr.org/papers/v11/cawley10a.html",
        "- Ojala & Garriga (2010), *Permutation Tests for Studying Classifier Performance*, https://www.jmlr.org/papers/v11/ojala10a.html",
        "- Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*, https://doi.org/10.2307/4615733",
        "",
    ]
    summary_path = target / "STUDY_SUMMARY.md"
    summary_path.write_text("\n".join(report), encoding="utf-8")
    interpretation = [
        "# Result interpretation",
        "",
        f"- Descriptive architecture leader: `{best['model_display']}`; no confirmatory winner is declared.",
        f"- ShapeFormer decision: {shape_interpretation}",
        f"- Fixed-epoch archive: `{fixed_leader}` is the point leader among {fixed_count} searched configurations and remains hypothesis-generating.",
        f"- SQI/loss/feature archive: `{extension_leader}` is the point leader among {extension_count} searched configurations and remains hypothesis-generating.",
        "- ROC/PR, calibration, participant-cluster CI and participant-exchange P remain unavailable because prediction-level OOF evidence was not archived.",
        "",
    ]
    (target / "RESULT_INTERPRETATION.md").write_text(
        "\n".join(interpretation), encoding="utf-8"
    )
    components = [
        "# Test components and fixed inputs",
        "",
        "## Historical sources",
        "",
        *(f"- `{source}`" for source in sources),
        "",
        "## Matched architecture contract",
        "",
        f"- Window: {window_seconds:g} s",
        f"- Overlap: {overlap_percent:g}%",
        f"- Patience: {patience}",
        f"- Extra input: {extra_input}",
        f"- Split seeds: {', '.join(str(seed) for seed in _SPLIT_SEEDS)}",
        "- Split: exact participant-grouped five-fold rosters audited across all four sources",
        f"- Consumed report JSON files: {len(consumed_reports)}",
        "- Scientific role: historical candidate/hypothesis generation only",
        "",
    ]
    (target / "TEST_COMPONENTS.md").write_text("\n".join(components), encoding="utf-8")
    methods = [
        "# Historical reporter methods",
        "",
        "- Analysis unit: one archived repeat summary; n=5 per historical configuration.",
        "- Display: mean ± sample SD.",
        "- Descriptive interval: mean ± t(0.975, n-1) × sample SD / sqrt(n).",
        "- Ranking: subject BA, then subject macro-F1; ranking remains post hoc.",
        "- The matched three-model table includes an exploratory exact sign-flip test across five aggregate repeat seeds, with Holm adjustment across three model pairs within each metric.",
        "- Formal V2 participant-exchange P values are unavailable because participant OOF rows/probabilities were not archived.",
        "- Participant-cluster CI means resampling participant IDs with replacement within true-class strata, carrying every selected participant's rows across all repeats, recomputing the metric per repeat, equally averaging repeats, and taking the 2.5th and 97.5th bootstrap percentiles. The historical aggregate archives cannot execute that procedure, so every absolute and paired cluster-CI cell is explicit N/A rather than a repeat-level substitute.",
        "- Every consumed report JSON plus the root CSV/manifests is hashed in `tables/source_evidence.csv`.",
        "- Seeds and exact held-out participant rosters are audited within and across all four source studies in the split-audit CSVs.",
        "- Every CSV table is also one worksheet in `tables/report_tables.xlsx`.",
        "- Every generated figure has a numerical companion in `tables/table_figure_pairs.csv`.",
        "",
    ]
    (target / "REPORT_METHODS.md").write_text("\n".join(methods), encoding="utf-8")
    output_evidence = [
        {
            "path": str(path.relative_to(target)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            (path for path in target.rglob("*") if path.is_file()),
            key=lambda path: str(path.relative_to(target)).encode("utf-8"),
        )
    ]
    output_index_path = target / "outputs_index.json"
    output_index_path.write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.historical_output_index.v2",
                "inventory_scope": "all_reporter_outputs_except_the_index_and_manifest",
                "artifacts": output_evidence,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_evidence.append(
        {
            "path": "outputs_index.json",
            "size_bytes": output_index_path.stat().st_size,
            "sha256": _sha256(output_index_path),
        }
    )
    output_paths = {row["path"] for row in output_evidence}
    output_paths.add("report_manifest.json")
    manifest = {
        "schema_version": "historical_major_report_v2.3",
        "status": "descriptive_reanalysis_complete",
        "scientific_status": "historical_hypothesis_generation_only",
        "sources": [str(source) for source in sources],
        "consumed_report_json_count": len(consumed_reports),
        "matched_contract": {
            "window_seconds": window_seconds,
            "overlap_percent": overlap_percent,
            "patience": patience,
            "extra_input": extra_input,
        },
        "output_evidence_excluding_self_manifest": output_evidence,
        "outputs": sorted(output_paths),
    }
    (target / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = ["run_historical_major_report"]
