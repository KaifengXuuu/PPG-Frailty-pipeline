"""Split, auditable V2-style reports for the selected historical searches.

The archived studies pre-date the V2 prediction-evidence contract.  This module
therefore reports every recoverable aggregate and per-class statistic while
keeping probability-dependent ROC-AUC and participant-cluster inference as
explicitly unavailable.  It never retrains a model or reconstructs participant
predictions from aggregate confusion matrices.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .historical import (
    _SPLIT_SEEDS,
    _absolute_historical_cluster_ci_rows,
    _as_records,
    _early_contract_rows,
    _early_pairwise_repeat_deltas,
    _metric_summary,
    _missing_statistics,
    _overfit_summary,
    _prepare_output_directory,
    _report_path,
    _sha256,
)
from .tabular import (
    format_interval,
    format_mean_sd,
    html_column_definitions_block,
    markdown_column_definitions_block,
    write_csv,
    write_excel_workbook_from_csv_directory,
    write_table_column_definitions,
)


_METRIC_FIELDS = (
    ("subject_balanced_accuracy", "Balanced accuracy"),
    ("subject_macro_f1", "Macro-F1"),
    ("subject_macro_roc_auc_ovr", "Macro ROC-AUC (OvR)"),
)
_MAX_HUMAN_TABLE_COLUMNS = 8
_LEADERBOARD_PERFORMANCE_COLUMNS = (
    "config_or_model",
    "model",
    "rank",
    "subject_BA_mean_sd_percent",
    "subject_BA_repeat_t_CI95_percent",
    "subject_macro_F1_mean_sd_percent",
    "subject_macro_F1_repeat_t_CI95_percent",
)
_LEADERBOARD_APPLICABILITY_COLUMNS = (
    "config_or_model",
    "model",
    "subject_macro_ROC_AUC",
    "ROC_AUC_applicability",
    "scientific_role",
)
_PARAMETER_INVENTORY_COLUMNS = (
    "parameter",
    "parameter_group",
    "source_study",
    "unique_value_count",
    "observed_values",
    "parameter_role",
    "comparison_interpretation",
)
_SUITE_INDEX_COLUMNS = (
    "Report",
    "Sources",
    "Configurations",
    "Runs",
    "Point leader",
    "BA",
    "Open",
)
_PARAMETER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("architecture", ("model", "resolved_model")),
    (
        "input_and_windowing",
        (
            "extra_input",
            "window_sec",
            "hop_sec",
            "overlap_pct",
            "max_windows_fraction",
            "train_overlap_pct",
            "cnn_target_fs",
        ),
    ),
    ("training_budget", ("cnn_epochs", "cnn_patience", "early_stopping_source")),
    (
        "optimization_and_regularization",
        (
            "cnn_lr",
            "cnn_weight_decay",
            "cnn_dropout",
            "cnn_label_smoothing",
            "loss_type",
            "class_weight_mode",
            "regularization_bundle",
        ),
    ),
    (
        "quality_feature_aggregation",
        (
            "sqi_mode",
            "aggregation",
            "manual_features",
            "quality_route_bundle",
        ),
    ),
    (
        "sampling",
        (
            "window_sampler",
            "windows_per_subject_per_epoch",
            "sampling_policy",
        ),
    ),
    (
        "study_factor",
        (
            "overfit_stage",
            "stage1_screen_group",
            "stage1_regularization_factor",
            "stage1_regularization_value",
            "is_reference",
        ),
    ),
    (
        "data_roles",
        (
            "dynamic_data_mode",
            "train_role_mode",
            "validation_role_mode",
            "test_role_mode",
            "eval_protocol",
            "n_splits",
        ),
    ),
)
_DIRECT_PLOT_PARAMETERS = (
    "model",
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
    "regularization_bundle",
    "quality_route_bundle",
    "sampling_policy",
    "train_overlap_pct",
    "overfit_stage",
    "stage1_screen_group",
    "stage1_regularization_factor",
)
_FACTOR_TO_PARAMETER = {
    "weight_decay": "cnn_weight_decay",
    "dropout": "cnn_dropout",
    "label_smoothing": "cnn_label_smoothing",
    "max_windows_fraction": "max_windows_fraction",
    "sqi_mode": "sqi_mode",
    "aggregation": "aggregation",
    "manual_features": "manual_features",
    "loss_type": "loss_type",
    "class_weight_mode": "class_weight_mode",
}


def _pd():
    try:
        import pandas as pandas
    except ImportError as exc:  # pragma: no cover - optional reporting dependency
        raise ImportError("historical suite reporting requires pandas") from exc
    return pandas


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as exc:  # pragma: no cover - optional reporting dependency
        raise ImportError("historical suite reporting requires matplotlib") from exc
    return pyplot


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parameter_group(column: str) -> str:
    for group, fields in _PARAMETER_GROUPS:
        if column in fields:
            return group
    return "other_archived_parameter"


def _derive_design_factors(runs: Any) -> Any:
    """Name joint design factors so coupled knobs are not misread as ablations."""

    frame = runs.copy()
    if {"cnn_weight_decay", "cnn_dropout", "cnn_label_smoothing"} <= set(frame):
        frame["regularization_bundle"] = frame.apply(
            lambda row: (
                f"WD={_normalise_plot_value(row['cnn_weight_decay'])}; "
                f"dropout={_normalise_plot_value(row['cnn_dropout'])}; "
                f"LS={_normalise_plot_value(row['cnn_label_smoothing'])}"
            ),
            axis=1,
        )
    if {"sqi_mode", "aggregation"} <= set(frame):
        frame["quality_route_bundle"] = frame.apply(
            lambda row: f"SQI={row['sqi_mode']}; aggregation={row['aggregation']}",
            axis=1,
        )
    if {"window_sampler", "windows_per_subject_per_epoch"} <= set(frame):
        frame["sampling_policy"] = frame.apply(
            lambda row: (
                f"sampler={row['window_sampler']}; quota={row['windows_per_subject_per_epoch']}"
            ),
            axis=1,
        )
    return frame


def _design_dependency_warnings(runs: Any, *, source_study: str) -> Any:
    pandas = _pd()
    rows: list[dict[str, Any]] = []
    if {"cnn_weight_decay", "cnn_label_smoothing"} <= set(runs):
        pairs = runs[["cnn_weight_decay", "cnn_label_smoothing"]].drop_duplicates()
        if (
            len(pairs) > 1
            and pairs.groupby("cnn_weight_decay")["cnn_label_smoothing"].nunique().max()
            == 1
            and pairs.groupby("cnn_label_smoothing")["cnn_weight_decay"].nunique().max()
            == 1
        ):
            rows.append(
                {
                    "source_study": source_study,
                    "design_factor": "regularization_bundle",
                    "coupled_parameters": "cnn_weight_decay;cnn_label_smoothing",
                    "status": "perfectly_co_varied_in_observed_design",
                    "reporting_rule": "report bundle; do not attribute separate WD or LS effect",
                }
            )
    if {"window_sampler", "windows_per_subject_per_epoch"} <= set(runs):
        pairs = runs[
            ["window_sampler", "windows_per_subject_per_epoch"]
        ].drop_duplicates()
        full_size = (
            runs["window_sampler"].nunique()
            * runs["windows_per_subject_per_epoch"].nunique()
        )
        if len(pairs) < full_size:
            rows.append(
                {
                    "source_study": source_study,
                    "design_factor": "sampling_policy",
                    "coupled_parameters": "window_sampler;windows_per_subject_per_epoch",
                    "status": "incomplete_factorial_structural_coupling",
                    "reporting_rule": "compare observed joint policies; do not infer independent sampler/quota effects",
                }
            )
    if {"sqi_mode", "aggregation"} <= set(runs) and runs["sqi_mode"].nunique() > 1:
        rows.append(
            {
                "source_study": source_study,
                "design_factor": "quality_route_bundle",
                "coupled_parameters": "sqi_mode;aggregation;training/evaluation window retention",
                "status": "route_level_intervention",
                "reporting_rule": "interpret as SQI route/composition, not threshold-only effect",
            }
        )
    if "stage1_screen_group" in runs and any(
        token in set(runs["stage1_screen_group"].dropna().astype(str))
        for token in (
            "strong_combo",
            "focused_combo",
            "quality_combo",
            "manual_loss_combo",
        )
    ):
        rows.append(
            {
                "source_study": source_study,
                "design_factor": "multi_parameter_compositions",
                "coupled_parameters": "stage1_screen_group composition-specific parameter sets",
                "status": "multiple_parameters_change_together",
                "reporting_rule": "composition/search comparison; not a single-factor ablation",
            }
        )
    return pandas.DataFrame(rows)


def _display_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        if abs(value) < 1e-3 and value != 0:
            return f"{value:.3g}"
        return f"{value:.6g}"
    return str(value)


def _normalise_plot_value(value: Any) -> str:
    """Collapse harmless floating representations without changing source CSVs."""

    number = _finite(value)
    if number is not None and not isinstance(value, str):
        return _display_value(number)
    return _display_value(value)


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return clean[:72] or "comparison"


def _configuration_parameter_table(
    runs: Any,
    *,
    source_study: str,
    source: Path,
) -> Any:
    pandas = _pd()
    config_column = (
        "overfit_config_id" if "overfit_config_id" in runs else "model_display"
    )
    sort_columns = [config_column]
    if "repeat" in runs:
        sort_columns.append("repeat")
    ordered = runs.sort_values(sort_columns)
    parameter_columns = [
        column
        for _group, fields in _PARAMETER_GROUPS
        for column in fields
        if column in ordered
    ]
    rows: list[dict[str, Any]] = []
    for config_id, group in ordered.groupby(config_column, dropna=False, sort=False):
        run = group.iloc[0]
        configs: list[dict[str, Any]] = []
        for _, repeat_run in group.iterrows():
            report = json.loads(
                _report_path(source, repeat_run["report_path"]).read_text(
                    encoding="utf-8"
                )
            )
            configs.append(
                {
                    str(key): value
                    for key, value in report.get("config", {}).items()
                    if str(key) != "seed"
                }
            )
        canonical = json.dumps(configs[0], sort_keys=True, ensure_ascii=False)
        if any(
            json.dumps(config, sort_keys=True, ensure_ascii=False) != canonical
            for config in configs[1:]
        ):
            raise ValueError(
                f"{source_study}/{config_id} report config drifts across repeat seeds"
            )
        report_config = configs[0]
        train_roles = run.get("train_role_mode", "B/R1-R4 static roles")
        validation_roles = run.get(
            "validation_role_mode", "participant-held-out B/R1-R4"
        )
        test_roles = run.get("test_role_mode", "participant-held-out B/R1-R4")
        native_fs = report_config.get("fs", "archived_value_unavailable")
        target_fs = report_config.get("cnn_target_fs", "archived_value_unavailable")
        row: dict[str, Any] = {
            "source_study": source_study,
            "config_id": run.get(config_column),
            "config_name": run.get("overfit_config_name", run.get("model_display")),
            "input_data": (
                "29-participant internal dataset; columns RED, IR, AX, AY, AZ, GX, GY, GZ; "
                f"train={train_roles}; validation={validation_roles}; test={test_roles}; "
                f"native_fs={native_fs} Hz; DL_target_fs={target_fs} Hz; historical DL "
                "tensor uses the source study's archived filtering/resampling/scaling; "
                "no participant-level prediction rows archived"
            ),
        }
        for column in parameter_columns:
            row[column] = run.get(column)
        for name, value in sorted(report_config.items()):
            row[f"report_config__{name}"] = value
        rows.append(row)
    return pandas.DataFrame(rows)


def _parameter_inventory(runs: Any, *, source_study: str) -> Any:
    pandas = _pd()
    rows: list[dict[str, Any]] = []
    for group, fields in _PARAMETER_GROUPS:
        for column in fields:
            if column not in runs:
                continue
            values = [
                value
                for value in runs[column].drop_duplicates().tolist()
                if not (isinstance(value, float) and math.isnan(value))
            ]
            display_values = sorted({_normalise_plot_value(value) for value in values})
            rows.append(
                {
                    "source_study": source_study,
                    "parameter_group": group,
                    "parameter": column,
                    "unique_value_count": len(display_values),
                    "observed_values": json.dumps(display_values, ensure_ascii=False),
                    "parameter_role": "varied" if len(display_values) > 1 else "fixed",
                    "comparison_interpretation": (
                        "marginal_descriptive_not_single_factor_causal"
                        if len(display_values) > 1
                        else "fixed_within_archive"
                    ),
                }
            )
    return pandas.DataFrame(rows)


def _long_metric_rows(
    runs: Any,
    *,
    source_study: str,
    parameter: str,
    subset_label: str,
    subset: Any | None = None,
) -> Any:
    pandas = _pd()
    selected = runs if subset is None else subset
    config_column = (
        "overfit_config_id" if "overfit_config_id" in selected else "model_display"
    )
    rows: list[dict[str, Any]] = []
    for _, run in selected.iterrows():
        value = _normalise_plot_value(run.get(parameter))
        model = str(
            run.get(
                "resolved_model", run.get("model_display", run.get("model", "model"))
            )
        )
        rows.append(
            {
                "source_study": source_study,
                "parameter_group": _parameter_group(parameter),
                "parameter": parameter,
                "subset_contract": subset_label,
                "parameter_value": value,
                "model": model,
                "plot_category": value,
                "config_id": run.get(config_column),
                "repeat": int(run.get("repeat")),
                "split_seed": int(run.get("seed")),
                "subject_balanced_accuracy": _finite(
                    run.get("subject_balanced_accuracy")
                ),
                "subject_macro_f1": _finite(run.get("subject_macro_f1")),
                "subject_macro_roc_auc_ovr": None,
                "macro_roc_auc_ovr_applicability": (
                    "N/A_continuous_participant_oof_probabilities_not_archived"
                ),
                "observation_unit": "archived_config_repeat_summary",
                "independence_warning": (
                    "rows share participants and many configurations; boxplots are descriptive"
                ),
            }
        )
    frame = pandas.DataFrame(rows)
    if frame["model"].nunique() > 1:
        frame["plot_category"] = frame["parameter_value"] + " · " + frame["model"]
    return frame


def _evaluation_level_metric_rows(
    runs: Any,
    *,
    source_study: str,
    parameter: str,
    subset_label: str,
    evaluation_level: str,
) -> Any:
    if evaluation_level not in {"window", "file", "subject"}:
        raise ValueError(f"unsupported evaluation level: {evaluation_level}")
    frame = _long_metric_rows(
        runs,
        source_study=source_study,
        parameter=parameter,
        subset_label=subset_label,
    )
    frame["evaluation_level"] = evaluation_level
    frame["subject_balanced_accuracy"] = (
        runs[f"{evaluation_level}_balanced_accuracy"].astype(float).to_numpy()
    )
    frame["subject_macro_f1"] = (
        runs[f"{evaluation_level}_macro_f1"].astype(float).to_numpy()
    )
    return frame


def _plot_specs(runs: Any) -> list[tuple[str, str, Any]]:
    """Return auditable plot contracts as (parameter, subset label, rows)."""

    specs: list[tuple[str, str, Any]] = []
    seen: set[tuple[str, str]] = set()
    is_generalization_grid = "stage1_regularization_factor" in runs and set(
        runs["stage1_regularization_factor"].dropna().astype(str)
    ) == {"generalization_grid"}
    generalization_skip = {
        "cnn_weight_decay",
        "cnn_dropout",
        "cnn_label_smoothing",
        "sqi_mode",
        "aggregation",
    }
    for parameter in _DIRECT_PLOT_PARAMETERS:
        if is_generalization_grid and parameter in generalization_skip:
            continue
        if parameter not in runs:
            continue
        values = {
            _normalise_plot_value(value) for value in runs[parameter].dropna().tolist()
        }
        if not 2 <= len(values) <= 16:
            continue
        key = (parameter, "all_archived_config_repeats")
        if key not in seen:
            specs.append((parameter, key[1], runs))
            seen.add(key)

    if "stage1_regularization_factor" in runs:
        factors = sorted(
            str(value)
            for value in runs["stage1_regularization_factor"].dropna().unique()
        )
        for factor in factors:
            parameter = _FACTOR_TO_PARAMETER.get(factor)
            if parameter is None or parameter not in runs:
                continue
            subset = runs.loc[
                runs["stage1_regularization_factor"].astype(str).eq(factor)
            ]
            values = {
                _normalise_plot_value(value) for value in subset[parameter].dropna()
            }
            if not 2 <= len(values) <= 16:
                continue
            label = f"stage1_regularization_factor={factor}"
            key = (parameter, label)
            if key not in seen:
                specs.append((parameter, label, subset))
                seen.add(key)

    if (
        "stage1_regularization_value" in runs
        and 2 <= runs["stage1_regularization_value"].dropna().nunique() <= 16
    ):
        key = ("stage1_regularization_value", "all_archived_config_repeats")
        if key not in seen:
            specs.append((key[0], key[1], runs))
    return specs


def _parameter_value_metric_summary(rows: Any) -> Any:
    """Summarize marginal factor views after equal weighting within each seed."""

    pandas = _pd()
    if rows.empty:
        return pandas.DataFrame()
    output: list[dict[str, Any]] = []
    keys = (
        "source_study",
        "parameter_group",
        "parameter",
        "subset_contract",
        "parameter_value",
        "model",
        "plot_category",
    )
    for key, group in rows.groupby(list(keys), dropna=False, sort=False):
        by_seed = (
            group.groupby(["repeat", "split_seed"], as_index=False)[
                ["subject_balanced_accuracy", "subject_macro_f1"]
            ]
            .mean()
            .sort_values("split_seed")
        )
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "config_count": int(group["config_id"].nunique()),
                "config_repeat_row_count": int(len(group)),
                "repeat_seed_count": int(len(by_seed)),
                "within_seed_aggregation": "equal_mean_across_configs_with_parameter_value",
                "interpretation": "marginal_descriptive_not_single_factor_causal",
            }
        )
        for metric in ("subject_balanced_accuracy", "subject_macro_f1"):
            stats = _metric_summary(by_seed[metric].astype(float).tolist())
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        row.update(
            {
                "subject_macro_roc_auc_ovr_mean": None,
                "subject_macro_roc_auc_ovr_sample_sd": None,
                "subject_macro_roc_auc_ovr_repeat_t_ci95_low": None,
                "subject_macro_roc_auc_ovr_repeat_t_ci95_high": None,
                "subject_macro_roc_auc_ovr_applicability": (
                    "N/A_continuous_participant_oof_probabilities_not_archived"
                ),
                "formal_v2_p_value": None,
                "formal_v2_p_value_applicability": (
                    "N/A_participant_keyed_matched_oof_predictions_not_archived"
                ),
            }
        )
        output.append(row)
    return pandas.DataFrame(output)


def _factor_reference(values: Sequence[str], *, parameter: str) -> str:
    preferred_tokens = {
        "model": "inceptiontime",
        "quality_route_bundle": "SQI=none",
        "sampling_policy": "sampler=none",
    }
    token = preferred_tokens.get(parameter)
    if token is not None:
        matches = [value for value in values if token in value]
        if len(matches) == 1:
            return matches[0]
    numeric = [(_finite(value), value) for value in values]
    if all(number is not None for number, _value in numeric):
        return min(numeric, key=lambda item: float(item[0]))[1]
    return sorted(values)[0]


def _matched_marginal_factor_deltas(
    runs: Any,
    *,
    source_study: str,
    parameters: Sequence[str],
) -> tuple[Any, Any]:
    """Compute seed-matched marginal contrasts for a complete factorial grid."""

    pandas = _pd()
    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for parameter in parameters:
        if parameter not in runs:
            continue
        values = sorted(
            {_normalise_plot_value(value) for value in runs[parameter].dropna()}
        )
        if len(values) < 2:
            continue
        reference = _factor_reference(values, parameter=parameter)
        normalized = runs.copy()
        normalized["_factor_value"] = normalized[parameter].map(_normalise_plot_value)
        for candidate in (value for value in values if value != reference):
            comparison_id = f"{parameter}:{candidate}_vs_{reference}"
            ba_deltas: list[float] = []
            f1_deltas: list[float] = []
            for seed in _SPLIT_SEEDS:
                seed_rows = normalized.loc[normalized["seed"].astype(int).eq(seed)]
                left = seed_rows.loc[seed_rows["_factor_value"].eq(reference)]
                right = seed_rows.loc[seed_rows["_factor_value"].eq(candidate)]
                if left.empty or right.empty:
                    raise ValueError(
                        f"{source_study}/{comparison_id} is missing seed {seed}"
                    )
                reference_ba = float(left["subject_balanced_accuracy"].mean())
                candidate_ba = float(right["subject_balanced_accuracy"].mean())
                reference_f1 = float(left["subject_macro_f1"].mean())
                candidate_f1 = float(right["subject_macro_f1"].mean())
                ba_delta = candidate_ba - reference_ba
                f1_delta = candidate_f1 - reference_f1
                ba_deltas.append(ba_delta)
                f1_deltas.append(f1_delta)
                repeat_rows.append(
                    {
                        "source_study": source_study,
                        "comparison_id": comparison_id,
                        "parameter_group": _parameter_group(parameter),
                        "parameter": parameter,
                        "reference_value": reference,
                        "candidate_value": candidate,
                        "repeat": int(seed_rows["repeat"].iloc[0]),
                        "split_seed": seed,
                        "reference_config_count": int(
                            left["overfit_config_id"].nunique()
                        ),
                        "candidate_config_count": int(
                            right["overfit_config_id"].nunique()
                        ),
                        "reference_balanced_accuracy": reference_ba,
                        "candidate_balanced_accuracy": candidate_ba,
                        "balanced_accuracy_delta": ba_delta,
                        "reference_macro_f1": reference_f1,
                        "candidate_macro_f1": candidate_f1,
                        "macro_f1_delta": f1_delta,
                        "macro_roc_auc_ovr_delta": None,
                        "macro_roc_auc_ovr_applicability": (
                            "N/A_continuous_participant_oof_probabilities_not_archived"
                        ),
                        "comparison_role": (
                            "exploratory_seed_matched_marginal_factor_contrast_"
                            "not_participant_level_inference"
                        ),
                    }
                )
            row = {
                "source_study": source_study,
                "comparison_id": comparison_id,
                "parameter_group": _parameter_group(parameter),
                "parameter": parameter,
                "reference_value": reference,
                "candidate_value": candidate,
                "n_matched_repeat_seeds": len(ba_deltas),
                "aggregation_before_contrast": (
                    "equal_mean_across_all_nuisance_grid_cells_within_seed_and_factor_value"
                ),
            }
            for metric, values_for_metric in (
                ("balanced_accuracy_delta", ba_deltas),
                ("macro_f1_delta", f1_deltas),
            ):
                stats = _metric_summary(values_for_metric)
                for name, value in stats.items():
                    row[f"{metric}_{name}"] = value
            row.update(
                {
                    "macro_roc_auc_ovr_delta": None,
                    "participant_cluster_delta_ci95_low": None,
                    "participant_cluster_delta_ci95_high": None,
                    "formal_v2_p_value": None,
                    "formal_v2_inference_applicability": (
                        "N/A_participant_keyed_matched_oof_predictions_not_archived"
                    ),
                    "interpretation": (
                        "exploratory_complete_grid_marginal_effect; repeat_t_CI_is_not_"
                        "participant_cluster_CI"
                    ),
                }
            )
            summary_rows.append(row)
    return pandas.DataFrame(repeat_rows), pandas.DataFrame(summary_rows)


def _plot_delta_boxpanels(data: Any, path: Path, *, title: str) -> None:
    pyplot = _plt()
    categories = sorted(data["comparison_id"].astype(str).unique())
    height = max(5.0, min(15.0, 0.42 * len(categories) + 2.0))
    figure, axes = pyplot.subplots(1, 3, figsize=(18, height), sharey=True)
    fields = (
        ("balanced_accuracy_delta", "Balanced accuracy delta"),
        ("macro_f1_delta", "Macro-F1 delta"),
        ("macro_roc_auc_ovr_delta", "Macro ROC-AUC delta"),
    )
    for axis, (field, label) in zip(axes, fields, strict=True):
        if field not in data or not any(
            _finite(value) is not None for value in data[field]
        ):
            axis.text(
                0.5,
                0.5,
                "N/A\nOOF probabilities not archived",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            continue
        groups = [
            data.loc[data["comparison_id"].eq(category), field].astype(float).to_numpy()
            * 100.0
            for category in categories
        ]
        axis.boxplot(groups, vert=False, tick_labels=categories, showmeans=True)
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.set_xlabel("Candidate − reference (percentage points)")
        axis.set_title(label)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        title
        + "\nFive seed-matched marginal contrasts; descriptive, not participant inference"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def _plot_metric_boxpanels(data: Any, path: Path, *, title: str) -> None:
    pyplot = _plt()
    categories = sorted(data["plot_category"].dropna().astype(str).unique())
    height = max(4.8, min(15.0, 0.42 * len(categories) + 2.0))
    figure, axes = pyplot.subplots(1, 3, figsize=(18, height), sharey=True)
    for axis, (field, label) in zip(axes, _METRIC_FIELDS, strict=True):
        if field not in data or not any(
            _finite(value) is not None for value in data[field]
        ):
            axis.text(
                0.5,
                0.5,
                "N/A\ncontinuous participant-level\nOOF probabilities not archived",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            continue
        groups = [
            np.asarray(
                [
                    float(value) * 100.0
                    for value in data.loc[data["plot_category"].eq(category), field]
                    if _finite(value) is not None
                ],
                dtype=np.float64,
            )
            for category in categories
        ]
        positions = np.arange(1, len(categories) + 1)
        axis.boxplot(groups, vert=False, tick_labels=categories, showmeans=True)
        for position, values in zip(positions, groups, strict=True):
            if not len(values):
                continue
            offsets = (
                np.linspace(-0.08, 0.08, len(values))
                if len(values) > 1
                else np.zeros(1)
            )
            axis.scatter(values, position + offsets, s=12, alpha=0.45, zorder=3)
        axis.set_xlabel("Score (%)")
        axis.set_title(label)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        title
        + "\nDescriptive config-repeat distributions; shared participants/configuration confounding"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def _plot_runtime(data: Any, path: Path, *, title: str) -> None:
    pyplot = _plt()
    categories = sorted(data["model_display"].unique())
    groups = [
        data.loc[data["model_display"].eq(category), "duration_sec"].astype(float)
        for category in categories
    ]
    figure, axis = pyplot.subplots(figsize=(8, 4.8))
    axis.boxplot(groups, tick_labels=categories, showmeans=True)
    axis.set_yscale("log")
    axis.set_ylabel("Archived run duration (s; log scale)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def _leaderboard_display(summary: Any, *, config_column: str, model_column: str) -> Any:
    pandas = _pd()
    rows: list[dict[str, Any]] = []
    for _, item in summary.iterrows():
        rows.append(
            {
                "config_or_model": item.get(config_column),
                "model": item.get(model_column),
                "rank": int(item.get("descriptive_rank", len(rows) + 1)),
                "subject_BA_mean_sd_percent": format_mean_sd(
                    item.get("subject_balanced_accuracy_mean"),
                    item.get("subject_balanced_accuracy_sample_sd"),
                    percent=True,
                ),
                "subject_BA_repeat_t_CI95_percent": format_interval(
                    item.get("subject_balanced_accuracy_repeat_t_ci95_low"),
                    item.get("subject_balanced_accuracy_repeat_t_ci95_high"),
                    percent=True,
                ),
                "subject_macro_F1_mean_sd_percent": format_mean_sd(
                    item.get("subject_macro_f1_mean"),
                    item.get("subject_macro_f1_sample_sd"),
                    percent=True,
                ),
                "subject_macro_F1_repeat_t_CI95_percent": format_interval(
                    item.get("subject_macro_f1_repeat_t_ci95_low"),
                    item.get("subject_macro_f1_repeat_t_ci95_high"),
                    percent=True,
                ),
                "subject_macro_ROC_AUC": "N/A",
                "ROC_AUC_applicability": (
                    "continuous participant-level OOF probabilities not archived"
                ),
                "scientific_role": "historical_hypothesis_generation_only",
            }
        )
    return pandas.DataFrame(rows)


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if len(columns) > _MAX_HUMAN_TABLE_COLUMNS:
        raise ValueError(
            f"human-facing historical table has {len(columns)} columns; "
            f"maximum is {_MAX_HUMAN_TABLE_COLUMNS}"
        )
    definitions = html_column_definitions_block(columns)
    if not rows:
        return "<p><em>No rows.</em></p>" + definitions
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body: list[str] = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                rendered = "N/A"
            elif isinstance(value, float):
                rendered = f"{value:.4f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-scroll'><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        + definitions
    )


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
            value = row.get(column)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                value = "N/A"
            elif isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(("", markdown_column_definitions_block(columns)))
    return "\n".join(lines)


def _historical_display_tables(
    leaderboard: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Sequence[Mapping[str, Any]], tuple[str, ...]], ...]:
    """Return one shared Markdown/HTML schema for historical report tables."""

    return (
        (
            "Performance estimates",
            leaderboard,
            _LEADERBOARD_PERFORMANCE_COLUMNS,
        ),
        (
            "Probability-metric applicability",
            leaderboard,
            _LEADERBOARD_APPLICABILITY_COLUMNS,
        ),
        (
            "Parameter groups and observed values",
            inventory,
            _PARAMETER_INVENTORY_COLUMNS,
        ),
    )


def _md_display_tables(
    tables: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[str]]],
) -> str:
    return "\n\n".join(
        f"### {title}\n\n{_md_table(rows, columns)}" for title, rows, columns in tables
    )


def _html_display_tables(
    tables: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[str]]],
) -> str:
    return "".join(
        f"<h3>{html.escape(title)}</h3>" + _html_table(rows, columns)
        for title, rows, columns in tables
    )


def _source_evidence(source: Path, report_paths: Sequence[Path]) -> Any:
    pandas = _pd()
    rows: list[dict[str, Any]] = []
    for name in (
        "sweep_manifest.json",
        "sweep_runs.csv",
        "sweep_summary.csv",
        "overfitting_manifest.json",
        "overfitting_runs.csv",
        "overfitting_summary.csv",
    ):
        path = source / name
        if path.is_file():
            rows.append(
                {
                    "source_study": source.name,
                    "source_kind": "root_manifest_or_run_table",
                    "source_file": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    for path in sorted(set(report_paths), key=lambda item: str(item).encode("utf-8")):
        rows.append(
            {
                "source_study": source.name,
                "source_kind": "consumed_report_json",
                "source_file": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pandas.DataFrame(rows)


def _write_tables(target: Path, tables: Mapping[str, Any]) -> None:
    table_dir = target / "tables"
    for name, frame in tables.items():
        rows = _as_records(frame) if hasattr(frame, "to_dict") else list(frame)
        write_csv(table_dir / f"{name}.csv", rows)


def _table_or_status(frame: Any, *, table_name: str, reason: str) -> Any:
    """Keep N/A tables machine-readable instead of writing a headerless file."""

    if not getattr(frame, "empty", False):
        return frame
    pandas = _pd()
    return pandas.DataFrame(
        [
            {
                "table": table_name,
                "status": "N/A",
                "reason": reason,
                "scientific_role": "historical_hypothesis_generation_only",
            }
        ]
    )


def _write_common_support_files(
    target: Path,
    *,
    sources: Sequence[Path],
    report_kind: str,
    interpretation_lines: Sequence[str],
    component_lines: Sequence[str],
) -> None:
    (target / "RESULT_INTERPRETATION.md").write_text(
        "# Result interpretation\n\n"
        + "\n".join(f"- {line}" for line in interpretation_lines)
        + "\n",
        encoding="utf-8",
    )
    (target / "TEST_COMPONENTS.md").write_text(
        "# Test components and fixed inputs\n\n"
        + "\n".join(f"- {line}" for line in component_lines)
        + "\n",
        encoding="utf-8",
    )
    methods = [
        "# Historical reporter methods",
        "",
        "- No model was retrained and no archived source was modified.",
        "- Unit for repeat summaries: one archived complete participant-grouped 5-fold repeat (n=5).",
        "- Display: arithmetic mean ± sample SD across repeat seeds.",
        "- Descriptive CI95: mean ± t(0.975, n−1) × sample SD / sqrt(n). Repeated CV estimates are correlated, so this interval is descriptive rather than an unbiased generalization-error interval.",
        "- Ranking: subject balanced accuracy, then subject macro-F1; all rankings are post-hoc and selection-contaminated.",
        "- Parameter boxplots use archived config-repeat summaries. They share participants and often differ in multiple parameters; they show marginal associations, not causal single-factor effects.",
        "- Formal V2 participant-cluster CI would resample participant IDs within true-class strata, carry all rows for each sampled participant across repeats, recompute each repeat metric, equally average repeats, and take the 2.5th/97.5th percentiles. Required participant-keyed OOF rows are absent, so these cells remain N/A.",
        "- ROC-AUC requires continuous per-class participant OOF probabilities. Those probabilities were not archived; the ROC-AUC panel is intentionally marked N/A and no hard-label surrogate is invented.",
        "- Every displayed table has a CSV, every root CSV becomes one workbook sheet, every plot has a CSV data partner, and every table column has a generated definition/formula catalog.",
        "",
        "## References",
        "",
        "- Student (1908), *The Probable Error of a Mean*, Biometrika 6:1–25.",
        "- Brodersen et al. (2010), *The Balanced Accuracy and Its Posterior Distribution*, ICPR.",
        "- Sokolova & Lapalme (2009), *A systematic analysis of performance measures for classification tasks*, Information Processing & Management 45:427–437.",
        "- Fawcett (2006), *An introduction to ROC analysis*, Pattern Recognition Letters 27:861–874.",
        "- Bengio & Grandvalet (2004), *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, JMLR 5:1089–1105.",
        "- Varma & Simon (2006), *Bias in error estimation when using cross-validation for model selection*, BMC Bioinformatics 7:91.",
        "- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11:2079–2107.",
        "",
    ]
    (target / "REPORT_METHODS.md").write_text("\n".join(methods), encoding="utf-8")
    write_table_column_definitions(target / "tables", csv_directory=target / "tables")
    write_excel_workbook_from_csv_directory(
        target / "tables" / "report_tables.xlsx", target / "tables"
    )
    artifacts = [
        {
            "path": str(path.relative_to(target)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            (path for path in target.rglob("*") if path.is_file()),
            key=lambda item: str(item.relative_to(target)).encode("utf-8"),
        )
    ]
    (target / "outputs_index.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.historical_split_output_index.v1",
                "artifacts_excluding_this_index_and_manifest": artifacts,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "ppg_frailty.historical_split_report.v1",
        "report_kind": report_kind,
        "status": "descriptive_reanalysis_complete",
        "scientific_status": "historical_hypothesis_generation_only",
        "sources": [str(source) for source in sources],
        "outputs_index": "outputs_index.json",
    }
    (target / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _html_document(
    *,
    title: str,
    subtitle: str,
    leaderboard: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    figures: Sequence[Mapping[str, Any]],
    conclusions: Sequence[str],
) -> str:
    display_tables = _historical_display_tables(leaderboard, inventory)
    figure_html = "".join(
        "<figure><img src='"
        + html.escape(str(row["figure_path"]))
        + "' alt='"
        + html.escape(str(row["title"]))
        + "'><figcaption>"
        + html.escape(str(row["title"]))
        + " — numeric source: <a href='"
        + html.escape(str(row["table_path"]))
        + "'>"
        + html.escape(str(row["table_path"]))
        + "</a>. "
        + html.escape(str(row.get("interpretation", "descriptive only")))
        + "</figcaption></figure>"
        for row in figures
    )
    conclusions_html = "".join(f"<li>{html.escape(line)}</li>" for line in conclusions)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1.25rem;color:#17202a;line-height:1.48}}
h1,h2{{color:#17365d}} .notice{{background:#fff4ce;border-left:5px solid #d99b00;padding:1rem}}
table{{border-collapse:collapse;font-size:.86rem;width:100%}} th,td{{border:1px solid #ccd3db;padding:.42rem;vertical-align:top}} th{{background:#eaf0f7;position:sticky;top:0}}
.table-scroll{{overflow:auto;max-height:650px}} figure{{margin:2rem 0;border:1px solid #d8dee6;padding:1rem}} img{{max-width:100%;height:auto}} figcaption{{font-size:.9rem;color:#445}}
code{{white-space:pre-wrap}} details{{margin:.6rem 0 1.4rem}}
</style></head><body>
<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>
<div class="notice"><strong>Evidential status:</strong> historical post-hoc candidate/hypothesis generation. Repeat-t intervals are descriptive. Participant-cluster CI, formal participant-exchange P, and ROC-AUC are N/A because participant-keyed OOF probabilities were not archived.</div>
<h2>Conclusions and writing role</h2><ul>{conclusions_html}</ul>
<h2>V2-style results and parameter audit</h2>{_html_display_tables(display_tables)}
<h2>Metric and parameter comparison plots</h2>{figure_html}
<h2>Audit files</h2><ul><li><a href="tables/report_tables.xlsx">One-sheet-per-table workbook</a></li><li><a href="REPORT_METHODS.md">Methods and references</a></li><li><a href="TEST_COMPONENTS.md">Models, inputs and fixed parameters</a></li><li><a href="RESULT_INTERPRETATION.md">Result interpretation</a></li><li><a href="tables/TABLE_COLUMN_DEFINITIONS.md">Column definitions and formulas</a></li></ul>
</body></html>"""


def _early_report(
    target: Path,
    *,
    early_source: Path,
    shapeformer_source: Path,
    window_seconds: float,
    overlap_percent: float,
    patience: int,
    extra_input: str,
) -> dict[str, Any]:
    pandas = _pd()
    target.mkdir(parents=True)
    (target / "tables").mkdir()
    (target / "figures").mkdir()
    (
        repeats,
        summary,
        parameters,
        per_class,
        split_audit,
        exploratory_tests,
    ) = _early_contract_rows(
        early_source,
        shapeformer_source,
        window_seconds=window_seconds,
        overlap_percent=overlap_percent,
        patience=patience,
        extra_input=extra_input,
    )
    repeat_deltas, participant_inference = _early_pairwise_repeat_deltas(repeats)
    repeats = repeats.copy()
    repeats["subject_macro_roc_auc_ovr"] = np.nan
    model_plot = _evaluation_level_metric_rows(
        repeats,
        source_study="matched_early_three_model",
        parameter="model_display",
        subset_label="5s_50pct_overlap_patience20_no_extra_ppi_hrv",
        evaluation_level="subject",
    )
    model_plot["plot_category"] = model_plot["parameter_value"]
    plot_path = target / "figures" / "01_three_model_ba_f1_roc_boxplots.png"
    _plot_metric_boxpanels(
        model_plot,
        plot_path,
        title="CNN1D vs InceptionTime vs ShapeFormer-PISD (matched 5×5 archive)",
    )
    _plot_runtime(
        repeats,
        target / "figures" / "02_three_model_runtime_boxplot.png",
        title="Archived computational burden",
    )
    file_plot = _evaluation_level_metric_rows(
        repeats,
        source_study="matched_early_three_model",
        parameter="model_display",
        subset_label="5s_50pct_overlap_patience20_no_extra_ppi_hrv",
        evaluation_level="file",
    )
    file_plot["plot_category"] = file_plot["parameter_value"]
    _plot_metric_boxpanels(
        file_plot,
        target / "figures" / "03_three_model_file_ba_f1_roc_boxplots.png",
        title="Matched three-model file-level metrics",
    )
    window_plot = _evaluation_level_metric_rows(
        repeats,
        source_study="matched_early_three_model",
        parameter="model_display",
        subset_label="5s_50pct_overlap_patience20_no_extra_ppi_hrv",
        evaluation_level="window",
    )
    window_plot["plot_category"] = window_plot["parameter_value"]
    _plot_metric_boxpanels(
        window_plot,
        target / "figures" / "04_three_model_window_ba_f1_roc_boxplots.png",
        title="Matched three-model window-level metrics",
    )

    shape = repeats.loc[repeats["model_display"].eq("ShapeFormer-PISD")].set_index(
        "seed"
    )
    evidence_rows: list[dict[str, Any]] = []
    for comparator in ("CNN1D", "InceptionTime"):
        other = repeats.loc[repeats["model_display"].eq(comparator)].set_index("seed")
        for seed in _SPLIT_SEEDS:
            evidence_rows.append(
                {
                    "comparison": f"ShapeFormer-PISD_vs_{comparator}",
                    "repeat": int(shape.loc[seed, "repeat"]),
                    "split_seed": seed,
                    "shapeformer_subject_BA": float(
                        shape.loc[seed, "subject_balanced_accuracy"]
                    ),
                    "comparator_subject_BA": float(
                        other.loc[seed, "subject_balanced_accuracy"]
                    ),
                    "shapeformer_minus_comparator_BA": float(
                        shape.loc[seed, "subject_balanced_accuracy"]
                        - other.loc[seed, "subject_balanced_accuracy"]
                    ),
                    "shapeformer_subject_macro_F1": float(
                        shape.loc[seed, "subject_macro_f1"]
                    ),
                    "comparator_subject_macro_F1": float(
                        other.loc[seed, "subject_macro_f1"]
                    ),
                    "shapeformer_minus_comparator_macro_F1": float(
                        shape.loc[seed, "subject_macro_f1"]
                        - other.loc[seed, "subject_macro_f1"]
                    ),
                    "shapeformer_runtime_seconds": float(
                        shape.loc[seed, "duration_sec"]
                    ),
                    "comparator_runtime_seconds": float(
                        other.loc[seed, "duration_sec"]
                    ),
                    "shapeformer_runtime_ratio": float(
                        shape.loc[seed, "duration_sec"]
                        / other.loc[seed, "duration_sec"]
                    ),
                }
            )
    shape_evidence = pandas.DataFrame(evidence_rows)
    shape_summary_rows: list[dict[str, Any]] = []
    for comparison, group in shape_evidence.groupby("comparison", sort=False):
        row: dict[str, Any] = {
            "comparison": comparison,
            "n_matched_repeat_seeds": len(group),
            "difference_direction": "shapeformer_minus_comparator",
            "formal_v2_inference": False,
        }
        for metric in (
            "shapeformer_minus_comparator_BA",
            "shapeformer_minus_comparator_macro_F1",
            "shapeformer_runtime_ratio",
        ):
            stats = _metric_summary(group[metric].astype(float).tolist())
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        row.update(
            {
                "participant_cluster_delta_ci95_low": None,
                "participant_cluster_delta_ci95_high": None,
                "participant_cluster_ci_applicability": (
                    "N/A_participant_keyed_oof_predictions_not_archived"
                ),
            }
        )
        shape_summary_rows.append(row)
    shape_decision_summary = pandas.DataFrame(shape_summary_rows)
    shape_delta_plot = shape_evidence.rename(
        columns={
            "comparison": "comparison_id",
            "shapeformer_minus_comparator_BA": "balanced_accuracy_delta",
            "shapeformer_minus_comparator_macro_F1": "macro_f1_delta",
        }
    ).copy()
    shape_delta_plot["macro_roc_auc_ovr_delta"] = np.nan
    _plot_delta_boxpanels(
        shape_delta_plot,
        target / "figures" / "05_shapeformer_paired_delta_boxplots.png",
        title="ShapeFormer-PISD matched deltas against ordinary-model comparators",
    )
    leaderboard = _leaderboard_display(
        summary,
        config_column="model_display",
        model_column="resolved_model",
    )
    early_inventory = pandas.DataFrame(
        [
            {
                "source_study": "matched_early_three_model",
                "parameter_group": "matched_contract",
                "parameter": name,
                "unique_value_count": 1,
                "observed_values": value,
                "parameter_role": "fixed",
                "comparison_interpretation": "matched_across_all_three_models",
            }
            for name, value in (
                ("window_seconds", window_seconds),
                ("hop_seconds", window_seconds * (1.0 - overlap_percent / 100.0)),
                ("overlap_percent", overlap_percent),
                ("patience", patience),
                ("extra_input", extra_input),
                ("split_seeds", json.dumps(_SPLIT_SEEDS)),
            )
        ]
    )
    report_paths = [
        _report_path(Path(row.source_path), row.report_path)
        for row in repeats.itertuples()
    ]
    source_evidence = pandas.concat(
        [
            _source_evidence(
                source,
                [
                    path
                    for path in report_paths
                    if path.is_relative_to((source / "reports").resolve())
                ],
            )
            for source in (early_source, shapeformer_source)
        ],
        ignore_index=True,
    )
    absolute_cluster = _absolute_historical_cluster_ci_rows(
        summary,
        source_study="matched_early_three_model",
        classifier_id_column="model_display",
        model_column="resolved_model",
    )
    figures = pandas.DataFrame(
        [
            {
                "figure_id": "early_three_model_metrics",
                "title": "Matched model BA / macro-F1 / ROC-AUC comparison",
                "figure_path": "figures/01_three_model_ba_f1_roc_boxplots.png",
                "table_path": "tables/plot_01_three_model_metrics.csv",
                "interpretation": "BA/F1 use five matched repeats; ROC-AUC is explicit N/A.",
            },
            {
                "figure_id": "early_runtime",
                "title": "Archived run-time comparison",
                "figure_path": "figures/02_three_model_runtime_boxplot.png",
                "table_path": "tables/early_three_model_repeat_metrics.csv",
                "interpretation": "Runtime is descriptive hardware/workflow evidence, not model accuracy.",
            },
            {
                "figure_id": "early_file_metrics",
                "title": "Matched file-level BA / macro-F1 / ROC-AUC comparison",
                "figure_path": "figures/03_three_model_file_ba_f1_roc_boxplots.png",
                "table_path": "tables/plot_03_three_model_file_metrics.csv",
                "interpretation": "File BA/F1 are archived; ROC-AUC is explicit N/A.",
            },
            {
                "figure_id": "early_window_metrics",
                "title": "Matched window-level BA / macro-F1 / ROC-AUC comparison",
                "figure_path": "figures/04_three_model_window_ba_f1_roc_boxplots.png",
                "table_path": "tables/plot_04_three_model_window_metrics.csv",
                "interpretation": "Windows are nested within files/participants; this is not an independent-unit inference.",
            },
            {
                "figure_id": "shapeformer_paired_deltas",
                "title": "ShapeFormer-PISD paired BA / macro-F1 / ROC-AUC deltas",
                "figure_path": "figures/05_shapeformer_paired_delta_boxplots.png",
                "table_path": "tables/plot_05_shapeformer_paired_deltas.csv",
                "interpretation": "Five aggregate repeat deltas per comparator; participant-cluster inference is unavailable.",
            },
        ]
    )
    missing = pandas.DataFrame(_missing_statistics())
    tables = {
        "leaderboard_display": leaderboard,
        "early_three_model_repeat_metrics": repeats,
        "early_three_model_summary_numeric": summary,
        "early_three_model_parameters": parameters,
        "early_three_model_per_class_summary": per_class,
        "early_three_model_split_audit": split_audit,
        "early_three_model_exploratory_paired_tests": exploratory_tests,
        "pairwise_repeat_metric_deltas": repeat_deltas,
        "paired_participant_inference": participant_inference,
        "shapeformer_decision_evidence": shape_evidence,
        "shapeformer_decision_summary": shape_decision_summary,
        "plot_05_shapeformer_paired_deltas": shape_delta_plot,
        "parameter_inventory": early_inventory,
        "plot_01_three_model_metrics": model_plot,
        "plot_03_three_model_file_metrics": file_plot,
        "plot_04_three_model_window_metrics": window_plot,
        "table_figure_pairs": figures,
        "historical_absolute_participant_cluster_ci": absolute_cluster,
        "missing_v2_statistics": missing,
        "source_evidence": source_evidence,
    }
    _write_tables(target, tables)
    leader = summary.iloc[0]
    shape_summary = summary.loc[summary["model_display"].eq("ShapeFormer-PISD")].iloc[0]
    all_ba_lower = bool((shape_evidence["shapeformer_minus_comparator_BA"] < 0).all())
    all_f1_lower = bool(
        (shape_evidence["shapeformer_minus_comparator_macro_F1"] < 0).all()
    )
    runtime_ratio = float(
        shape_summary["duration_sec_mean"] / leader["duration_sec_mean"]
    )
    conclusions = [
        f"Descriptive leader: {leader['model_display']} with subject BA {100*leader['subject_balanced_accuracy_mean']:.1f} ± {100*leader['subject_balanced_accuracy_sample_sd']:.1f}% and macro-F1 {100*leader['subject_macro_f1_mean']:.1f} ± {100*leader['subject_macro_f1_sample_sd']:.1f}%.",
        f"ShapeFormer-PISD trails the leader by {100*(leader['subject_balanced_accuracy_mean']-shape_summary['subject_balanced_accuracy_mean']):.1f} BA points and {100*(leader['subject_macro_f1_mean']-shape_summary['subject_macro_f1_mean']):.1f} macro-F1 points.",
        f"ShapeFormer is below both comparators on BA in all 10 matched pair/repeat rows={all_ba_lower} and on macro-F1 in all 10 rows={all_f1_lower}; its mean runtime is {runtime_ratio:.1f}× the leader.",
        "This supports excluding this historical ShapeFormer-PISD implementation from the ordinary mega-study on utility/cost grounds; it does not establish that every ShapeFormer implementation is inferior.",
        "The exact five-repeat sign-flip P values are exploratory; only 32 sign patterns exist, so the minimum attainable two-sided P is 0.0625.",
        "The held-out fold supplied the historical best-epoch/early-stopping trajectory and the reported score, creating selection contamination; absolute scores are therefore candidate-generation evidence, not selection-unbiased OOF confirmation.",
    ]
    components = [
        f"Sources: {early_source}; {shapeformer_source}",
        "Models: CNN1D, InceptionTime, ShapeFormer-PISD",
        f"Matched filter: window={window_seconds:g}s; overlap={overlap_percent:g}%; patience={patience}; extra_input={extra_input}",
        "Input columns: RED, IR, AX, AY, AZ, GX, GY, GZ; historical DL view includes filtering, 64 Hz resampling and per-window robust scaling",
        "Files/participants/windows per selected run: 145 / 29 / 15,657",
        f"Split seeds: {', '.join(str(seed) for seed in _SPLIT_SEEDS)}; participant-grouped 5-fold rosters match exactly",
        "Epoch budget: up to 50 with historical held-out-fold validation trajectory and patience 20",
        "No PPI/HRV extra features",
    ]
    display_rows = _as_records(leaderboard)
    inventory_rows = _as_records(early_inventory)
    display_tables = _historical_display_tables(display_rows, inventory_rows)
    md = [
        "# Matched historical CNN1D–InceptionTime–ShapeFormer report",
        "",
        "> Historical post-hoc candidate evidence; not a confirmatory V2 test.",
        "",
        "## V2-style results and parameter audit",
        "",
        _md_display_tables(display_tables),
        "",
        "## Conclusions",
        "",
        *(f"- {line}" for line in conclusions),
        "",
        "## Missing V2 calculations",
        "",
        "ROC-AUC/ROC curves, PR-AUC, participant-cluster CI and formal participant-exchange P are N/A because participant-keyed OOF class probabilities were not archived. They are not reconstructed from aggregate confusion matrices.",
        "",
        "See `STUDY_SUMMARY.html` for all paired plots and `tables/report_tables.xlsx` for one worksheet per table.",
        "",
    ]
    (target / "STUDY_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    (target / "STUDY_SUMMARY.html").write_text(
        _html_document(
            title="Matched historical CNN1D–InceptionTime–ShapeFormer report",
            subtitle="5 s, 50% overlap, patience 20, no extra PPI/HRV input; 3 models × 5 matched repeat seeds.",
            leaderboard=display_rows,
            inventory=inventory_rows,
            figures=_as_records(figures),
            conclusions=conclusions,
        ),
        encoding="utf-8",
    )
    _write_common_support_files(
        target,
        sources=(early_source, shapeformer_source),
        report_kind="matched_early_three_model",
        interpretation_lines=conclusions,
        component_lines=components,
    )
    return {
        "report_id": target.name,
        "path": str(target),
        "source_studies": f"{early_source.name};{shapeformer_source.name}",
        "configuration_count": 3,
        "run_count": len(repeats),
        "leader": str(leader["model_display"]),
        "leader_subject_ba_mean": float(leader["subject_balanced_accuracy_mean"]),
        "scientific_status": "historical_hypothesis_generation_only",
    }


def _overfit_report(target: Path, *, source: Path, writing_role: str) -> dict[str, Any]:
    pandas = _pd()
    target.mkdir(parents=True)
    (target / "tables").mkdir()
    (target / "figures").mkdir()
    runs = _derive_design_factors(pandas.read_csv(source / "overfitting_runs.csv"))
    runs["subject_macro_roc_auc_ovr"] = np.nan
    analysis_runs = runs
    fixed_reference_exclusion = False
    if "overfit_stage" in runs and "generalization" in set(
        runs["overfit_stage"].dropna().astype(str)
    ):
        analysis_runs = runs.loc[
            runs["overfit_stage"].astype(str).eq("generalization")
        ].copy()
        fixed_reference_exclusion = len(analysis_runs) != len(runs)
    (
        summary,
        top,
        effects,
        archived_contract,
        split_audit,
        per_class_repeats,
        per_class_summary,
        factor_repeat_deltas,
        factor_participant_inference,
    ) = _overfit_summary(source)
    leaderboard = _leaderboard_display(
        summary,
        config_column="overfit_config_id",
        model_column="resolved_model",
    )
    parameter_inventory = _parameter_inventory(runs, source_study=source.name)
    dependency_warnings = _design_dependency_warnings(
        analysis_runs, source_study=source.name
    )
    configuration_parameters = _configuration_parameter_table(
        runs, source_study=source.name, source=source
    )
    absolute_cluster = _absolute_historical_cluster_ci_rows(
        summary,
        source_study=source.name,
        classifier_id_column="overfit_config_id",
        model_column="resolved_model",
    )
    report_paths = tuple(_report_path(source, raw) for raw in runs["report_path"])
    source_evidence = _source_evidence(source, report_paths)
    missing = pandas.DataFrame(_missing_statistics())
    tables: dict[str, Any] = {
        "leaderboard_display": leaderboard,
        "all_config_summary_numeric": summary,
        "top15_config_summary_numeric": top,
        "all_config_repeat_metrics": runs,
        "configuration_parameters": configuration_parameters,
        "parameter_inventory": parameter_inventory,
        "parameter_dependency_warnings": dependency_warnings,
        "factor_signals": _table_or_status(
            effects,
            table_name="factor_signals",
            reason="archive has no declared same-epoch baseline factor rows for this design",
        ),
        "per_class_repeat_results": per_class_repeats,
        "classifier_per_class_results": per_class_summary,
        "factor_pairwise_repeat_metric_deltas": _table_or_status(
            factor_repeat_deltas,
            table_name="factor_pairwise_repeat_metric_deltas",
            reason="archive has no declared baseline-vs-candidate factor signal pairs",
        ),
        "factor_paired_participant_inference": _table_or_status(
            factor_participant_inference,
            table_name="factor_paired_participant_inference",
            reason="no declared factor pairs; participant-keyed OOF predictions are also absent",
        ),
        "historical_absolute_participant_cluster_ci": absolute_cluster,
        "archived_parameter_contract": archived_contract,
        "split_audit": split_audit,
        "missing_v2_statistics": missing,
        "source_evidence": source_evidence,
    }
    figures: list[dict[str, Any]] = []
    top_ids = set(top["overfit_config_id"].astype(str))
    top_runs = runs.loc[runs["overfit_config_id"].astype(str).isin(top_ids)].copy()
    top_plot = _long_metric_rows(
        top_runs,
        source_study=source.name,
        parameter="overfit_config_id",
        subset_label="top15_by_subject_BA_then_macro_F1",
    )
    top_plot["plot_category"] = top_plot["parameter_value"]
    tables["plot_01_top15_configs"] = top_plot
    _plot_metric_boxpanels(
        top_plot,
        target / "figures" / "01_top15_config_ba_f1_roc_boxplots.png",
        title=f"{source.name}: top 15 observed configurations",
    )
    figures.append(
        {
            "figure_id": "top15_configs",
            "title": "Top-15 configuration BA / macro-F1 / ROC-AUC boxplots",
            "figure_path": "figures/01_top15_config_ba_f1_roc_boxplots.png",
            "table_path": "tables/plot_01_top15_configs.csv",
            "interpretation": "Post-hoc top-15 display; ROC-AUC unavailable.",
        }
    )
    combined_parameter_rows: list[dict[str, Any]] = []
    for index, (parameter, subset_label, subset) in enumerate(
        _plot_specs(analysis_runs), start=2
    ):
        plot_data = _long_metric_rows(
            runs,
            source_study=source.name,
            parameter=parameter,
            subset_label=subset_label,
            subset=subset,
        )
        combined_parameter_rows.extend(_as_records(plot_data))
        stem = f"plot_{index:02d}_{_slug(parameter + '_' + subset_label)}"
        tables[stem] = plot_data
        figure_name = (
            f"{index:02d}_{_slug(parameter + '_' + subset_label)}_boxplots.png"
        )
        _plot_metric_boxpanels(
            plot_data,
            target / "figures" / figure_name,
            title=f"{source.name}: {parameter} ({subset_label})",
        )
        figures.append(
            {
                "figure_id": stem,
                "title": f"Parameter comparison: {parameter} — {subset_label}",
                "figure_path": f"figures/{figure_name}",
                "table_path": f"tables/{stem}.csv",
                "interpretation": (
                    "Marginal config-repeat association; shared participants and "
                    "uncontrolled co-parameters prevent a causal ablation claim."
                ),
            }
        )
    parameter_metric_long = pandas.DataFrame(combined_parameter_rows)
    tables["parameter_value_metric_long"] = parameter_metric_long
    tables["parameter_value_metric_summary"] = _parameter_value_metric_summary(
        parameter_metric_long
    )
    if fixed_reference_exclusion:
        (
            marginal_repeat_deltas,
            marginal_delta_summary,
        ) = _matched_marginal_factor_deltas(
            analysis_runs,
            source_study=source.name,
            parameters=(
                "model",
                "cnn_epochs",
                "regularization_bundle",
                "quality_route_bundle",
                "train_overlap_pct",
                "sampling_policy",
            ),
        )
        tables["generalization_matched_marginal_repeat_deltas"] = marginal_repeat_deltas
        tables["generalization_matched_marginal_delta_summary"] = marginal_delta_summary
        delta_index = len(figures) + 1
        delta_name = f"{delta_index:02d}_matched_marginal_factor_delta_boxplots.png"
        _plot_delta_boxpanels(
            marginal_repeat_deltas,
            target / "figures" / delta_name,
            title=f"{source.name}: complete-grid marginal factor contrasts",
        )
        figures.append(
            {
                "figure_id": "generalization_matched_marginal_deltas",
                "title": "Matched marginal factor BA / macro-F1 / ROC-AUC deltas",
                "figure_path": f"figures/{delta_name}",
                "table_path": "tables/generalization_matched_marginal_repeat_deltas.csv",
                "interpretation": (
                    "Each seed first averages nuisance grid cells; CI/P remain descriptive "
                    "repeat evidence, not participant-level inference."
                ),
            }
        )
    figure_frame = pandas.DataFrame(figures)
    tables["table_figure_pairs"] = figure_frame
    _write_tables(target, tables)

    leader = summary.iloc[0]
    runner = summary.iloc[1]
    leader_id = str(leader["overfit_config_id"])
    runner_id = str(runner["overfit_config_id"])
    conclusions = [
        f"Point leader: {leader_id}, subject BA {100*leader['subject_balanced_accuracy_mean']:.1f} ± {100*leader['subject_balanced_accuracy_sample_sd']:.1f}% and macro-F1 {100*leader['subject_macro_f1_mean']:.1f} ± {100*leader['subject_macro_f1_sample_sd']:.1f}%.",
        f"Runner-up: {runner_id}, subject BA {100*runner['subject_balanced_accuracy_mean']:.1f} ± {100*runner['subject_balanced_accuracy_sample_sd']:.1f}%; the same archive searched {len(summary)} configurations, so neither row is an unbiased final estimate.",
        f"Writing role: {writing_role}",
        "Parameter-value boxplots are descriptive marginal views. A parameter can appear favorable because other settings differ; only later matched V2 ablations can support a component-effect claim.",
        "Repeat Student-t CI95 is available; participant-cluster CI, formal V2 P, ROC-AUC and ROC curves are N/A because participant-keyed OOF probabilities were not archived.",
    ]
    if fixed_reference_exclusion:
        conclusions.append(
            "Fixed historical references remain in the overall leaderboard as anchors but are excluded from generalization-factor boxplots; those plots use the new generalization grid only."
        )
    for warning in _as_records(dependency_warnings):
        conclusions.append(
            f"Design dependency — {warning['design_factor']}: {warning['reporting_rule']}."
        )
    stage_leaders: list[str] = []
    if "overfit_stage" in summary:
        for stage, rows in summary.groupby("overfit_stage", dropna=False, sort=False):
            item = rows.sort_values(
                ["subject_balanced_accuracy_mean", "subject_macro_f1_mean"],
                ascending=False,
            ).iloc[0]
            stage_leaders.append(
                f"{stage}: {item['overfit_config_id']} ({100*item['subject_balanced_accuracy_mean']:.1f}% BA)"
            )
        conclusions.append(
            "Stage-specific point leaders — " + "; ".join(stage_leaders) + "."
        )
    components = [
        f"Source: {source}",
        f"Models/resolved models: {', '.join(sorted(runs['resolved_model'].astype(str).unique()))}",
        f"Configurations/runs: {runs['overfit_config_id'].nunique()} / {len(runs)}",
        "Input: 29-participant internal dataset; columns RED, IR, AX, AY, AZ, GX, GY, GZ; detailed per-config roles and processing fields are in configuration_parameters.csv",
        f"Split seeds: {', '.join(str(seed) for seed in _SPLIT_SEEDS)}; participant-grouped 5 folds; all configs share the same roster per seed",
        "All observed fixed and varying parameter values are preserved in parameter_inventory.csv and configuration_parameters.csv",
        "No continuous participant OOF probabilities were found",
    ]
    for field, label in (
        ("report_config__fs", "native sampling rate(s)"),
        ("report_config__cnn_target_fs", "DL target sampling rate(s)"),
        ("report_config__cnn_batch_size", "batch size(s)"),
        ("report_config__cnn_select_best_epoch", "best-epoch selection flag(s)"),
    ):
        if field in configuration_parameters:
            values = sorted(
                {
                    _display_value(value)
                    for value in configuration_parameters[field].dropna().tolist()
                }
            )
            components.append(f"{label}: {', '.join(values)}")
    display_rows = _as_records(leaderboard.head(15))
    inventory_rows = _as_records(parameter_inventory)
    display_tables = _historical_display_tables(display_rows, inventory_rows)
    md = [
        f"# Historical V2-style report: {source.name}",
        "",
        "> Historical post-hoc search evidence; not a confirmatory V2 test.",
        "",
        "## Top-15 V2-style results and parameter audit",
        "",
        _md_display_tables(display_tables),
        "",
        "## Conclusions and writing role",
        "",
        *(f"- {line}" for line in conclusions),
        "",
        "## Parameter comparison caution",
        "",
        "Every varied parameter and factor group is recorded. Boxplots use archived config-repeat summaries and are intentionally labelled descriptive because participants recur and many settings vary jointly.",
        "",
        "## Missing V2 calculations",
        "",
        "ROC-AUC/ROC curves, PR-AUC, participant-cluster CI and formal participant-exchange P are N/A. The required participant-keyed OOF probability rows were not archived; aggregate confusion matrices are not a valid replacement.",
        "",
        "See `STUDY_SUMMARY.html` for all plots and `tables/report_tables.xlsx` for one worksheet per table.",
        "",
    ]
    (target / "STUDY_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    (target / "STUDY_SUMMARY.html").write_text(
        _html_document(
            title=f"Historical V2-style report: {source.name}",
            subtitle=f"{len(summary)} configurations × five matched repeat seeds; {writing_role}",
            leaderboard=display_rows,
            inventory=inventory_rows,
            figures=figures,
            conclusions=conclusions,
        ),
        encoding="utf-8",
    )
    _write_common_support_files(
        target,
        sources=(source,),
        report_kind="separate_historical_overfitting_search",
        interpretation_lines=conclusions,
        component_lines=components,
    )
    return {
        "report_id": target.name,
        "path": str(target),
        "source_studies": source.name,
        "configuration_count": int(runs["overfit_config_id"].nunique()),
        "run_count": len(runs),
        "leader": leader_id,
        "leader_subject_ba_mean": float(leader["subject_balanced_accuracy_mean"]),
        "scientific_status": "historical_hypothesis_generation_only",
    }


def _write_suite_index(target: Path, reports: Sequence[Mapping[str, Any]]) -> None:
    pandas = _pd()
    write_csv(target / "report_index.csv", reports)
    split_by_report: dict[str, Any] = {}
    contract_rows: list[dict[str, Any]] = []
    for report in reports:
        report_dir = Path(str(report["path"]))
        if report["report_id"] == "01_early_three_model_matched":
            split = pandas.read_csv(
                report_dir / "tables/early_three_model_split_audit.csv"
            )
            parameters = pandas.read_csv(
                report_dir / "tables/early_three_model_parameters.csv"
            )
            repeats = pandas.read_csv(
                report_dir / "tables/early_three_model_repeat_metrics.csv"
            )
            parameter_values = lambda name: sorted(  # noqa: E731
                {
                    str(value)
                    for value in parameters.loc[
                        parameters["parameter"].astype(str).eq(name), "value"
                    ].dropna()
                }
            )
            contract_rows.append(
                {
                    "report_id": report["report_id"],
                    "source_studies": report["source_studies"],
                    "models": ";".join(
                        sorted(repeats["resolved_model"].astype(str).unique())
                    ),
                    "native_fs_hz": ";".join(parameter_values("fs")),
                    "dl_target_fs_hz": ";".join(parameter_values("cnn_target_fs")),
                    "batch_size": ";".join(parameter_values("cnn_batch_size")),
                    "all_config_evaluation_window_seconds": ";".join(
                        sorted(
                            {_display_value(value) for value in repeats["window_sec"]}
                        )
                    ),
                    "all_config_evaluation_hop_seconds": ";".join(
                        sorted({_display_value(value) for value in repeats["hop_sec"]})
                    ),
                    "analysis_grid_evaluation_hop_seconds": ";".join(
                        sorted({_display_value(value) for value in repeats["hop_sec"]})
                    ),
                    "analysis_grid_train_overlap_percent": "N/A_not_declared",
                    "derived_analysis_grid_training_hop_seconds": "N/A_not_declared",
                    "best_epoch_selection": "true_with_patience20_held_out_fold_selection_contamination",
                    "roles": "B/R1-R4 static",
                    "comparison_boundary": "matched_three_model_only",
                }
            )
        else:
            split = pandas.read_csv(report_dir / "tables/split_audit.csv")
            parameters = pandas.read_csv(
                report_dir / "tables/configuration_parameters.csv"
            )

            def unique_field(name: str) -> str:
                if name not in parameters:
                    return "N/A_not_archived"
                return ";".join(
                    sorted(
                        {_display_value(value) for value in parameters[name].dropna()}
                    )
                )

            analysis_parameters = parameters
            if "overfit_stage" in parameters and "generalization" in set(
                parameters["overfit_stage"].dropna().astype(str)
            ):
                analysis_parameters = parameters.loc[
                    parameters["overfit_stage"].astype(str).eq("generalization")
                ]

            def unique_analysis_field(name: str) -> str:
                if name not in analysis_parameters:
                    return "N/A_not_declared"
                return (
                    ";".join(
                        sorted(
                            {
                                _display_value(value)
                                for value in analysis_parameters[name].dropna()
                            }
                        )
                    )
                    or "N/A_not_declared"
                )

            train_overlaps = [
                float(value)
                for value in analysis_parameters.get(
                    "train_overlap_pct", pandas.Series(dtype=float)
                ).dropna()
            ]
            training_hops = sorted(
                {5.0 * (1.0 - overlap / 100.0) for overlap in train_overlaps}
            )

            contract_rows.append(
                {
                    "report_id": report["report_id"],
                    "source_studies": report["source_studies"],
                    "models": unique_field("resolved_model"),
                    "native_fs_hz": unique_field("report_config__fs"),
                    "dl_target_fs_hz": unique_field("report_config__cnn_target_fs"),
                    "batch_size": unique_field("report_config__cnn_batch_size"),
                    "all_config_evaluation_window_seconds": unique_field("window_sec"),
                    "all_config_evaluation_hop_seconds": unique_field("hop_sec"),
                    "analysis_grid_evaluation_hop_seconds": unique_analysis_field(
                        "hop_sec"
                    ),
                    "analysis_grid_train_overlap_percent": unique_analysis_field(
                        "train_overlap_pct"
                    ),
                    "derived_analysis_grid_training_hop_seconds": (
                        ";".join(_display_value(value) for value in training_hops)
                        if training_hops
                        else "N/A_not_declared"
                    ),
                    "best_epoch_selection": unique_field(
                        "report_config__cnn_select_best_epoch"
                    ),
                    "roles": unique_field("train_role_mode"),
                    "comparison_boundary": (
                        "within_report_only; preprocessing/input contracts differ across reports"
                    ),
                }
            )
        split_by_report[str(report["report_id"])] = split

    split_rows: list[dict[str, Any]] = []
    for seed in _SPLIT_SEEDS:
        signatures: dict[str, str] = {}
        for report_id, split in split_by_report.items():
            values = split.loc[
                split["split_seed"].astype(int).eq(seed), "fold_roster_sha256"
            ].drop_duplicates()
            if len(values) != 1:
                raise ValueError(
                    f"suite split audit has ambiguous roster for {report_id}/seed={seed}"
                )
            signatures[report_id] = str(values.iloc[0])
        split_rows.append(
            {
                "split_seed": seed,
                **{
                    f"{name}_fold_roster_sha256": value
                    for name, value in signatures.items()
                },
                "all_report_units_share_exact_participant_fold_roster": (
                    len(set(signatures.values())) == 1
                ),
                "independence_interpretation": (
                    "same_participants_and_splits_enable_pairing_but_are_not_independent_evidence"
                ),
            }
        )
    cross_split = pandas.DataFrame(split_rows)
    if not cross_split["all_report_units_share_exact_participant_fold_roster"].all():
        raise ValueError(
            "historical report units do not share the frozen split registry"
        )
    contracts = pandas.DataFrame(contract_rows)
    write_csv(target / "cross_study_split_audit.csv", _as_records(cross_split))
    write_csv(target / "cross_study_contract_summary.csv", _as_records(contracts))
    write_table_column_definitions(target, csv_directory=target)
    write_excel_workbook_from_csv_directory(target / "report_tables.xlsx", target)

    early, fixed, extension, generalization = reports
    narrative = [
        "# Paper narrative and cross-study interpretation",
        "",
        "## Evidence order",
        "",
        f"1. **Historical architecture candidate generation.** The matched early archive selects `{early['leader']}` descriptively (BA {100*float(early['leader_subject_ba_mean']):.1f}%) and supplies implementation-specific evidence for moving ShapeFormer-PISD to a separate diagnostic route.",
        f"2. **Fixed-epoch/regularization hypothesis generation.** `{fixed['leader']}` is the 20260608 point leader (BA {100*float(fixed['leader_subject_ba_mean']):.1f}%) among {fixed['configuration_count']} searched configurations; use it to motivate later fixed-epoch and regularization tests, not as an unbiased final estimate.",
        f"3. **SQI/loss/feature hypothesis generation.** `{extension['leader']}` is the 20260625 point leader (BA {100*float(extension['leader_subject_ba_mean']):.1f}%) among {extension['configuration_count']} configurations; treat SQI/aggregation/manual-feature/loss compositions as route hypotheses.",
        f"4. **Generalization-grid stress test.** The overall point leader `{generalization['leader']}` (BA {100*float(generalization['leader_subject_ba_mean']):.1f}%) is a fixed historical reference. The new-grid leader is reported separately inside report 04; fixed references are excluded from factor boxplots.",
        "5. **Move confirmation to V2.** Only later matched V2 ablations, frozen hyperparameter studies, representation selection, SQI–motion–denoiser composition, and a final locked 5×5 run should support confirmatory claims.",
        "",
        "## Cross-study boundaries",
        "",
        "- All report units reuse the same 29 participants and the same participant-fold roster for each split seed. This supports matched descriptive contrasts but means the studies are not independent replications.",
        "- The historical preprocessing contract changes: 20260608 uses a 64 Hz DL target, whereas 20260625 and 20260630 use 400 Hz. Baselines and available modules also change. Do not interpret their point leaders as a one-factor cross-study ablation.",
        "- The early archive uses held-out-fold epoch selection and is selection-contaminated. The June fixed-epoch searches remove best-epoch selection but still use the same pooled participant-grouped CV evidence for search and ranking.",
        "- ROC-AUC/ROC curves, PR-AUC, calibration, participant-cluster CI and formal participant-exchange P cannot be recovered because participant-keyed OOF probability rows and model checkpoints were not archived.",
        "- Repeat Student-t CI95 quantifies dispersion across five correlated repeated-CV summaries. It must not be renamed participant-cluster CI or treated as model-selection-adjusted uncertainty.",
        "",
        "## Recommended manuscript tables and plots",
        "",
        "- Main historical model table: report 01 leaderboard plus matched repeat-delta and runtime plots.",
        "- Fixed-epoch evidence: report 02 top configurations and parameter/factor boxplots, explicitly labelled hypothesis-generating.",
        "- SQI/loss/feature evidence: report 03 route-composition plots and per-class tables.",
        "- Generalization evidence: report 04 new-grid-only factor plots, seven-level joint sampling-policy comparison and matched marginal-delta table; fixed references shown only as anchors.",
        "- Use `cross_study_contract_summary.csv` beside these results so sampling-rate and selection-protocol changes remain visible.",
        "",
    ]
    (target / "PAPER_NARRATIVE.md").write_text("\n".join(narrative), encoding="utf-8")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['report_id']))}</td>"
        f"<td>{html.escape(str(row['source_studies']))}</td>"
        f"<td>{row['configuration_count']}</td><td>{row['run_count']}</td>"
        f"<td>{html.escape(str(row['leader']))}</td>"
        f"<td>{100*float(row['leader_subject_ba_mean']):.1f}%</td>"
        f"<td><a href='{html.escape(str(Path(row['path']).name))}/STUDY_SUMMARY.html'>HTML</a> · "
        f"<a href='{html.escape(str(Path(row['path']).name))}/STUDY_SUMMARY.md'>Markdown</a></td>"
        "</tr>"
        for row in reports
    )
    index_header = "".join(
        f"<th>{html.escape(column)}</th>" for column in _SUITE_INDEX_COLUMNS
    )
    index_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Historical V2 report suite</title>
<style>body{{font-family:system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3db;padding:.5rem}}th{{background:#eaf0f7}}</style></head><body>
<h1>Historical V2 report suite</h1><p>One matched early-model report plus three separately rendered historical-search reports. No source archive was modified and no model was retrained.</p>
<table><thead><tr>{index_header}</tr></thead><tbody>{rows}</tbody></table>
<p>All reports are historical hypothesis-generation evidence. Probability-dependent V2 statistics are explicit N/A where source evidence is absent.</p>
<p><a href="PAPER_NARRATIVE.md">Paper narrative and cross-study interpretation</a> · <a href="report_tables.xlsx">Root audit workbook</a> · <a href="cross_study_split_audit.csv">Cross-study split audit</a> · <a href="cross_study_contract_summary.csv">Cross-study contract summary</a></p></body></html>"""
    (target / "STUDY_INDEX.html").write_text(index_html, encoding="utf-8")
    md = [
        "# Historical V2 report suite",
        "",
        "One matched early-model report plus three separately rendered historical-search reports. No source archive was modified and no model was retrained.",
        "",
        "| " + " | ".join(_SUITE_INDEX_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _SUITE_INDEX_COLUMNS) + " |",
    ]
    for row in reports:
        name = Path(str(row["path"])).name
        md.append(
            f"| [{row['report_id']}]({name}/STUDY_SUMMARY.md) | {row['source_studies']} | "
            f"{row['configuration_count']} | {row['run_count']} | {row['leader']} | "
            f"{100*float(row['leader_subject_ba_mean']):.1f}% | "
            f"[HTML]({name}/STUDY_SUMMARY.html) / "
            f"[Markdown]({name}/STUDY_SUMMARY.md) |"
        )
    md.extend(
        (
            "",
            "Open `STUDY_INDEX.html` for the plot-rich index.",
            "",
            "Cross-study writing order and evidential boundaries: [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md).",
            "",
        )
    )
    (target / "STUDY_INDEX.md").write_text("\n".join(md), encoding="utf-8")
    evidence = [
        {
            "path": str(path.relative_to(target)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            (path for path in target.rglob("*") if path.is_file()),
            key=lambda item: str(item.relative_to(target)).encode("utf-8"),
        )
    ]
    (target / "suite_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ppg_frailty.historical_report_suite.v1",
                "status": "complete",
                "report_count": len(reports),
                "reports": list(reports),
                "output_evidence_excluding_manifest": evidence,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_historical_report_suite(
    *,
    early_source: str | Path,
    shapeformer_source: str | Path,
    fixed_epoch_source: str | Path,
    extension_source: str | Path,
    generalization_source: str | Path,
    output_dir: str | Path,
    window_seconds: float = 5.0,
    overlap_percent: float = 50.0,
    patience: int = 20,
    extra_input: str = "0",
) -> Path:
    """Generate four immutable reports and a suite index without retraining."""

    sources = tuple(
        Path(value).resolve()
        for value in (
            early_source,
            shapeformer_source,
            fixed_epoch_source,
            extension_source,
            generalization_source,
        )
    )
    missing = [str(source) for source in sources if not source.is_dir()]
    if missing:
        raise FileNotFoundError(f"historical source directories are missing: {missing}")
    target = _prepare_output_directory(output_dir)
    reports = [
        _early_report(
            target / "01_early_three_model_matched",
            early_source=sources[0],
            shapeformer_source=sources[1],
            window_seconds=window_seconds,
            overlap_percent=overlap_percent,
            patience=patience,
            extra_input=extra_input,
        ),
        _overfit_report(
            target / f"02_{sources[2].name}",
            source=sources[2],
            writing_role="motivate fixed-epoch and regularization hypotheses for later matched V2 tests",
        ),
        _overfit_report(
            target / f"03_{sources[3].name}",
            source=sources[3],
            writing_role="motivate SQI, loss, aggregation and engineered-feature hypotheses",
        ),
        _overfit_report(
            target / f"04_{sources[4].name}",
            source=sources[4],
            writing_role="stress-test generalization/sampling hypotheses and compare frozen historical references",
        ),
    ]
    _write_suite_index(target, reports)
    return target


__all__ = ["run_historical_report_suite"]
