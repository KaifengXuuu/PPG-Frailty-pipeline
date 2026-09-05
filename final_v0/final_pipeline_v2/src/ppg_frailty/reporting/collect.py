"""Normalize real cell, history, OOF, quality, and operational artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import yaml

from .cache_audit import collect_preprocessing_cache_rows


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"cannot convert artifact row to mapping: {type(value)!r}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_table(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def _first_shallow(paths: Iterable[Path]) -> Path | None:
    values = _shallowest(paths)
    return values[0] if values else None


def _shallowest(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Select a root aggregate, or every equally shallow per-cell artifact."""

    values = tuple(paths)
    if not values:
        return ()
    minimum_depth = min(len(path.parts) for path in values)
    return tuple(
        sorted(
            (path for path in values if len(path.parts) == minimum_depth),
            key=str,
        )
    )


def _cell_rows(case_id: str, result: Mapping[str, Any], case_directory: Path) -> list[dict[str, Any]]:
    cells = result.get("cell_results")
    if not isinstance(cells, list):
        cells = None
    if cells is None:
        metrics_file = _first_shallow(case_directory.rglob("metrics_per_fold_seed.json"))
        if metrics_file is not None:
            payload = _read_json(metrics_file)
            if isinstance(payload, Mapping) and isinstance(payload.get("cells"), list):
                cells = payload["cells"]
    rows: list[dict[str, Any]] = []
    for raw in cells or ():
        if not isinstance(raw, Mapping):
            continue
        cell = dict(raw)
        metrics = cell.get("metrics") if isinstance(cell.get("metrics"), Mapping) else {}
        row: dict[str, Any] = {
            "case_id": case_id,
            "status": cell.get("status", "unknown"),
            "repeat": cell.get("repeat_index", cell.get("repeat")),
            "fold": cell.get("fold_index", cell.get("fold")),
            "split_seed": cell.get("split_seed"),
            "training_seed": cell.get("training_seed"),
            "model_id": cell.get("model_machine_id", cell.get("model_id")),
            "representation_mode": cell.get("representation_mode"),
            "elapsed_seconds": cell.get("elapsed_seconds"),
            "retained_train_record_count": cell.get("retained_train_record_count"),
            "retained_oof_record_count": cell.get("retained_oof_record_count"),
            "selected_record_count": cell.get("selected_record_count"),
            "oof_window_prediction_count": cell.get("oof_window_prediction_count"),
        }
        for name in (
            "balanced_accuracy",
            "macro_f1",
            "abstention_aware_balanced_accuracy",
            "abstention_aware_macro_precision",
            "abstention_aware_macro_recall",
            "abstention_aware_macro_f1",
            "abstention_count",
            "multiclass_log_loss",
            "multiclass_brier",
            "expected_calibration_error",
            "worst_class_precision",
            "worst_class_recall",
            "worst_class_f1",
            "coverage_rate",
            "n_total",
            "n_retained",
            "n_dropped",
        ):
            row[name] = metrics.get(name)
        row["confusion_matrix"] = metrics.get("confusion_matrix")
        row["class_order"] = metrics.get("class_order", cell.get("class_order"))
        row["per_class"] = metrics.get("per_class")
        row["abstention_counts_by_class"] = metrics.get(
            "abstention_counts_by_class",
            metrics.get("per_class_abstention"),
        )
        row["abstention_aware_per_class"] = metrics.get(
            "abstention_aware_per_class"
        )
        row["abstention_probability_metrics_scope"] = metrics.get(
            "abstention_probability_metrics_scope"
        )
        operational = cell.get("operational_metrics")
        if isinstance(operational, Mapping):
            row["operational_status"] = operational.get("status")
            row["parameter_count"] = operational.get("parameter_count")
            row["model_latency_p50_ms"] = operational.get("model_latency_p50_ms")
            row["model_latency_p95_ms"] = operational.get("model_latency_p95_ms")
        rows.append(row)
    return rows


def _learning_curve_contract_fields(payload: Any) -> dict[str, Any]:
    """Project curve provenance next to every history row.

    Report code must be able to distinguish an inner/train learning metric from
    an outer held-out result without reopening the experiment result.  The
    prefix also prevents contract metadata from being mistaken for a plotted
    metric.
    """

    if not isinstance(payload, Mapping):
        return {}
    return {
        "learning_curve_status": payload.get("status"),
        "learning_curve_training_data_scope": payload.get("training_data_scope"),
        "learning_curve_outer_heldout_used": payload.get(
            "outer_heldout_used_for_epoch_selection_or_curve"
        ),
        "learning_curve_validation_metric": payload.get("validation_metric"),
    }


def _history_rows(
    case_id: str,
    result: Mapping[str, Any],
    case_directory: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = (
        result.get("cell_results", ())
        if isinstance(result.get("cell_results"), list)
        else ()
    )
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        history = cell.get("training_history", cell.get("history", ()))
        if not isinstance(history, list):
            continue
        contract = _learning_curve_contract_fields(
            cell.get("learning_curve_contract")
        )
        for item in history:
            if isinstance(item, Mapping):
                rows.append(
                    {
                        "case_id": case_id,
                        "repeat": cell.get("repeat_index", cell.get("repeat")),
                        "fold": cell.get("fold_index", cell.get("fold")),
                        **dict(item),
                        **contract,
                    }
                )
    for path in sorted(case_directory.rglob("training_history.json")):
        payload = _read_json(path)
        values = (
            payload.get("rows", payload)
            if isinstance(payload, Mapping)
            else payload
        )
        contract = _learning_curve_contract_fields(
            payload.get("learning_curve_contract")
            if isinstance(payload, Mapping)
            else None
        )
        if isinstance(values, list):
            for item in values:
                if isinstance(item, Mapping):
                    rows.append({"case_id": case_id, **dict(item), **contract})
    for path in sorted(case_directory.rglob("training_history.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            for item in csv.DictReader(stream):
                rows.append({"case_id": case_id, **dict(item)})
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=True, default=str)
        deduplicated[key] = row
    return list(deduplicated.values())


def _oof_rows(
    case_id: str,
    artifact_root: Path,
    *,
    filename: str,
) -> tuple[list[dict[str, Any]], str | None]:
    targets = _shallowest(artifact_root.rglob(filename))
    if not targets:
        return [], f"{filename} not found"
    from ppg_frailty.training import read_oof_parquet

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for target in targets:
        try:
            values = read_oof_parquet(target)
            rows.extend(
                {"case_id": case_id, **_mapping(value)}
                for value in values
            )
        except Exception as error:  # noqa: BLE001 - report the exact limitation.
            errors.append(
                f"cannot read {target}: {type(error).__name__}: {error}"
            )
    return rows, "; ".join(errors) if errors else None


def _case_artifact_root(
    case_directory: Path,
    record: Mapping[str, Any],
) -> Path:
    raw = record.get("artifact_root")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("case_result.json lacks an explicit artifact_root")
    target = (case_directory / raw).resolve()
    target.relative_to(case_directory.resolve())
    if not target.is_dir():
        raise FileNotFoundError(f"case artifact_root is not a directory: {target}")
    return target


def _manifest_case_directory(
    study_root: Path,
    case: Mapping[str, Any],
) -> Path:
    raw = case.get("case_directory")
    if isinstance(raw, str) and raw.strip():
        relative = Path(raw)
        if relative.is_absolute():
            raise ValueError("manifest case_directory must be relative")
        target = (study_root / relative).resolve()
        target.relative_to(study_root.resolve())
        return target
    case_id = str(case["case_id"])
    return study_root / "cases" / case_id


def _resolved_aggregation_config(
    study_root: Path,
    *,
    case_id: str,
    case: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read the persisted per-case aggregation controls for report replay.

    Report-time Line A/Line B reaggregation must retain scientific modifiers
    from the fitted case.  The aggregation block and evaluation-statistics
    policy are retained in memory; the full resolved config remains the source
    of truth on disk.
    """

    raw = case.get("resolved_config_path")
    try:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("manifest case lacks resolved_config_path")
        relative = Path(raw)
        if relative.is_absolute():
            raise ValueError("resolved_config_path must be relative")
        target = (study_root / relative).resolve()
        target.relative_to(study_root.resolve())
        if not target.is_file():
            raise FileNotFoundError(target)
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("resolved config root must be a mapping")
        aggregation = payload.get("aggregation")
        if not isinstance(aggregation, Mapping):
            raise TypeError("resolved config aggregation must be a mapping")
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping) or not isinstance(
            evaluation.get("statistics"), Mapping
        ):
            raise TypeError(
                "resolved config evaluation.statistics must be a mapping"
            )
        return (
            {
                "case_id": case_id,
                "resolved_config_path": relative.as_posix(),
                "aggregation": dict(aggregation),
                "evaluation_statistics": dict(evaluation["statistics"]),
            },
            None,
        )
    except Exception as error:  # noqa: BLE001 - preserve report limitation.
        return (
            None,
            {
                "case_id": case_id,
                "resolved_config_path": raw,
                "error": f"{type(error).__name__}: {error}",
            },
        )


_ROUTE_REPORT_SCALAR_FIELDS = (
    "schema_version",
    "state",
    "quality_mode",
    "source_signal",
    "quality_tier",
    "motion_state",
    "canonical_hybrid_waveform_created",
    "denoiser_invocation_count",
    "direct_hr_bpm",
    "direct_median_valid_ppi_s",
    "direct_peak_count",
    "direct_valid_ppi_count",
    "heart_rate_estimator",
    "motion_file_median_probability_diagnostic_only",
    "motion_record_probability",
    "motion_threshold",
    "motion_window_count",
    "native_routing_window_count",
    "post_denoise_hr_bpm",
    "post_denoise_median_valid_ppi_s",
    "post_denoise_peak_count",
    "post_denoise_valid_ppi_count",
    "post_minus_direct_hr_bpm",
    "short_record_action",
    "abstained",
    "abstention_reason",
    "denoiser_attempted",
    "denoiser_id",
    "denoiser_status",
    "reducer_status",
    "artifact_reducer_status",
    "direct_q_rate_score",
    "direct_q_rate_coverage",
    "direct_q_rate_state",
    "direct_q_morph_score",
    "direct_q_morph_coverage",
    "direct_q_morph_state",
    "post_q_rate_score",
    "post_q_rate_coverage",
    "post_q_rate_state",
    # Compact pre-timeline route artifacts use these fields.
    "affects_aggregation",
    "affects_prediction",
    "affects_retention",
    "classification_action",
    "end_sample",
    "segment_id",
    "start_sample",
)


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_mapping_field(rows: Iterable[Mapping[str, Any]], field: str) -> float | None:
    values = [
        number
        for row in rows
        if (number := _finite_number(row.get(field))) is not None
    ]
    return sum(values) / len(values) if values else None


def _collapse_mapping_field(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> str | None:
    values = sorted(
        {
            str(value)
            for row in rows
            if (value := row.get(field)) is not None and str(value).strip()
        }
    )
    if not values:
        return None
    return values[0] if len(values) == 1 else "mixed:" + "|".join(values)


def _numeric_field_summary(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [
        number
        for row in rows
        if (number := _finite_number(row.get(field))) is not None
    ]
    unique = sorted(set(values))
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "unique_count": len(unique),
        "single_value": unique[0] if len(unique) == 1 else None,
    }


def _state_counts(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    states: Iterable[str],
) -> dict[str, int]:
    normalized = [
        (
            "unavailable"
            if row.get(field) is None or not str(row.get(field)).strip()
            else str(row.get(field)).strip().lower()
        )
        for row in rows
    ]
    return {
        state: sum(value == state for value in normalized)
        for state in states
    }


def _nested_sqi_mean(
    evidence: Mapping[str, Any],
    *,
    stage: str,
    component: str,
    field: str,
) -> float | None:
    rows: list[Mapping[str, Any]] = []
    for raw in _native_sqi_window_rows(evidence):
        stage_payload = raw.get(stage)
        if not isinstance(stage_payload, Mapping):
            continue
        component_payload = stage_payload.get(component)
        if isinstance(component_payload, Mapping):
            rows.append(component_payload)
    return _mean_mapping_field(rows, field)


def _nested_sqi_state(
    evidence: Mapping[str, Any],
    *,
    stage: str,
    component: str,
) -> str | None:
    rows: list[Mapping[str, Any]] = []
    for raw in _native_sqi_window_rows(evidence):
        stage_payload = raw.get(stage)
        if not isinstance(stage_payload, Mapping):
            continue
        component_payload = stage_payload.get(component)
        if isinstance(component_payload, Mapping):
            rows.append(component_payload)
    return _collapse_mapping_field(rows, "state")


def _native_sqi_window_rows(
    evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return keyed-window compact SQI rows without treating metadata as windows."""

    return tuple(
        raw
        for raw in evidence.values()
        if isinstance(raw, Mapping)
        and ("direct" in raw or "post_reduction" in raw)
    )


def _project_route_artifact(
    raw: Any,
    *,
    diagnostic_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the bounded record-level route view consumed by reports.

    The authoritative per-window routing cells and full SQI component evidence
    remain in the cell artifacts named by the surrounding diagnostic row.  A
    root report must not duplicate those high-cardinality objects into CSV,
    JSON, and XLSX cells.
    """

    if not isinstance(raw, Mapping):
        return {}
    scalar_types = (str, int, float, bool, type(None))
    projected = {
        field: raw[field]
        for field in _ROUTE_REPORT_SCALAR_FIELDS
        if field in raw and isinstance(raw[field], scalar_types)
    }
    for field in ("motion_provenance", "routing_grid"):
        if isinstance(raw.get(field), Mapping):
            projected[field] = dict(raw[field])
    if isinstance(raw.get("reasons"), (list, tuple)):
        projected["reasons"] = [str(value) for value in raw["reasons"]]

    raw_cells = raw.get("cells")
    cells = [
        dict(value)
        for value in (raw_cells if isinstance(raw_cells, (list, tuple)) else ())
        if isinstance(value, Mapping)
    ]
    native_sqi = (
        dict(raw["native_window_sqi_evidence"])
        if isinstance(raw.get("native_window_sqi_evidence"), Mapping)
        else {}
    )
    if cells:
        threshold_summary = _numeric_field_summary(cells, "motion_threshold")
        aliases = {
            "motion_threshold": threshold_summary["single_value"],
            "direct_q_rate_score": _mean_mapping_field(
                cells, "direct_q_rate_score"
            ),
            "direct_q_morph_score": _mean_mapping_field(
                cells, "direct_q_morph_score"
            ),
            "post_q_rate_score": _mean_mapping_field(cells, "post_q_rate_score"),
            "direct_q_rate_state": _collapse_mapping_field(
                cells, "direct_q_rate_state"
            ),
            "direct_q_morph_state": _collapse_mapping_field(
                cells, "direct_q_morph_state"
            ),
            "post_q_rate_state": _collapse_mapping_field(
                cells, "post_q_rate_state"
            ),
            "quality_tier": _collapse_mapping_field(cells, "final_tier"),
            "source_signal": _collapse_mapping_field(cells, "source_route"),
        }
        for field, value in aliases.items():
            if projected.get(field) is None and value is not None:
                projected[field] = value
        if threshold_summary["unique_count"] == 1:
            projected["motion_threshold"] = threshold_summary["single_value"]
        elif threshold_summary["unique_count"] > 1:
            projected.pop("motion_threshold", None)
        motion_states = {
            str(cell.get("motion_state", "unavailable")).strip().lower()
            for cell in cells
        }
        projected["motion_state"] = (
            "low_only"
            if motion_states == {"low"}
            else "high_only"
            if motion_states == {"high"}
            else "off"
            if motion_states == {"off"}
            else "unavailable"
            if motion_states <= {"", "none", "unavailable"}
            else "mixed:" + "|".join(sorted(motion_states))
        )
        projected.update(
            {
                "motion_threshold_unique_count": threshold_summary["unique_count"],
                "motion_threshold_minimum": threshold_summary["minimum"],
                "motion_threshold_maximum": threshold_summary["maximum"],
                "motion_threshold_consistency_status": (
                    "single_frozen_value"
                    if threshold_summary["unique_count"] == 1
                    else "not_evaluated"
                    if threshold_summary["unique_count"] == 0
                    else "inconsistent_multiple_values"
                ),
            }
        )

    native_sqi_rows = _native_sqi_window_rows(native_sqi)
    if native_sqi_rows:
        native_sqi_aliases = {
            "direct_q_rate_score": _nested_sqi_mean(
                native_sqi,
                stage="direct",
                component="q_rate",
                field="score",
            ),
            "direct_q_rate_coverage": _nested_sqi_mean(
                native_sqi,
                stage="direct",
                component="q_rate",
                field="coverage",
            ),
            "direct_q_rate_state": _nested_sqi_state(
                native_sqi,
                stage="direct",
                component="q_rate",
            ),
            "direct_q_morph_score": _nested_sqi_mean(
                native_sqi,
                stage="direct",
                component="q_morph",
                field="score",
            ),
            "direct_q_morph_coverage": _nested_sqi_mean(
                native_sqi,
                stage="direct",
                component="q_morph",
                field="coverage",
            ),
            "direct_q_morph_state": _nested_sqi_state(
                native_sqi,
                stage="direct",
                component="q_morph",
            ),
            "post_q_rate_score": _nested_sqi_mean(
                native_sqi,
                stage="post_reduction",
                component="q_rate",
                field="score",
            ),
            "post_q_rate_coverage": _nested_sqi_mean(
                native_sqi,
                stage="post_reduction",
                component="q_rate",
                field="coverage",
            ),
            "post_q_rate_state": _nested_sqi_state(
                native_sqi,
                stage="post_reduction",
                component="q_rate",
            ),
        }
        for field, value in native_sqi_aliases.items():
            if value is not None:
                projected[field] = value

    motion_probability = _finite_number(
        raw.get(
            "motion_record_probability",
            raw.get("motion_file_median_probability_diagnostic_only"),
        )
    )
    if motion_probability is not None:
        projected["motion_record_probability"] = motion_probability
    motion_window_count = raw.get(
        "motion_window_count", raw.get("native_routing_window_count")
    )
    if motion_window_count is not None:
        projected["motion_window_count"] = motion_window_count

    requested_cells = [
        cell for cell in cells if bool(cell.get("denoiser_requested", False))
    ]
    invocation_count = int(raw.get("denoiser_invocation_count") or 0)
    denoiser_attempted = bool(invocation_count or requested_cells)
    projected["denoiser_attempted"] = denoiser_attempted
    if denoiser_attempted:
        reducer_id = diagnostic_row.get("artifact_reducer_name")
        if reducer_id is not None:
            projected["denoiser_id"] = str(reducer_id)
        successful = sum(
            str(cell.get("denoiser_status", "")).lower() == "success"
            for cell in requested_cells
        )
        status = (
            "success"
            if requested_cells and successful == len(requested_cells)
            else "failed"
            if requested_cells and successful == 0
            else "partial_failure"
            if requested_cells
            else "not_reported"
        )
        projected["denoiser_status"] = status
        projected["reducer_status"] = status

    motion_probability_summary = _numeric_field_summary(
        cells, "motion_probability"
    )
    direct_q_rate_summary = _numeric_field_summary(cells, "direct_q_rate_score")
    direct_q_morph_summary = _numeric_field_summary(cells, "direct_q_morph_score")
    post_q_rate_summary = _numeric_field_summary(cells, "post_q_rate_score")
    motion_state_counts = _state_counts(
        cells, "motion_state", ("high", "low", "off", "unavailable")
    )
    tier_counts = _state_counts(
        cells, "final_tier", ("excellent", "acceptable", "unfit", "unavailable")
    )
    q_rate_counts = _state_counts(
        cells, "direct_q_rate_state", ("pass", "fail", "unavailable")
    )
    q_morph_counts = _state_counts(
        cells, "direct_q_morph_state", ("pass", "fail", "unavailable")
    )
    post_q_rate_counts = _state_counts(
        requested_cells,
        "post_q_rate_state",
        ("pass", "fail", "not_applicable", "unavailable"),
    )
    recovery_eligible_cells = [
        cell
        for cell in requested_cells
        if str(cell.get("direct_q_rate_state", "")).lower()
        not in {"", "pass", "none", "not_applicable", "unavailable"}
    ]
    recovery_count = sum(
        str(cell.get("post_q_rate_state", "")).lower() == "pass"
        for cell in recovery_eligible_cells
    )

    projected.update(
        {
            "routing_cell_count": len(cells),
            "native_sqi_window_count": len(native_sqi_rows),
            "denoiser_requested_cell_count": len(requested_cells),
            "denoiser_success_cell_count": sum(
                str(cell.get("denoiser_status", "")).lower() == "success"
                for cell in requested_cells
            ),
            "motion_high_cell_count": motion_state_counts["high"],
            "motion_low_cell_count": motion_state_counts["low"],
            "motion_off_cell_count": motion_state_counts["off"],
            "motion_unavailable_cell_count": motion_state_counts["unavailable"],
            "motion_probability_cell_count": motion_probability_summary["count"],
            "motion_probability_cell_mean": motion_probability_summary["mean"],
            "motion_probability_cell_median": motion_probability_summary["median"],
            "motion_probability_cell_minimum": motion_probability_summary["minimum"],
            "motion_probability_cell_maximum": motion_probability_summary["maximum"],
            "excellent_cell_count": tier_counts["excellent"],
            "acceptable_cell_count": tier_counts["acceptable"],
            "unfit_cell_count": tier_counts["unfit"],
            "direct_q_rate_pass_cell_count": q_rate_counts["pass"],
            "direct_q_rate_fail_cell_count": q_rate_counts["fail"],
            "direct_q_rate_score_cell_mean": direct_q_rate_summary["mean"],
            "direct_q_rate_score_cell_median": direct_q_rate_summary["median"],
            "direct_q_morph_pass_cell_count": q_morph_counts["pass"],
            "direct_q_morph_fail_cell_count": q_morph_counts["fail"],
            "direct_q_morph_score_cell_mean": direct_q_morph_summary["mean"],
            "direct_q_morph_score_cell_median": direct_q_morph_summary["median"],
            "post_q_rate_pass_cell_count": post_q_rate_counts["pass"],
            "post_q_rate_fail_cell_count": post_q_rate_counts["fail"],
            "post_q_rate_score_cell_mean": post_q_rate_summary["mean"],
            "post_q_rate_score_cell_median": post_q_rate_summary["median"],
            "post_q_rate_recovery_eligible_cell_count": len(
                recovery_eligible_cells
            ),
            "post_q_rate_recovered_cell_count": recovery_count,
            "post_q_rate_recovery_cell_rate": (
                recovery_count / len(recovery_eligible_cells)
                if recovery_eligible_cells
                else None
            ),
            "report_projection": {
                "schema_version": "ppg_frailty.route_report_projection.v1",
                "detail_fields_omitted": [
                    field for field, value in raw.items()
                    if (
                        field in {"cells", "native_window_sqi_evidence"}
                        or (
                            isinstance(value, (Mapping, list, tuple))
                            and field not in {"motion_provenance", "routing_grid", "reasons"}
                        )
                    )
                ],
                "full_detail_retained_in_source_artifact": True,
            },
        }
    )
    return projected


def _quality_rows(case_id: str, case_directory: Path) -> list[dict[str, Any]]:
    def projected(
        item: Mapping[str, Any],
        target: Path,
        *,
        artifact_field: str,
        artifact_sha256: str,
    ) -> dict[str, Any]:
        value = dict(item)
        components = (
            value.get("components")
            if isinstance(value.get("components"), Mapping)
            else None
        )
        if components is not None:
            value["components"] = {
                name: dict(components[name])
                for name in ("predictor_availability", "non_predictor_features")
                if isinstance(components.get(name), Mapping)
            }
        if "route_artifact" in value:
            value["route_artifact"] = _project_route_artifact(
                value.get("route_artifact"),
                diagnostic_row=value,
            )
        value[artifact_field] = target.relative_to(
            case_directory
        ).as_posix()
        value[f"{artifact_field}_sha256"] = artifact_sha256
        return value

    def artifact_rows(filename: str, artifact_field: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        projected_payload_cache: dict[
            str,
            tuple[tuple[Any, Any, Any, dict[str, Any]], ...],
        ] = {}
        for target in _shallowest(case_directory.rglob(filename)):
            digest = _artifact_sha256(target)
            cached = projected_payload_cache.get(digest)
            if cached is None:
                payload = _read_json(target)
                if not isinstance(payload, Mapping):
                    continue
                cells = payload.get("cells")
                if not isinstance(cells, list):
                    cells = (
                        {
                            "repeat_index": payload.get("repeat_index"),
                            "fold_index": payload.get("fold_index"),
                            "quality_mode": payload.get("quality_mode"),
                            "rows": payload.get("rows", ()),
                        },
                    )
                built: list[tuple[Any, Any, Any, dict[str, Any]]] = []
                for cell in cells:
                    if not isinstance(cell, Mapping):
                        continue
                    for item in cell.get("rows", ()):
                        if isinstance(item, Mapping):
                            built.append(
                                (
                                    cell.get("repeat_index"),
                                    cell.get("fold_index"),
                                    cell.get("quality_mode"),
                                    projected(
                                        item,
                                        target,
                                        artifact_field=artifact_field,
                                        artifact_sha256=digest,
                                    ),
                                )
                            )
                cached = tuple(built)
                projected_payload_cache[digest] = cached
            directory_parts = target.parent.name.split("_")
            directory_repeat = None
            directory_fold = None
            if (
                len(directory_parts) == 4
                and directory_parts[0] == "repeat"
                and directory_parts[2] == "fold"
            ):
                directory_repeat = int(directory_parts[1])
                directory_fold = int(directory_parts[3])
            relative_artifact = target.relative_to(case_directory).as_posix()
            for repeat, fold, quality_mode, raw_row in cached:
                row = dict(raw_row)
                row[artifact_field] = relative_artifact
                rows.append(
                    {
                        "case_id": case_id,
                        "repeat": repeat if repeat is not None else directory_repeat,
                        "fold": fold if fold is not None else directory_fold,
                        "quality_mode": quality_mode,
                        **row,
                    }
                )
        return rows

    diagnostics = artifact_rows(
        "quality_diagnostics.json",
        "quality_diagnostics_artifact",
    )
    routes = artifact_rows("route_artifacts.json", "route_artifacts_artifact")

    def annotate_outer_partition(
        rows: list[dict[str, Any]],
        *,
        artifact_field: str,
        manifest_count_field: str,
    ) -> None:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(artifact_field, "")), []).append(row)
        for relative, artifact_rows_for_cell in groups.items():
            artifact_path = case_directory / relative
            manifest_path = artifact_path.parent / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = _read_json(manifest_path)
            cell = manifest.get("cell") if isinstance(manifest, Mapping) else None
            if not isinstance(cell, Mapping):
                continue
            expected = cell.get(manifest_count_field)
            record_ids = [row.get("record_id") for row in artifact_rows_for_cell]
            if (
                isinstance(expected, bool)
                or not isinstance(expected, int)
                or expected != len(artifact_rows_for_cell)
                or any(record_id is None for record_id in record_ids)
                or len(set(record_ids)) != len(record_ids)
            ):
                continue
            provenance = cell.get("fitted_provenance")
            fitted = (
                provenance.get("fitted_participant_ids", ())
                if isinstance(provenance, Mapping)
                else ()
            )
            if (
                not isinstance(fitted, (list, tuple))
                or not fitted
                or any(not str(value).strip() for value in fitted)
            ):
                continue
            fitted_participants = set(map(str, fitted))
            for row in artifact_rows_for_cell:
                row["outer_partition"] = (
                    "outer_train"
                    if str(row.get("participant_id", "")) in fitted_participants
                    else "outer_oof"
                )

    annotate_outer_partition(
        diagnostics,
        artifact_field="quality_diagnostics_artifact",
        manifest_count_field="quality_diagnostic_row_count",
    )
    annotate_outer_partition(
        routes,
        artifact_field="route_artifacts_artifact",
        manifest_count_field="route_artifacts_row_count",
    )
    merged = list(diagnostics)

    def merge_key(row: Mapping[str, Any]) -> tuple[int, int, str] | None:
        record_id = row.get("record_id")
        if record_id is None or not str(record_id).strip():
            return None
        try:
            repeat = int(row.get("repeat"))
            fold = int(row.get("fold"))
        except (TypeError, ValueError):
            # A record id alone is not a safe key in a multi-cell study.
            return None
        return repeat, fold, str(record_id)

    by_record: dict[tuple[int, int, str], int] = {}
    for index, row in enumerate(merged):
        key = merge_key(row)
        if key is None:
            continue
        if key in by_record:
            raise ValueError(
                "duplicate quality diagnostic key while joining route artifact: "
                f"repeat={key[0]}, fold={key[1]}, record_id={key[2]}"
            )
        by_record[key] = index

    seen_routes: set[tuple[int, int, str]] = set()
    for route in routes:
        key = merge_key(route)
        if key is not None and key in seen_routes:
            raise ValueError(
                "duplicate route artifact key while joining quality diagnostic: "
                f"repeat={key[0]}, fold={key[1]}, record_id={key[2]}"
            )
        if key is not None:
            seen_routes.add(key)
        existing_index = by_record.get(key) if key is not None else None
        if existing_index is None:
            if key is not None:
                by_record[key] = len(merged)
            merged.append(route)
            continue
        diagnostic = merged[existing_index]
        for identity_field in ("participant_id", "role"):
            diagnostic_value = diagnostic.get(identity_field)
            route_value = route.get(identity_field)
            if (
                diagnostic_value is not None
                and route_value is not None
                and str(diagnostic_value) != str(route_value)
            ):
                raise ValueError(
                    "quality/route identity mismatch while joining route artifact: "
                    f"repeat={key[0]}, fold={key[1]}, record_id={key[2]}, "
                    f"field={identity_field}"
                )
        # The diagnostic row remains authoritative for record identity and SQI
        # components.  The dedicated route artifact is authoritative only for
        # routing evidence and its source pointer.
        combined = {**route, **diagnostic}
        if "route_artifact" in route:
            combined["route_artifact"] = route["route_artifact"]
        for pointer in (
            "route_artifacts_artifact",
            "route_artifacts_artifact_sha256",
            "route_artifacts_row_count",
        ):
            if pointer in route:
                combined[pointer] = route[pointer]
        if diagnostic.get("quality_mode") is not None:
            combined["quality_mode"] = diagnostic["quality_mode"]
        merged[existing_index] = combined
    return merged


def _config_metrics(case_id: str, case_directory: Path) -> dict[str, Any] | None:
    target = _first_shallow(case_directory.rglob("config_metrics_v2.json"))
    if target is None:
        return None
    payload = _read_json(target)
    if not isinstance(payload, Mapping):
        return None
    metrics = payload.get("config_metrics")
    if not isinstance(metrics, Mapping):
        return None
    bootstrap_fields: dict[str, Any] = {}
    bootstrap_results = payload.get("bootstrap_results", ())
    if isinstance(bootstrap_results, list):
        for raw in bootstrap_results:
            if not isinstance(raw, Mapping):
                continue
            metric = str(raw.get("metric", "")).strip()
            if metric not in {
                "balanced_accuracy",
                "macro_f1",
                "macro_roc_auc_ovr",
            }:
                continue
            prefix = f"participant_cluster_{metric}"
            bootstrap_fields.update(
                {
                    f"{prefix}_estimate": raw.get("estimate"),
                    f"{prefix}_ci95_low": raw.get("ci95_lower"),
                    f"{prefix}_ci95_high": raw.get("ci95_upper"),
                    f"{prefix}_n_resamples": raw.get("n_resamples"),
                    f"{prefix}_seed": raw.get("seed"),
                    f"{prefix}_n_participants": raw.get("n_participants"),
                    f"{prefix}_n_repeats": raw.get("n_repeats"),
                    f"{prefix}_interval_method": raw.get("interval_method"),
                    f"{prefix}_cluster_unit": raw.get("cluster_unit"),
                }
            )
    return {
        "case_id": case_id,
        **dict(metrics),
        **bootstrap_fields,
        "metrics_source": "config_metrics_v2",
        "metrics_status": payload.get("status"),
    }


@dataclass(frozen=True)
class CollectedStudy:
    root: Path
    plan: Mapping[str, Any]
    manifest: Mapping[str, Any]
    case_records: tuple[Mapping[str, Any], ...]
    varied_parameters: tuple[Mapping[str, Any], ...]
    controlled_parameters: tuple[Mapping[str, Any], ...]
    cell_rows: tuple[Mapping[str, Any], ...]
    history_rows: tuple[Mapping[str, Any], ...]
    file_oof_rows: tuple[Mapping[str, Any], ...]
    subject_oof_rows: tuple[Mapping[str, Any], ...]
    role_oof_rows: tuple[Mapping[str, Any], ...]
    quality_rows: tuple[Mapping[str, Any], ...]
    trusted_config_metrics: tuple[Mapping[str, Any], ...]
    limitations: tuple[str, ...]
    oof_read_failures: tuple[Mapping[str, Any], ...] = ()
    window_oof_rows: tuple[Mapping[str, Any], ...] = ()
    resolved_aggregation_configs: tuple[Mapping[str, Any], ...] = ()
    resolved_config_failures: tuple[Mapping[str, Any], ...] = ()
    preprocessing_cache_rows: tuple[Mapping[str, Any], ...] = ()


def collect_study(root: str | Path) -> CollectedStudy:
    study_root = Path(root).resolve()
    if not study_root.is_dir():
        raise FileNotFoundError(study_root)
    plan_path = study_root / "study_plan.yaml"
    manifest_path = study_root / "study_manifest.json"
    if not plan_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("study_plan.yaml and study_manifest.json are required")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    manifest = _read_json(manifest_path)
    if not isinstance(plan, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("study plan and manifest roots must be mappings")
    run_records: dict[str, Mapping[str, Any]] = {}
    run_result_path = study_root / "study_run_result.json"
    if run_result_path.is_file():
        run_result = _read_json(run_result_path)
        if isinstance(run_result, Mapping):
            run_records = {
                str(row["case_id"]): row
                for row in run_result.get("case_records", ())
                if isinstance(row, Mapping) and row.get("case_id") is not None
            }
    case_records: list[Mapping[str, Any]] = []
    cell_rows: list[Mapping[str, Any]] = []
    history_rows: list[Mapping[str, Any]] = []
    window_oof_rows: list[Mapping[str, Any]] = []
    file_oof_rows: list[Mapping[str, Any]] = []
    oof_rows: list[Mapping[str, Any]] = []
    role_oof_rows: list[Mapping[str, Any]] = []
    quality_rows: list[Mapping[str, Any]] = []
    config_metrics: list[Mapping[str, Any]] = []
    limitations: list[str] = []
    oof_read_failures: list[Mapping[str, Any]] = []
    resolved_aggregation_configs: list[Mapping[str, Any]] = []
    resolved_config_failures: list[Mapping[str, Any]] = []
    preprocessing_cache_rows: list[Mapping[str, Any]] = []
    for case in manifest.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case["case_id"])
        resolved_aggregation, config_failure = _resolved_aggregation_config(
            study_root,
            case_id=case_id,
            case=case,
        )
        if resolved_aggregation is not None:
            resolved_aggregation_configs.append(resolved_aggregation)
        if config_failure is not None:
            resolved_config_failures.append(config_failure)
            limitations.append(
                f"{case_id}: resolved config unavailable for aggregation replay: "
                f"{config_failure['error']}"
            )
        case_directory = _manifest_case_directory(study_root, case)
        result_path = case_directory / "case_result.json"
        if not result_path.is_file():
            case_records.append(
                {
                    "case_id": case_id,
                    "status": "not_run",
                    "output_group": case.get("output_group"),
                    "case_directory": case.get(
                        "case_directory",
                        (Path("cases") / case_id).as_posix(),
                    ),
                    "resolved_config_path": case.get("resolved_config_path"),
                }
            )
            limitations.append(f"{case_id}: case_result.json not found")
            continue
        record = run_records.get(case_id)
        if record is None:
            record = _read_json(result_path)
        if not isinstance(record, Mapping):
            raise TypeError(f"case result root must be a mapping: {result_path}")
        result = (
            record.get("result")
            if isinstance(record.get("result"), Mapping)
            else {}
        )
        case_records.append(
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "result"
                },
                "output_group": case.get("output_group"),
                "case_directory": case.get(
                    "case_directory",
                    (Path("cases") / case_id).as_posix(),
                ),
                "resolved_config_path": case.get("resolved_config_path"),
            }
        )
        artifact_root = _case_artifact_root(case_directory, record)
        current_cache_rows, cache_limitations = collect_preprocessing_cache_rows(
            case_id,
            artifact_root,
        )
        preprocessing_cache_rows.extend(current_cache_rows)
        limitations.extend(cache_limitations)
        cell_rows.extend(_cell_rows(case_id, result, artifact_root))
        history_rows.extend(_history_rows(case_id, result, artifact_root))
        current_window_oof, window_limitation = _oof_rows(
            case_id,
            artifact_root,
            filename="oof_window_predictions.parquet",
        )
        window_oof_rows.extend(current_window_oof)
        if window_limitation is not None:
            limitations.append(f"{case_id}: {window_limitation}")
            oof_read_failures.append(
                {
                    "case_id": case_id,
                    "oof_level": "window",
                    "error": window_limitation,
                }
            )
        current_file_oof, file_limitation = _oof_rows(
            case_id,
            artifact_root,
            filename="oof_file_predictions.parquet",
        )
        file_oof_rows.extend(current_file_oof)
        if file_limitation is not None:
            limitations.append(f"{case_id}: {file_limitation}")
            oof_read_failures.append(
                {
                    "case_id": case_id,
                    "oof_level": "file",
                    "error": file_limitation,
                }
            )
        current_oof, limitation = _oof_rows(
            case_id,
            artifact_root,
            filename="oof_subject_predictions.parquet",
        )
        oof_rows.extend(current_oof)
        if limitation is not None:
            limitations.append(f"{case_id}: {limitation}")
            oof_read_failures.append(
                {
                    "case_id": case_id,
                    "oof_level": "participant",
                    "error": limitation,
                }
            )
        current_role_oof, role_limitation = _oof_rows(
            case_id,
            artifact_root,
            filename="oof_role_predictions.parquet",
        )
        role_oof_rows.extend(current_role_oof)
        if role_limitation is not None:
            limitations.append(f"{case_id}: {role_limitation}")
        quality_rows.extend(_quality_rows(case_id, artifact_root))
        trusted = _config_metrics(case_id, artifact_root)
        if trusted is not None:
            config_metrics.append(trusted)
    if not history_rows:
        limitations.append(
            "training history unavailable; learning curves are N/A (classical models "
            "may be legitimately not applicable)"
        )
    return CollectedStudy(
        root=study_root,
        plan=dict(plan),
        manifest=dict(manifest),
        case_records=tuple(case_records),
        varied_parameters=_read_csv_table(
            study_root / "tables" / "varied_parameters.csv"
        ),
        controlled_parameters=_read_csv_table(
            study_root / "tables" / "controlled_parameters.csv"
        ),
        cell_rows=tuple(cell_rows),
        history_rows=tuple(history_rows),
        file_oof_rows=tuple(file_oof_rows),
        subject_oof_rows=tuple(oof_rows),
        role_oof_rows=tuple(role_oof_rows),
        quality_rows=tuple(quality_rows),
        trusted_config_metrics=tuple(config_metrics),
        limitations=tuple(dict.fromkeys(limitations)),
        oof_read_failures=tuple(oof_read_failures),
        window_oof_rows=tuple(window_oof_rows),
        resolved_aggregation_configs=tuple(resolved_aggregation_configs),
        resolved_config_failures=tuple(resolved_config_failures),
        preprocessing_cache_rows=tuple(preprocessing_cache_rows),
    )
