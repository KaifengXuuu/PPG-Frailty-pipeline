"""Normalize real cell, history, OOF, quality, and operational artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


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
    from the fitted case.  Only the small aggregation block is retained in the
    in-memory collection; the full resolved config remains the source of truth
    on disk.
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
        return (
            {
                "case_id": case_id,
                "resolved_config_path": relative.as_posix(),
                "aggregation": dict(aggregation),
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


def _quality_rows(case_id: str, case_directory: Path) -> list[dict[str, Any]]:
    def projected(
        item: Mapping[str, Any],
        target: Path,
        *,
        artifact_field: str,
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
        value[artifact_field] = target.relative_to(
            case_directory
        ).as_posix()
        return value

    def artifact_rows(filename: str, artifact_field: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for target in _shallowest(case_directory.rglob(filename)):
            payload = _read_json(target)
            if not isinstance(payload, Mapping):
                continue
            cells = payload.get("cells")
            if not isinstance(cells, list):
                repeat = payload.get("repeat_index")
                fold = payload.get("fold_index")
                directory_parts = target.parent.name.split("_")
                if (
                    len(directory_parts) == 4
                    and directory_parts[0] == "repeat"
                    and directory_parts[2] == "fold"
                ):
                    repeat = (
                        repeat if repeat is not None else int(directory_parts[1])
                    )
                    fold = fold if fold is not None else int(directory_parts[3])
                cells = (
                    {
                        "repeat_index": repeat,
                        "fold_index": fold,
                        "quality_mode": payload.get("quality_mode"),
                        "rows": payload.get("rows", ()),
                    },
                )
            for cell in cells:
                if not isinstance(cell, Mapping):
                    continue
                for item in cell.get("rows", ()):
                    if isinstance(item, Mapping):
                        rows.append(
                            {
                                "case_id": case_id,
                                "repeat": cell.get("repeat_index"),
                                "fold": cell.get("fold_index"),
                                "quality_mode": cell.get("quality_mode"),
                                **projected(
                                    item,
                                    target,
                                    artifact_field=artifact_field,
                                ),
                            }
                        )
        return rows

    diagnostics = artifact_rows(
        "quality_diagnostics.json",
        "quality_diagnostics_artifact",
    )
    routes = artifact_rows("route_artifacts.json", "route_artifacts_artifact")
    merged = list(diagnostics)
    by_record = {
        (row.get("repeat"), row.get("fold"), row.get("record_id")): index
        for index, row in enumerate(merged)
        if row.get("record_id") is not None
    }
    for route in routes:
        key = (route.get("repeat"), route.get("fold"), route.get("record_id"))
        existing_index = (
            by_record.get(key) if route.get("record_id") is not None else None
        )
        if existing_index is None:
            if route.get("record_id") is not None:
                by_record[key] = len(merged)
            merged.append(route)
            continue
        diagnostic = merged[existing_index]
        combined = {**diagnostic, **route}
        if route.get("quality_mode") is None:
            combined["quality_mode"] = diagnostic.get("quality_mode")
        if "components" in diagnostic and "components" not in route:
            combined["components"] = diagnostic["components"]
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
    return {
        "case_id": case_id,
        **dict(metrics),
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
    )
