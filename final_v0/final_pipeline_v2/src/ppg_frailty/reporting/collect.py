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
    values = tuple(paths)
    return min(values, key=lambda path: (len(path.parts), str(path))) if values else None


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
        operational = cell.get("operational_metrics")
        if isinstance(operational, Mapping):
            row["operational_status"] = operational.get("status")
            row["parameter_count"] = operational.get("parameter_count")
            row["model_latency_p50_ms"] = operational.get("model_latency_p50_ms")
            row["model_latency_p95_ms"] = operational.get("model_latency_p95_ms")
        rows.append(row)
    return rows


def _history_rows(case_id: str, result: Mapping[str, Any], case_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in result.get("cell_results", ()) if isinstance(result.get("cell_results"), list) else ():
        if not isinstance(cell, Mapping):
            continue
        history = cell.get("training_history", cell.get("history", ()))
        if not isinstance(history, list):
            continue
        for item in history:
            if isinstance(item, Mapping):
                rows.append(
                    {
                        "case_id": case_id,
                        "repeat": cell.get("repeat_index", cell.get("repeat")),
                        "fold": cell.get("fold_index", cell.get("fold")),
                        **dict(item),
                    }
                )
    for path in sorted(case_directory.rglob("training_history.json")):
        payload = _read_json(path)
        values = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
        if isinstance(values, list):
            for item in values:
                if isinstance(item, Mapping):
                    rows.append({"case_id": case_id, **dict(item)})
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
    candidates = tuple(artifact_root.rglob(filename))
    target = _first_shallow(candidates)
    if target is None:
        return [], f"{filename} not found"
    try:
        from ppg_frailty.training import read_oof_parquet

        values = read_oof_parquet(target)
        rows = [{"case_id": case_id, **_mapping(value)} for value in values]
        return rows, None
    except Exception as error:  # noqa: BLE001 - reporting records the limitation.
        return [], f"cannot read {target.name}: {type(error).__name__}: {error}"


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


def _quality_rows(case_id: str, case_directory: Path) -> list[dict[str, Any]]:
    target = _first_shallow(case_directory.rglob("quality_diagnostics.json"))
    if target is None:
        return []
    payload = _read_json(target)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("cells"), list):
        for cell in payload["cells"]:
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
                            **dict(item),
                        }
                    )
    elif isinstance(payload, Mapping):
        for item in payload.get("rows", ()):
            if isinstance(item, Mapping):
                rows.append({"case_id": case_id, **dict(item)})
    return rows


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
    subject_oof_rows: tuple[Mapping[str, Any], ...]
    role_oof_rows: tuple[Mapping[str, Any], ...]
    quality_rows: tuple[Mapping[str, Any], ...]
    trusted_config_metrics: tuple[Mapping[str, Any], ...]
    limitations: tuple[str, ...]


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
    case_records: list[Mapping[str, Any]] = []
    cell_rows: list[Mapping[str, Any]] = []
    history_rows: list[Mapping[str, Any]] = []
    oof_rows: list[Mapping[str, Any]] = []
    role_oof_rows: list[Mapping[str, Any]] = []
    quality_rows: list[Mapping[str, Any]] = []
    config_metrics: list[Mapping[str, Any]] = []
    limitations: list[str] = []
    for case in manifest.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case["case_id"])
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
        record = _read_json(result_path)
        if not isinstance(record, Mapping):
            raise TypeError(f"case result root must be a mapping: {result_path}")
        case_records.append(
            {
                **dict(record),
                "output_group": case.get("output_group"),
                "case_directory": case.get(
                    "case_directory",
                    (Path("cases") / case_id).as_posix(),
                ),
                "resolved_config_path": case.get("resolved_config_path"),
            }
        )
        artifact_root = _case_artifact_root(case_directory, record)
        result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
        cell_rows.extend(_cell_rows(case_id, result, artifact_root))
        history_rows.extend(_history_rows(case_id, result, artifact_root))
        current_oof, limitation = _oof_rows(
            case_id,
            artifact_root,
            filename="oof_subject_predictions.parquet",
        )
        oof_rows.extend(current_oof)
        if limitation is not None:
            limitations.append(f"{case_id}: {limitation}")
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
        subject_oof_rows=tuple(oof_rows),
        role_oof_rows=tuple(role_oof_rows),
        quality_rows=tuple(quality_rows),
        trusted_config_metrics=tuple(config_metrics),
        limitations=tuple(dict.fromkeys(limitations)),
    )
