"""V5 output paths and recoverable post-training Excel exports."""

from __future__ import annotations

from datetime import datetime, timezone
import json, math, os
from pathlib import Path
import re, time
from typing import Any, Iterable, Mapping

from ppg_frailty.reporting.tabular import ReportTable, write_excel_workbook

from .io import atomic_json, resolve_path
from .results import _read_csv


V5_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_OUTPUT_ROOT = V5_ROOT / "pipeline_output"
REPORT_OUTPUT_ROOT = V5_ROOT / "report_output"
MODEL_CONFIG_ROOT = V5_ROOT / "model_config"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXCEL_MAX_ROWS = 1_048_576
_PREDICTION_SHEETS = {
    "file": "file_predictions", "role": "role_predictions",
    "participant": "participant_predictions", "ensemble_member": "member_predictions",
}

def safe_output_name(value: str, *, label: str = "output name") -> str:
    """Return a portable single-directory name or fail closed."""
    name = str(value).strip()
    if not name or name in {".", ".."} or not _SAFE_NAME.fullmatch(name) or Path(name).name != name:
        raise ValueError(f"{label} must match {_SAFE_NAME.pattern!r} and contain no path separator")
    return name

def utc_stamp() -> str:
    """Filesystem-safe UTC timestamp used in immutable automatic run names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

def automatic_run_name(source_yaml: str | Path) -> str:
    """Name a run after its mandatory YAML source plus a UTC timestamp."""
    source = Path(source_yaml)
    stem = safe_output_name(source.stem, label="YAML filename stem")
    return f"{stem}_{utc_stamp()}"

def _inside(root: Path, value: str | Path, *, label: str) -> Path:
    return resolve_path(value, base=V5_ROOT, within=root, label=label)

def pipeline_run_path(*, source_yaml: str | Path, run_name: str | None = None) -> Path:
    """Resolve a new run path without creating it."""
    name = automatic_run_name(source_yaml) if run_name is None else safe_output_name(run_name)
    return PIPELINE_OUTPUT_ROOT / name

def existing_pipeline_run(value: str | Path) -> Path:
    """Resolve an existing canonical V5 run below ``pipeline_output``."""
    target = _inside(PIPELINE_OUTPUT_ROOT, value, label="pipeline run")
    if not target.is_dir():
        raise FileNotFoundError(target)
    if not (target / "study_manifest.json").is_file():
        raise FileNotFoundError(f"pipeline run lacks study_manifest.json: {target}")
    return target

def report_path_for_run(pipeline_run: str | Path, *, output_name: str | None = None) -> Path:
    """Resolve the immutable report target paired with one pipeline run."""
    run = existing_pipeline_run(pipeline_run)
    name = run.name if output_name is None else safe_output_name(output_name)
    return REPORT_OUTPUT_ROOT / name

def _excel_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _excel_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)

def _csv_table(path: Path) -> list[dict[str, Any]]:
    _, rows = _read_csv(path)
    return [{str(key): _excel_value(value) for key, value in row.items()} for row in rows]

def _prediction_files(root: Path) -> dict[str, list[Path]]:
    index_path = root / "tables" / "v5_fold_predictions.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    grouped = {level: [] for level in _PREDICTION_SHEETS}
    for row in _csv_table(index_path):
        level = str(row.get("level", ""))
        if level not in grouped or str(row.get("artifact_state", "")) == "empty":
            continue
        raw = Path(str(row.get("path", "")))
        if raw.is_absolute():
            raise ValueError(f"prediction index path must be relative: {raw}")
        target = resolve_path(raw, base=root, within=root, must_exist=True, label="prediction index path")
        if not target.is_file():
            raise FileNotFoundError(target)
        grouped[level].append(target)
    return grouped

def _parquet_shape(paths: Iterable[Path]) -> tuple[int, tuple[str, ...]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - dependency contract.
        raise RuntimeError("pyarrow is required for pipeline Excel export") from error
    total, fields = 0, ()
    for path in paths:
        file = parquet.ParquetFile(path)
        names = tuple(file.schema_arrow.names)
        if fields and fields != names:
            raise ValueError(f"prediction schemas differ while exporting Excel: {path}")
        fields = names
        total += int(file.metadata.num_rows)
    return total, fields

def _parquet_rows(paths: Iterable[Path], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    rows: list[dict[str, Any]] = []
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                rows.append({field: _excel_value(row.get(field)) for field in fields})
    return rows

def _workbook_tables(root: Path) -> tuple[list[ReportTable], dict[str, str]]:
    tables: list[ReportTable] = []
    skipped: dict[str, str] = {}
    def add(name: str, rows: list[dict[str, Any]], description: str) -> None:
        tables.append(ReportTable(name=name, rows=rows, description=description, compact=False))
    overview = (
        ("schema_version", "ppg_frailty.v5_pipeline_workbook.v1"), ("pipeline_run", root.name),
        ("source_manifest", "v5_data_manifest.json"), ("authoritative_predictions", "per-fold Parquet"),
        ("window_predictions_in_workbook", False),
        ("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    rows = [{"field": key, "value": value} for key, value in overview]
    add("run_overview", rows, "Pipeline workbook provenance")
    for sheet_name, filename in (("fold_prediction_index", "v5_fold_predictions.csv"),
                                 ("fold_models", "v5_fold_models.csv"),
                                 ("config_parameters", "v5_config_parameters.csv")):
        source = root / "tables" / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        add(sheet_name, _csv_table(source), f"Pipeline data table from {source.name}")
    for level, paths in _prediction_files(root).items():
        if not paths:
            skipped[level] = "no non-empty per-fold artifacts"
            continue
        row_count, fields = _parquet_shape(paths)
        if row_count + 1 > _EXCEL_MAX_ROWS:
            skipped[level] = f"{row_count} rows exceed Excel worksheet limit; use indexed Parquet"
            continue
        add(_PREDICTION_SHEETS[level], _parquet_rows(paths, fields), f"Per-fold {level} predictions")
    return tables, skipped

def _excel_status(status: str, **details: Any) -> dict[str, Any]:
    return {
        "schema_version": "ppg_frailty.v5_pipeline_excel_status.v1", "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workbook": "tables/pipeline_data.xlsx", **details,
    }

def export_pipeline_excel(pipeline_run: str | Path, *, allow_legacy_location: bool = False,
                          replace: bool = False) -> Mapping[str, Any]:
    """Create ``tables/pipeline_data.xlsx``; indexed Parquet remains authoritative."""
    raw = Path(pipeline_run)
    if allow_legacy_location:
        root = raw.resolve() if raw.is_absolute() else (V5_ROOT / raw).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
    else:
        root = existing_pipeline_run(raw)
    manifest = root / "v5_data_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"run must be indexed before Excel export: {manifest}; "
                                "run pipeline.py index first for a legacy V5 run")
    target = root / "tables" / "pipeline_data.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{time.time_ns()}.xlsx")
    if (target.exists() or target.is_symlink()) and not replace:
        raise FileExistsError(f"pipeline Excel already exists: {target}")

    try:
        tables, skipped = _workbook_tables(root)
        sheet_rows = {table.name: len(table.rows) for table in tables}
        write_excel_workbook(temporary, tables)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    status = _excel_status(
        "complete", sheet_rows=sheet_rows, skipped_prediction_layers=skipped,
        authoritative_predictions="per-fold Parquet; workbook is a convenience view",
    )
    atomic_json(root / "pipeline_excel_status.json", status)
    return status

def try_export_pipeline_excel(pipeline_run: str | Path, *, allow_legacy_location: bool = False,
                              replace: bool = False) -> Mapping[str, Any]:
    """Best-effort post-training export that never raises into model execution."""
    root = resolve_path(pipeline_run, base=V5_ROOT)
    try:
        return export_pipeline_excel(root, allow_legacy_location=allow_legacy_location, replace=replace)
    except Exception as error:  # noqa: BLE001 - recoverable presentation export.
        status = _excel_status(
            "failed_recoverable", error_type=type(error).__name__, error=str(error),
            pipeline_data_preserved=True,
            recovery="python sweep.py export-excel --pipeline-output <run>",
        )
        if root.is_dir():
            atomic_json(root / "pipeline_excel_status.json", status)
        return status


__all__ = [
    "MODEL_CONFIG_ROOT", "PIPELINE_OUTPUT_ROOT", "REPORT_OUTPUT_ROOT", "V5_ROOT",
    "automatic_run_name", "existing_pipeline_run", "export_pipeline_excel", "pipeline_run_path",
    "report_path_for_run", "safe_output_name", "try_export_pipeline_excel", "utc_stamp",
]
