"""Transactional, V5-confined report artifact writer."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

from ppg_frailty.reporting.tabular import (
    write_csv,
    write_excel_workbook_from_csv_directory,
)
from ppg_frailty.v5.output_contract import REPORT_OUTPUT_ROOT

from .contracts import (
    AnalysisProducts,
    LoadedReportData,
    ReportContractError,
    ReportRequest,
    ResolvedSelection,
    ValidationReport,
)
from .plots import generate_selected_figures
from .registry import MODULE_BY_NAME

V5_ROOT = Path(__file__).resolve().parents[3]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_outputs_index(directory: Path) -> None:
    rows: list[Mapping[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=str):
        if not path.is_file() or path.name == "outputs_index.json":
            continue
        relative = path.relative_to(directory).as_posix()
        rows.append({
            "path":
            relative,
            "bytes":
            path.stat().st_size,
            "sha256":
            _sha256(path),
            "kind":
            ("table_csv" if relative.startswith("tables/") and path.suffix == ".csv" else
             "table_json" if relative.startswith("tables/") and path.suffix == ".json" else "table_workbook"
             if relative == "tables/report_tables.xlsx" else "figure" if relative.startswith("figures/") else
             "summary_html" if path.suffix == ".html" else "summary_markdown" if path.suffix == ".md" else "manifest"),
        })
    _write_json(
        directory / "outputs_index.json",
        {
            "schema_version": "ppg_frailty.v5_output_index.v1",
            "self_indexed": False,
            "entries": rows,
        },
    )


def resolve_output_path(raw: Path) -> Path:
    if raw.is_absolute():
        target = raw.resolve()
    elif raw.parts and raw.parts[0] == REPORT_OUTPUT_ROOT.name:
        target = (V5_ROOT / raw).resolve()
    else:
        target = (REPORT_OUTPUT_ROOT / raw).resolve()
    try:
        target.relative_to(REPORT_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ReportContractError(f"report output must stay inside {REPORT_OUTPUT_ROOT}: {target}") from error
    if target == REPORT_OUTPUT_ROOT.resolve():
        raise ReportContractError("report output requires a child directory name")
    if target.exists() or target.is_symlink():
        raise ReportContractError(f"report output already exists; choose a new directory: {target}")
    return target


def _existing_report_output(raw: str | Path) -> Path:
    value = Path(raw)
    if value.is_absolute():
        target = value.resolve()
    elif value.parts and value.parts[0] == REPORT_OUTPUT_ROOT.name:
        target = (V5_ROOT / value).resolve()
    else:
        target = (REPORT_OUTPUT_ROOT / value).resolve()
    try:
        target.relative_to(REPORT_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ReportContractError(f"report output must stay inside {REPORT_OUTPUT_ROOT}: {target}") from error
    if not target.is_dir():
        raise FileNotFoundError(target)
    if not (target / "analysis_manifest.json").is_file():
        raise FileNotFoundError(f"report output lacks analysis_manifest.json: {target}")
    return target


def export_report_excel(
    report_output: str | Path,
    *,
    replace: bool = False,
) -> Mapping[str, Any]:
    """Regenerate ``report_tables.xlsx`` from authoritative derived CSV files."""

    root = _existing_report_output(report_output)
    table_directory = root / "tables"
    if not any(table_directory.glob("*.csv")):
        raise FileNotFoundError(f"report output has no derived CSV tables: {root}")
    target = table_directory / "report_tables.xlsx"
    if (target.exists() or target.is_symlink()) and not replace:
        raise FileExistsError(f"report Excel already exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp-{time.time_ns()}.xlsx")
    try:
        write_excel_workbook_from_csv_directory(temporary, table_directory)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    status = {
        "schema_version": "ppg_frailty.v5_report_excel_status.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workbook": "tables/report_tables.xlsx",
        "source": "derived report CSV tables",
    }
    _write_json(root / "report_excel_status.json", status)
    _write_outputs_index(root)
    return status


def _write_human_summary(
    directory: Path,
    *,
    data: LoadedReportData,
    request: ReportRequest,
    selection: ResolvedSelection,
    validation: ValidationReport,
    table_status: tuple[Mapping[str, Any], ...],
    figure_status: tuple[Mapping[str, Any], ...],
    module_status: tuple[Mapping[str, Any], ...],
    workbook_available: bool,
) -> None:
    """Write a portable HTML/Markdown index over modular report products."""

    title = f"PPG Frailty analysis — {directory.name}"
    figure_items = []
    for row in figure_status:
        name = html.escape(str(row.get("figure", "figure")))
        path = str(row.get("path", ""))
        status = html.escape(str(row.get("status", "unknown")))
        if status == "generated" and path:
            safe_path = html.escape(path, quote=True)
            figure_items.append(f"<figure><figcaption>{name}</figcaption>"
                                f'<img src="{safe_path}" alt="{name}"></figure>')
        else:
            reason = html.escape(str(row.get("reason", "")))
            figure_items.append(f"<p><strong>{name}</strong>: {status} — {reason}</p>")
    table_items = "".join('<li><a href="{csv}">{name}.csv</a> · '
                          '<a href="{json}">JSON</a> ({rows} rows)</li>'.format(
                              csv=html.escape(str(row.get("csv", "")), quote=True),
                              json=html.escape(str(row.get("json", "")), quote=True),
                              name=html.escape(str(row.get("table", "table"))),
                              rows=int(row.get("row_count", 0)),
                          ) for row in table_status)
    workbook_link = ('<p><a href="tables/report_tables.xlsx">Download report_tables.xlsx</a></p>'
                     if workbook_available else "<p>Excel workbook unavailable; CSV/JSON tables remain complete.</p>")
    modules = "".join(f"<tr><td>{html.escape(str(row.get('module', '')))}</td>"
                      f"<td>{html.escape(str(row.get('status', '')))}</td></tr>" for row in module_status)
    source_rows = "".join(f"<li>{html.escape(run.case_id)}: {html.escape(str(run.path))}</li>" for run in request.runs)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}}img{{max-width:100%;height:auto}}figure{{margin:2rem 0;padding:1rem;border:1px solid #d5d8dc}}table{{border-collapse:collapse}}th,td{{padding:.4rem .7rem;border:1px solid #ccd1d1}}code{{background:#f4f6f7;padding:.1rem .3rem}}</style>
</head><body><h1>{html.escape(title)}</h1>
<p>Mode: <code>{html.escape(request.mode)}</code>; validation: <code>{html.escape(validation.status)}</code>; source kind: <code>{html.escape(data.source_kind)}</code>.</p>
<h2>Inputs</h2><ul>{source_rows}</ul>
<h2>Modules</h2><table><thead><tr><th>Module</th><th>Status</th></tr></thead><tbody>{modules}</tbody></table>
<h2>Tables</h2><ul>{table_items}</ul>{workbook_link}
<h2>Figures</h2>{''.join(figure_items)}
<p>Exact parameters and provenance are recorded in <a href="analysis_manifest.json">analysis_manifest.json</a>.</p>
</body></html>"""
    (directory / "STUDY_SUMMARY.html").write_text(document, encoding="utf-8")
    markdown = [
        f"# {title}",
        "",
        f"- Mode: `{request.mode}`",
        f"- Validation: `{validation.status}`",
        f"- Source kind: `{data.source_kind}`",
        "",
        "## Tables",
        "",
    ]
    markdown.extend(f"- [{row['table']}.csv]({row['csv']}) ({row['row_count']} rows)" for row in table_status)
    markdown.extend(("", "## Figures", ""))
    markdown.extend(f"- `{row.get('figure')}`: {row.get('status')}" for row in figure_status)
    (directory / "STUDY_SUMMARY.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _request_manifest(request: ReportRequest) -> Mapping[str, Any]:
    return {
        **asdict(request),
        "runs": tuple({
            "case_id": run.case_id,
            "path": str(run.path)
        } for run in request.runs),
        "output_dir": None if request.output_dir is None else str(request.output_dir),
    }


def _module_rows(
    selection: ResolvedSelection,
    table_rows: tuple[Mapping[str, Any], ...],
    figure_rows: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    table_by_name = {str(row["table"]): row for row in table_rows}
    figure_by_name = {str(row["figure"]): row for row in figure_rows}
    output: list[Mapping[str, Any]] = []
    for name in selection.modules:
        spec = MODULE_BY_NAME[name]
        selected_tables = tuple(value for value in spec.tables if value in selection.tables)
        selected_figures = tuple(value for value in spec.figures if value in selection.figures)
        states = [str(table_by_name[value]["status"]) for value in selected_tables]
        states.extend(str(figure_by_name[value]["status"]) for value in selected_figures)
        status = ("computed_no_output_selected" if not states else "completed" if all(
            value == "generated" for value in states) else "completed_with_missing_outputs")
        output.append({
            "module": name,
            "status": status,
            "selected_tables": selected_tables,
            "selected_figures": selected_figures,
        })
    return tuple(output)


def write_report(
    data: LoadedReportData,
    products: AnalysisProducts,
    request: ReportRequest,
    selection: ResolvedSelection,
    validation: ValidationReport,
) -> Path:
    """Write a new report atomically; never mutate or merge an old report."""

    if request.output_dir is None:
        raise ReportContractError("run mode requires --output-dir")
    target = resolve_output_path(request.output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        table_status: list[Mapping[str, Any]] = []
        for name in selection.tables:
            if name not in products.tables:
                raise ReportContractError(f"analysis did not provide selected table {name!r}")
            rows = tuple(_jsonable(row) for row in products.tables[name])
            write_csv(staging / "tables" / f"{name}.csv", rows)
            _write_json(staging / "tables" / f"{name}.json", rows)
            table_status.append({
                "table": name,
                "status": "generated",
                "row_count": len(rows),
                "csv": f"tables/{name}.csv",
                "json": f"tables/{name}.json",
            })

        figure_status = generate_selected_figures(
            data,
            products,
            request,
            selection.figures,
            staging / "figures",
        )
        module_status = _module_rows(
            selection,
            tuple(table_status),
            figure_status,
        )
        workbook = staging / "tables" / "report_tables.xlsx"
        if table_status:
            try:
                write_excel_workbook_from_csv_directory(
                    workbook,
                    staging / "tables",
                )
                excel_status: Mapping[str, Any] = {
                    "status": "complete",
                    "workbook": "tables/report_tables.xlsx",
                    "source": "derived report CSV tables",
                }
            except Exception as error:  # noqa: BLE001 - CSV/JSON remain authoritative.
                excel_status = {
                    "status": "failed_recoverable",
                    "workbook": "tables/report_tables.xlsx",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "report_tables_preserved": True,
                }
        else:
            excel_status = {
                "status": "not_applicable_no_tables_selected",
                "workbook": None,
            }
        _write_json(staging / "table_status.json", table_status)
        _write_json(staging / "figure_status.json", figure_status)
        _write_json(staging / "module_status.json", module_status)
        _write_json(staging / "validation_report.json", validation)
        _write_json(staging / "report_excel_status.json", excel_status)
        _write_human_summary(
            staging,
            data=data,
            request=request,
            selection=selection,
            validation=validation,
            table_status=tuple(table_status),
            figure_status=figure_status,
            module_status=module_status,
            workbook_available=excel_status["status"] == "complete",
        )
        manifest_status = ("complete" if
                           (all(row.get("status") == "generated" for row in figure_status)
                            and excel_status["status"] in {"complete", "not_applicable_no_tables_selected"}) else
                           "complete_with_missing_outputs")
        _write_json(
            staging / "analysis_manifest.json",
            {
                "schema_version":
                "ppg_frailty.v5_analysis_manifest.v1",
                "created_utc":
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status":
                manifest_status,
                "source_kind":
                data.source_kind,
                "request":
                _request_manifest(request),
                "selection":
                selection,
                "validation":
                validation,
                "source_plan":
                data.collected.plan,
                "source_manifest":
                data.collected.manifest,
                "input_artifacts":
                tuple({
                    **asdict(row), "path": str(row.path)
                } for row in data.artifact_records),
                "notes":
                products.notes,
                "excel":
                excel_status,
                "html":
                "STUDY_SUMMARY.html",
                "input_hash_policy": ("not_computed_for_large_prediction_tables; exact OOF schema and "
                                      "row-level provenance hashes were validated"),
            },
        )

        _write_outputs_index(staging)
        os.rename(staging, target)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return target


__all__ = [
    "V5_ROOT",
    "export_report_excel",
    "resolve_output_path",
    "write_report",
]
