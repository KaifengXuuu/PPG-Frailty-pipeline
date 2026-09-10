"""Execution-only reports for failed or interrupted V5 pipeline runs.

This module deliberately reads only small execution metadata.  It never opens
prediction tables or model weights, and it publishes into ``report_output``
without modifying the source ``pipeline_output`` tree.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from ppg_frailty.reporting.tabular import write_csv, write_excel_workbook_from_csv_directory
from ppg_frailty.v5.output_contract import PIPELINE_OUTPUT_ROOT, REPORT_OUTPUT_ROOT, safe_output_name

from .contracts import ReportContractError
from .writer import _write_json, _write_outputs_index, resolve_output_path


_PASS = {"passed", "success", "complete", "completed"}
_FAIL = {"failed", "failed_closed", "error", "aborted", "killed"}
_FLAGS = {
    "formal_result_available": False,
    "ranking_eligible": False,
    "inference_eligible": False,
    "selection_eligible": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    target = value.resolve() if value.is_absolute() else (PIPELINE_OUTPUT_ROOT.parent / value).resolve()
    root = PIPELINE_OUTPUT_ROOT.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ReportContractError(f"execution-audit input must stay inside {root}: {target}") from error
    if target == root or not target.is_dir():
        raise FileNotFoundError(f"pipeline run directory not found: {target}")
    if not (target / "study_plan.yaml").is_file() or (target / "study_plan.yaml").is_symlink():
        raise FileNotFoundError(f"execution-audit input lacks study_plan.yaml: {target}")
    return target


def _mapping(path: Path, *, yaml_input: bool = False) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_input else json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(value, Mapping):
        raise ReportContractError(f"execution evidence must contain a mapping: {path}")
    return dict(value)


def _case_id(row: Any) -> str | None:
    if not isinstance(row, Mapping):
        return None
    for field in ("case_id", "catalog_case_id", "config_id", "profile_id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in value if isinstance(row, Mapping)) if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ) else ()


def _planned_cases(plan: Mapping[str, Any]) -> tuple[str, ...]:
    values = [case_id for field in ("cases", "candidates") for row in _rows(plan.get(field, ()))
              if (case_id := _case_id(row))]
    legacy = plan.get("legacy_bridge")
    if isinstance(legacy, Mapping):
        values.extend(case_id for row in _rows(legacy.get("profiles", ())) if (case_id := _case_id(row)))
    return tuple(dict.fromkeys(values))


def _planned_cells(plan: Mapping[str, Any]) -> int:
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        execution = plan.get("resource")
    if not isinstance(execution, Mapping):
        return 0
    repeats, folds = execution.get("repeats"), execution.get("folds")
    return len(repeats) * len(folds) if all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in (repeats, folds)
    ) else 0


def _progress(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    paths: list[Path] = []
    for path in sorted(root.rglob("progress_events.jsonl"), key=str):
        if path.is_symlink() or "result_backup" in path.parts:
            continue
        paths.append(path)
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError("JSON root is not an object")
            except (json.JSONDecodeError, TypeError) as error:
                failures.append({
                    "case_id": "", "repeat": None, "fold": None, "classification": "malformed_progress",
                    "message": f"{type(error).__name__}: {error}", "source": relative, "line": line_number,
                })
                continue
            event = {**dict(value), "_source": relative, "_line": line_number}
            events.append(event)
            status = str(event.get("status", "")).lower()
            name = str(event.get("event", event.get("stage", ""))).lower()
            if status in _FAIL or any(token in name for token in ("fail", "error", "abort", "killed")):
                failures.append({
                    "case_id": event.get("case_id", ""),
                    "repeat": event.get("repeat", event.get("repeat_index")),
                    "fold": event.get("fold", event.get("fold_index")),
                    "classification": "failure_progress_event",
                    "message": event.get("message", event.get("error", status or name)),
                    "source": relative,
                    "line": line_number,
                })
    return events, failures, paths


def _case_results(root: Path, run_result: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[Path]]:
    records: dict[str, Mapping[str, Any]] = {}
    paths: list[Path] = []
    for path in sorted((root / ".runner_state").glob("*/case_result.json"), key=str):
        if path.is_file() and not path.is_symlink():
            payload = _mapping(path)
            paths.append(path)
            if case_id := _case_id(payload):
                records[case_id] = payload
    for row in _rows(run_result.get("case_records", ())):
        if case_id := _case_id(row):
            records[case_id] = row
    return records, paths


def _cell_rows(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    result = record.get("result")
    return _rows(result.get("cell_results", ())) if isinstance(result, Mapping) else ()


def collect_execution_audit(raw: str | Path) -> dict[str, Any]:
    """Collect execution state without reading prediction/model artifacts."""

    root = _source_path(raw)
    plan_path = root / "study_plan.yaml"
    plan = _mapping(plan_path, yaml_input=True)
    optional_names = ("study_manifest.json", "study_run_result.json", "v5_run_status.json", "v5_data_manifest.json")
    evidence_paths = [plan_path]
    documents: dict[str, Mapping[str, Any]] = {}
    for name in optional_names:
        path = root / name
        if path.is_symlink():
            raise ReportContractError(f"execution evidence must not be a symlink: {path}")
        if path.is_file():
            documents[name] = _mapping(path)
            evidence_paths.append(path)
    manifest = documents.get("study_manifest.json", {})
    run_result = documents.get("study_run_result.json", {})
    data_manifest = documents.get("v5_data_manifest.json", {})
    if str(data_manifest.get("status", "")).lower() in _PASS or (
        str(manifest.get("status", "")).lower() in _PASS
        and str(run_result.get("status", "")).lower() in _PASS
    ):
        raise ReportContractError("successful complete run must use the normal analyse_report.py run workflow")

    events, failure_rows, progress_paths = _progress(root)
    evidence_paths.extend(progress_paths)
    records, result_paths = _case_results(root, run_result)
    evidence_paths.extend(result_paths)
    planned = _planned_cases(plan)
    manifest_cases = tuple(case_id for row in _rows(manifest.get("cases", ())) if (case_id := _case_id(row)))
    event_cases = tuple(str(row["case_id"]) for row in events if row.get("case_id") not in (None, ""))
    case_ids = tuple(dict.fromkeys((*planned, *manifest_cases, *records, *event_cases)))
    expected_default = _planned_cells(plan)
    case_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        record = records.get(case_id, {})
        cells = _cell_rows(record)
        passed = sum(str(row.get("status", "")).lower() in _PASS for row in cells)
        failed = sum(str(row.get("status", "")).lower() in _FAIL for row in cells)
        started = {
            (row.get("repeat", row.get("repeat_index")), row.get("fold", row.get("fold_index")))
            for row in events
            if str(row.get("case_id", "")) == case_id and str(row.get("event", "")) == "cell_start"
        }
        started.discard((None, None))
        expected = expected_default or len(cells) or len(started)
        open_cells = max(0, len(started) - passed - failed)
        not_started = max(0, expected - passed - failed - open_cells)
        raw_status = str(record.get("status", "not_available"))
        if raw_status.lower() in _PASS and expected and passed == expected and not failed:
            status = "complete"
        elif failed or raw_status.lower() in _FAIL:
            status = "partial_failed" if passed else "failed"
        elif passed or started:
            status = "partial_interrupted"
        else:
            status = "not_started"
        result = record.get("result")
        reasons = list(result.get("failure_reasons", ())) if isinstance(result, Mapping) else []
        if record.get("error"):
            reasons.insert(0, str(record["error"]))
        for reason in reasons:
            failure_rows.append({
                "case_id": case_id, "repeat": None, "fold": None, "classification": "case_result_failure",
                "message": str(reason),
                "source": "study_run_result.json or .runner_state/case_result.json",
                "line": None,
            })
        case_events = [row for row in events if str(row.get("case_id", "")) == case_id]
        case_rows.append({
            "case_id": case_id, "planned": case_id in planned, "case_status": status,
            "case_result_status": raw_status, "expected_cell_count": expected,
            "passed_cell_count": passed, "failed_closed_cell_count": failed,
            "started_without_terminal_event_cell_count": open_cells, "not_started_cell_count": not_started,
            "last_event": case_events[-1].get("event", "") if case_events else "",
            "last_timestamp_utc": case_events[-1].get("timestamp_utc", "") if case_events else "", **_FLAGS,
        })

    totals = lambda field: sum(int(row[field]) for row in case_rows)
    failed_count = totals("failed_closed_cell_count")
    open_count = totals("started_without_terminal_event_cell_count")
    source_status = str(manifest.get("status", run_result.get("status", "absent_unfinalized")))
    status = "incomplete_failed" if failed_count or source_status.lower() in _FAIL else (
        "incomplete_interrupted" if open_count else "incomplete_unfinalized"
    )
    study = plan.get("study")
    summary = {
        "schema_version": "ppg_frailty.v5_execution_audit.v1",
        "study_id": study.get("study_id", root.name) if isinstance(study, Mapping) else root.name,
        "study_status": status,
        "source_study_status": source_status,
        "report_scope": "execution_audit_only",
        "planned_case_count": max(len(planned), len(case_rows)),
        "complete_case_count": sum(row["case_status"] == "complete" for row in case_rows),
        "incomplete_case_count": sum(row["case_status"] != "complete" for row in case_rows),
        "planned_cell_count": totals("expected_cell_count"),
        "passed_cell_count": totals("passed_cell_count"),
        "failed_closed_cell_count": failed_count,
        "started_without_terminal_event_cell_count": open_count,
        "not_started_cell_count": totals("not_started_cell_count"),
        "progress_event_count": len(events),
        "failure_event_count": len(failure_rows),
        **_FLAGS,
    }
    unique_evidence = tuple(dict.fromkeys(path for path in evidence_paths if path.is_file()))
    evidence = tuple({
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)
    } for path in unique_evidence)
    return {
        "root": root,
        "summary": summary,
        "tables": {
            "execution_completeness": (summary,),
            "incomplete_cases": tuple(case_rows),
            "failure_events": tuple(failure_rows),
            "input_evidence": evidence,
        },
    }


def write_execution_audit(raw: str | Path, *, output_name: str | None = None) -> dict[str, Any]:
    """Atomically publish an execution-only report below ``report_output``."""

    audit = collect_execution_audit(raw)
    source = audit["root"]
    name = safe_output_name(output_name or source.name, label="report name")
    target = resolve_output_path(REPORT_OUTPUT_ROOT / name)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        for name, rows in audit["tables"].items():
            write_csv(staging / "tables" / f"{name}.csv", rows)
            _write_json(staging / "tables" / f"{name}.json", rows)
        write_excel_workbook_from_csv_directory(staging / "tables/report_tables.xlsx", staging / "tables")
        summary = audit["summary"]
        text = (
            f"# {summary['study_id']} execution audit\n\n"
            f"Status: **{summary['study_status']}**\n\n"
            "> Execution audit only. No predictions, performance metrics, rankings, confidence intervals, "
            "P values, or model-selection conclusions were read or produced.\n\n"
            f"- Planned cells: {summary['planned_cell_count']}\n"
            f"- Passed cells: {summary['passed_cell_count']}\n"
            f"- Failed-closed cells: {summary['failed_closed_cell_count']}\n"
            f"- Interrupted cells: {summary['started_without_terminal_event_cell_count']}\n"
            f"- Not-started cells: {summary['not_started_cell_count']}\n"
        )
        (staging / "STUDY_SUMMARY.md").write_text(text, encoding="utf-8")
        (staging / "STUDY_SUMMARY.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Execution audit</title><h1>"
            + escape(str(summary["study_id"])) + " execution audit</h1><p>Status: <strong>"
            + escape(str(summary["study_status"]))
            + "</strong></p><blockquote>Execution audit only. "
            "No performance or inference result exists.</blockquote><pre>"
            + escape(text) + "</pre>",
            encoding="utf-8",
        )
        relative_source = source.relative_to(PIPELINE_OUTPUT_ROOT.resolve()).as_posix()
        _write_json(staging / "analysis_manifest.json", {
            "schema_version": "ppg_frailty.v5_execution_audit_report.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "complete",
            "report_scope": "execution_audit_only",
            "source_pipeline_output": relative_source,
            **_FLAGS,
            "summary": summary,
            "input_evidence": audit["tables"]["input_evidence"],
            "prediction_artifacts_read": False,
            "model_weights_read": False,
            "tables": tuple(audit["tables"]),
            "excel": "tables/report_tables.xlsx",
            "html": "STUDY_SUMMARY.html",
        })
        _write_outputs_index(staging)
        os.rename(staging, target)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return {
        "status": "complete",
        "report_scope": "execution_audit_only",
        "output_dir": str(target),
        "source_pipeline_output": relative_source,
        **_FLAGS,
    }
