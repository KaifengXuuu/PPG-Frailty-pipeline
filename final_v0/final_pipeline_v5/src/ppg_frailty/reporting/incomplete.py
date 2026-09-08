"""Execution-only inventory for an interrupted study.

No performance value is reconstructed when predictions or the terminal study
manifest are absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .components import build_pipeline_test_component_rows, write_test_component_markdown
from .profiles import reporter_profile_rows, write_reporter_methods
from .tabular import ReportTable, write_csv, write_excel_workbook


@dataclass(frozen=True)
class IncompleteStudyReportResult:
    study_directory: Path
    summary_markdown: Path
    summary_html: Path
    methods_markdown: Path
    interpretation_markdown: Path
    outputs_index: Path
    status: str
    table_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "incomplete_report_regenerated",
            "study_status": self.status,
            "report_scope": "execution_audit_only",
            "formal_result_available": False,
            "ranking_eligible": False,
            "inference_eligible": False,
            "selection_eligible": False,
            "study_dir": str(self.study_directory),
            "root_report": str(self.summary_markdown),
            "root_report_html": str(self.summary_html),
            "outputs_index": str(self.outputs_index),
        }


def is_incomplete_study_directory(study_directory: str | Path) -> bool:
    root = Path(study_directory)
    return root.is_dir() and (root / "study_plan.yaml").is_file() and not (root / "study_manifest.json").exists()


def _mapping(path: Path) -> dict[str, Any]:
    value = (yaml.safe_load(path.read_text(
        encoding="utf-8")) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8")))
    return dict(value) if isinstance(value, Mapping) else {}


def _case_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("cases", "candidates", "configurations"):
        values = plan.get(key)
        if isinstance(values, (list, tuple)):
            return tuple(
                str(row.get("case_id", row.get("config_id", row.get("name")))) for row in values
                if isinstance(row, Mapping))
    return ()


def _events(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.rglob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"event": "invalid_json", "message": line[:300]}
            if isinstance(value, Mapping):
                output.append({"source": path.relative_to(root).as_posix(), "line": line_no, **value})
    return output


def _cases(root: Path, declared: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    paths = {path.parent.name: path for path in root.rglob("case_result.json")}
    for case_id in sorted(set(declared) | set(paths)):
        payload = _mapping(paths[case_id]) if case_id in paths else {}
        status = str(payload.get("status", "not_run"))
        output.append({
            "case_id": case_id,
            "status": status,
            "complete": status.lower() in {"passed", "success", "complete", "completed"},
            "case_result": paths[case_id].relative_to(root).as_posix() if case_id in paths else None,
            "error": payload.get("error", payload.get("message")),
        })
    return output


def generate_incomplete_study_report(study_directory: str | Path) -> IncompleteStudyReportResult:
    root = Path(study_directory).resolve()
    plan_path = root / "study_plan.yaml"
    if not root.is_dir() or not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if (root / "study_manifest.json").exists():
        raise ValueError("use the formal analyzer when study_manifest.json exists")
    plan = _mapping(plan_path)
    cases = _cases(root, _case_ids(plan))
    events = _events(root)
    failures = [
        row for row in events if any(word in str(row.get("event", row.get("status", ""))).lower()
                                     for word in ("fail", "error", "abort", "kill"))
    ]
    summary = {
        "study_id":
        plan.get("study", {}).get("study_id", root.name) if isinstance(plan.get("study"), Mapping) else root.name,
        "study_status": "incomplete",
        "planned_case_count": len(cases),
        "completed_case_count": sum(bool(row["complete"]) for row in cases),
        "event_count": len(events),
        "failure_count": len(failures),
        "formal_result_available": False,
    }
    # A synthetic manifest lets the shared component collector inspect configs
    # that were materialized before interruption.
    synthetic = {
        "cases": [{
            "case_id": row["case_id"],
            "resolved_config_path": f"cases/{row['case_id']}/resolved_config.yaml"
        } for row in cases]
    }
    components = build_pipeline_test_component_rows(root, synthetic)
    profiles = reporter_profile_rows(components)
    unavailable = [{
        "module": module,
        "status": "N/A_incomplete_study",
        "reason": "terminal prediction evidence is unavailable"
    } for module in ("classification_metrics", "ranking", "participant_cluster_inference", "comparison_selection")]
    products = {
        "execution_completeness": [summary],
        "incomplete_cases": [row for row in cases if not row["complete"]],
        "case_status": cases,
        "failure_events": failures,
        "progress_events": events,
        "test_components": components,
        "reporter_profiles": profiles,
        "unavailable_analysis": unavailable,
    }
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    for name, rows in products.items():
        write_csv(tables / f"{name}.csv", rows)
        (tables / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    write_excel_workbook(tables / "report_tables.xlsx",
                         [ReportTable(name, rows, compact=False) for name, rows in products.items()])
    methods = write_reporter_methods(root, components)
    write_test_component_markdown(root, components)
    interpretation = root / "RESULT_INTERPRETATION.md"
    interpretation.write_text(
        "# Result interpretation\n\nThe study is incomplete; no performance, P value, ranking, or model selection is reported.\n",
        encoding="utf-8",
    )
    summary_md = root / "STUDY_SUMMARY.md"
    summary_md.write_text(
        f"# {summary['study_id']}\n\nStatus: **incomplete**\n\n- Completed cases: {summary['completed_case_count']}/{summary['planned_case_count']}\n- Failures: {summary['failure_count']}\n\nExecution audit only; scientific outputs are unavailable.\n",
        encoding="utf-8",
    )
    summary_html = root / "STUDY_SUMMARY.html"
    summary_html.write_text(
        f"<!doctype html><meta charset='utf-8'><h1>{summary['study_id']}</h1><p>Incomplete execution audit; scientific outputs are unavailable.</p>",
        encoding="utf-8",
    )
    index = root / "outputs_index.json"
    files = [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size
    } for path in sorted(root.rglob("*")) if path.is_file() and path != index]
    index.write_text(
        json.dumps({
            "schema_version": "ppg_frailty.incomplete_report.v2",
            "artifacts": files
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return IncompleteStudyReportResult(root, summary_md, summary_html, methods, interpretation, index, "incomplete",
                                       len(products))


__all__ = ["IncompleteStudyReportResult", "generate_incomplete_study_report", "is_incomplete_study_directory"]
