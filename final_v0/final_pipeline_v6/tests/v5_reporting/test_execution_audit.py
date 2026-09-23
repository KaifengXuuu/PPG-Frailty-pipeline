from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import ppg_frailty.v5_reporting.execution_audit as execution_audit
import ppg_frailty.v5_reporting.writer as writer
from ppg_frailty.v5_reporting.cli import main
from ppg_frailty.v5_reporting.contracts import ReportContractError


def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    pipeline, report = tmp_path / "pipeline_output", tmp_path / "report_output"
    pipeline.mkdir()
    report.mkdir()
    monkeypatch.setattr(execution_audit, "PIPELINE_OUTPUT_ROOT", pipeline)
    monkeypatch.setattr(execution_audit, "REPORT_OUTPUT_ROOT", report)
    monkeypatch.setattr(writer, "REPORT_OUTPUT_ROOT", report)
    return pipeline, report


def _plan(run: Path, *, repeats: tuple[int, ...] = (0, 1), folds: tuple[int, ...] = (0, 1)) -> None:
    (run / "study_plan.yaml").write_text(yaml.safe_dump({
        "study": {"study_id": "failed-study"},
        "cases": [{"case_id": "case-a"}],
        "execution": {"repeats": list(repeats), "folds": list(folds)},
    }), encoding="utf-8")


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*") if path.is_file()
    }


def test_cli_writes_atomic_execution_only_report_without_reading_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline, report = _roots(tmp_path, monkeypatch)
    run = pipeline / "failed-run"
    run.mkdir()
    _plan(run)
    (run / "study_manifest.json").write_text(json.dumps({
        "status": "failed", "cases": [{"case_id": "case-a"}],
    }), encoding="utf-8")
    (run / "study_run_result.json").write_text(json.dumps({
        "status": "failed",
        "case_records": [{
            "case_id": "case-a", "status": "failed", "error": "executor failed",
            "result": {
                "cell_results": [
                    {"repeat_index": 0, "fold_index": 0, "status": "passed"},
                    {"repeat_index": 0, "fold_index": 1, "status": "failed_closed"},
                ],
                "failure_reasons": ["r0_f1:synthetic failure"],
            },
        }],
    }), encoding="utf-8")
    (run / "progress_events.jsonl").write_text(
        json.dumps({"event": "cell_start", "case_id": "case-a", "repeat": 0, "fold": 0}) + "\n",
        encoding="utf-8",
    )
    (run / "oof_subject_predictions.parquet").write_bytes(b"must-not-be-read")
    before = _snapshot(run)

    assert main(["execution-audit", "--input", str(run)]) == 0
    response = json.loads(capsys.readouterr().out)
    output = report / "failed-run"
    assert Path(response["output_dir"]) == output
    assert before == _snapshot(run)
    assert not (output / "figures").exists()
    for relative in (
        "analysis_manifest.json", "outputs_index.json", "STUDY_SUMMARY.md", "STUDY_SUMMARY.html",
        "tables/execution_completeness.csv", "tables/execution_completeness.json",
        "tables/incomplete_cases.csv", "tables/failure_events.json", "tables/input_evidence.json",
        "tables/report_tables.xlsx",
    ):
        assert (output / relative).is_file()
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["report_scope"] == "execution_audit_only"
    assert manifest["prediction_artifacts_read"] is False
    assert all(manifest[field] is False for field in execution_audit._FLAGS)
    summary = json.loads((output / "tables/execution_completeness.json").read_text(encoding="utf-8"))[0]
    assert (
        summary["planned_cell_count"], summary["passed_cell_count"], summary["failed_closed_cell_count"]
    ) == (4, 1, 1)
    with pytest.raises(ReportContractError, match="already exists"):
        execution_audit.write_execution_audit(run)
    assert not any(path.name.startswith(".failed-run.staging-") for path in report.iterdir())


def test_interrupted_plan_only_input_and_complete_or_outside_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _ = _roots(tmp_path, monkeypatch)
    interrupted = pipeline / "interrupted"
    interrupted.mkdir()
    _plan(interrupted, repeats=(0,), folds=(0,))
    (interrupted / "progress_events.jsonl").write_text(
        json.dumps({"event": "cell_start", "case_id": "case-a", "repeat": 0, "fold": 0}) + "\n{bad\n",
        encoding="utf-8",
    )
    audit = execution_audit.collect_execution_audit(interrupted)
    assert audit["summary"]["study_status"] == "incomplete_interrupted"
    assert audit["summary"]["started_without_terminal_event_cell_count"] == 1
    assert audit["summary"]["failure_event_count"] == 1

    complete = pipeline / "complete"
    complete.mkdir()
    _plan(complete)
    (complete / "v5_data_manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
    with pytest.raises(ReportContractError, match="normal analyse_report"):
        execution_audit.collect_execution_audit(complete)

    outside = tmp_path / "outside"
    outside.mkdir()
    _plan(outside)
    with pytest.raises(ReportContractError, match="inside"):
        execution_audit.collect_execution_audit(outside)
