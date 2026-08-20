#!/usr/bin/env python3
"""Build a lightweight, reportable study from completed historical V2 cases.

The source studies remain untouched.  The derived study copies only resolved
configs, compact cell results, OOF predictions, confusion matrices, and
learning histories required by the V2 reporter; large diagnostics stay solely
in their original study folders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_SOURCES = (
    (
        "legacy214823",
        PIPELINE_ROOT
        / "artifacts/studies/20260817_214823_catalog_sweep_static-line-b-all-models-v2",
    ),
    (
        "stage1",
        PIPELINE_ROOT
        / "artifacts/studies/static_line_b_staged_v2"
        / "20260818_234543_catalog_sweep_staged-static-01-representation-baselines-v2",
    ),
    (
        "stage2",
        PIPELINE_ROOT
        / "artifacts/studies/static_line_b_staged_v2"
        / "20260819_074738_catalog_sweep_staged-static-02-competitive-routes-models-v2",
    ),
)
DEFAULT_OUTPUT = (
    PIPELINE_ROOT
    / "artifacts/studies/recovered_completed_cases_v2"
    / "20260819_completed_cases_legacy_stage1_stage2_v2"
)
REPORT_ARTIFACTS = (
    "oof_window_predictions.parquet",
    "oof_file_predictions.parquet",
    "oof_role_predictions.parquet",
    "oof_subject_predictions.parquet",
    "training_history.json",
    "confusion_matrices.json",
)
LARGE_CELL_FIELDS = frozenset(
    {
        "quality_diagnostics",
        "physical_recording_qc",
        "route_artifacts",
        "training_history",
        "dropped_records",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PIPELINE_ROOT.parent.parent).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_.-")
    if not cleaned:
        raise ValueError(f"cannot derive case id from {value!r}")
    return cleaned


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output
    return {prefix: value}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in values for key in row))
    if not fields:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _relocate_and_verify_report_metadata(working: Path, final: Path) -> None:
    """Replace staging-only metadata and revalidate every indexed artifact."""

    summary_path = working / "study_summary.json"
    index_path = working / "outputs_index.json"
    html_path = working / "STUDY_SUMMARY.html"
    for required in (summary_path, index_path, html_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["study_directory"] = str(final)
    _write_json(summary_path, summary)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["study_directory"] = str(final)
    for row in index.get("artifacts", ()):
        if row.get("path") == "outputs_index.json":
            continue
        path = working / str(row.get("path"))
        if not path.is_file():
            raise FileNotFoundError(path)
        row["bytes"] = path.stat().st_size
        row["sha256"] = _sha256(path)
    _write_json(index_path, index)

    html = html_path.read_text(encoding="utf-8")
    image_paths = re.findall(r"<img src='([^']+)'", html)
    if not image_paths or any(not (working / path).is_file() for path in image_paths):
        raise ValueError("HTML report does not reference a complete local PNG gallery")
    verified = json.loads(index_path.read_text(encoding="utf-8"))
    for row in verified.get("artifacts", ()):
        if row.get("path") == "outputs_index.json":
            continue
        path = working / str(row["path"])
        if _sha256(path) != row.get("sha256"):
            raise ValueError(f"output index hash mismatch after relocation: {path}")


def _completed_case_records(
    source_name: str,
    source_root: Path,
) -> list[dict[str, Any]]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    completed: list[dict[str, Any]] = []
    for result_path in sorted(source_root.glob("*/*/case_result.json")):
        record = json.loads(result_path.read_text(encoding="utf-8"))
        result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
        cells = result.get("cell_results") if isinstance(result, Mapping) else None
        if (
            record.get("status") != "passed"
            or not isinstance(cells, list)
            or not cells
            or any(not isinstance(cell, Mapping) or cell.get("status") != "passed" for cell in cells)
        ):
            continue
        identities = {
            (int(cell["repeat_index"]), int(cell["fold_index"]))
            for cell in cells
        }
        if len(identities) != len(cells):
            raise ValueError(f"duplicate repeat/fold cell in {result_path}")
        case_directory = result_path.parent
        artifact_root_raw = record.get("artifact_root")
        if not isinstance(artifact_root_raw, str) or not artifact_root_raw.strip():
            raise ValueError(f"missing artifact_root in {result_path}")
        artifact_root = (case_directory / artifact_root_raw).resolve()
        artifact_root.relative_to(case_directory.resolve())
        if not artifact_root.is_dir():
            raise FileNotFoundError(artifact_root)
        resolved_config = case_directory / "resolved_config.yaml"
        if not resolved_config.is_file():
            raise FileNotFoundError(resolved_config)
        relative_case = case_directory.relative_to(source_root).as_posix()
        completed.append(
            {
                "source_name": source_name,
                "source_root": source_root,
                "source_case": relative_case,
                "source_case_directory": case_directory,
                "source_result_path": result_path,
                "source_artifact_root": artifact_root,
                "source_resolved_config": resolved_config,
                "record": record,
                "cells": cells,
                "case_id": _safe_case_id(f"{source_name}__{relative_case}"),
            }
        )
    return completed


def _project_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in cell.items()
        if str(key) not in LARGE_CELL_FIELDS
    }


def _copy_report_artifacts(
    source_root: Path,
    destination_root: Path,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.name not in REPORT_ARTIFACTS:
            continue
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = _sha256(source)
        destination_hash = _sha256(destination)
        if source_hash != destination_hash:
            raise RuntimeError(f"copy hash mismatch: {source} -> {destination}")
        copied.append(
            {
                "source_path": _repo_path(source),
                "recovered_path": destination.relative_to(destination_root.parents[1]).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": destination_hash,
            }
        )
    return copied


def recover(
    *,
    sources: Iterable[tuple[str, Path]],
    output: Path,
    write_report: bool,
) -> Path:
    final_output = output.resolve()
    if final_output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing recovery directory: {final_output}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = final_output.with_name(
        f".{final_output.name}.staging-{os.getpid()}"
    )
    if output.exists():
        raise FileExistsError(f"recovery staging directory already exists: {output}")
    all_cases: list[dict[str, Any]] = []
    for name, root in sources:
        all_cases.extend(_completed_case_records(name, root.resolve()))
    if not all_cases:
        raise ValueError("no completed source cases were found")
    case_ids = [str(row["case_id"]) for row in all_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("recovered case ids are not unique")
    expected_cells = {
        (int(cell["repeat_index"]), int(cell["fold_index"]))
        for cell in all_cases[0]["cells"]
    }
    for source_case in all_cases:
        observed_cells = {
            (int(cell["repeat_index"]), int(cell["fold_index"]))
            for cell in source_case["cells"]
        }
        if observed_cells != expected_cells:
            raise ValueError(
                f"completed cases do not share one repeat/fold grid: {source_case['case_id']}"
            )

    output.mkdir()
    manifest_cases: list[dict[str, Any]] = []
    run_case_records: list[dict[str, Any]] = []
    recovery_cases: list[dict[str, Any]] = []
    flattened_configs: dict[str, dict[str, Any]] = {}
    total_cells = 0

    for source_case in all_cases:
        case_id = str(source_case["case_id"])
        case_directory = output / "cases" / case_id
        artifact_root = case_directory / "artifacts"
        case_directory.mkdir(parents=True)
        config_destination = case_directory / "resolved_config.yaml"
        shutil.copy2(source_case["source_resolved_config"], config_destination)
        config = yaml.safe_load(config_destination.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise TypeError(f"resolved config root is not a mapping: {config_destination}")
        config_file_sha256 = _sha256(config_destination)
        config_hash = hashlib.sha256(
            json.dumps(
                config,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        declared_hash = str(source_case["record"].get("config_sha256", ""))
        if declared_hash and declared_hash != config_hash:
            raise ValueError(
                f"resolved config canonical hash disagrees with case_result for {case_id}"
            )
        flattened_configs[case_id] = _flatten(config)

        copied = _copy_report_artifacts(
            source_case["source_artifact_root"], artifact_root
        )
        projected_cells = [_project_cell(cell) for cell in source_case["cells"]]
        total_cells += len(projected_cells)
        source_record = source_case["record"]
        result_payload = {
            "status": "passed",
            "config_id": source_record.get("result", {}).get("config_id", case_id),
            "cell_results": projected_cells,
            "failure_reasons": [],
            "output_dir": "artifacts",
        }
        recovered_record = {
            "schema_version": "ppg_frailty.recovered_case_result.v2",
            "case_id": case_id,
            "status": "passed",
            "artifact_root": "artifacts",
            "attempt": source_record.get("attempt"),
            "config_sha256": config_hash,
            "resolved_config_file_sha256": config_file_sha256,
            "elapsed_seconds": source_record.get("elapsed_seconds"),
            "started_utc": source_record.get("started_utc"),
            "finished_utc": source_record.get("finished_utc"),
            "source_study": _repo_path(source_case["source_root"]),
            "source_case": source_case["source_case"],
            "source_case_result_sha256": _sha256(source_case["source_result_path"]),
            "recovery_policy": "resolved_config_compact_cells_oof_confusion_history_no_diagnostics",
            "result": result_payload,
        }
        _write_json(case_directory / "case_result.json", recovered_record)
        output_group = str(source_case["source_case"]).split("/", 1)[0]
        manifest_cases.append(
            {
                "case_id": case_id,
                "case_directory": f"cases/{case_id}",
                "resolved_config_path": f"cases/{case_id}/resolved_config.yaml",
                "output_group": output_group,
                "is_reference": case_id == "stage1__raw__raw__compact_cnn",
                "source_name": source_case["source_name"],
                "source_case": source_case["source_case"],
                "config_sha256": config_hash,
                "resolved_config_file_sha256": config_file_sha256,
            }
        )
        run_case_records.append(recovered_record)
        recovery_cases.append(
            {
                "case_id": case_id,
                "source_name": source_case["source_name"],
                "source_study": _repo_path(source_case["source_root"]),
                "source_case": source_case["source_case"],
                "source_case_result_sha256": recovered_record[
                    "source_case_result_sha256"
                ],
                "config_sha256": config_hash,
                "resolved_config_file_sha256": config_file_sha256,
                "passed_cell_count": len(projected_cells),
                "copied_artifact_count": len(copied),
                "copied_bytes": sum(int(row["bytes"]) for row in copied),
                "copied_artifacts": copied,
            }
        )

    all_paths = sorted(
        set().union(*(set(values) for values in flattened_configs.values()))
    )
    varied_paths = [
        path
        for path in all_paths
        if len(
            {
                _json_value(flattened_configs[case_id].get(path, "__MISSING__"))
                for case_id in case_ids
            }
        )
        > 1
    ]
    controlled_paths = [path for path in all_paths if path not in varied_paths]
    resolved_parameter_rows = [
        {
            "case_id": case_id,
            "parameter_path": path,
            "value": _json_value(flattened_configs[case_id].get(path, "__MISSING__")),
        }
        for case_id in case_ids
        for path in all_paths
    ]
    varied_parameter_rows = [
        {
            "case_id": case_id,
            "parameter_path": path,
            "value": _json_value(flattened_configs[case_id].get(path, "__MISSING__")),
        }
        for path in varied_paths
        for case_id in case_ids
    ]
    controlled_parameter_rows = [
        {
            "parameter_path": path,
            "value": _json_value(flattened_configs[case_ids[0]][path]),
        }
        for path in controlled_paths
    ]
    tables = output / "tables"
    _write_csv(tables / "resolved_parameters.csv", resolved_parameter_rows)
    _write_csv(tables / "varied_parameters.csv", varied_parameter_rows)
    _write_csv(tables / "controlled_parameters.csv", controlled_parameter_rows)
    _write_csv(
        tables / "source_case_index.csv",
        (
            {
                key: value
                for key, value in row.items()
                if key != "copied_artifacts"
            }
            for row in recovery_cases
        ),
    )

    reference_case_id = next(
        (
            str(row["case_id"])
            for row in manifest_cases
            if bool(row.get("is_reference"))
        ),
        str(manifest_cases[0]["case_id"]),
    )
    plan = {
        "schema_version": "ppg_frailty.recovered_study_plan.v2",
        "study": {
            "study_id": "completed_cases_legacy_stage1_stage2_recovery_v2",
            "kind": "catalog_sweep",
            "purpose": "Read-only recovery of completed historical, Stage 1, and Stage 2 cases for complete reporting.",
            "flow_position": "Derived reporting archive; no new training and no new selection evidence.",
            "decision_role": "descriptive_recovery",
            "thesis_sections": [
                "All-class plots",
                "Confusion matrices",
                "Learning curves",
                "Parallel aggregation sensitivity",
            ],
        },
        "catalog": {
            "path": "tables/source_case_index.csv",
            "scope": "completed_cases_from_three_existing_studies",
            "balance_line": "source_case_declared_line_with_report_only_parallel_views",
        },
        "search": {
            "method": "completed_case_recovery",
            "runtime_sampling": "none",
            "interpretation": "descriptive only; source runs are unchanged",
        },
        "axes": [],
        "execution": {
            "repeats": sorted(
                {
                    int(cell["repeat_index"])
                    for source_case in all_cases
                    for cell in source_case["cells"]
                }
            ),
            "folds": sorted(
                {
                    int(cell["fold_index"])
                    for source_case in all_cases
                    for cell in source_case["cells"]
                }
            ),
            "jobs": 0,
        },
        "report": {
            "top_k": len(all_cases),
            "write_html": True,
            "write_static_figures": True,
            "calibration_bins": 10,
        },
        "recovery_sources": [
            {"name": name, "path": _repo_path(root)} for name, root in sources
        ],
    }
    manifest = {
        "schema_version": "ppg_frailty.recovered_study_manifest.v2",
        "status": "passed",
        "created_or_resumed_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "study": plan["study"],
        "execution": plan["execution"],
        "effective_jobs": 0,
        "reference_case_id": reference_case_id,
        "planned_case_count": len(all_cases),
        "passed_case_count": len(all_cases),
        "failed_case_count": 0,
        "not_run_case_count": 0,
        "planned_cell_count": total_cells,
        "reported_cell_count": total_cells,
        "passed_cell_count": total_cells,
        "failed_cell_count": 0,
        "not_run_cell_count": 0,
        "resumed_case_count": 0,
        "cases": manifest_cases,
    }
    _write_json(output / "study_manifest.json", manifest)
    (output / "study_plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        output / "study_run_result.json",
        {
            "schema_version": "ppg_frailty.recovered_study_run_result.v2",
            "status": "passed",
            "output_directory": str(final_output),
            "case_records": run_case_records,
            "recovery_only": True,
        },
    )
    _write_json(
        output / "recovery_manifest.json",
        {
            "schema_version": "ppg_frailty.completed_case_recovery.v2",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "policy": {
                "source_mutation": "none",
                "training_executed": False,
                "copied": list(REPORT_ARTIFACTS),
                "excluded": sorted(LARGE_CELL_FIELDS),
            },
            "case_count": len(all_cases),
            "cell_count": total_cells,
            "cases": recovery_cases,
        },
    )

    if write_report:
        from ppg_frailty.reporting import generate_study_report

        generate_study_report(output)
        _relocate_and_verify_report_metadata(output, final_output)
    os.replace(output, final_output)
    return final_output


def _source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return _safe_case_id(name.strip()), Path(path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover completed V2 cases into a lightweight reportable study without modifying sources."
    )
    parser.add_argument(
        "--source",
        action="append",
        type=_source,
        help="NAME=PATH; repeat to replace the three built-in source studies.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-report", action="store_true", help="Build recovery inputs but skip report generation."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = tuple(args.source) if args.source else DEFAULT_SOURCES
    output = recover(
        sources=sources,
        output=args.output,
        write_report=not args.no_report,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
