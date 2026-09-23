#!/usr/bin/env python3
"""Compare the scientific fold outputs of one V2 and one V5 case.

Directories may use V2's flat or V5's nested fold layout.  Paths, source-tree
identities, timings, cache accounting and learned-weight files are deliberately
excluded: they are operational side effects, not numerical pipeline results.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import numbers
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.parquet as pq


_FLAT_CELL = re.compile(r"repeat_(\d+)_fold_(\d+)$")
_REPEAT = re.compile(r"repeat_(\d+)$")
_FOLD = re.compile(r"fold_(\d+)$")
_METRICS_SCHEMA = "ppg_frailty.metrics_per_fold_seed.v2"
_OOF_FILES = frozenset(
    {
        "oof_window_predictions.parquet",
        "oof_file_predictions.parquet",
        "oof_role_predictions.parquet",
        "oof_subject_predictions.parquet",
        "oof_member_predictions.parquet",
    }
)
_STRUCTURED_FILES = frozenset(
    {
        "metrics_per_fold_seed.json",
        "confusion_matrices.json",
        "training_history.json",
        "quality_diagnostics.json",
        "physical_recording_qc.json",
        "route_artifacts.json",
        "route_window_sqi_evidence.jsonl.gz",
    }
)
_IDENTITY_COLUMNS = (
    "participant_id",
    "file_id",
    "role",
    "window_id",
    "member_index",
    "prediction_kind",
    "level",
)
_NON_SCIENTIFIC_COLUMNS = frozenset({"code_commit", "source_snapshot_hash"})
_CELL_RUNTIME_FIELDS = frozenset(
    {
        "code_commit",
        "source_version",
        "elapsed_seconds",
        "preprocessing_cache_summary",
        "learned_model_checkpoint",
    }
)
_INFERENCE_TIMINGS = frozenset(
    {
        "cpu_batch1_model_only_p50_ms",
        "cpu_batch1_model_only_p95_ms",
    }
)

def _validate_contract_arguments(*, atol: float, expected_folds: int | None) -> None:
    if (
        isinstance(atol, bool)
        or not isinstance(atol, numbers.Real)
        or not math.isfinite(float(atol))
        or not 0.0 <= float(atol) <= 1.0e-6
    ):
        raise ValueError("atol must be finite and within the frozen range [0, 1e-6]")
    if expected_folds is not None and (
        isinstance(expected_folds, bool) or not isinstance(expected_folds, numbers.Integral) or int(expected_folds) <= 0
    ):
        raise ValueError("expected_folds must be a positive integer")

def _coordinate(directory: Path) -> tuple[int, int] | None:
    flat = _FLAT_CELL.fullmatch(directory.name)
    if flat:
        return int(flat.group(1)), int(flat.group(2))
    fold, repeat = _FOLD.fullmatch(directory.name), _REPEAT.fullmatch(directory.parent.name)
    return (int(repeat.group(1)), int(fold.group(1))) if fold and repeat else None

def _discover(root: Path, names: Iterable[str]) -> dict[tuple[int, int, str], Path]:
    found: dict[tuple[int, int, str], Path] = {}
    for name in names:
        for path in root.rglob(name):
            coordinate = _coordinate(path.parent)
            if coordinate is None:
                continue
            key = (*coordinate, name)
            if key in found:
                raise ValueError("output contains duplicate fold artifacts; pass one case directory: " f"{key}")
            found[key] = path
    return found

def _normal(value: Any) -> Any:
    if value is None:
        return "<NULL>"
    if isinstance(value, float) and math.isnan(value):
        return "<NAN>"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_normal(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_normal(item) for item in value)
    return value

def _compare_value(left: Any, right: Any, *, atol: float) -> tuple[bool, float]:
    """Compare floats by ``atol`` and every discrete value exactly."""

    left, right = _normal(left), _normal(right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False, 0.0
        if set(left) != set(right):
            return False, 0.0
        results = [_compare_value(left[key], right[key], atol=atol) for key in left]
    elif isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return False, 0.0
        if len(left) != len(right):
            return False, 0.0
        results = [_compare_value(a, b, atol=atol) for a, b in zip(left, right)]
    elif isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right, 0.0
    elif isinstance(left, numbers.Integral) or isinstance(right, numbers.Integral):
        return (
            isinstance(left, numbers.Integral) and isinstance(right, numbers.Integral) and int(left) == int(right),
            0.0,
        )
    elif isinstance(left, numbers.Real) or isinstance(right, numbers.Real):
        if not isinstance(left, numbers.Real) or not isinstance(right, numbers.Real):
            return False, 0.0
        a, b = float(left), float(right)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b), 0.0
        difference = abs(a - b)
        return difference <= atol, difference
    else:
        return type(left) is type(right) and left == right, 0.0
    return all(match for match, _ in results), max((difference for _, difference in results), default=0.0)

def _aligned_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    keys = [name for name in _IDENTITY_COLUMNS if name in table.column_names]
    rows.sort(key=lambda row: tuple(str(_normal(row.get(name))) for name in keys))
    return table.column_names, rows

def _compare_table(v2_path: Path, v5_path: Path, *, atol: float) -> dict[str, Any]:
    v2_columns, v2_rows = _aligned_rows(v2_path)
    v5_columns, v5_rows = _aligned_rows(v5_path)
    failures = [] if v2_columns == v5_columns else ["schema_columns_differ"]
    if len(v2_rows) != len(v5_rows):
        failures.append("row_count_differs")
        return {
            "status": "failed",
            "rows_v2": len(v2_rows),
            "rows_v5": len(v5_rows),
            "max_abs_difference": None,
            "failures": failures,
        }
    maxima: list[float] = []
    for name in sorted(set(v2_columns) & set(v5_columns) - _NON_SCIENTIFIC_COLUMNS):
        results = [_compare_value(left.get(name), right.get(name), atol=atol) for left, right in zip(v2_rows, v5_rows)]
        maxima.append(max((value for _, value in results), default=0.0))
        if not all(match for match, _ in results):
            failures.append(f"column_differs:{name}")
    if "probabilities" in v2_columns and "probabilities" in v5_columns:
        predicted = lambda rows: tuple(  # noqa: E731 - compact local projection
            int(np.argmax(np.asarray(row["probabilities"], dtype=np.float64))) for row in rows
        )
        if predicted(v2_rows) != predicted(v5_rows):
            failures.append("predicted_class_differs")
    return {
        "status": "passed" if not failures else "failed",
        "rows_v2": len(v2_rows),
        "rows_v5": len(v5_rows),
        "max_abs_difference": max(maxima, default=0.0),
        "failures": failures,
    }

def _read_metrics(oof_path: Path) -> Mapping[str, Any]:
    manifest_path = oof_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell = manifest.get("cell", {}) if isinstance(manifest, Mapping) else {}
    metrics = cell.get("metrics", {}) if isinstance(cell, Mapping) else {}
    return metrics if isinstance(metrics, Mapping) else {}

def _read_structured(path: Path) -> Any:
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))

def _normalise_metrics(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove only non-scientific runtime fields from fold summaries."""

    if not isinstance(payload, Mapping):
        raise ValueError("metrics payload must be an object")
    value = copy.deepcopy(dict(payload))
    if value.get("schema_version") != _METRICS_SCHEMA:
        raise ValueError("metrics_per_fold_seed schema differs from contract")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("metrics_per_fold_seed.cells must be a list")
    evidence: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"metrics.cells[{index}] must be an object")
        excluded = sorted(field for field in _CELL_RUNTIME_FIELDS if cell.pop(field, None) is not None)
        operational = cell.get("operational_metrics")
        if isinstance(operational, dict) and isinstance(operational.get("inference_cost"), dict):
            inference = operational["inference_cost"]
            for field in _INFERENCE_TIMINGS:
                if inference.pop(field, None) is not None:
                    excluded.append(f"operational_metrics.inference_cost.{field}")
        evidence.append(
            {
                "repeat": cell.get("repeat_index"),
                "fold": cell.get("fold_index"),
                "excluded_fields": sorted(excluded),
            }
        )
    return value, evidence

def _compare_structured(v2_path: Path, v5_path: Path, *, atol: float) -> dict[str, Any]:
    left, right = _read_structured(v2_path), _read_structured(v5_path)
    excluded: dict[str, Any] | None = None
    failures: list[str] = []
    if v2_path.name == "metrics_per_fold_seed.json":
        try:
            left, v2_excluded = _normalise_metrics(left)
            right, v5_excluded = _normalise_metrics(right)
            excluded = {"v2": v2_excluded, "v5": v5_excluded}
        except (TypeError, ValueError) as exc:
            failures.append(f"metrics_contract_invalid:{exc}")
    matches, maximum = _compare_value(left, right, atol=atol)
    if not matches:
        failures.append("structured_content_differs")
    result = {
        "status": "passed" if not failures else "failed",
        "max_abs_difference": maximum,
        "failures": failures,
    }
    if excluded is not None:
        result["excluded_non_scientific_runtime"] = excluded
    return result

def compare_outputs(
    v2_output: str | Path,
    v5_output: str | Path,
    *,
    atol: float = 1.0e-6,
    expected_folds: int | None = None,
) -> dict[str, Any]:
    """Return a read-only, machine-readable scientific comparison."""

    _validate_contract_arguments(atol=atol, expected_folds=expected_folds)
    v2_root, v5_root = Path(v2_output).resolve(), Path(v5_output).resolve()
    for root in (v2_root, v5_root):
        if not root.is_dir():
            raise FileNotFoundError(root)
    v2_files, v5_files = _discover(v2_root, _OOF_FILES), _discover(v5_root, _OOF_FILES)
    v2_structured = _discover(v2_root, _STRUCTURED_FILES)
    v5_structured = _discover(v5_root, _STRUCTURED_FILES)

    missing_v5, unexpected_v5 = sorted(set(v2_files) - set(v5_files)), sorted(set(v5_files) - set(v2_files))
    missing_structured = sorted(set(v2_structured) - set(v5_structured))
    unexpected_structured = sorted(set(v5_structured) - set(v2_structured))
    tables, metric_failures, structured = [], [], []
    for key in sorted(set(v2_files) & set(v5_files)):
        row = {"repeat": key[0], "fold": key[1], "artifact": key[2]}
        row.update(_compare_table(v2_files[key], v5_files[key], atol=atol))
        tables.append(row)
        if key[2] == "oof_subject_predictions.parquet":
            matches, maximum = _compare_value(_read_metrics(v2_files[key]), _read_metrics(v5_files[key]), atol=atol)
            if not matches:
                metric_failures.append(
                    {
                        "repeat": key[0],
                        "fold": key[1],
                        "reason": "metrics_differ",
                        "maximum_float_difference": maximum,
                    }
                )
    for key in sorted(set(v2_structured) & set(v5_structured)):
        row = {"repeat": key[0], "fold": key[1], "artifact": key[2]}
        row.update(_compare_structured(v2_structured[key], v5_structured[key], atol=atol))
        structured.append(row)

    folds = {(repeat, fold) for repeat, fold, _ in v5_files}
    failed = any(
        (
            missing_v5,
            unexpected_v5,
            missing_structured,
            unexpected_structured,
            metric_failures,
            expected_folds is not None and len(folds) != expected_folds,
            any(row["status"] != "passed" for row in tables),
            any(row["status"] != "passed" for row in structured),
        )
    )
    rows = [*tables, *structured]
    return {
        "schema_version": "ppg_frailty.v2_v5_numeric_equivalence.v2",
        "status": "failed" if failed else "passed",
        "contract": {"atol": atol, "rtol": 0.0},
        "excluded_non_numerical_provenance_columns": sorted(_NON_SCIENTIFIC_COLUMNS),
        "excluded_runtime_fields": sorted(_CELL_RUNTIME_FIELDS | _INFERENCE_TIMINGS),
        "v2_output": str(v2_root),
        "v5_output": str(v5_root),
        "fold_count": len(folds),
        "expected_folds": expected_folds,
        "missing_v5": [list(item) for item in missing_v5],
        "unexpected_v5": [list(item) for item in unexpected_v5],
        "missing_structured_v5": [list(item) for item in missing_structured],
        "unexpected_structured_v5": [list(item) for item in unexpected_structured],
        "metric_failures": metric_failures,
        "tables": tables,
        "structured_artifacts": structured,
        "maximum_absolute_difference": max(
            (float(row["max_abs_difference"]) for row in rows if row["max_abs_difference"] is not None),
            default=0.0,
        ),
    }

def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare V2/V5 scientific fold outputs at atol<=1e-6, rtol=0.")
    parser.add_argument("--v2-output", required=True)
    parser.add_argument("--v5-output", required=True)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--expected-folds", type=int)
    parser.add_argument("--write", help="Optional JSON result path.")
    args = parser.parse_args(argv)
    try:
        _validate_contract_arguments(atol=args.atol, expected_folds=args.expected_folds)
    except ValueError as exc:
        parser.error(str(exc))
    result = compare_outputs(
        args.v2_output,
        args.v5_output,
        atol=args.atol,
        expected_folds=args.expected_folds,
    )
    if args.write:
        _atomic_json(Path(args.write).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
