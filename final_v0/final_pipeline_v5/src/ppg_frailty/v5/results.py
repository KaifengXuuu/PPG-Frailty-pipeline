"""Index per-fold data products without duplicating prediction rows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .io import atomic_json, file_sha256, resolve_path


_FLAT_CELL = re.compile(r"^repeat_(\d+)_fold_(\d+)$")
_REPEAT = re.compile(r"^repeat_(\d+)$")
_FOLD = re.compile(r"^fold_(\d+)$")
_COMPLETE = frozenset({"passed", "success", "complete", "completed"})
_PREDICTIONS = (
    ("window", "oof_window_predictions.parquet"),
    ("file", "oof_file_predictions.parquet"),
    ("role", "oof_role_predictions.parquet"),
    ("participant", "oof_subject_predictions.parquet"),
    ("ensemble_member", "oof_member_predictions.parquet"),
)
_PREDICTION_FIELDS = """case_id repeat fold level prediction_kind rows columns row_groups bytes
    parquet_schema_version artifact_state empty_reason sha256 path""".split()
_FOLD_FIELDS = """case_id repeat fold status config_id config_hash split_seed training_seed model_id model_machine_id
    representation_mode parameter_count model_hash state_hash fold_hash preprocessing_hash feature_hash aggregation_rule
    balanced_accuracy macro_f1 fitted_provenance_json model_factory_provenance_json frozen_model_run_provenance_json
    run_manifest_path checkpoint_schema learned_weight_checkpoint checkpoint_manifest_sha256 checkpoint_state_sha256
    checkpoint_deployment_status""".split()
_PARAMETER_FIELDS = "case_id config_id parameter_path value_json".split()

def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, Mapping):
        raise TypeError(f"file root must be a mapping: {path}")
    return dict(value)

def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    materialized = tuple(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(fields), extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]

def _parquet_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - explicit dependency error.
        raise RuntimeError("pyarrow is required to index formal prediction tables") from error
    file = parquet.ParquetFile(path)
    metadata = file.schema_arrow.metadata or {}

    def decoded(key: str) -> str | None:
        raw = metadata.get(key.encode())
        return None if raw is None else raw.decode()

    return {
        "rows": int(file.metadata.num_rows), "columns": int(file.metadata.num_columns),
        "row_groups": int(file.metadata.num_row_groups), "schema_version": decoded("schema_version"),
        "artifact_state": decoded("artifact_state"), "empty_reason": decoded("empty_reason"),
    }

def _case_artifact_root(study: Path, case: Mapping[str, Any]) -> Path | None:
    directory = resolve_path(case["case_directory"], base=study, within=study, label="case directory")
    result_path = directory / "case_result.json"
    if not result_path.is_file():
        return None
    raw = _read_mapping(result_path).get("artifact_root")
    if raw in (None, ""):
        return None
    artifact = resolve_path(str(raw), base=directory, within=directory, label="case artifact root")
    return artifact if artifact.is_dir() else None

def _cell_coordinates(manifest: Path) -> tuple[int, int] | None:
    flat = _FLAT_CELL.fullmatch(manifest.parent.name)
    if flat:
        return int(flat.group(1)), int(flat.group(2))
    fold, repeat = _FOLD.fullmatch(manifest.parent.name), _REPEAT.fullmatch(manifest.parent.parent.name)
    return None if fold is None or repeat is None else (int(repeat.group(1)), int(fold.group(1)))

def _cell_manifests(artifact: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in artifact.rglob("run_manifest.json") if _cell_coordinates(path)),
                        key=Path.as_posix))

def _checkpoint_row(root: Path, manifest: Path, cell: Mapping[str, Any]) -> dict[str, str]:
    declared = cell.get("learned_model_checkpoint")
    if not isinstance(declared, Mapping):
        return {
            "checkpoint_schema": "", "learned_weight_checkpoint": "not_persisted_by_legacy_v2_outer_cv",
            "checkpoint_manifest_sha256": "", "checkpoint_state_sha256": "",
            "checkpoint_deployment_status": "unavailable",
        }
    raw = declared.get("manifest_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"checkpoint manifest path missing in {manifest}")
    checkpoint = resolve_path(raw, base=manifest.parent, within=root, must_exist=True, label="checkpoint manifest")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed, expected = file_sha256(checkpoint), str(declared.get("manifest_sha256", ""))
    if expected and observed != expected:
        raise ValueError(f"checkpoint manifest hash mismatch: {checkpoint}")
    return {
        "checkpoint_schema": str(declared.get("schema_version", "")),
        "learned_weight_checkpoint": checkpoint.relative_to(root).as_posix(),
        "checkpoint_manifest_sha256": observed, "checkpoint_state_sha256": str(declared.get("state_sha256", "")),
        "checkpoint_deployment_status": str(declared.get("deployment_status", "")),
    }

def _safe_case_name(case_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._")
    return name or hashlib.sha256(case_id.encode()).hexdigest()[:12]

def _publish_median_selections(root: Path, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from .checkpoints import select_median_fold

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    published = []
    for case_id in sorted(grouped):
        selection = select_median_fold(grouped[case_id])
        if selection is None:
            continue
        path = root / "models" / _safe_case_name(case_id) / "median_fold" / "selection.json"
        payload = {
            **selection, "case_id": case_id, "complete_5x5": int(selection["eligible_fold_count"]) == 25,
            "publication": "reference_to_authoritative_per_fold_bundle_no_weight_copy",
        }
        atomic_json(path, payload)
        published.append({
            "case_id": case_id, "model_role": selection["selection_role"],
            "deployment_status": selection["deployment_status"],
            "selection_manifest": path.relative_to(root).as_posix(),
            "bundle_manifest": selection["checkpoint_manifest"],
            "bundle_manifest_sha256": selection["checkpoint_manifest_sha256"], "repeat": selection["repeat"],
            "fold": selection["fold"], "balanced_accuracy": selection["balanced_accuracy"],
            "complete_5x5": payload["complete_5x5"],
            "model_input_boundary": "already_preprocessed_fold_model_input",
        })
    return published

def _flatten_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _flatten_leaves(value[key], f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        yield prefix, list(value)
    else:
        yield prefix, value

def _resolved_config(study: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(case["resolved_config_path"], base=study, within=study, must_exist=True,
                        label="resolved config")
    return _read_mapping(path)

def _fold_row(root: Path, manifest: Path, case_id: str, cell: Mapping[str, Any],
              config: Mapping[str, Any]) -> dict[str, Any]:
    nested = {name: value if isinstance(value, Mapping) else {} for name, value in (
        ("fitted", cell.get("fitted_provenance")), ("factory", cell.get("model_factory_provenance")),
        ("frozen", cell.get("frozen_model_run_provenance")), ("metrics", cell.get("metrics")),
        ("operational", cell.get("operational_metrics")),
    )}
    parameter_count = nested["factory"].get("parameter_count")
    parameter_count = nested["operational"].get("parameter_count") if parameter_count is None else parameter_count
    row = {
        "case_id": case_id, "repeat": int(cell["repeat_index"]), "fold": int(cell["fold_index"]),
        "status": str(cell.get("status", "")), "config_id": str(cell.get("config_id", config.get("config_id", ""))),
        "config_hash": str(cell.get("config_hash", "")), "split_seed": cell.get("split_seed"),
        "training_seed": cell.get("training_seed"), "model_id": str(cell.get("model_id", "")),
        "model_machine_id": str(cell.get("model_machine_id", "")),
        "representation_mode": str(cell.get("representation_mode", "")), "parameter_count": parameter_count,
        "model_hash": str(cell.get("model_hash", "")), "state_hash": str(nested["fitted"].get("state_hash", "")),
        "fold_hash": str(nested["fitted"].get("fold_hash", cell.get("fold_hash", ""))),
        "preprocessing_hash": str(cell.get("preprocessing_hash", "")),
        "feature_hash": str(cell.get("feature_hash", "")), "aggregation_rule": str(cell.get("balance_line", "")),
        "balanced_accuracy": nested["metrics"].get("balanced_accuracy"), "macro_f1": nested["metrics"].get("macro_f1"),
        "fitted_provenance_json": _json_cell(nested["fitted"]),
        "model_factory_provenance_json": _json_cell(nested["factory"]),
        "frozen_model_run_provenance_json": _json_cell(nested["frozen"]),
        "run_manifest_path": manifest.relative_to(root).as_posix(),
    }
    return {**row, **_checkpoint_row(root, manifest, cell)}

def _prediction_rows(root: Path, manifest: Path, case_id: str, repeat: int, fold: int,
                     hash_files: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, missing = [], []
    for level, filename in _PREDICTIONS:
        path = manifest.parent / filename
        if not path.is_file():
            missing.append({"case_id": case_id, "repeat": repeat, "fold": fold, "level": level,
                            "reason": f"missing:{filename}"})
            continue
        metadata = _parquet_metadata(path)
        rows.append({
            "case_id": case_id, "repeat": repeat, "fold": fold, "level": level,
            "prediction_kind": "ensemble_member" if level == "ensemble_member" else "declared",
            "rows": metadata["rows"], "columns": metadata["columns"], "row_groups": metadata["row_groups"],
            "bytes": path.stat().st_size, "parquet_schema_version": metadata["schema_version"] or "",
            "artifact_state": metadata["artifact_state"] or "", "empty_reason": metadata["empty_reason"] or "",
            "sha256": file_sha256(path) if hash_files else "not_computed", "path": path.relative_to(root).as_posix(),
        })
    return rows, missing

def build_study_data_index(study_directory: str | Path, *, hash_prediction_files: bool = False) -> Mapping[str, Any]:
    """Write economical CSV indexes for every case/repeat/fold prediction and learned model."""
    root = Path(study_directory).resolve()
    manifest_path = root / "study_manifest.json"
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    study = _read_mapping(manifest_path)
    cases = tuple(row for row in study.get("cases", ()) if isinstance(row, Mapping))
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    status = str(study.get("status", ""))
    if status and status not in _COMPLETE:
        missing.append({"reason": f"study_status:{status}"})
    execution = study.get("execution") if isinstance(study.get("execution"), Mapping) else {}
    repeats, folds = execution.get("repeats"), execution.get("folds")
    expected = ({(int(repeat), int(fold)) for repeat in repeats for fold in folds}
                if isinstance(repeats, list) and isinstance(folds, list) else None)
    require_checkpoints = study.get("output_layout") == "comparison/repeat/fold"

    for case in cases:
        case_id = str(case["case_id"])
        directory = resolve_path(case["case_directory"], base=root, within=root, label="case directory")
        case_result = directory / "case_result.json"
        if case_result.is_file():
            case_status = str(_read_mapping(case_result).get("status", ""))
            if case_status not in _COMPLETE:
                missing.append({"case_id": case_id, "reason": f"case_status:{case_status}"})
        config = _resolved_config(root, case)
        parameter_rows.extend({
            "case_id": case_id, "config_id": str(config.get("config_id", "")), "parameter_path": path,
            "value_json": _json_cell(value),
        } for path, value in _flatten_leaves(config))
        artifact = _case_artifact_root(root, case)
        if artifact is None:
            missing.append({"case_id": case_id, "reason": "artifact_root_missing"})
            continue
        manifests = _cell_manifests(artifact)
        if not manifests:
            missing.append({"case_id": case_id, "reason": "fold_manifests_missing"})
            continue
        observed: set[tuple[int, int]] = set()
        for manifest in manifests:
            payload = _read_mapping(manifest)
            cell = payload.get("cell")
            if not isinstance(cell, Mapping):
                raise TypeError(f"run manifest lacks cell mapping: {manifest}")
            coordinates = _cell_coordinates(manifest)
            assert coordinates is not None
            repeat, fold = int(cell["repeat_index"]), int(cell["fold_index"])
            if (repeat, fold) != coordinates:
                raise ValueError(f"fold coordinate mismatch in {manifest}: manifest={(repeat, fold)}, "
                                 f"directory={coordinates}")
            if coordinates in observed:
                missing.append({"case_id": case_id, "repeat": repeat, "fold": fold,
                                "reason": "duplicate_fold_coordinate"})
            observed.add(coordinates)
            row = _fold_row(root, manifest, case_id, cell, config)
            row["status"] = str(cell.get("status", payload.get("status", "")))
            fold_rows.append(row)
            if row["status"] not in _COMPLETE:
                missing.append({"case_id": case_id, "repeat": repeat, "fold": fold,
                                "reason": f"fold_status:{row['status']}"})
            if require_checkpoints and row["checkpoint_deployment_status"] == "unavailable":
                missing.append({"case_id": case_id, "repeat": repeat, "fold": fold,
                                "reason": "learned_weight_checkpoint_missing"})
            current, absent = _prediction_rows(root, manifest, case_id, repeat, fold, hash_prediction_files)
            prediction_rows.extend(current)
            missing.extend(absent)
        if expected is not None:
            missing.extend({"case_id": case_id, "repeat": repeat, "fold": fold, "reason": reason}
                           for coordinates, reason in ((expected - observed, "planned_fold_missing"),
                                                       (observed - expected, "undeclared_fold_present"))
                           for repeat, fold in sorted(coordinates))

    tables = root / "tables"
    paths = {
        "fold_predictions": tables / "v5_fold_predictions.csv", "fold_models": tables / "v5_fold_models.csv",
        "config_parameters": tables / "v5_config_parameters.csv",
    }
    for path, rows, fields in ((paths["fold_predictions"], prediction_rows, _PREDICTION_FIELDS),
                               (paths["fold_models"], fold_rows, _FOLD_FIELDS),
                               (paths["config_parameters"], parameter_rows, _PARAMETER_FIELDS)):
        _write_csv(path, rows, fields)
    published = _publish_median_selections(root, fold_rows)
    manifest = {
        "schema_version": "ppg_frailty.v5_data_products.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "study_directory": str(root),
        "source_study_manifest": "study_manifest.json", "status": "complete" if not missing else "incomplete",
        "case_count": len(cases), "fold_count": len(fold_rows),
        "expected_fold_count": None if expected is None else len(cases) * len(expected),
        "prediction_artifact_count": len(prediction_rows),
        "prediction_row_count": sum(int(row["rows"]) for row in prediction_rows),
        "config_parameter_count": len(parameter_rows), "hash_prediction_files": hash_prediction_files,
        "prediction_storage": "per-fold Parquet files are authoritative; this manifest and CSV index do not "
                              "duplicate prediction rows",
        "model_parameter_semantics": "resolved hyperparameters, fitted provenance/state hashes, and reloadable "
                                     "per-fold learned-weight bundles at the model-ready input boundary",
        "published_models": published, "tables": {name: path.relative_to(root).as_posix()
                                                    for name, path in paths.items()}, "missing": missing,
    }
    atomic_json(root / "v5_data_manifest.json", manifest)
    return manifest


__all__ = ["build_study_data_index"]
