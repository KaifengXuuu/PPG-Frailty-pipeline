"""Strict recovery of a complete, interrupted full-experiment staging tree.

The recovery path never invents predictions or resumes model fitting.  It is
eligible only when every requested outer cell already has its complete typed
OOF and audit artifacts.  The original interrupted staging tree is left intact;
recovery hard-links its cell files into a new staging tree, compacts duplicated
SQI evidence there, rebuilds root aggregates, and only then atomically publishes
the recovered experiment.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


_CELL_POINTER_FIELDS = (
    "quality_diagnostics_artifact",
    "training_history_artifact",
    "sampling_diagnostics_artifact",
    "physical_recording_qc_artifact",
    "route_artifacts_artifact",
    "route_window_sqi_evidence_artifact",
    "preprocessing_cache_artifact",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from ppg_frailty.provenance import atomic_write_json

    atomic_write_json(path, dict(payload), root=path.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json_value(value: Any) -> str:
    """Hash one canonical strict-JSON value without building a giant string."""

    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _safe_mandatory_artifacts(cell_directory: Path) -> dict[str, Any]:
    manifest = _load_json(cell_directory / "run_manifest.json")
    if (
        manifest.get("schema_version") != "ppg_frailty.run_manifest.v2"
        or manifest.get("status") != "passed"
    ):
        raise ValueError(f"interrupted cell manifest is not passed: {cell_directory}")
    mandatory = manifest.get("mandatory_artifacts")
    if not isinstance(mandatory, list) or not mandatory:
        raise ValueError(f"interrupted cell lacks mandatory artifact roster: {cell_directory}")
    for raw_name in mandatory:
        name = str(raw_name)
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise ValueError(f"unsafe mandatory artifact name: {name}")
        target = cell_directory / relative
        if not target.is_file() or target.is_symlink() or target.stat().st_size <= 0:
            raise ValueError(f"missing interrupted cell artifact: {target}")
    return manifest


def _prefix_cell_pointers(summary: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    result = dict(summary)
    for field in _CELL_POINTER_FIELDS:
        value = result.get(field)
        if value is None:
            continue
        raw = Path(str(value))
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"unsafe cell artifact pointer: {field}={value}")
        result[field] = f"{prefix}/{raw.name}"
    return result


def _externalize_legacy_sqi_payload(
    cell_directory: Path,
    *,
    config_id: str,
    config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compact one pre-fix cell while retaining exact components in gzip JSONL."""

    from ppg_frailty import experiment as experiment_module

    route_path = cell_directory / "route_artifacts.json"
    route_payload = _load_json(route_path)
    if route_payload.get("schema_version") != "ppg_frailty.route_artifacts.v2":
        raise ValueError("route artifact schema drift during recovery")
    rows = route_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("route artifact rows missing during recovery")

    evidence_records: list[dict[str, Any]] = []
    route_record_ids: set[str] = set()
    original_route_sha256: dict[str, str] = {}
    route_identities: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("route artifact contains a non-object row")
        record_id = str(row.get("record_id", "")).strip()
        participant_id = str(row.get("participant_id", "")).strip()
        role = str(row.get("role", "")).strip()
        if not record_id or not participant_id or not role or record_id in route_record_ids:
            raise ValueError("route artifact record identity is missing or duplicated")
        route_record_ids.add(record_id)
        route_identities[record_id] = (participant_id, role)
        route = row.get("route_artifact")
        if not isinstance(route, dict):
            raise ValueError("route artifact row lacks its routing payload")
        original_route_sha256[record_id] = _sha256_json_value(route)
        cells = route.get("cells")
        if isinstance(cells, list):
            for timeline_cell in cells:
                if not isinstance(timeline_cell, Mapping):
                    raise ValueError("route timeline contains a non-object cell")
                if str(timeline_cell.get("config_sha256", "")) != config_hash:
                    raise ValueError("route timeline config hash drift during recovery")
        full_evidence = route.get("native_window_sqi_evidence")
        if isinstance(full_evidence, Mapping) and full_evidence:
            evidence_records.append(
                {
                    "record_id": record_id,
                    "participant_id": participant_id,
                    "role": role,
                    "windows": full_evidence,
                }
            )
            route["native_window_sqi_evidence"] = (
                experiment_module._compact_window_sqi_evidence(full_evidence)
            )

    repeat_index = int(route_payload.get("repeat_index"))
    fold_index = int(route_payload.get("fold_index"))
    evidence_summary = {
        "repeat_index": repeat_index,
        "fold_index": fold_index,
        "config_hash": config_hash,
        "route_window_sqi_evidence": evidence_records,
    }
    evidence_count, evidence_sha256 = (
        experiment_module._write_route_window_sqi_evidence(
            cell_directory,
            evidence_summary,
        )
    )
    _atomic_json(route_path, route_payload)
    del route_payload, rows, evidence_records
    gc.collect()

    quality_path = cell_directory / "quality_diagnostics.json"
    quality_payload = _load_json(quality_path)
    if (
        quality_payload.get("schema_version")
        != "ppg_frailty.quality_diagnostics.v2"
    ):
        raise ValueError("quality diagnostic schema drift during recovery")
    quality_rows = quality_payload.get("rows")
    if not isinstance(quality_rows, list):
        raise ValueError("quality diagnostic rows missing during recovery")
    quality_record_ids: set[str] = set()
    for row in quality_rows:
        if isinstance(row, dict):
            record_id = str(row.get("record_id", "")).strip()
            if not record_id or record_id in quality_record_ids:
                raise ValueError(
                    "quality diagnostic record identity is missing or duplicated"
                )
            quality_record_ids.add(record_id)
            if (
                str(row.get("participant_id", "")).strip(),
                str(row.get("role", "")).strip(),
            ) != route_identities.get(record_id):
                raise ValueError("quality/route participant or role identity drift")
            # This was a byte-for-byte semantic duplicate of the authoritative
            # route_artifacts row and carried no independent diagnostic value.
            duplicate_route = row.pop("route_artifact", None)
            if (
                not isinstance(duplicate_route, Mapping)
                or _sha256_json_value(duplicate_route)
                != original_route_sha256.get(record_id)
            ):
                raise ValueError(
                    "quality/route duplicated routing payload drift"
                )
    if quality_record_ids != route_record_ids:
        raise ValueError("quality and route record rosters differ during recovery")
    _atomic_json(quality_path, quality_payload)

    metrics_path = cell_directory / "metrics_per_fold_seed.json"
    metrics_payload = _load_json(metrics_path)
    cells = metrics_payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 1 or not isinstance(cells[0], dict):
        raise ValueError("cell metrics payload is not singular")
    summary = cells[0]
    if (
        int(summary.get("repeat_index", -1)) != repeat_index
        or int(summary.get("fold_index", -1)) != fold_index
        or str(summary.get("status")) != "passed"
    ):
        raise ValueError("route and metrics cell identity drift during recovery")
    for field in ("config_hash", "canonical_config_hash"):
        prior_hash = summary.get(field)
        if prior_hash not in (None, "") and str(prior_hash) != config_hash:
            raise ValueError(f"recovery cell {field} drift")
    prior_config_id = summary.get("config_id")
    if prior_config_id not in (None, "") and str(prior_config_id) != config_id:
        raise ValueError("recovery cell config_id drift")
    for count_field, observed_count in (
        ("route_artifacts_row_count", len(route_record_ids)),
        ("quality_diagnostic_row_count", len(quality_record_ids)),
    ):
        declared = summary.get(count_field)
        if isinstance(declared, bool) or int(declared) != observed_count:
            raise ValueError(f"recovery cell {count_field} drift")
    summary.update(
        {
            "config_id": config_id,
            "config_hash": config_hash,
            "canonical_config_hash": config_hash,
            "route_window_sqi_evidence_artifact": (
                "route_window_sqi_evidence.jsonl.gz"
            ),
            "route_window_sqi_evidence_row_count": evidence_count,
            "route_window_sqi_evidence_compression": "gzip_mtime0_jsonl",
            "route_window_sqi_evidence_report_consumed": False,
            "route_window_sqi_evidence_sha256": evidence_sha256,
        }
    )
    _atomic_json(metrics_path, metrics_payload)

    manifest_path = cell_directory / "run_manifest.json"
    manifest = _load_json(manifest_path)
    manifest_cell = manifest.get("cell")
    if not isinstance(manifest_cell, dict):
        raise ValueError("cell run manifest lacks compact cell summary")
    if (
        int(manifest_cell.get("repeat_index", -1)) != repeat_index
        or int(manifest_cell.get("fold_index", -1)) != fold_index
        or str(manifest_cell.get("status")) != "passed"
    ):
        raise ValueError("manifest and route cell identity drift during recovery")
    manifest_cell.update(summary)
    mandatory = manifest.get("mandatory_artifacts")
    if not isinstance(mandatory, list):
        raise ValueError("cell run manifest lacks mandatory artifacts")
    evidence_name = "route_window_sqi_evidence.jsonl.gz"
    if evidence_name not in mandatory:
        mandatory.append(evidence_name)
    _atomic_json(manifest_path, manifest)
    return dict(summary), quality_payload


def _read_cell_oof(
    cell_directory: Path,
    *,
    expected_config_hash: str,
    repeat_index: int,
    fold_index: int,
    expected_split_seed: int,
    expected_participant_ids: Iterable[str],
    expected_source_version: str,
    expected_code_commit: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    from ppg_frailty.training import (
        read_oof_parquet,
        validate_unique_subject_oof,
    )

    artifact_contracts = (
        ("oof_file_predictions.parquet", "file"),
        ("oof_subject_predictions.parquet", "participant"),
        ("oof_window_predictions.parquet", "window"),
        ("oof_role_predictions.parquet", "role"),
        ("oof_member_predictions.parquet", "participant"),
    )
    rows_by_level = tuple(
        read_oof_parquet(cell_directory / filename)
        for filename, _expected_level in artifact_contracts
    )
    if not rows_by_level[0] or not rows_by_level[1]:
        raise ValueError("recovery requires non-empty file and subject OOF")
    expected_participants = set(map(str, expected_participant_ids))
    for (filename, expected_level), rows in zip(
        artifact_contracts,
        rows_by_level,
    ):
        validate_unique_subject_oof(rows)
        for row in rows:
            if (
                str(row.config_hash) != expected_config_hash
                or int(row.repeat) != repeat_index
                or int(row.fold) != fold_index
                or int(row.split_seed) != expected_split_seed
                or str(row.level) != expected_level
                or str(row.participant_id) not in expected_participants
                or str(row.source_snapshot_hash) != expected_source_version
                or str(row.code_commit) != expected_code_commit
            ):
                raise ValueError(
                    f"recovery OOF identity drift in {filename}"
                )
    subject_rows = rows_by_level[1]
    subject_participants = [str(row.participant_id) for row in subject_rows]
    if (
        len(subject_participants) != len(expected_participants)
        or set(subject_participants) != expected_participants
    ):
        raise ValueError("recovery participant OOF roster is not exact-once")
    member_rows = rows_by_level[4]
    if member_rows and {str(row.participant_id) for row in member_rows} != (
        expected_participants
    ):
        raise ValueError("recovery member OOF participant roster drift")
    return rows_by_level


def _validate_compacted_cell_artifacts(
    cell_directory: Path,
    summary: Mapping[str, Any],
    *,
    repeat_index: int,
    fold_index: int,
) -> None:
    """Validate non-predictive artifacts and all externalized pointers."""

    row_artifacts = (
        (
            "training_history.json",
            "ppg_frailty.training_history.v2",
            "training_history_row_count",
        ),
        (
            "physical_recording_qc.json",
            "ppg_frailty.physical_recording_qc.v2",
            "physical_recording_qc_row_count",
        ),
    )
    for filename, schema_version, count_field in row_artifacts:
        payload = _load_json(cell_directory / filename)
        rows = payload.get("rows")
        if (
            payload.get("schema_version") != schema_version
            or int(payload.get("repeat_index", -1)) != repeat_index
            or int(payload.get("fold_index", -1)) != fold_index
            or not isinstance(rows, list)
            or isinstance(summary.get(count_field), bool)
            or int(summary.get(count_field, -1)) != len(rows)
        ):
            raise ValueError(f"recovery {filename} schema/identity/count drift")

    confusion = _load_json(cell_directory / "confusion_matrices.json")
    confusion_cells = confusion.get("cells")
    if (
        confusion.get("schema_version")
        != "ppg_frailty.confusion_matrices.v2"
        or confusion.get("pipeline_generation") != "final_pipeline_v2"
        or not isinstance(confusion_cells, list)
        or len(confusion_cells) != 1
        or int(confusion_cells[0].get("repeat_index", -1)) != repeat_index
        or int(confusion_cells[0].get("fold_index", -1)) != fold_index
        or confusion_cells[0].get("class_order") != summary.get("class_order")
        or confusion_cells[0].get("confusion_matrix")
        != summary.get("metrics", {}).get("confusion_matrix")
    ):
        raise ValueError("recovery confusion artifact identity drift")

    cache = _load_json(cell_directory / "preprocessing_cache.json")
    if (
        cache.get("schema_version")
        != "ppg_frailty.preprocessing_cache_audit.v1"
        or cache.get("affects_predictions") is not False
    ):
        raise ValueError("recovery preprocessing cache audit contract drift")

    pointer_contract = {
        "quality_diagnostics_artifact": "quality_diagnostics.json",
        "training_history_artifact": "training_history.json",
        "physical_recording_qc_artifact": "physical_recording_qc.json",
        "route_artifacts_artifact": "route_artifacts.json",
        "route_window_sqi_evidence_artifact": (
            "route_window_sqi_evidence.jsonl.gz"
        ),
        "preprocessing_cache_artifact": "preprocessing_cache.json",
    }
    for field, expected_name in pointer_contract.items():
        if summary.get(field) != expected_name:
            raise ValueError(f"recovery externalized pointer drift: {field}")

    metrics = _load_json(cell_directory / "metrics_per_fold_seed.json")
    metric_cells = metrics.get("cells")
    if (
        metrics.get("schema_version") != "ppg_frailty.metrics_per_fold_seed.v2"
        or metrics.get("pipeline_generation") != "final_pipeline_v2"
        or not isinstance(metric_cells, list)
        or len(metric_cells) != 1
        or metric_cells[0] != dict(summary)
    ):
        raise ValueError("recovery compact metrics summary drift")

    manifest = _load_json(cell_directory / "run_manifest.json")
    mandatory = manifest.get("mandatory_artifacts")
    if (
        manifest.get("schema_version") != "ppg_frailty.run_manifest.v2"
        or manifest.get("pipeline_generation") != "final_pipeline_v2"
        or manifest.get("status") != "passed"
        or manifest.get("scientific_scope")
        != "frozen_5x5_scientific_benchmark"
        or manifest.get("cell") != dict(summary)
        or not isinstance(mandatory, list)
        or any(not (cell_directory / str(name)).is_file() for name in mandatory)
    ):
        raise ValueError("recovery compact cell manifest drift")


def recover_completed_full_experiment_staging(
    config_path: str | Path,
    *,
    interrupted_staging: str | Path,
    output_dir: str | Path,
    repeats: Iterable[int],
    folds: Iterable[int],
    measure_operational_costs: bool,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any | None:
    """Publish a complete 5x5 staging tree without rerunning any training cell."""

    repeat_values = tuple(int(value) for value in repeats)
    fold_values = tuple(int(value) for value in folds)
    expected_keys = {
        (repeat_index, fold_index)
        for repeat_index in repeat_values
        for fold_index in fold_values
    }
    if expected_keys != {
        (repeat_index, fold_index)
        for repeat_index in range(5)
        for fold_index in range(5)
    }:
        return None

    source = Path(interrupted_staging).resolve()
    target = Path(output_dir).resolve()
    if (
        not source.is_dir()
        or not source.name.startswith(".experiment.staging.")
        or target.exists()
    ):
        return None
    observed_cells = {
        (int(path.name[7:9]), int(path.name[15:17])): path
        for path in source.iterdir()
        if path.is_dir()
        and len(path.name) == len("repeat_00_fold_00")
        and path.name.startswith("repeat_")
        and "_fold_" in path.name
    }
    if set(observed_cells) != expected_keys:
        return None
    for path in observed_cells.values():
        _safe_mandatory_artifacts(path)

    from ppg_frailty import experiment as experiment_module
    from ppg_frailty.pipeline import PipelinePaths, preflight_pipeline

    paths = PipelinePaths.discover()
    report, config, _, registry = preflight_pipeline(
        config_path,
        mode="full",
        paths=paths,
    )
    recovery_staging = target.with_name(
        f".{target.name}.recovery-staging.{time.time_ns()}"
    )
    root_oof_names = (
        "oof_file_predictions.parquet",
        "oof_subject_predictions.parquet",
        "oof_window_predictions.parquet",
        "oof_role_predictions.parquet",
        "oof_member_predictions.parquet",
    )
    source_root_oof_present = {
        filename: (source / filename).is_file() for filename in root_oof_names
    }
    if any(source_root_oof_present.values()) and not all(
        source_root_oof_present.values()
    ):
        raise ValueError("interrupted staging contains a partial root OOF set")
    source_root_oof_sha256 = (
        {
            filename: _sha256_file(source / filename)
            for filename in root_oof_names
        }
        if all(source_root_oof_present.values())
        else {}
    )
    recovery_staging.mkdir(parents=True, exist_ok=False)
    cells: list[Any] = []
    indexed_summaries: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    training_source_versions: set[str] = set()
    training_code_versions: set[str] = set()
    started = time.perf_counter()
    try:
        ordered_keys = sorted(expected_keys)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "run_start",
                    "message": "recovering 25 completed cells; no model refit",
                    "total_cells": len(ordered_keys),
                    "output_dir": str(target),
                }
            )
        for cell_number, (repeat_index, fold_index) in enumerate(
            ordered_keys,
            start=1,
        ):
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "cell_start",
                        "message": "validating and compacting interrupted cell",
                        "current_cell": cell_number,
                        "total_cells": len(ordered_keys),
                        "repeat_index": repeat_index,
                        "fold_index": fold_index,
                    }
                )
            name = f"repeat_{repeat_index:02d}_fold_{fold_index:02d}"
            recovered_cell_directory = recovery_staging / name
            source_cell_directory = observed_cells[(repeat_index, fold_index)]
            transformed_names = (
                "route_artifacts.json",
                "quality_diagnostics.json",
                "metrics_per_fold_seed.json",
                "run_manifest.json",
            )
            source_sha256 = {
                filename: _sha256_file(source_cell_directory / filename)
                for filename in transformed_names
            }
            source_prediction_sha256 = {
                filename: _sha256_file(source_cell_directory / filename)
                for filename in root_oof_names
            }
            shutil.copytree(
                source_cell_directory,
                recovered_cell_directory,
                copy_function=os.link,
            )
            summary, quality_payload = _externalize_legacy_sqi_payload(
                recovered_cell_directory,
                config_id=str(config.config_id),
                config_hash=str(config.sha256),
            )
            if (
                int(summary.get("repeat_index", -1)) != repeat_index
                or int(summary.get("fold_index", -1)) != fold_index
                or str(summary.get("status")) != "passed"
            ):
                raise ValueError("recovery cell summary identity drift")
            source_version = str(summary.get("source_version", ""))
            code_version = str(summary.get("code_commit", ""))
            if len(source_version) != 64 or not code_version:
                raise ValueError("recovery cell training source provenance missing")
            training_source_versions.add(source_version)
            training_code_versions.add(code_version)
            split = registry.get_split(repeat_index, fold_index)
            file_rows, subject_rows, window_rows, role_rows, member_rows = (
                _read_cell_oof(
                    recovered_cell_directory,
                    expected_config_hash=str(config.sha256),
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    expected_split_seed=int(split["split_seed"]),
                    expected_participant_ids=split["oof_participant_ids"],
                    expected_source_version=source_version,
                    expected_code_commit=code_version,
                )
            )
            _validate_compacted_cell_artifacts(
                recovered_cell_directory,
                summary,
                repeat_index=repeat_index,
                fold_index=fold_index,
            )
            quality_rows = quality_payload.get("rows", [])
            root_summary = _prefix_cell_pointers(summary, name)
            root_summary["quality_diagnostics"] = quality_rows
            cells.append(
                experiment_module._CellResult(
                    summary=root_summary,
                    file_rows=file_rows,
                    subject_rows=subject_rows,
                    window_rows=window_rows,
                    role_rows=role_rows,
                    member_rows=member_rows,
                )
            )
            indexed = dict(root_summary)
            indexed.pop("quality_diagnostics", None)
            indexed_summaries.append(indexed)
            recovered_sha256 = {
                filename: _sha256_file(recovered_cell_directory / filename)
                for filename in transformed_names
            }
            recovered_sha256["route_window_sqi_evidence.jsonl.gz"] = str(
                summary["route_window_sqi_evidence_sha256"]
            )
            recovered_prediction_sha256 = {
                filename: _sha256_file(recovered_cell_directory / filename)
                for filename in root_oof_names
            }
            if recovered_prediction_sha256 != source_prediction_sha256:
                raise ValueError("recovery changed a cell prediction artifact")
            lineage_rows.append(
                {
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "source_cell": name,
                    "recovered_cell": name,
                    "source_artifact_sha256": source_sha256,
                    "recovered_artifact_sha256": recovered_sha256,
                    "source_prediction_sha256": source_prediction_sha256,
                    "recovered_prediction_sha256": (
                        recovered_prediction_sha256
                    ),
                    "prediction_artifacts_hard_linked_unchanged": True,
                    "model_refit": False,
                }
            )
            del quality_payload
            gc.collect()
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "cell_complete",
                        "message": "recovered completed cell",
                        "current_cell": cell_number,
                        "total_cells": len(ordered_keys),
                        "repeat_index": repeat_index,
                        "fold_index": fold_index,
                        "status": "passed",
                    }
                )

        scope_values = {str(cell.summary.get("scientific_scope")) for cell in cells}
        if scope_values != {"frozen_5x5_scientific_benchmark"}:
            raise ValueError("recovery scientific scope is not the frozen 5x5 grid")
        if len(training_source_versions) != 1 or len(training_code_versions) != 1:
            raise ValueError("recovery cells mix training source provenance")
        training_source_version = next(iter(training_source_versions))
        training_code_version = next(iter(training_code_versions))
        recovery_source_version = experiment_module._source_version()
        recovery_code_version = experiment_module._code_version()
        result = experiment_module.ExperimentResult(
            status="passed",
            scientific_scope="frozen_5x5_scientific_benchmark",
            config_id=str(config.config_id),
            config_hash=str(config.sha256),
            repeat_indices=repeat_values,
            fold_indices=fold_values,
            output_dir=str(target),
            cell_results=tuple(indexed_summaries),
            metrics={
                "requested_cell_count": 25,
                "passed_cell_count": 25,
                "failed_cell_count": 0,
                "elapsed_seconds": sum(
                    float(cell.summary.get("elapsed_seconds", 0.0))
                    for cell in cells
                ),
                "recovery_finalize_seconds": time.perf_counter() - started,
            },
            provenance={
                "preflight_status": report.status,
                "manifest_hash": report.manifest_hash,
                "fold_hash": report.fold_hash,
                "frozen_outer_split": True,
                "data_shortening": False,
                "record_cap": None,
                "epoch_override": None,
                "operational_measurement_requested": bool(
                    measure_operational_costs
                ),
                "code_version": training_code_version,
                "source_version": training_source_version,
                "recovered_from_complete_interrupted_staging": True,
                "interrupted_staging_preserved": str(source),
                "recovery_transform": (
                    "externalize_duplicate_full_sqi_evidence_v1"
                ),
                "recovery_model_refit": False,
                "recovery_predictions_changed": False,
                "recovery_code_version": recovery_code_version,
                "recovery_source_version": recovery_source_version,
                "recovery_manifest": "recovery_manifest.json",
                "trusted_run_replay_required_before_publish": True,
            },
        )
        experiment_module._write_full_root_artifacts(
            recovery_staging,
            cells,
            result,
        )
        recovered_root_oof_sha256 = {
            filename: _sha256_file(recovery_staging / filename)
            for filename in root_oof_names
        }
        if (
            source_root_oof_sha256
            and recovered_root_oof_sha256 != source_root_oof_sha256
        ):
            raise ValueError("recovered root OOF bytes differ from interrupted staging")
        trusted = experiment_module._read_trusted_comparison_run(
            str(config.config_id),
            recovery_staging,
            n_bootstrap_resamples=None,
            bootstrap_seed=None,
        )
        if (
            str(trusted.get("config_hash")) != str(config.sha256)
            or str(trusted.get("machine_id"))
            != str(cells[0].summary.get("model_machine_id"))
        ):
            raise ValueError("recovery trusted run replay identity drift")
        del trusted
        gc.collect()
        experiment_module._strict_json(
            recovery_staging / "recovery_manifest.json",
            {
                "schema_version": "ppg_frailty.recovery_manifest.v1",
                "pipeline_generation": "final_pipeline_v2",
                "status": "passed_before_atomic_publish",
                "purpose": "finalize_complete_interrupted_5x5_without_refit",
                "source_staging": str(source),
                "source_staging_preserved": True,
                "target": str(target),
                "config_id": str(config.config_id),
                "config_hash": str(config.sha256),
                "training_code_version": training_code_version,
                "training_source_version": training_source_version,
                "recovery_code_version": recovery_code_version,
                "recovery_source_version": recovery_source_version,
                "transformation": (
                    "full_component_sqi_to_gzip_jsonl_once;"
                    "compact_report_projection;remove_quality_route_duplicate"
                ),
                "model_refit": False,
                "prediction_artifacts_changed": False,
                "trusted_run_replay": "passed",
                "source_root_oof_sha256": source_root_oof_sha256,
                "recovered_root_oof_sha256": recovered_root_oof_sha256,
                "root_oof_byte_identity": (
                    "identical"
                    if source_root_oof_sha256
                    else "source_root_not_yet_materialized"
                ),
                "cells": lineage_rows,
            },
        )
        experiment_module._strict_json(
            recovery_staging / "experiment_result.json",
            result.to_dict(),
        )
        experiment_module._commit_staging(recovery_staging, target)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "run_complete",
                    "message": "interrupted experiment recovered without refitting",
                    "status": "passed",
                    "total_cells": len(ordered_keys),
                    "passed_cells": len(ordered_keys),
                    "failed_cells": 0,
                    "output_dir": str(target),
                }
            )
        return result
    except Exception:
        if recovery_staging.exists():
            shutil.rmtree(recovery_staging)
        raise


def validate_published_recovered_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    repeats: Iterable[int],
    folds: Iterable[int],
) -> dict[str, Any] | None:
    """Validate an atomically published recovery whose case index was interrupted."""

    if (
        tuple(map(int, repeats)) != tuple(range(5))
        or tuple(map(int, folds)) != tuple(range(5))
    ):
        return None
    root = Path(output_dir).resolve()
    required = (
        "experiment_result.json",
        "run_manifest.json",
        "recovery_manifest.json",
        "config_metrics_v2.json",
        "oof_file_predictions.parquet",
        "oof_subject_predictions.parquet",
        "oof_window_predictions.parquet",
        "oof_role_predictions.parquet",
        "oof_member_predictions.parquet",
    )
    if not root.is_dir() or any(
        not (root / filename).is_file() for filename in required
    ):
        return None

    from ppg_frailty import experiment as experiment_module
    from ppg_frailty.pipeline import PipelinePaths, preflight_pipeline

    _report, config, _rows, _registry = preflight_pipeline(
        config_path,
        mode="full",
        paths=PipelinePaths.discover(),
    )
    result = _load_json(root / "experiment_result.json")
    manifest = _load_json(root / "run_manifest.json")
    recovery = _load_json(root / "recovery_manifest.json")
    for payload in (result, manifest):
        if (
            payload.get("status") != "passed"
            or payload.get("scientific_scope")
            != "frozen_5x5_scientific_benchmark"
            or payload.get("config_id") != config.config_id
            or payload.get("config_hash") != config.sha256
            or tuple(payload.get("repeat_indices", ())) != tuple(range(5))
            or tuple(payload.get("fold_indices", ())) != tuple(range(5))
        ):
            raise ValueError("published recovery result identity drift")
    mandatory = manifest.get("mandatory_artifacts")
    if (
        manifest.get("schema_version") != "ppg_frailty.run_manifest.v2"
        or not isinstance(mandatory, list)
        or "recovery_manifest.json" not in mandatory
        or any(not (root / str(name)).is_file() for name in mandatory)
    ):
        raise ValueError("published recovery mandatory artifact drift")
    if (
        recovery.get("schema_version") != "ppg_frailty.recovery_manifest.v1"
        or recovery.get("status") != "passed_before_atomic_publish"
        or recovery.get("config_id") != config.config_id
        or recovery.get("config_hash") != config.sha256
        or recovery.get("model_refit") is not False
        or recovery.get("prediction_artifacts_changed") is not False
        or recovery.get("trusted_run_replay") != "passed"
    ):
        raise ValueError("published recovery chain-of-custody drift")
    recovered_oof = recovery.get("recovered_root_oof_sha256")
    if not isinstance(recovered_oof, Mapping) or any(
        recovered_oof.get(filename) != _sha256_file(root / filename)
        for filename in (
            "oof_file_predictions.parquet",
            "oof_subject_predictions.parquet",
            "oof_window_predictions.parquet",
            "oof_role_predictions.parquet",
            "oof_member_predictions.parquet",
        )
    ):
        raise ValueError("published recovery root OOF hash drift")
    trusted = experiment_module._read_trusted_comparison_run(
        str(config.config_id),
        root,
        n_bootstrap_resamples=None,
        bootstrap_seed=None,
    )
    if str(trusted.get("config_hash")) != str(config.sha256):
        raise ValueError("published recovery trusted replay config drift")
    return result


__all__ = [
    "recover_completed_full_experiment_staging",
    "validate_published_recovered_experiment",
]
