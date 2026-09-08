"""Strictly recover a complete interrupted experiment without fitting models.

Cells are authenticated, hard-linked into new staging, compacted, aggregated,
and atomically published; the interrupted tree and prediction bytes stay intact.
"""
from __future__ import annotations
import gc, gzip, hashlib, json, os, shutil, time
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
_EVIDENCE_FIELDS = (
    "route_window_sqi_evidence_artifact",
    "route_window_sqi_evidence_row_count",
    "route_window_sqi_evidence_compression",
    "route_window_sqi_evidence_report_consumed",
    "route_window_sqi_evidence_sha256",
)
_ROOT_OOF = tuple((f"oof_{level}_predictions.parquet" for level in ("file", "subject", "window", "role", "member")))
_TRANSFORMED_CELL_FILES = (
    "route_artifacts.json",
    "quality_diagnostics.json",
    "metrics_per_fold_seed.json",
    "run_manifest.json",
)


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _matches(payload: Mapping[str, Any], **expected: Any) -> bool:
    return all((payload.get(field) == value for field, value in expected.items()))


def _indices_match(payload: Mapping[str, Any], repeat: int, fold: int) -> bool:
    return int(payload.get("repeat_index", -1)) == repeat and int(payload.get("fold_index", -1)) == fold


def _hash_files(directory: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: _sha256_file(directory / name) for name in names}


def _row_identity(row: Mapping[str, Any], *, window: bool = False, strip: bool = True) -> tuple[str, ...]:
    fields = (("record_id", "participant_id", "role", "routing_window_id") if window else
              ("record_id", "participant_id", "role"))
    values = tuple((str(row.get(field, "")) for field in fields))
    return tuple((value.strip() for value in values)) if strip else values


def _declared_count(summary: Mapping[str, Any], field: str, observed: int, message: str) -> None:
    declared = summary.get(field)
    _require(not isinstance(declared, bool) and int(declared if declared is not None else -1) == observed, message)


def _emit(callback: Callable[[Mapping[str, Any]], None] | None, payload: Mapping[str, Any]) -> None:
    if callback is not None:
        callback(payload)


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
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {token}")),
        )
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
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
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _safe_relative_artifact(directory: Path, raw_name: Any) -> Path:
    _require(isinstance(raw_name, str) and bool(raw_name), f"unsafe mandatory artifact name: {raw_name!r}")
    relative = Path(raw_name)
    _require(
        not (relative.is_absolute() or relative == Path(".") or ".." in relative.parts or ("\\" in raw_name) or
             (relative.as_posix() != raw_name)),
        f"unsafe mandatory artifact name: {raw_name}",
    )
    root, target = directory.resolve(strict=True), directory
    for part in relative.parts:
        target = target / part
        _require(not target.is_symlink(), f"unsafe mandatory artifact symlink: {raw_name}")
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"unsafe or missing mandatory artifact: {raw_name}") from error
    _require(resolved.is_file() and resolved.stat().st_size > 0, f"missing interrupted cell artifact: {resolved}")
    return resolved


def _validate_fold_checkpoint_artifacts(cell_directory: Path, cell: Mapping[str, Any], mandatory: set[str]) -> None:
    checkpoint_name = "model_checkpoint/manifest.json"
    declaration = cell.get("learned_model_checkpoint")
    if declaration is None and checkpoint_name not in mandatory:
        return
    _require(
        isinstance(declaration, Mapping) and checkpoint_name in mandatory,
        "fold checkpoint declaration/mandatory roster is incomplete",
    )
    _require(declaration.get("manifest_path") == checkpoint_name, "fold checkpoint manifest path drift")
    manifest_path = _safe_relative_artifact(cell_directory, checkpoint_name)
    from ppg_frailty.training.bundle import verify_bundle

    declared_manifest = _load_json(manifest_path)
    hashes = declared_manifest.get("file_hashes")
    expected_files = {"manifest.json", *hashes} if isinstance(hashes, Mapping) else set()
    actual_files = {path.name for path in manifest_path.parent.iterdir()}
    _require(not actual_files - expected_files, "fold checkpoint has missing or extra payloads")
    _require(not expected_files - actual_files, "fold checkpoint payload integrity drift")
    try:
        manifest = verify_bundle(manifest_path.parent, load_model=False)
    except (OSError, ValueError) as error:
        raise ValueError("fold checkpoint payload integrity drift") from error
    assert isinstance(manifest, dict)
    state_name = str(manifest["state_file"])
    expected = {
        "manifest_sha256": _sha256_file(manifest_path),
        "state_file": f"model_checkpoint/{state_name}",
        "state_sha256": manifest["file_hashes"][state_name],
        "golden_parity_atol": manifest["golden_parity_atol"],
    }
    _require(
        not any((declaration.get(name) != value for name, value in expected.items())),
        "fold checkpoint declaration disagrees with verified bundle",
    )


def _safe_mandatory_artifacts(cell_directory: Path) -> dict[str, Any]:
    manifest = _load_json(cell_directory / "run_manifest.json")
    _require(
        manifest.get("schema_version") == "ppg_frailty.run_manifest.v2" and manifest.get("status") == "passed",
        f"interrupted cell manifest is not passed: {cell_directory}",
    )
    mandatory = manifest.get("mandatory_artifacts")
    _require(
        isinstance(mandatory, list) and bool(mandatory),
        f"interrupted cell lacks mandatory artifact roster: {cell_directory}",
    )
    seen: set[str] = set()
    for raw_name in mandatory:
        _require(isinstance(raw_name, str), f"unsafe mandatory artifact name: {raw_name!r}")
        _require(raw_name not in seen, f"duplicate mandatory artifact name: {raw_name}")
        _safe_relative_artifact(cell_directory, raw_name)
        seen.add(raw_name)
    cell = manifest.get("cell")
    _validate_fold_checkpoint_artifacts(cell_directory, cell if isinstance(cell, Mapping) else {}, seen)
    return manifest


def _load_json_line(line: str, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {token}")),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid externalized SQI JSONL at {path}:{line_number}") from error
    _require(isinstance(value, dict), f"externalized SQI JSONL row is not an object: {path}:{line_number}")
    return value


def _validate_externalized_sqi_evidence(
    cell_directory: Path,
    summary: Mapping[str, Any],
    *,
    config_hash: str,
    repeat_index: int,
    fold_index: int,
    expected_rows: set[tuple[str, str, str, str]],
) -> tuple[int, str]:
    artifact_name = "route_window_sqi_evidence.jsonl.gz"
    contract_ok = (_matches(
        summary,
        route_window_sqi_evidence_artifact=artifact_name,
        route_window_sqi_evidence_compression="gzip_mtime0_jsonl",
    ) and summary.get("route_window_sqi_evidence_report_consumed") is False)
    _require(contract_ok, "externalized SQI evidence contract drift")
    expected_count = summary.get("route_window_sqi_evidence_row_count")
    expected_sha256 = summary.get("route_window_sqi_evidence_sha256")
    valid_metadata = (not isinstance(expected_count, bool) and isinstance(expected_count, int) and expected_count >= 0
                      and isinstance(expected_sha256, str) and len(expected_sha256) == 64
                      and not any(character not in "0123456789abcdef" for character in expected_sha256))
    _require(valid_metadata, "externalized SQI evidence metadata drift")
    path = _safe_relative_artifact(cell_directory, artifact_name)
    actual_sha256 = _sha256_file(path)
    _require(actual_sha256 == expected_sha256, "externalized SQI evidence SHA-256 drift")
    observed_rows: set[tuple[str, str, str, str]] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            header_line = stream.readline()
            _require(bool(header_line), "externalized SQI evidence lacks header")
            header = _load_json_line(header_line, path=path, line_number=1)
            header_ok = (_matches(
                header, schema_version="ppg_frailty.route_window_sqi_evidence.v1", record_type="header")
                         and _indices_match(header, repeat_index, fold_index)
                         and str(header.get("config_hash", "")) == config_hash
                         and header.get("payload_contract") == "one_full_component_evidence_object_per_routing_window"
                         and header.get("report_consumed") is False)
            _require(header_ok, "externalized SQI evidence header drift")
            for line_number, line in enumerate(stream, start=2):
                _require(bool(line.strip()), f"blank externalized SQI JSONL row: {path}:{line_number}")
                row = _load_json_line(line, path=path, line_number=line_number)
                identity = _row_identity(row, window=True, strip=False)
                row_ok = (_matches(
                    row, schema_version="ppg_frailty.route_window_sqi_evidence.v1", record_type="window_evidence")
                          and all(identity) and isinstance(row.get("evidence"), Mapping)
                          and identity not in observed_rows)
                _require(row_ok, "externalized SQI evidence row drift")
                observed_rows.add(identity)
    except (OSError, EOFError) as error:
        raise ValueError("invalid externalized SQI gzip stream") from error
    _require(
        len(observed_rows) == expected_count and observed_rows == expected_rows,
        "externalized SQI evidence roster/count drift",
    )
    return (expected_count, actual_sha256)


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


def _prepare_route_payload(
    cell_directory: Path,
    *,
    config_hash: str,
    repeat_index: int,
    fold_index: int,
    externalized: bool,
) -> tuple[Any, ...]:
    from ppg_frailty import experiment as experiment_module

    payload = _load_json(cell_directory / "route_artifacts.json")
    _require(
        payload.get("schema_version") == "ppg_frailty.route_artifacts.v2"
        and _indices_match(payload, repeat_index, fold_index),
        "route artifact schema drift during recovery",
    )
    rows = payload.get("rows")
    _require(isinstance(rows, list), "route artifact rows missing during recovery")
    record_ids, identities, route_hashes = set(), {}, {}
    evidence_records, projected = [], {}
    for row in rows:
        _require(isinstance(row, dict), "route artifact contains a non-object row")
        record_id, participant_id, role = _row_identity(row)
        _require(
            bool(record_id and participant_id and role) and record_id not in record_ids,
            "route artifact record identity is missing or duplicated",
        )
        record_ids.add(record_id)
        identities[record_id] = (participant_id, role)
        route = row.get("route_artifact")
        _require(isinstance(route, dict), "route artifact row lacks its routing payload")
        cells = route.get("cells")
        if isinstance(cells, list):
            for timeline_cell in cells:
                _require(isinstance(timeline_cell, Mapping), "route timeline contains a non-object cell")
                _require(
                    str(timeline_cell.get("config_sha256", "")) == config_hash,
                    "route timeline config hash drift during recovery",
                )
        full_evidence = route.get("native_window_sqi_evidence")
        _require(full_evidence is None or isinstance(full_evidence, Mapping),
                 "route window SQI projection is not an object")
        if externalized:
            for window_id, compact in (full_evidence or {}).items():
                _require(isinstance(compact, Mapping), "route window SQI projection row is not an object")
                identity = (record_id, participant_id, role, str(window_id))
                _require(identity not in projected, "duplicate route window SQI projection identity")
                projected[identity] = compact
        else:
            route_hashes[record_id] = _sha256_json_value(route)
            if isinstance(full_evidence, Mapping) and full_evidence:
                evidence_records.append({
                    "record_id": record_id,
                    "participant_id": participant_id,
                    "role": role,
                    "windows": full_evidence
                })
                route["native_window_sqi_evidence"] = experiment_module._compact_window_sqi_evidence(full_evidence)
    return (payload, record_ids, identities, evidence_records, projected, route_hashes)


def _prepare_quality_payload(
    cell_directory: Path,
    *,
    summary: Mapping[str, Any],
    route_ids: set[str],
    route_identities: Mapping[str, tuple[str, str]],
    route_hashes: Mapping[str, str],
    externalized: bool,
) -> dict[str, Any]:
    payload = _load_json(cell_directory / "quality_diagnostics.json")
    _require(
        payload.get("schema_version") == "ppg_frailty.quality_diagnostics.v2",
        "quality diagnostic schema drift during recovery",
    )
    rows = payload.get("rows")
    _require(isinstance(rows, list), "quality diagnostic rows missing during recovery")
    quality_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        record_id, participant_id, role = _row_identity(row)
        _require(
            bool(record_id) and record_id not in quality_ids,
            "quality diagnostic record identity is missing or duplicated",
        )
        quality_ids.add(record_id)
        _require(
            (participant_id, role) == route_identities.get(record_id),
            "quality/route participant or role identity drift",
        )
        if externalized:
            _require("route_artifact" not in row, "externalized quality diagnostics retain route duplicate")
        else:
            duplicate = row.pop("route_artifact", None)
            _require(
                isinstance(duplicate, Mapping) and _sha256_json_value(duplicate) == route_hashes.get(record_id),
                "quality/route duplicated routing payload drift",
            )
    if not externalized:
        _require(quality_ids == route_ids, "quality and route record rosters differ during recovery")
    _declared_count(summary, "quality_diagnostic_row_count", len(quality_ids),
                    "recovery cell quality_diagnostic_row_count drift")
    return payload


def _externalize_legacy_sqi_payload(
    cell_directory: Path,
    *,
    config_id: str,
    config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Compact legacy SQI once, or authenticate and reuse an existing V5 archive."""
    from ppg_frailty import experiment as experiment_module

    metrics_path = cell_directory / "metrics_per_fold_seed.json"
    metrics_payload = _load_json(metrics_path)
    cells = metrics_payload.get("cells")
    _require(
        isinstance(cells, list) and len(cells) == 1 and isinstance(cells[0], dict),
        "cell metrics payload is not singular",
    )
    summary = cells[0]
    repeat_index, fold_index = int(summary.get("repeat_index", -1)), int(summary.get("fold_index", -1))
    _require(
        repeat_index >= 0 and fold_index >= 0 and (str(summary.get("status")) == "passed"),
        "cell metrics identity/status drift during recovery",
    )
    for field in ("config_hash", "canonical_config_hash"):
        prior_hash = summary.get(field)
        _require(prior_hash in (None, "") or str(prior_hash) == config_hash, f"recovery cell {field} drift")
    prior_config_id = summary.get("config_id")
    _require(prior_config_id in (None, "") or str(prior_config_id) == config_id, "recovery cell config_id drift")
    evidence_name = "route_window_sqi_evidence.jsonl.gz"
    evidence_path = cell_directory / evidence_name
    field_presence = tuple((field in summary for field in _EVIDENCE_FIELDS))
    archive_present = evidence_path.exists() or evidence_path.is_symlink()
    _require(not (any(field_presence) and (not all(field_presence))), "partial externalized SQI evidence metadata")
    externalized = archive_present and all(field_presence)
    _require(archive_present == all(field_presence), "externalized SQI evidence file/metadata conflict")
    _require(
        not (externalized and "route_window_sqi_evidence" in summary),
        "externalized SQI evidence is also present inline",
    )
    route_payload, route_ids, route_identities, evidence_records, projected, route_hashes = _prepare_route_payload(
        cell_directory,
        config_hash=config_hash,
        repeat_index=repeat_index,
        fold_index=fold_index,
        externalized=externalized,
    )
    _declared_count(summary, "route_artifacts_row_count", len(route_ids),
                    "recovery cell route_artifacts_row_count drift")
    quality_payload = _prepare_quality_payload(
        cell_directory,
        summary=summary,
        route_ids=route_ids,
        route_identities=route_identities,
        route_hashes=route_hashes,
        externalized=externalized,
    )
    manifest_path = cell_directory / "run_manifest.json"
    manifest = _load_json(manifest_path)
    manifest_cell = manifest.get("cell")
    _require(isinstance(manifest_cell, dict), "cell run manifest lacks compact cell summary")
    _require(
        _indices_match(manifest_cell, repeat_index, fold_index) and str(manifest_cell.get("status")) == "passed",
        "manifest and route cell identity drift during recovery",
    )
    mandatory = manifest.get("mandatory_artifacts")
    _require(isinstance(mandatory, list), "cell run manifest lacks mandatory artifacts")
    if externalized:
        evidence_count, evidence_sha256 = _validate_externalized_sqi_evidence(
            cell_directory,
            summary,
            config_hash=config_hash,
            repeat_index=repeat_index,
            fold_index=fold_index,
            expected_rows=set(projected),
        )
        _require(evidence_count == len(projected), "externalized route/SQI row-count drift")
        with gzip.open(evidence_path, "rt", encoding="utf-8", newline="") as stream:
            next(stream)
            for line_number, line in enumerate(stream, start=2):
                row = _load_json_line(line, path=evidence_path, line_number=line_number)
                identity = _row_identity(row, window=True, strip=False)
                compact = experiment_module._compact_window_sqi_evidence({
                    identity[-1]: row["evidence"]
                }).get(identity[-1])
                _require(compact == projected.get(identity), "externalized route/SQI projection drift")
        _require(manifest_cell == summary, "externalized manifest/metrics cell summary drift")
        _require(evidence_name in mandatory, "externalized SQI evidence is not mandatory")
        _require(_sha256_file(evidence_path) == evidence_sha256, "externalized SQI evidence changed during validation")
        return (dict(summary), quality_payload, "validate_and_reuse_externalized_sqi_evidence_v1")
    evidence_summary = {
        "repeat_index": repeat_index,
        "fold_index": fold_index,
        "config_hash": config_hash,
        "route_window_sqi_evidence": evidence_records,
    }
    evidence_count, evidence_sha256 = experiment_module._write_route_window_sqi_evidence(
        cell_directory, evidence_summary)
    _atomic_json(cell_directory / "route_artifacts.json", route_payload)
    _atomic_json(cell_directory / "quality_diagnostics.json", quality_payload)
    summary.update(config_id=config_id, config_hash=config_hash, canonical_config_hash=config_hash)
    summary.update(zip(_EVIDENCE_FIELDS, (evidence_name, evidence_count, "gzip_mtime0_jsonl", False, evidence_sha256)))
    _atomic_json(metrics_path, metrics_payload)
    manifest_cell.update(summary)
    if evidence_name not in mandatory:
        mandatory.append(evidence_name)
    _atomic_json(manifest_path, manifest)
    del route_payload, evidence_records
    gc.collect()
    return (dict(summary), quality_payload, "externalize_duplicate_full_sqi_evidence_v1")


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
    from ppg_frailty.training import read_oof_parquet, validate_unique_subject_oof

    artifact_contracts = (
        ("oof_file_predictions.parquet", "file"),
        ("oof_subject_predictions.parquet", "participant"),
        ("oof_window_predictions.parquet", "window"),
        ("oof_role_predictions.parquet", "role"),
        ("oof_member_predictions.parquet", "participant"),
    )
    rows_by_level = tuple((read_oof_parquet(cell_directory / filename) for filename, _level in artifact_contracts))
    _require(bool(rows_by_level[0] and rows_by_level[1]), "recovery requires non-empty file and subject OOF")
    expected_participants = set(map(str, expected_participant_ids))
    for (filename, expected_level), rows in zip(artifact_contracts, rows_by_level):
        validate_unique_subject_oof(rows)
        for row in rows:
            valid_identity = (str(row.config_hash) == expected_config_hash and int(row.repeat) == repeat_index
                              and int(row.fold) == fold_index and int(row.split_seed) == expected_split_seed
                              and str(row.level) == expected_level and str(row.participant_id) in expected_participants
                              and str(row.source_snapshot_hash) == expected_source_version
                              and str(row.code_commit) == expected_code_commit)
            _require(valid_identity, f"recovery OOF identity drift in {filename}")
    subject_rows = rows_by_level[1]
    subject_participants = [str(row.participant_id) for row in subject_rows]
    _require(
        len(subject_participants) == len(expected_participants) and set(subject_participants) == expected_participants,
        "recovery participant OOF roster is not exact-once",
    )
    member_rows = rows_by_level[4]
    _require(
        not member_rows or {str(row.participant_id)
                            for row in member_rows} == expected_participants,
        "recovery member OOF participant roster drift",
    )
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
        ("training_history.json", "ppg_frailty.training_history.v2", "training_history_row_count"),
        ("physical_recording_qc.json", "ppg_frailty.physical_recording_qc.v2", "physical_recording_qc_row_count"),
    )
    for filename, schema_version, count_field in row_artifacts:
        payload = _load_json(cell_directory / filename)
        rows = payload.get("rows")
        valid_rows = (payload.get("schema_version") == schema_version
                      and _indices_match(payload, repeat_index, fold_index) and isinstance(rows, list)
                      and not isinstance(summary.get(count_field), bool)
                      and int(summary.get(count_field, -1)) == len(rows))
        _require(valid_rows, f"recovery {filename} schema/identity/count drift")
    confusion = _load_json(cell_directory / "confusion_matrices.json")
    confusion_cells = confusion.get("cells")
    valid_confusion = (
        _matches(confusion, schema_version="ppg_frailty.confusion_matrices.v2", pipeline_generation="final_pipeline_v2")
        and isinstance(confusion_cells, list) and len(confusion_cells) == 1
        and _indices_match(confusion_cells[0], repeat_index, fold_index)
        and confusion_cells[0].get("class_order") == summary.get("class_order")
        and confusion_cells[0].get("confusion_matrix") == summary.get("metrics", {}).get("confusion_matrix"))
    _require(valid_confusion, "recovery confusion artifact identity drift")
    cache = _load_json(cell_directory / "preprocessing_cache.json")
    _require(
        cache.get("schema_version") == "ppg_frailty.preprocessing_cache_audit.v1"
        and cache.get("affects_predictions") is False,
        "recovery preprocessing cache audit contract drift",
    )
    pointer_contract = {
        "quality_diagnostics_artifact": "quality_diagnostics.json",
        "training_history_artifact": "training_history.json",
        "physical_recording_qc_artifact": "physical_recording_qc.json",
        "route_artifacts_artifact": "route_artifacts.json",
        "route_window_sqi_evidence_artifact": "route_window_sqi_evidence.jsonl.gz",
        "preprocessing_cache_artifact": "preprocessing_cache.json",
    }
    for field, expected_name in pointer_contract.items():
        _require(summary.get(field) == expected_name, f"recovery externalized pointer drift: {field}")
    metrics = _load_json(cell_directory / "metrics_per_fold_seed.json")
    metric_cells = metrics.get("cells")
    valid_metrics = (_matches(
        metrics, schema_version="ppg_frailty.metrics_per_fold_seed.v2", pipeline_generation="final_pipeline_v2")
                     and isinstance(metric_cells, list) and len(metric_cells) == 1 and metric_cells[0] == dict(summary))
    _require(valid_metrics, "recovery compact metrics summary drift")
    manifest = _load_json(cell_directory / "run_manifest.json")
    mandatory = manifest.get("mandatory_artifacts")
    valid_manifest = (_matches(
        manifest,
        schema_version="ppg_frailty.run_manifest.v2",
        pipeline_generation="final_pipeline_v2",
        status="passed",
        scientific_scope="frozen_5x5_scientific_benchmark",
    ) and manifest.get("cell") == dict(summary) and isinstance(mandatory, list)
                      and not any(not (cell_directory / str(name)).is_file() for name in mandatory))
    _require(valid_manifest, "recovery compact cell manifest drift")


def _cell_path(repeat: int, fold: int, layout: str) -> Path:
    if layout == "nested":
        return Path(f"repeat_{repeat:02d}") / f"fold_{fold:02d}"
    return Path(f"repeat_{repeat:02d}_fold_{fold:02d}")


def _discover_staged_cells(source: Path, layout: str) -> dict[tuple[int, int], Path]:
    if layout == "flat":
        return {(int(path.name[7:9]), int(path.name[15:17])): path
                for path in source.iterdir() if path.is_dir() and len(path.name) == len("repeat_00_fold_00")
                and path.name.startswith("repeat_") and "_fold_" in path.name}
    return {(int(repeat.name[7:9]), int(fold.name[5:7])): fold
            for repeat in source.iterdir()
            if repeat.is_dir() and len(repeat.name) == len("repeat_00") and repeat.name.startswith("repeat_")
            for fold in repeat.iterdir()
            if fold.is_dir() and len(fold.name) == len("fold_00") and fold.name.startswith("fold_")}


def _recover_cell(
    *,
    source_cell: Path,
    recovery_staging: Path,
    repeat: int,
    fold: int,
    layout: str,
    config: Any,
    registry: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any], str, str, str]:
    from ppg_frailty import experiment as experiment_module

    relative = _cell_path(repeat, fold, layout)
    name = relative.as_posix()
    recovered = recovery_staging / relative
    source_sha256 = _hash_files(source_cell, _TRANSFORMED_CELL_FILES)
    source_prediction_sha256 = _hash_files(source_cell, _ROOT_OOF)
    shutil.copytree(source_cell, recovered, copy_function=os.link)
    summary, quality, mode = _externalize_legacy_sqi_payload(recovered,
                                                             config_id=str(config.config_id),
                                                             config_hash=str(config.sha256))
    _require(
        _indices_match(summary, repeat, fold) and str(summary.get("status")) == "passed",
        "recovery cell summary identity drift",
    )
    source_version, code_version = (str(summary.get("source_version", "")), str(summary.get("code_commit", "")))
    _require(len(source_version) == 64 and bool(code_version), "recovery cell training source provenance missing")
    split = registry.get_split(repeat, fold)
    oof_rows = _read_cell_oof(
        recovered,
        expected_config_hash=str(config.sha256),
        repeat_index=repeat,
        fold_index=fold,
        expected_split_seed=int(split["split_seed"]),
        expected_participant_ids=split["oof_participant_ids"],
        expected_source_version=source_version,
        expected_code_commit=code_version,
    )
    _validate_compacted_cell_artifacts(recovered, summary, repeat_index=repeat, fold_index=fold)
    root_summary = _prefix_cell_pointers(summary, name)
    root_summary["quality_diagnostics"] = quality.get("rows", [])
    cell = experiment_module._CellResult(
        summary=root_summary,
        file_rows=oof_rows[0],
        subject_rows=oof_rows[1],
        window_rows=oof_rows[2],
        role_rows=oof_rows[3],
        member_rows=oof_rows[4],
    )
    indexed = dict(root_summary)
    indexed.pop("quality_diagnostics", None)
    recovered_sha256 = _hash_files(recovered, _TRANSFORMED_CELL_FILES)
    recovered_sha256["route_window_sqi_evidence.jsonl.gz"] = str(summary["route_window_sqi_evidence_sha256"])
    recovered_prediction_sha256 = _hash_files(recovered, _ROOT_OOF)
    _require(recovered_prediction_sha256 == source_prediction_sha256, "recovery changed a cell prediction artifact")
    lineage = {
        "repeat_index": repeat,
        "fold_index": fold,
        "source_cell": name,
        "recovered_cell": name,
        "source_artifact_sha256": source_sha256,
        "recovered_artifact_sha256": recovered_sha256,
        "source_prediction_sha256": source_prediction_sha256,
        "recovered_prediction_sha256": recovered_prediction_sha256,
        "prediction_artifacts_hard_linked_unchanged": True,
        "sqi_recovery_mode": mode,
        "model_refit": False,
    }
    return (cell, indexed, lineage, mode, source_version, code_version)


def recover_completed_full_experiment_staging(
    config_path: str | Path,
    *,
    interrupted_staging: str | Path,
    output_dir: str | Path,
    repeats: Iterable[int],
    folds: Iterable[int],
    measure_operational_costs: bool,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    cell_directory_layout: str = "flat",
) -> Any | None:
    """Publish a complete 5x5 staging tree without rerunning any training cell."""
    _require(cell_directory_layout in {"flat", "nested"}, "cell_directory_layout must be flat or nested")
    repeat_values, fold_values = tuple(map(int, repeats)), tuple(map(int, folds))
    expected_keys = {(repeat_index, fold_index) for repeat_index in repeat_values for fold_index in fold_values}
    if expected_keys != {(repeat_index, fold_index) for repeat_index in range(5) for fold_index in range(5)}:
        return None
    source, target = Path(interrupted_staging).resolve(), Path(output_dir).resolve()
    invalid_source = (not source.is_dir() or source.parent != target.parent
                      or not source.name.startswith(f".{target.name}.staging.") or target.exists())
    if invalid_source:
        return None
    observed_cells = _discover_staged_cells(source, cell_directory_layout)
    if set(observed_cells) != expected_keys:
        return None
    for path in observed_cells.values():
        _safe_mandatory_artifacts(path)
    from ppg_frailty import experiment as experiment_module
    from ppg_frailty.pipeline import PipelinePaths, preflight_pipeline

    report, config, _, registry = preflight_pipeline(config_path, mode="full", paths=PipelinePaths.discover())
    recovery_staging = target.with_name(f".{target.name}.recovery-staging.{time.time_ns()}")
    source_root_oof_present = {filename: (source / filename).is_file() for filename in _ROOT_OOF}
    if any(source_root_oof_present.values()) and (not all(source_root_oof_present.values())):
        raise ValueError("interrupted staging contains a partial root OOF set")
    source_root_oof_sha256 = _hash_files(source, _ROOT_OOF) if all(source_root_oof_present.values()) else {}
    recovery_staging.mkdir(parents=True, exist_ok=False)
    cells, indexed_summaries, lineage_rows = [], [], []
    sqi_recovery_modes, training_source_versions, training_code_versions = set(), set(), set()
    started = time.perf_counter()
    try:
        ordered_keys = sorted(expected_keys)
        _emit(
            progress_callback,
            {
                "event": "run_start",
                "message": "recovering 25 completed cells; no model refit",
                "total_cells": len(ordered_keys),
                "output_dir": str(target),
            },
        )
        for cell_number, (repeat_index, fold_index) in enumerate(ordered_keys, start=1):
            _emit(
                progress_callback,
                {
                    "event": "cell_start",
                    "message": "validating and compacting interrupted cell",
                    "current_cell": cell_number,
                    "total_cells": len(ordered_keys),
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                },
            )
            cell, indexed, lineage, mode, source_version, code_version = _recover_cell(
                source_cell=observed_cells[repeat_index, fold_index],
                recovery_staging=recovery_staging,
                repeat=repeat_index,
                fold=fold_index,
                layout=cell_directory_layout,
                config=config,
                registry=registry,
            )
            cells.append(cell)
            indexed_summaries.append(indexed)
            lineage_rows.append(lineage)
            sqi_recovery_modes.add(mode)
            training_source_versions.add(source_version)
            training_code_versions.add(code_version)
            gc.collect()
            _emit(
                progress_callback,
                {
                    "event": "cell_complete",
                    "message": "recovered completed cell",
                    "current_cell": cell_number,
                    "total_cells": len(ordered_keys),
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "status": "passed",
                },
            )
        if {str(cell.summary.get("scientific_scope")) for cell in cells} != {"frozen_5x5_scientific_benchmark"}:
            raise ValueError("recovery scientific scope is not the frozen 5x5 grid")
        if len(training_source_versions) != 1 or len(training_code_versions) != 1:
            raise ValueError("recovery cells mix training source provenance")
        training_source_version = next(iter(training_source_versions))
        training_code_version = next(iter(training_code_versions))
        recovery_source_version = experiment_module._source_version()
        recovery_code_version = experiment_module._code_version()
        recovery_transform = (next(iter(sqi_recovery_modes))
                              if len(sqi_recovery_modes) == 1 else "mixed_externalize_or_reuse_sqi_evidence_v1")
        metrics = dict(
            requested_cell_count=25,
            passed_cell_count=25,
            failed_cell_count=0,
            elapsed_seconds=sum(float(cell.summary.get("elapsed_seconds", 0.0)) for cell in cells),
            recovery_finalize_seconds=time.perf_counter() - started,
        )
        provenance = dict(
            preflight_status=report.status,
            manifest_hash=report.manifest_hash,
            fold_hash=report.fold_hash,
            frozen_outer_split=True,
            data_shortening=False,
            record_cap=None,
            epoch_override=None,
            operational_measurement_requested=bool(measure_operational_costs),
            code_version=training_code_version,
            source_version=training_source_version,
            recovered_from_complete_interrupted_staging=True,
            interrupted_staging_preserved=str(source),
            recovery_transform=recovery_transform,
            recovery_model_refit=False,
            recovery_predictions_changed=False,
            recovery_code_version=recovery_code_version,
            recovery_source_version=recovery_source_version,
            recovery_manifest="recovery_manifest.json",
            trusted_run_replay_required_before_publish=True,
        )
        result = experiment_module.ExperimentResult(
            status="passed",
            scientific_scope="frozen_5x5_scientific_benchmark",
            config_id=str(config.config_id),
            config_hash=str(config.sha256),
            repeat_indices=repeat_values,
            fold_indices=fold_values,
            output_dir=str(target),
            cell_results=tuple(indexed_summaries),
            metrics=metrics,
            provenance=provenance,
        )
        experiment_module._write_full_root_artifacts(recovery_staging,
                                                     cells,
                                                     result,
                                                     cell_directory_layout=cell_directory_layout)
        recovered_root_oof_sha256 = _hash_files(recovery_staging, _ROOT_OOF)
        if source_root_oof_sha256 and recovered_root_oof_sha256 != source_root_oof_sha256:
            raise ValueError("recovered root OOF bytes differ from interrupted staging")
        trusted = experiment_module._read_trusted_comparison_run(str(config.config_id),
                                                                 recovery_staging,
                                                                 n_bootstrap_resamples=None,
                                                                 bootstrap_seed=None)
        if str(trusted.get("config_hash")) != str(config.sha256) or str(trusted.get("machine_id")) != str(
                cells[0].summary.get("model_machine_id")):
            raise ValueError("recovery trusted run replay identity drift")
        del trusted
        gc.collect()
        recovery_manifest = dict(
            schema_version="ppg_frailty.recovery_manifest.v1",
            pipeline_generation="final_pipeline_v2",
            status="passed_before_atomic_publish",
            purpose="finalize_complete_interrupted_5x5_without_refit",
            source_staging=str(source),
            source_staging_preserved=True,
            target=str(target),
            config_id=str(config.config_id),
            config_hash=str(config.sha256),
            training_code_version=training_code_version,
            training_source_version=training_source_version,
            recovery_code_version=recovery_code_version,
            recovery_source_version=recovery_source_version,
            transformation=recovery_transform,
            model_refit=False,
            prediction_artifacts_changed=False,
            trusted_run_replay="passed",
            source_root_oof_sha256=source_root_oof_sha256,
            recovered_root_oof_sha256=recovered_root_oof_sha256,
            root_oof_byte_identity="identical" if source_root_oof_sha256 else "source_root_not_yet_materialized",
            cells=lineage_rows,
        )
        experiment_module._strict_json(recovery_staging / "recovery_manifest.json", recovery_manifest)
        experiment_module._strict_json(recovery_staging / "experiment_result.json", result.to_dict())
        experiment_module._commit_staging(recovery_staging, target)
        _emit(
            progress_callback,
            {
                "event": "run_complete",
                "message": "interrupted experiment recovered without refitting",
                "status": "passed",
                "total_cells": len(ordered_keys),
                "passed_cells": len(ordered_keys),
                "failed_cells": 0,
                "output_dir": str(target),
            },
        )
        return result
    except Exception:
        if recovery_staging.exists():
            shutil.rmtree(recovery_staging)
        raise


def _validate_published(
    config_path: str | Path,
    output_dir: str | Path,
    repeats: Iterable[int],
    folds: Iterable[int],
    *,
    layout: str,
    recovered: bool,
) -> dict[str, Any] | None:
    """Read-only validation shared by normal and recovery publication."""
    _require(layout in {"flat", "nested"}, "cell_directory_layout must be flat or nested")
    repeat_values, fold_values = (tuple(map(int, repeats)), tuple(map(int, folds)))
    if repeat_values != tuple(range(5)) or fold_values != tuple(range(5)):
        return None
    root = Path(output_dir).resolve()
    required = ("experiment_result.json", "run_manifest.json", "config_metrics_v2.json", *_ROOT_OOF)
    required += ("recovery_manifest.json", ) if recovered else ()
    if not root.is_dir() or any((not (root / name).is_file() for name in required)):
        return None
    from ppg_frailty import experiment as experiment_module
    from ppg_frailty.pipeline import PipelinePaths, preflight_pipeline

    _, config, _, _ = preflight_pipeline(config_path, mode="full", paths=PipelinePaths.discover())
    result, manifest = _load_json(root / "experiment_result.json"), _load_json(root / "run_manifest.json")
    label = "recovery" if recovered else "experiment"
    for payload in (result, manifest):
        identity_ok = (_matches(
            payload,
            status="passed",
            scientific_scope="frozen_5x5_scientific_benchmark",
            config_id=config.config_id,
            config_hash=config.sha256,
        ) and tuple(payload.get("repeat_indices", ())) == repeat_values and tuple(payload.get("fold_indices",
                                                                                              ())) == fold_values)
        _require(identity_ok, f"published {label} result identity drift")
    mandatory = manifest.get("mandatory_artifacts")
    complete_manifest = (manifest.get("schema_version") == "ppg_frailty.run_manifest.v2"
                         and isinstance(mandatory, list) and all((root / str(name)).is_file() for name in mandatory)
                         and (not recovered or "recovery_manifest.json" in mandatory))
    _require(complete_manifest, f"published {label} mandatory artifact drift")
    if recovered:
        recovery = _load_json(root / "recovery_manifest.json")
        custody = (_matches(
            recovery,
            schema_version="ppg_frailty.recovery_manifest.v1",
            status="passed_before_atomic_publish",
            config_id=config.config_id,
            config_hash=config.sha256,
        ) and recovery.get("model_refit") is False and recovery.get("prediction_artifacts_changed") is False
                   and recovery.get("trusted_run_replay") == "passed")
        _require(custody, "published recovery chain-of-custody drift")
        hashes = recovery.get("recovered_root_oof_sha256")
        _require(
            isinstance(hashes, Mapping) and (not any(
                (hashes.get(name) != _sha256_file(root / name) for name in _ROOT_OOF))),
            "published recovery root OOF hash drift",
        )
    else:
        cells = [root / _cell_path(repeat, fold, layout) for repeat in repeat_values for fold in fold_values]
        _require(not any((not cell.is_dir() for cell in cells)), "published experiment cell roster is incomplete")
        for cell in cells:
            _safe_mandatory_artifacts(cell)
        rows = result.get("cell_results")
        _require(
            isinstance(rows, list) and len(rows) == 25 and (not any(
                (not isinstance(row, Mapping) or row.get("status") != "passed" for row in rows))),
            "published experiment result does not contain 25 passed cells",
        )
    trusted = experiment_module._read_trusted_comparison_run(str(config.config_id),
                                                             root,
                                                             n_bootstrap_resamples=None,
                                                             bootstrap_seed=None)
    _require(str(trusted.get("config_hash")) == str(config.sha256), f"published {label} trusted replay config drift")
    return result


def validate_published_complete_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    repeats: Iterable[int],
    folds: Iterable[int],
    cell_directory_layout: str = "flat",
) -> dict[str, Any] | None:
    return _validate_published(config_path, output_dir, repeats, folds, layout=cell_directory_layout, recovered=False)


def validate_published_recovered_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    repeats: Iterable[int],
    folds: Iterable[int],
) -> dict[str, Any] | None:
    return _validate_published(config_path, output_dir, repeats, folds, layout="flat", recovered=True)


__all__ = [
    "recover_completed_full_experiment_staging",
    "validate_published_complete_experiment",
    "validate_published_recovered_experiment",
]
