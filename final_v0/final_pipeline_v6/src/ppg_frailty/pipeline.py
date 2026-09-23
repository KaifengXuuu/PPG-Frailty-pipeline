"""Frozen-input preflight and recording loading shared by V5 workflows.

Preflight reads the materialized manifest and fold registry without regenerating
splits. ``smoke`` checks one split; ``full`` checks the complete 5-by-5 roster.
Training, predictions and reports are owned by their dedicated V5 services.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, load_config
from .contracts import to_strict_json_value
from .data.folds import (
    FrozenFoldRegistry,
    M2_SEEDS,
    M2_SPLIT_FILE_SHA256,
    M2_SPLIT_PAYLOAD_SHA256,
)
from .data.manifest import (
    M2_FILE_MANIFEST_SHA256,
    audit_manifest,
    load_internal_manifest,
)
from .module_registry import (
    registry_sha256,
    resolve_artifact_config,
    resolve_peak_detector_config,
    resolve_window_config,
    validate_model_config,
)
from .provenance import sha256_file
from .paths import resolve_repository_root

@dataclass(frozen=True)
class PipelinePaths:
    """Locate packaged pipeline files separately from external recording data."""

    pipeline_root: Path
    repository_root: Path

    @classmethod
    def discover(cls) -> "PipelinePaths":
        """Locate from installed source."""

        root = Path(__file__).resolve().parents[2]
        return cls(root, resolve_repository_root(root))

    def input_path(self, relative: str | Path) -> Path:
        """Restrict configured inputs to the V2 root."""

        candidate = (self.pipeline_root / Path(relative)).resolve()
        candidate.relative_to(self.pipeline_root)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def output_path(self, path: str | Path) -> Path:
        """Restrict every output to the V2 root."""

        candidate = Path(path)
        candidate = candidate.resolve() if candidate.is_absolute() else (self.pipeline_root / candidate).resolve()
        candidate.relative_to(self.pipeline_root)
        return candidate

@dataclass(frozen=True)
class PreflightReport:
    """Machine-verifiable formal preflight."""

    status: str
    config_id: str
    config_hash: str
    representation_mode: str
    model: dict[str, str]
    artifact: dict[str, Any]
    peak_detector: dict[str, Any]
    manifest_path: str
    manifest_hash: str
    fold_path: str
    fold_hash: str
    manifest_authority_hash: str
    fold_authority_file_hash: str
    fold_authority_payload_hash: str
    manifest_materialization_report_hash: str
    source_unit_schema_hash: str
    record_count: int
    selected_record_count: int
    participant_count: int
    split_count: int
    split_seeds: tuple[int, ...]
    module_registry_hash: str
    window_profiles: dict[str, dict[str, Any]]

def _config_path(path: str | Path, paths: PipelinePaths) -> Path:
    """Resolve a config path.

    Explicit absolute YAMLs may live in an external study archive. Relative
    configs remain rooted in V2, while every manifest/fold/data reference inside
    the config is still resolved through :meth:`PipelinePaths.input_path`.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (paths.pipeline_root / candidate).resolve()
        candidate.relative_to(paths.pipeline_root)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("pipeline config must be a YAML file")
    return candidate

def preflight_pipeline(
    config_path: str | Path,
    *,
    mode: str = "smoke",
    paths: PipelinePaths | None = None,
) -> tuple[PreflightReport, PipelineConfig, list[Any], FrozenFoldRegistry]:
    """Run fail-closed preflight."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    resolved_paths = paths or PipelinePaths.discover()
    config = load_config(_config_path(config_path, resolved_paths))
    payload = config.to_dict()
    artifact = resolve_artifact_config(config.section("artifact"))
    peak_detector = resolve_peak_detector_config(config.section("signal"))
    windows = resolve_window_config(config.section("windows"))
    model = validate_model_config(config.section("model"), config.representation_mode)
    manifest_section = config.section("manifest")
    split_section = config.section("splits")
    if split_section.get("runtime_recompute") is not False:
        raise ValueError("formal pipeline forbids runtime fold recomputation")
    if (
        int(split_section.get("n_splits", 0)) != 5
        or int(split_section.get("n_repeats", 0)) != 5
        or tuple(int(value) for value in split_section.get("split_seeds", ())) != M2_SEEDS
    ):
        raise ValueError("formal pipeline requires the frozen corrected 5x5 registry")
    manifest_path = resolved_paths.input_path(str(manifest_section["path"]))
    fold_path = resolved_paths.input_path(str(split_section["path"]))
    if manifest_section.get("source_manifest_sha256") != M2_FILE_MANIFEST_SHA256:
        raise ValueError("config internal-manifest authority SHA drift")
    if split_section.get("source_registry_file_sha256") != M2_SPLIT_FILE_SHA256:
        raise ValueError("config fold authority file SHA drift")
    if split_section.get("source_registry_payload_sha256") != M2_SPLIT_PAYLOAD_SHA256:
        raise ValueError("config fold authority payload SHA drift")
    materialization_report_path = resolved_paths.input_path("reports/internal_manifest_v2_report.json")
    materialization = json.loads(materialization_report_path.read_text(encoding="utf-8"))
    expected_materialized = materialization.get("generated_artifact", {})
    if (
        materialization.get("schema_version") != "ppg_frailty.internal_manifest_materialization.v2"
        or materialization.get("pipeline_generation") != "final_pipeline_v2"
        or materialization.get("status") != "passed"
        or materialization.get("all_261_source_hashes_verified") is not True
        or expected_materialized.get("path") != manifest_path.relative_to(resolved_paths.pipeline_root).as_posix()
        or expected_materialized.get("sha256") != sha256_file(manifest_path)
        or int(expected_materialized.get("bytes", -1)) != manifest_path.stat().st_size
    ):
        raise ValueError("internal V2 manifest is not bound to a passed byte-rehash report")
    rows = load_internal_manifest(manifest_path)
    summary = audit_manifest(rows)
    if int(manifest_section.get("expected_record_count", -1)) != int(summary["record_count"]):
        raise ValueError("manifest record count differs from frozen config")
    if int(manifest_section.get("expected_participant_count", -1)) != int(summary["participant_count"]):
        raise ValueError("manifest participant count differs from frozen config")
    expected_channels = tuple(str(value) for value in manifest_section["channel_order"])
    if any(tuple(row.channel_schema) != expected_channels for row in rows):
        raise ValueError("manifest channel order differs from frozen config")
    unit_schemas = {json.dumps(row.channel_units, sort_keys=True, separators=(",", ":")) for row in rows}
    if len(unit_schemas) != 1:
        raise ValueError("internal records do not share one frozen source-unit schema")
    source_unit_schema_hash = hashlib.sha256(next(iter(unit_schemas)).encode("utf-8")).hexdigest()
    fold_registry = FrozenFoldRegistry.from_csv(fold_path)
    if any(
        row.source_registry_file_sha256 != M2_SPLIT_FILE_SHA256
        or row.source_registry_payload_sha256 != M2_SPLIT_PAYLOAD_SHA256
        for row in fold_registry.assignments
    ):
        raise ValueError("materialized fold CSV is not bound to the frozen authority")
    participants = {row.participant_id for row in rows}
    if set(fold_registry.participant_ids) != participants:
        raise ValueError("manifest and fold participant rosters differ")
    roles = set(payload["roles"])
    selected = [row for row in rows if row.role in roles and row.qc_status in {"pass", "pass_with_warnings"}]
    if not selected:
        raise ValueError("configuration selected no eligible records")
    # Resolve every split: this exposes missing repeat/fold memberships and
    # detects missing repeat/fold memberships without ever regenerating them.
    splits = [fold_registry.get_split(repeat, fold) for repeat in range(5) for fold in range(5)]
    if mode == "smoke":
        splits = splits[:1]
    report = PreflightReport(
        status="passed",
        config_id=config.config_id,
        config_hash=config.sha256,
        representation_mode=config.representation_mode,
        model=model,
        artifact=artifact,
        peak_detector=peak_detector,
        manifest_path=manifest_path.relative_to(resolved_paths.pipeline_root).as_posix(),
        manifest_hash=sha256_file(manifest_path),
        fold_path=fold_path.relative_to(resolved_paths.pipeline_root).as_posix(),
        fold_hash=sha256_file(fold_path),
        manifest_authority_hash=M2_FILE_MANIFEST_SHA256,
        fold_authority_file_hash=M2_SPLIT_FILE_SHA256,
        fold_authority_payload_hash=M2_SPLIT_PAYLOAD_SHA256,
        manifest_materialization_report_hash=sha256_file(materialization_report_path),
        source_unit_schema_hash=source_unit_schema_hash,
        record_count=int(summary["record_count"]),
        selected_record_count=len(selected),
        participant_count=int(summary["participant_count"]),
        split_count=len(splits),
        split_seeds=tuple(sorted({int(item["split_seed"]) for item in splits})),
        module_registry_hash=registry_sha256(),
        window_profiles=windows,
    )
    return report, config, rows, fold_registry

def _audit_source_identity(
    row: Any,
    paths: PipelinePaths,
) -> tuple[Path, bytes, str]:
    """Read once, then verify source bytes and row/header/unit identity."""

    source = (paths.repository_root / row.source_path).resolve()
    source.relative_to(paths.repository_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != row.source_hash:
        raise ValueError(f"source hash drift: {row.record_id}")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source UTF-8 decode failed: {row.record_id}") from exc
    with io.StringIO(source_text, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        row_count = 0
        for line_number, values in enumerate(reader, start=2):
            if len(values) != 8:
                raise ValueError(f"source column-count drift: {row.record_id}:line={line_number}")
            row_count += 1
    if tuple(header or ()) != tuple(row.channel_schema):
        raise ValueError(f"source channel order drift: {row.record_id}")
    if row_count != int(row.n_samples):
        raise ValueError(f"source sample-count drift: {row.record_id}:{row_count}!={row.n_samples}")
    units = dict(row.channel_units)
    expected_units = {
        "RED": "raw_device_counts_adc_scale_unknown",
        "IR": "raw_device_counts_adc_scale_unknown",
        "AX": "g_source_declared",
        "AY": "g_source_declared",
        "AZ": "g_source_declared",
        "GX": "degree_per_second_source_declared",
        "GY": "degree_per_second_source_declared",
        "GZ": "degree_per_second_source_declared",
    }
    if units != expected_units:
        raise ValueError(f"source unit schema drift: {row.record_id}")
    if not np.isclose(float(row.duration_s), row_count / float(row.fs), atol=1e-9):
        raise ValueError(f"source duration/sample-rate drift: {row.record_id}")
    return source, source_bytes, source_sha256

def _load_record(row: Any, paths: PipelinePaths, *, max_samples: int | None) -> dict[str, Any]:
    """Full-load, audit physical QC, then optionally return a leading slice."""

    source, source_bytes, source_sha256 = _audit_source_identity(row, paths)
    source_text = source_bytes.decode("utf-8")
    full_values = np.loadtxt(
        io.StringIO(source_text, newline=""),
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    full_values = np.atleast_2d(full_values)
    if full_values.ndim != 2 or full_values.shape[1] != 8:
        raise ValueError(f"source numeric structure failed: {row.record_id}")
    from .data.qc import assess_manifest_record, require_recording_qc_pass

    thresholds = physical_recording_qc_thresholds_v2()
    admission = assess_manifest_record(
        row,
        full_values,
        observed_channel_names=tuple(row.channel_schema),
        observed_fs=float(row.fs),
        thresholds=thresholds,
        timestamps_s=None,
    )
    require_recording_qc_pass(admission)
    if max_samples is None:
        returned_samples = int(full_values.shape[0])
    else:
        returned_samples = int(max_samples)
        if returned_samples <= 0 or returned_samples > int(full_values.shape[0]):
            raise ValueError(f"invalid max_samples for {row.record_id}: {max_samples}")
    values = full_values[:returned_samples]
    return {
        "record_id": row.record_id,
        "fs_hz": float(row.fs),
        "ppg": values[:, :2],
        "acc": values[:, 2:5],
        "gyro": values[:, 5:8],
        "acc_unit": "g",
        "gyro_unit": "deg/s",
        "source_path": source,
        "recording_qc": to_strict_json_value(
            {
                **dict(admission.evidence),
                "source_byte_identity": {
                    "expected_sha256": row.source_hash,
                    "observed_buffer_sha256": source_sha256,
                    "read_operation_count": 1,
                    "header_parsed_from_same_buffer": True,
                    "numeric_values_parsed_from_same_buffer": True,
                },
            }
        ),
        "recording_qc_profile": physical_recording_qc_profile_v2(),
        "full_record_n_samples_before_slice": int(full_values.shape[0]),
        "returned_n_samples_after_slice": returned_samples,
    }

def physical_recording_qc_thresholds_v2() -> Any:
    """Return the named non-device physical admission profile.

    Zero standard-deviation/span floors reject only exact constants. Any
    non-finite run is rejected. CSV files have no source timestamp column, so
    sampling-grid identity remains manifest-bound. Device rails, absolute scale,
    clipping and saturation remain deferred and are not executed.
    """

    from .data.qc import physical_recording_qc_thresholds_v2 as canonical

    return canonical()

def physical_recording_qc_profile_v2() -> dict[str, Any]:
    """Serialize the exact applied thresholds without invented device limits."""

    from .data.qc import physical_recording_qc_profile_v2 as canonical

    return canonical()
