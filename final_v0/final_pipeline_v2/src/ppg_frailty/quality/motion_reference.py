"""Canonical motion entries bound to authoritative source files.

The functions in this module perform no work at import time. The internal
entry reads and re-hashes all 261 frozen files itself. The PTT entry reads all
66 authoritative CSV records itself and requires the exact hash-bound V2-036
unit evidence before any unit conversion, EKF, materialization, or evaluation.
An optional observer receives progress counters only; it cannot inject data,
models, splits, or scientific parameters.  The training device is an explicit
operational input and is restricted to CUDA.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import ManifestRow
from ..data.external_manifest import (
    M2_EXTERNAL_MANIFEST_SHA256,
    M2_EXTERNAL_RELATIVE_PATH,
    PTT_ADOPTED_ACCELERATION_CONVERSION,
    PTT_ADOPTED_ACCELERATION_UNIT,
    PTT_ADOPTED_GYROSCOPE_CONVERSION,
    PTT_ADOPTED_GYROSCOPE_UNIT,
    PTT_DATASET_ID,
    PTT_CHANNEL_MAPPING_PROVENANCE,
    PTT_IMU_UNIT_CONFLICT_PROVENANCE,
    PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH,
    PTT_IMU_UNIT_EVIDENCE_SHA256,
    ExternalRecord,
    adapt_ptt_synchronized_channels,
    load_m2_external_manifest,
)
from ..data.manifest import (
    M2_DATASET_VERSION_ID,
    M2_FILE_MANIFEST,
    M2_FILE_MANIFEST_SHA256,
    load_m2_internal_manifest,
)
from ..data.qc import (
    assess_manifest_record,
    physical_recording_qc_profile_v2,
    physical_recording_qc_thresholds_v2,
    require_recording_qc_pass,
)
from ..provenance import stable_payload_sha256
from ..representations.motion import (
    MOTION_HOP_SAMPLES,
    MOTION_NETWORK_SCHEMA_SHA256,
    MOTION_WINDOW_SAMPLES,
    MOTION_WINDOW_SECONDS,
)
from ..signal.motion_imu import (
    PTT_STATIC_CALIBRATION_ROLE,
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
)
from .motion import load_motion_fold_jobs, motion_activity_label
from .motion_adapters import (
    FormalMotionTrainerConfig,
    MotionRecordingInput,
    fit_formal_motion_model,
    load_formal_motion_model,
    materialize_motion_window_examples,
    predict_formal_motion_probability,
    require_formal_motion_cuda,
    write_formal_motion_input_schema,
)
from .motion_runner import (
    MotionExternalRunResult,
    MotionInternalRunResult,
    MotionPttTrainingRunResult,
    ProgressCallback,
    _notify_progress,
    _run_internal_reverse_evaluation_impl,
    _run_internal_motion_oof_impl,
    _run_ptt_external_evaluation_impl,
    _run_ptt_motion_training_ablation_impl,
)


FORMAL_INTERNAL_MOTION_ENTRY_ID = "formal_internal_motion_reference_source_bound_v2"
FORMAL_PTT_MOTION_ENTRY_ID = "formal_ptt_motion_reference_source_bound_v2"
FORMAL_INTERNAL_SOURCE_EVIDENCE_SCHEMA = (
    "ppg_frailty.formal_internal_motion_source_evidence.v3"
)
PTT_IMU_UNIT_EVIDENCE_SCHEMA = "ppg_frailty.ptt_imu_unit_evidence.v3"
PTT_UNRESOLVED_IMU_UNIT_STATUS = "declared_g_but_values_and_code_inference_conflict"
PTT_IMU_UNIT_DECISION_ID = "V2-036"
PTT_IMU_UNIT_DECISION_DATE = "2026-08-17"
PTT_IMU_UNIT_DECISION_AUTHORITY = "user_confirmed_project_decision"
PTT_IMU_UNIT_RESOLUTION_BASIS = "v2_036_user_adoption_hash_bound_source_evidence"
PTT_ACCELERATION_DECISION_BASIS = (
    "user_confirmed_from_hash_bound_sit_gravity_magnitude"
)
PTT_GYROSCOPE_DECISION_BASIS = (
    "wfdb_header_deg_per_s_plus_historical_deg2rad_no_numeric_conflict"
)
PTT_SOURCE_ROOT = Path("physionet.org/files/pulse-transit-time-ppg/1.1.0")
MOTION_SPLIT_RELATIVE_PATH = Path(
    "final_v0/final_pipeline_v2/splits/sgkf5_seed42_v2.csv"
)
PTT_SPLIT_RELATIVE_PATH = Path(
    "final_v0/final_pipeline_v2/splits/ptt_formal_repeated_grouped_5x5_v2.csv"
)
INTERNAL_CHANNEL_SCHEMA = ("RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ")
PTT_REQUIRED_COLUMNS = (
    "pleth_1",
    "pleth_2",
    "a_x",
    "a_y",
    "a_z",
    "g_x",
    "g_y",
    "g_z",
)
PTT_CSV_COLUMNS = (
    "time",
    "ecg",
    "peaks",
    "pleth_1",
    "pleth_2",
    "pleth_3",
    "pleth_4",
    "pleth_5",
    "pleth_6",
    "lc_1",
    "lc_2",
    "temp_1",
    "temp_2",
    "temp_3",
    "a_x",
    "a_y",
    "a_z",
    "g_x",
    "g_y",
    "g_z",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PttImuUnitEvidenceRequired(RuntimeError):
    """Structured data-readiness status until exact V2-036 units are supplied."""

    def __init__(self, statuses: Mapping[str, int]) -> None:
        self.payload = {
            "schema_version": "ppg_frailty.ptt_imu_unit_readiness.v3",
            "ready": False,
            "reason": "ptt_v2_036_unit_evidence_not_supplied",
            "source_manifest_sha256": M2_EXTERNAL_MANIFEST_SHA256,
            "observed_manifest_status_counts": dict(statuses),
            "concrete_conflict_evidence": PTT_IMU_UNIT_CONFLICT_PROVENANCE,
            "required_evidence_schema": PTT_IMU_UNIT_EVIDENCE_SCHEMA,
            "required_evidence_relative_path": (
                PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH.as_posix()
            ),
            "required_evidence_sha256": PTT_IMU_UNIT_EVIDENCE_SHA256,
            "unit_guessing_allowed": False,
        }
        super().__init__(json.dumps(self.payload, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class PttImuUnitEvidence:
    """Exact human-approved V2-036 decision bound to all 66 source hashes."""

    acceleration_unit: str
    acceleration_conversion: str
    acceleration_decision_basis: str
    gyroscope_unit: str
    gyroscope_conversion: str
    gyroscope_decision_basis: str
    decision_id: str
    decision_date: str
    decision_authority: str
    wfdb_header_relative_path: str
    wfdb_header_sha256: str
    numeric_evidence_record_id: str
    numeric_evidence_relative_path: str
    numeric_evidence_sha256: str
    numeric_evidence_first_acceleration_xyz: tuple[float, float, float]
    numeric_evidence_first_acceleration_norm: float
    numeric_evidence_first_gyroscope_xyz: tuple[float, float, float]
    numeric_evidence_first_gyroscope_norm: float
    historical_transform_relative_path: str
    historical_transform_sha256: str
    artifact_sha256: str
    source_record_sha256_by_id: Mapping[str, str]
    schema_version: str = PTT_IMU_UNIT_EVIDENCE_SCHEMA
    decision_status: str = "approved_for_v2_formal_motion"
    source_manifest_sha256: str = M2_EXTERNAL_MANIFEST_SHA256
    dataset_id: str = PTT_DATASET_ID
    resolution_basis: str = PTT_IMU_UNIT_RESOLUTION_BASIS

    def validate(self, expected_records: Sequence[ExternalRecord]) -> None:
        expected = {
            row.record_id: row.checksum_sha256
            for row in expected_records
            if row.dataset_id == PTT_DATASET_ID
        }
        if (
            self.schema_version != PTT_IMU_UNIT_EVIDENCE_SCHEMA
            or self.decision_status != "approved_for_v2_formal_motion"
            or self.decision_id != PTT_IMU_UNIT_DECISION_ID
            or self.decision_date != PTT_IMU_UNIT_DECISION_DATE
            or self.decision_authority != PTT_IMU_UNIT_DECISION_AUTHORITY
            or self.source_manifest_sha256 != M2_EXTERNAL_MANIFEST_SHA256
            or self.dataset_id != PTT_DATASET_ID
            or self.resolution_basis != PTT_IMU_UNIT_RESOLUTION_BASIS
            or self.artifact_sha256 != PTT_IMU_UNIT_EVIDENCE_SHA256
        ):
            raise ValueError("PTT IMU unit evidence identity/status drift")
        if (
            self.acceleration_unit != PTT_ADOPTED_ACCELERATION_UNIT
            or self.acceleration_conversion != PTT_ADOPTED_ACCELERATION_CONVERSION
            or self.acceleration_decision_basis != PTT_ACCELERATION_DECISION_BASIS
        ):
            raise ValueError(
                "PTT acceleration must be source m/s^2 with identity/no-scale conversion"
            )
        if (
            self.gyroscope_unit != PTT_ADOPTED_GYROSCOPE_UNIT
            or self.gyroscope_conversion != PTT_ADOPTED_GYROSCOPE_CONVERSION
            or self.gyroscope_decision_basis != PTT_GYROSCOPE_DECISION_BASIS
        ):
            raise ValueError(
                "PTT gyroscope must follow hash-bound deg/s-to-rad/s evidence"
            )
        if len(expected) != 66 or dict(self.source_record_sha256_by_id) != expected:
            raise ValueError("PTT IMU unit evidence is not bound to the exact 66 records")
        header = PTT_IMU_UNIT_CONFLICT_PROVENANCE["wfdb_header_declaration"]
        numeric = PTT_IMU_UNIT_CONFLICT_PROVENANCE["canonical_csv_numeric_evidence"]
        historical = PTT_IMU_UNIT_CONFLICT_PROVENANCE["historical_code_transform"]
        if (
            self.wfdb_header_relative_path != header["relative_path"]
            or self.wfdb_header_sha256 != header["sha256"]
            or self.numeric_evidence_record_id != numeric["record_id"]
            or self.numeric_evidence_relative_path != numeric["relative_path"]
            or self.numeric_evidence_sha256 != numeric["sha256"]
            or self.historical_transform_relative_path != historical["relative_path"]
            or self.historical_transform_sha256 != historical["sha256"]
        ):
            raise ValueError("PTT IMU unit evidence source provenance drift")
        expected_acceleration = np.asarray(
            numeric["first_acceleration_xyz"], dtype=np.float64
        )
        expected_gyroscope = np.asarray(
            numeric["first_gyroscope_xyz"], dtype=np.float64
        )
        observed_acceleration = np.asarray(
            self.numeric_evidence_first_acceleration_xyz, dtype=np.float64
        )
        observed_gyroscope = np.asarray(
            self.numeric_evidence_first_gyroscope_xyz, dtype=np.float64
        )
        acceleration_norm = float(numeric["first_acceleration_norm"])
        gyroscope_norm = float(numeric["first_gyroscope_norm"])
        if (
            observed_acceleration.shape != (3,)
            or not np.array_equal(observed_acceleration, expected_acceleration)
            or observed_gyroscope.shape != (3,)
            or not np.array_equal(observed_gyroscope, expected_gyroscope)
            or not np.isclose(
                self.numeric_evidence_first_acceleration_norm,
                acceleration_norm,
                rtol=0.0,
                atol=8.0 * abs(float(np.spacing(np.float64(acceleration_norm)))),
            )
            or not np.isclose(
                self.numeric_evidence_first_gyroscope_norm,
                gyroscope_norm,
                rtol=0.0,
                atol=8.0 * abs(float(np.spacing(np.float64(gyroscope_norm)))),
            )
        ):
            raise ValueError("PTT IMU unit evidence numeric provenance drift")


def load_ptt_imu_unit_evidence(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_records: Sequence[ExternalRecord],
) -> PttImuUnitEvidence:
    """Load only the exact immutable V2-036 evidence artifact."""

    source = Path(path).resolve()
    if str(expected_sha256) != PTT_IMU_UNIT_EVIDENCE_SHA256:
        raise ValueError("expected PTT IMU unit evidence is not the frozen V2-036 SHA-256")
    payload_bytes = source.read_bytes()
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != PTT_IMU_UNIT_EVIDENCE_SHA256:
        raise ValueError("PTT IMU unit evidence file SHA-256 mismatch")
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PTT IMU unit evidence must be one JSON object")
    expected_keys = {
        "schema_version",
        "decision_status",
        "decision_id",
        "decision_date",
        "decision_authority",
        "source_manifest_sha256",
        "dataset_id",
        "resolution_basis",
        "acceleration_unit",
        "acceleration_conversion",
        "acceleration_decision_basis",
        "gyroscope_unit",
        "gyroscope_conversion",
        "gyroscope_decision_basis",
        "wfdb_header_relative_path",
        "wfdb_header_sha256",
        "numeric_evidence_record_id",
        "numeric_evidence_relative_path",
        "numeric_evidence_sha256",
        "numeric_evidence_first_acceleration_xyz",
        "numeric_evidence_first_acceleration_norm",
        "numeric_evidence_first_gyroscope_xyz",
        "numeric_evidence_first_gyroscope_norm",
        "historical_transform_relative_path",
        "historical_transform_sha256",
        "source_record_sha256_by_id",
    }
    if set(payload) != expected_keys:
        raise ValueError("PTT IMU unit evidence field schema drift")
    evidence = PttImuUnitEvidence(
        acceleration_unit=str(payload["acceleration_unit"]),
        acceleration_conversion=str(payload["acceleration_conversion"]),
        acceleration_decision_basis=str(payload["acceleration_decision_basis"]),
        gyroscope_unit=str(payload["gyroscope_unit"]),
        gyroscope_conversion=str(payload["gyroscope_conversion"]),
        gyroscope_decision_basis=str(payload["gyroscope_decision_basis"]),
        wfdb_header_relative_path=str(payload["wfdb_header_relative_path"]),
        wfdb_header_sha256=str(payload["wfdb_header_sha256"]),
        numeric_evidence_record_id=str(payload["numeric_evidence_record_id"]),
        numeric_evidence_relative_path=str(payload["numeric_evidence_relative_path"]),
        numeric_evidence_sha256=str(payload["numeric_evidence_sha256"]),
        numeric_evidence_first_acceleration_xyz=tuple(
            float(value)
            for value in payload["numeric_evidence_first_acceleration_xyz"]
        ),
        numeric_evidence_first_acceleration_norm=float(
            payload["numeric_evidence_first_acceleration_norm"]
        ),
        numeric_evidence_first_gyroscope_xyz=tuple(
            float(value) for value in payload["numeric_evidence_first_gyroscope_xyz"]
        ),
        numeric_evidence_first_gyroscope_norm=float(
            payload["numeric_evidence_first_gyroscope_norm"]
        ),
        historical_transform_relative_path=str(
            payload["historical_transform_relative_path"]
        ),
        historical_transform_sha256=str(payload["historical_transform_sha256"]),
        artifact_sha256=observed,
        source_record_sha256_by_id=dict(payload["source_record_sha256_by_id"]),
        schema_version=str(payload["schema_version"]),
        decision_status=str(payload["decision_status"]),
        decision_id=str(payload["decision_id"]),
        decision_date=str(payload["decision_date"]),
        decision_authority=str(payload["decision_authority"]),
        source_manifest_sha256=str(payload["source_manifest_sha256"]),
        dataset_id=str(payload["dataset_id"]),
        resolution_basis=str(payload["resolution_basis"]),
    )
    evidence.validate(expected_records)
    return evidence


def _manifest_roster_sha256(rows: Sequence[ManifestRow]) -> str:
    return stable_payload_sha256(
        [
            {
                "record_id": row.record_id,
                "participant_id": row.participant_id,
                "role": row.role,
                "source_path": row.source_path,
                "source_hash": row.source_hash,
                "n_samples": row.n_samples,
                "fs": row.fs,
                "channel_units": dict(sorted(row.channel_units.items())),
            }
            for row in sorted(rows, key=lambda item: item.record_id)
        ]
    )


def _internal_materialization_eligibility(
    rows: Sequence[ManifestRow],
) -> dict[str, Any]:
    """Describe complete-window eligibility without padding short records."""

    excluded = {
        row.record_id: int(row.n_samples)
        for row in sorted(rows, key=lambda item: item.record_id)
        if int(row.n_samples) < MOTION_WINDOW_SAMPLES
    }
    eligible_rows = tuple(row for row in rows if row.record_id not in excluded)
    all_participants = {row.participant_id for row in rows}
    eligible_participants = {row.participant_id for row in eligible_rows}
    expected_window_count = sum(
        1 + (int(row.n_samples) - MOTION_WINDOW_SAMPLES) // MOTION_HOP_SAMPLES
        for row in eligible_rows
    )
    labels_by_participant = {
        participant_id: {
            motion_activity_label(row.role)
            for row in eligible_rows
            if row.participant_id == participant_id
        }
        for participant_id in sorted(all_participants)
    }
    return {
        "policy_id": "complete_frozen_motion_window_only_v1",
        "window_seconds": MOTION_WINDOW_SECONDS,
        "window_samples": MOTION_WINDOW_SAMPLES,
        "short_record_action": "exclude_record",
        "padding": "none",
        "partial_windows": False,
        "eligible_record_count": len(rows) - len(excluded),
        "excluded_record_count": len(excluded),
        "excluded_record_n_samples_by_id": excluded,
        "expected_window_count": expected_window_count,
        "eligible_participant_count": len(eligible_participants),
        "participants_without_eligible_records": sorted(
            all_participants - eligible_participants
        ),
        "participants_missing_static_records": sorted(
            participant_id
            for participant_id, labels in labels_by_participant.items()
            if 0 not in labels
        ),
        "participants_missing_motion_records": sorted(
            participant_id
            for participant_id, labels in labels_by_participant.items()
            if 1 not in labels
        ),
    }


def _resolve_bound_source(
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> Path:
    source, _ = _read_bound_source_bytes(
        repository_root, relative_path, expected_sha256
    )
    return source


def _read_bound_source_bytes(
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> tuple[Path, bytes]:
    """Read once, hash those exact bytes, and return the immutable parse buffer."""

    source = (repository_root / relative_path).resolve()
    try:
        source.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("formal motion source escapes repository root") from exc
    if not source.is_file():
        raise ValueError("formal motion source file missing or SHA-256 mismatch")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("formal motion source file missing or SHA-256 mismatch")
    return source, payload


def _load_internal_numeric_source(
    repository_root: Path,
    row: ManifestRow,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    _, payload = _read_bound_source_bytes(
        repository_root, row.source_path, row.source_hash
    )
    text = payload.decode("utf-8")
    header = tuple(next(csv.reader(io.StringIO(text, newline=""))))
    if header != INTERNAL_CHANNEL_SCHEMA:
        raise ValueError("formal internal source header drift")
    values = np.loadtxt(
        io.StringIO(text, newline=""),
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    if values.shape != (row.n_samples, len(INTERNAL_CHANNEL_SCHEMA)):
        raise ValueError("formal internal source shape differs from manifest")
    if not np.isfinite(values).all():
        raise ValueError("formal internal source contains non-finite values")
    admission = assess_manifest_record(
        row,
        values,
        observed_channel_names=INTERNAL_CHANNEL_SCHEMA,
        observed_fs=float(row.fs),
        thresholds=physical_recording_qc_thresholds_v2(),
        timestamps_s=None,
    )
    require_recording_qc_pass(admission)
    return values, dict(admission.evidence)


def _internal_source_units(row: ManifestRow) -> tuple[str, str]:
    acceleration = {row.channel_units[name] for name in ("AX", "AY", "AZ")}
    gyroscope = {row.channel_units[name] for name in ("GX", "GY", "GZ")}
    if acceleration != {"g_source_declared"}:
        raise ValueError("formal internal acceleration-unit declaration drift")
    if gyroscope != {"degree_per_second_source_declared"}:
        raise ValueError("formal internal gyroscope-unit declaration drift")
    return "g", "deg/s"


def _materialized_examples_sha256(examples: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.window_id):
        values = np.ascontiguousarray(example.values, dtype="<f4")
        identity = {
            "window_id": example.window_id,
            "participant_id": example.participant_id,
            "file_id": example.file_id,
            "role_or_activity": example.role_or_activity,
            "activity_label": int(example.activity_label),
            "dataset_id": example.dataset_id,
            "shape": list(values.shape),
            "dtype": "<f4",
        }
        digest.update(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _seal_source_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["source_evidence_sha256"] = stable_payload_sha256(sealed)
    return sealed


def _build_internal_materialization(
    repository_root: Path,
    config: RollPitchEkfConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    rows = tuple(load_m2_internal_manifest(repository_root, verify_sources=True))
    by_participant: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_participant.setdefault(row.participant_id, []).append(row)
    if len(rows) != 261 or len(by_participant) != 29:
        raise ValueError("formal internal source roster is not the exact 261/29 snapshot")

    materialization_eligibility = _internal_materialization_eligibility(rows)
    excluded_record_ids = set(
        materialization_eligibility["excluded_record_n_samples_by_id"]
    )
    eligible_record_ids = {
        row.record_id for row in rows if row.record_id not in excluded_record_ids
    }
    if (
        materialization_eligibility["eligible_participant_count"]
        != len(by_participant)
        or materialization_eligibility["participants_without_eligible_records"]
        or materialization_eligibility["participants_missing_static_records"]
        or materialization_eligibility["participants_missing_motion_records"]
    ):
        raise ValueError(
            "formal internal short-record exclusions break participant/class coverage"
        )
    recordings: list[MotionRecordingInput] = []
    calibration_hashes: dict[str, str] = {}
    calibration_records: dict[str, str] = {}
    lineage_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    physical_qc_evidence: dict[str, Mapping[str, Any]] = {}
    participant_rows_by_id = sorted(by_participant.items())
    for participant_index, (participant_id, participant_rows) in enumerate(
        participant_rows_by_id
    ):
        _notify_progress(
            progress_callback,
            participant_index,
            len(participant_rows_by_id),
            f"preprocess internal participant {participant_id}",
        )
        baseline_rows = [row for row in participant_rows if row.role == "B"]
        if len(baseline_rows) != 1:
            raise ValueError("formal internal participant must have exactly one B source")
        baseline = baseline_rows[0]
        baseline_values, baseline_qc = _load_internal_numeric_source(
            repository_root, baseline
        )
        physical_qc_evidence[baseline.record_id] = baseline_qc
        acceleration_unit, gyroscope_unit = _internal_source_units(baseline)
        calibration = fit_motion_imu_calibration(
            baseline_values[:, 2:5],
            baseline_values[:, 5:8],
            participant_id=participant_id,
            file_id=baseline.record_id,
            source_role="B",
            fs_hz=baseline.fs,
            acceleration_unit=acceleration_unit,
            gyroscope_unit=gyroscope_unit,
            config=config,
        )
        calibration_hashes[participant_id] = calibration.artifact_sha256
        calibration_records[participant_id] = baseline.record_id
        for row in sorted(participant_rows, key=lambda item: item.record_id):
            if row.record_id == baseline.record_id:
                values = baseline_values
            else:
                values, row_qc = _load_internal_numeric_source(repository_root, row)
                physical_qc_evidence[row.record_id] = row_qc
            if row.record_id in excluded_record_ids:
                continue
            acceleration_unit, gyroscope_unit = _internal_source_units(row)
            motion = preprocess_motion_imu_calibrated_ekf(
                values[:, 2:5],
                values[:, 5:8],
                fs_hz=row.fs,
                acceleration_unit=acceleration_unit,
                gyroscope_unit=gyroscope_unit,
                participant_id=participant_id,
                calibration=calibration,
                config=config,
            )
            lineage_hashes[row.record_id] = str(motion.diagnostics["lineage_sha256"])
            output_hashes[row.record_id] = str(
                motion.diagnostics["output_values_sha256"]
            )
            recordings.append(
                MotionRecordingInput(
                    ppg_red_ir=np.asarray(values[:, :2], dtype=np.float64),
                    motion_imu=motion,
                    record_id=row.record_id,
                    participant_id=participant_id,
                    role_or_activity=row.role,
                    dataset_id=M2_DATASET_VERSION_ID,
                    fs_hz=row.fs,
                )
            )
    _notify_progress(
        progress_callback,
        len(participant_rows_by_id),
        len(participant_rows_by_id),
        "completed internal participant preprocessing",
    )
    _notify_progress(
        progress_callback, 0, 1, f"materialize {len(recordings)} internal recordings"
    )
    examples = materialize_motion_window_examples(recordings, dataset_kind="internal")
    _notify_progress(progress_callback, 1, 1, "completed internal motion windows")
    materialized_record_ids = {example.file_id for example in examples}
    if (
        materialized_record_ids != eligible_record_ids
        or len(examples) != materialization_eligibility["expected_window_count"]
    ):
        raise ValueError("formal internal eligible-record materialization is incomplete")
    record_ids = {row.record_id for row in rows}
    role_counts = {
        role: sum(row.role == role for row in rows)
        for role in sorted({row.role for row in rows})
    }
    payload = {
        "schema_version": FORMAL_INTERNAL_SOURCE_EVIDENCE_SCHEMA,
        "formal_entry_id": FORMAL_INTERNAL_MOTION_ENTRY_ID,
        "repository_root": str(repository_root),
        "source_manifest_path": str((repository_root / M2_FILE_MANIFEST).resolve()),
        "source_manifest_sha256": M2_FILE_MANIFEST_SHA256,
        "dataset_version_id": M2_DATASET_VERSION_ID,
        "record_count": len(rows),
        "participant_count": len(by_participant),
        "role_counts": role_counts,
        "manifest_roster_sha256": _manifest_roster_sha256(rows),
        "record_source_sha256_by_id": {
            row.record_id: row.source_hash for row in sorted(rows, key=lambda item: item.record_id)
        },
        "physical_recording_qc_profile": physical_recording_qc_profile_v2(),
        "physical_recording_qc_evidence_by_id": physical_qc_evidence,
        "all_records_physical_qc_admitted": (
            len(physical_qc_evidence) == len(rows)
            and all(bool(item.get("admitted")) for item in physical_qc_evidence.values())
        ),
        "calibration_source_role": "same_participant_B_only",
        "calibration_source_record_id_by_participant": calibration_records,
        "calibration_artifact_sha256_by_participant": calibration_hashes,
        "record_ekf_lineage_sha256_by_id": lineage_hashes,
        "record_motion_values_sha256_by_id": output_hashes,
        "ekf_config_sha256": stable_payload_sha256(asdict(config)),
        "tensor_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
        "materialization_record_eligibility": materialization_eligibility,
        "materialized_record_count": len(materialized_record_ids),
        "materialized_window_count": len(examples),
        "materialized_window_values_sha256": _materialized_examples_sha256(examples),
        "source_loader_id": (
            "same_hashed_bytes_csv_header_shape_finite_manifest_physical_qc_"
            "complete_motion_windows_v3"
        ),
    }
    if (
        set(payload["record_source_sha256_by_id"]) != record_ids
        or set(lineage_hashes) != eligible_record_ids
        or set(output_hashes) != eligible_record_ids
    ):
        raise ValueError("formal internal record lineage coverage is incomplete")
    return examples, _seal_source_evidence(payload)


def verify_formal_internal_source_evidence(
    source_evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    """Re-hash the complete source roster and validate formal materialization lineage."""

    reasons: list[str] = []
    evidence = dict(source_evidence)
    expected_fields = {
        "schema_version",
        "formal_entry_id",
        "repository_root",
        "source_manifest_path",
        "source_manifest_sha256",
        "dataset_version_id",
        "record_count",
        "participant_count",
        "role_counts",
        "manifest_roster_sha256",
        "record_source_sha256_by_id",
        "physical_recording_qc_profile",
        "physical_recording_qc_evidence_by_id",
        "all_records_physical_qc_admitted",
        "calibration_source_role",
        "calibration_source_record_id_by_participant",
        "calibration_artifact_sha256_by_participant",
        "record_ekf_lineage_sha256_by_id",
        "record_motion_values_sha256_by_id",
        "ekf_config_sha256",
        "tensor_schema_sha256",
        "materialization_record_eligibility",
        "materialized_record_count",
        "materialized_window_count",
        "materialized_window_values_sha256",
        "source_loader_id",
        "source_evidence_sha256",
    }
    if set(evidence) != expected_fields:
        return ("formal_source_evidence_field_schema_drift",)
    sealed_hash = evidence.pop("source_evidence_sha256", "")
    if stable_payload_sha256(evidence) != sealed_hash:
        reasons.append("formal_source_evidence_hash_drift")
    if (
        evidence.get("schema_version") != FORMAL_INTERNAL_SOURCE_EVIDENCE_SCHEMA
        or evidence.get("formal_entry_id") != FORMAL_INTERNAL_MOTION_ENTRY_ID
        or evidence.get("source_manifest_sha256") != M2_FILE_MANIFEST_SHA256
        or evidence.get("dataset_version_id") != M2_DATASET_VERSION_ID
        or evidence.get("tensor_schema_sha256") != MOTION_NETWORK_SCHEMA_SHA256
        or evidence.get("calibration_source_role") != "same_participant_B_only"
        or evidence.get("source_loader_id")
        != (
            "same_hashed_bytes_csv_header_shape_finite_manifest_physical_qc_"
            "complete_motion_windows_v3"
        )
    ):
        reasons.append("formal_source_evidence_identity_drift")
    try:
        repository_root = Path(str(evidence["repository_root"])).resolve()
        rows = tuple(load_m2_internal_manifest(repository_root, verify_sources=True))
    except (KeyError, OSError, TypeError, ValueError):
        return tuple(dict.fromkeys([*reasons, "formal_source_roster_reload_failed"]))
    expected_manifest_path = str((repository_root / M2_FILE_MANIFEST).resolve())
    expected_sources = {row.record_id: row.source_hash for row in rows}
    participants = sorted({row.participant_id for row in rows})
    baseline_by_participant = {
        participant: next(
            row.record_id
            for row in rows
            if row.participant_id == participant and row.role == "B"
        )
        for participant in participants
    }
    expected_roles = {
        role: sum(row.role == role for row in rows)
        for role in sorted({row.role for row in rows})
    }
    expected_eligibility = _internal_materialization_eligibility(rows)
    excluded_record_ids = set(
        expected_eligibility["excluded_record_n_samples_by_id"]
    )
    eligible_record_ids = set(expected_sources) - excluded_record_ids
    if (
        evidence.get("source_manifest_path") != expected_manifest_path
        or evidence.get("record_count") != 261
        or evidence.get("participant_count") != 29
        or evidence.get("role_counts") != expected_roles
        or evidence.get("manifest_roster_sha256") != _manifest_roster_sha256(rows)
        or evidence.get("record_source_sha256_by_id") != expected_sources
        or evidence.get("calibration_source_record_id_by_participant")
        != baseline_by_participant
    ):
        reasons.append("formal_source_manifest_or_roster_drift")
    if (
        evidence.get("materialization_record_eligibility") != expected_eligibility
        or evidence.get("materialized_record_count") != len(eligible_record_ids)
        or evidence.get("materialized_window_count")
        != expected_eligibility["expected_window_count"]
    ):
        reasons.append("formal_source_materialization_eligibility_drift")
    observed_eligibility = evidence.get("materialization_record_eligibility")
    if (
        not isinstance(observed_eligibility, Mapping)
        or observed_eligibility.get("eligible_participant_count") != 29
        or observed_eligibility.get("participants_without_eligible_records") != []
        or observed_eligibility.get("participants_missing_static_records") != []
        or observed_eligibility.get("participants_missing_motion_records") != []
    ):
        reasons.append("formal_source_materialization_training_coverage_drift")
    physical_profile = evidence.get("physical_recording_qc_profile")
    physical_by_id = evidence.get("physical_recording_qc_evidence_by_id")
    expected_qc_fields = {
        "schema_version",
        "record_id",
        "manifest_version",
        "source_hash",
        "admitted",
        "status",
        "reasons",
        "metrics",
        "device_dependent_qc_status",
        "sqi_or_classifier_effect",
    }
    physical_qc_valid = (
        physical_profile == physical_recording_qc_profile_v2()
        and evidence.get("all_records_physical_qc_admitted") is True
        and isinstance(physical_by_id, Mapping)
        and set(physical_by_id) == set(expected_sources)
    )
    if physical_qc_valid:
        for record_id, source_hash in expected_sources.items():
            item = physical_by_id[record_id]
            if (
                not isinstance(item, Mapping)
                or set(item) != expected_qc_fields
                or item.get("schema_version")
                != "ppg_frailty.recording_qc_admission.v2"
                or item.get("record_id") != record_id
                or item.get("source_hash") != source_hash
                or item.get("admitted") is not True
                or item.get("status") != "pass"
                or item.get("reasons") != []
                or item.get("device_dependent_qc_status")
                != "deferred_missing_device_metadata"
                or item.get("sqi_or_classifier_effect")
                != "none_recording_safety_admission_only"
                or not isinstance(item.get("metrics"), Mapping)
                or item["metrics"].get("device_dependent_checks_executed")
                is not False
            ):
                physical_qc_valid = False
                break
    if not physical_qc_valid:
        reasons.append("formal_source_physical_qc_evidence_drift")
    hash_maps = (
        "calibration_artifact_sha256_by_participant",
        "record_ekf_lineage_sha256_by_id",
        "record_motion_values_sha256_by_id",
    )
    expected_keys = (set(participants), eligible_record_ids, eligible_record_ids)
    for name, keys in zip(hash_maps, expected_keys, strict=True):
        value = evidence.get(name)
        if (
            not isinstance(value, Mapping)
            or set(value) != keys
            or any(not _SHA256_PATTERN.fullmatch(str(item)) for item in value.values())
        ):
            reasons.append("formal_source_lineage_hash_coverage_drift")
    for name in (
        "ekf_config_sha256",
        "materialized_window_values_sha256",
    ):
        if not _SHA256_PATTERN.fullmatch(str(evidence.get(name, ""))):
            reasons.append("formal_source_materialization_hash_missing")
    if int(evidence.get("materialized_window_count", 0)) <= 0:
        reasons.append("formal_source_materialization_is_empty")
    return tuple(dict.fromkeys(reasons))


def run_formal_internal_motion_reference(
    repository_root: str | Path,
    *,
    output_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    training_device: str = "cuda",
) -> MotionInternalRunResult:
    """Canonical source-bound SGKF5 entry with no scientific injection API."""

    repository = Path(repository_root).resolve()
    trainer_config = FormalMotionTrainerConfig(device=str(training_device))
    # This preflight deliberately precedes source hashing/materialization so a
    # missing or invalid CUDA runtime fails in seconds, not after CPU-heavy
    # preprocessing of all 29 participants.
    require_formal_motion_cuda(trainer_config)
    config = RollPitchEkfConfig()
    examples, source_evidence = _build_internal_materialization(
        repository,
        config,
        progress_callback,
    )
    root = Path(output_dir).resolve()
    schema_path, schema_file_sha256 = write_formal_motion_input_schema(
        root / "formal_motion_input_schema.json"
    )
    split_path = (repository / MOTION_SPLIT_RELATIVE_PATH).resolve()
    jobs = load_motion_fold_jobs(split_path)
    return _run_internal_motion_oof_impl(
        examples,
        jobs,
        fit_model=partial(fit_formal_motion_model, config=trainer_config),
        predict_probability=predict_formal_motion_probability,
        model_input_schema_path=schema_path,
        expected_model_input_schema_sha256=schema_file_sha256,
        output_dir=root,
        motion_split_csv_path=split_path,
        execution_mode="formal",
        write_artifacts=True,
        formal_source_evidence=source_evidence,
        progress_callback=progress_callback,
    )


def _validate_ptt_unit_conflict(
    repository_root: Path,
    records: Sequence[ExternalRecord],
) -> dict[str, int]:
    statuses = {
        status: sum(row.imu_unit_status == status for row in records)
        for status in sorted({row.imu_unit_status for row in records})
    }
    if len(records) != 66 or statuses != {PTT_UNRESOLVED_IMU_UNIT_STATUS: 66}:
        raise ValueError("PTT IMU unit-conflict roster differs from frozen evidence")
    header_evidence = PTT_IMU_UNIT_CONFLICT_PROVENANCE["wfdb_header_declaration"]
    _, header_payload = _read_bound_source_bytes(
        repository_root,
        str(header_evidence["relative_path"]),
        str(header_evidence["sha256"]),
    )
    header_text = header_payload.decode("utf-8")
    if (
        not all(f" 0 {name}" in header_text for name in ("a_x", "a_y", "a_z"))
        or header_text.count("/g ") < 3
        or not all(f" 0 {name}" in header_text for name in ("g_x", "g_y", "g_z"))
        or header_text.count("/deg/s ") < 3
    ):
        raise ValueError("PTT WFDB unit declaration evidence drift")
    numeric_evidence = PTT_IMU_UNIT_CONFLICT_PROVENANCE[
        "canonical_csv_numeric_evidence"
    ]
    _, csv_payload = _read_bound_source_bytes(
        repository_root,
        str(numeric_evidence["relative_path"]),
        str(numeric_evidence["sha256"]),
    )
    with io.StringIO(csv_payload.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PTT_CSV_COLUMNS:
            raise ValueError("PTT representative CSV schema drift")
        first = next(reader)
    observed_acceleration = np.asarray(
        [float(first[name]) for name in ("a_x", "a_y", "a_z")],
        dtype=np.float64,
    )
    observed_gyroscope = np.asarray(
        [float(first[name]) for name in ("g_x", "g_y", "g_z")],
        dtype=np.float64,
    )
    expected_acceleration = np.asarray(
        numeric_evidence["first_acceleration_xyz"], dtype=np.float64
    )
    expected_gyroscope = np.asarray(
        numeric_evidence["first_gyroscope_xyz"], dtype=np.float64
    )
    acceleration_norm = float(numeric_evidence["first_acceleration_norm"])
    gyroscope_norm = float(numeric_evidence["first_gyroscope_norm"])
    if (
        not np.array_equal(observed_acceleration, expected_acceleration)
        or not np.array_equal(observed_gyroscope, expected_gyroscope)
        or not np.isclose(
            np.linalg.norm(observed_acceleration),
            acceleration_norm,
            rtol=0.0,
            atol=8.0 * abs(float(np.spacing(np.float64(acceleration_norm)))),
        )
        or not np.isclose(
            np.linalg.norm(observed_gyroscope),
            gyroscope_norm,
            rtol=0.0,
            atol=8.0 * abs(float(np.spacing(np.float64(gyroscope_norm)))),
        )
    ):
        raise ValueError("PTT hash-bound numeric unit-conflict evidence drift")
    historical = PTT_IMU_UNIT_CONFLICT_PROVENANCE["historical_code_transform"]
    _resolve_bound_source(
        repository_root,
        str(historical["relative_path"]),
        str(historical["sha256"]),
    )
    return statuses


def _load_ptt_numeric_source(
    repository_root: Path,
    row: ExternalRecord,
) -> Mapping[str, np.ndarray]:
    if (
        row.container_grid_fs_hz != 500.0
        or row.target_internal_fs_hz != 400.0
        or not row.resampling_required
    ):
        raise ValueError("formal PTT record is not declared synchronized 500-to-400")
    relative_path = (PTT_SOURCE_ROOT / row.canonical_representation).as_posix()
    _, payload = _read_bound_source_bytes(
        repository_root, relative_path, row.checksum_sha256
    )
    text = payload.decode("utf-8")
    header = tuple(next(csv.reader(io.StringIO(text, newline=""))))
    if header != PTT_CSV_COLUMNS:
        raise ValueError("formal PTT canonical CSV header drift")
    indices = tuple(header.index(name) for name in PTT_REQUIRED_COLUMNS)
    values = np.loadtxt(
        io.StringIO(text, newline=""),
        delimiter=",",
        skiprows=1,
        usecols=indices,
        dtype=np.float64,
        ndmin=2,
    )
    if (
        values.ndim != 2
        or values.shape[1] != len(PTT_REQUIRED_COLUMNS)
        or values.shape[0] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("formal PTT numeric source is empty, malformed, or non-finite")
    return {
        source_name: values[:, index]
        for index, source_name in enumerate(
            ("pleth_1", "pleth_2", "AX", "AY", "AZ", "GX", "GY", "GZ")
        )
    }


def _ptt_manifest_roster_sha256(records: Sequence[ExternalRecord]) -> str:
    return stable_payload_sha256(
        [
            {
                "record_id": row.record_id,
                "subject_id": row.subject_id,
                "activity": row.activity_raw,
                "canonical_representation": row.canonical_representation,
                "source_sha256": row.checksum_sha256,
                "imu_unit_status": row.imu_unit_status,
            }
            for row in sorted(records, key=lambda item: item.record_id)
        ]
    )


def _build_ptt_materialization(
    repository_root: Path,
    records: Sequence[ExternalRecord],
    unit_evidence: PttImuUnitEvidence,
    unit_evidence_path: Path,
    config: RollPitchEkfConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    unit_evidence.validate(records)
    expected_unit_evidence_path = (
        repository_root / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH
    ).resolve()
    if unit_evidence_path.resolve() != expected_unit_evidence_path:
        raise ValueError("formal PTT unit evidence path is not the canonical V2-036 artifact")
    by_subject: dict[str, list[ExternalRecord]] = {}
    for row in records:
        by_subject.setdefault(row.subject_id, []).append(row)
    if len(records) != 66 or len(by_subject) != 22:
        raise ValueError("formal PTT source roster is not the exact 66/22 snapshot")

    recordings: list[MotionRecordingInput] = []
    calibration_hashes: dict[str, str] = {}
    mapping_hashes: dict[str, str] = {}
    resampling_hashes: dict[str, str] = {}
    adapted_source_hashes: dict[str, str] = {}
    adapted_output_hashes: dict[str, str] = {}
    source_schema_hashes: dict[str, str] = {}
    target_schema_hashes: dict[str, str] = {}
    lineage_hashes: dict[str, str] = {}
    subject_rows_by_id = sorted(by_subject.items())
    for subject_index, (subject_id, subject_rows) in enumerate(subject_rows_by_id):
        _notify_progress(
            progress_callback,
            subject_index,
            len(subject_rows_by_id),
            f"preprocess PTT participant {subject_id}",
        )
        by_activity = {row.activity_raw.strip().lower(): row for row in subject_rows}
        if set(by_activity) != {"sit", "walk", "run"}:
            raise ValueError("formal PTT subject must have exactly sit/walk/run")
        adapted_by_activity: dict[str, Any] = {}
        for activity in ("sit", "walk", "run"):
            row = by_activity[activity]
            channels = _load_ptt_numeric_source(repository_root, row)
            adapted = adapt_ptt_synchronized_channels(
                channels,
                external_record=row,
                observed_source_file_sha256=row.checksum_sha256,
                additional_channel_order=("AX", "AY", "AZ", "GX", "GY", "GZ"),
            )
            adapted_by_activity[activity] = adapted
            mapping_hashes[row.record_id] = adapted.mapping_sha256
            resampling_hashes[row.record_id] = adapted.resampling_config_sha256
            adapted_source_hashes[row.record_id] = adapted.source_values_sha256
            adapted_output_hashes[row.record_id] = adapted.output_values_sha256
            source_schema_hashes[row.record_id] = (
                adapted.source_channel_schema_sha256
            )
            target_schema_hashes[row.record_id] = (
                adapted.target_channel_schema_sha256
            )
        sit_row = by_activity["sit"]
        sit = adapted_by_activity["sit"]
        calibration = fit_motion_imu_calibration(
            sit.values[:, 2:5],
            sit.values[:, 5:8],
            participant_id=subject_id,
            file_id=sit_row.record_id,
            source_role=PTT_STATIC_CALIBRATION_ROLE,
            fs_hz=400.0,
            acceleration_unit=unit_evidence.acceleration_unit,
            gyroscope_unit=unit_evidence.gyroscope_unit,
            config=config,
        )
        calibration_hashes[subject_id] = calibration.artifact_sha256
        for activity in ("sit", "walk", "run"):
            row = by_activity[activity]
            adapted = adapted_by_activity[activity]
            motion = preprocess_motion_imu_calibrated_ekf(
                adapted.values[:, 2:5],
                adapted.values[:, 5:8],
                fs_hz=400.0,
                acceleration_unit=unit_evidence.acceleration_unit,
                gyroscope_unit=unit_evidence.gyroscope_unit,
                participant_id=subject_id,
                calibration=calibration,
                config=config,
            )
            lineage_hashes[row.record_id] = str(motion.diagnostics["lineage_sha256"])
            recordings.append(
                MotionRecordingInput(
                    ppg_red_ir=adapted.ppg_red_ir,
                    motion_imu=motion,
                    record_id=row.record_id,
                    participant_id=subject_id,
                    role_or_activity=activity,
                    dataset_id=PTT_DATASET_ID,
                    fs_hz=400.0,
                )
            )
    _notify_progress(
        progress_callback,
        len(subject_rows_by_id),
        len(subject_rows_by_id),
        "completed PTT participant preprocessing",
    )
    _notify_progress(
        progress_callback, 0, 1, f"materialize {len(recordings)} PTT recordings"
    )
    examples = materialize_motion_window_examples(recordings, dataset_kind="ptt")
    _notify_progress(progress_callback, 1, 1, "completed PTT motion windows")
    source_hashes = {
        row.record_id: row.checksum_sha256
        for row in sorted(records, key=lambda item: item.record_id)
    }
    source_evidence = _seal_source_evidence(
        {
            "schema_version": "ppg_frailty.formal_ptt_motion_source_evidence.v2",
            "formal_entry_id": FORMAL_PTT_MOTION_ENTRY_ID,
            "repository_root": str(repository_root),
            "source_manifest_path": str(
                (repository_root / M2_EXTERNAL_RELATIVE_PATH).resolve()
            ),
            "source_manifest_sha256": M2_EXTERNAL_MANIFEST_SHA256,
            "dataset_id": PTT_DATASET_ID,
            "record_count": len(records),
            "participant_count": len(by_subject),
            "activity_counts": {
                activity: sum(row.activity_raw == activity for row in records)
                for activity in ("sit", "walk", "run")
            },
            "manifest_roster_sha256": _ptt_manifest_roster_sha256(records),
            "record_source_sha256_by_id": source_hashes,
            "distal_mapping_sha256_by_id": mapping_hashes,
            "resampling_config_sha256_by_id": resampling_hashes,
            "adapted_source_values_sha256_by_id": adapted_source_hashes,
            "adapted_output_values_sha256_by_id": adapted_output_hashes,
            "source_channel_schema_sha256_by_id": source_schema_hashes,
            "target_channel_schema_sha256_by_id": target_schema_hashes,
            "calibration_source_role": PTT_STATIC_CALIBRATION_ROLE,
            "calibration_artifact_sha256_by_participant": calibration_hashes,
            "record_ekf_lineage_sha256_by_id": lineage_hashes,
            "unit_evidence_artifact_sha256": unit_evidence.artifact_sha256,
            "unit_evidence_artifact_path": str(unit_evidence_path.resolve()),
            "unit_evidence_schema_version": unit_evidence.schema_version,
            "unit_evidence_decision_id": unit_evidence.decision_id,
            "unit_evidence_decision_date": unit_evidence.decision_date,
            "unit_evidence_decision_authority": unit_evidence.decision_authority,
            "unit_evidence_resolution_basis": unit_evidence.resolution_basis,
            "unit_evidence_acceleration_conversion": (
                unit_evidence.acceleration_conversion
            ),
            "unit_evidence_acceleration_decision_basis": (
                unit_evidence.acceleration_decision_basis
            ),
            "unit_evidence_gyroscope_conversion": unit_evidence.gyroscope_conversion,
            "unit_evidence_gyroscope_decision_basis": (
                unit_evidence.gyroscope_decision_basis
            ),
            "unit_evidence_wfdb_header_sha256": unit_evidence.wfdb_header_sha256,
            "unit_evidence_numeric_evidence_sha256": (
                unit_evidence.numeric_evidence_sha256
            ),
            "unit_evidence_historical_transform_sha256": (
                unit_evidence.historical_transform_sha256
            ),
            "acceleration_unit": unit_evidence.acceleration_unit,
            "gyroscope_unit": unit_evidence.gyroscope_unit,
            "ekf_config_sha256": stable_payload_sha256(asdict(config)),
            "tensor_schema_sha256": MOTION_NETWORK_SCHEMA_SHA256,
            "materialized_window_count": len(examples),
            "materialized_window_values_sha256": _materialized_examples_sha256(examples),
        }
    )
    return examples, source_evidence


def verify_formal_ptt_source_evidence(
    source_evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    """Re-hash the exact 66-record roster and validate all formal PTT lineage maps."""

    evidence = dict(source_evidence)
    expected_fields = {
        "schema_version",
        "formal_entry_id",
        "repository_root",
        "source_manifest_path",
        "source_manifest_sha256",
        "dataset_id",
        "record_count",
        "participant_count",
        "activity_counts",
        "manifest_roster_sha256",
        "record_source_sha256_by_id",
        "distal_mapping_sha256_by_id",
        "resampling_config_sha256_by_id",
        "adapted_source_values_sha256_by_id",
        "adapted_output_values_sha256_by_id",
        "source_channel_schema_sha256_by_id",
        "target_channel_schema_sha256_by_id",
        "calibration_source_role",
        "calibration_artifact_sha256_by_participant",
        "record_ekf_lineage_sha256_by_id",
        "unit_evidence_artifact_sha256",
        "unit_evidence_artifact_path",
        "unit_evidence_schema_version",
        "unit_evidence_decision_id",
        "unit_evidence_decision_date",
        "unit_evidence_decision_authority",
        "unit_evidence_resolution_basis",
        "unit_evidence_acceleration_conversion",
        "unit_evidence_acceleration_decision_basis",
        "unit_evidence_gyroscope_conversion",
        "unit_evidence_gyroscope_decision_basis",
        "unit_evidence_wfdb_header_sha256",
        "unit_evidence_numeric_evidence_sha256",
        "unit_evidence_historical_transform_sha256",
        "acceleration_unit",
        "gyroscope_unit",
        "ekf_config_sha256",
        "tensor_schema_sha256",
        "materialized_window_count",
        "materialized_window_values_sha256",
        "source_evidence_sha256",
    }
    if set(evidence) != expected_fields:
        return ("formal_ptt_source_evidence_field_schema_drift",)
    reasons: list[str] = []
    sealed_hash = evidence.pop("source_evidence_sha256", "")
    if stable_payload_sha256(evidence) != sealed_hash:
        reasons.append("formal_ptt_source_evidence_hash_drift")
    if (
        evidence.get("schema_version")
        != "ppg_frailty.formal_ptt_motion_source_evidence.v2"
        or evidence.get("formal_entry_id") != FORMAL_PTT_MOTION_ENTRY_ID
        or evidence.get("source_manifest_sha256") != M2_EXTERNAL_MANIFEST_SHA256
        or evidence.get("dataset_id") != PTT_DATASET_ID
        or evidence.get("record_count") != 66
        or evidence.get("participant_count") != 22
        or evidence.get("calibration_source_role") != PTT_STATIC_CALIBRATION_ROLE
        or evidence.get("tensor_schema_sha256") != MOTION_NETWORK_SCHEMA_SHA256
        or evidence.get("unit_evidence_artifact_sha256")
        != PTT_IMU_UNIT_EVIDENCE_SHA256
        or evidence.get("unit_evidence_schema_version")
        != PTT_IMU_UNIT_EVIDENCE_SCHEMA
        or evidence.get("unit_evidence_decision_id") != PTT_IMU_UNIT_DECISION_ID
        or evidence.get("unit_evidence_decision_date") != PTT_IMU_UNIT_DECISION_DATE
        or evidence.get("unit_evidence_decision_authority")
        != PTT_IMU_UNIT_DECISION_AUTHORITY
    ):
        reasons.append("formal_ptt_source_evidence_identity_drift")
    try:
        repository_root = Path(str(evidence["repository_root"])).resolve()
        records = tuple(
            row
            for row in load_m2_external_manifest(
                repository_root / M2_EXTERNAL_RELATIVE_PATH
            )
            if row.dataset_id == PTT_DATASET_ID
        )
        _validate_ptt_unit_conflict(repository_root, records)
    except (KeyError, OSError, TypeError, ValueError):
        return tuple(
            dict.fromkeys([*reasons, "formal_ptt_source_roster_reload_failed"])
        )
    record_ids = {row.record_id for row in records}
    participants = {row.subject_id for row in records}
    expected_source_hashes = {row.record_id: row.checksum_sha256 for row in records}
    if (
        evidence.get("source_manifest_path")
        != str((repository_root / M2_EXTERNAL_RELATIVE_PATH).resolve())
        or evidence.get("activity_counts") != {"sit": 22, "walk": 22, "run": 22}
        or evidence.get("manifest_roster_sha256") != _ptt_manifest_roster_sha256(records)
        or evidence.get("record_source_sha256_by_id") != expected_source_hashes
    ):
        reasons.append("formal_ptt_manifest_or_roster_drift")
    per_record_maps = (
        "distal_mapping_sha256_by_id",
        "resampling_config_sha256_by_id",
        "adapted_source_values_sha256_by_id",
        "adapted_output_values_sha256_by_id",
        "source_channel_schema_sha256_by_id",
        "target_channel_schema_sha256_by_id",
        "record_ekf_lineage_sha256_by_id",
    )
    for name in per_record_maps:
        value = evidence.get(name)
        if (
            not isinstance(value, Mapping)
            or set(value) != record_ids
            or any(not _SHA256_PATTERN.fullmatch(str(item)) for item in value.values())
        ):
            reasons.append("formal_ptt_record_lineage_hash_coverage_drift")
    expected_mapping_hash = stable_payload_sha256(
        dict(PTT_CHANNEL_MAPPING_PROVENANCE)
    )
    expected_resampling_hash = stable_payload_sha256(
        {
            "source_fs_hz": 500.0,
            "target_fs_hz": 400.0,
            "up": 4,
            "down": 5,
            "method": "scipy_signal_resample_poly_anti_alias_line_pad_v2",
            "axis": 0,
        }
    )
    expected_source_schema_hash = stable_payload_sha256(
        ["pleth_1", "pleth_2", "AX", "AY", "AZ", "GX", "GY", "GZ"]
    )
    expected_target_schema_hash = stable_payload_sha256(
        ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"]
    )
    mapping_values = evidence.get("distal_mapping_sha256_by_id")
    if (
        not isinstance(mapping_values, Mapping)
        or set(mapping_values.values()) != {expected_mapping_hash}
    ):
        reasons.append("formal_ptt_distal_mapping_hash_drift")
    resampling_values = evidence.get("resampling_config_sha256_by_id")
    if (
        not isinstance(resampling_values, Mapping)
        or set(resampling_values.values()) != {expected_resampling_hash}
    ):
        reasons.append("formal_ptt_resampling_hash_drift")
    source_schema_values = evidence.get("source_channel_schema_sha256_by_id")
    target_schema_values = evidence.get("target_channel_schema_sha256_by_id")
    if (
        not isinstance(source_schema_values, Mapping)
        or set(source_schema_values.values()) != {expected_source_schema_hash}
        or not isinstance(target_schema_values, Mapping)
        or set(target_schema_values.values()) != {expected_target_schema_hash}
    ):
        reasons.append("formal_ptt_channel_schema_hash_drift")
    calibration = evidence.get("calibration_artifact_sha256_by_participant")
    if (
        not isinstance(calibration, Mapping)
        or set(calibration) != participants
        or any(
            not _SHA256_PATTERN.fullmatch(str(item))
            for item in calibration.values()
        )
    ):
        reasons.append("formal_ptt_calibration_hash_coverage_drift")
    for name in (
        "unit_evidence_artifact_sha256",
        "unit_evidence_wfdb_header_sha256",
        "unit_evidence_numeric_evidence_sha256",
        "unit_evidence_historical_transform_sha256",
        "ekf_config_sha256",
        "materialized_window_values_sha256",
    ):
        if not _SHA256_PATTERN.fullmatch(str(evidence.get(name, ""))):
            reasons.append("formal_ptt_required_hash_missing")
    if (
        evidence.get("acceleration_unit") != PTT_ADOPTED_ACCELERATION_UNIT
        or evidence.get("unit_evidence_acceleration_conversion")
        != PTT_ADOPTED_ACCELERATION_CONVERSION
        or evidence.get("unit_evidence_acceleration_decision_basis")
        != PTT_ACCELERATION_DECISION_BASIS
    ):
        reasons.append("formal_ptt_resolved_acceleration_unit_invalid")
    if (
        evidence.get("gyroscope_unit") != PTT_ADOPTED_GYROSCOPE_UNIT
        or evidence.get("unit_evidence_gyroscope_conversion")
        != PTT_ADOPTED_GYROSCOPE_CONVERSION
        or evidence.get("unit_evidence_gyroscope_decision_basis")
        != PTT_GYROSCOPE_DECISION_BASIS
    ):
        reasons.append("formal_ptt_resolved_gyroscope_unit_invalid")
    expected_unit_evidence_path = (
        repository_root / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH
    ).resolve()
    if (
        Path(str(evidence.get("unit_evidence_artifact_path", ""))).resolve()
        != expected_unit_evidence_path
        or evidence.get("unit_evidence_resolution_basis")
        != PTT_IMU_UNIT_RESOLUTION_BASIS
    ):
        reasons.append("formal_ptt_unit_evidence_path_or_basis_drift")
    try:
        unit_evidence = load_ptt_imu_unit_evidence(
            Path(str(evidence["unit_evidence_artifact_path"])).resolve(),
            expected_sha256=str(evidence["unit_evidence_artifact_sha256"]),
            expected_records=records,
        )
    except (KeyError, OSError, TypeError, ValueError):
        reasons.append("formal_ptt_v2_036_unit_evidence_reload_failed")
    else:
        if (
            unit_evidence.schema_version
            != evidence.get("unit_evidence_schema_version")
            or unit_evidence.decision_id != evidence.get("unit_evidence_decision_id")
            or unit_evidence.decision_date
            != evidence.get("unit_evidence_decision_date")
            or unit_evidence.decision_authority
            != evidence.get("unit_evidence_decision_authority")
            or unit_evidence.resolution_basis
            != evidence.get("unit_evidence_resolution_basis")
            or unit_evidence.acceleration_conversion
            != evidence.get("unit_evidence_acceleration_conversion")
            or unit_evidence.acceleration_decision_basis
            != evidence.get("unit_evidence_acceleration_decision_basis")
            or unit_evidence.gyroscope_conversion
            != evidence.get("unit_evidence_gyroscope_conversion")
            or unit_evidence.gyroscope_decision_basis
            != evidence.get("unit_evidence_gyroscope_decision_basis")
            or unit_evidence.wfdb_header_sha256
            != evidence.get("unit_evidence_wfdb_header_sha256")
            or unit_evidence.numeric_evidence_sha256
            != evidence.get("unit_evidence_numeric_evidence_sha256")
            or unit_evidence.historical_transform_sha256
            != evidence.get("unit_evidence_historical_transform_sha256")
            or unit_evidence.acceleration_unit != evidence.get("acceleration_unit")
            or unit_evidence.gyroscope_unit != evidence.get("gyroscope_unit")
        ):
            reasons.append("formal_ptt_v2_036_unit_evidence_semantic_drift")
    if int(evidence.get("materialized_window_count", 0)) <= 0:
        reasons.append("formal_ptt_materialization_is_empty")
    return tuple(dict.fromkeys(reasons))


def _materialize_formal_ptt_source(
    repository: Path,
    *,
    unit_evidence_path: str | Path | None,
    expected_unit_evidence_sha256: str | None,
    progress_callback: ProgressCallback | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Load the single registered PTT source path used by both directions."""

    records = tuple(
        row
        for row in load_m2_external_manifest(repository / M2_EXTERNAL_RELATIVE_PATH)
        if row.dataset_id == PTT_DATASET_ID
    )
    statuses = _validate_ptt_unit_conflict(repository, records)
    if unit_evidence_path is None and expected_unit_evidence_sha256 is None:
        raise PttImuUnitEvidenceRequired(statuses)
    if unit_evidence_path is None or expected_unit_evidence_sha256 is None:
        raise ValueError("PTT unit evidence path and out-of-band SHA-256 are both required")
    canonical_unit_evidence_path = (
        repository / PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH
    ).resolve()
    if (
        Path(unit_evidence_path).resolve() != canonical_unit_evidence_path
        or expected_unit_evidence_sha256 != PTT_IMU_UNIT_EVIDENCE_SHA256
    ):
        raise ValueError("PTT formal entry requires the exact canonical V2-036 artifact")
    unit_evidence = load_ptt_imu_unit_evidence(
        canonical_unit_evidence_path,
        expected_sha256=expected_unit_evidence_sha256,
        expected_records=records,
    )
    return _build_ptt_materialization(
        repository,
        records,
        unit_evidence,
        canonical_unit_evidence_path,
        RollPitchEkfConfig(),
        progress_callback,
    )


def run_formal_ptt_motion_reference(
    repository_root: str | Path,
    *,
    internal_evidence_path: str | Path,
    expected_internal_evidence_sha256: str,
    output_dir: str | Path,
    unit_evidence_path: str | Path | None = None,
    expected_unit_evidence_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MotionExternalRunResult:
    """Canonical 66-record PTT entry using exact V2-036 unit evidence."""

    repository = Path(repository_root).resolve()
    examples, source_evidence = _materialize_formal_ptt_source(
        repository,
        unit_evidence_path=unit_evidence_path,
        expected_unit_evidence_sha256=expected_unit_evidence_sha256,
        progress_callback=progress_callback,
    )
    return _run_ptt_external_evaluation_impl(
        examples,
        internal_evidence_path=internal_evidence_path,
        expected_internal_evidence_sha256=expected_internal_evidence_sha256,
        ptt_split_csv=(repository / PTT_SPLIT_RELATIVE_PATH).resolve(),
        load_frozen_model=load_formal_motion_model,
        predict_probability=predict_formal_motion_probability,
        output_dir=output_dir,
        formal_source_evidence=source_evidence,
        progress_callback=progress_callback,
    )


def run_formal_ptt_motion_training_ablation(
    repository_root: str | Path,
    *,
    output_dir: str | Path,
    unit_evidence_path: str | Path | None = None,
    expected_unit_evidence_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
    training_device: str = "cuda",
) -> MotionPttTrainingRunResult:
    """Train the registered motion model on PTT repeat-0 SGKF5 plus all 22."""

    repository = Path(repository_root).resolve()
    trainer_config = FormalMotionTrainerConfig(device=str(training_device))
    require_formal_motion_cuda(trainer_config)
    examples, source_evidence = _materialize_formal_ptt_source(
        repository,
        unit_evidence_path=unit_evidence_path,
        expected_unit_evidence_sha256=expected_unit_evidence_sha256,
        progress_callback=progress_callback,
    )
    root = Path(output_dir).resolve()
    schema_path, schema_file_sha256 = write_formal_motion_input_schema(
        root / "formal_motion_input_schema.json"
    )
    return _run_ptt_motion_training_ablation_impl(
        examples,
        ptt_split_csv=(repository / PTT_SPLIT_RELATIVE_PATH).resolve(),
        fit_model=partial(fit_formal_motion_model, config=trainer_config),
        predict_probability=predict_formal_motion_probability,
        model_input_schema_path=schema_path,
        expected_model_input_schema_sha256=schema_file_sha256,
        output_dir=root,
        formal_source_evidence=source_evidence,
        progress_callback=progress_callback,
    )


def run_formal_internal_reverse_evaluation(
    repository_root: str | Path,
    *,
    ptt_training_evidence_path: str | Path,
    expected_ptt_training_evidence_sha256: str,
    output_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    runtime_device: str = "cuda",
) -> MotionExternalRunResult:
    """Evaluate the frozen PTT-trained model once on all Frailty29 windows."""

    repository = Path(repository_root).resolve()
    runtime_config = FormalMotionTrainerConfig(device=str(runtime_device))
    require_formal_motion_cuda(runtime_config)
    examples, source_evidence = _build_internal_materialization(
        repository,
        RollPitchEkfConfig(),
        progress_callback,
    )
    jobs = load_motion_fold_jobs((repository / MOTION_SPLIT_RELATIVE_PATH).resolve())
    load_model = partial(load_formal_motion_model, runtime_device=runtime_config.device)
    return _run_internal_reverse_evaluation_impl(
        examples,
        ptt_training_evidence_path=ptt_training_evidence_path,
        expected_ptt_training_evidence_sha256=expected_ptt_training_evidence_sha256,
        internal_fold_jobs=jobs,
        load_frozen_model=load_model,
        predict_probability=predict_formal_motion_probability,
        output_dir=output_dir,
        formal_source_evidence=source_evidence,
        progress_callback=progress_callback,
    )


__all__ = [
    "FORMAL_INTERNAL_MOTION_ENTRY_ID",
    "FORMAL_INTERNAL_SOURCE_EVIDENCE_SCHEMA",
    "FORMAL_PTT_MOTION_ENTRY_ID",
    "PTT_IMU_UNIT_EVIDENCE_SCHEMA",
    "PTT_UNRESOLVED_IMU_UNIT_STATUS",
    "PttImuUnitEvidence",
    "PttImuUnitEvidenceRequired",
    "load_ptt_imu_unit_evidence",
    "run_formal_internal_motion_reference",
    "run_formal_internal_reverse_evaluation",
    "run_formal_ptt_motion_training_ablation",
    "run_formal_ptt_motion_reference",
    "verify_formal_internal_source_evidence",
    "verify_formal_ptt_source_evidence",
]
