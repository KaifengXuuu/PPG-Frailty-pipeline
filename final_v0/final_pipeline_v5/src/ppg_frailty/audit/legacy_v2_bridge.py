"""Optional read-only Phase-0 audit for the historical-to-V2 bridge.

The auditor reads current source bytes, the materialized V2 manifest/split, the
independently maintained label table, and (when present) the exact historical
L0 cache.  It never edits a source, label, unit declaration, manifest, split,
or cache.  Its only writes are the nine protocol-declared audit artifacts.
The pipeline records the result as advisory evidence; no decision returned by
this module can enable, disable, or otherwise alter training execution.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from ppg_frailty.data.folds import (
    M2_SPLIT_FILE_SHA256,
    M2_SPLIT_PAYLOAD_SHA256,
    M2_SPLIT_REGISTRY_ID,
    FrozenFoldRegistry,
)
from ppg_frailty.data.manifest import load_internal_manifest
from ppg_frailty.data.schema import CANONICAL_CHANNEL_SCHEMA, CANONICAL_CLASS_NAMES
from ppg_frailty.legacy_bridge import build_legacy_bridge_raw_windows, resolve_legacy_bridge_profile
from ppg_frailty.provenance import sha256_file
from ppg_frailty.signal.imu import STANDARD_GRAVITY, convert_acceleration, convert_gyro
from ppg_frailty.signal.motion_imu import (
    RollPitchEkfConfig,
    fit_motion_imu_calibration,
    preprocess_motion_imu_calibrated_ekf,
)

PHASE0_RESULT_SCHEMA_VERSION = "ppg_frailty.legacy_v2_phase0_result.v1"
_REGISTERED_SOURCE_SPECIFICATION = (
    "AA_TODO/old_version_compare_V2/CODEX_LEGACY_V2_BRIDGE_REVISED_9_CASES_WITH_PHASE0.md")
_REGISTERED_SOURCE_SPECIFICATION_SHA256 = "7ad847a630f81e76304c8d7a38924a8c2b0ca9b16f9344488ade750cbee3c49b"
_EXPECTED_OUTPUTS = (
    "artifacts/audit/legacy_v2_manifest_record_diff.csv",
    "artifacts/audit/legacy_v2_source_hash_audit.csv",
    "artifacts/audit/legacy_v2_source_hash_audit.json",
    "artifacts/audit/legacy_v2_channel_qc.csv",
    "artifacts/audit/legacy_v2_participant_alias_map.csv",
    "artifacts/audit/legacy_v2_imu_unit_ekf_audit.csv",
    "artifacts/audit/legacy_v2_cache_audit.json",
    "artifacts/audit/legacy_v2_split_audit.json",
    "artifacts/audit/LEGACY_V2_PHASE0_DATA_AUDIT.md",
)
_TARGET_CACHE_NAME = "frailty3_cnn_windows_B_R1_R2_R3_R4_fs64_s15_h3_mf090.npz"
_STATIC_ROLE_RE = re.compile("^(?P<participant>.+)_(?P<role>B|R[1-4])$")
_ANY_ROLE_RE = re.compile("^(?P<participant>.+)_(?P<role>B|R[1-4]|S[12]|W[12])$")
_VISIT_SUFFIX_RE = re.compile("^(?P<alias>.+)_(?P<suffix>\\d{2})$")


@dataclass(frozen=True)
class Phase0Result:
    """Hash-bound advisory Phase-0 outcome."""

    decision: str
    advisory_checks_passed: bool
    stop_reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    outputs: Mapping[str, str]
    source_specification: str
    source_specification_sha256: str
    audit_spec_sha256: str
    manifest_sha256: str | None
    split_sha256: str | None
    schema_version: str = PHASE0_RESULT_SCHEMA_VERSION

    @property
    def phase0_spec_sha256(self) -> str:
        """Compatibility alias retained in existing audit artifacts."""
        return self.audit_spec_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "advisory_checks_passed": self.advisory_checks_passed,
            "stop_reasons": list(self.stop_reasons),
            "limitations": list(self.limitations),
            "outputs": dict(sorted(self.outputs.items())),
            "source_specification": self.source_specification,
            "source_specification_sha256": self.source_specification_sha256,
            "audit_spec_sha256": self.audit_spec_sha256,
            "phase0_spec_sha256": self.phase0_spec_sha256,
            "manifest_sha256": self.manifest_sha256,
            "split_sha256": self.split_sha256,
        }


@dataclass(frozen=True)
class _DiscoveryRow:
    source_path: str
    participant_id: str
    participant_alias: str
    removed_suffix: str
    role: str
    class_id: int
    class_name: str
    label_source_id: str
    cohort: str


def _bridge_field(bridge: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(bridge, Mapping):
        return bridge[name]
    return getattr(bridge, name)


def _strict_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve(strict=False)
    target.relative_to(root.resolve())
    return target


def _atomic_text(path: Path, text: str, root: Path) -> None:
    path.resolve(strict=False).relative_to(root.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Any, root: Path) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", root)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], root: Path) -> None:
    path.resolve(strict=False).relative_to(root.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _read_label_table(path: Path) -> tuple[dict[str, int], list[str]]:
    issues: list[str] = []
    mapping: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"ID", "FRAILTY-STATUS"}
            if not required <= set(reader.fieldnames or ()):
                return ({}, ["historical_label_table_columns_missing"])
            for line_number, row in enumerate(reader, start=2):
                label_id = str(row.get("ID", "")).strip()
                status_text = str(row.get("FRAILTY-STATUS", "")).strip()
                if not label_id or status_text not in {"2", "3"}:
                    continue
                class_id = 0 if status_text == "2" else 1
                previous = mapping.setdefault(label_id, class_id)
                if previous != class_id:
                    issues.append(f"label_conflict:{label_id}:line{line_number}")
    except OSError as exc:
        return ({}, [f"historical_label_table_unreadable:{type(exc).__name__}"])
    return (mapping, issues)


def _participant_parts(value: str) -> tuple[str, str]:
    match = _VISIT_SUFFIX_RE.fullmatch(value)
    if match is None:
        return (value, "")
    return (match.group("alias"), "_" + match.group("suffix"))


def _discover_static(repository_root: Path, data_root: Path,
                     labels: Mapping[str, int]) -> tuple[list[_DiscoveryRow], list[dict[str, Any]], list[str]]:
    discovered: list[_DiscoveryRow] = []
    issues: list[str] = []
    for directory_name, cohort in (("StudyData", "older"), ("TestDataYoungers", "young")):
        directory = data_root / directory_name
        if not directory.is_dir():
            issues.append(f"historical_discovery_directory_missing:{directory_name}")
            continue
        for path in sorted(directory.glob("*.csv"), key=lambda item: item.name.encode("utf-8")):
            match = _STATIC_ROLE_RE.fullmatch(path.stem)
            if match is None:
                continue
            participant = match.group("participant")
            if cohort == "young":
                # Young IDs keep their visit suffix; only the older label join
                # collapses the trailing two-digit visit suffix.
                alias, suffix = (participant, "")
                class_id = 2
                label_source_id = "TestDataYoungers"
            else:
                alias, suffix = _participant_parts(participant)
                label_source_id = alias
                if alias not in labels:
                    issues.append(f"historical_label_alias_missing:{participant}->{alias}")
                    class_id = -1
                else:
                    class_id = int(labels[alias])
            try:
                relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
            except ValueError:
                issues.append(f"historical_source_escapes_repository:{path}")
                continue
            discovered.append(
                _DiscoveryRow(
                    source_path=relative,
                    participant_id=participant,
                    participant_alias=alias,
                    removed_suffix=suffix,
                    role=match.group("role"),
                    class_id=class_id,
                    class_name=CANONICAL_CLASS_NAMES.get(class_id, "UNRESOLVED"),
                    label_source_id=label_source_id,
                    cohort=cohort,
                ))
    by_participant: dict[str, list[_DiscoveryRow]] = defaultdict(list)
    for row in discovered:
        by_participant[row.participant_id].append(row)
    alias_to_participants: dict[str, set[str]] = defaultdict(set)
    for participant, rows in by_participant.items():
        alias_to_participants[rows[0].participant_alias].add(participant)
    alias_rows: list[dict[str, Any]] = []
    for participant in sorted(by_participant):
        rows = by_participant[participant]
        aliases = {row.participant_alias for row in rows}
        roles = {row.role for row in rows}
        class_ids = {row.class_id for row in rows}
        alias = next(iter(aliases)) if len(aliases) == 1 else ""
        one_to_one = len(aliases) == 1 and len(alias_to_participants.get(alias, set())) == 1 and (len(class_ids) == 1)
        row_issues: list[str] = []
        if not one_to_one:
            row_issues.append("participant_alias_not_one_to_one")
        if roles != {"B", "R1", "R2", "R3", "R4"}:
            row_issues.append("static_role_coverage_mismatch")
        alias_rows.append({
            "historical_file_participant_id":
            participant,
            "historical_participant_alias":
            alias,
            "removed_suffix":
            rows[0].removed_suffix,
            "alias_rule":
            "remove_single_trailing_underscore_two_digit_visit_suffix",
            "cohort":
            rows[0].cohort,
            "label_source_id":
            rows[0].label_source_id,
            "class_id":
            next(iter(class_ids)) if len(class_ids) == 1 else "",
            "class_name":
            CANONICAL_CLASS_NAMES.get(next(iter(class_ids)), "") if len(class_ids) == 1 else "",
            "roles":
            ";".join(sorted(roles)),
            "one_to_one":
            str(bool(one_to_one)).lower(),
            "issues":
            ";".join(row_issues),
        })
    return (discovered, alias_rows, issues)


def _load_numeric_csv(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = tuple((str(value).strip() for value in next(csv.reader(handle))))
    try:
        matrix = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    except ValueError:
        matrix = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64, ndmin=2, invalid_raise=False)
    return (header, np.asarray(matrix, dtype=np.float64))


def _constant_run_stats(values: np.ndarray, minimum: int) -> tuple[int, int]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return (0, 0)
    finite = np.isfinite(data)
    equal = finite[1:] & finite[:-1] & (data[1:] == data[:-1])
    padded = np.concatenate((np.asarray([False]), equal, np.asarray([False])))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    # k equal adjacencies represent k + 1 constant samples.
    lengths = stops - starts + 1
    longest = int(np.max(lengths)) if lengths.size else 1 if finite.any() else 0
    return (int(np.sum(lengths >= int(minimum))), longest)


def _gap_counts(values: np.ndarray) -> tuple[int, int, int]:
    finite = np.isfinite(values)
    if not finite.any():
        return (int(values.size), 0, 0)
    first = int(np.flatnonzero(finite)[0])
    last = int(np.flatnonzero(finite)[-1])
    return (first, int(values.size - 1 - last), int(np.sum(~finite[first:last + 1])))


def _path_identity(path_text: str) -> tuple[str, str] | None:
    match = _ANY_ROLE_RE.fullmatch(Path(path_text).stem)
    if match is None:
        return None
    return (match.group("participant"), match.group("role"))


def _audit_sources(
    repository_root: Path,
    rows: Sequence[Any],
    independent_classes: Mapping[str, int],
    required_channels: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    source_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    stops: list[str] = []
    limitations: list[str] = ["sampling_rate_not_independently_observable_no_timestamp_column"]
    observed_hashes: list[str] = []
    for row in rows:
        issues: list[str] = []
        path = _strict_path(repository_root, row.source_path)
        observed_hash = ""
        observed_samples: int | str = ""
        header: tuple[str, ...] = ()
        matrix = np.empty((0, len(required_channels)), dtype=np.float64)
        if not path.is_file():
            issues.append("source_missing")
        else:
            observed_hash = sha256_file(path)
            observed_hashes.append(observed_hash)
            if observed_hash != row.source_hash:
                issues.append("source_hash_mismatch")
            try:
                header, matrix = _load_numeric_csv(path)
                observed_samples = int(matrix.shape[0])
            except Exception as exc:  # noqa: BLE001 - retain complete audit evidence.
                issues.append(f"source_csv_unreadable:{type(exc).__name__}")
        if header and header != required_channels:
            issues.append("required_channel_order_mismatch")
        if matrix.size and matrix.shape[1] != len(required_channels):
            issues.append("required_channel_count_mismatch")
        if observed_samples != "" and int(observed_samples) != int(row.n_samples):
            issues.append("n_samples_mismatch")
        duration_observed = float(observed_samples) / float(row.fs) if observed_samples != "" else None
        if duration_observed is not None and (not math.isclose(
                duration_observed, float(row.duration_s), abs_tol=max(1e-09, 0.51 / float(row.fs)))):
            issues.append("duration_mismatch")
        identity = _path_identity(row.source_path)
        if identity is None or identity != (row.participant_id, row.role):
            issues.append("participant_or_role_filename_conflict")
        independent_class = independent_classes.get(row.source_path)
        if independent_class is None or independent_class != row.class_id:
            issues.append("independent_class_conflict")
        source_rows.append({
            "record_id": row.record_id,
            "source_path": row.source_path,
            "expected_source_hash": row.source_hash,
            "observed_source_hash": observed_hash,
            "source_hash_match": str(observed_hash == row.source_hash).lower(),
            "expected_n_samples": row.n_samples,
            "observed_n_samples": observed_samples,
            "expected_duration_s": row.duration_s,
            "observed_duration_s": duration_observed if duration_observed is not None else "",
            "participant_id": row.participant_id,
            "role": row.role,
            "class_id": row.class_id,
            "sampling_rate_hz_declared": row.fs,
            "channel_order_observed": _json_cell(list(header)),
            "issues": ";".join(issues),
        })
        if matrix.shape[1] == len(required_channels):
            threshold = max(2, int(round(float(row.fs))))
            for index, channel in enumerate(required_channels):
                values = matrix[:, index]
                leading, trailing, internal = _gap_counts(values)
                run_count, longest = _constant_run_stats(values, threshold)
                channel_rows.append({
                    "record_id": row.record_id,
                    "source_path": row.source_path,
                    "channel_index": index,
                    "channel": channel,
                    "header_order_match": str(header == required_channels).lower(),
                    "n_samples": matrix.shape[0],
                    "nonfinite_count": int(np.sum(~np.isfinite(values))),
                    "all_missing": str(not np.isfinite(values).any()).lower(),
                    "leading_gap_samples": leading,
                    "trailing_gap_samples": trailing,
                    "internal_gap_samples": internal,
                    "constant_run_count_ge_1s": run_count,
                    "longest_constant_run_samples": longest,
                })
    if any(("source_hash_mismatch" in row["issues"] or "source_missing" in row["issues"] for row in source_rows)):
        stops.append("source_hash_mismatch_unresolved")
    if any(("required_channel" in row["issues"] or "source_csv_unreadable" in row["issues"] for row in source_rows)):
        stops.append("required_channel_missing_or_semantics_ambiguous")
    if any(("participant_or_role" in row["issues"] or "independent_class" in row["issues"] for row in source_rows)):
        stops.append("role_class_or_participant_identity_conflict")
    declared_hashes = [row.source_hash for row in rows]
    record_ids = [row.record_id for row in rows]
    identities = [(row.participant_id, row.role) for row in rows]
    if (len(set(declared_hashes)) != len(declared_hashes) or len(set(record_ids)) != len(record_ids)
            or len(set(identities)) != len(identities) or (len(set(observed_hashes)) != len(observed_hashes))):
        stops.append("duplicate_records_unresolved")
    return (source_rows, channel_rows, stops, limitations)


def _manifest_diff(
    discovered: Sequence[_DiscoveryRow],
    manifest_rows: Sequence[Any],
    observed_hash_by_path: Mapping[str, str],
    required_channels: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_static = {
        row.source_path: row
        for row in manifest_rows if row.role in {"B", "R1", "R2", "R3", "R4"} and (
            "/StudyData/" in f"/{row.source_path}" or "/TestDataYoungers/" in f"/{row.source_path}")
    }
    discovered_by_path = {row.source_path: row for row in discovered}
    output: list[dict[str, Any]] = []
    stops: list[str] = []
    for source_path in sorted(set(manifest_static) | set(discovered_by_path)):
        legacy = discovered_by_path.get(source_path)
        current = manifest_static.get(source_path)
        reasons: list[str] = []
        if legacy is None:
            reasons.append("missing_from_independent_historical_discovery")
        if current is None:
            reasons.append("missing_from_v2_static_manifest")
        if legacy is not None and current is not None:
            comparisons = {
                "participant": legacy.participant_id == current.participant_id,
                "role": legacy.role == current.role,
                "class": legacy.class_id == current.class_id,
                "source_hash": observed_hash_by_path.get(source_path) == current.source_hash,
                "channel_schema": tuple(current.channel_schema) == required_channels,
            }
            reasons.extend((f"{name}_mismatch" for name, matches in comparisons.items() if not matches))
        output.append({
            "legacy_source_path": source_path if legacy else "",
            "v2_record_id": current.record_id if current else "",
            "legacy_file_participant_id": legacy.participant_id if legacy else "",
            "legacy_participant_alias": legacy.participant_alias if legacy else "",
            "v2_participant_id": current.participant_id if current else "",
            "role": legacy.role if legacy else current.role if current else "",
            "legacy_class_id": legacy.class_id if legacy else "",
            "v2_class_id": current.class_id if current else "",
            "legacy_class_name": legacy.class_name if legacy else "",
            "v2_class_name": current.class_name if current else "",
            "observed_source_hash": observed_hash_by_path.get(source_path, ""),
            "v2_source_hash": current.source_hash if current else "",
            "v2_n_samples": current.n_samples if current else "",
            "v2_channel_schema": _json_cell(list(current.channel_schema)) if current else "",
            "status": "MATCH" if not reasons else "MISMATCH",
            "reasons": ";".join(reasons),
        })
    participants = {row.participant_id: row.class_id for row in discovered}
    class_counts = Counter(participants.values())
    expected = (len(discovered) == 145 and len(participants) == 29 and (class_counts == Counter({
        0: 9,
        1: 12,
        2: 8
    })) and all((row["status"] == "MATCH" for row in output)))
    if not expected:
        stops.append("historical_static_set_differs_from_v2")
    return (output, stops)


def _finite_stats(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {name: None for name in ("median", "iqr", "p01", "p99", "rms")}
    q01, q25, q75, q99 = np.percentile(finite, [1, 25, 75, 99])
    return {
        "median": float(np.median(finite)),
        "iqr": float(q75 - q25),
        "p01": float(q01),
        "p99": float(q99),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
    }


def _declared_unit(row: Any, channels: Sequence[str], expected: str) -> bool:
    return all((str(row.channel_units.get(channel, "")) == expected for channel in channels))


def _audit_imu(repository_root: Path, rows: Sequence[Any],
               config: RollPitchEkfConfig) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output: list[dict[str, Any]] = []
    stops: list[str] = []
    limitations: list[str] = []
    for row in sorted((item for item in rows if item.role == "B"), key=lambda item: item.participant_id):
        red_flags: list[str] = []
        failure = ""
        result_row: dict[str, Any] = {
            "participant_id": row.participant_id,
            "record_id": row.record_id,
            "source_path": row.source_path,
            "declared_acceleration_unit": "g",
            "declared_gyroscope_unit": "deg/s",
            "conversion_policy": "g_to_m_per_s2_and_deg_per_s_to_rad_per_s",
        }
        try:
            header, matrix = _load_numeric_csv(_strict_path(repository_root, row.source_path))
            if header != CANONICAL_CHANNEL_SCHEMA or matrix.shape[1] != 8:
                raise ValueError("required_channel_order_mismatch")
            if not _declared_unit(row, ("AX", "AY", "AZ"), "g_source_declared"):
                raise ValueError("acceleration_unit_semantics_ambiguous")
            if not _declared_unit(row, ("GX", "GY", "GZ"), "degree_per_second_source_declared"):
                raise ValueError("gyroscope_unit_semantics_ambiguous")
            acc_raw = matrix[:, 2:5]
            gyro_raw = matrix[:, 5:8]
            acc_norm_raw = np.linalg.norm(acc_raw, axis=1)
            raw_acc = _finite_stats(acc_norm_raw)
            result_row.update({f"raw_acc_norm_{key}": value for key, value in raw_acc.items()})
            for axis, values in zip(("gx", "gy", "gz"), gyro_raw.T):
                stats = _finite_stats(values)
                result_row.update({f"raw_{axis}_{key}": value for key, value in stats.items() if key != "iqr"})
            result_row["raw_imu_nonfinite_count"] = int(np.sum(~np.isfinite(np.column_stack((acc_raw, gyro_raw)))))
            result_row["raw_imu_constant_run_count_ge_1s"] = sum(
                (_constant_run_stats(values, max(2, int(round(row.fs))))[0]
                 for values in np.column_stack((acc_raw, gyro_raw)).T))
            raw_median = raw_acc["median"]
            if raw_median is not None and 4.0 <= raw_median <= 15.0:
                red_flags.append("raw_acc_norm_near_si_while_manifest_declares_g")
            acc_si = convert_acceleration(acc_raw, "g")
            gyro_si = convert_gyro(gyro_raw, "deg/s")
            converted_acc = _finite_stats(np.linalg.norm(acc_si, axis=1))
            result_row.update({f"converted_acc_norm_{key}": value for key, value in converted_acc.items()})
            converted_median = converted_acc["median"]
            if converted_median is None or not 0.5 * STANDARD_GRAVITY <= converted_median <= 1.5 * STANDARD_GRAVITY:
                red_flags.append("converted_acc_norm_materially_inconsistent_with_gravity")
            for axis, values in zip(("gx", "gy", "gz"), gyro_si.T):
                stats = _finite_stats(values)
                result_row.update({f"converted_{axis}_{key}": value for key, value in stats.items() if key != "iqr"})
            calibration = fit_motion_imu_calibration(
                acc_raw,
                gyro_raw,
                participant_id=row.participant_id,
                file_id=row.record_id,
                source_role="B",
                fs_hz=row.fs,
                acceleration_unit="g",
                gyroscope_unit="deg/s",
                config=config,
            )
            processed = preprocess_motion_imu_calibrated_ekf(
                acc_raw,
                gyro_raw,
                fs_hz=row.fs,
                acceleration_unit="g",
                gyroscope_unit="deg/s",
                participant_id=row.participant_id,
                calibration=calibration,
                config=config,
            )
            gravity_stats = _finite_stats(np.linalg.norm(processed.gravity_mps2, axis=1))
            dynamic = np.asarray(processed.values[:, :3], dtype=np.float64)
            result_row.update({
                "calibration_file_id":
                calibration.file_id,
                "calibration_artifact_sha256":
                calibration.artifact_sha256,
                "acceleration_bias_mps2":
                _json_cell(calibration.acceleration_bias_mps2.tolist()),
                "gyroscope_bias_rads":
                _json_cell(calibration.gyroscope_bias_rads.tolist()),
                "initial_roll_rad":
                calibration.initial_roll_rad,
                "initial_pitch_rad":
                calibration.initial_pitch_rad,
                "gravity_norm_median_mps2":
                gravity_stats["median"],
                "gravity_norm_iqr_mps2":
                gravity_stats["iqr"],
                "dynamic_acc_mean_mps2":
                _json_cell(np.mean(dynamic, axis=0).tolist()),
                "dynamic_acc_rms_mps2":
                _json_cell(np.sqrt(np.mean(np.square(dynamic), axis=0)).tolist()),
                "dynamic_acc_p99_abs_mps2":
                _json_cell(np.percentile(np.abs(dynamic), 99, axis=0).tolist()),
                "ekf_status":
                "PASS",
                "ekf_failure_reason":
                "",
                "ekf_lineage_sha256":
                processed.diagnostics.get("lineage_sha256", ""),
            })
            if float(np.percentile(np.linalg.norm(dynamic, axis=1), 99)) > 5.0:
                red_flags.append("large_residual_dynamic_acceleration_on_B")
        except Exception as exc:  # noqa: BLE001 - one B file must not suppress the others.
            failure = f"{type(exc).__name__}:{exc}"
            result_row.update({"ekf_status": "FAIL", "ekf_failure_reason": failure})
            red_flags.append("imu_calibration_or_ekf_failure")
        result_row["red_flags"] = _json_cell(red_flags)
        output.append(result_row)
        if failure or any((flag in {
                "raw_acc_norm_near_si_while_manifest_declares_g",
                "converted_acc_norm_materially_inconsistent_with_gravity",
        } for flag in red_flags)):
            stops.append("required_channel_missing_or_semantics_ambiguous")
        elif red_flags:
            limitations.extend((f"imu_red_flag:{row.record_id}:{flag}" for flag in red_flags))
    return (output, stops, limitations)


def _fresh_legacy_windows(path: Path) -> np.ndarray:
    header, matrix = _load_numeric_csv(path)
    if header != CANONICAL_CHANNEL_SCHEMA or not np.isfinite(matrix).all():
        raise ValueError("fresh_legacy_materialization_requires_finite_canonical_csv")
    windows = build_legacy_bridge_raw_windows(
        {
            "fs_hz": 400.0,
            "ppg": matrix[:, :2],
            "acc": matrix[:, 2:5],
            "gyro": matrix[:, 5:8]
        },
        resolve_legacy_bridge_profile("L0"),
    )
    return np.asarray(windows.values, dtype="<f4")


def _npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                reader = (np.lib.format.read_array_header_1_0
                          if version == (1, 0) else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(handle)
            headers[member[:-4]] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran)}
    return headers


def _selected_npz_rows(path: Path, wanted_positions: Mapping[int, np.ndarray]) -> tuple[dict[int, np.ndarray], str]:
    """Stream ``x.npy`` once and retain only selected file rows.

    This avoids materialising the roughly 390 MB historical tensor merely to
    compare the six protocol-required B/R4 representatives.
    """
    with zipfile.ZipFile(path) as archive, archive.open("x.npy") as handle:
        version = np.lib.format.read_magic(handle)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(handle)
        if fortran or len(shape) != 3:
            raise ValueError("historical x array must be C-order windows-by-channels-by-samples")
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * int(dtype.itemsize)
        position_lookup: dict[int, tuple[int, int]] = {}
        selected: dict[int, np.ndarray] = {}
        for file_index, positions_raw in wanted_positions.items():
            positions = np.asarray(positions_raw, dtype=np.int64)
            selected[file_index] = np.empty((len(positions), *shape[1:]), dtype=dtype)
            for local_index, position in enumerate(positions.tolist()):
                if position in position_lookup:
                    raise ValueError("selected cache row belongs to multiple files")
                position_lookup[position] = (file_index, local_index)
        digest = hashlib.sha256()
        for row_index in range(int(shape[0])):
            payload = handle.read(row_bytes)
            if len(payload) != row_bytes:
                raise ValueError("historical x array payload is truncated")
            digest.update(payload)
            target = position_lookup.get(row_index)
            if target is not None:
                file_index, local_index = target
                selected[file_index][local_index] = np.frombuffer(payload, dtype=dtype).reshape(shape[1:])
        if handle.read(1):
            raise ValueError("historical x array contains unexpected trailing payload")
    return (selected, digest.hexdigest())


def _audit_cache(repository_root: Path, cache_roots: Sequence[Path],
                 discovered: Sequence[_DiscoveryRow]) -> tuple[dict[str, Any], list[str]]:
    candidates: set[Path] = set()
    for root in cache_roots:
        if root.is_file() and root.name == _TARGET_CACHE_NAME:
            candidates.add(root.resolve())
        elif root.is_dir():
            candidates.update((path.resolve() for path in root.rglob(_TARGET_CACHE_NAME)))
    if not candidates:
        return (
            {
                "schema_version": "ppg_frailty.legacy_v2_cache_audit.v1",
                "status": "historical_cache_not_available",
                "target_cache_name": _TARGET_CACHE_NAME,
                "training_use": "forbidden_audit_only",
                "cache_files": [],
            },
            ["historical_cache_not_available"],
        )
    discovery_by_path = {row.source_path: row for row in discovered}
    cache_rows: list[dict[str, Any]] = []
    limitations: list[str] = []
    for path in sorted(candidates):
        try:
            relative = path.relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        entry: dict[str, Any] = {
            "path": relative,
            "sha256": sha256_file(path),
            "array_headers": _npz_headers(path),
            "training_use": "forbidden_audit_only",
        }
        try:
            with np.load(path, allow_pickle=False) as cached:
                names = set(cached.files)
                entry["array_names"] = sorted(names)
                required = {"x", "y", "subjects", "file_index", "paths"}
                if not required <= names:
                    entry["status"] = "cache_schema_incomplete"
                    entry["missing_arrays"] = sorted(required - names)
                    limitations.append(f"historical_cache_schema_incomplete:{relative}")
                    cache_rows.append(entry)
                    continue
                paths = cached["paths"].astype(str)
                labels = cached["y"].astype(np.int64)
                subjects = cached["subjects"].astype(str)
                file_index = cached["file_index"].astype(np.int64)
            metadata_ok = (len(paths) == 145 and set(paths.tolist()) == set(discovery_by_path)
                           and (labels.shape == subjects.shape == file_index.shape)
                           and np.all((file_index >= 0) & (file_index < len(paths))))
            metadata_issues: list[str] = []
            stored_file_metadata: list[dict[str, Any]] = []
            if metadata_ok:
                for index, source_path in enumerate(paths.tolist()):
                    discovered_row = discovery_by_path[source_path]
                    mask = file_index == index
                    cached_labels = sorted((int(value) for value in np.unique(labels[mask]).tolist()))
                    cached_participants = sorted((str(value) for value in np.unique(subjects[mask]).tolist()))
                    identity = _path_identity(source_path)
                    stored_file_metadata.append({
                        "file_index": int(index),
                        "source_path": source_path,
                        "participant_id_from_path": "" if identity is None else identity[0],
                        "role_from_path": "" if identity is None else identity[1],
                        "stored_participant_ids": cached_participants,
                        "stored_class_ids": cached_labels,
                        "stored_window_count": int(np.sum(mask)),
                    })
                    if not np.all(labels[mask] == discovered_row.class_id):
                        metadata_issues.append(f"label_mismatch:{source_path}")
                    if not np.all(subjects[mask] == discovered_row.participant_alias):
                        metadata_issues.append(f"participant_alias_mismatch:{source_path}")
            else:
                metadata_issues.append("path_or_row_identity_mismatch")
            entry["metadata_issues"] = metadata_issues
            entry["stored_file_metadata"] = stored_file_metadata
            entry["metadata_array_payload_sha256"] = {
                "paths": hashlib.sha256(np.ascontiguousarray(paths).tobytes(order="C")).hexdigest(),
                "labels": hashlib.sha256(np.ascontiguousarray(labels).tobytes(order="C")).hexdigest(),
                "participants": hashlib.sha256(np.ascontiguousarray(subjects).tobytes(order="C")).hexdigest(),
                "file_index": hashlib.sha256(np.ascontiguousarray(file_index).tobytes(order="C")).hexdigest(),
            }
            entry["stored_path_count"] = int(len(paths))
            entry["stored_window_count"] = int(len(labels))
            entry["stored_class_counts"] = {
                str(key): int(value)
                for key, value in sorted(Counter(labels.tolist()).items())
            }
            provenance_names = {"source_hashes", "preprocessing_hash", "schema_hash", "code_version", "provenance"}
            entry["required_provenance_present"] = sorted(names & provenance_names)
            entry["byte_equivalence_proven"] = False
            selected_indices: list[int] = []
            selected_participant_by_class: dict[int, str] = {}
            for row in discovered:
                selected_participant_by_class.setdefault(row.class_id, row.participant_id)
            for index, source_path in enumerate(paths.tolist()):
                row = discovery_by_path.get(source_path)
                if (row is None or row.participant_id != selected_participant_by_class.get(row.class_id)
                        or row.role not in {"B", "R4"}):
                    continue
                selected_indices.append(index)
            wanted = {index: np.flatnonzero(file_index == index) for index in selected_indices}
            selected_cached, cached_x_sha = _selected_npz_rows(path, wanted)
            entry["cached_x_payload_sha256"] = cached_x_sha
            selected: list[dict[str, Any]] = []
            for index in selected_indices:
                source_path = paths[index]
                row = discovery_by_path[source_path]
                fresh = _fresh_legacy_windows(_strict_path(repository_root, source_path))
                cached_values = np.asarray(selected_cached[index])
                comparison: dict[str, Any] = {
                    "source_path": source_path,
                    "role": row.role,
                    "class_id": row.class_id,
                    "fresh_shape": list(fresh.shape),
                    "cached_shape": list(cached_values.shape),
                    "cached_row_count": int(len(wanted[index])),
                    "fresh_sha256": hashlib.sha256(fresh.tobytes(order="C")).hexdigest(),
                    "cached_sha256": hashlib.sha256(np.ascontiguousarray(cached_values).tobytes(order="C")).hexdigest(),
                }
                if fresh.shape == cached_values.shape:
                    difference = np.abs(fresh.astype(np.float64) - cached_values.astype(np.float64))
                    comparison.update({
                        "max_absolute_difference": float(np.max(difference)),
                        "mean_absolute_difference": float(np.mean(difference)),
                        "numeric_equal": bool(np.array_equal(fresh, cached_values)),
                    })
                else:
                    comparison.update({
                        "max_absolute_difference": None,
                        "mean_absolute_difference": None,
                        "numeric_equal": False
                    })
                selected.append(comparison)
            entry["selected_fresh_B_R4_regeneration"] = selected
            entry["status"] = ("historical_cache_untraceable_to_current_source_bytes"
                               if not entry["required_provenance_present"] else
                               "historical_cache_provenance_present_numeric_comparison_required")
            limitations.append(f"historical_cache_byte_equivalence_unproven:{relative}")
        except Exception as exc:  # noqa: BLE001 - cache is audit-only, never a training source.
            entry["status"] = "historical_cache_inspection_failed"
            entry["error"] = f"{type(exc).__name__}:{exc}"
            limitations.append(f"historical_cache_inspection_failed:{relative}")
        cache_rows.append(entry)
    return (
        {
            "schema_version": "ppg_frailty.legacy_v2_cache_audit.v1",
            "status": "audit_complete_training_source_forbidden",
            "target_cache_name": _TARGET_CACHE_NAME,
            "training_use": "forbidden_audit_only",
            "cache_files": cache_rows,
        },
        limitations,
    )


def _audit_split(split_path: Path, manifest_rows: Sequence[Any]) -> tuple[dict[str, Any], list[str]]:
    stops: list[str] = []
    result: dict[str, Any] = {
        "schema_version": "ppg_frailty.legacy_v2_split_audit.v1",
        "split_path": split_path.as_posix(),
        "split_sha256": sha256_file(split_path) if split_path.is_file() else None,
        "runtime_split_recomputation": False,
    }
    try:
        registry = FrozenFoldRegistry.from_csv(split_path)
        repeat0 = [row for row in registry.assignments if row.repeat_index == 0]
        participants = {row.participant_id: row.class_id for row in manifest_rows}
        held_counts = Counter((row.participant_id for row in repeat0))
        folds = sorted({row.fold_index for row in repeat0})
        declarations = {
            "registry_id": sorted({row.source_registry_id
                                   for row in repeat0}),
            "file_sha256": sorted({row.source_registry_file_sha256
                                   for row in repeat0}),
            "payload_sha256": sorted({row.source_registry_payload_sha256
                                      for row in repeat0}),
        }
        class_consistent = all((participants.get(row.participant_id) == row.class_id for row in repeat0))
        exact_once = set(held_counts.values()) == {1} and set(held_counts) == set(participants)
        disjoint = True
        fold_rows: list[dict[str, Any]] = []
        for fold in range(5):
            split = registry.get_split(0, fold)
            train = set(split["train_participant_ids"])
            oof = set(split["oof_participant_ids"])
            disjoint &= not bool(train & oof) and train | oof == set(participants)
            fold_rows.append({
                "fold_index": fold,
                "train_participant_count": len(train),
                "held_out_participant_count": len(oof),
                "held_out_participants": sorted(oof),
                "train_oof_disjoint": not bool(train & oof),
            })
        declarations_match = declarations == {
            "registry_id": [M2_SPLIT_REGISTRY_ID],
            "file_sha256": [M2_SPLIT_FILE_SHA256],
            "payload_sha256": [M2_SPLIT_PAYLOAD_SHA256],
        }
        passed = (len(repeat0) == 29 and folds == list(range(5)) and exact_once and disjoint and class_consistent
                  and declarations_match)
        result.update({
            "repeat0_assignment_count": len(repeat0),
            "fold_indices": folds,
            "held_out_exactly_once": exact_once,
            "train_oof_disjoint": disjoint,
            "class_consistent_with_manifest": class_consistent,
            "all_manifest_files_inherit_participant_fold": exact_once and class_consistent,
            "declared_registry": declarations,
            "declared_registry_hashes_match_v2_authority": declarations_match,
            "folds": fold_rows,
            "status": "PASS" if passed else "STOP",
        })
        if not passed:
            stops.append("participant_leakage_or_split_registry_mismatch")
    except Exception as exc:  # noqa: BLE001 - convert split failures into STOP evidence.
        result.update({"status": "STOP", "error": f"{type(exc).__name__}:{exc}"})
        stops.append("participant_leakage_or_split_registry_mismatch")
    return (result, stops)


def _decision(stops: Iterable[str], limitations: Iterable[str]) -> tuple[str, bool, tuple[str, ...], tuple[str, ...]]:
    stop_values = tuple(sorted(set((str(value) for value in stops if str(value)))))
    limitation_values = tuple(sorted(set((str(value) for value in limitations if str(value)))))
    if stop_values:
        return ("STOP", False, stop_values, limitation_values)
    if limitation_values:
        return ("PASS_WITH_DECLARED_LIMITATIONS", True, (), limitation_values)
    return ("PASS", True, (), ())


def _markdown_summary(
    result: Phase0Result,
    manifest_diff: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    imu_rows: Sequence[Mapping[str, Any]],
    cache_audit: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> str:
    mismatches = [row for row in manifest_diff if row.get("status") != "MATCH"]
    source_mismatches = [row for row in source_rows if row.get("issues")]
    imu_flags = [row for row in imu_rows if row.get("red_flags") not in {"[]", ""}]
    cache_files = list(cache_audit.get("cache_files", ()))
    if not cache_files:
        cache_traceability = "historical_cache_not_available"
    elif all((bool(row.get("byte_equivalence_proven")) for row in cache_files)):
        cache_traceability = "byte_equivalence_to_current_sources_proven"
    else:
        cache_traceability = "cannot_be_traced_byte_for_byte_to_current_sources"
    static_set_identical = len(manifest_diff) == 145 and (not mismatches)
    current_source_bytes_match = bool(source_rows) and (not source_mismatches)
    lines = [
        "# Legacy-to-V2 Phase 0 Data Audit",
        "",
        f"- Decision: **{result.decision}**",
        f"- Advisory checks passed: **{str(result.advisory_checks_passed).lower()}**",
        f"- Source specification: `{result.source_specification}`",
        f"- Source specification SHA-256: `{result.source_specification_sha256}`",
        f"- Audit-spec SHA-256: `{result.audit_spec_sha256}`",
        f"- Manifest SHA-256: `{result.manifest_sha256}`",
        f"- Split CSV SHA-256: `{result.split_sha256}`",
        "- Mutation policy: report-only; no unit, label, source, manifest, split, or cache correction was performed.",
        "- Training input policy: fresh current raw CSV bytes only; historical cache use is forbidden.",
        "",
        "## Identity summary",
        "",
        f"- Historical/V2 static rows compared: {len(manifest_diff)}",
        f"- Matched rows: {len(manifest_diff) - len(mismatches)}",
        f"- Mismatched rows: {len(mismatches)}",
        f"- Static 145-record set identical: **{str(static_set_identical).lower()}**",
        f"- Current source rows re-hashed: {len(source_rows)}",
        f"- Current source mismatches: {len(source_mismatches)}",
        f"- Current source bytes match V2 manifest: **{str(current_source_bytes_match).lower()}**",
        f"- B-record IMU/EKF rows: {len(imu_rows)}",
        f"- IMU rows with red flags: {len(imu_flags)}",
        f"- Historical cache status: `{cache_audit.get('status')}`",
        f"- Historical cache traceability: **{cache_traceability}**",
        f"- Frozen split status: `{split_audit.get('status')}`",
        "",
        "## STOP reasons",
        "",
    ]
    lines.extend((f"- `{value}`" for value in result.stop_reasons))
    if not result.stop_reasons:
        lines.append("- None")
    lines.extend(["", "## Declared limitations", ""])
    lines.extend((f"- `{value}`" for value in result.limitations))
    if not result.limitations:
        lines.append("- None")
    if mismatches:
        lines.extend(["", "## Static-set mismatch details", ""])
        lines.extend((f"- `{row.get('legacy_source_path') or row.get('v2_record_id')}`: {row.get('reasons')}"
                      for row in mismatches))
    if source_mismatches:
        lines.extend(["", "## Current source mismatch details", ""])
        lines.extend((f"- `{row.get('record_id')}`: {row.get('issues')}" for row in source_mismatches))
    lines.extend(["", "## IMU unit / calibration / EKF red flags", ""])
    if imu_flags:
        lines.extend((f"- `{row.get('record_id')}`: {row.get('red_flags')}" for row in imu_flags))
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def run_legacy_v2_phase0(
    repository_root: str | Path,
    *,
    legacy_bridge: Mapping[str, Any] | Any | None = None,
    phase0_spec: Mapping[str, Any] | None = None,
    source_specification: str | None = None,
    source_specification_sha256: str | None = None,
    pipeline_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    historical_data_root: str | Path | None = None,
    label_table_path: str | Path | None = None,
    cache_search_roots: Sequence[str | Path] | None = None,
    imu_config: RollPitchEkfConfig | None = None,
    generate_report: bool = True,
) -> Phase0Result:
    """Run the complete read-only Phase-0 audit and write its data artifacts.

    ``legacy_bridge`` accepts either ``LegacyBridgeSpec`` or its YAML mapping.
    A STOP is returned only as advisory data, never raised, so the complete
    evidence package is retained without controlling pipeline execution.  The
    historical Markdown summary remains enabled by default; V5's data-only
    application boundary disables only that presentation side effect.
    """
    repository = Path(repository_root).resolve()
    pipeline = Path(pipeline_root).resolve() if pipeline_root is not None else repository / "final_v0/final_pipeline_v2"
    artifacts = Path(artifact_root).resolve() if artifact_root is not None else pipeline
    if legacy_bridge is not None:
        phase0 = dict(_bridge_field(legacy_bridge, "phase0"))
        bridge_source = str(_bridge_field(legacy_bridge, "source_specification"))
        bridge_source_sha = str(_bridge_field(legacy_bridge, "source_specification_sha256"))
        if source_specification is not None and source_specification != bridge_source:
            raise ValueError("legacy bridge source specification argument drift")
        if source_specification_sha256 is not None and source_specification_sha256 != bridge_source_sha:
            raise ValueError("legacy bridge source specification SHA argument drift")
        source_specification = bridge_source
        source_specification_sha256 = bridge_source_sha
    else:
        if phase0_spec is None:
            raise TypeError("legacy_bridge or phase0_spec is required")
        phase0 = dict(phase0_spec)
        source_specification = source_specification or _REGISTERED_SOURCE_SPECIFICATION
        source_specification_sha256 = source_specification_sha256 or _REGISTERED_SOURCE_SPECIFICATION_SHA256
    declared_source_sha = str(source_specification_sha256)
    audit_spec_sha = hashlib.sha256(
        json.dumps(phase0, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    outputs_raw = tuple((str(value) for value in phase0.get("audit_outputs", ())))
    if outputs_raw != _EXPECTED_OUTPUTS:
        raise ValueError("legacy bridge Phase-0 output contract drift")
    output_paths = {Path(relative).name: _strict_path(artifacts, relative) for relative in outputs_raw}
    stops: list[str] = []
    limitations: list[str] = []
    source_spec_path = _strict_path(repository, source_specification)
    observed_source_sha = sha256_file(source_spec_path) if source_spec_path.is_file() else ""
    if observed_source_sha != declared_source_sha:
        stops.append("source_specification_hash_mismatch")
    manifest_path = _strict_path(pipeline, str(phase0["manifest_path"]))
    split_path = _strict_path(pipeline, str(phase0["split_path"]))
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    split_sha = sha256_file(split_path) if split_path.is_file() else None
    try:
        manifest_rows = load_internal_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001 - retain the remaining audit evidence.
        manifest_rows = []
        stops.append(f"manifest_unreadable_or_invalid:{type(exc).__name__}")
    if len(manifest_rows) != int(phase0["manifest_expected_rows"]):
        stops.append("manifest_record_count_mismatch")
    data_root = (Path(historical_data_root).resolve() if historical_data_root is not None else repository /
                 "PPG_Testing_05_01_2026")
    label_path = (Path(label_table_path).resolve() if label_table_path is not None else data_root /
                  "StudyData_frailtyScored/StudyData_V7_standard.csv")
    labels, label_issues = _read_label_table(label_path)
    discovered, alias_rows, discovery_issues = _discover_static(repository, data_root, labels)
    if label_issues or discovery_issues:
        stops.append("role_class_or_participant_identity_conflict")
    if any((row["one_to_one"] != "true" for row in alias_rows)):
        stops.append("participant_alias_not_one_to_one")
    independent_classes = {row.source_path: row.class_id for row in discovered}
    # Extend the independently reconstructed identity to S/W roles without
    # treating manifest labels as independent proof.
    class_by_participant = {row.participant_id: row.class_id for row in discovered}
    for row in manifest_rows:
        if row.participant_id in class_by_participant:
            independent_classes.setdefault(row.source_path, class_by_participant[row.participant_id])
    source_rows, channel_rows, source_stops, source_limitations = _audit_sources(
        repository,
        manifest_rows,
        independent_classes,
        tuple((str(value) for value in phase0["required_channel_order"])),
    )
    stops.extend(source_stops)
    limitations.extend(source_limitations)
    observed_by_path = {row["source_path"]: row["observed_source_hash"] for row in source_rows}
    diff_rows, diff_stops = _manifest_diff(discovered, manifest_rows, observed_by_path,
                                           tuple((str(value) for value in phase0["required_channel_order"])))
    stops.extend(diff_stops)
    imu_rows, imu_stops, imu_limitations = _audit_imu(repository, manifest_rows, imu_config or RollPitchEkfConfig())
    stops.extend(imu_stops)
    limitations.extend(imu_limitations)
    roots = (tuple((Path(value).resolve() for value in cache_search_roots)) if cache_search_roots is not None else
             (repository / "datasets", ))
    cache_audit, cache_limitations = _audit_cache(repository, roots, discovered)
    limitations.extend(cache_limitations)
    split_audit, split_stops = _audit_split(split_path, manifest_rows)
    stops.extend(split_stops)
    decision, checks_passed, stop_values, limitation_values = _decision(stops, limitations)
    published_outputs = {
        name: path
        for name, path in output_paths.items() if generate_report or path.suffix.lower() != ".md"
    }
    result = Phase0Result(
        decision=decision,
        advisory_checks_passed=checks_passed,
        stop_reasons=stop_values,
        limitations=limitation_values,
        outputs={name: str(path)
                 for name, path in published_outputs.items()},
        source_specification=source_specification,
        source_specification_sha256=observed_source_sha,
        audit_spec_sha256=audit_spec_sha,
        manifest_sha256=manifest_sha,
        split_sha256=split_sha,
    )
    _write_csv(
        output_paths["legacy_v2_manifest_record_diff.csv"],
        diff_rows,
        (
            "legacy_source_path",
            "v2_record_id",
            "legacy_file_participant_id",
            "legacy_participant_alias",
            "v2_participant_id",
            "role",
            "legacy_class_id",
            "v2_class_id",
            "legacy_class_name",
            "v2_class_name",
            "observed_source_hash",
            "v2_source_hash",
            "v2_n_samples",
            "v2_channel_schema",
            "status",
            "reasons",
        ),
        artifacts,
    )
    source_fields = (tuple(source_rows[0]) if source_rows else
                     ("record_id", "source_path", "expected_source_hash", "observed_source_hash", "issues"))
    _write_csv(output_paths["legacy_v2_source_hash_audit.csv"], source_rows, source_fields, artifacts)
    _write_json(
        output_paths["legacy_v2_source_hash_audit.json"],
        {
            "schema_version": "ppg_frailty.legacy_v2_source_hash_audit.v1",
            "phase0_result": result.to_dict(),
            "manifest_version": sorted({row.manifest_version
                                        for row in manifest_rows}),
            "manifest_record_count": len(manifest_rows),
            "manifest_sha256": manifest_sha,
            "source_match_count": sum((row.get("source_hash_match") == "true" for row in source_rows)),
            "source_mismatch_count": sum((row.get("source_hash_match") != "true" for row in source_rows)),
            "label_table_issues": label_issues,
            "discovery_issues": discovery_issues,
        },
        artifacts,
    )
    channel_fields = (tuple(channel_rows[0]) if channel_rows else
                      ("record_id", "source_path", "channel_index", "channel", "nonfinite_count"))
    _write_csv(output_paths["legacy_v2_channel_qc.csv"], channel_rows, channel_fields, artifacts)
    alias_fields = (tuple(alias_rows[0]) if alias_rows else
                    ("historical_file_participant_id", "historical_participant_alias", "one_to_one", "issues"))
    _write_csv(output_paths["legacy_v2_participant_alias_map.csv"], alias_rows, alias_fields, artifacts)
    imu_fields = tuple(dict.fromkeys((key for row in imu_rows for key in row))) or (
        "participant_id",
        "record_id",
        "ekf_status",
        "ekf_failure_reason",
        "red_flags",
    )
    _write_csv(output_paths["legacy_v2_imu_unit_ekf_audit.csv"], imu_rows, imu_fields, artifacts)
    _write_json(output_paths["legacy_v2_cache_audit.json"], cache_audit, artifacts)
    _write_json(output_paths["legacy_v2_split_audit.json"], split_audit, artifacts)
    if generate_report:
        _atomic_text(
            output_paths["LEGACY_V2_PHASE0_DATA_AUDIT.md"],
            _markdown_summary(result, diff_rows, source_rows, imu_rows, cache_audit, split_audit),
            artifacts,
        )
    return result


__all__ = ["PHASE0_RESULT_SCHEMA_VERSION", "Phase0Result", "run_legacy_v2_phase0"]
