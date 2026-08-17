"""数据层机器合同 / Machine contracts for the data layer.

中文：这里集中定义 manifest CSV 编码、QC reason code 和物化 fold 行。
CSV 中的复合字段使用 strict JSON；读取后立即恢复强类型并执行 fail-closed 校验。

English: This module centralizes manifest CSV encoding, QC reason codes, and
materialized fold rows. Composite CSV fields use strict JSON and are restored to
typed values followed by fail-closed validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ppg_frailty.contracts import ManifestRow


CANONICAL_CHANNEL_SCHEMA = (
    "RED",
    "IR",
    "AX",
    "AY",
    "AZ",
    "GX",
    "GY",
    "GZ",
)
CANONICAL_CLASS_NAMES = {
    0: "Pre-Frail",
    1: "Robust/Non-Frail",
    2: "Young",
}
REGISTERED_ROLES = (
    "B",
    "R1",
    "R2",
    "R3",
    "R4",
    "S1",
    "S2",
    "W1",
    "W2",
)
MANIFEST_VERSION = "internal_records_v1"
MANIFEST_COLUMNS = (
    "record_id",
    "participant_id",
    "class_id",
    "class_name",
    "role",
    "source_path",
    "source_hash",
    "source_version",
    "fs",
    "n_samples",
    "duration_s",
    "channel_schema",
    "channel_units",
    "synchrony_status",
    "reference_available",
    "qc_status",
    "qc_reasons",
    "manifest_version",
)
FOLD_COLUMNS = (
    "source_registry_id",
    "source_registry_file_sha256",
    "source_registry_payload_sha256",
    "dataset_version_id",
    "repeat_index",
    "split_seed",
    "fold_index",
    "fold_number",
    "participant_id",
    "class_id",
    "class_name",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class QCStatus(str, Enum):
    """Recording 级 QC 结论 / Recording-level QC disposition."""

    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"


class QCReason(str, Enum):
    """规范要求的 QC reason code / Contract-required QC reason codes."""

    PARSE_FAILURE = "parse_failure"
    MISSING_REQUIRED_CHANNEL = "missing_required_channel"
    ALL_NONFINITE_CHANNEL = "all_nonfinite_channel"
    EXCESSIVE_NONFINITE_GAP = "excessive_nonfinite_gap"
    INSUFFICIENT_DURATION = "insufficient_duration"
    FLATLINE = "flatline"
    CLIPPING = "clipping"
    SATURATION = "saturation"
    IMPLAUSIBLE_SCALE = "implausible_scale"
    TIMESTAMP_FAILURE = "timestamp_failure"
    SYNCHRONY_FAILURE = "synchrony_failure"
    DUPLICATE_RECORD = "duplicate_record"


@dataclass(frozen=True)
class FoldAssignment:
    """一个 participant 在一个 repeat 中的 outer fold / One frozen assignment."""

    source_registry_id: str
    source_registry_file_sha256: str
    source_registry_payload_sha256: str
    dataset_version_id: str
    repeat_index: int
    split_seed: int
    fold_index: int
    fold_number: int
    participant_id: str
    class_id: int
    class_name: str


def _strict_json(value: Any) -> str:
    """输出稳定 strict JSON / Render stable strict JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_json_field(value: str, *, field_name: str) -> Any:
    """解析 CSV 中的 JSON 字段 / Parse a JSON-valued CSV field."""

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {field_name}") from exc


def validate_manifest_row(row: ManifestRow) -> None:
    """验证一个规范 recording 行 / Validate one canonical recording row."""

    if not row.record_id or not row.participant_id:
        raise ValueError("record_id and participant_id must be non-empty")
    if row.class_id not in CANONICAL_CLASS_NAMES:
        raise ValueError(f"unsupported class_id: {row.class_id}")
    if row.class_name != CANONICAL_CLASS_NAMES[row.class_id]:
        raise ValueError("class_id/class_name mismatch")
    if row.role not in REGISTERED_ROLES:
        raise ValueError(f"unregistered role: {row.role}")
    if not row.source_path or row.source_path.startswith("/"):
        raise ValueError("source_path must be a repository-relative path")
    if not _SHA256_PATTERN.fullmatch(row.source_hash):
        raise ValueError("source_hash must be lowercase SHA-256")
    if not row.source_version or row.manifest_version != MANIFEST_VERSION:
        raise ValueError("source/manifest version is missing or unsupported")
    if row.fs <= 0.0 or row.n_samples <= 0 or row.duration_s <= 0.0:
        raise ValueError("sampling metadata must be positive")
    if tuple(row.channel_schema) != CANONICAL_CHANNEL_SCHEMA:
        raise ValueError("internal recordings require exact RED/IR/IMU channel order")
    if set(row.channel_units) != set(CANONICAL_CHANNEL_SCHEMA):
        raise ValueError("channel_units must cover the exact channel schema")
    if not row.synchrony_status:
        raise ValueError("synchrony_status must be explicit")
    if row.qc_status not in {status.value for status in QCStatus}:
        raise ValueError(f"unsupported qc_status: {row.qc_status}")
    if row.qc_status == QCStatus.FAIL.value and not row.qc_reasons:
        raise ValueError("failed QC requires at least one reason")
    if any(not str(reason).strip() for reason in row.qc_reasons):
        raise ValueError("QC reasons cannot contain empty values")


def manifest_row_to_csv(row: ManifestRow) -> dict[str, str]:
    """把强类型行编码为可移植 CSV / Encode a typed row for portable CSV."""

    validate_manifest_row(row)
    return {
        "record_id": row.record_id,
        "participant_id": row.participant_id,
        "class_id": str(row.class_id),
        "class_name": row.class_name,
        "role": row.role,
        "source_path": row.source_path,
        "source_hash": row.source_hash,
        "source_version": row.source_version,
        "fs": f"{row.fs:.12g}",
        "n_samples": str(row.n_samples),
        "duration_s": f"{row.duration_s:.12g}",
        "channel_schema": _strict_json(list(row.channel_schema)),
        "channel_units": _strict_json(row.channel_units),
        "synchrony_status": row.synchrony_status,
        "reference_available": "true" if row.reference_available else "false",
        "qc_status": row.qc_status,
        "qc_reasons": _strict_json(list(row.qc_reasons)),
        "manifest_version": row.manifest_version,
    }


def manifest_row_from_csv(raw: Mapping[str, str]) -> ManifestRow:
    """从 CSV 恢复强类型行并校验 / Restore and validate a typed row."""

    missing = [field for field in MANIFEST_COLUMNS if field not in raw]
    if missing:
        raise ValueError(f"manifest row missing fields: {missing}")
    reference_text = str(raw["reference_available"]).strip().lower()
    if reference_text not in {"true", "false"}:
        raise ValueError("reference_available must be true or false")
    row = ManifestRow(
        record_id=str(raw["record_id"]),
        participant_id=str(raw["participant_id"]),
        class_id=int(raw["class_id"]),
        class_name=str(raw["class_name"]),
        role=str(raw["role"]),
        source_path=str(raw["source_path"]),
        source_hash=str(raw["source_hash"]),
        source_version=str(raw["source_version"]),
        fs=float(raw["fs"]),
        n_samples=int(raw["n_samples"]),
        duration_s=float(raw["duration_s"]),
        channel_schema=tuple(
            str(value)
            for value in _parse_json_field(
                raw["channel_schema"], field_name="channel_schema"
            )
        ),
        channel_units={
            str(key): str(value)
            for key, value in dict(
                _parse_json_field(
                    raw["channel_units"], field_name="channel_units"
                )
            ).items()
        },
        synchrony_status=str(raw["synchrony_status"]),
        reference_available=reference_text == "true",
        qc_status=str(raw["qc_status"]),
        qc_reasons=tuple(
            str(value)
            for value in _parse_json_field(
                raw["qc_reasons"], field_name="qc_reasons"
            )
        ),
        manifest_version=str(raw["manifest_version"]),
    )
    validate_manifest_row(row)
    return row


def fold_assignment_to_csv(row: FoldAssignment) -> dict[str, str]:
    """编码一个物化 fold 行 / Encode one materialized fold assignment."""

    if not _SHA256_PATTERN.fullmatch(row.source_registry_file_sha256):
        raise ValueError("invalid source registry file hash")
    if not _SHA256_PATTERN.fullmatch(row.source_registry_payload_sha256):
        raise ValueError("invalid source registry payload hash")
    if row.class_name != CANONICAL_CLASS_NAMES.get(row.class_id):
        raise ValueError("fold assignment class mismatch")
    return {
        "source_registry_id": row.source_registry_id,
        "source_registry_file_sha256": row.source_registry_file_sha256,
        "source_registry_payload_sha256": row.source_registry_payload_sha256,
        "dataset_version_id": row.dataset_version_id,
        "repeat_index": str(row.repeat_index),
        "split_seed": str(row.split_seed),
        "fold_index": str(row.fold_index),
        "fold_number": str(row.fold_number),
        "participant_id": row.participant_id,
        "class_id": str(row.class_id),
        "class_name": row.class_name,
    }


def fold_assignment_from_csv(raw: Mapping[str, str]) -> FoldAssignment:
    """恢复一个物化 fold 行 / Restore one materialized fold assignment."""

    missing = [field for field in FOLD_COLUMNS if field not in raw]
    if missing:
        raise ValueError(f"fold row missing fields: {missing}")
    row = FoldAssignment(
        source_registry_id=str(raw["source_registry_id"]),
        source_registry_file_sha256=str(raw["source_registry_file_sha256"]),
        source_registry_payload_sha256=str(
            raw["source_registry_payload_sha256"]
        ),
        dataset_version_id=str(raw["dataset_version_id"]),
        repeat_index=int(raw["repeat_index"]),
        split_seed=int(raw["split_seed"]),
        fold_index=int(raw["fold_index"]),
        fold_number=int(raw["fold_number"]),
        participant_id=str(raw["participant_id"]),
        class_id=int(raw["class_id"]),
        class_name=str(raw["class_name"]),
    )
    # 中文：复用编码器的全部一致性检查，避免读写两套规则。
    # English: Reuse encoder validation so read/write rules cannot diverge.
    fold_assignment_to_csv(row)
    return row


__all__ = [
    "CANONICAL_CHANNEL_SCHEMA",
    "CANONICAL_CLASS_NAMES",
    "FOLD_COLUMNS",
    "FoldAssignment",
    "MANIFEST_COLUMNS",
    "MANIFEST_VERSION",
    "QCReason",
    "QCStatus",
    "REGISTERED_ROLES",
    "fold_assignment_from_csv",
    "fold_assignment_to_csv",
    "manifest_row_from_csv",
    "manifest_row_to_csv",
    "validate_manifest_row",
]
