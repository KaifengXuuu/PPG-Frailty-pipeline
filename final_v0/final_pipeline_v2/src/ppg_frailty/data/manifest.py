"""从 M2 权威快照导入规范 manifest / Import the canonical M2 snapshot.

中文：本模块不从文件名重新推断标签，也不静默跳过异常行。导入时逐行收集
所有错误，任何错误都会使整个物化过程失败；可选的 source verification 会重新
计算每个原始文件的 SHA-256。

English: Labels are never re-inferred from filenames and malformed rows are never
silently skipped. Import collects every row error and fails the entire materialization;
optional source verification re-hashes every original recording.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from ppg_frailty.contracts import ManifestRow
from ppg_frailty.provenance import sha256_file

from .schema import (
    CANONICAL_CHANNEL_SCHEMA,
    CANONICAL_CLASS_NAMES,
    MANIFEST_COLUMNS,
    MANIFEST_VERSION,
    QCStatus,
    REGISTERED_ROLES,
    canonicalize_role_family,
    is_default_classifier_role,
    manifest_row_from_csv,
    manifest_row_to_csv,
    validate_manifest_row,
)


M2_RELATIVE_ROOT = Path("final_v0/M2_data_manifest_and_evaluation_protocol")
M2_FILE_MANIFEST = M2_RELATIVE_ROOT / "manifests/frailty3_file_manifest.csv"
M2_FILE_MANIFEST_SHA256 = (
    "bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90"
)
M2_DATASET_VERSION_ID = "frailty3_m2_20260815_a054800abda272f6"
M2_SNAKE_CLASS_NAMES = {
    0: "pre_frail",
    1: "robust_non_frail",
    2: "young",
}
SYNCHRONY_STATUS = "row_aligned_eight_channel_fixed_grid_no_timestamp"


class ManifestImportError(ValueError):
    """包含全部导入错误 / Aggregate all import failures without data loss."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__("manifest import failed: " + " | ".join(self.issues))


def _parse_m2_json(value: str, *, field_name: str) -> object:
    """解析 M2 CSV 的复合字段 / Parse a composite M2 CSV field."""

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid M2 {field_name} JSON") from exc


def convert_m2_row(raw: Mapping[str, str]) -> ManifestRow:
    """无损映射一个 M2 行 / Losslessly map one M2 row to ManifestRow."""

    if raw.get("dataset_version_id") != M2_DATASET_VERSION_ID:
        raise ValueError("unexpected M2 dataset_version_id")
    class_id = int(raw["class_id"])
    if raw.get("class_name") != M2_SNAKE_CLASS_NAMES.get(class_id):
        raise ValueError("M2 class ID/name mismatch")
    role = str(raw["role"])
    role_family = canonicalize_role_family(role)
    if raw.get("role_family") != role_family:
        raise ValueError("M2 role/role_family mismatch")
    channels = tuple(
        str(value)
        for value in _parse_m2_json(raw["channels"], field_name="channels")
    )
    units_raw = _parse_m2_json(raw["units"], field_name="units")
    if not isinstance(units_raw, dict):
        raise ValueError("M2 units must decode to an object")
    if channels != CANONICAL_CHANNEL_SCHEMA:
        raise ValueError("M2 channel order drift")
    if raw.get("numeric_full_scan") != "passed_finite_8_columns":
        raise ValueError("M2 numeric full scan did not pass")
    if raw.get("inclusion_status") != "included":
        raise ValueError("frozen internal roster contains a non-included row")
    warning_codes = tuple(
        value for value in str(raw.get("warning_codes", "")).split(";") if value
    )
    reference_text = str(raw.get("reference_available", "")).lower()
    if reference_text not in {"true", "false"}:
        raise ValueError("invalid M2 reference_available value")
    row = ManifestRow(
        record_id=str(raw["file_id"]),
        participant_id=str(raw["subject_id"]),
        class_id=class_id,
        class_name=CANONICAL_CLASS_NAMES[class_id],
        class_name_provenance_alias=str(raw["class_name"]),
        class_source=str(raw["class_source"]),
        label_record_id=str(raw["label_record_id"]),
        role=role,
        source_path=Path(str(raw["relative_path"])).as_posix(),
        source_hash=str(raw["sha256"]),
        source_version=M2_DATASET_VERSION_ID,
        fs=float(raw["raw_fs_hz"]),
        n_samples=int(raw["n_samples"]),
        duration_s=float(raw["duration_seconds"]),
        channel_schema=channels,
        channel_units={str(key): str(value) for key, value in units_raw.items()},
        synchrony_status=SYNCHRONY_STATUS,
        reference_available=reference_text == "true",
        qc_status=(
            QCStatus.PASS_WITH_WARNINGS.value
            if warning_codes
            else QCStatus.PASS.value
        ),
        # 中文：M2 的 duration warnings 是资格诊断，不自动把短 S/W 记录排除。
        # English: Duration warnings are eligibility diagnostics, not global exclusion.
        qc_reasons=warning_codes,
        manifest_version=MANIFEST_VERSION,
    )
    validate_manifest_row(row)
    return row


def _validate_manifest_set(rows: list[ManifestRow]) -> None:
    """验证完整 roster 和唯一性 / Validate the complete roster and uniqueness."""

    issues: list[str] = []
    if len(rows) != 261:
        issues.append(f"expected 261 records, observed {len(rows)}")
    record_ids = [row.record_id for row in rows]
    source_paths = [row.source_path for row in rows]
    participant_roles = [(row.participant_id, row.role) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        issues.append("duplicate record_id")
    if len(set(source_paths)) != len(source_paths):
        issues.append("duplicate source_path")
    if len(set(participant_roles)) != len(participant_roles):
        issues.append("duplicate participant/role recording")
    participants = sorted({row.participant_id for row in rows})
    if len(participants) != 29:
        issues.append(f"expected 29 participants, observed {len(participants)}")
    for participant in participants:
        roles = {row.role for row in rows if row.participant_id == participant}
        if roles != set(REGISTERED_ROLES):
            issues.append(f"role coverage mismatch for {participant}: {sorted(roles)}")
    class_counts = Counter(
        next(row.class_id for row in rows if row.participant_id == participant)
        for participant in participants
    )
    if class_counts != Counter({0: 9, 1: 12, 2: 8}):
        issues.append(f"participant class-count drift: {dict(class_counts)}")
    if issues:
        raise ManifestImportError(issues)


def load_m2_internal_manifest(
    repository_root: str | Path,
    *,
    verify_sources: bool,
) -> list[ManifestRow]:
    """加载、校验并可选重哈希 M2 / Load and optionally re-hash M2 sources."""

    repo = Path(repository_root).resolve()
    source = repo / M2_FILE_MANIFEST
    digest = sha256_file(source)
    if digest != M2_FILE_MANIFEST_SHA256:
        raise ManifestImportError(
            [f"M2 file manifest SHA drift: {digest} != {M2_FILE_MANIFEST_SHA256}"]
        )
    issues: list[str] = []
    rows: list[ManifestRow] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            try:
                rows.append(convert_m2_row(raw))
            except Exception as exc:  # noqa: BLE001 - aggregate every row failure.
                # 中文：继续仅用于收集全部错误；最终绝不会返回不完整 manifest。
                # English: Continue only to collect all errors; partial data is never returned.
                issues.append(f"line {line_number}: {type(exc).__name__}: {exc}")
    if issues:
        raise ManifestImportError(issues)
    _validate_manifest_set(rows)
    if verify_sources:
        source_issues: list[str] = []
        for row in rows:
            path = (repo / row.source_path).resolve(strict=False)
            try:
                path.relative_to(repo)
            except ValueError:
                source_issues.append(f"source escapes repository: {row.record_id}")
                continue
            if not path.is_file():
                source_issues.append(f"missing source: {row.record_id}")
                continue
            observed = sha256_file(path)
            if observed != row.source_hash:
                source_issues.append(
                    f"source hash drift: {row.record_id}:{observed}"
                )
        if source_issues:
            raise ManifestImportError(source_issues)
    return sorted(rows, key=lambda row: row.record_id.encode("utf-8"))


def _checked_target(path: str | Path, *, output_root: str | Path) -> Path:
    """限制生成写入目录 / Restrict generated writes to the package root."""

    root = Path(output_root).resolve()
    target = Path(path).resolve(strict=False)
    target.relative_to(root)
    return target


def write_manifest_csv(
    path: str | Path,
    rows: Iterable[ManifestRow],
    *,
    output_root: str | Path,
) -> None:
    """原子写规范 manifest / Atomically write the canonical manifest."""

    target = _checked_target(path, output_root=output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    ordered = sorted(rows, key=lambda row: row.record_id.encode("utf-8"))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(manifest_row_to_csv(row) for row in ordered)
    temporary.replace(target)


def load_internal_manifest(path: str | Path) -> list[ManifestRow]:
    """读取已物化的规范 manifest / Load a materialized canonical manifest."""

    rows: list[ManifestRow] = []
    issues: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            try:
                rows.append(manifest_row_from_csv(raw))
            except Exception as exc:  # noqa: BLE001 - report every invalid row.
                issues.append(f"line {line_number}: {type(exc).__name__}: {exc}")
    if issues:
        raise ManifestImportError(issues)
    _validate_manifest_set(rows)
    return rows


def manifest_summary(rows: Iterable[ManifestRow]) -> dict[str, object]:
    """生成无预测字段的审计摘要 / Summarize identity/QC without predictors."""

    materialized = list(rows)
    participants = {
        row.participant_id: row.class_id for row in materialized
    }
    warning_counts = Counter(
        reason for row in materialized for reason in row.qc_reasons
    )
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version_id": M2_DATASET_VERSION_ID,
        "record_count": len(materialized),
        "participant_count": len(participants),
        "class_participant_counts": {
            CANONICAL_CLASS_NAMES[class_id]: sum(
                value == class_id for value in participants.values()
            )
            for class_id in sorted(CANONICAL_CLASS_NAMES)
        },
        "role_record_counts": dict(
            sorted(Counter(row.role for row in materialized).items())
        ),
        "role_family_record_counts": dict(
            sorted(Counter(canonicalize_role_family(row.role) for row in materialized).items())
        ),
        "default_classifier_record_count": sum(
            is_default_classifier_role(row.role) for row in materialized
        ),
        "default_classifier_role_families": ["B", "R"],
        "class_name_provenance_aliases": {
            str(class_id): M2_SNAKE_CLASS_NAMES[class_id]
            for class_id in sorted(M2_SNAKE_CLASS_NAMES)
        },
        "class_source_record_counts": dict(
            sorted(Counter(row.class_source for row in materialized).items())
        ),
        "label_record_id_status_counts": {
            "frailty_status_record_present": sum(
                row.class_id in {0, 1} and bool(row.label_record_id.strip())
                for row in materialized
            ),
            "cohort_source_record_present": sum(
                row.class_id == 2 and bool(row.label_record_id.strip())
                for row in materialized
            ),
            "cohort_source_record_absent": sum(
                row.class_id == 2 and not row.label_record_id.strip()
                for row in materialized
            ),
        },
        "qc_status_counts": dict(
            sorted(Counter(row.qc_status for row in materialized).items())
        ),
        "qc_reason_counts": dict(sorted(warning_counts.items())),
        "reference_available_count": sum(
            row.reference_available for row in materialized
        ),
        "technical_fields_excluded_from_predictors": [
            "n_samples",
            "duration_s",
            "source_path",
            "source_hash",
            "record_id",
            "participant_id",
            "class_name_provenance_alias",
            "class_source",
            "label_record_id",
        ],
    }


def audit_manifest(rows: Iterable[ManifestRow]) -> dict[str, object]:
    """执行完整 roster 审计并返回摘要 / Audit the full roster and summarize it.

    中文：公共审计入口会先执行与 M2 导入相同的完整性检查，因此调用方不能
    仅凭一个看似合理的计数摘要接受部分 manifest。

    English: The public audit entry point first runs the same completeness checks
    as the M2 importer, so callers cannot accept a partial manifest from plausible
    looking summary counts alone.
    """

    materialized = list(rows)
    _validate_manifest_set(materialized)
    return manifest_summary(materialized)


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """加载规范内部 manifest 的稳定别名 / Stable canonical loader alias."""

    return load_internal_manifest(path)


def build_internal_manifest(
    source_csv: str | Path,
    output_csv: str | Path,
) -> list[ManifestRow]:
    """从唯一 M2 权威 CSV 构建内部合同 / Build from the sole M2 authority.

    中文：此 API 故意不接受任意同形 CSV。它验证传入路径确为 repository 内
    M2 权威路径、验证 manifest hash，并逐一重算 261 个原始 recording hash。
    输出被限制在 final_pipeline_v2 内。

    English: This API intentionally rejects arbitrary look-alike CSV files. It
    verifies that the source is the repository's authoritative M2 path, validates
    its digest, and re-hashes all 261 source recordings. Output is constrained to
    final_pipeline_v2.
    """

    source = Path(source_csv).resolve()
    # manifest.py -> data -> ppg_frailty -> src -> final_pipeline_v2
    pipeline_root = Path(__file__).resolve().parents[3]
    # final_pipeline_v2 -> final_v0 -> repository
    repository_root = pipeline_root.parents[1]
    expected = (repository_root / M2_FILE_MANIFEST).resolve()
    if source != expected:
        raise ManifestImportError(
            [f"unsupported manifest authority: {source}; expected {expected}"]
        )
    rows = load_m2_internal_manifest(repository_root, verify_sources=True)
    write_manifest_csv(output_csv, rows, output_root=pipeline_root)
    return rows


__all__ = [
    "M2_DATASET_VERSION_ID",
    "M2_FILE_MANIFEST",
    "M2_FILE_MANIFEST_SHA256",
    "ManifestImportError",
    "audit_manifest",
    "build_internal_manifest",
    "convert_m2_row",
    "load_internal_manifest",
    "load_manifest",
    "load_m2_internal_manifest",
    "manifest_summary",
    "write_manifest_csv",
]
