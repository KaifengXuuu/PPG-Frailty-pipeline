"""外部 heartbeat/motion 数据合同 / External heartbeat-motion data contract.

中文：本模块只适配 M2 已审核的 external_record_manifest.csv。它保留来源数据的
原始通道语义，尤其不会把 PTT 的 pleth_1..pleth_6 猜测为 RED/IR。PTT 和 SIM
数据可用于 heartbeat/motion 方法开发或对照，但本合同明确不宣称独立外部测试。

English: This module adapts only the M2-audited external_record_manifest.csv.
Source channel semantics are preserved; in particular, PTT pleth_1..pleth_6 are
never guessed to be RED/IR. PTT and SIM may support heartbeat/motion development
and comparisons, but this contract makes no independent external-test claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ppg_frailty.provenance import sha256_file


M2_EXTERNAL_RELATIVE_PATH = Path(
    "final_v0/M2_data_manifest_and_evaluation_protocol/"
    "manifests/external_record_manifest.csv"
)
M2_EXTERNAL_MANIFEST_SHA256 = (
    "43ab3273346469e9f689ce32da9c5ad280d0a53a8bc8864adf5716f40f9f024e"
)
EXTERNAL_MANIFEST_VERSION = "external_records_v1"
PTT_DATASET_ID = "ptt_ppg_1_1_0_local"
SIM_DATASET_ID = "simultaneous_measurements_1_0_0_local"
PTT_WAVELENGTH_STATUS = "unresolved_red_ir_mapping_conflict"
INDEPENDENCE_CLAIM = "none_not_an_independent_external_test"
TARGET_INTERNAL_FS_HZ = 400.0

M2_EXTERNAL_COLUMNS = (
    "dataset_id",
    "record_id",
    "subject_id",
    "source_files",
    "canonical_representation",
    "activity_raw",
    "activity_binary",
    "activity_label_source",
    "container_grid_fs_hz",
    "channel_rate_detail",
    "ppg_channels",
    "ppg_placement",
    "ppg_wavelength_status",
    "ecg_channels",
    "ecg_reference_type",
    "imu_channels",
    "imu_unit_status",
    "checksum_sha256",
    "checksum_status",
    "inclusion_status",
    "inclusion_reason",
    "known_quality_flags",
)
EXTERNAL_MANIFEST_COLUMNS = (
    "contract_schema_version",
    *M2_EXTERNAL_COLUMNS,
    "evaluation_role",
    "independence_claim",
    "resampling_required",
    "target_internal_fs_hz",
    "source_manifest_sha256",
)
PROVISIONAL_EXTERNAL_SPLIT_REGISTRY_ID = (
    "v1_provisional_external_grouped_split_seed42"
)
PROVISIONAL_EXTERNAL_SPLIT_COLUMNS = (
    "registry_id",
    "registry_status",
    "seed",
    "n_splits",
    "fold_index",
    "fold_number",
    "split",
    "subject_id",
    "record_ids",
    "activity_raw",
    "activity_binary",
    "runtime_split_recomputation_allowed",
    "source_manifest_sha256",
    "independence_claim",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExternalManifestError(ValueError):
    """聚合外部合同错误 / Aggregate external-contract failures."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__(
            "external manifest validation failed: " + " | ".join(self.issues)
        )


@dataclass(frozen=True)
class ExternalRecord:
    """一个保留来源语义的外部记录 / One source-semantics-preserving record."""

    contract_schema_version: str
    dataset_id: str
    record_id: str
    subject_id: str
    source_files: tuple[str, ...]
    canonical_representation: str
    activity_raw: str
    activity_binary: str
    activity_label_source: str
    container_grid_fs_hz: float
    channel_rate_detail: str
    ppg_channels: str
    ppg_placement: str
    ppg_wavelength_status: str
    ecg_channels: str
    ecg_reference_type: str
    imu_channels: str
    imu_unit_status: str
    checksum_sha256: str
    checksum_status: str
    inclusion_status: str
    inclusion_reason: str
    known_quality_flags: str
    evaluation_role: str
    independence_claim: str
    resampling_required: bool
    target_internal_fs_hz: float
    source_manifest_sha256: str


def _strict_json(value: object) -> str:
    """输出稳定 JSON 单元格 / Render a stable JSON-valued CSV cell."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_source_files(value: str) -> tuple[str, ...]:
    """解析并拒绝空 source file 列表 / Parse and reject empty source lists."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("source_files is not valid JSON") from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or not item for item in parsed)
    ):
        raise ValueError("source_files must be a non-empty JSON string list")
    return tuple(parsed)


def _evaluation_role(raw: Mapping[str, str]) -> str:
    """给来源记录登记用途但不制造 test 声明 / Register use without test claims."""

    if raw["inclusion_status"] != "included":
        return "excluded_source_record"
    if raw["dataset_id"] == PTT_DATASET_ID:
        return "heartbeat_motion_benchmark_candidate"
    if raw["dataset_id"] == SIM_DATASET_ID:
        return "interval_heartbeat_motion_benchmark_candidate"
    raise ValueError(f"unsupported external dataset: {raw['dataset_id']}")


def _adapt_m2_row(raw: Mapping[str, str]) -> ExternalRecord:
    """把 M2 行映射为 V1 合同 / Map one M2 row into the V1 contract."""

    fs = float(raw["container_grid_fs_hz"])
    return ExternalRecord(
        contract_schema_version=EXTERNAL_MANIFEST_VERSION,
        dataset_id=str(raw["dataset_id"]),
        record_id=str(raw["record_id"]),
        subject_id=str(raw["subject_id"]),
        source_files=_parse_source_files(str(raw["source_files"])),
        canonical_representation=str(raw["canonical_representation"]),
        activity_raw=str(raw["activity_raw"]),
        activity_binary=str(raw["activity_binary"]),
        activity_label_source=str(raw["activity_label_source"]),
        container_grid_fs_hz=fs,
        channel_rate_detail=str(raw["channel_rate_detail"]),
        ppg_channels=str(raw["ppg_channels"]),
        ppg_placement=str(raw["ppg_placement"]),
        ppg_wavelength_status=str(raw["ppg_wavelength_status"]),
        ecg_channels=str(raw["ecg_channels"]),
        ecg_reference_type=str(raw["ecg_reference_type"]),
        imu_channels=str(raw["imu_channels"]),
        imu_unit_status=str(raw["imu_unit_status"]),
        checksum_sha256=str(raw["checksum_sha256"]),
        checksum_status=str(raw["checksum_status"]),
        inclusion_status=str(raw["inclusion_status"]),
        inclusion_reason=str(raw["inclusion_reason"]),
        known_quality_flags=str(raw["known_quality_flags"]),
        evaluation_role=_evaluation_role(raw),
        independence_claim=INDEPENDENCE_CLAIM,
        resampling_required=not abs(fs - TARGET_INTERNAL_FS_HZ) < 1e-12,
        target_internal_fs_hz=TARGET_INTERNAL_FS_HZ,
        source_manifest_sha256=M2_EXTERNAL_MANIFEST_SHA256,
    )


def _validate_external_record(row: ExternalRecord) -> None:
    """验证一行且禁止 PTT 波长推断 / Validate one row and forbid inference."""

    if row.contract_schema_version != EXTERNAL_MANIFEST_VERSION:
        raise ValueError("external contract schema version drift")
    if not row.record_id or not row.subject_id or not row.source_files:
        raise ValueError("external record identity is incomplete")
    if row.dataset_id not in {PTT_DATASET_ID, SIM_DATASET_ID}:
        raise ValueError("unsupported external dataset")
    if row.container_grid_fs_hz <= 0.0 or row.target_internal_fs_hz <= 0.0:
        raise ValueError("external sampling rate must be positive")
    # 中文：PTT 是单个 canonical CSV hash；SIM 权威表保存一个
    # file-path -> SHA-256 的完整 snapshot JSON。两种编码均无损保留。
    # English: PTT stores one canonical-CSV digest, whereas SIM stores a complete
    # file-path -> SHA-256 snapshot object. Both authority encodings are preserved.
    if not _SHA256_PATTERN.fullmatch(row.checksum_sha256):
        try:
            checksum_bundle = json.loads(row.checksum_sha256)
        except json.JSONDecodeError as exc:
            raise ValueError("external checksum is neither SHA-256 nor JSON map") from exc
        if (
            not isinstance(checksum_bundle, dict)
            or not checksum_bundle
            or any(
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or not _SHA256_PATTERN.fullmatch(digest)
                for path, digest in checksum_bundle.items()
            )
        ):
            raise ValueError("external checksum JSON map is invalid")
    if row.source_manifest_sha256 != M2_EXTERNAL_MANIFEST_SHA256:
        raise ValueError("external source-manifest identity drift")
    if row.inclusion_status not in {"included", "excluded"}:
        raise ValueError("external inclusion status is invalid")
    if row.independence_claim != INDEPENDENCE_CLAIM:
        raise ValueError("external independence claim drift")
    expected_resampling = not abs(
        row.container_grid_fs_hz - row.target_internal_fs_hz
    ) < 1e-12
    if row.resampling_required != expected_resampling:
        raise ValueError("resampling flag/rate mismatch")
    if (
        row.dataset_id == PTT_DATASET_ID
        and row.ppg_wavelength_status != PTT_WAVELENGTH_STATUS
    ):
        raise ValueError("PTT wavelength mapping must remain unresolved")


def _record_to_csv(row: ExternalRecord) -> dict[str, str]:
    """编码强类型外部记录 / Encode one typed external record."""

    _validate_external_record(row)
    return {
        "contract_schema_version": row.contract_schema_version,
        "dataset_id": row.dataset_id,
        "record_id": row.record_id,
        "subject_id": row.subject_id,
        "source_files": _strict_json(list(row.source_files)),
        "canonical_representation": row.canonical_representation,
        "activity_raw": row.activity_raw,
        "activity_binary": row.activity_binary,
        "activity_label_source": row.activity_label_source,
        "container_grid_fs_hz": f"{row.container_grid_fs_hz:.12g}",
        "channel_rate_detail": row.channel_rate_detail,
        "ppg_channels": row.ppg_channels,
        "ppg_placement": row.ppg_placement,
        "ppg_wavelength_status": row.ppg_wavelength_status,
        "ecg_channels": row.ecg_channels,
        "ecg_reference_type": row.ecg_reference_type,
        "imu_channels": row.imu_channels,
        "imu_unit_status": row.imu_unit_status,
        "checksum_sha256": row.checksum_sha256,
        "checksum_status": row.checksum_status,
        "inclusion_status": row.inclusion_status,
        "inclusion_reason": row.inclusion_reason,
        "known_quality_flags": row.known_quality_flags,
        "evaluation_role": row.evaluation_role,
        "independence_claim": row.independence_claim,
        "resampling_required": "true" if row.resampling_required else "false",
        "target_internal_fs_hz": f"{row.target_internal_fs_hz:.12g}",
        "source_manifest_sha256": row.source_manifest_sha256,
    }


def _record_from_csv(raw: Mapping[str, str]) -> ExternalRecord:
    """从物化 CSV 恢复外部记录 / Restore one materialized external record."""

    missing = [name for name in EXTERNAL_MANIFEST_COLUMNS if name not in raw]
    if missing:
        raise ValueError(f"external row missing fields: {missing}")
    flag = str(raw["resampling_required"]).lower()
    if flag not in {"true", "false"}:
        raise ValueError("resampling_required must be true or false")
    row = ExternalRecord(
        contract_schema_version=str(raw["contract_schema_version"]),
        dataset_id=str(raw["dataset_id"]),
        record_id=str(raw["record_id"]),
        subject_id=str(raw["subject_id"]),
        source_files=_parse_source_files(str(raw["source_files"])),
        canonical_representation=str(raw["canonical_representation"]),
        activity_raw=str(raw["activity_raw"]),
        activity_binary=str(raw["activity_binary"]),
        activity_label_source=str(raw["activity_label_source"]),
        container_grid_fs_hz=float(raw["container_grid_fs_hz"]),
        channel_rate_detail=str(raw["channel_rate_detail"]),
        ppg_channels=str(raw["ppg_channels"]),
        ppg_placement=str(raw["ppg_placement"]),
        ppg_wavelength_status=str(raw["ppg_wavelength_status"]),
        ecg_channels=str(raw["ecg_channels"]),
        ecg_reference_type=str(raw["ecg_reference_type"]),
        imu_channels=str(raw["imu_channels"]),
        imu_unit_status=str(raw["imu_unit_status"]),
        checksum_sha256=str(raw["checksum_sha256"]),
        checksum_status=str(raw["checksum_status"]),
        inclusion_status=str(raw["inclusion_status"]),
        inclusion_reason=str(raw["inclusion_reason"]),
        known_quality_flags=str(raw["known_quality_flags"]),
        evaluation_role=str(raw["evaluation_role"]),
        independence_claim=str(raw["independence_claim"]),
        resampling_required=flag == "true",
        target_internal_fs_hz=float(raw["target_internal_fs_hz"]),
        source_manifest_sha256=str(raw["source_manifest_sha256"]),
    )
    _validate_external_record(row)
    return row


def audit_external_manifest(
    rows: Iterable[ExternalRecord],
) -> dict[str, object]:
    """执行精确 roster/activity/inclusion 审计 / Audit the exact roster."""

    materialized = list(rows)
    issues: list[str] = []
    for index, row in enumerate(materialized):
        try:
            _validate_external_record(row)
        except Exception as exc:  # noqa: BLE001 - aggregate every row issue.
            issues.append(f"row {index}: {type(exc).__name__}: {exc}")
    identities = [(row.dataset_id, row.record_id) for row in materialized]
    if len(set(identities)) != len(identities):
        issues.append("duplicate dataset/record identity")
    by_dataset = Counter(row.dataset_id for row in materialized)
    if by_dataset != Counter({PTT_DATASET_ID: 66, SIM_DATASET_ID: 14}):
        issues.append(f"external dataset-count drift: {dict(by_dataset)}")

    ptt = [row for row in materialized if row.dataset_id == PTT_DATASET_ID]
    ptt_subjects = sorted({row.subject_id for row in ptt})
    if len(ptt_subjects) != 22 or any(
        {
            row.activity_raw
            for row in ptt
            if row.subject_id == subject_id
        }
        != {"sit", "walk", "run"}
        for subject_id in ptt_subjects
    ):
        issues.append("PTT must contain 22 grouped subjects with sit/walk/run")
    if any(row.inclusion_status != "included" for row in ptt):
        issues.append("all 66 PTT records must remain included")
    if any(row.ppg_wavelength_status != PTT_WAVELENGTH_STATUS for row in ptt):
        issues.append("PTT wavelength semantics were inferred or changed")

    sim = [row for row in materialized if row.dataset_id == SIM_DATASET_ID]
    sim_included = [row for row in sim if row.inclusion_status == "included"]
    sim_excluded = [row for row in sim if row.inclusion_status == "excluded"]
    if len(sim_included) != 13 or len(sim_excluded) != 1:
        issues.append("SIM must contain 13 included and one excluded record")
    if issues:
        raise ExternalManifestError(issues)
    return {
        "contract_schema_version": EXTERNAL_MANIFEST_VERSION,
        "source_manifest_sha256": M2_EXTERNAL_MANIFEST_SHA256,
        "record_count_total": len(materialized),
        "record_count_included": sum(
            row.inclusion_status == "included" for row in materialized
        ),
        "record_count_excluded": sum(
            row.inclusion_status == "excluded" for row in materialized
        ),
        "dataset_record_counts": dict(sorted(by_dataset.items())),
        "ptt_subject_count": len(ptt_subjects),
        "ptt_activity_counts": dict(
            sorted(Counter(row.activity_raw for row in ptt).items())
        ),
        "sim_included_count": len(sim_included),
        "ptt_wavelength_interpretation": "unresolved_do_not_infer",
        "independence_claim": INDEPENDENCE_CLAIM,
    }


def load_m2_external_manifest(source_csv: str | Path) -> list[ExternalRecord]:
    """读取并验证 M2 权威外部清单 / Load the authoritative M2 external table."""

    source = Path(source_csv)
    observed = sha256_file(source)
    if observed != M2_EXTERNAL_MANIFEST_SHA256:
        raise ExternalManifestError(
            [
                "M2 external manifest SHA drift: "
                f"{observed} != {M2_EXTERNAL_MANIFEST_SHA256}"
            ]
        )
    rows: list[ExternalRecord] = []
    issues: list[str] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != M2_EXTERNAL_COLUMNS:
            raise ExternalManifestError(["M2 external column order drift"])
        for line_number, raw in enumerate(reader, start=2):
            try:
                rows.append(_adapt_m2_row(raw))
            except Exception as exc:  # noqa: BLE001 - never silently skip.
                issues.append(f"line {line_number}: {type(exc).__name__}: {exc}")
    if issues:
        raise ExternalManifestError(issues)
    audit_external_manifest(rows)
    return sorted(rows, key=lambda row: (row.dataset_id, row.record_id))


def load_external_manifest(path: str | Path) -> list[ExternalRecord]:
    """读取已物化 V1 外部清单 / Load the materialized V1 external manifest."""

    rows: list[ExternalRecord] = []
    issues: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXTERNAL_MANIFEST_COLUMNS:
            raise ExternalManifestError(["V1 external column order drift"])
        for line_number, raw in enumerate(reader, start=2):
            try:
                rows.append(_record_from_csv(raw))
            except Exception as exc:  # noqa: BLE001 - never silently skip.
                issues.append(f"line {line_number}: {type(exc).__name__}: {exc}")
    if issues:
        raise ExternalManifestError(issues)
    audit_external_manifest(rows)
    return rows


def _checked_target(path: str | Path, *, output_root: str | Path) -> Path:
    """限制所有生成写入范围 / Restrict every generated write."""

    root = Path(output_root).resolve()
    target = Path(path).resolve(strict=False)
    target.relative_to(root)
    return target


def write_external_manifest_csv(
    path: str | Path,
    rows: Iterable[ExternalRecord],
    *,
    output_root: str | Path,
) -> None:
    """原子写外部 manifest / Atomically write the external manifest."""

    materialized = list(rows)
    audit_external_manifest(materialized)
    target = _checked_target(path, output_root=output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(EXTERNAL_MANIFEST_COLUMNS),
        )
        writer.writeheader()
        writer.writerows(
            _record_to_csv(row)
            for row in sorted(
                materialized,
                key=lambda item: (item.dataset_id, item.record_id),
            )
        )
    temporary.replace(target)


def build_external_manifest(
    source_csv: str | Path,
    output_csv: str | Path,
) -> list[ExternalRecord]:
    """从唯一 M2 路径构建 V1 外部合同 / Build V1 from the sole M2 path."""

    source = Path(source_csv).resolve()
    pipeline_root = Path(__file__).resolve().parents[3]
    repository_root = pipeline_root.parents[1]
    expected = (repository_root / M2_EXTERNAL_RELATIVE_PATH).resolve()
    if source != expected:
        raise ExternalManifestError(
            [f"unsupported external authority: {source}; expected {expected}"]
        )
    rows = load_m2_external_manifest(source)
    write_external_manifest_csv(output_csv, rows, output_root=pipeline_root)
    return rows


def _provisional_fold_by_subject(
    rows: Iterable[ExternalRecord],
) -> dict[str, int]:
    """用 seed42 hash 排序后轮转分组 / Hash-rank groups then round-robin."""

    included_ptt = [
        row
        for row in rows
        if row.dataset_id == PTT_DATASET_ID
        and row.inclusion_status == "included"
    ]
    subjects = sorted({row.subject_id for row in included_ptt})
    if len(subjects) != 22:
        raise ExternalManifestError(["provisional split requires 22 PTT subjects"])
    # 中文：SHA-256(seed:subject) 比运行时 splitter 或语言特定 shuffle 更可复现。
    # English: SHA-256(seed:subject) avoids a runtime splitter and RNG-version drift.
    ranked = sorted(
        subjects,
        key=lambda subject: (
            hashlib.sha256(f"42:{subject}".encode("utf-8")).hexdigest(),
            subject,
        ),
    )
    return {subject: rank % 5 for rank, subject in enumerate(ranked)}


def _validate_provisional_rows(rows: Iterable[Mapping[str, str]]) -> None:
    """验证 provisional 分组和 activity 覆盖 / Audit grouping and coverage."""

    materialized = list(rows)
    issues: list[str] = []
    if len(materialized) != 110:
        issues.append(f"expected 110 fold/subject rows, observed {len(materialized)}")
    keys = [(row["fold_index"], row["subject_id"]) for row in materialized]
    if len(set(keys)) != len(keys):
        issues.append("duplicate provisional fold/subject row")
    subjects = sorted({row["subject_id"] for row in materialized})
    if len(subjects) != 22:
        issues.append("provisional split must contain 22 subject groups")
    oof_count = Counter(
        row["subject_id"] for row in materialized if row["split"] == "oof"
    )
    if set(oof_count.values()) != {1} or set(oof_count) != set(subjects):
        issues.append("each PTT subject must be OOF exactly once")
    for fold_index in range(5):
        fold_rows = [
            row for row in materialized if int(row["fold_index"]) == fold_index
        ]
        train = {row["subject_id"] for row in fold_rows if row["split"] == "train"}
        oof = {row["subject_id"] for row in fold_rows if row["split"] == "oof"}
        if train & oof or train | oof != set(subjects):
            issues.append(f"fold {fold_index} is not an exact grouped partition")
        if len(oof) not in {4, 5}:
            issues.append(f"fold {fold_index} OOF size is not 4 or 5")
        activity_union = {
            activity
            for row in fold_rows
            if row["split"] == "oof"
            for activity in json.loads(row["activity_raw"])
        }
        if activity_union != {"sit", "walk", "run"}:
            issues.append(f"fold {fold_index} lacks full PTT activity coverage")
    if issues:
        raise ExternalManifestError(issues)


def materialize_provisional_external_grouped_split(
    records: Iterable[ExternalRecord],
    output_csv: str | Path,
    *,
    output_root: str | Path,
) -> list[dict[str, str]]:
    """物化待 V2 确认的 PTT 参与者五折 / Materialize provisional grouped folds.

    中文：该 CSV 是不可运行时重算的开发期 grouped CV registry，不是独立 test。
    每个 subject 的 sit/walk/run 三条记录通过 subject_id 继承同一个 OOF fold。

    English: This CSV is a development grouped-CV registry that must not be
    regenerated at runtime; it is not an independent test. Each subject's
    sit/walk/run records inherit the same OOF fold through subject_id.
    """

    materialized = list(records)
    audit_external_manifest(materialized)
    included_ptt = [
        row
        for row in materialized
        if row.dataset_id == PTT_DATASET_ID
        and row.inclusion_status == "included"
    ]
    fold_by_subject = _provisional_fold_by_subject(included_ptt)
    by_subject = {
        subject: [row for row in included_ptt if row.subject_id == subject]
        for subject in sorted(fold_by_subject)
    }
    output_rows: list[dict[str, str]] = []
    for fold_index in range(5):
        for subject, subject_rows in by_subject.items():
            output_rows.append(
                {
                    "registry_id": PROVISIONAL_EXTERNAL_SPLIT_REGISTRY_ID,
                    "registry_status": (
                        "provisional_pending_v2_human_confirmation"
                    ),
                    "seed": "42",
                    "n_splits": "5",
                    "fold_index": str(fold_index),
                    "fold_number": str(fold_index + 1),
                    "split": (
                        "oof"
                        if fold_by_subject[subject] == fold_index
                        else "train"
                    ),
                    "subject_id": subject,
                    "record_ids": _strict_json(
                        sorted(row.record_id for row in subject_rows)
                    ),
                    "activity_raw": _strict_json(
                        sorted({row.activity_raw for row in subject_rows})
                    ),
                    "activity_binary": _strict_json(
                        sorted({row.activity_binary for row in subject_rows})
                    ),
                    "runtime_split_recomputation_allowed": "false",
                    "source_manifest_sha256": M2_EXTERNAL_MANIFEST_SHA256,
                    "independence_claim": INDEPENDENCE_CLAIM,
                }
            )
    _validate_provisional_rows(output_rows)
    target = _checked_target(output_csv, output_root=output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(PROVISIONAL_EXTERNAL_SPLIT_COLUMNS),
        )
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(target)
    return output_rows


def load_provisional_external_split(
    path: str | Path,
) -> list[dict[str, str]]:
    """读取并审计 provisional registry / Load and audit the provisional registry."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROVISIONAL_EXTERNAL_SPLIT_COLUMNS:
            raise ExternalManifestError(["provisional split column order drift"])
        rows = [dict(row) for row in reader]
    _validate_provisional_rows(rows)
    if any(
        row["registry_id"] != PROVISIONAL_EXTERNAL_SPLIT_REGISTRY_ID
        or row["registry_status"]
        != "provisional_pending_v2_human_confirmation"
        or row["runtime_split_recomputation_allowed"] != "false"
        or row["independence_claim"] != INDEPENDENCE_CLAIM
        for row in rows
    ):
        raise ExternalManifestError(["provisional split identity/status drift"])
    return rows


__all__ = [
    "EXTERNAL_MANIFEST_COLUMNS",
    "EXTERNAL_MANIFEST_VERSION",
    "ExternalManifestError",
    "ExternalRecord",
    "INDEPENDENCE_CLAIM",
    "M2_EXTERNAL_MANIFEST_SHA256",
    "M2_EXTERNAL_RELATIVE_PATH",
    "PROVISIONAL_EXTERNAL_SPLIT_REGISTRY_ID",
    "PTT_DATASET_ID",
    "SIM_DATASET_ID",
    "audit_external_manifest",
    "build_external_manifest",
    "load_external_manifest",
    "load_m2_external_manifest",
    "load_provisional_external_split",
    "materialize_provisional_external_grouped_split",
    "write_external_manifest_csv",
]
