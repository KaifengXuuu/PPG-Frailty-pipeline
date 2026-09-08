"""外部 heartbeat/motion 数据合同 / External heartbeat-motion data contract.

中文：本模块只适配 M2 已审核的 external_record_manifest.csv。它保留来源数据的
原始 PTT 状态，同时显式记录 V2 项目采用的 distal 映射：
``pleth_1=RED``、``pleth_2=IR``。PhysioNet v1.0.0 官方页明确说明
pleth_1 为 distal red、pleth_2 为 distal infrared；用户确认项目采用该对。
早期 M2 unresolved/conflict 仍作为历史 provenance 保留。

English: This module adapts only the M2-audited external_record_manifest.csv.
The historical unresolved M2 status is retained as provenance. The official
PhysioNet v1.0.0 page identifies pleth_1 as distal red and pleth_2 as distal
infrared; V2 records the user's project adoption without an independent-test claim.
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

import numpy as np

from ppg_frailty.provenance import sha256_file


M2_EXTERNAL_RELATIVE_PATH = Path(
    "final_v0/M2_data_manifest_and_evaluation_protocol/" "manifests/external_record_manifest.csv"
)
M2_EXTERNAL_MANIFEST_SHA256 = "43ab3273346469e9f689ce32da9c5ad280d0a53a8bc8864adf5716f40f9f024e"
EXTERNAL_MANIFEST_VERSION = "external_records_v2"
PTT_DATASET_ID = "ptt_ppg_1_1_0_local"
SIM_DATASET_ID = "simultaneous_measurements_1_0_0_local"
PTT_SOURCE_WAVELENGTH_STATUS = "unresolved_red_ir_mapping_conflict"
PTT_WAVELENGTH_STATUS = "project_adopted_distal_pleth_1_red_pleth_2_ir_v2"
PTT_SOURCE_PAGE_URL = "https://physionet.org/content/pulse-transit-time-ppg/1.0.0/"
PTT_DISTAL_CHANNEL_MAPPING = {"RED": "pleth_1", "IR": "pleth_2"}
PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH = Path("final_v0/final_pipeline_v5/manifests/ptt_imu_unit_evidence_v2_036.json")
PTT_IMU_UNIT_EVIDENCE_SHA256 = "c03669029f9e58e53ae24e5e9208aa47541b93d8fcc0b9dea2f215165dc70ee5"
PTT_ADOPTED_ACCELERATION_UNIT = "m/s^2"
PTT_ADOPTED_ACCELERATION_CONVERSION = "identity_m_per_s2_no_scale"
PTT_ADOPTED_GYROSCOPE_UNIT = "deg/s"
PTT_ADOPTED_GYROSCOPE_CONVERSION = "degrees_per_second_to_radians_per_second"
PTT_CHANNEL_MAPPING_PROVENANCE = {
    "historical_m2_status": PTT_SOURCE_WAVELENGTH_STATUS,
    "official_page_url": PTT_SOURCE_PAGE_URL,
    "official_page_version": "1.0.0",
    "official_page_evidence_status": "official_v1_0_0_page_statement",
    "official_page_verified_on": "2026-08-16",
    "official_channel_statement": {
        "pleth_1": {
            "sensor": "MAX30101",
            "wavelength_label": "red",
            "site": "distal",
            "sampling_rate_hz": 500.0,
        },
        "pleth_2": {
            "sensor": "MAX30101",
            "wavelength_label": "infrared",
            "site": "distal",
            "sampling_rate_hz": 500.0,
        },
    },
    "published_statement_reconciliation": {
        "v1_0_data_description": {
            "url": "https://physionet.org/content/pulse-transit-time-ppg/1.0.0/",
            "version": "1.0.0",
            "section": "Data Description",
            "statement": "pleth_1 red; pleth_2 infrared",
        },
        "v1_0_methods": {
            "url": "https://physionet.org/content/pulse-transit-time-ppg/1.0.0/",
            "version": "1.0.0",
            "section": "Methods",
            "statement": "pleth_1 infrared; pleth_2 red",
        },
        "v1_1_data_description": {
            "url": "https://physionet.org/content/pulse-transit-time-ppg/1.1.0/",
            "version": "1.1.0",
            "section": "Data Description",
            "statement": "pleth_1 red; pleth_2 infrared",
        },
        "resolution": "project_adopted_distal_pleth_1_red_pleth_2_ir_v2",
    },
    "project_mapping": PTT_DISTAL_CHANNEL_MAPPING,
    "project_site": "left_index_finger_distal",
    "project_adoption_decision": "user_confirmed_v2_002",
    "source_manifest_rewritten": False,
}
PTT_IMU_UNIT_CONFLICT_PROVENANCE = {
    "status": "project_resolved_v2_036_source_manifest_conflict_retained",
    "historical_source_status": "unresolved_source_level_unit_contradiction",
    "official_declarations": [
        {
            "url": "https://physionet.org/content/pulse-transit-time-ppg/1.0.0/",
            "version": "1.0.0",
            "section": "Data Description",
            "acceleration_unit": "g",
            "gyroscope_unit": "degrees/s",
        },
        {
            "url": "https://physionet.org/content/pulse-transit-time-ppg/1.1.0/",
            "version": "1.1.0",
            "section": "Data Description",
            "acceleration_unit": "g",
            "gyroscope_unit": "degrees/s",
        },
    ],
    "wfdb_header_declaration": {
        "relative_path": "physionet.org/files/pulse-transit-time-ppg/1.1.0/s1_sit.hea",
        "sha256": "8f4570b9f58c43d9f0bf1dac3eba71d91acd0d85c6bff33067b7db897ae7209f",
        "acceleration_unit": "g",
        "gyroscope_unit": "deg/s",
    },
    "canonical_csv_numeric_evidence": {
        "record_id": "s1_sit",
        "relative_path": "physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s1_sit.csv",
        "sha256": "25601a60611e01fe138a88829b202c95f047764f5ee232b4d3457df460133391",
        "first_acceleration_xyz": [4.298409, 1.371349, -8.450766],
        "first_acceleration_norm": 9.5797893503896,
        "first_gyroscope_xyz": [0.007759, -0.000482, 0.004583],
        "first_gyroscope_norm": 0.00902431681624709,
        "interpretation_boundary": (
            "V2-036 adopts source acceleration as m/s^2 from the sitting gravity "
            "magnitude; gyro values do not contradict the header's deg/s declaration "
            "but are not an independent device calibration"
        ),
    },
    "historical_code_transform": {
        "relative_path": "pttppg_denoiser_onnx_runtime.py",
        "sha256": "bc76aa4387987408464bd686d4c3dbb464497d37f7f4a8c746f1c5304e773cd7",
        "function_lines": {
            "deg2rad": 125,
            "acceleration_g_to_m_per_s2": 130,
        },
        "acceleration": ("legacy multiply source values by 9.80665; prohibited for V2-036 PTT"),
        "gyroscope": "legacy degrees-to-radians conversion retained",
        "authority": "historical_implementation_only_not_unit_resolution",
        "formal_v2_036_use": "forbidden_for_acceleration_provenance_only",
    },
    "project_adopted_units": {
        "decision_id": "V2-036",
        "decision_date": "2026-08-17",
        "decision_authority": "user_confirmed_project_decision",
        "acceleration_unit": PTT_ADOPTED_ACCELERATION_UNIT,
        "acceleration_conversion": PTT_ADOPTED_ACCELERATION_CONVERSION,
        "gyroscope_unit": PTT_ADOPTED_GYROSCOPE_UNIT,
        "gyroscope_conversion": PTT_ADOPTED_GYROSCOPE_CONVERSION,
        "gyroscope_evidence_boundary": (
            "WFDB header plus historical deg2rad path with no observed numeric conflict; "
            "not independently device-calibrated"
        ),
    },
    "canonical_evidence_artifact": {
        "relative_path": PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH.as_posix(),
        "sha256": PTT_IMU_UNIT_EVIDENCE_SHA256,
        "schema_version": "ppg_frailty.ptt_imu_unit_evidence.v3",
    },
    "formal_action": (
        "require exact V2-036 artifact path/hash before conversion, EKF, " "materialization, or evaluation"
    ),
}
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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

class ExternalManifestError(ValueError):
    """聚合外部合同错误 / Aggregate external-contract failures."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__("external manifest validation failed: " + " | ".join(self.issues))

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

@dataclass(frozen=True)
class PttSynchronizedSignals:
    """Canonical distal PTT channels on one deterministic 400-Hz grid."""

    record_id: str
    source_values: np.ndarray
    values: np.ndarray
    channel_schema: tuple[str, ...]
    timestamps_s: np.ndarray
    source_channel_schema: tuple[str, ...]
    source_fs_hz: float
    target_fs_hz: float
    up: int
    down: int
    mapping_provenance: Mapping[str, object]
    source_file_sha256: str
    source_manifest_sha256: str
    source_values_sha256: str
    output_values_sha256: str
    source_channel_schema_sha256: str
    target_channel_schema_sha256: str
    resampling_config_sha256: str
    mapping_sha256: str

    @property
    def ppg_red_ir(self) -> np.ndarray:
        return np.asarray(self.values[:, :2], dtype=np.float64)

    def validate(self) -> None:
        matrix = np.asarray(self.values)
        source = np.asarray(self.source_values)
        times = np.asarray(self.timestamps_s)
        if not self.record_id:
            raise ValueError("PTT adapted record_id must be non-empty")
        if self.channel_schema[:2] != ("RED", "IR"):
            raise ValueError("PTT adapted channels must begin with canonical RED/IR")
        if self.source_channel_schema[:2] != ("pleth_1", "pleth_2"):
            raise ValueError("PTT source channels must begin with distal pleth_1/pleth_2")
        if matrix.ndim != 2 or matrix.shape[1] != len(self.channel_schema):
            raise ValueError("PTT adapted values/schema are misaligned")
        if source.ndim != 2 or source.shape[1] != len(self.source_channel_schema):
            raise ValueError("PTT source values/schema are misaligned")
        if times.shape != (matrix.shape[0],) or not np.isfinite(matrix).all():
            raise ValueError("PTT adapted signal/timeline is invalid")
        if self.source_fs_hz != 500.0 or self.target_fs_hz != TARGET_INTERNAL_FS_HZ:
            raise ValueError("formal PTT adapter is frozen to synchronized 500-to-400 Hz")
        if (self.up, self.down) != (4, 5):
            raise ValueError("formal PTT resampling ratio must be exactly 4/5")
        if dict(self.mapping_provenance) != PTT_CHANNEL_MAPPING_PROVENANCE:
            raise ValueError("PTT mapping provenance drift")
        expected_hashes = {
            "source_values_sha256": _array_identity_sha256(source, self.source_channel_schema),
            "output_values_sha256": _array_identity_sha256(matrix, self.channel_schema),
            "source_channel_schema_sha256": _payload_identity_sha256(list(self.source_channel_schema)),
            "target_channel_schema_sha256": _payload_identity_sha256(list(self.channel_schema)),
            "resampling_config_sha256": _payload_identity_sha256(
                {
                    "source_fs_hz": self.source_fs_hz,
                    "target_fs_hz": self.target_fs_hz,
                    "up": self.up,
                    "down": self.down,
                    "method": "scipy_signal_resample_poly_anti_alias_line_pad_v2",
                    "axis": 0,
                }
            ),
            "mapping_sha256": _payload_identity_sha256(dict(self.mapping_provenance)),
        }
        for name, expected in expected_hashes.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PTT adapted {name} drift")
        if not _SHA256_PATTERN.fullmatch(self.source_file_sha256):
            raise ValueError("PTT source file SHA-256 is invalid")
        if self.source_manifest_sha256 != M2_EXTERNAL_MANIFEST_SHA256:
            raise ValueError("PTT source manifest SHA-256 drift")

def _payload_identity_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _array_identity_sha256(
    values: np.ndarray,
    channel_schema: tuple[str, ...],
) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "dtype": "<f8",
                "shape": list(array.shape),
                "channel_schema": list(channel_schema),
                "order": "C",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

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
    if not isinstance(parsed, list) or not parsed or any(not isinstance(item, str) or not item for item in parsed):
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

def select_ptt_distal_red_ir(channels: Mapping[str, np.ndarray]) -> np.ndarray:
    """Select the adopted distal PTT pair in canonical [RED, IR] order."""

    missing = [source for source in PTT_DISTAL_CHANNEL_MAPPING.values() if source not in channels]
    if missing:
        raise ValueError("PTT distal mapping missing channels: " + ",".join(missing))
    red = np.asarray(channels[PTT_DISTAL_CHANNEL_MAPPING["RED"]], dtype=np.float64).reshape(-1)
    infrared = np.asarray(channels[PTT_DISTAL_CHANNEL_MAPPING["IR"]], dtype=np.float64).reshape(-1)
    if red.size == 0 or red.shape != infrared.shape:
        raise ValueError("PTT distal RED/IR channels must be non-empty and aligned")
    pair = np.column_stack((red, infrared))
    if not np.isfinite(pair).all():
        raise ValueError("PTT distal RED/IR channels must be finite")
    return pair

def adapt_ptt_synchronized_channels(
    channels: Mapping[str, np.ndarray],
    *,
    record_id: str | None = None,
    external_record: ExternalRecord | None = None,
    observed_source_file_sha256: str | None = None,
    additional_channel_order: Iterable[str] = (),
    source_fs_hz: float = 500.0,
    target_fs_hz: float = TARGET_INTERNAL_FS_HZ,
) -> PttSynchronizedSignals:
    """Select distal RED/IR and jointly resample optional ECG/IMU channels.

    The additional channel list is explicit and ordered. All selected source
    arrays are stacked before one polyphase operation, so PPG, ECG reference,
    and IMU cannot acquire separate rounding or padding offsets.
    """

    if float(source_fs_hz) != 500.0 or float(target_fs_hz) != TARGET_INTERNAL_FS_HZ:
        raise ValueError("formal PTT adapter only supports synchronized 500-to-400 Hz")
    if external_record is not None:
        _validate_external_record(external_record)
        if external_record.dataset_id != PTT_DATASET_ID:
            raise ValueError("PTT adapter external_record is not from the PTT dataset")
        if record_id is not None and str(record_id) != external_record.record_id:
            raise ValueError("PTT adapter record_id conflicts with external manifest")
        record_id = external_record.record_id
        if not _SHA256_PATTERN.fullmatch(external_record.checksum_sha256):
            raise ValueError("formal PTT adapter requires one source-file SHA-256")
        if observed_source_file_sha256 != external_record.checksum_sha256:
            raise ValueError("PTT observed source-file SHA-256 differs from manifest")
        source_file_sha256 = external_record.checksum_sha256
    else:
        source_file_sha256 = str(observed_source_file_sha256 or "")
    if not record_id or not _SHA256_PATTERN.fullmatch(source_file_sha256):
        raise ValueError("PTT adapter requires record_id and observed source-file SHA-256")
    extras = tuple(str(name) for name in additional_channel_order)
    if len(set(extras)) != len(extras) or any(
        name in {"pleth_1", "pleth_2", "RED", "IR"} or not name for name in extras
    ):
        raise ValueError("PTT additional channel order is duplicated or aliases distal PPG")
    source_names = ("pleth_1", "pleth_2", *extras)
    missing = [name for name in source_names if name not in channels]
    if missing:
        raise ValueError("PTT synchronized adapter missing channels: " + ",".join(missing))
    arrays = [np.asarray(channels[name], dtype=np.float64).reshape(-1) for name in source_names]
    if not arrays or arrays[0].size == 0 or any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("PTT synchronized source channels must be non-empty and aligned")
    from ..signal.resample import resample_synchronized_channels

    source_values = np.column_stack(arrays)
    target_schema = ("RED", "IR", *extras)
    resampled = resample_synchronized_channels(
        source_values,
        channel_schema=target_schema,
        source_fs_hz=source_fs_hz,
        target_fs_hz=target_fs_hz,
    )
    result = PttSynchronizedSignals(
        record_id=str(record_id),
        source_values=source_values,
        values=resampled.values,
        channel_schema=resampled.channel_schema,
        timestamps_s=resampled.timestamps_s,
        source_channel_schema=source_names,
        source_fs_hz=resampled.source_fs_hz,
        target_fs_hz=resampled.target_fs_hz,
        up=resampled.up,
        down=resampled.down,
        mapping_provenance=PTT_CHANNEL_MAPPING_PROVENANCE,
        source_file_sha256=source_file_sha256,
        source_manifest_sha256=M2_EXTERNAL_MANIFEST_SHA256,
        source_values_sha256=_array_identity_sha256(source_values, source_names),
        output_values_sha256=_array_identity_sha256(resampled.values, target_schema),
        source_channel_schema_sha256=_payload_identity_sha256(list(source_names)),
        target_channel_schema_sha256=_payload_identity_sha256(list(target_schema)),
        resampling_config_sha256=_payload_identity_sha256(
            {
                "source_fs_hz": float(source_fs_hz),
                "target_fs_hz": float(target_fs_hz),
                "up": resampled.up,
                "down": resampled.down,
                "method": resampled.method,
                "axis": 0,
            }
        ),
        mapping_sha256=_payload_identity_sha256(PTT_CHANNEL_MAPPING_PROVENANCE),
    )
    result.validate()
    return result

def _adapt_m2_row(raw: Mapping[str, str]) -> ExternalRecord:
    """把 M2 行映射为 V2 合同 / Map one M2 row into the V2 contract."""

    fs = float(raw["container_grid_fs_hz"])
    payload: dict[str, object] = {name: str(raw[name]) for name in M2_EXTERNAL_COLUMNS}
    payload.update(
        contract_schema_version=EXTERNAL_MANIFEST_VERSION,
        source_files=_parse_source_files(str(raw["source_files"])),
        container_grid_fs_hz=fs,
        ppg_wavelength_status=(
            PTT_WAVELENGTH_STATUS if raw["dataset_id"] == PTT_DATASET_ID else str(raw["ppg_wavelength_status"])
        ),
        evaluation_role=_evaluation_role(raw),
        independence_claim=INDEPENDENCE_CLAIM,
        resampling_required=not abs(fs - TARGET_INTERNAL_FS_HZ) < 1e-12,
        target_internal_fs_hz=TARGET_INTERNAL_FS_HZ,
        source_manifest_sha256=M2_EXTERNAL_MANIFEST_SHA256,
    )
    return ExternalRecord(**payload)  # type: ignore[arg-type]

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
    expected_resampling = not abs(row.container_grid_fs_hz - row.target_internal_fs_hz) < 1e-12
    if row.resampling_required != expected_resampling:
        raise ValueError("resampling flag/rate mismatch")
    if row.dataset_id == PTT_DATASET_ID and row.ppg_wavelength_status != PTT_WAVELENGTH_STATUS:
        raise ValueError("PTT wavelength mapping differs from adopted distal V2 mapping")
    if row.dataset_id == PTT_DATASET_ID and row.ppg_channels != "pleth_1..pleth_6":
        raise ValueError("PTT source channel roster drift")

def _record_to_csv(row: ExternalRecord) -> dict[str, str]:
    """编码强类型外部记录 / Encode one typed external record."""

    _validate_external_record(row)
    encoded = {name: str(getattr(row, name)) for name in EXTERNAL_MANIFEST_COLUMNS}
    encoded.update(
        source_files=_strict_json(list(row.source_files)),
        container_grid_fs_hz=f"{row.container_grid_fs_hz:.12g}",
        resampling_required="true" if row.resampling_required else "false",
        target_internal_fs_hz=f"{row.target_internal_fs_hz:.12g}",
    )
    return encoded

def _record_from_csv(raw: Mapping[str, str]) -> ExternalRecord:
    """从物化 CSV 恢复外部记录 / Restore one materialized external record."""

    missing = [name for name in EXTERNAL_MANIFEST_COLUMNS if name not in raw]
    if missing:
        raise ValueError(f"external row missing fields: {missing}")
    flag = str(raw["resampling_required"]).lower()
    if flag not in {"true", "false"}:
        raise ValueError("resampling_required must be true or false")
    payload: dict[str, object] = {name: str(raw[name]) for name in EXTERNAL_MANIFEST_COLUMNS}
    payload.update(
        source_files=_parse_source_files(str(raw["source_files"])),
        container_grid_fs_hz=float(raw["container_grid_fs_hz"]),
        resampling_required=flag == "true",
        target_internal_fs_hz=float(raw["target_internal_fs_hz"]),
    )
    row = ExternalRecord(**payload)  # type: ignore[arg-type]
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
        {row.activity_raw for row in ptt if row.subject_id == subject_id} != {"sit", "walk", "run"}
        for subject_id in ptt_subjects
    ):
        issues.append("PTT must contain 22 grouped subjects with sit/walk/run")
    if any(row.inclusion_status != "included" for row in ptt):
        issues.append("all 66 PTT records must remain included")
    if any(row.ppg_wavelength_status != PTT_WAVELENGTH_STATUS for row in ptt):
        issues.append("PTT project-adopted distal mapping drift")

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
        "record_count_included": sum(row.inclusion_status == "included" for row in materialized),
        "record_count_excluded": sum(row.inclusion_status == "excluded" for row in materialized),
        "dataset_record_counts": dict(sorted(by_dataset.items())),
        "ptt_subject_count": len(ptt_subjects),
        "ptt_activity_counts": dict(sorted(Counter(row.activity_raw for row in ptt).items())),
        "sim_included_count": len(sim_included),
        "ptt_wavelength_interpretation": PTT_WAVELENGTH_STATUS,
        "ptt_source_wavelength_status": PTT_SOURCE_WAVELENGTH_STATUS,
        "ptt_distal_channel_mapping": dict(PTT_DISTAL_CHANNEL_MAPPING),
        "ptt_channel_mapping_provenance": dict(PTT_CHANNEL_MAPPING_PROVENANCE),
        "ptt_imu_unit_conflict_provenance": dict(PTT_IMU_UNIT_CONFLICT_PROVENANCE),
        "independence_claim": INDEPENDENCE_CLAIM,
    }

def load_m2_external_manifest(source_csv: str | Path) -> list[ExternalRecord]:
    """读取并验证 M2 权威外部清单 / Load the authoritative M2 external table."""

    source = Path(source_csv)
    observed = sha256_file(source)
    if observed != M2_EXTERNAL_MANIFEST_SHA256:
        raise ExternalManifestError(["M2 external manifest SHA drift: " f"{observed} != {M2_EXTERNAL_MANIFEST_SHA256}"])
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
    """读取已物化 V2 外部清单 / Load the materialized V2 external manifest."""

    rows: list[ExternalRecord] = []
    issues: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXTERNAL_MANIFEST_COLUMNS:
            raise ExternalManifestError(["V2 external column order drift"])
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
    """从唯一 M2 路径构建 V2 外部合同 / Build V2 from the sole M2 path."""

    source = Path(source_csv).resolve()
    pipeline_root = Path(__file__).resolve().parents[3]
    repository_root = pipeline_root.parents[1]
    expected = (repository_root / M2_EXTERNAL_RELATIVE_PATH).resolve()
    if source != expected:
        raise ExternalManifestError([f"unsupported external authority: {source}; expected {expected}"])
    rows = load_m2_external_manifest(source)
    write_external_manifest_csv(output_csv, rows, output_root=pipeline_root)
    return rows

def _reject_historical_v1_provisional_split() -> None:
    """Keep every archived provisional splitter entry point inert in V2."""

    raise ExternalManifestError(
        [
            "historical_v1_provisional_split_is_not_executable_in_final_pipeline_v2; "
            "use data.external_folds formal repeated grouped folds"
        ]
    )

def _provisional_fold_by_subject(
    rows: Iterable[ExternalRecord],
) -> dict[str, int]:
    """Archived V1 name retained for import compatibility."""
    del rows
    _reject_historical_v1_provisional_split()

def _validate_provisional_rows(rows: Iterable[Mapping[str, str]]) -> None:
    """Archived V1 name retained for import compatibility."""
    del rows
    _reject_historical_v1_provisional_split()

def _historical_v1_materialize_provisional_external_grouped_split(
    records: Iterable[ExternalRecord],
    output_csv: str | Path,
    *,
    output_root: str | Path,
) -> list[dict[str, str]]:
    """Archived V1 name retained for import compatibility."""
    del records, output_csv, output_root
    _reject_historical_v1_provisional_split()

def _historical_v1_load_provisional_external_split(
    path: str | Path,
) -> list[dict[str, str]]:
    """Archived V1 name retained for import compatibility."""
    del path
    _reject_historical_v1_provisional_split()

def materialize_provisional_external_grouped_split(*args: object, **kwargs: object) -> None:
    """Reject the archived provisional V1 splitter from active V2 code."""
    del args, kwargs
    _reject_historical_v1_provisional_split()

def load_provisional_external_split(*args: object, **kwargs: object) -> None:
    """Reject loading provisional V1 fold identities through the active V2 API."""
    del args, kwargs
    _reject_historical_v1_provisional_split()


__all__ = [
    "EXTERNAL_MANIFEST_COLUMNS",
    "EXTERNAL_MANIFEST_VERSION",
    "ExternalManifestError",
    "ExternalRecord",
    "PttSynchronizedSignals",
    "INDEPENDENCE_CLAIM",
    "M2_EXTERNAL_MANIFEST_SHA256",
    "M2_EXTERNAL_RELATIVE_PATH",
    "PTT_ADOPTED_ACCELERATION_CONVERSION",
    "PTT_ADOPTED_ACCELERATION_UNIT",
    "PTT_ADOPTED_GYROSCOPE_CONVERSION",
    "PTT_ADOPTED_GYROSCOPE_UNIT",
    "PTT_CHANNEL_MAPPING_PROVENANCE",
    "PTT_DATASET_ID",
    "PTT_DISTAL_CHANNEL_MAPPING",
    "PTT_IMU_UNIT_CONFLICT_PROVENANCE",
    "PTT_IMU_UNIT_EVIDENCE_RELATIVE_PATH",
    "PTT_IMU_UNIT_EVIDENCE_SHA256",
    "PTT_SOURCE_PAGE_URL",
    "PTT_SOURCE_WAVELENGTH_STATUS",
    "PTT_WAVELENGTH_STATUS",
    "SIM_DATASET_ID",
    "audit_external_manifest",
    "adapt_ptt_synchronized_channels",
    "build_external_manifest",
    "load_external_manifest",
    "load_m2_external_manifest",
    "select_ptt_distal_red_ir",
    "write_external_manifest_csv",
]
