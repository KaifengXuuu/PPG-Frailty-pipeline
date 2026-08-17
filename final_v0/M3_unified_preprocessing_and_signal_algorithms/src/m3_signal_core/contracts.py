"""M3 公共返回合同与状态类型 / M3 shared result contracts and status types.

中文：所有算法都返回显式状态、原因码、mask 和 provenance，禁止把无效或不足
数据静默转换为零值特征。

English: Every algorithm returns explicit status, reason codes, masks, and provenance.
Invalid or insufficient data must never be silently converted into zero-valued features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ProcessingStatus(str, Enum):
    """统一处理状态 / Canonical processing status."""

    VALID = "valid"
    REPAIRED = "repaired"
    PARTIAL = "partial"
    INVALID = "invalid"
    INSUFFICIENT = "insufficient"
    INITIALIZATION_PENDING = "initialization_pending"
    NO_ESTIMATE = "no_estimate"


@dataclass(frozen=True)
class QualityIssue:
    """一条可追溯质量问题 / One traceable quality issue."""

    code: str
    severity: str
    detail: str
    channel: str | None = None
    start_index: int | None = None
    stop_index: int | None = None


@dataclass
class QualityAssessment:
    """清洗与异常门控结果 / Cleaning and anomaly-gate result."""

    status: ProcessingStatus
    signal: np.ndarray
    valid_mask: np.ndarray
    repair_mask: np.ndarray
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    profile_id: str = ""


@dataclass
class PpgPreprocessResult:
    """PPG 预处理结果 / PPG preprocessing result."""

    status: ProcessingStatus
    filtered: np.ndarray | None
    quality: QualityAssessment
    raw_metrics: dict[str, float]
    filter_metadata: dict[str, Any]
    final_filter_state: np.ndarray | None = None
    # 中文：保留未经修复的输入和已修复视图，避免插值后数据冒充 source raw。
    # English: Preserve source and repaired views so interpolation never masquerades as raw.
    source_raw: np.ndarray | None = None
    repaired_raw: np.ndarray | None = None


@dataclass
class ExternalResampleResult:
    """外部 PPG 统一重采样结果 / Profile-bound external PPG resampling result.

    中文：信号、时间、有效 mask 和峰标注共享同一有理数时间映射，避免不同
    payload 各自四舍五入后失去同步。

    English: Signal, time, validity mask, and peak annotations share one rational
    time mapping so independently rounded payloads cannot drift out of sync.
    """

    status: ProcessingStatus
    signal: np.ndarray
    timestamps_s: np.ndarray
    valid_mask: np.ndarray
    peak_annotations: np.ndarray
    reason_codes: list[str]
    profile_id: str
    metadata: dict[str, Any]


@dataclass
class PeakResult:
    """峰值、PPI 与极性选择结果 / Peak, PPI, and polarity-selection result."""

    status: ProcessingStatus
    peaks: np.ndarray
    peak_confidence: np.ndarray
    polarity: int
    raw_ppi_sec: np.ndarray
    ppi_valid_mask: np.ndarray
    valid_ppi_sec: np.ndarray
    corrected_nni_sec: np.ndarray
    score: float
    reason_codes: list[str] = field(default_factory=list)
    # 中文：算法和预处理 profile 是 provenance，不得伪装成失败原因码。
    # English: Algorithm/profile provenance is separate from failure reason codes.
    algorithm_id: str = "m3_peak_corrected_v1"
    profile_id: str = ""
    nni_semantics: str = "hard_valid_ppi_no_imputation_v1"


@dataclass
class HrvResult:
    """HR/HRV 结果及覆盖状态 / HR/HRV result with coverage status."""

    status: ProcessingStatus
    metrics: dict[str, Any]
    reason_codes: list[str]
    duration_sec: float
    valid_beat_count: int
    # 中文：明确记录 PPG-derived PRV 版本，避免沿用 ECG-HRV 名称造成歧义。
    # English: Version PPG-derived PRV explicitly instead of implying ECG HRV.
    algorithm_id: str = "m3_hr_ppi_prv_corrected_v1"
    source_peak_algorithm_id: str = ""
    source_profile_id: str = ""


@dataclass
class ImuPreprocessResult:
    """IMU 主/对照路线公共输出 / Shared IMU main/comparator output."""

    status: ProcessingStatus
    gravity_mps2: np.ndarray | None
    dynamic_acc_mps2: np.ndarray | None
    gyro_rads: np.ndarray | None
    jerk_mps3: np.ndarray | None
    sample_valid_mask: np.ndarray
    quality: QualityAssessment
    diagnostics: dict[str, Any]
    profile_id: str


def _json_value(value: Any) -> Any:
    """递归转换 NumPy 值；recursively convert NumPy values for strict JSON."""

    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def map_processing_status_to_m1(
    status: ProcessingStatus,
    *,
    end_of_stream: bool,
) -> str:
    """把 M3 模块状态确定性映射到 M1 output / Map M3 status to M1."""

    mapping = {
        ProcessingStatus.VALID: "ok",
        ProcessingStatus.REPAIRED: "partial",
        ProcessingStatus.PARTIAL: "partial",
        ProcessingStatus.INVALID: "invalid_input",
        ProcessingStatus.INSUFFICIENT: "insufficient_quality",
        ProcessingStatus.NO_ESTIMATE: "insufficient_quality",
    }
    if status == ProcessingStatus.INITIALIZATION_PENDING:
        return "insufficient_quality" if end_of_stream else "processing_lag"
    return mapping[status]


def to_serializable(result: Any) -> dict[str, Any]:
    """把 dataclass 转换为严格 JSON 对象 / Convert a dataclass to strict JSON."""

    return _json_value(asdict(result))
